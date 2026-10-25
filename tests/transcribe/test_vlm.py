"""VLM judge (tx-vlm): frames, grammar/logprob parsing, binding rules and an end-to-end run with a fake judge.
No model, no server, no network: synthetic frames (a bright square = a 'face'), a fake face finder, canned
llama-server responses. The live check against a real llama-server only runs with UNSPOKEN_LIVE=1."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from unspoken.transcribe.contracts import load_asd, load_transcript
from unspoken.transcribe.vlm_frames import (ArrayFrames, SeatBoxes, UnitSampler, crop_square, faces_grid, sample_times,
                                            seat_of, windows)
from unspoken.transcribe.vlm_judge import (GRAMMAR_JSON, GRAMMAR_WORD, OPTIONS, FakeJudge, LlamaServerJudge,
                                           UnitResult, build_prompt, decide, grammar_for, parse_response,
                                           probs_from_logprobs, rescore, run, segment_probs, units_for)
from unspoken.transcribe.contracts import AsrResult, AsrSegment

W, H = 160, 80


def _frame(left_bright: int, right_bright: int) -> np.ndarray:
    f = np.full((H, W, 3), 30, dtype=np.uint8)
    f[20:60, 20:60] = left_bright          # 'face' in the left half
    f[20:60, 100:140] = right_bright       # 'face' in the right half
    return f


def _finder(frame: np.ndarray):
    """Two fixed boxes (normalised), like a detector that always sees both faces."""
    return [(20 / W, 20 / H, 40 / W, 40 / H), (100 / W, 20 / H, 40 / W, 40 / H)]


def test_sample_times_and_windows():
    assert sample_times(0.0, 2.0, 1) == [1.0]
    assert sample_times(0.0, 3.0, 3) == [0.5, 1.5, 2.5]
    assert windows(0.0, 1.0, 1.5) == [(0.0, 1.0)]                       # short unit stays whole
    w = windows(0.0, 6.0, 1.5)
    assert len(w) == 4 and math.isclose(w[-1][1], 6.0) and all(math.isclose(b - a, 1.5) for a, b in w)
    assert len(windows(0.0, 3.2, 1.5)) == 2                             # tail joins


def test_seat_of_and_fallback_boxes():
    assert seat_of((0.1, 0.1, 0.2, 0.2)) == "left" and seat_of((0.6, 0.1, 0.2, 0.2)) == "right"
    sb = SeatBoxes()
    assert sb.update([(0.1, 0.2, 0.1, 0.2)]) == {"left": (0.1, 0.2, 0.1, 0.2)}
    assert sb.box("left") == (0.1, 0.2, 0.1, 0.2)
    rb = sb.box("right")                                               # never seen: centre of the right zone
    assert 0.5 < rb[0] + rb[2] / 2 < 1.0 and sb.mean_box("right") is None
    sb.update([(0.12, 0.2, 0.1, 0.2)])
    assert sb.seen["left"] == 2 and math.isclose(sb.mean_box("left")[0], 0.11)


def test_crop_and_grid_shapes():
    fr = _frame(200, 90)
    c = crop_square(fr, (20 / W, 20 / H, 40 / W, 40 / H), zoom=2.0, size=64)
    assert c.shape == (64, 64, 3)
    boxes = [{"left": (20 / W, 20 / H, 40 / W, 40 / H), "right": (100 / W, 20 / H, 40 / W, 40 / H)}] * 3
    g = faces_grid([fr] * 3, boxes, tile=64, zoom=2.0)
    assert g.shape == (28 + 3 * 64, 2 * 64, 3)                          # header + 3 rows x 2 columns
    gd = faces_grid([fr] * 2, boxes[:2], tile=64, zoom=2.0, diffs=[_frame(90, 90)] * 2)
    assert gd.shape == (28 + 2 * 64, 4 * 64, 3)                         # face, motion, face, motion
    body = gd[28:]
    assert body[:, 64:128].mean() > body[:, 192:256].mean()             # left changed (200 vs 90), right did not


def test_unit_sampler_views():
    src = ArrayFrames([_frame(200, 90)] * 50, fps=25.0)
    for view, n_imgs in (("faces", 1), ("faces-diff", 1), ("full", 3)):
        s = UnitSampler(src, view=view, k=3, tile=32, finder=_finder)
        p = s.sample(0.2, 1.4)
        assert p is not None and len(p.images) == n_imgs and len(p.times) == 3 and len(p.boxes) == 3
        assert p.boxes[0]["left"][0] < 0.5 < p.boxes[0]["right"][0]
    s = UnitSampler(src, view="faces", k=2, tile=32, finder=None)        # no detector: seat zones
    assert s.sample(0.0, 1.0).images[0].shape == (28 + 2 * 32, 64, 3)
    assert s.sample(10.0, 11.0) is not None                              # past the end: clamped to the last frame
    assert UnitSampler(src, view="faces", k=1, finder=_finder).frames_read == 0
    with pytest.raises(ValueError):
        UnitSampler(src, view="mouth")


def _content(first_token: str, tops: list[tuple[str, float]], prefix: str = "") -> list[dict]:
    out = [{"token": ch, "logprob": -0.01, "top_logprobs": []} for ch in prefix]
    out.append({"token": first_token, "logprob": dict(tops).get(first_token, -0.5),
                "top_logprobs": [{"token": t, "logprob": lp} for t, lp in tops]})
    return out


def test_probs_from_logprobs_word_grammar():
    c = _content("right", [("right", math.log(0.5)), ("left", math.log(0.3)), ("ne", math.log(0.1)),
                           ("LEFT", math.log(0.05)), ("The", math.log(0.05))])
    p = probs_from_logprobs(c)
    assert set(p) == set(OPTIONS) and math.isclose(sum(p.values()), 1.0)
    assert math.isclose(p["right"], 0.5 / 0.95) and math.isclose(p["left"], 0.35 / 0.95)
    assert math.isclose(p["neither"], 0.1 / 0.95) and p["both"] == 0.0
    assert probs_from_logprobs([]) is None
    assert probs_from_logprobs([{"token": "The", "logprob": -0.1, "top_logprobs": []}]) is None


def test_probs_from_logprobs_json_offset():
    prefix = '{"left_mouth": "open", "right_mouth": "closed", "speaker": "'
    c = _content("left", [("left", math.log(0.9)), ("right", math.log(0.1))], prefix=prefix)
    p = probs_from_logprobs(c, answer_offset=len(prefix))
    assert math.isclose(p["left"], 0.9) and math.isclose(p["right"], 0.1)
    assert probs_from_logprobs(c, answer_offset=3) is None               # offset inside a token: no answer


def _resp(text: str, content: list[dict] | None, prompt_tokens: int = 321) -> dict:
    return {"choices": [{"message": {"content": text}, "logprobs": {"content": content} if content else None}],
            "usage": {"prompt_tokens": prompt_tokens}}


def test_parse_response():
    j = parse_response(_resp("left", _content("left", [("left", math.log(0.7)), ("right", math.log(0.3))])), "word", 1.5)
    assert j.choice == "left" and math.isclose(j.probs["left"], 0.7) and j.prompt_tokens == 321 and j.latency_s == 1.5
    assert math.isclose(j.seat_scores["left"], 0.7) and math.isclose(j.seat_scores["right"], 0.3)
    j = parse_response(_resp("both", None), "word")                       # no logprobs: one-hot on the answer
    assert j.probs == {"left": 0.0, "right": 0.0, "neither": 0.0, "both": 1.0}
    assert j.seat_scores == {"left": 1.0, "right": 1.0}
    txt = '{"left_mouth": "closed", "right_mouth": "moving", "speaker": "right"}'
    pre = txt[: txt.index('"speaker": "') + len('"speaker": "')]
    j = parse_response(_resp(txt, _content("right", [("right", math.log(0.8)), ("neither", math.log(0.2))], pre)), "json")
    assert j.choice == "right" and math.isclose(j.probs["right"], 0.8) and math.isclose(j.probs["neither"], 0.2)
    assert parse_response(_resp("garbage", None), "word").choice == "neither"


def test_request_body_and_prompts():
    j = LlamaServerJudge("http://127.0.0.1:1", name="x", timeout_s=1)      # name given: no network in __init__
    body = j.request_body(["data:image/jpeg;base64,AAAA", "data:image/jpeg;base64,BBBB"], "who?", GRAMMAR_WORD)
    user = body["messages"][1]["content"]
    assert [c["type"] for c in user] == ["image_url", "image_url", "text"] and user[-1]["text"] == "who?"
    assert body["grammar"] == GRAMMAR_WORD and body["logprobs"] is True and body["top_logprobs"] == 10
    assert body["temperature"] == 0.0 and body["max_tokens"] == 4 and body["stream"] is False
    assert j.request_body([], "q", GRAMMAR_JSON)["max_tokens"] > 4
    v = LlamaServerJudge("http://127.0.0.1:1", name="v", constraint="guided_choice", model="qwen").request_body([], "q", GRAMMAR_WORD)
    assert v["guided_choice"] == list(OPTIONS) and v["model"] == "qwen" and "grammar" not in v
    with pytest.raises(ValueError):
        LlamaServerJudge("http://127.0.0.1:1", name="v", constraint="guided_choice").request_body([], "q", GRAMMAR_JSON)
    with pytest.raises(ValueError):
        LlamaServerJudge("http://127.0.0.1:1", name="v", constraint="magic")
    assert "grammar" not in LlamaServerJudge("http://127.0.0.1:1", name="n", constraint="none").request_body([], "q", GRAMMAR_WORD)
    assert grammar_for("json") == GRAMMAR_JSON and grammar_for("cues") == GRAMMAR_WORD
    for prompt in ("plain", "cues", "json"):
        for view in ("faces", "faces-diff", "full"):
            t = build_prompt(prompt, view, 3)
            assert "{" not in t and "LEFT" in t and "3" in t
    assert "neither" in build_prompt("cues", "faces", 1)


def _asr(n: int = 4) -> AsrResult:
    return AsrResult("fake asr", [AsrSegment(i * 2.0, i * 2.0 + 1.6, f"segment {i}") for i in range(n)])


def test_units_and_segment_probs():
    asr = _asr(2)
    assert units_for(asr, "segments", 1.5) == [(0, 0.0, 1.6), (1, 2.0, 3.6)]
    asr.segments[1].end = 6.5
    u = units_for(asr, "windows", 1.5)
    assert u[0] == (0, 0.0, 1.6) and len([x for x in u if x[0] == 1]) == 3
    units = [UnitResult(0, 0.0, 1.0, "left", {"left": 1.0, "right": 0.0, "neither": 0.0, "both": 0.0}, 0.1),
             UnitResult(0, 1.0, 2.0, "right", {"left": 0.0, "right": 1.0, "neither": 0.0, "both": 0.0}, 0.1),
             UnitResult(1, 2.0, 2.5, "both", {"left": 0.0, "right": 0.0, "neither": 0.0, "both": 1.0}, 0.1)]
    sp = segment_probs(units, 3)
    assert math.isclose(sp[0]["left"], 0.5) and math.isclose(sp[0]["right"], 0.5)
    assert sp[1]["both"] == 1.0 and sum(sp[2].values()) == 0.0


def test_decide_binds():
    sure_left = {"left": 0.9, "right": 0.1, "neither": 0.0, "both": 0.0}
    unsure = {"left": 0.4, "right": 0.35, "neither": 0.25, "both": 0.0}
    nobody = {"left": 0.1, "right": 0.1, "neither": 0.8, "both": 0.0}
    both = {"left": 0.1, "right": 0.0, "neither": 0.0, "both": 0.9}
    probs, voices, durs = [sure_left, unsure, nobody, both], ["voice-A", "voice-A", "voice-B", "voice-C"], [2.0, 1.0, 1.0, 1.0]
    none = decide(probs, voices, durs, "none")
    assert [c for c, _, _ in none] == ["left", "left", "neither", "left"]     # 'both' -> the stronger seat
    assert all(src == "video" for _, _, src in none)
    voice = decide(probs, voices, durs, "voice", min_conf=0.6)
    assert voice[0][2] == "video" and voice[1] == ("left", pytest.approx((2 * 0.9 + 0.4) / 3), "fused")
    assert voice[2][0] == "neither"                                           # confident 'neither' stays
    fuse = decide(probs, voices, durs, "fuse", fuse_w=0.5)
    assert fuse[1][0] == "left" and all(src == "fused" for _, _, src in fuse)
    vs = decide(probs, voices, durs, "voice-seat")
    assert [c for c, _, _ in vs] == ["left", "left", "neither", "left"]        # per voice: A left, B off, C left
    assert vs[0][1] == vs[1][1]                                               # same voice, same affinity
    with pytest.raises(ValueError):
        decide(probs, voices, durs, "magic")


def _case(tmp: Path, n_frames: int = 250) -> tuple[Path, Path, Path, ArrayFrames]:
    """Synthetic case: 10 s, left 'face' bright during segments 0 and 2, right one during segment 1; refs match."""
    frames = []
    for i in range(n_frames):
        t = i / 25.0
        left_on = 0.0 <= t < 1.6 or 4.0 <= t < 5.6
        right_on = 2.0 <= t < 3.6
        frames.append(_frame(230 if left_on else 90, 230 if right_on else 90))
    case = tmp / "case"
    case.mkdir()
    (case / "meta.json").write_text(json.dumps({"case": "synthetic", "duration": 10.0, "kind": "duo"}))
    words = []
    for i, (spk, seat) in enumerate((("A", "left"), ("C", "right"), ("A", "left"))):
        for j in range(2):
            words.append({"speaker": spk, "seat": seat, "on_camera": True, "start": i * 2.0 + 0.2 + 0.6 * j,
                          "end": i * 2.0 + 0.6 + 0.6 * j, "text": f"word{j}"})
    seg = [{"session_id": "s", "speaker": s, "start_time": i * 2.0, "end_time": i * 2.0 + 1.6, "words": "word0 word1"}
           for i, s in enumerate("ACA")]
    (case / "ref.words.json").write_text(json.dumps(words))
    (case / "ref.seglst.json").write_text(json.dumps(seg))
    asr = tmp / "asr.json"
    asr.write_text(json.dumps({"model": "fake", "segments": [{"start": i * 2.0, "end": i * 2.0 + 1.6,
                                                              "text": "word0 word1", "words": []} for i in range(3)]}))
    diar = tmp / "diar.json"
    diar.write_text(json.dumps({"model": "fake", "turns": [{"start": i * 2.0, "end": i * 2.0 + 1.6,
                                                            "voice": v, "conf": 1.0} for i, v in enumerate(("voice-A", "voice-B", "voice-A"))]}))
    return case, asr, diar, ArrayFrames(frames, fps=25.0)


def test_run_end_to_end_with_fake_judge(tmp_path):
    case, asr, diar, src = _case(tmp_path)
    out = tmp_path / "out"
    judge = FakeJudge()                                                     # brighter column = speaking
    r = run(case, out, judge, asr_path=asr, diar_path=diar, view="faces", k=3, prompt="cues", unit="segments",
            binds=("voice", "none"), tile=32, log=None, source=src, faces="none", threads=6)
    assert judge.calls == 3 and r["timing"]["units"] == 3 and r["timing"]["frames_read"] == 9
    assert r["timing"]["threads"] == 6 and r["timing"]["rtf"] >= 0
    tr = load_transcript(out / "transcript.json")
    assert [u.speaker for u in tr.utterances] == ["person-left", "person-right", "person-left"]
    assert tr.speakers["person-left"].seat == "left" and tr.speakers["person-right"].on_camera
    assert all(u.source == "video" for u in tr.utterances) and tr.meta["bind"] == "voice"
    asd = load_asd(out / "asd.json")
    assert [f.track_id for f in asd.faces] == ["seat-left", "seat-right"] and len(asd.frames) == 3
    assert asd.frames[1].scores["seat-right"] > asd.frames[1].scores["seat-left"]
    assert asd.faces[0].box[0] < W / 2 < asd.faces[1].box[0]                 # pixel boxes, per the contract
    sc = json.loads((out / "scores.json").read_text())
    assert sc["seat_acc"] == 1.0 and sc["sa_acc"] == 1.0 and sc["bind"] == "voice"
    assert (out / "transcript.none.json").exists() and (out / "scores.none.json").exists()
    units = json.loads((out / "units.json").read_text())
    assert [u["choice"] for u in units["units"]] == ["left", "right", "left"] and units["settings"]["frames"] == 3
    sc2 = rescore(case, out, asr_path=asr, diar_path=diar, binds=("voice-seat", "fuse"), min_conf=0.6, log=None)
    assert sc2["voice-seat"]["seat_acc"] == 1.0 and (out / "transcript.voice-seat.json").exists()
    assert json.loads((out / "scores.json").read_text())["bind"] == "voice"   # the primary file is untouched


def test_run_windows_and_neither_to_voice(tmp_path):
    case, asr, diar, src = _case(tmp_path)
    out = tmp_path / "out"
    judge = FakeJudge(lambda pics: "neither")
    r = run(case, out, judge, asr_path=asr, diar_path=diar, view="full", k=1, prompt="plain", unit="windows",
            window=0.8, binds=("none",), log=None, source=src, faces="none")
    assert r["timing"]["units"] == 6 and judge.calls == 6                    # 1.6 s segments -> two 0.8 s windows
    tr = load_transcript(out / "transcript.json")
    assert [u.speaker for u in tr.utterances] == ["voice-A", "voice-B", "voice-A"]   # 'neither' -> the voice label
    assert not tr.speakers["voice-A"].on_camera and tr.speakers["voice-A"].voice == "voice-A"
    sc = json.loads((out / "scores.json").read_text())
    assert sc["hyp_speakers"] == 2 and sc["seat_acc"] == 0.0        # nobody on camera was named


def test_video_frames_roundtrip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from unspoken.transcribe.vlm_frames import VideoFrames

    path = tmp_path / "v.mp4"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (W, H))
    if not wr.isOpened():
        pytest.skip("no mp4 encoder in this OpenCV build")
    for i in range(50):
        wr.write(_frame(230 if i < 25 else 90, 90))
    wr.release()
    v = VideoFrames(path)
    assert (v.width, v.height) == (W, H) and v.fps == 25.0 and v.n_frames == 50
    a, b = v.read(0.2), v.read(1.6)
    assert a is not None and b is not None and a[40, 40].mean() > b[40, 40].mean()
    assert v.read(5.0) is None and v.read(0.24) is not None               # forward decode path
    v.close()


@pytest.mark.skipif(os.environ.get("UNSPOKEN_LIVE") != "1", reason="needs a running llama-server with --mmproj")
def test_live_server_answers_with_grammar():
    j = LlamaServerJudge(os.environ.get("UNSPOKEN_VLM_SERVER", "http://127.0.0.1:8795"))
    assert j.healthy()
    src = ArrayFrames([_frame(230, 90)] * 25)
    pics = UnitSampler(src, view="faces", k=1, tile=64, finder=_finder).sample(0.0, 1.0)
    out = j.judge(pics, build_prompt("cues", "faces", 1), GRAMMAR_WORD)
    assert out.choice in OPTIONS and math.isclose(sum(out.probs.values()), 1.0, abs_tol=1e-6) and out.error is None
