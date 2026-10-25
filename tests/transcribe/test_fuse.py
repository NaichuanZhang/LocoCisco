"""Fusion unit tests on synthetic fixtures: no models, no servers, no test data files.

The fixture is a 2-seat, 2-voice scene (left person speaks 0-4 s and 8-12 s, right person 4-8 s) plus an
off-camera voice at 12-14 s; the fake ASD gives the talking face a high score. Every method must recover it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from unspoken.core.decider import FakeDecider
from unspoken.transcribe import fuse as F
from unspoken.transcribe.contracts import (AsdFrame, AsdResult, AsrResult, AsrSegment, DiarResult, FaceInfo, Transcript,
                                           Utterance, VoiceTurn, Word, load_transcript, to_json)
from unspoken.transcribe.fuse_decider import FakeCascade, ask_decider, build_question, build_state
from unspoken.transcribe.fuse_llm import correct_boundaries, parse_reply, tagged
from unspoken.transcribe.render import fmt_srt, fmt_time, render_files, render_md, render_srt, render_txt


def words_for(text: str, start: float, end: float) -> list[Word]:
    toks = text.split()
    step = (end - start) / len(toks)
    return [Word(round(start + i * step, 3), round(start + (i + 1) * step, 3), t) for i, t in enumerate(toks)]


def scene(word_timings: bool = True, swap_asd: bool = False, off_camera: bool = True) -> tuple[AsrResult, DiarResult, AsdResult]:
    segs = [("okay everybody ready let us start with you", 0.0, 4.0),
            ("I am the industrial designer of this group", 4.0, 8.0),
            ("great and I am the project manager today", 8.0, 12.0)]
    if off_camera:
        segs.append(("hello from the back of the room", 12.0, 14.0))
    asr = AsrResult("fake asr", [AsrSegment(a, b, t, words_for(t, a, b) if word_timings else []) for t, a, b in segs])
    turns = [VoiceTurn(0.0, 4.0, "voice-A", 0.9), VoiceTurn(4.0, 8.0, "voice-B", 0.9), VoiceTurn(8.0, 12.0, "voice-A", 0.9)]
    if off_camera:
        turns.append(VoiceTurn(12.0, 14.0, "voice-C", 0.8))
    diar = DiarResult("fake diar", turns)
    faces = [FaceInfo("f1", "left", (0.2, 0.4, 0.1, 0.3), 140), FaceInfo("f2", "right", (0.7, 0.4, 0.1, 0.3), 140)]
    frames = []
    for k in range(140):                                    # 10 Hz, 0..14 s
        t = k * 0.1
        talking = "left" if t < 4.0 or 8.0 <= t < 12.0 else ("right" if t < 8.0 else None)
        if swap_asd and talking:
            talking = "right" if talking == "left" else "left"
        jitter = 0.05 * math.sin(k)                         # deterministic noise
        hi, lo = 0.8 + jitter, 0.1 + jitter
        frames.append(AsdFrame(round(t, 2), {"f1": round(hi if talking == "left" else lo, 3),
                                             "f2": round(hi if talking == "right" else lo, 3)}))
    asd = AsdResult("fake asd", 10.0, faces, frames)
    return asr, diar, asd


# ----------------------------------------------------------------------------------------------- alignment
def test_units_from_words_and_interpolated():
    asr, _, _ = scene(word_timings=True)
    units = F.units_of(asr, F.Settings())
    assert len(units) == 8 + 8 + 8 + 7 and all(u.word is not None for u in units)
    asr2, _, _ = scene(word_timings=False)
    units2 = F.units_of(asr2, F.Settings(interp_words=True))
    assert len(units2) == len(units) and all(u.word is None for u in units2)
    assert units2[0].start == 0.0 and abs(units2[7].end - 4.0) < 1e-6 and units2[1].start > units2[0].start
    seg_units = F.units_of(asr2, F.Settings(interp_words=False))
    assert len(seg_units) == 4 and seg_units[0].text.startswith("okay everybody")


def test_assign_most_overlap_and_nearest():
    units = [F.Unit(0.0, 1.0, "a", 0), F.Unit(0.9, 1.5, "b", 0), F.Unit(5.0, 5.5, "far", 0)]
    turns = [VoiceTurn(0.0, 1.0, "voice-A"), VoiceTurn(1.0, 2.0, "voice-B")]
    F.assign_units(units, turns, F.Settings())
    assert [u.voice for u in units] == ["voice-A", "voice-B", "voice-B"]
    assert units[0].conf > units[1].conf > units[2].conf == 0.1       # far word: a guess


def test_islands_and_merge():
    units = [F.Unit(0.0, 0.5, "we", 0), F.Unit(0.5, 0.8, "um", 0), F.Unit(0.8, 1.3, "start", 0), F.Unit(1.4, 2.0, "now", 0),
             F.Unit(4.0, 4.5, "yes", 0)]
    turns = [VoiceTurn(0.0, 0.5, "voice-A"), VoiceTurn(0.5, 0.8, "voice-B"), VoiceTurn(0.8, 2.0, "voice-A"),
             VoiceTurn(4.0, 4.5, "voice-A")]
    F.assign_units(units, turns, F.Settings())
    groups = F.group_units(units, F.Settings(merge_gap=0.3))
    assert [(g.voice, g.text) for g in groups] == [("voice-A", "we um start now"), ("voice-A", "yes")]
    assert groups[0].source == "rules"
    F.assign_units(units, turns, F.Settings())                 # islands relabel units in place: start over
    groups_off = F.group_units(units, F.Settings(island_max_words=0, merge_gap=0.3))
    assert len(groups_off) == 4


# ----------------------------------------------------------------------------------------------- binding
def test_binding_hungarian_and_off_camera():
    asr, diar, asd = scene()
    ev = F.SeatEvidence(asd, [(t.start, t.end) for t in diar.turns])
    b = F.bind_voices(diar.turns, ev, F.Settings())
    assert b.seat_of_voice == {"voice-A": "left", "voice-B": "right"}        # voice-C stays off camera
    assert b.conf["voice-A"] > 0.9 and "voice-C" not in b.conf
    assert b.table["voice-C"]["left"][0] < 0 and b.table["voice-C"]["right"][0] < 0


def test_binding_full_room_uses_elimination():
    """Two voices, two seats: the dominant voice's own face is flat, the other face goes quiet -> still bound."""
    asr, diar, asd = scene(off_camera=False)
    for fr in asd.frames:                                   # the left face never moves (flat score)
        fr.scores["f1"] = 0.2
    ev = F.SeatEvidence(asd, [(t.start, t.end) for t in diar.turns])
    b = F.bind_voices(diar.turns, ev, F.Settings())
    assert b.seat_of_voice == {"voice-A": "left", "voice-B": "right"}
    assert "everyone on camera" in b.note


def test_binding_room_mode_needs_positive_evidence():
    asr, diar, asd = scene(off_camera=True)
    for fr in asd.frames:
        fr.scores["f1"] = 0.2
    ev = F.SeatEvidence(asd, [(t.start, t.end) for t in diar.turns])
    b = F.bind_voices(diar.turns, ev, F.Settings())
    assert b.seat_of_voice == {"voice-B": "right"} and "voice-A" not in b.seat_of_voice


def test_seat_bias_does_not_matter():
    """A face that always scores high is not 'talking all the time': ranks remove the bias."""
    asr, diar, asd = scene()
    for fr in asd.frames:
        fr.scores["f2"] = round(0.5 + 0.5 * fr.scores["f2"], 3)      # right face biased up
    ev = F.SeatEvidence(asd, [(t.start, t.end) for t in diar.turns])
    b = F.bind_voices(diar.turns, ev, F.Settings())
    assert b.seat_of_voice == {"voice-A": "left", "voice-B": "right"}


# ----------------------------------------------------------------------------------------------- methods
def _check(tr: Transcript, expect: list[tuple[str, str]]) -> None:
    got = [(u.speaker, u.text) for u in tr.utterances]
    assert got == expect, got


EXPECT = [("person-left", "okay everybody ready let us start with you"),
          ("person-right", "I am the industrial designer of this group"),
          ("person-left", "great and I am the project manager today"),
          ("voice-C", "hello from the back of the room")]


@pytest.mark.parametrize("word_timings", [True, False])
def test_methods_align_bind_av(word_timings):
    asr, diar, asd = scene(word_timings=word_timings)
    tr = F.fuse(asr, diar, asd, "align")
    assert [u.speaker for u in tr.utterances] == ["voice-A", "voice-B", "voice-A", "voice-C"]
    assert all(s.on_camera is False for s in tr.speakers.values())
    tr = F.fuse(asr, diar, asd, "bind", session="t", duration=14.0)
    _check(tr, EXPECT)
    assert tr.speakers["person-left"].seat == "left" and tr.speakers["person-left"].voice == "voice-A"
    assert tr.speakers["person-left"].on_camera and not tr.speakers["voice-C"].on_camera
    assert tr.duration == 14.0 and tr.session_id == "t"
    tr = F.fuse(asr, diar, asd, "av")
    _check(tr, EXPECT)
    assert all(u.conf > 0.5 for u in tr.utterances[:3])
    assert len(tr.meta["evidence"]) == len(tr.utterances)
    if word_timings:
        assert all(w.speaker == u.speaker for u in tr.utterances for w in u.words) and tr.utterances[0].words


def test_av_video_overrides_a_wrong_voice():
    """The diarizer mislabels one turn; a reliable ASD (self-calibrated weight ~1) fixes it."""
    asr, diar, asd = scene(off_camera=False)
    diar.turns.append(VoiceTurn(8.0, 12.0, "voice-B", 0.9))          # second half mislabelled as the right voice
    diar.turns = [t for t in diar.turns if not (t.start == 8.0 and t.voice == "voice-A")]
    tr = F.fuse(asr, diar, asd, "bind")
    assert tr.utterances[1].speaker == "person-right" and tr.utterances[1].text.startswith("I am the industrial")
    assert tr.utterances[1].end == 12.0                                 # bind: 4-12 s is one right-person turn
    tr = F.fuse(asr, diar, asd, "av", F.Settings(av_video_weight=0.8))
    _check(tr, EXPECT[:3])
    assert tr.utterances[2].source == "video"
    calib = F.fuse(asr, diar, asd, "av").meta["av"]["video"]           # self-calibrated: video agrees with 2 of 3
    assert calib["how"].startswith("agreement") and calib["n"] == 3


def test_av_video_weight_self_calibrates():
    asr, diar, asd = scene(swap_asd=True, off_camera=False)
    # swapped ASD: the binding follows it (consistently wrong is still consistent), video agrees -> weight > 0
    tr = F.fuse(asr, diar, asd, "av")
    calib = tr.meta["av"]["video"]
    assert calib["agreement"] == 1.0 and 0.3 < calib["weight"] < 1.0        # shrunk: only 3 utterances
    rng_scores = [(0.1 + 0.8 * ((7 * k) % 10) / 10, 0.1 + 0.8 * ((3 * k) % 10) / 10) for k in range(len(asd.frames))]
    for fr, (a, b) in zip(asd.frames, rng_scores):            # pseudo-random: no relation to who talks
        fr.scores = {"f1": round(a, 3), "f2": round(b, 3)}
    tr = F.fuse(asr, diar, asd, "av", F.Settings(bind_min_margin=0.0))
    assert tr.meta["av"]["video"]["weight"] < 0.35


def test_no_asd_means_nobody_bound():
    asr, diar, _ = scene()
    tr = F.fuse(asr, diar, None, "av")
    assert all(k.startswith("voice-") for k in tr.speakers) and "note" in tr.meta["binding"]


# ----------------------------------------------------------------------------------------------- decider
def test_decider_asks_only_low_confidence_and_applies_answer():
    asr, diar, asd = scene()
    tr = F.fuse(asr, diar, asd, "av")
    for u in tr.utterances:
        u.conf = 0.9
    tr.utterances[1].conf = 0.1                                           # make exactly one line "hard"
    fake = FakeDecider(script=[("industrial designer", "*", ("Left person", 0.9))])
    out = ask_decider(tr, FakeCascade(fake), F.Settings(decider_threshold=0.35))
    assert len(fake.calls) == 1
    state, questions = fake.calls[0]
    assert "industrial designer" in state and "Line before" in state and "Line after" in state
    assert set(questions["who"]["criteria"]) == {"Left person", "Right person"}
    assert out.utterances[1].speaker == "person-left" and out.utterances[1].source == "decider"
    assert out.utterances[1].conf == 0.9 and all(w.speaker == "person-left" for w in out.utterances[1].words)
    assert out.meta["decider"]["asked"] == 1 and out.meta["decider"]["changed"] == 1


def test_decider_state_and_question_text():
    asr, diar, asd = scene()
    tr = F.fuse(asr, diar, asd, "av")
    st = build_state(tr, 2, tr.meta["evidence"][2], {"voice-A": {"person-left": 1}})
    assert "attributed to Left person in 1 of its 1 other lines" in st and "project manager" in st
    q = build_question(tr, ["person-left", "voice-C"])
    assert q["criteria"]["Left person"].endswith("left of the camera") and "not on camera" in q["criteria"]["Off-camera C"]


def test_decider_without_client_is_a_noop():
    asr, diar, asd = scene()
    tr = F.fuse(asr, diar, asd, "decider")
    assert tr.meta["decider"]["skipped_no_client"] is True and len(tr.utterances) == 4


# ----------------------------------------------------------------------------------------------- llm
class FakeLlm:
    def __init__(self, replies: dict[str, str]) -> None:
        self.replies, self.prompts = replies, []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.get(prompt, prompt)             # default: unchanged boundary


def test_parse_reply():
    shown = "a b c d".split()
    assert parse_reply("<spk1> a b c </spk1> <spk2> d </spk2>", shown) == 3
    assert parse_reply("<spk1> a </spk1> <spk2> b c d </spk2>", shown) == 1
    assert parse_reply("<spk1> a b </spk1> <spk2> c </spk2>", shown) is None          # dropped a word
    assert parse_reply("<spk2> a b </spk2> <spk1> c d </spk1>", shown) is None        # swapped speakers
    assert parse_reply("sure! here it is", shown) is None


@pytest.mark.parametrize("word_timings", [True, False])
def test_llm_moves_boundary_words(word_timings):
    asr, diar, asd = scene(word_timings=word_timings, off_camera=False)
    tr = F.fuse(asr, diar, asd, "av")
    a, b = tr.utterances[0], tr.utterances[1]
    prompt = tagged(a.text.split()[-12:], b.text.split()[:12])
    reply = "<spk1> okay everybody ready let us start with </spk1> <spk2> you I am the industrial designer of this group </spk2>"
    llm = FakeLlm({prompt: reply})
    out = correct_boundaries(tr, llm, F.Settings())
    assert out.utterances[0].text == "okay everybody ready let us start with"
    assert out.utterances[1].text.startswith("you I am the industrial")
    assert out.meta["llm"]["moved"] == 1 and out.meta["llm"]["words_moved"] == 1 and out.meta["llm"]["asked"] == 2
    assert out.utterances[0].end == out.utterances[1].start < 4.0
    if word_timings:
        assert out.utterances[1].words[0].text == "you" and out.utterances[1].words[0].speaker == "person-right"
        assert out.utterances[0].words[-1].text == "with"


def test_llm_ignores_bad_replies():
    asr, diar, asd = scene(off_camera=False)
    tr = F.fuse(asr, diar, asd, "av")
    before = [(u.speaker, u.text, u.start, u.end) for u in tr.utterances]
    out = correct_boundaries(tr, FakeLlm({p: "garbage" for p in []}) if False else
                             type("L", (), {"complete": staticmethod(lambda p: "I cannot do that")})(), F.Settings())
    assert [(u.speaker, u.text, u.start, u.end) for u in out.utterances] == before
    assert out.meta["llm"]["unparsed"] == 2 and out.meta["llm"]["moved"] == 0


# ----------------------------------------------------------------------------------------------- render + CLI
def test_render_formats(tmp_path: Path):
    asr, diar, asd = scene()
    tr = F.fuse(asr, diar, asd, "av", session="t")
    txt = render_txt(tr)
    assert txt.splitlines()[0] == "[00:00.0] Left person: okay everybody ready let us start with you"
    assert "[00:12.0] Off-camera C: hello from the back of the room" in txt
    srt = render_srt(tr)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:04,000\nLeft person: okay")
    md = render_md(tr)
    assert "| `person-left` | Left person | left | yes | voice-A |" in md
    assert fmt_time(72.34) == "01:12.3" and fmt_srt(3725.5) == "01:02:05,500"
    paths = render_files(tr, tmp_path / "transcript")
    assert {p.suffix for p in paths} == {".txt", ".srt", ".md"} and all(p.exists() for p in paths)


def test_cli_roundtrip(tmp_path: Path):
    asr, diar, asd = scene()
    to_json(asr, tmp_path / "asr.json")
    to_json(diar, tmp_path / "diar.json")
    to_json(asd, tmp_path / "asd.json")
    out = tmp_path / "out" / "transcript.json"
    F.main(["--asr", str(tmp_path / "asr.json"), "--diar", str(tmp_path / "diar.json"), "--asd", str(tmp_path / "asd.json"),
            "--out", str(out), "--method", "av", "--render", "--set", "merge_gap=0.5", "--session", "cli"])
    tr = load_transcript(out)
    _check(tr, EXPECT)
    assert tr.meta["settings"]["merge_gap"] == 0.5 and tr.session_id == "cli"
    assert (out.with_suffix(".txt")).exists() and (out.with_suffix(".srt")).exists()
    with pytest.raises(SystemExit):
        F.Settings.parse(["no_such=1"])
