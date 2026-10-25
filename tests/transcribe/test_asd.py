"""tx-asd unit tests: tracker + seats + coasting, the Light-ASD crop/chunk plumbing with a fake scorer, the model
definition (random weights, torch only), the ASD scorer, and the CLI on a synthetic video. No downloads, no
servers, no real faces: detections are scripted (FakeDetector)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval.transcribe.score_asd import average_precision, score_asd, speech_intervals
from unspoken.transcribe.asd import LightASDMethod, Settings, build_parser, run_asd, settings_from_args
from unspoken.transcribe.contracts import load_asd, to_json
from unspoken.transcribe.faces import HALVES, FaceTracks, FakeDetector, crop_square

LEFT = (0.15, 0.30, 0.20, 0.40)
RIGHT = (0.65, 0.30, 0.20, 0.40)
FPS = 25


# ----------------------------------------------------------------------------------------------- tracker
def test_tracks_keep_ids_and_seats_and_coast():
    def boxes_at(i):
        return [LEFT] if 4 <= i <= 5 else [LEFT, RIGHT]        # right face missed on frames 4-5
    ft = FaceTracks(FakeDetector(boxes_at), zones=HALVES, coast_s=0.5)
    per_frame = [ft.update(np.zeros((100, 200, 3), np.uint8), i / FPS) for i in range(10)]
    ids = {b.track_id for f in per_frame for b in f}
    assert ids == {"f1", "f2"}
    seats = {b.track_id: b.seat for f in per_frame for b in f}
    assert seats == {"f1": "left", "f2": "right"}
    coasted = [(i, b.track_id) for i, f in enumerate(per_frame) for b in f if not b.detected]
    assert coasted == [(4, "f2"), (5, "f2")]
    assert all(len(f) == 2 for f in per_frame)                # coasting keeps both faces every frame


def test_tracks_drop_after_coast_window():
    def boxes_at(i):
        return [LEFT, RIGHT] if i < 3 else [LEFT]
    ft = FaceTracks(FakeDetector(boxes_at), zones=HALVES, coast_s=0.1, max_lost_s=0.2)
    out = [ft.update(np.zeros((100, 200, 3), np.uint8), i / FPS) for i in range(12)]
    assert [len(f) for f in out] == [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1]   # coasted 2 frames (0.08 s), then gone
    assert "f2" not in ft.alive()


def test_crop_square_shape_and_padding():
    g = np.full((100, 200), 50, np.uint8)
    g[60:100, 160:200] = 200
    c = crop_square(g, (0.8, 0.6, 0.2, 0.4))                  # mouth-shifted crop leaves the bottom -> grey pad
    assert c.shape == (112, 112) and c.dtype == np.uint8
    assert (c == 110).any() and (c == 200).any()
    assert crop_square(g, (0.5, 0.5, 0.0, 0.0)).shape == (112, 112)   # degenerate box -> grey patch


# ----------------------------------------------------------------------------------------------- synthetic video
def _write_video(path: Path, n: int = 50, w: int = 200, h: int = 100) -> Path:
    cv2 = pytest.importorskip("cv2")
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    if not out.isOpened():
        pytest.skip("OpenCV cannot write mp4 here")
    for i in range(n):
        img = np.full((h, w, 3), 40, np.uint8)
        img[30:70, 30:70] = 230                               # bright "face" left
        img[30:70, 130:170] = 20                              # dark face right ...
        img[56:70, 138:162] = 20 + (i % 2) * 60               # ... whose mouth region flickers (mouth motion!)
        out.write(img)
    out.release()
    return path


class BrightScorer:
    """Fake Light-ASD: P(speaking) = mean crop intensity > 128. Records what it was asked to score."""
    label = "fake bright-scorer"

    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    def score(self, crops, mfcc):
        self.calls.append((len(crops), len(mfcc)))
        return (crops.reshape(len(crops), -1).mean(1) > 128).astype(np.float32)


def test_run_asd_lightasd_plumbing_with_fake_scorer(tmp_path):
    video = _write_video(tmp_path / "v.mp4")
    def boxes_at(i):
        return [LEFT] if 20 <= i < 25 else [LEFT, RIGHT]      # right face lost for 0.2 s then back
    scorer = BrightScorer()
    s = Settings(method="lightasd", chunk_s=1.0, min_track_s=0.2, coast_s=0.5)
    res = run_asd(video, s, detector=FakeDetector(boxes_at), scorer=scorer, audio=np.zeros(16000 * 2, np.int16))
    assert res.fps == FPS and len(res.frames) == 50
    assert {(f.track_id, f.seat) for f in res.faces} == {("f1", "left"), ("f2", "right")}
    assert all(len(f.scores) == 2 for f in res.frames)        # coasted frames 20-24 are kept (face came back)
    assert all(f.scores["f1"] == 1.0 and f.scores["f2"] == 0.0 for f in res.frames)
    assert all(a == 4 * v for v, a in scorer.calls)           # 4 MFCC rows per video frame
    assert sum(v for v, _ in scorer.calls) == 100             # every frame of both tracks scored exactly once
    d = json.loads(to_json(res))
    assert d["faces"][0]["box"][2] > 30                       # pixel box, not normalised
    back = load_asd(_save(tmp_path, res))
    assert back.frames[0].scores == res.frames[0].scores


def _save(tmp: Path, res) -> Path:
    p = tmp / "asd.json"
    to_json(res, p)
    return p


def test_run_asd_drops_dead_end_coast_and_short_tracks(tmp_path):
    video = _write_video(tmp_path / "v.mp4")
    def boxes_at(i):
        extra = [(0.4, 0.6, 0.1, 0.1)] if i == 10 else []     # one-frame spurious detection
        return ([LEFT, RIGHT] if i < 30 else [LEFT]) + extra
    s = Settings(method="lightasd", chunk_s=10.0, min_track_s=0.2, coast_s=0.4)
    res = run_asd(video, s, detector=FakeDetector(boxes_at), scorer=BrightScorer(), audio=np.zeros(16000 * 2, np.int16))
    assert [f.track_id for f in res.faces] == ["f1", "f2"]    # f3 (1 frame) dropped
    assert all("f2" in f.scores for f in res.frames[:30])
    assert all("f2" not in f.scores for f in res.frames[30:])  # coasted to nowhere -> not reported


def test_run_asd_motion_and_avcorr_on_synthetic_video(tmp_path):
    video = _write_video(tmp_path / "v.mp4")
    audio = (np.sin(np.arange(16000 * 2) * 0.05) * 8000).astype(np.int16)     # steady tone = voice present
    for method in ("motion", "avcorr"):
        res = run_asd(video, Settings(method=method, min_track_s=0.2), detector=FakeDetector(lambda i: [LEFT, RIGHT]),
                      audio=audio)
        assert len(res.frames) == 50 and len(res.faces) == 2
        ps = [p for f in res.frames for p in f.scores.values()]
        assert ps and all(0.0 <= p <= 1.0 for p in ps)
        assert res.timing["rtf"] > 0 and res.timing["threads"] == 6
    # motion: the flickering right "face" moves its mouth region, the static left one does not
    res = run_asd(video, Settings(method="motion", min_track_s=0.2), detector=FakeDetector(lambda i: [LEFT, RIGHT]),
                  audio=audio)
    right = np.mean([f.scores["f2"] for f in res.frames[5:]])
    left = np.mean([f.scores["f1"] for f in res.frames[5:]])
    assert right > left


def test_lightasd_method_repeats_crops_for_slow_video():
    """A 12.5 fps video (every second 25-fps tick) is up-sampled to the model's 25 fps grid."""
    class Rec:
        label = "rec"
        def score(self, crops, mfcc):
            self.n = len(crops)
            return np.full(len(crops), 0.7, np.float32)
    rec = Rec()
    m = LightASDMethod(rec, np.zeros((1000, 13), np.float32), chunk_s=100)
    gray = np.zeros((100, 200), np.uint8)
    from unspoken.transcribe.faces import TrackBox
    for k in range(10):
        m.on_frame(k, k / 12.5, None, gray, [TrackBox("f1", LEFT, "left", True)], np.zeros(0), {"f1"})
    out = m.finish()
    assert rec.n == 19 and set(out["f1"]) == set(range(10)) and all(v == 0.7 for v in out["f1"].values())


# ----------------------------------------------------------------------------------------------- model (torch)
def test_lightasd_model_shapes_random_weights():
    pytest.importorskip("torch")
    from unspoken.transcribe.asd_lightasd import LightASDScorer, build_model
    model = build_model()
    assert sum(p.numel() for p in model.parameters()) == 1_021_120 + 258     # the paper's 1.02 M + FC head
    s = LightASDScorer(model=model, durations=(1,), threads=2)
    crops = np.random.randint(0, 255, (30, 112, 112), dtype=np.uint8)
    p = s.score(crops, np.random.randn(100, 13).astype(np.float32))         # short MFCC -> edge padded to 120
    assert p.shape == (30,) and np.all((p >= 0) & (p <= 1))
    assert s.score(crops[:0], np.zeros((0, 13), np.float32)).shape == (0,)


def test_mfcc_rate():
    pytest.importorskip("python_speech_features")
    from unspoken.transcribe.asd_lightasd import mfcc_features
    x = np.random.randint(-3000, 3000, 16000 * 2 + 160).astype(np.int16)
    m = mfcc_features(x)
    assert m.shape == (201, 13) and m.dtype == np.float32
    assert mfcc_features(np.zeros(100, np.int16)).shape == (1, 13)          # too short for one window -> zeros


# ----------------------------------------------------------------------------------------------- scorer
def _case(tmp: Path) -> Path:
    words = [{"speaker": "A", "seat": "left", "on_camera": True, "start": 0.0, "end": 0.9, "text": "hello"},
             {"speaker": "A", "seat": "left", "on_camera": True, "start": 1.0, "end": 2.0, "text": "there"},
             {"speaker": "C", "seat": "right", "on_camera": True, "start": 3.0, "end": 4.0, "text": "yes"},
             {"speaker": "D", "seat": None, "on_camera": False, "start": 5.0, "end": 6.0, "text": "offcam"}]
    meta = {"case": "t", "speakers": {"A": {"seat": "left", "on_camera": True}, "C": {"seat": "right", "on_camera": True},
                                      "D": {"seat": None, "on_camera": False}}}
    (tmp / "ref.words.json").write_text(json.dumps(words))
    (tmp / "meta.json").write_text(json.dumps(meta))
    return tmp


def _asd(left_on, right_on, seats=("left", "right"), n=70, fps=10):
    frames = []
    for i in range(n):
        t = i / fps
        frames.append({"t": t, "scores": {"f1": 1.0 if left_on(t) else 0.0, "f2": 1.0 if right_on(t) else 0.0}})
    return {"model": "x", "fps": fps, "faces": [{"track_id": "f1", "seat": seats[0], "box": [0, 0, 1, 1], "frames": n},
                                               {"track_id": "f2", "seat": seats[1], "box": [0, 0, 1, 1], "frames": n}],
            "frames": frames, "timing": {"rtf": 0.5, "threads": 6}}


def test_speech_intervals_bridge():
    words = [{"speaker": "A", "start": 0.0, "end": 0.9}, {"speaker": "A", "start": 1.0, "end": 2.0},
             {"speaker": "A", "start": 2.5, "end": 3.0}, {"speaker": "B", "start": 0.0, "end": 9.0}]
    assert speech_intervals(words, "A") == [(0.0, 2.0), (2.5, 3.0)]
    assert average_precision(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1])) == pytest.approx((1 + 2 / 3) / 2)


def test_score_asd_oracle_and_swapped(tmp_path):
    case = _case(tmp_path)
    oracle = _asd(lambda t: t <= 2.0, lambda t: 3.0 <= t <= 4.0)
    r = score_asd(case, oracle)
    assert r["map"] == 1.0 and r["acc"] == 1.0 and r["which_seat_acc"] == 1.0 and r["talk_f1"] == 1.0
    assert r["seats"]["left"]["face_cov"] == 1.0 and r["seats"]["right"]["ref_speaker"] == "C"
    swapped = _asd(lambda t: t <= 2.0, lambda t: 3.0 <= t <= 4.0, seats=("right", "left"))
    r2 = score_asd(case, swapped)
    assert r2["which_seat_acc"] == 0.0 and r2["map"] < 0.5
    silent = _asd(lambda t: False, lambda t: False)
    r3 = score_asd(case, silent)
    assert r3["which_seat_acc"] == 0.0 and r3["talk_f1"] == 0.0 and r3["which_seat_abstain"] == 1.0
    nobody = {**silent, "frames": [{"t": f["t"], "scores": {}} for f in silent["frames"]]}
    assert score_asd(case, nobody)["which_seat_abstain"] == 1.0


# ----------------------------------------------------------------------------------------------- CLI
def test_cli_settings():
    a = build_parser().parse_args(["--in", "v.mp4", "--out", "o.json", "--method", "motion", "--fps", "12.5",
                                   "--durations", "1,3", "--threads", "4"])
    s = settings_from_args(a)
    assert s.method == "motion" and s.fps == 12.5 and s.durations == (1, 3) and s.threads == 4 and s.seats is None
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--in", "v.mp4", "--out", "o.json", "--method", "nope"])


# ----------------------------------------------------------------------------------------------- live session
def test_lightasd_live_session_with_fake_scorer():
    from unspoken.contracts import FaceTrack
    from unspoken.transcribe.asd_live import LightASDLive

    class Rec:
        label = "rec"
        calls: list[tuple[int, int]] = []
        def score(self, crops, mfcc):
            Rec.calls.append((len(crops), len(mfcc)))
            return (crops.reshape(len(crops), -1).mean(1) > 128).astype(np.float32)

    live = LightASDLive(options={"scorer": Rec(), "window_s": 1.0, "every_s": 0.2, "min_s": 0.4})
    sess = live.session()
    frame = np.full((100, 200, 3), 40, np.uint8)
    frame[30:70, 30:70] = 230                                   # bright face left, dark face right
    faces = [FaceTrack("f1", LEFT, "left"), FaceTrack("f2", RIGHT, "right")]
    pcm = np.zeros(640, np.int16).tobytes()                     # 40 ms of silence per 25 fps frame
    out = None
    for k in range(30):
        out = sess.update(frame, k / 25, faces, pcm, 16000)
    assert {f.track_id: f.speaking for f in out} == {"f1": 1.0, "f2": 0.0}
    assert Rec.calls and all(a == 4 * v for v, a in Rec.calls) and all(v <= 25 for v, _ in Rec.calls)
    out = sess.update(frame, 31 / 25, faces[:1], pcm, 16000)   # f2 left the frame: forgotten
    assert [f.track_id for f in out] == ["f1"] and "f2" not in sess.crops
    sess.reset()
    assert not sess.crops and not sess.speaking
    assert sess.update(None, 2.0, [], None, 16000) == []       # no frame, no faces: nothing scored, no error
