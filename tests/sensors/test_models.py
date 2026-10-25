"""Model providers: faster-whisper WER on the fixture, WeSpeaker diarization purity, YuNet faces + tracker,
MotionASD on synthetic frames. Model-file tests skip when the file is not cached (nothing downloads)."""
from __future__ import annotations

import asyncio

import numpy as np

from unspoken.contracts import ASR, VAD, ActiveSpeaker, Diarizer, FaceTracker, FaceTrack, RoleSpec
from unspoken.sensors.asd import MotionASD, mouth_box
from unspoken.sensors.audio import float_to_pcm16
from unspoken.sensors.diar import UNKNOWN, EmbeddingDiarizer, voice_name
from unspoken.sensors.faces import IoUTracker, YuNetFaces, iou
from unspoken.sensors.vad import SileroVAD

from .conftest import FACE_JPG, SR, line_pcm, needs_speaker_model, needs_whisper, normalize, purity, wer


# ----------------------------------------------------------------------------- protocols
def test_factories_take_a_rolespec_and_satisfy_the_protocols():
    from unspoken.sensors.asr_local import FasterWhisperASR

    assert isinstance(SileroVAD(RoleSpec("vad", "python:x", options={"threshold": 0.6})), VAD)
    asr = FasterWhisperASR(RoleSpec("asr", "python:x", model="Systran/faster-whisper-tiny.en",
                                    options={"compute_type": "int8"}))
    assert isinstance(asr, ASR) and asr.model_name.endswith("tiny.en") and "tiny.en" in asr.label
    d = EmbeddingDiarizer(RoleSpec("diarization", "python:x", options={"threshold": 0.7, "model_path": "/nope.onnx"}))
    assert isinstance(d, Diarizer) and d.threshold == 0.7
    assert isinstance(YuNetFaces(RoleSpec("faces", "python:x", options={"seats": {"s1": [0, 0, 1, 1]}})), FaceTracker)
    assert isinstance(MotionASD(RoleSpec("asd", "python:x")), ActiveSpeaker)


# ----------------------------------------------------------------------------- ASR
@needs_whisper
async def test_faster_whisper_gets_most_words(meeting_pcm, truth):
    from unspoken.sensors.asr_local import FasterWhisperASR

    asr = FasterWhisperASR(options={"local_files_only": True})
    assert await asr.healthy()
    ref: list[str] = []
    hyp: list[str] = []
    latencies = []
    for line in truth["lines"]:
        tr = await asr.transcribe(line_pcm(meeting_pcm, line), SR)
        ref += normalize(line["text"])
        hyp += normalize(tr.text)
        latencies.append(tr.latency_ms)
        assert tr.lang == "en" and 0 < tr.conf <= 1 and tr.words and tr.words[0].start >= 0
    assert wer(ref, hyp) <= 0.15, (ref, hyp)
    assert all(0 < ms < 10_000 for ms in latencies)
    assert "deck" in hyp and "friday" in hyp and "record" in hyp


@needs_whisper
async def test_faster_whisper_empty_and_tiny_audio():
    from unspoken.sensors.asr_local import FasterWhisperASR

    asr = FasterWhisperASR(options={"local_files_only": True})
    tr = await asr.transcribe(b"\x00\x00" * 800, SR)
    assert tr.text == "" and tr.conf == 0.0


# ----------------------------------------------------------------------------- diarization
def test_voice_names_are_session_letters():
    assert [voice_name(i) for i in (0, 1, 25, 26, 27)] == ["voice-A", "voice-B", "voice-Z", "voice-AA", "voice-AB"]


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_online_clustering_assigns_updates_and_resets():
    d = EmbeddingDiarizer(model_path="/nope.onnx", threshold=0.6, max_voices=2)
    s = d.session()
    a, b = _unit([1, 0, 0, 0]), _unit([0, 1, 0, 0])
    assert s.assign(a).label == "voice-A"
    assert s.assign(b).label == "voice-B"
    near_a = _unit([0.95, 0.05, 0, 0])
    la = s.assign(near_a)
    assert la.label == "voice-A" and la.conf > 0.5
    far = _unit([0, 0, 1, 0])                        # a third voice, but max_voices=2: nearest wins
    assert s.assign(far).label in ("voice-A", "voice-B")
    assert s.voices == ["voice-A", "voice-B"]
    s.reset()
    assert s.voices == [] and s.assign(b).label == "voice-A"   # labels restart: nothing survives a reset
    assert s.assign(np.zeros(0, np.float32)).label == UNKNOWN


async def test_short_audio_is_unknown_without_touching_the_model():
    d = EmbeddingDiarizer(model_path="/nope.onnx", min_seconds=0.6)
    vl = await d.session().label(b"\x00\x00" * 4000, SR)        # 0.25 s
    assert vl.label == UNKNOWN and vl.conf == 0.0


@needs_speaker_model
async def test_wespeaker_separates_the_three_voices(meeting_pcm, truth):
    d = EmbeddingDiarizer()
    assert d.healthy()
    s = d.session()
    labels = [(await s.label(line_pcm(meeting_pcm, line), SR)).label for line in truth["lines"]]
    pure, n = purity([l["speaker"] for l in truth["lines"]], labels)
    assert pure >= 0.9 and n == 3, (labels, pure)
    assert labels[0] == "voice-A" and labels[1] == "voice-B" and labels[2] == "voice-C"
    assert 0 < s.last_ms < 5000
    s.reset()
    assert s.voices == []


# ----------------------------------------------------------------------------- faces
def test_iou_and_tracker_keep_ids_and_seats():
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0 and iou((0, 0, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)) == 0.0
    trk = IoUTracker(zones={"s1": (0, 0, 0.5, 1), "screen": (0.5, 0, 0.5, 1)}, max_lost_s=0.5)
    a = trk.update([((0.1, 0.2, 0.2, 0.3), 0.9, []), ((0.6, 0.2, 0.2, 0.3), 0.9, [])], 0.0)
    assert [(t.track_id, t.seat) for t in a] == [("f1", "s1"), ("f2", "screen")]
    b = trk.update([((0.12, 0.21, 0.2, 0.3), 0.9, [])], 0.2)          # f2 briefly lost: not reported, kept
    assert [t.track_id for t in b] == ["f1"]
    c = trk.update([((0.12, 0.21, 0.2, 0.3), 0.9, []), ((0.61, 0.2, 0.2, 0.3), 0.9, [])], 0.4)
    assert [t.track_id for t in c] == ["f1", "f2"]                     # same id when it comes back in time
    d = trk.update([((0.9, 0.9, 0.05, 0.05), 0.9, [])], 1.5)           # everyone else gone > max_lost_s
    assert [t.track_id for t in d] == ["f3"] and len(trk.tracks) == 1
    # sticky seat: a face drifting between zones keeps its last seat
    trk2 = IoUTracker(zones={"s1": (0, 0, 0.4, 1)})
    trk2.update([((0.1, 0.2, 0.2, 0.3), 0.9, [])], 0.0)
    out = trk2.update([((0.5, 0.2, 0.2, 0.3), 0.9, [])], 0.2)
    assert out == [] or out[0].seat in ("s1", None)


def test_yunet_detects_the_fixture_face_and_assigns_seats():
    import cv2

    img = cv2.imread(str(FACE_JPG))
    assert img is not None
    faces = YuNetFaces(seats={"left": [0, 0, 0.5, 1], "right": [0.5, 0, 0.5, 1]})
    assert faces.healthy()
    s = faces.session()
    t1 = s.update(img, 0.0)
    assert len(t1) == 1 and t1[0].seat == "left" and 0 < t1[0].bbox[2] < 0.5 and t1[0].speaking == 0.0
    t2 = s.update(img, 0.2)
    assert t2[0].track_id == t1[0].track_id and s.last_ms < 500
    room = np.full((360, 640, 3), 70, np.uint8)
    f = cv2.resize(img, (160, 160))
    room[100:260, 40:200] = f
    room[100:260, 440:600] = f
    s2 = faces.session()
    tracks = s2.update(room, 0.0)
    assert sorted(t.seat for t in tracks) == ["left", "right"]
    assert s2.update(np.full((360, 640, 3), 70, np.uint8), 0.2) == []   # no face, no track reported
    faces.configure_seats({"screen": (0.0, 0.0, 1.0, 1.0)})
    assert s2.update(img, 0.4)[0].seat == "screen"


# ----------------------------------------------------------------------------- ASD
def _frames(n: int, talker_moves: bool, rng) -> list[np.ndarray]:
    """64x36 frames with two faces; the first face's mouth region flickers when talker_moves."""
    frames = []
    base = rng.integers(60, 120, (36, 64, 3), dtype=np.uint8)
    for _ in range(n):
        f = base.copy()
        if talker_moves:
            mx, my, mw, mh = mouth_box((0.1, 0.2, 0.3, 0.5))
            x0, y0 = int(mx * 64), int(my * 36)
            x1, y1 = int((mx + mw) * 64), int((my + mh) * 36)
            f[y0:y1, x0:x1] = rng.integers(0, 255, (y1 - y0, x1 - x0, 3), dtype=np.uint8)
        frames.append(f)
    return frames


FACES = [FaceTrack("f1", (0.1, 0.2, 0.3, 0.5), seat="s1"), FaceTrack("f2", (0.6, 0.2, 0.3, 0.5), seat="s2")]
VOICE = float_to_pcm16(np.sin(np.linspace(0, 300, 3200)).astype(np.float32) * 0.3)
QUIET = b"\x00\x00" * 3200


def test_motion_asd_picks_the_moving_mouth_when_there_is_voice():
    rng = np.random.default_rng(1)
    s = MotionASD().session()
    s.audio_gate(QUIET)                          # settle the noise floor
    out = []
    for f in _frames(6, True, rng):
        out = s.update(f, 0.0, FACES, VOICE)
    by = {t.track_id: t.speaking for t in out}
    assert by["f1"] > 0.5 > by["f2"], by
    assert out[0].seat == "s1" and out[0].bbox == FACES[0].bbox


def test_motion_asd_is_quiet_without_voice_or_without_motion():
    rng = np.random.default_rng(2)
    s = MotionASD().session()
    s.audio_gate(QUIET)
    for f in _frames(6, True, rng):
        out = s.update(f, 0.0, FACES, QUIET)          # mouth moves but nobody is heard
    assert all(t.speaking < 0.2 for t in out)
    s2 = MotionASD().session()
    s2.audio_gate(QUIET)
    for f in _frames(6, False, rng):
        out = s2.update(f, 0.0, FACES, VOICE)         # voice but static faces: two faces, no one favoured
    assert all(t.speaking < 0.2 for t in out)


def test_motion_asd_single_face_floor_and_screen_exclusion():
    rng = np.random.default_rng(3)
    s = MotionASD().session()
    s.audio_gate(QUIET)
    one = [FaceTrack("f1", (0.1, 0.2, 0.3, 0.5), seat="s1")]
    for f in _frames(3, False, rng):
        out = s.update(f, 0.0, one, VOICE)
    assert out[0].speaking >= 0.6                     # the only person in the room is heard: it is them
    with_screen = one + [FaceTrack("f9", (0.6, 0.1, 0.3, 0.3), seat="screen")]
    for f in _frames(3, False, rng):
        out = s.update(f, 0.1, with_screen, VOICE)
    assert out[0].speaking >= 0.6 and out[1].speaking < 0.2   # the screen face does not count as a person
    assert s.update(None, 0.2, one, None)[0].speaking == 0.0  # no frame, unknown audio: nothing claimed
    s.reset()
    assert s._prev == {} and s._score == {}


async def test_asyncio_marker_sanity():
    await asyncio.sleep(0)
