"""start_sensors end to end with fakes: the fixture WAV through the real VAD, fake ASR/diarizer, a fake video
source with scripted faces and ASD -> MT.SEGMENT and MT.SPEAKERS events; stop() wipes sessions; disabled
roles; the replay CLI; Mp4FileSource on a generated clip; the local registry."""
from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import pytest

from unspoken.contracts import FaceTrack, RoleDisabled
from unspoken.core.bus import Bus
from unspoken.meeting.contracts import MT
from unspoken.sensors import SensorHandle, start_sensors
from unspoken.sensors.audio import WavFileSource
from unspoken.sensors.fakes import (FakeASD, FakeASR, FakeDiarizer, FakeFaceTracker, FakeRegistry,
                                    FakeVideoSource)
from unspoken.sensors.vad import EnergyVAD, SileroVAD

from .conftest import MEETING_WAV, FakeClock

SEATS = {"s1": [0.0, 0.0, 0.5, 1.0], "screen": [0.5, 0.0, 0.5, 0.5]}
PERSON = FaceTrack("f1", (0.1, 0.3, 0.2, 0.4))              # seat left to the pipeline (s1)
SCREEN_FACE = FaceTrack("f2", (0.7, 0.1, 0.15, 0.3), seat="screen")


def _wall_clock():
    base = time.monotonic()
    return lambda: time.monotonic() - base


async def _run(config, registry, clock=None, timeout=60.0):
    bus = Bus()
    handle = await start_sensors(bus, registry, config, clock or _wall_clock())
    await handle.wait(timeout=timeout)
    await bus.drain()
    await handle.stop()
    return bus, handle


async def test_fixture_replay_publishes_twelve_segments_in_order(truth):
    script = [l["text"] for l in truth["lines"]]
    asr = FakeASR(script)
    diar = FakeDiarizer()
    reg = FakeRegistry(vad=SileroVAD(), asr=asr, diarizer=diar, labels={"asr": "fake whisper"})
    config = {"audio": {"path": str(MEETING_WAV)}, "replay": {"realtime": False}, "seats": SEATS}
    bus, handle = await _run(config, reg)
    segs = [e for e in bus.history if e.topic == MT.SEGMENT]
    assert len(segs) == 12 and handle.stats.summary()["segments_published"] == 12
    assert handle.models == {"vad": "fake vad", "asr": "fake whisper", "diarization": "fake diarization",
                             "faces": "disabled", "asd": "disabled"}
    for ev, line in zip(segs, truth["lines"]):
        d = ev.data
        assert ev.source == "sensors.audio" and ev.ts == d["end"]
        assert d["text"] == line["text"] and d["lang"] == "en"
        assert abs(d["start"] - line["start"]) < 0.5 and abs(d["end"] - line["end"]) < 0.7
        assert d["voice"] in ("voice-A", "voice-B", "voice-C") and 0 < d["voice_conf"] <= 1
        assert d["asr_ms"] == 12.0 and d["diar_ms"] == 3.0 and 0 < d["asr_conf"] <= 1
        assert len(d["segment_id"]) == 10
        assert set(d) == {"segment_id", "start", "end", "text", "lang", "asr_conf", "voice", "voice_conf",
                          "asr_ms", "diar_ms"}
    assert [e.data["start"] for e in segs] == sorted(e.data["start"] for e in segs)
    # the ASR saw exactly the segment PCM (16 kHz) and the diarizer was reset at stop()
    assert all(sr == 16000 and n > 16000 for n, sr in asr.calls)
    assert diar.sessions[0].calls == 12 and diar.sessions[0].resets == 1
    assert handle.video_done.is_set() and handle.audio_done.is_set()
    assert bus.errors == []


async def test_empty_transcripts_are_not_published():
    reg = FakeRegistry(vad=EnergyVAD(), asr=FakeASR(""), diarizer=None)
    config = {"audio": {"path": str(MEETING_WAV)}, "replay": {"realtime": False}}
    bus, handle = await _run(config, reg)
    assert [e for e in bus.history if e.topic == MT.SEGMENT] == []
    s = handle.stats.summary()
    assert s["segments"] == 12 and s["segments_empty"] == 12 and s["segments_published"] == 0


async def test_no_diarizer_gives_voice_none_and_no_faces_gives_no_video():
    reg = FakeRegistry(vad=EnergyVAD(), asr=FakeASR("hello"))
    config = {"audio": {"path": str(MEETING_WAV)}, "replay": {"realtime": False}}
    bus, handle = await _run(config, reg)
    segs = [e for e in bus.history if e.topic == MT.SEGMENT]
    assert len(segs) == 12 and all(e.data["voice"] is None and e.data["voice_conf"] == 0.0 for e in segs)
    assert all(e.data["diar_ms"] == 0.0 for e in segs)
    assert [e for e in bus.history if e.topic == MT.SPEAKERS] == []


async def test_video_path_publishes_speakers_with_seats_and_screen_talking(truth):
    clock = _wall_clock()
    audio = WavFileSource(MEETING_WAV, clock=clock, realtime=False)
    video = FakeVideoSource(clock=clock, seconds=audio.duration, fps=10, pace=lambda: audio.media_time)
    faces = FakeFaceTracker([(0.0, [PERSON]), (20.0, [PERSON, SCREEN_FACE])])
    # the screen face "talks" after t=20 whenever there is voice; the person otherwise
    asd = FakeASD(talker=lambda t: "f2" if t >= 20.0 else "f1")
    reg = FakeRegistry(vad=SileroVAD(), asr=FakeASR("x"), diarizer=FakeDiarizer(), faces=faces, asd=asd)
    config = {"audio": {"source": audio}, "video": {"source": video}, "seats": SEATS, "speakers": {"hz": 5}}
    bus, handle = await _run(config, reg, clock)
    sp = [e for e in bus.history if e.topic == MT.SPEAKERS]
    assert 150 <= len(sp) <= 190, len(sp)                       # ~5 Hz over 36 s from a 10 fps source
    assert all(e.source == "sensors.video" and e.ts == e.data["t"] for e in sp)
    gaps = np.diff([e.data["t"] for e in sp])
    assert gaps.min() >= 0.18 and gaps.max() <= 0.31
    assert faces.seats_configured and set(faces.seats_configured) == {"s1", "screen"}
    early = [e for e in sp if e.data["t"] < 19.0]
    late = [e for e in sp if e.data["t"] > 21.0]
    assert all(e.data["face_count"] == 1 and e.data["faces"][0]["seat"] == "s1" for e in early)  # seat from zones
    assert all(e.data["face_count"] == 1 and len(e.data["faces"]) == 2 for e in late)            # screen not a person
    assert any(e.data["faces"][0]["speaking"] == 1.0 for e in early) and not any(e.data["screen_talking"] for e in early)
    assert any(e.data["screen_talking"] for e in late)
    # ASD got the PCM since its previous call (about 0.2 s = 6400 bytes) and nothing more
    seen = [n for n in asd.sessions[0].audio_seen if n]
    assert seen and max(seen) <= 16000 * 2 * 0.5
    s = handle.stats.summary()
    assert s["frames_skipped"] > 100 and s["speakers_published"] == len(sp)
    assert faces.sessions[0].resets == 1 and asd.sessions[0].resets == 1
    assert bus.errors == []


async def test_stop_cancels_a_live_run_and_wipes_sessions():
    clock = _wall_clock()
    audio = WavFileSource(MEETING_WAV, clock=clock, realtime=True)       # 36 s if left alone
    video = FakeVideoSource(clock=clock, seconds=36, fps=5)
    diar = FakeDiarizer()
    faces = FakeFaceTracker([(0.0, [PERSON])])
    reg = FakeRegistry(vad=EnergyVAD(), asr=FakeASR("x"), diarizer=diar, faces=faces, asd=FakeASD("f1"))
    bus = Bus()
    handle = await start_sensors(bus, reg, {"audio": {"source": audio}, "video": {"source": video}}, clock)
    await asyncio.sleep(0.4)
    assert not handle.audio_done.is_set()
    await handle.stop()
    assert handle.audio_done.is_set() and handle.video_done.is_set()
    assert all(t.done() for t in handle.tasks) and video.closed and audio.finished.is_set()
    assert diar.sessions[0].resets == 1 and faces.sessions[0].resets == 1
    await handle.stop()                                                  # idempotent


async def test_disabled_vad_or_asr_turns_the_audio_path_off():
    reg = FakeRegistry(asr=FakeASR("x"), faces=FakeFaceTracker())
    bus, handle = await _run({"audio": {"path": str(MEETING_WAV)}, "replay": {"realtime": False}}, reg)
    assert handle.models["vad"] == "disabled" and [e for e in bus.history if e.topic == MT.SEGMENT] == []


def test_fake_registry_disabled_roles_raise():
    reg = FakeRegistry(vad=EnergyVAD())
    assert reg.enabled("vad") and not reg.enabled("asr")
    with pytest.raises(RoleDisabled):
        reg.asr()


# ----------------------------------------------------------------------------- replay CLI
async def test_replay_cli_prints_json_lines(monkeypatch, capsys, truth):
    from unspoken.sensors import replay

    script = [l["text"] for l in truth["lines"]]
    monkeypatch.setattr(replay, "make_registry",
                        lambda profile, local, overrides=None: FakeRegistry(vad=SileroVAD(), asr=FakeASR(script),
                                                                             diarizer=FakeDiarizer()))
    rc = await replay.run(replay.build_parser().parse_args(["--audio", str(MEETING_WAV), "--fast", "--local"]))
    out, err = capsys.readouterr()
    lines = [json.loads(l) for l in out.strip().splitlines()]
    assert rc == 0 and len(lines) == 12
    assert lines[0]["topic"] == MT.SEGMENT and lines[0]["data"]["text"] == truth["lines"][0]["text"]
    assert "done:" in err and '"segments_published": 12' in err


# ----------------------------------------------------------------------------- video file source
async def test_mp4_source_paced_by_audio_position(tmp_path):
    import cv2

    from unspoken.sensors.video import Mp4FileSource, find_v4l2_device, resize_to

    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 180))
    if not w.isOpened():
        pytest.skip("no mp4v encoder in this OpenCV build")
    for i in range(90):   # 3 s
        frame = np.full((180, 320, 3), i * 2 % 255, np.uint8)
        w.write(frame)
    w.release()
    clock = FakeClock(10.0)
    pos = [0.0]
    src = Mp4FileSource(path, clock=clock, realtime=False, fps=5, width=160, height=90, pace=lambda: pos[0])

    async def advance():
        for k in range(1, 40):
            await asyncio.sleep(0.01)
            pos[0] = k * 0.1

    task = asyncio.create_task(advance())
    frames = [f async for f in src]
    await task
    assert 14 <= len(frames) <= 16
    assert frames[0].t == 10.0 and frames[0].image.shape == (90, 160, 3)
    assert all(b.t - a.t == pytest.approx(0.2, abs=0.05) for a, b in zip(frames, frames[1:]))
    assert frames[1].dropped == 5                                     # 30 fps -> 5 fps keeps every 6th
    assert find_v4l2_device("3") == 3 and find_v4l2_device("/dev/video9") == "/dev/video9"
    assert resize_to(frames[0].image, 160, 90).shape == (90, 160, 3)


# ----------------------------------------------------------------------------- local registry
async def test_sensors_registry_builds_python_providers_and_overrides():
    from unspoken.sensors.local_registry import SensorsRegistry

    reg = SensorsRegistry(overrides={"vad.provider": "python:unspoken.sensors.vad:EnergyVAD",
                                     "vad.options.min_silence_ms": 300, "asr.model": "Systran/faster-whisper-tiny.en"},
                          disabled={"diarization"})
    assert isinstance(reg.vad(), EnergyVAD) and reg.vad().settings.min_silence_ms == 300
    assert reg.spec("asr").model == "Systran/faster-whisper-tiny.en"
    assert not reg.enabled("diarization") and reg.enabled("faces") and not reg.enabled("llm")
    with pytest.raises(RoleDisabled):
        reg.diarizer()
    with pytest.raises(RoleDisabled):
        reg.decider()
    health = await reg.check(["vad", "asd", "diarization"])
    assert [(h.role, h.ok) for h in health] == [("vad", True), ("asd", True), ("diarization", False)]
    assert health[2].detail == "disabled" and reg.label("asd").startswith("MotionASD")
    assert isinstance(SensorHandle(), SensorHandle)
