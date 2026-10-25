"""Recording telemetry shows actual public observations and obeys stream privacy barriers."""
from __future__ import annotations

import json

import pytest

from unspoken.core.events import Event
from unspoken.meeting.contracts import MT
from unspoken.meeting.web import server as web
from unspoken.meeting.web.dev_fake import FakeMeetingRuntime


class SensorRuntime(FakeMeetingRuntime):
    def snapshot(self):
        return {**super().snapshot(), "sensors": {"models": {}, "stats": {}}}


@pytest.fixture
def runtime():
    return SensorRuntime(fast=True)


def segment(start=1.0, end=3.0, **extra):
    return Event(MT.SEGMENT, {
        "start": start, "end": end, "text": "PRIVATE RAW TRANSCRIPT", "asr_ms": 94.5,
        "diar_ms": 28.2, "voice": "voice-A", "voice_conf": 0.82, **extra,
    }, source="sensors.audio", ts=end)


def vision(t=3.0, **extra):
    return Event(MT.SPEAKERS, {
        "t": t, "face_count": 1, "screen_talking": False,
        "faces": [{"track_id": "f1", "seat": "left", "bbox": [0.2, 0.3, 0.1, 0.2],
                   "speaking": 0.8, "name": "PRIVATE FACE NAME", "image": "PRIVATE IMAGE"}],
        **extra,
    }, source="sensors.video", ts=t)


def queued(q):
    result = []
    while not q.empty():
        result.append(q.get_nowait())
    return result


async def test_only_allowlisted_sensor_observations_reach_dashboard(runtime):
    hub = web.StreamHub(runtime)
    q = hub.connect()
    await hub._on_event(vision())
    await hub._on_event(segment(pcm="PRIVATE AUDIO", arbitrary="PRIVATE METADATA"))
    frames = queued(q)
    assert {frame["topic"] for frame in frames} == {web.TELEMETRY_TOPIC}
    assert "PRIVATE" not in json.dumps(frames)
    data = frames[-1]["data"]
    assert data["counts"] == {"segments": 1, "frames": 1}
    assert data["audio"] == {
        "start": 1.0, "end": 3.0, "asr_ms": 94.5, "diar_ms": 28.2,
        "voice": "voice-A", "voice_conf": 0.82,
    }
    assert data["vision"] == {
        "t": 3.0, "face_count": 1, "screen_talking": False,
        "faces": [{"track_id": "f1", "seat": "left", "bbox": [0.2, 0.3, 0.1, 0.2], "speaking": 0.8}],
    }
    # A reconnect has the latest projection, not a fabricated model-start event.
    assert hub.browser_snapshot()["telemetry"] == data
    data["vision"]["faces"][0]["bbox"][0] = 0.99
    assert hub.browser_snapshot()["telemetry"]["vision"]["faces"][0]["bbox"][0] == 0.2


async def test_video_throttles_but_counts_all_observations_and_audio_sends_immediately(runtime, monkeypatch):
    monkeypatch.setattr(web, "TELEMETRY_MIN_INTERVAL", 100)
    hub = web.StreamHub(runtime)
    q = hub.connect()
    for t in range(20):
        await hub._on_event(vision(float(t)))
    frames = queued(q)
    assert len(frames) == 1
    assert frames[0]["data"]["counts"]["frames"] == 1
    assert hub.browser_snapshot()["telemetry"]["vision"]["t"] == 19
    await hub._on_event(segment(18, 20))
    frame = q.get_nowait()
    assert frame["data"]["counts"] == {"segments": 1, "frames": 20}
    assert frame["data"]["vision"]["t"] == 19


@pytest.mark.parametrize("privacy", ["off_record", "paused"])
async def test_privacy_clears_queued_held_and_snapshot_telemetry(runtime, privacy):
    hub = web.StreamHub(runtime)
    q = hub.connect()
    await hub._on_event(vision())
    held_by_pump = q.get_nowait()
    await hub._on_event(segment())
    runtime.privacy = {"state": privacy}
    await hub._on_event(Event(MT.PRIVACY, dict(runtime.privacy), source="test", ts=60))
    assert hub.ready(held_by_pump) is None
    assert not any(frame["topic"] == web.TELEMETRY_TOPIC for frame in queued(q))
    assert hub.browser_snapshot()["telemetry"] is None
    private_segment = segment(60, 61)
    await hub._on_event(private_segment)
    await hub._on_event(vision(61))
    assert hub.browser_snapshot()["telemetry"] is None
    runtime.privacy = {"state": "on"}
    await hub._on_event(Event(MT.PRIVACY, dict(runtime.privacy), source="test", ts=62))
    queued(q)
    assert hub.browser_snapshot()["telemetry"] == {
        "audio": None, "vision": None, "counts": {"segments": 0, "frames": 0},
    }
    # Both old queued events and newly returned segments spanning the private interval are rejected.
    await hub._on_event(private_segment)
    await hub._on_event(segment(61, 63))
    await hub._on_event(vision(61))
    assert not any(frame["topic"] == web.TELEMETRY_TOPIC for frame in queued(q))
    await hub._on_event(segment(63, 65))
    assert q.get_nowait()["data"]["counts"] == {"segments": 1, "frames": 0}


async def test_reset_removes_telemetry_before_teardown_and_rejects_old_events(runtime):
    hub = web.StreamHub(runtime)
    q = hub.connect()
    old = segment()
    await hub._on_event(old)
    held_by_pump = q.get_nowait()
    hub.begin_reset()
    assert hub.browser_snapshot()["telemetry"] is None
    assert hub.ready(held_by_pump) is None
    await hub._on_event(vision(4))  # sensor finishing during teardown
    hub.end_reset()
    queued(q)
    await hub._on_event(old)
    assert hub.browser_snapshot()["telemetry"] == {
        "audio": None, "vision": None, "counts": {"segments": 0, "frames": 0},
    }
    assert not any(frame["topic"] == web.TELEMETRY_TOPIC for frame in queued(q))


async def test_invalid_geometry_and_non_numeric_timings_do_not_enter_telemetry(runtime):
    hub = web.StreamHub(runtime)
    await hub._on_event(vision(faces=[
        {"track_id": "PRIVATE NAME", "bbox": [float("nan"), 0, 1, 1], "speaking": 1},
        {"track_id": "f2", "seat": "left", "bbox": [0, 0, 1, 1], "speaking": float("inf")},
    ]))
    await hub._on_event(segment(asr_ms="PRIVATE LATENCY", diar_ms=False, voice="PRIVATE VOICE"))
    data = hub.browser_snapshot()["telemetry"]
    assert len(data["vision"]["faces"]) == 1
    assert data["vision"]["faces"][0]["speaking"] is None
    assert data["audio"]["asr_ms"] is data["audio"]["diar_ms"] is data["audio"]["voice"] is None
    assert "PRIVATE" not in json.dumps(data, allow_nan=False)


async def test_non_sensor_screen_fake_does_not_claim_real_model_observations():
    hub = web.StreamHub(FakeMeetingRuntime(fast=True))
    q = hub.connect()
    await hub._on_event(segment())
    await hub._on_event(vision())
    assert q.empty()
    assert "telemetry" not in hub.browser_snapshot()
