from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from unspoken.core.events import Event
from unspoken.meeting import demo
from unspoken.meeting.contracts import MT
from unspoken.meeting.fakes import FakeNotifier
from unspoken.models import load_registry


@pytest.fixture
async def setup(tmp_path):
    runtime = demo.DemoRuntime(profile="fake", registry=load_registry(profile="fake"),
                               fake_io=True, notifier=FakeNotifier(echo=False), echo=False)
    app = demo.create_demo_app(runtime, tmp_path / "uploads")
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://127.0.0.1") as client:
            yield runtime, app.state.replay, client


def media_info(path):
    return demo.MediaInfo(path=path, duration=5, origin=0, audio_start=0, video_start=0,
                          audio_end=5, video_end=5, audio_stream_index=1, video_stream_index=0,
                          sample_rate=16000, channels=1, width=64, height=48, fps=25)


async def test_demo_rejects_fake_models_as_real_readiness(setup, tmp_path, monkeypatch):
    runtime, controller, client = setup
    path = tmp_path / "meeting.mp4"
    path.write_bytes(b"fixture-media")
    monkeypatch.setattr(demo, "probe_media", media_info)
    await controller.load(path)
    state = (await client.get("/api/replay")).json()
    assert not state["ready"]
    assert "asr" in state["unavailable_roles"]
    response = await client.post("/api/replay/control", json={"action": "play"})
    assert response.status_code == 409
    assert "not ready" in response.json()["detail"]
    assert str(tmp_path) not in str(state)
    assert runtime.snapshot()["egress"]["total_bytes"] == 0


async def test_browser_origin_and_host_boundaries(setup):
    _, _, client = setup
    response = await client.post("/api/replay/control", json={"action": "pause"},
                                 headers={"Origin": "https://foreign.example"})
    assert response.status_code == 403
    response = await client.get("/api/replay", headers={"Host": "foreign.example"})
    assert response.status_code == 400
    response = await client.post("/api/replay/control", json={"action": "pause"},
                                 headers={"Origin": "http://127.0.0.1"})
    assert response.status_code == 200


async def test_uploaded_media_has_an_opaque_url_and_range_support(setup, monkeypatch):
    _, controller, client = setup
    monkeypatch.setattr(demo, "probe_media", media_info)
    response = await client.post("/api/replay/upload",
                                  files={"file": ("private-meeting.mp4", b"0123456789", "video/mp4")})
    assert response.status_code == 200
    media = response.json()["media"]
    assert "private-meeting" not in media["url"]
    assert str(controller.media_dir) not in str(response.json())
    response = await client.get(media["url"], headers={"Range": "bytes=2-5"})
    assert response.status_code == 206 and response.content == b"2345"
    assert (await client.get("/api/replay/media/missing")).status_code == 404


async def test_invalid_upload_is_removed_without_resetting_the_current_meeting(setup, monkeypatch):
    _, controller, client = setup
    before = controller.generation

    def reject(path):
        raise demo.MediaValidationError("No audio.")

    monkeypatch.setattr(demo, "probe_media", reject)
    response = await client.post("/api/replay/upload", files={"file": ("silent.mp4", b"bad", "video/mp4")})
    assert response.status_code == 422
    assert controller.generation == before
    assert list(controller.media_dir.iterdir()) == []
    assert (await client.post("/api/replay/upload", files={"file": ("bad.txt", b"bad")})).status_code == 422


async def test_pause_restart_and_generation_gate_discard_old_output(setup, tmp_path, monkeypatch):
    runtime, controller, _ = setup
    path = tmp_path / "meeting.mp4"
    path.write_bytes(b"fixture-media")
    monkeypatch.setattr(demo, "probe_media", media_info)
    await controller.load(path)
    monkeypatch.setattr(controller, "unavailable", lambda: [])
    stopped = []

    async def start(*args):
        async def stop():
            stopped.append(True)
        return SimpleNamespace(tasks=[], stats=SimpleNamespace(errors=0), audio_done=asyncio.Event(),
                               video_done=asyncio.Event(), stop=stop)

    monkeypatch.setattr(demo, "start_sensors", start)
    await controller.control("play")
    old = demo.GenerationBus(controller, controller.generation)
    await controller.control("pause")
    assert controller.state == "paused" and not controller.clock.playing
    runtime.state.transcript.append({"text": "Old take"})
    await controller.control("restart")
    assert stopped and controller.clock.now() == 0
    assert runtime.transcript() == [] and controller.state == "ready"
    await old.publish(Event("test.stale", {"text": "Late ASR output"}, source="test"))
    assert runtime.bus.find("test.stale") == []
    fresh = demo.GenerationBus(controller, controller.generation)
    await fresh.publish(Event("test.fresh", {}, source="test"))
    assert len(runtime.bus.find("test.fresh")) == 1


async def test_question_is_cancelled_when_privacy_changes(setup, monkeypatch):
    import strands

    runtime, controller, _ = setup
    entered = asyncio.Event()

    class Agent:
        def __init__(self, **kwargs):
            pass

        async def invoke_async(self, prompt):
            entered.set()
            await asyncio.Future()

    monkeypatch.setattr(strands, "Agent", Agent)
    monkeypatch.setattr(runtime.registry, "strands_model", lambda role: object())
    runtime.state.transcript.append({"text": "On-record meeting content", "speaker": {}})
    task = asyncio.create_task(controller.ask("What was decided?"))
    await entered.wait()
    await controller._privacy_changed(Event(MT.PRIVACY, {"state": "off_record"}, source="test"))
    with pytest.raises(ValueError, match="discarded"):
        await task
    assert not controller._questions


async def test_health_failure_is_not_overwritten_by_finishing_source(setup, monkeypatch):
    runtime, controller, _ = setup
    done, entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    done.set()
    runtime._sensors = SimpleNamespace(tasks=[], stats=SimpleNamespace(errors=0),
                                       audio_done=done, video_done=done)

    async def drain(*args, **kwargs):
        entered.set()
        await release.wait()

    monkeypatch.setattr(runtime.bus, "drain", drain)
    watcher = asyncio.create_task(controller._watch(controller.generation))
    await entered.wait()
    controller.state = "error"
    controller.error = "A required model is down."
    release.set()
    await watcher
    assert controller.state == "error"


def segment(text, start, end, *, voice="voice-A"):
    return Event(MT.SEGMENT, {"start": start, "end": end, "text": text, "voice": voice,
                              "voice_conf": 0.9, "lang": "en"}, source="sensors.audio", ts=end)


async def test_privacy_wipes_spans_caches_previews_and_not_older_transcript(setup, monkeypatch):
    runtime, controller, _ = setup
    now = [100.0]
    monkeypatch.setattr(controller.clock, "now", lambda: now[0])
    for text, start, end in [("Earlier public context", 10, 11), ("CROSSING_SECRET", 39, 41),
                             ("RECENT_SECRET", 90, 91)]:
        await runtime.bus.publish(Event(MT.UTTERANCE, {
            "utterance_id": text, "text": text, "start": start, "end": end,
            "speaker": {}, "source": "person"}, source="test", ts=end))
        await runtime.bus.drain()
    await runtime.bus.publish(Event(MT.PROPOSAL, {
        "mode": "act", "action": "create_task", "args": {"text": "RECENT_SECRET"}, "confidence": 0.5},
        source="test", ts=91))
    await runtime.bus.publish(Event(MT.PROPOSAL, {
        "mode": "act", "action": "set_light", "args": {"scene": "RECENT_SECRET"}, "confidence": 1},
        source="test", ts=91))
    await runtime.bus.drain()
    assert runtime.agent("actuator").pending
    assert runtime.notifier.posts and runtime.notifier.dms
    # Replay's off-record floor is 60 seconds, even if general room settings were changed.
    runtime.settings.wipe_seconds = 0
    await asyncio.wait_for(runtime.set_privacy("off_record", "RECENT_SECRET"), 2)
    assert [u["text"] for u in runtime.transcript()] == ["Earlier public context"]
    assert [c["text"] for c in runtime.state.screen["captions"]] == ["Earlier public context"]
    assert not runtime.state.items and not runtime.state.screen["cards"]
    assert not runtime.notifier.posts and not runtime.notifier.dms
    assert "SECRET" not in json.dumps(runtime.state.devices)
    assert "SECRET" not in json.dumps([device.commands for device in runtime.devices.values()])
    assert not runtime.agent("policy")._utterances and not runtime.agent("actuator").pending
    assert "SECRET" not in json.dumps([e.to_json() for e in runtime.bus.history])
    assert runtime.bus.find(MT.PRIVACY)[-1].data["wiped_from"] == 40
    # No private check-in gets into fusion, nor private diagnostics into history.
    await runtime.bus.publish(segment("Hi, I'm Secret.", 101, 102))
    await runtime.bus.publish(Event(MT.LOG, {"text": "PRIVATE_SECRET"}, source="test", ts=102))
    await runtime.bus.drain()
    assert not runtime.state.people
    now[0] = 120
    await runtime.set_privacy("on", "resumed")
    assert "SECRET" not in json.dumps(runtime.snapshot())
    assert "SECRET" not in json.dumps([e.to_json() for e in runtime.bus.history])
    assert not runtime.bus.errors


async def test_delayed_asr_and_queued_audio_cross_privacy_using_real_sensor_pipeline(setup, monkeypatch):
    """Actual sensor worker + fusion/reflex/runtime; only source, VAD and ASR are doubles.

    This deliberately does not claim model readiness or run hardware/inference servers.
    The worker's first ASR is in flight and its second segment is queued during privacy.
    """
    from unspoken.contracts import SpeechSegment, Transcript
    from unspoken.sensors.audio import AudioChunk
    from unspoken.sensors.fakes import FakeRegistry

    runtime, controller, _ = setup
    entered, release = asyncio.Event(), asyncio.Event()
    now = [100.0]
    monkeypatch.setattr(controller.clock, "now", lambda: now[0])

    class Audio:
        async def __aiter__(self):
            for t in (99, 105, 130):
                yield AudioChunk(b"\0\0" * 16000, 16000, t)

        async def aclose(self):
            pass

    class VAD:
        def stream(self, sample_rate):
            self.times = iter(((0, 1), (6, 7), (31, 32)))
            return self

        def feed(self, pcm):
            start, end = next(self.times)
            return [SpeechSegment(pcm16=pcm, sample_rate=16000, start=start, end=end)]

        def flush(self):
            return []

    class ASR:
        def __init__(self):
            self.calls = 0

        async def transcribe(self, pcm, sample_rate):
            self.calls += 1
            if self.calls == 1:
                entered.set()
                await release.wait()
            return Transcript(text=["WIPED_SECRET", "PRIVATE_SECRET", "Fresh public discussion"][self.calls - 1],
                              lang="en", conf=0.9)

    asr = ASR()
    runtime._sensors = await demo.start_sensors(
        demo.GenerationBus(controller, controller.generation), FakeRegistry(vad=VAD(), asr=asr),
        {"audio": {"source": Audio()}}, controller.clock.now)
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await runtime.set_privacy("off_record", "operator")
        now[0] = 120
        await runtime.set_privacy("on", "operator")
        release.set()
        await runtime._sensors.wait(timeout=2)
        await runtime.bus.drain()
        assert asr.calls == 3
        assert [u["text"] for u in runtime.transcript()] == ["Fresh public discussion"]
        assert "SECRET" not in json.dumps([e.to_json() for e in runtime.bus.history])
        assert not runtime.bus.errors
    finally:
        release.set()
        await runtime.stop_sensors()


async def test_backlogged_straddling_segments_stay_private_but_future_input_resumes(setup, monkeypatch):
    runtime, controller, _ = setup
    now = [100.0]
    monkeypatch.setattr(controller.clock, "now", lambda: now[0])
    gate = demo.GenerationBus(controller, controller.generation)
    await runtime.set_privacy("off_record", "operator")
    now[0] = 120
    await runtime.set_privacy("on", "operator")
    for start, end in [(39, 41), (80, 90), (100, 105), (119, 121), (120, 122)]:
        await gate.publish(segment("SECRET_BACKLOG", start, end))
    await gate.publish(segment("Fresh public speech", 121, 122))
    await runtime.bus.drain()
    assert [u["text"] for u in runtime.transcript()] == ["Fresh public speech"]
    assert "SECRET" not in json.dumps([e.to_json() for e in runtime.bus.history])
    # Neither source teardown nor clock reset/seek was needed for either transition.
    assert controller.generation == 0 and controller.clock.generation == 0


async def test_starting_playback_preserves_prior_privacy_without_poisoning_future_input(setup, tmp_path, monkeypatch):
    runtime, controller, _ = setup
    path = tmp_path / "meeting.mp4"
    path.write_bytes(b"fixture-media")
    monkeypatch.setattr(demo, "probe_media", media_info)
    monkeypatch.setattr(controller, "unavailable", lambda: [])
    await controller.load(path)

    async def start(*args):
        async def stop():
            pass
        return SimpleNamespace(tasks=[], stats=SimpleNamespace(errors=0), audio_done=asyncio.Event(),
                               video_done=asyncio.Event(), stop=stop)

    monkeypatch.setattr(demo, "start_sensors", start)  # control wiring only, not a real-model readiness claim
    await runtime.set_privacy("off_record", "operator")
    generation = controller.generation
    await controller.control("play")
    assert runtime.state.privacy["state"] == "off_record"
    gate = demo.GenerationBus(controller, generation)
    await gate.publish(segment("PRIVATE_SECRET", 0, 1))
    await gate.publish(segment("back on the record", 1, 2))
    await runtime.bus.drain()
    assert runtime.state.privacy["state"] == "on"
    await gate.publish(segment("New public speech", 3, 4))
    await runtime.bus.drain()
    assert [u["text"] for u in runtime.transcript()] == ["New public speech"]
    assert generation == controller.generation


@pytest.mark.parametrize("privacy", ["off_record", "paused"])
async def test_private_start_has_no_transient_on_state_or_stream_event(setup, tmp_path, monkeypatch, privacy):
    """Observe every publication/broadcast, not only the state after drain/restoration.

    Runtime agents and StreamHub are real; media decoding/model readiness are doubles.
    """
    runtime, controller, client = setup
    path = tmp_path / "meeting.mp4"
    path.write_bytes(b"fixture-media")
    monkeypatch.setattr(demo, "probe_media", media_info)
    monkeypatch.setattr(controller, "unavailable", lambda: [])
    await controller.load(path)
    await runtime.set_privacy(privacy, "operator")
    before_epoch = controller.privacy_epoch
    publications, broadcasts, samples, capture_states = [], [], [], []
    publish, broadcast = runtime.bus.publish, controller._hub.broadcast

    async def observe(event):
        await publish(event)
        publications.append(json.loads(json.dumps(event.to_json())))
        samples.append((event.topic, runtime.state.privacy["state"], runtime.state.screen["privacy"]))

    def observe_broadcast(payload):
        broadcasts.append(json.loads(json.dumps(payload)))
        broadcast(payload)

    async def start(bus, registry, config, clock):
        capture_states.append((runtime.state.privacy["state"], runtime.state.screen["privacy"]))
        assert isinstance(config["audio"]["source"], demo.ReplayAudioSource)
        assert isinstance(config["video"]["source"], demo.ReplayVideoSource)

        async def stop():
            pass
        return SimpleNamespace(tasks=[], stats=SimpleNamespace(errors=0), audio_done=asyncio.Event(),
                               video_done=asyncio.Event(), stop=stop)

    monkeypatch.setattr(runtime.bus, "publish", observe)
    monkeypatch.setattr(controller._hub, "broadcast", observe_broadcast)
    monkeypatch.setattr(demo, "start_sensors", start)
    response = await client.post("/api/replay/control", json={"action": "play"})
    assert response.status_code == 200
    await runtime.bus.drain()
    controller._hub._send_snapshot()  # exercise the trailing snapshot before any explicit Resume

    # A temporary runtime flip is dangerous even if StreamHub masks the main screen:
    # replay polling and nested snapshot.demo are also used to mute/hide source media.
    assert samples and all(state == screen == privacy for _, state, screen in samples), samples
    assert capture_states == [(privacy, privacy)]
    assert any(event["topic"] == MT.STATUS and event["data"]["state"] == "started" for event in publications)
    assert all(event["data"]["privacy"] == privacy for event in publications if event["topic"] == MT.SCREEN)
    assert all(event["data"]["state"] == privacy for event in publications if event["topic"] == MT.PRIVACY)
    assert broadcasts
    for payload in broadcasts:
        if payload["topic"] == "ui.reset":
            assert payload["data"]["privacy"] == privacy
        if payload["topic"] == "ui.snapshot":
            snapshot = payload["data"]["snapshot"]
            assert snapshot["privacy"]["state"] == snapshot["screen"]["privacy"] == snapshot["demo"]["privacy"] == privacy
        if payload["topic"] == MT.SCREEN:
            assert payload["data"]["privacy"] == privacy
            assert not payload["data"]["captions"] and not payload["data"]["cards"]
    assert controller.privacy_epoch == before_epoch  # Play itself is not a privacy transition.
    assert not runtime.bus.errors


async def test_replay_privacy_epoch_is_exposed_and_monotonic_across_resumes_and_reset(setup):
    runtime, controller, client = setup
    first = (await client.get("/api/replay")).json()
    assert isinstance(first["privacy_epoch"], int)
    previous = first["privacy_epoch"]
    generation = first["generation"]
    for privacy in ("off_record", "on", "paused", "on"):
        await runtime.set_privacy(privacy, "operator")
        replay = (await client.get("/api/replay")).json()
        assert replay["privacy"] == privacy
        assert replay["privacy_epoch"] > previous
        assert replay["generation"] == generation
        previous = replay["privacy_epoch"]
        nested = (await client.get("/api/snapshot")).json()["snapshot"]["demo"]
        assert nested["privacy_epoch"] == previous
        # Playback pause is not a privacy transition, and control responses carry
        # the same epoch as polling and nested runtime snapshots.
        paused = (await client.post("/api/replay/control", json={"action": "pause"})).json()
        assert paused["privacy_epoch"] == previous
    restarted = (await client.post("/api/replay/control", json={"action": "restart"})).json()
    assert restarted["privacy_epoch"] > previous  # never reset this counter back to zero
    assert restarted["generation"] > generation
    assert restarted["privacy_epoch"] == controller.snapshot()["privacy_epoch"]
    assert first["privacy_epoch"] < restarted["privacy_epoch"]  # a delayed old response is identifiable


async def test_resume_during_source_start_is_not_overwritten_and_control_returns_latest_epoch(setup, tmp_path, monkeypatch):
    runtime, controller, client = setup
    path = tmp_path / "meeting.mp4"
    path.write_bytes(b"fixture-media")
    monkeypatch.setattr(demo, "probe_media", media_info)
    monkeypatch.setattr(controller, "unavailable", lambda: [])
    await controller.load(path)
    await runtime.set_privacy("off_record", "operator")
    private_epoch = controller.privacy_epoch
    entered, release = asyncio.Event(), asyncio.Event()
    startup_states = []

    async def start(*args):
        startup_states.append((runtime.state.privacy["state"], runtime.state.screen["privacy"]))
        entered.set()
        await release.wait()

        async def stop():
            pass
        return SimpleNamespace(tasks=[], stats=SimpleNamespace(errors=0), audio_done=asyncio.Event(),
                               video_done=asyncio.Event(), stop=stop)

    monkeypatch.setattr(demo, "start_sensors", start)  # only decoder/provider startup is a double
    playing = asyncio.create_task(client.post("/api/replay/control", json={"action": "play"}))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert startup_states == [("off_record", "off_record")]
        before_resume = (await client.get("/api/replay")).json()
        assert before_resume["privacy"] == "off_record" and before_resume["privacy_epoch"] == private_epoch
        assert runtime.state.status["state"] == "started"
        response = await client.post("/api/chat", json={"command": "resume", "user": "operator"})
        assert response.status_code == 200
        resumed_epoch = controller.privacy_epoch
        assert resumed_epoch > private_epoch
    finally:
        release.set()
        response = await asyncio.wait_for(playing, 2)
    assert response.status_code == 200
    assert response.json()["privacy"] == "on" and response.json()["privacy_epoch"] == resumed_epoch
    assert runtime.state.privacy["state"] == runtime.state.screen["privacy"] == "on"
    assert not runtime.bus.errors


@pytest.mark.parametrize("media_kind", ["voice", "frames"])
async def test_private_spoken_resume_is_ephemeral_and_media_cannot_resume(setup, monkeypatch, media_kind):
    runtime, controller, _ = setup
    now = [100.0]
    monkeypatch.setattr(controller.clock, "now", lambda: now[0])
    await runtime.set_privacy("off_record", "operator")
    if media_kind == "frames":
        await runtime.bus.publish(Event(MT.SPEAKERS, {
            "t": 102, "faces": [], "screen_talking": True}, source="sensors.video", ts=102))
        await runtime.bus.drain()
    await runtime.bus.publish(segment("back on the record and PRIVATE_SECRET", 101, 102,
                                      voice="voice-screen-1" if media_kind == "voice" else "voice-A"))
    await runtime.bus.drain()
    assert runtime.state.privacy["state"] == "off_record"
    # A stale resume heard before the off-record request cannot undo the request.
    await runtime.bus.publish(segment("back on the record", 90, 91))
    await runtime.bus.drain()
    assert runtime.state.privacy["state"] == "off_record"
    now[0] = 105
    await runtime.bus.publish(segment("back on the record and PRIVATE_SECRET", 104, 105))
    await runtime.bus.drain()
    assert runtime.state.privacy["state"] == "on"
    assert runtime.transcript() == []
    assert runtime.bus.find(MT.PRIVACY)[-1].data["state"] == "on"
    assert "PRIVATE_SECRET" not in json.dumps([e.to_json() for e in runtime.bus.history])
    assert not runtime.bus.find(MT.PROPOSAL)
    assert not runtime.bus.errors


async def cross_privacy_while_provider_ignores_cancel(runtime, controller, cancelled, release, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(controller.clock, "now", lambda: now[0])
    # Real runtime APIs must finish even if a quarantined provider ignores cancellation.
    old_workers = list(runtime.bus._workers)
    off = asyncio.create_task(runtime.set_privacy("off_record", "operator"))
    await asyncio.wait_for(cancelled.wait(), 2)
    await asyncio.wait_for(off, 2)
    assert runtime.state.privacy["state"] == "off_record"
    now[0] = 120
    await asyncio.wait_for(runtime.set_privacy("on", "operator"), 2)
    release.set()
    await asyncio.wait_for(asyncio.gather(*old_workers, return_exceptions=True), 2)
    await runtime.bus.drain()


async def test_inflight_decision_cannot_publish_or_act_after_offrecord_resume(setup, monkeypatch):
    from unspoken.meeting.fakes import KeywordDecider

    runtime, controller, _ = setup
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class SlowDecider(KeywordDecider):
        async def ask(self, state, questions):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()  # model/thread can ignore cancellation and still return
            return await super().ask(state, questions)

    provider = SlowDecider()
    monkeypatch.setattr(runtime.registry, "decider", lambda role="decider": provider)
    runtime.settings.escalate = False
    await runtime.bus.publish(segment("We decided PRIVATE_DECISION.", 90, 91))
    await asyncio.wait_for(entered.wait(), 2)
    cause = runtime.bus.find(MT.UTTERANCE)[-1]
    await cross_privacy_while_provider_ignores_cancel(runtime, controller, cancelled, release, monkeypatch)
    assert len(provider.calls) == 1  # it actually returned a decision after resume
    # Independently published descendants must also carry valid cause provenance.
    await runtime.bus.publish(Event(MT.PROPOSAL, {
        "mode": "act", "action": "log_decision", "args": {"text": "PRIVATE_DECISION"}, "confidence": 1},
        source="test.late-model", ts=130, cause=cause.id))
    await runtime.bus.drain()
    assert not runtime.items() and not runtime.notifier.posts and not runtime.notifier.dms
    assert not runtime.bus.find(MT.PROPOSAL)
    assert "PRIVATE_DECISION" not in json.dumps([e.to_json() for e in runtime.bus.history])
    assert [e.data["state"] for e in runtime.bus.find(MT.PRIVACY)] == ["off_record", "on"]
    assert not runtime.bus.errors
    # New work is not permanently disabled by the fence.
    await runtime.bus.publish(segment("We decided to use the public design.", 130, 131))
    await runtime.bus.drain()
    assert runtime.items(kind="decision") and runtime.notifier.posts


@pytest.mark.parametrize("fails", [False, True])
async def test_inflight_actuator_answer_cannot_mutate_current_state_after_resume(setup, monkeypatch, caplog, fails):
    runtime, controller, _ = setup
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    returned = asyncio.Event()

    async def search(question, k=3):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        returned.set()
        if fails:
            raise RuntimeError("PRIVATE_ANSWER_FAILURE")
        return [{"doc": "local.txt", "text": "PRIVATE_ANSWER", "score": 1}]

    monkeypatch.setattr(runtime.docs, "search", search)
    await runtime.bus.publish(Event(MT.PROPOSAL, {
        "mode": "act", "action": "answer_question", "args": {"question": "PRIVATE_QUESTION"}, "confidence": 1},
        source="test", ts=90))
    await asyncio.wait_for(entered.wait(), 2)
    await cross_privacy_while_provider_ignores_cancel(runtime, controller, cancelled, release, monkeypatch)
    assert returned.is_set()
    assert not runtime.items() and not runtime.state.screen["cards"]
    assert not runtime.notifier.posts and not runtime.notifier.dms
    assert "PRIVATE_" not in json.dumps(runtime.snapshot())
    assert "PRIVATE_" not in json.dumps([e.to_json() for e in runtime.bus.history])
    assert "PRIVATE_" not in caplog.text
    assert not runtime.bus.errors


async def test_handler_failure_diagnostics_do_not_print_model_text(setup, monkeypatch, caplog):
    runtime, controller, _ = setup

    async def fail(question, k=3):
        raise RuntimeError("PROMPT_SECRET")

    monkeypatch.setattr(runtime.docs, "search", fail)
    monkeypatch.setattr(controller.clock, "now", lambda: 100.0)
    await runtime.bus.publish(Event(MT.VERDICT, {
        "verdict": "act", "action": "answer_question", "args": {"question": "Public question"}},
        source="test", ts=90))
    await runtime.bus.drain()
    assert runtime.bus.errors and runtime.bus.errors[0][2].__context__ is None
    assert "PROMPT_SECRET" not in caplog.text
    assert "PROMPT_SECRET" not in json.dumps([e.to_json() for e in runtime.bus.history])
    await runtime.set_privacy("off_record", "operator")
    assert not runtime.bus.errors


@pytest.mark.parametrize("api", ["browser_question", "runtime_recap", "chat_recap"])
async def test_answer_returning_after_resume_is_discarded(setup, monkeypatch, api):
    import strands

    runtime, controller, _ = setup
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    returned = asyncio.Event()

    class Agent:
        def __init__(self, **kwargs):
            pass

        async def invoke_async(self, prompt):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            returned.set()
            return "PRIVATE_ANSWER"

    monkeypatch.setattr(strands, "Agent", Agent)
    monkeypatch.setattr(runtime.registry, "strands_model", lambda role: object())
    monkeypatch.setattr(runtime, "_llm_factory", lambda: lambda: object())
    runtime.state.transcript.append({"text": "PRIVATE_CONTEXT", "start": 90, "end": 91, "speaker": {}})
    invoke = {"browser_question": lambda: controller.ask("What was decided?"),
              "runtime_recap": runtime.recap,
              "chat_recap": lambda: runtime.chat({"command": "recap", "user": "operator"})}[api]
    task = asyncio.create_task(invoke())
    await asyncio.wait_for(entered.wait(), 2)
    await cross_privacy_while_provider_ignores_cancel(runtime, controller, cancelled, release, monkeypatch)
    with pytest.raises(ValueError, match="discarded"):
        await asyncio.wait_for(task, 2)
    assert returned.is_set()
    assert not controller._questions


async def test_spoken_offrecord_uses_runtime_barrier_and_private_chat_cannot_answer(setup, monkeypatch):
    runtime, controller, client = setup
    monkeypatch.setattr(controller.clock, "now", lambda: 100.0)
    await runtime.bus.publish(segment("PRIVATE_SECRET off the record", 99, 100))
    await asyncio.wait_for(runtime.bus.drain(), 2)
    assert runtime.state.privacy["state"] == "off_record"
    assert runtime.bus.find(MT.PRIVACY)[-1].data["wiped_from"] == 40
    assert "PRIVATE_SECRET" not in json.dumps([e.to_json() for e in runtime.bus.history])
    for command in ("recap", "ask", "approve", "here"):
        response = await client.post("/api/chat", json={"command": command, "user": "operator", "text": "PRIVATE_SECRET"})
        assert response.status_code == 422
    response = await client.post("/api/chat", json={"command": "resume", "user": "operator"})
    assert response.status_code == 200 and runtime.state.privacy["state"] == "on"
    assert not runtime.notifier.posts and not runtime.notifier.dms
    assert not runtime.bus.errors


async def test_restart_clears_local_notifier_identity_and_output_history(setup, monkeypatch):
    runtime, controller, _ = setup
    await runtime.bus.publish(segment("Hi, I'm Alex.", 10, 11))
    await runtime.bus.drain()
    assert runtime.state.people
    await runtime.notifier.post("notes", "OLD_SESSION_SECRET")
    await runtime.notifier.dm("Alex", "OLD_SESSION_SECRET")
    runtime.state.items["secret"] = {"text": "OLD_SESSION_SECRET"}
    runtime.state.screen["cards"] = [{"text": "OLD_SESSION_SECRET"}]
    await runtime.set_privacy("off_record", "operator")
    await runtime.notifier.post("notes", "OLD_SESSION_SECRET")
    old_notifier = runtime.notifier
    await controller.control("restart")
    assert runtime.notifier is old_notifier
    assert not old_notifier.posts and not old_notifier.dms and old_notifier._n == 0
    assert not runtime.state.people and not runtime.items() and not runtime.transcript()
    assert runtime.state.screen["captions"] == [] and runtime.state.screen["cards"] == []
    assert runtime.state.privacy["state"] == "on"
    assert runtime.state.egress["total_bytes"] == 0
    assert not runtime.agent("fusion").frames and not runtime.agent("policy")._utterances
    assert not runtime.agent("actuator").pending and not controller._private_spans
    assert not runtime.bus.history
    await runtime.bus.publish(segment("New session public discussion", 1, 2))
    await runtime.bus.drain()
    assert runtime.transcript()[0]["text"] == "New session public discussion"


async def test_restart_installs_stream_barrier_before_awaiting_sensor_teardown(setup):
    runtime, controller, _ = setup
    hub = controller._hub
    queue = hub.connect()
    entered, release = asyncio.Event(), asyncio.Event()

    async def stop():
        entered.set()
        await release.wait()

    runtime._sensors = SimpleNamespace(stop=stop)
    await runtime.bus.publish(Event(MT.LOG, {"text": "OLD_SESSION_SECRET"}, source="test"))
    await runtime.bus.drain()
    captured = []
    while not queue.empty():
        captured.append(queue.get_nowait())
    assert any("OLD_SESSION_SECRET" in json.dumps(payload) for payload in captured)
    reset = asyncio.create_task(controller.control("restart"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert hub._resetting
        assert all(hub.ready(payload) is None for payload in captured)
        assert "OLD_SESSION_SECRET" not in json.dumps(hub.browser_snapshot())
    finally:
        release.set()
        await asyncio.wait_for(reset, 2)
        hub.disconnect(queue)
    assert not hub._resetting and not runtime.bus.history


def test_demo_cli_uses_empty_local_documents_and_agenda(tmp_path, monkeypatch):
    import uvicorn

    made = []
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(demo, "load_registry", lambda **kwargs: load_registry(profile="fake"))
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: made.append(app))
    demo.main(["--media-dir", str(tmp_path)])
    runtime = made[0].state.replay.runtime
    cfg = runtime.config["meeting"]
    assert cfg["docs_dir"] == str(tmp_path / "documents")
    assert list((tmp_path / "documents").iterdir()) == []
    assert cfg["agenda"] == []
    assert "examples/docs" not in cfg["docs_dir"]
    assert runtime.docs.chunks == []


def test_demo_cli_refuses_nonempty_documents_without_deleting_them(tmp_path, monkeypatch):
    documents = tmp_path / "documents"
    documents.mkdir()
    existing = documents / "pricing.md"
    existing.write_text("Not evidence for the AMI recording.")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(demo, "load_registry", lambda **kwargs: load_registry(profile="fake"))
    with pytest.raises(SystemExit):
        demo.main(["--media-dir", str(tmp_path)])
    assert existing.read_text() == "Not evidence for the AMI recording."


def test_demo_rejects_trace_writing_and_nonlocal_outputs(tmp_path):
    with pytest.raises(ValueError, match="trace"):
        demo.DemoRuntime(trace_path=str(tmp_path / "trace.jsonl"))
    with pytest.raises(ValueError, match="FakeNotifier"):
        demo.DemoRuntime(notifier=object())
    with pytest.raises(ValueError, match="FakeNotifier"):
        demo.DemoRuntime(notifier=FakeNotifier(echo=True))
    assert not (tmp_path / "trace.jsonl").exists()


@pytest.mark.parametrize("confidence,status", [(0.9, "open"), (0.5, "pending_approval")])
async def test_task_preview_is_sanitized_before_retention_and_subscribers(setup, confidence, status):
    """Controlled policy input, not model/ASR evidence; shared actuator and screen are real."""
    runtime, _, client = setup
    observed = []

    async def observe(event):
        observed.append(event.to_json())

    runtime.bus.subscribe("*", observe, name="preview-regression")
    await runtime.bus.publish(Event(MT.PROPOSAL, {
        "mode": "act", "action": "create_task", "confidence": confidence,
        "args": {"text": "CONTROLLED TEST: send notes", "owner": "Test participant", "due": "2026-10-30"},
    }, source="controlled-test", ts=10))
    await runtime.bus.drain()
    snapshot = (await client.get("/api/snapshot")).json()["snapshot"]
    item = snapshot["items"][0]
    assert item["status"] == status
    assert item["owner"] == "Test participant" and item["due"] == "2026-10-30"
    assert item["message"] is None and item["preview_message"]["id"].startswith("m")
    assert item["preview_message"]["external_delivery"] is False
    assert len(runtime.notifier.posts) == 1
    assert len(runtime.notifier.dms) == (status == "pending_approval")
    expected_bytes = sum(len(r["text"].encode("utf-8")) for r in runtime.notifier.posts + runtime.notifier.dms)
    assert snapshot["previews"] == {
        "delivery": "local_preview", "external_delivery": False, "posts": 1,
        "dms": int(status == "pending_approval"), "text_bytes": expected_bytes,
    }
    for meter in [runtime.state.egress, snapshot["egress"], *[e.data for e in runtime.bus.find(MT.EGRESS)]]:
        assert all(meter[key] == 0 for key in ("bytes", "total_bytes", "audio_bytes_out", "video_bytes_out"))
    for events in (observed, [e.to_json() for e in runtime.bus.history]):
        for event in events:
            data = event["data"]
            if event["topic"] == MT.ITEM or (event["topic"] == MT.COMMAND and data.get("target") == "notifier"):
                assert data["preview"] and data["delivery"] == "local_preview" and not data["external_delivery"]
            if event["topic"] == MT.SCREEN:
                for card in data["cards"]:
                    assert card["preview"] and card["delivery"] == "local_preview" and not card["external_delivery"]
                    assert "Discord" not in card.get("undo", "") + card.get("text", "")
    assert snapshot["screen"]["cards"]
    assert not runtime.bus.errors


@pytest.mark.parametrize("kind,title", [("recap", "Recap preview"), ("docs", "Docs preview")])
@pytest.mark.parametrize("fail", [False, True])
async def test_notifier_commands_never_claim_external_delivery_even_on_failure(setup, kind, title, fail):
    runtime, _, _ = setup
    runtime.notifier.fail = RuntimeError("controlled local failure") if fail else None
    cause = Event(MT.LOG, {"text": "controlled preview regression"}, source="controlled-test", ts=10)
    await runtime.bus.publish(cause)
    await runtime.agent("actuator")._post("notes", "CONTROLLED TEST: preview content", cause, None, kind=kind)
    await runtime.bus.drain()
    assert len(runtime.notifier.posts) == (not fail)
    assert runtime.snapshot()["previews"]["posts"] == (not fail)
    assert runtime.snapshot()["egress"]["total_bytes"] == 0
    assert runtime.state.screen["cards"][0]["card"] == "preview"
    assert runtime.state.screen["cards"][0]["title"] == title
    assert runtime.state.screen["cards"][0]["external_delivery"] is False
    for event in runtime.bus.find(MT.SCREEN):
        assert all(card["card"] != "posted" for card in event.data["cards"])
    assert not runtime.bus.errors


@pytest.mark.parametrize("action", ["off_record", "paused", "restart"])
async def test_privacy_and_reset_clear_all_local_preview_counts_and_data(setup, action):
    runtime, controller, _ = setup
    await runtime.bus.publish(Event(MT.PROPOSAL, {
        "mode": "act", "action": "create_task", "confidence": 0.5,
        "args": {"text": "PREVIEW_SECRET"},
    }, source="controlled-test", ts=10))
    await runtime.bus.drain()
    assert runtime.snapshot()["previews"]["text_bytes"] > 0
    if action == "restart":
        await controller.control("restart")
    else:
        await runtime.set_privacy(action, "operator")
    result = runtime.snapshot()
    assert result["previews"]["posts"] == result["previews"]["dms"] == result["previews"]["text_bytes"] == 0
    assert not result["items"] and not result["screen"]["cards"]
    assert result["egress"]["total_bytes"] == 0
    assert "PREVIEW_SECRET" not in json.dumps(result)
    assert "PREVIEW_SECRET" not in json.dumps([event.to_json() for event in runtime.bus.history])
    assert not runtime.notifier.posts and not runtime.notifier.dms


async def test_preview_adapter_does_not_change_shared_runtime_delivery_contract():
    from unspoken.meeting.fakes import MeteredNotifier
    from unspoken.meeting.runtime import MeetingRuntime

    # No real delivery: a metered fake isolates the shared runtime's existing contract.
    notifier = MeteredNotifier(FakeNotifier(echo=False))
    runtime = MeetingRuntime(profile="fake", registry=load_registry(profile="fake"),
                             fake_io=True, notifier=notifier, echo=False)
    notifier.on_egress = runtime._on_egress
    try:
        await runtime.start(ticking=False)
        cause = Event(MT.LOG, {"text": "controlled shared-runtime regression"}, source="controlled-test")
        await runtime.bus.publish(cause)
        await runtime.agent("actuator")._post("notes", "Unchanged contract", cause, None, kind="recap")
        await runtime.bus.drain()
        assert runtime.snapshot()["egress"]["total_bytes"] == len("Unchanged contract")
        assert runtime.state.screen["cards"][0]["card"] == "posted"
        assert "previews" not in runtime.snapshot()
        assert not runtime.bus.errors
    finally:
        await runtime.close()
