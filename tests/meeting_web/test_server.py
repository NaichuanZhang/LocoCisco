"""Tests for unspoken.meeting.web.server with the dev fake. No model servers, no hardware, no network."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from unspoken.core.events import Event
from unspoken.meeting.contracts import MT
from unspoken.meeting.web import server as web
from unspoken.meeting.web.dev_fake import FakeMeetingRuntime


@pytest.fixture
def runtime() -> FakeMeetingRuntime:
    return FakeMeetingRuntime(fast=True)


@pytest.fixture
async def client(runtime: FakeMeetingRuntime):
    app = web.create_app(runtime)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        c.app = app  # type: ignore[attr-defined]
        yield c
    await runtime.stop_scenario()


# --------------------------------------------------------------------------- page + snapshot
async def test_index_and_static(client: httpx.AsyncClient):
    r = await client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert client.app.title == "LocoCisco room screen"
    assert "<title>LocoCisco · meeting intelligence</title>" in r.text
    assert r.headers["cache-control"] == "no-store"
    assert "static/screen.js" in r.text and "static/screen.css" in r.text and "static/xray.js" in r.text
    for name in ("screen.js", "screen.css", "xray.js", "index.html"):
        assert (await client.get(f"/static/{name}")).status_code == 200, name
    assert (await client.get("/static/nope.js")).status_code == 404
    # no docs endpoints are exposed on the room screen
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert (await client.get(path)).status_code == 404, path


async def test_snapshot_shape(client: httpx.AsyncClient):
    r = await client.get("/api/snapshot")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-lococisco-stream-epoch"] == "0"
    body = r.json()
    assert set(body) == {"snapshot", "settings"}
    snap = body["snapshot"]
    assert set(snap) == {"status", "privacy", "participants", "screen", "items", "devices", "egress", "models"}
    assert set(snap["screen"]) >= {"privacy", "captions", "agenda", "cards", "banner"}
    assert snap["privacy"]["state"] == "on" and snap["screen"]["privacy"] == "on"
    assert snap["egress"]["audio_bytes_out"] == 0 and snap["egress"]["video_bytes_out"] == 0
    assert body["settings"]["act_threshold"] == 0.8 and body["settings"]["caption_lines"] == 6


async def test_models_marks_local_targets(client: httpx.AsyncClient):
    r = await client.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    rows = body["models"]
    assert {m["role"] for m in rows} >= {"decider", "llm", "asr", "vad", "diarization", "faces", "asd", "embed", "pii"}
    assert all({"role", "provider", "label", "target", "ok", "local"} <= set(m) for m in rows)
    assert all(m["local"] for m in rows) and body["all_local"] is True


def test_target_is_local():
    for t in ("in-process", "", "http://127.0.0.1:8765", "http://localhost:8000/v1", "http://[::1]:8000",
              "http://192.168.1.20:8000/v1", "http://10.0.0.5:8000", "http://172.16.0.9:11434", "http://gb10.local:8000",
              "/models/whisper-large-v3-turbo", "disabled"):
        assert web.target_is_local(t), t
    for t in ("https://bedrock-runtime.us-east-1.amazonaws.com", "https://api.openai.com/v1", "http://8.8.8.8:80",
              "https://example.com"):
        assert not web.target_is_local(t), t


async def test_scenarios_run_stop_reset(client: httpx.AsyncClient, runtime: FakeMeetingRuntime):
    r = await client.get("/api/scenarios")
    assert r.status_code == 200
    ids = [s["id"] for s in r.json()]
    assert "demo_story" in ids and all({"id", "title", "blurb", "tags", "steps"} <= set(s) for s in r.json())
    assert (await client.post("/api/scenario/does_not_exist")).status_code == 404
    r = await client.post("/api/scenario/beat_task")
    assert r.status_code == 200 and r.json() == {"ok": True, "scenario_id": "beat_task"}
    await runtime._task
    await runtime.bus.drain()
    states = [e.data["state"] for e in runtime.bus.find(MT.SCENARIO)]
    assert states[0] == "started" and states[-1] == "finished"
    assert runtime.bus.find(MT.SCENARIO, state="finished")[0].data["failed"] == 0
    assert any(i["kind"] == "task" and i["owner"] == "Sam" for i in runtime.items())
    assert (await client.post("/api/stop")).status_code == 200
    r = await client.post("/api/reset")
    assert r.status_code == 200
    snap = r.json()["snapshot"]
    assert snap["items"] == [] and snap["screen"]["captions"] == [] and snap["egress"]["total_bytes"] == 0


async def test_stop_cancels_a_running_scenario(runtime: FakeMeetingRuntime):
    slow = FakeMeetingRuntime()  # real sleeps so the story is still running when we stop it
    app = web.create_app(slow)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.post("/api/scenario/demo_story")).status_code == 200
        await asyncio.sleep(0.05)
        assert (await c.post("/api/stop")).status_code == 200
    await slow.bus.drain()
    assert slow.bus.find(MT.SCENARIO, state="stopped")


# --------------------------------------------------------------------------- chat (rehearsal)
async def test_chat_commands(client: httpx.AsyncClient, runtime: FakeMeetingRuntime):
    await runtime.run_scenario("beat_task")
    await runtime._task
    # off the record from Discord, then back
    r = await client.post("/api/chat", json={"command": "offrecord", "user": "Priya"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert runtime.privacy["state"] == "off_record" and runtime.snapshot()["screen"]["captions"] == []
    r = await client.post("/api/chat", json={"command": "resume", "user": "Priya"})
    assert r.status_code == 200 and runtime.privacy["state"] == "on"
    # undo the last task
    task = [i for i in runtime.items("task") if i["owner"] == "Sam"][0]
    r = await client.post("/api/chat", json={"command": "undo", "user": "Sam", "args": {"item_id": task["item_id"]}})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert runtime._items[task["item_id"]]["status"] == "undone"
    assert not any(c.get("item_id") == task["item_id"] for c in runtime.snapshot()["screen"]["cards"])
    # a late joiner checks in and gets a name
    await runtime._do_join({"who": "sam", "seat": "s2", "voice": "voice-B", "checkin": None})
    assert any(p["guest"] for p in runtime.snapshot()["participants"])
    r = await client.post("/api/chat", json={"command": "here", "user": "Sam", "args": {"name": "Sam"}})
    assert r.status_code == 200 and "Sam" in r.json()["text"]
    assert not any(p["guest"] for p in runtime.snapshot()["participants"])
    # recap on demand leaves as text and is metered
    before = runtime.egress["total_bytes"]
    r = await client.post("/api/chat", json={"command": "recap", "user": "Priya"})
    assert r.status_code == 200 and "Recap" in r.json()["text"]
    assert runtime.egress["total_bytes"] > before and runtime.egress["audio_bytes_out"] == 0
    # validation
    for bad in ({"command": "fly"}, {"user": "x"}, {"command": "here", "extra": 1}, {"command": "ask", "text": "x" * 2001},
                {"command": "here", "user": ""}):
        assert (await client.post("/api/chat", json=bad)).status_code == 422, bad
    assert (await client.post("/api/chat", json={"command": "approve", "user": "Priya"})).json()["ok"] is False


# --------------------------------------------------------------------------- settings
async def test_settings_validation(client: httpx.AsyncClient, runtime: FakeMeetingRuntime):
    r = await client.post("/api/settings", json={"act_threshold": 0.9, "speed": 4.0, "caption_lines": 8})
    assert r.status_code == 200
    body = r.json()
    assert body["act_threshold"] == 0.9 and body["speed"] == 4.0 and body["caption_lines"] == 8
    assert runtime.settings.act_threshold == 0.9 and runtime.settings.mode == "sim"
    for bad in ({"act_threshold": 1.5}, {"ask_threshold": -0.1}, {"speed": 0}, {"speed": 500}, {"caption_lines": 0},
                {"ask_threshold": 0.95}, {"act_threshold": 0.3}, {"mode": "live"}, {"nope": 1}, {"escalate": "yes"},
                {"wipe_seconds": -1}):
        assert (await client.post("/api/settings", json=bad)).status_code == 422, bad
    assert runtime.settings.act_threshold == 0.9  # nothing above was applied
    r = await client.post("/api/settings", json={"escalate": False, "wipe_seconds": 120})
    assert r.status_code == 200 and runtime.settings.escalate is False and runtime.settings.wipe_seconds == 120
    assert (await client.post("/api/settings", json={})).status_code == 200


# --------------------------------------------------------------------------- WebSocket stream
class _WsSession:
    """Drives the ASGI app's websocket scope directly so the stream can be read frame by frame in one loop."""

    def __init__(self, app) -> None:
        self.app = app
        self.out: asyncio.Queue[dict] = asyncio.Queue()
        self.inbox: asyncio.Queue[dict] = asyncio.Queue()
        self.inbox.put_nowait({"type": "websocket.connect"})

    async def _receive(self) -> dict:
        return await self.inbox.get()

    async def _send(self, message: dict) -> None:
        await self.out.put(message)

    def start(self) -> asyncio.Task:
        scope = {"type": "websocket", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
                 "scheme": "ws", "path": "/ws", "raw_path": b"/ws", "query_string": b"", "root_path": "",
                 "headers": [(b"host", b"test")], "client": ("127.0.0.1", 40000), "server": ("127.0.0.1", 18751),
                 "subprotocols": []}
        self.task = asyncio.create_task(self.app(scope, self._receive, self._send))
        return self.task

    async def frame(self, timeout: float = 3.0) -> dict:
        while True:
            msg = await asyncio.wait_for(self.out.get(), timeout)
            if msg["type"] == "websocket.accept":
                self.accepted = True
                continue
            if msg["type"] == "websocket.send":
                return json.loads(msg["text"])
            if msg["type"] == "websocket.close":
                return {"closed": msg.get("code")}

    def disconnect(self) -> None:
        self.inbox.put_nowait({"type": "websocket.disconnect", "code": 1000})


async def test_ws_streams_bus_events_and_snapshots(runtime: FakeMeetingRuntime):
    app = web.create_app(runtime)
    hub: web.StreamHub = app.state.hub
    sess = _WsSession(app)
    sess.start()
    first = await sess.frame()
    assert sess.accepted and first["topic"] == web.SNAPSHOT_TOPIC
    assert first["data"]["snapshot"]["privacy"]["state"] == "on" and first["data"]["settings"]["act_threshold"] == 0.8
    assert hub.clients == 1
    # a streamed topic arrives unchanged
    ev = Event(MT.LOG, {"agent": "test", "level": "info", "text": "hello"}, source="test", ts=1.5)
    await runtime.bus.publish(ev)
    assert await sess.frame() == ev.to_json()
    # sensor topics never reach the page
    await runtime.bus.publish(Event(MT.SEGMENT, {"segment_id": "s1", "text": "raw words"}, source="sensors.audio", ts=2.0))
    await runtime.bus.publish(Event(MT.SPEAKERS, {"t": 2.0, "faces": []}, source="sensors.video", ts=2.0))
    # a screen.state event is followed by a synthetic snapshot
    await runtime.bus.publish(Event(MT.SCREEN, runtime._screen(), source="agent.screen", ts=2.5))
    got = await sess.frame()
    assert got["topic"] == MT.SCREEN, got["topic"]
    snap = await sess.frame()
    assert snap["topic"] == web.SNAPSHOT_TOPIC and snap["source"] == "web" and snap["ts"] == 2.5
    sess.disconnect()
    await asyncio.wait_for(sess.task, 3)
    assert hub.clients == 0


async def test_ws_heartbeat_and_settings_broadcast(monkeypatch, runtime: FakeMeetingRuntime):
    monkeypatch.setattr(web, "HEARTBEAT_SECONDS", 0.05)
    app = web.create_app(runtime)
    sess = _WsSession(app)
    sess.start()
    await sess.frame()  # snapshot
    beat = await sess.frame()
    assert beat["topic"] == web.PING_TOPIC
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.post("/api/settings", json={"act_threshold": 0.85})).status_code == 200
    frames = [await sess.frame() for _ in range(3)]
    settings = [f for f in frames if f["topic"] == web.SETTINGS_TOPIC]
    assert settings and settings[0]["data"]["settings"]["act_threshold"] == 0.85
    sess.disconnect()
    await asyncio.wait_for(sess.task, 3)


async def test_ws_sees_the_whole_story(runtime: FakeMeetingRuntime):
    """A public scenario still streams the complete pipeline; private queues are tested separately."""
    app = web.create_app(runtime)
    sess = _WsSession(app)
    sess.start()
    await sess.frame()
    await runtime.run_scenario("beat_task")
    await runtime._task
    await runtime.bus.drain()
    await asyncio.sleep(0.3)   # trailing throttled snapshot
    topics = []
    while not sess.out.empty():
        msg = sess.out.get_nowait()
        if msg["type"] == "websocket.send":
            topics.append(json.loads(msg["text"])["topic"])
    assert {MT.UTTERANCE, MT.PROPOSAL, MT.VERDICT, MT.PRIVACY, MT.SCREEN, MT.LOG, MT.SCENARIO, MT.ENV, web.SNAPSHOT_TOPIC} <= set(topics)
    assert MT.SEGMENT not in topics and MT.SPEAKERS not in topics
    sess.disconnect()
    await asyncio.wait_for(sess.task, 3)


async def test_snapshot_throttle_and_bounded_queue(runtime: FakeMeetingRuntime):
    hub = web.StreamHub(runtime, queue_size=3, min_interval=0.2)
    q = hub.connect()
    for i in range(5):
        hub.broadcast({"n": i})
    assert q.qsize() == 3 and q.get_nowait() == {"n": 2} and hub.dropped == 2
    hub2 = web.StreamHub(runtime, queue_size=100, min_interval=0.2)
    q2 = hub2.connect()
    for i in range(20):
        await runtime.bus.publish(Event(MT.EGRESS, {"dest": "discord", "kind": "text", "bytes": 1, "total_bytes": i},
                                        source="bridge", ts=float(i)))
    await runtime.bus.drain()
    await asyncio.sleep(0.3)
    items = []
    while not q2.empty():
        items.append(q2.get_nowait())
    assert len([x for x in items if x["topic"] == MT.EGRESS]) == 20
    snaps = [x for x in items if x["topic"] == web.SNAPSHOT_TOPIC]
    assert len(snaps) == 2 and snaps[-1]["ts"] == 19.0, "one immediate snapshot plus one trailing snapshot for a burst"


def _queued(q: asyncio.Queue) -> list[dict]:
    frames = []
    while not q.empty():
        frames.append(q.get_nowait())
    return frames


async def test_private_stream_filters_content_not_public_metadata(runtime: FakeMeetingRuntime):
    hub = web.StreamHub(runtime, min_interval=0)
    q = hub.connect()
    public = Event(MT.UTTERANCE, {"text": "old caption"}, source="test")
    await hub._on_event(public)
    held_by_pump = q.get_nowait()
    # Runtime state changes before the web callback for its privacy event gets CPU time.
    runtime.privacy = {"state": "off_record", "reason": "private reason", "wiped_count": 1}
    await hub._on_event(Event(MT.PRIVACY, dict(runtime.privacy), source="test", ts=60))
    assert hub.ready(held_by_pump) is None
    assert not any(f["topic"] == MT.UTTERANCE for f in _queued(q))
    for topic in (MT.UTTERANCE, MT.PROPOSAL, MT.VERDICT, MT.COMMAND, MT.ITEM, MT.LOG, "meeting.future_content"):
        await hub._on_event(Event(topic, {"text": "PRIVATE", "args": {"text": "PRIVATE"}}, source="test", ts=61))
    await hub._on_event(Event(MT.SCREEN, {
        "privacy": "on", "captions": [{"text": "PRIVATE", "t": 61}],
        "cards": [{"text": "PRIVATE"}], "banner": "PRIVATE", "agenda": {"minutes_left": 9},
    }, source="test", ts=61))
    for topic, data in (
        (MT.STATUS, {"state": "started", "title": "Meeting"}),
        (MT.EGRESS, {"total_bytes": 100}),
        (MT.ENV, {"device_id": "lamp", "state": {"scene": "off_record"}}),
        (MT.CLOCK, {"t": 61}),
        (MT.SCENE, {"mode": "talk", "text": "PRIVATE"}),
        (MT.SCENARIO, {"state": "step", "step_index": 1, "text": "PRIVATE"}),
    ):
        await hub._on_event(Event(topic, data, source="test", ts=61))
    frames = _queued(q)
    assert "PRIVATE" not in json.dumps(frames)
    assert {MT.STATUS, MT.EGRESS, MT.ENV, MT.CLOCK, MT.SCENE, MT.SCENARIO, MT.SCREEN} <= {
        f["topic"] for f in frames
    }
    screens = [f["data"] for f in frames if f["topic"] == MT.SCREEN]
    assert screens[-1]["privacy"] == "off_record"
    assert screens[-1]["captions"] == screens[-1]["cards"] == []
    assert screens[-1]["agenda"]["minutes_left"] == 9


async def test_privacy_event_order_and_delayed_content(runtime: FakeMeetingRuntime):
    hub = web.StreamHub(runtime, min_interval=0)
    q = hub.connect()
    # The runtime already resumed, but its off/on callbacks are still queued.
    off = Event(MT.PRIVACY, {"state": "off_record"}, source="test", ts=60)
    private = Event(MT.PROPOSAL, {"request": {"state": "PRIVATE"}}, source="test", ts=61)
    resume = Event(MT.PRIVACY, {"state": "on"}, source="test", ts=62)
    await hub._on_event(off)
    assert hub.snapshot_event()["data"]["snapshot"]["privacy"]["state"] == "off_record"
    await hub._on_event(private)
    await hub._on_event(resume)
    await hub._on_event(private)  # callback delayed past resume must not repopulate X-ray
    await hub._on_event(off)      # older privacy callback must not re-close a resumed session
    assert hub.snapshot_event()["data"]["snapshot"]["privacy"]["state"] == "on"
    public = Event(MT.UTTERANCE, {"text": "new public"}, source="test", ts=63)
    await hub._on_event(public)
    frames = _queued(q)
    assert "PRIVATE" not in json.dumps(frames)
    assert any(f["id"] == public.id for f in frames)
    # Conversely, an on callback cannot reopen the UI while runtime privacy is already off.
    runtime.privacy = {"state": "off_record"}
    await hub._on_event(Event(MT.PRIVACY, {"state": "on"}, source="test", ts=64))
    assert hub.snapshot_event()["data"]["snapshot"]["privacy"]["state"] == "off_record"


async def test_snapshots_do_not_restore_wiped_browser_cards(client, runtime):
    hub = client.app.state.hub
    runtime.captions = [{"text": "PRIVATE", "t": 59}]
    runtime.cards = [{"card": "task", "item_id": "task-old", "text": "PRIVATE"}]
    runtime._items["task-old"] = {"item_id": "task-old", "text": "PRIVATE"}
    runtime.privacy = {"state": "off_record", "reason": "PRIVATE"}
    await hub._on_event(Event(MT.PRIVACY, dict(runtime.privacy), source="test", ts=60))
    private = (await client.get("/api/snapshot")).json()
    assert "PRIVATE" not in json.dumps(private)
    assert runtime.captions[0]["text"] == "PRIVATE", "browser redaction must not pretend to fix backend retention"
    runtime.privacy = {"state": "on"}
    await hub._on_event(Event(MT.PRIVACY, dict(runtime.privacy), source="test", ts=62))
    resumed = (await client.get("/api/snapshot")).json()
    assert "PRIVATE" not in json.dumps(resumed)
    runtime.captions.append({"text": "new public", "t": 63})
    runtime.cards.append({"card": "task", "item_id": "new", "text": "new public"})
    visible = (await client.get("/api/snapshot")).json()["snapshot"]["screen"]
    assert visible["captions"] == [{"text": "new public", "t": 63}]
    assert visible["cards"] == [{"card": "task", "item_id": "new", "text": "new public"}]


async def test_reset_purges_queues_before_awaiting_teardown_and_rejects_old_callbacks(client, runtime, monkeypatch):
    hub = client.app.state.hub
    q = hub.connect()
    old = Event(MT.COMMAND, {"target": "notifier", "args": {"text": "PRIVATE"}}, source="test")
    await hub._on_event(old)
    held_by_pump = q.get_nowait()
    hub.broadcast(old.to_json())
    hub.broadcast(hub.snapshot_event())
    started, release = asyncio.Event(), asyncio.Event()
    original = runtime.reset

    async def reset():
        started.set()
        await release.wait()
        await original()

    monkeypatch.setattr(runtime, "reset", reset)
    request = asyncio.create_task(client.post("/api/reset"))
    await started.wait()
    assert hub.ready(held_by_pump) is None
    assert "PRIVATE" not in json.dumps(_queued(q))
    during_reset = Event(MT.PROPOSAL, {"reply": "PRIVATE"}, source="test")
    await hub._on_event(during_reset)
    release.set()
    assert (await request).status_code == 200
    for event in (old, during_reset):
        await hub._on_event(event)
    fresh = Event(MT.PROPOSAL, {"reply": "public"}, source="test")
    await hub._on_event(fresh)
    frames = _queued(q)
    assert "PRIVATE" not in json.dumps(frames)
    assert any(f["topic"] == web.RESET_TOPIC for f in frames)
    assert any(f["id"] == fresh.id for f in frames)


async def test_tiny_queue_still_marks_epoch_when_privacy_frame_is_dropped(runtime):
    hub = web.StreamHub(runtime, queue_size=1, min_interval=0)
    q = hub.connect()
    await hub._on_event(Event(MT.PRIVACY, {"state": "off_record"}, source="test", ts=1))
    await hub._on_event(Event(MT.PRIVACY, {"state": "on"}, source="test", ts=2))
    await hub._on_event(Event(MT.EGRESS, {"total_bytes": 2}, source="test", ts=3))
    frame = q.get_nowait()
    assert frame["stream_epoch"] > 0, "even a surviving metadata/snapshot frame must clear browser caches"
    assert frame["stream_privacy"] == "on"
    assert hub.ready(frame) is not None


# --------------------------------------------------------------------------- loopback only
def test_serve_refuses_non_loopback(runtime: FakeMeetingRuntime):
    for host in ("0.0.0.0", "192.168.1.20", "10.0.0.1", "::", "example.com", "", "0.0.0.0:8780"):
        with pytest.raises(ValueError):
            web.serve(runtime, host=host, port=18799)
        with pytest.raises(ValueError):
            web.require_loopback(host)
    for host in ("127.0.0.1", "127.0.0.2", "localhost", "::1", "[::1]"):
        assert web.is_loopback(host), host


async def test_serve_async_refuses_non_loopback(runtime: FakeMeetingRuntime):
    with pytest.raises(ValueError):
        await web.serve_async(runtime, host="0.0.0.0", port=18799)
