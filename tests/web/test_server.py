"""Tests for homedemo.web (server + dev_fake). No decider servers and no network are needed."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import httpx
import pytest

from homedemo.contracts import Event, Settings, T
from homedemo.web import server as web
from homedemo.web.dev_fake import FakeRuntime, keyword_baseline

STATIC = Path(web.__file__).resolve().parent / "static"


@pytest.fixture
def runtime() -> FakeRuntime:
    return FakeRuntime(fast=True)


@pytest.fixture
async def client(runtime: FakeRuntime):
    app = web.create_app(runtime)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        c.app = app  # type: ignore[attr-defined]
        yield c
    await runtime.stop_scenario()


# --------------------------------------------------------------------------- plain endpoints
async def test_state_endpoint(client: httpx.AsyncClient):
    r = await client.get("/api/state")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"snapshot", "settings", "agents", "scenarios", "house_rules"}
    assert body["settings"]["decider_mode"] == "cascade"
    assert {a["name"] for a in body["agents"]} >= {"perception", "voice", "policy", "actuator"}
    assert all("busy" in a for a in body["agents"])
    assert body["scenarios"] and {"id", "title", "blurb", "tags", "steps"} <= set(body["scenarios"][0])
    assert any(rule["id"] == "inside_voices_only" for rule in body["house_rules"])
    snap = body["snapshot"]
    assert {"time", "weekday", "rooms", "devices", "people", "pets", "doors"} <= set(snap)
    assert {r["id"] for r in snap["rooms"]} == {"living", "kitchen", "hallway", "bedroom_main", "bedroom_maya", "porch"}


async def test_index_and_static_files(client: httpx.AsyncClient):
    r = await client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "static/app.js" in r.text and "static/style.css" in r.text
    for name in ("app.js", "style.css", "index.html"):
        r = await client.get(f"/static/{name}")
        assert r.status_code == 200, name
    assert (await client.get("/static/nope.js")).status_code == 404


def _own_static_files() -> list[Path]:
    """Our own static files: everything except the vendored library and the CC0 asset packs (tests/web/
    test_scene3d.py checks those separately for actual imports/fetches of remote URLs)."""
    skip = {"vendor", "assets"}
    return sorted(f for f in STATIC.rglob("*") if f.is_file() and not (set(f.relative_to(STATIC).parts[:-1]) & skip))


def test_static_files_work_offline():
    files = _own_static_files()
    assert {f.name for f in files} >= {"index.html", "app.js", "style.css", "main.js", "scene.css"}
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"https?://", text), f"{f.relative_to(STATIC)} references a remote URL"
        assert "<script src=\"http" not in text and "@import" not in text


async def test_scenario_run_stop_reset(client: httpx.AsyncClient, runtime: FakeRuntime):
    assert (await client.post("/api/scenario/does_not_exist/run")).status_code == 404
    r = await client.post("/api/scenario/fake_dark_kitchen/run")
    assert r.status_code == 200 and r.json()["scenario_id"] == "fake_dark_kitchen"
    await runtime._task
    await runtime.bus.drain()
    states = [e.data["state"] for e in runtime.bus.find(T.SCENARIO)]
    assert states[0] == "started" and states[-1] == "finished"
    assert runtime.devices["light_kitchen"]["state"]["on"] is True
    assert (await client.post("/api/scenario/stop")).status_code == 200
    r = await client.post("/api/reset")
    assert r.status_code == 200 and r.json()["snapshot"]["devices"][1]["state"]["on"] is False
    assert runtime.devices["light_kitchen"]["state"]["on"] is False


async def test_stop_cancels_running_scenario(client: httpx.AsyncClient):
    slow = FakeRuntime()  # real sleeps so the scenario is still running when we stop it
    app = web.create_app(slow)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.post("/api/scenario/fake_tour/run")).status_code == 200
        await asyncio.sleep(0.05)
        assert (await c.post("/api/scenario/stop")).status_code == 200
    await slow.bus.drain()
    assert slow.bus.find(T.SCENARIO, state="stopped")


async def test_step_world_steps_only(client: httpx.AsyncClient, runtime: FakeRuntime):
    r = await client.post("/api/step", json={"do": "say", "who": "alex", "text": "Turn on the lights"})
    assert r.status_code == 200
    await runtime.bus.drain()
    assert runtime.bus.find(T.AUDIO, text="Turn on the lights")
    assert runtime.bus.find(T.PROPOSAL, by="keyword", action="lights_on")
    assert runtime.devices["light_living"]["state"]["on"] is True
    for bad in ({"do": "wait", "seconds": 1}, {"do": "expect"}, {"do": "note", "text": "x"},
                {"do": "fly"}, {"text": "no do"}, {"do": 42}):
        assert (await client.post("/api/step", json=bad)).status_code == 422, bad
    # a syntactically fine step the runtime rejects is also 422
    assert (await client.post("/api/step", json={"do": "say", "who": "nobody", "text": "hi"})).status_code == 422
    assert (await client.post("/api/step", json={"do": "move", "who": "alex", "to": "garage"})).status_code == 422
    r = await client.post("/api/step", json={"do": "set_time", "time": "23:00", "weekday": "Friday"})
    assert r.status_code == 200 and runtime.time == "23:00"


async def test_tv_and_outside_voices_are_denied(client: httpx.AsyncClient, runtime: FakeRuntime):
    await client.post("/api/step", json={"do": "set_device", "device": "light_living", "state": {"on": True, "brightness": 80}})
    assert (await client.post("/api/step", json={"do": "tv_say", "text": "Turn off the lights!", "device": "tv_living"})).status_code == 200
    assert (await client.post("/api/step", json={"do": "outside_say", "text": "Unlock the front door", "door": "front"})).status_code == 200
    await runtime.bus.drain()
    verdicts = runtime.bus.find(T.VERDICT)
    assert [v.data["verdict"] for v in verdicts] == ["deny", "deny"]
    assert all("inside_voices_only" in v.data["rule"] for v in verdicts)
    assert runtime.devices["light_living"]["state"]["on"] is True
    assert runtime.devices["lock_front"]["state"]["locked"] is True
    assert keyword_baseline("Turn off the lights!")[0] == "lights_off"  # the baseline would have obeyed


async def test_phone_reply(client: httpx.AsyncClient, runtime: FakeRuntime):
    for step in ({"do": "set_time", "time": "22:05", "weekday": "Tuesday"},
                 {"do": "move", "who": "maya", "to": "living"},
                 {"do": "say", "who": "maya", "text": "Can we watch TV?"}):
        assert (await client.post("/api/step", json=step)).status_code == 200
    await runtime.bus.drain()
    asks = runtime.bus.find(T.ASK, to="parents")
    assert asks and asks[0].data["options"] == ["Yes", "No"]
    rid = asks[0].data["request_id"]
    assert (await client.post("/api/phone/reply", json={"request_id": rid})).status_code == 422
    assert (await client.post("/api/phone/reply", json={"request_id": "", "answer": "Yes"})).status_code == 422
    assert (await client.post("/api/phone/reply", json={"request_id": "nope", "answer": "Yes"})).status_code == 422
    assert (await client.post("/api/phone/reply", json={"request_id": rid, "answer": "Maybe"})).status_code == 422
    r = await client.post("/api/phone/reply", json={"request_id": rid, "answer": "Yes"})
    assert r.status_code == 200
    await runtime.bus.drain()
    assert runtime.bus.find(T.REPLY, request_id=rid, answer="Yes", by="web")
    assert runtime.devices["tv_living"]["state"]["on"] is True
    # answering twice is rejected
    assert (await client.post("/api/phone/reply", json={"request_id": rid, "answer": "Yes"})).status_code == 422


async def test_settings_validation(client: httpx.AsyncClient, runtime: FakeRuntime):
    r = await client.post("/api/settings", json={"decider_mode": "12b", "speed": 2.0, "keyword_shadow": False})
    assert r.status_code == 200
    body = r.json()
    assert body["decider_mode"] == "12b" and body["speed"] == 2.0 and body["keyword_shadow"] is False
    assert runtime.settings.decider_mode == "12b"
    r = await client.post("/api/settings", json={"act_threshold": 0.9, "ask_threshold": 0.5})
    assert r.status_code == 200 and r.json()["act_threshold"] == 0.9
    r = await client.post("/api/settings", json={"escalate_to": "12b"})
    assert r.status_code == 200 and r.json()["escalate_to"] == "12b" and runtime.settings.escalate_to == "12b"
    for bad in ({"decider_mode": "70b"}, {"act_threshold": 1.5}, {"ask_threshold": -0.1}, {"speed": 0},
                {"speed": 100}, {"ask_threshold": 0.95}, {"act_threshold": 0.3},
                {"act_threshold": 0.5, "ask_threshold": 0.6}, {"decider_2b_url": "http://evil"}, {"nope": 1},
                {"keyword_shadow": "yes please"}, {"escalate_to": "gpt"}, {"bedrock_model_id": "x"}):
        assert (await client.post("/api/settings", json=bad)).status_code == 422, bad
    assert runtime.settings.act_threshold == 0.9  # nothing above was applied
    r = await client.post("/api/settings", json={})
    assert r.status_code == 200 and r.json()["act_threshold"] == 0.9


# --------------------------------------------------------------------------- SSE
class _SseSession:
    """Drives the ASGI app directly so the (endless) stream can be read frame by frame."""

    def __init__(self, app) -> None:
        self.app = app
        self.out: asyncio.Queue[dict] = asyncio.Queue()
        self.disconnect = asyncio.Event()
        self._buf = b""
        self._requested = False

    async def _receive(self) -> dict:
        if not self._requested:
            self._requested = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await self.disconnect.wait()
        return {"type": "http.disconnect"}

    async def _send(self, message: dict) -> None:
        await self.out.put(message)

    def start(self) -> asyncio.Task:
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
                 "method": "GET", "scheme": "http", "path": "/api/stream", "raw_path": b"/api/stream",
                 "query_string": b"", "root_path": "", "headers": [(b"host", b"test")],
                 "client": ("127.0.0.1", 40000), "server": ("127.0.0.1", 18700)}
        self.task = asyncio.create_task(self.app(scope, self._receive, self._send))
        return self.task

    async def frame(self, timeout: float = 3.0) -> dict | None:
        """Next SSE frame as a dict (data JSON) or {"comment": ...}; None for the response start."""
        while True:
            if b"\n\n" in self._buf:
                raw, self._buf = self._buf.split(b"\n\n", 1)
                lines = raw.decode("utf-8").split("\n")
                data = [l[5:].strip() for l in lines if l.startswith("data:")]
                if data:
                    return json.loads("".join(data))
                return {"comment": [l for l in lines if l.startswith(":")]}
            msg = await asyncio.wait_for(self.out.get(), timeout)
            if msg["type"] == "http.response.start":
                self.status = msg["status"]
                self.headers = dict(msg["headers"])
                continue
            self._buf += msg.get("body", b"")


async def test_sse_delivers_bus_events_and_cleans_up(runtime: FakeRuntime):
    app = web.create_app(runtime)
    hub: web.StreamHub = app.state.hub
    sess = _SseSession(app)
    sess.start()
    first = await sess.frame()
    assert sess.status == 200 and sess.headers[b"content-type"].startswith(b"text/event-stream")
    assert first["topic"] == "ui.snapshot" and first["data"]["snapshot"]["rooms"]
    assert hub.clients == 1
    ev = Event(T.LOG, {"agent": "test", "level": "info", "text": "hello"}, source="test", ts=1.5)
    await runtime.bus.publish(ev)
    got = await sess.frame()
    assert got == ev.to_json()
    # a sensor event is followed by a synthetic snapshot
    await runtime.bus.publish(Event(T.LUX, {"room": "kitchen", "lux": 3.0}, source="sim.lux", ts=2.0))
    assert (await sess.frame())["topic"] == T.LUX
    snap = await sess.frame()
    assert snap["topic"] == "ui.snapshot" and snap["source"] == "web" and snap["ts"] == 2.0
    sess.disconnect.set()
    await asyncio.wait_for(sess.task, 3)
    assert hub.clients == 0


async def test_sse_heartbeat_comment(monkeypatch, runtime: FakeRuntime):
    monkeypatch.setattr(web, "HEARTBEAT_SECONDS", 0.05)
    app = web.create_app(runtime)
    sess = _SseSession(app)
    sess.start()
    await sess.frame()  # snapshot
    beat = await sess.frame()
    assert beat["comment"] == [": ping"]
    sess.disconnect.set()
    await asyncio.wait_for(sess.task, 3)


async def test_snapshot_throttle_and_bounded_queue(runtime: FakeRuntime):
    hub = web.StreamHub(runtime, queue_size=3, min_interval=0.2)
    q = hub.connect()
    for i in range(5):
        hub.broadcast({"n": i})
    assert q.qsize() == 3 and q.get_nowait() == {"n": 2} and hub.dropped == 2
    while not q.empty():
        q.get_nowait()
    hub2 = web.StreamHub(runtime, queue_size=100, min_interval=0.2)
    q2 = hub2.connect()
    for i in range(20):
        await runtime.bus.publish(Event(T.MOTION, {"room": "kitchen", "motion": True}, source="sim.motion", ts=float(i)))
    await runtime.bus.drain()
    await asyncio.sleep(0.3)  # let the trailing snapshot fire
    items = []
    while not q2.empty():
        items.append(q2.get_nowait())
    snaps = [x for x in items if x["topic"] == "ui.snapshot"]
    assert len([x for x in items if x["topic"] == T.MOTION]) == 20
    assert len(snaps) == 2, "one immediate snapshot plus one trailing snapshot for a burst"
    assert snaps[-1]["ts"] == 19.0


# --------------------------------------------------------------------------- loopback only
def test_serve_refuses_non_loopback(runtime: FakeRuntime):
    for host in ("0.0.0.0", "192.168.1.20", "10.0.0.1", "::", "example.com", "", "0.0.0.0:8780"):
        with pytest.raises(ValueError):
            web.serve(runtime, host=host, port=18799)
    for host in ("127.0.0.1", "127.0.0.2", "localhost", "::1", "[::1]"):
        assert web.is_loopback(host), host


async def test_serve_async_refuses_non_loopback(runtime: FakeRuntime):
    with pytest.raises(ValueError):
        await web.serve_async(runtime, host="0.0.0.0", port=18799)


# --------------------------------------------------------------------------- fake runtime
async def test_fake_tour_is_believable(runtime: FakeRuntime):
    await runtime.run_scenario("fake_tour")
    await runtime._task
    await runtime.bus.drain()
    finished = runtime.bus.find(T.SCENARIO, state="finished")[0].data
    assert finished["failed"] == 0 and finished["passed"] >= 7
    # reflex in the dark kitchen
    assert runtime.bus.find(T.PROPOSAL, by="reflex", action="lights_on", target="light_kitchen")
    # cascade on "I can't see a thing": 2B unsure (0.55) -> published as a shadow -> Claude on Bedrock acts
    two_b = [p for p in runtime.bus.find(T.PROPOSAL, by="decider-2b", action="lights_on") if p.data["confidence"] == 0.55]
    bedrock = runtime.bus.find(T.PROPOSAL, by="bedrock", action="lights_on")
    assert two_b and two_b[0].data["mode"] == "shadow" and two_b[0].data["route"] == "local"
    assert bedrock and bedrock[0].data["escalated_from"] == two_b[0].id and bedrock[0].data["mode"] == "act"
    b = bedrock[0].data
    assert b["route"] == "bedrock" and "Bedrock" in b["model"] and b["tokens"]["input"] > 0 and b["tokens"]["output"] > 0
    assert b["tool_calls"] == [{"name": "lights_on", "input": {"room": "living", "target": "light_living"}}]
    assert b["reply"] and "living room" in b["reply"]
    assert not runtime.bus.find(T.PROPOSAL, by="decider-12b")
    # the scene path escalates too: hazard 2B 0.74 (shadow) -> Bedrock alert with a tool call and a reply
    hazard = runtime.bus.find(T.PROPOSAL, by="bedrock", action="alert_parent")
    assert hazard and hazard[0].data["kind"] == "scene" and hazard[0].data["tool_calls"][0]["name"] == "alert_parent"
    # TV voice denied, child stove alert, curfew ask answered
    assert any("inside_voices_only" in v.data["rule"] for v in runtime.bus.find(T.VERDICT, verdict="deny"))
    assert runtime.bus.find(T.COMMAND, command="notify")
    assert runtime.bus.find(T.ASK, to="parents") and runtime.bus.find(T.REPLY, by="scenario")
    assert runtime.devices["tv_living"]["state"]["on"] is True
    assert not runtime.bus.errors
    # every event is JSON-able and carries the envelope the UI relies on
    for e in runtime.bus.history:
        j = json.loads(json.dumps(e.to_json(), default=str))
        assert {"topic", "data", "source", "ts", "id", "cause", "wall"} <= set(j)


async def test_fake_proposals_carry_the_xray_fields(runtime: FakeRuntime):
    """Every decider proposal carries the typed request, raw answers, route, model and threshold the
    Decider X-ray panel renders; local 2B decisions cost no tokens."""
    await runtime.step({"do": "set_time", "time": "21:30"})
    await runtime.step({"do": "say", "who": "alex", "text": "Turn on the lights"})
    await runtime.step({"do": "say", "who": "alex", "text": "I can't see a thing in here."})
    await runtime.bus.drain()
    deciders = [p.data for p in runtime.bus.find(T.PROPOSAL) if p.data["by"] in ("decider-2b", "decider-12b", "bedrock")]
    assert len(deciders) == 3
    for d in deciders:
        assert d["route"] in ("local", "bedrock", "12b") and d["model"] and d["threshold"] == runtime.settings.act_threshold
        req = d["request"]
        assert req["state"] == d["state_text"] and req["state"].endswith('"')  # ends with the quoted utterance
        assert set(req["questions"]) == {"request", "intent"}
        assert req["questions"]["request"]["type"] == "noul" and req["questions"]["intent"]["type"] == "choice"
        assert d["question"] == req["questions"]["intent"]["instructions"]
        assert abs(sum(d["probabilities"].values()) - 1) < 1e-3
    local = [d for d in deciders if d["route"] == "local" and d["mode"] == "act"][0]
    assert local["tokens"] is None and "tool_calls" not in local
    assert set(local["answers"]) == {"request", "intent"}
    assert local["answers"]["intent"]["choice"] == "Lights on" and local["answers"]["intent"]["confidence"] == 0.91
    assert abs(sum(local["answers"]["intent"]["probabilities"].values()) - 1) < 1e-3
    assert local["answers"]["request"]["noul"] == 0.91
    shadow = [d for d in deciders if d["mode"] == "shadow"][0]
    assert shadow["by"] == "decider-2b" and shadow["confidence"] < shadow["threshold"] and "Bedrock" in shadow["reason"]
    # the keyword baseline and the reflex rule say where they were decided too
    kw = runtime.bus.find(T.PROPOSAL, by="keyword")[0].data
    assert kw["route"] == "rule" and kw["tokens"] is None


async def test_fake_runtime_escalates_to_the_12b_when_asked(runtime: FakeRuntime):
    await runtime.update_settings(escalate_to="12b")
    await runtime.step({"do": "set_time", "time": "21:30"})
    await runtime.step({"do": "say", "who": "alex", "text": "I can't see a thing in here."})
    await runtime.bus.drain()
    two_b = runtime.bus.find(T.PROPOSAL, by="decider-2b", mode="shadow")
    twelve_b = runtime.bus.find(T.PROPOSAL, by="decider-12b")
    assert two_b and twelve_b and twelve_b[0].data["escalated_from"] == two_b[0].id
    d = twelve_b[0].data
    assert d["route"] == "12b" and d["model"] == "Strands Decider 12B" and d["confidence"] == 0.95 and d["tokens"] is None
    assert "tool_calls" not in d and "reply" not in d and abs(sum(d["probabilities"].values()) - 1) < 1e-3
    assert not runtime.bus.find(T.PROPOSAL, by="bedrock")
    assert runtime.devices["light_living"]["state"]["on"] is True


async def test_fake_runtime_modes_and_reset(runtime: FakeRuntime):
    await runtime.update_settings(decider_mode="2b")
    await runtime.step({"do": "set_time", "time": "21:30"})
    await runtime.step({"do": "say", "who": "alex", "text": "I can't see a thing in here."})
    await runtime.bus.drain()
    assert runtime.bus.find(T.VERDICT, verdict="ask_speaker") and not runtime.bus.find(T.PROPOSAL, by="decider-12b")
    await runtime.reset()
    assert runtime.time == "19:00" and not runtime._open_asks
    assert runtime.snapshot()["people"][0]["room"] == "living"
    assert all(a["busy"] is False for a in runtime.agent_graph())
    with pytest.raises(ValueError):
        await runtime.update_settings(bogus=1)
    assert isinstance(await runtime.update_settings(speed=3.0), Settings)
