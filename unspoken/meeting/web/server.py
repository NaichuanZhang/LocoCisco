"""FastAPI app for the room screen: the static page, a JSON API and a WebSocket stream of the bus.

There is NO authentication on purpose: serve() refuses to bind to anything but a loopback address, so
only processes on this machine (the kiosk browser on the GB10's HDMI out, or an ssh -L tunnel the user
opened) can reach it. The page itself makes no network request except to this server.

The stream hub subscribes to the bus when the app is created, so call create_app() BEFORE the runtime
starts its agents: the Bus has no unsubscribe and the hub must see every event from the first one.
"""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import math
import re
import time
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from ...core.bus import topic_matches
from ...core.events import Event
from ..contracts import MT, MeetingRuntimeAPI, MeetingSettings

log = logging.getLogger("unspoken.meeting.web")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Bus topics the page receives. Raw sensor events stay off the wire; sensor runtimes also expose
# a bounded, text/image-free telemetry projection for the recording dashboard.
STREAM_TOPICS: tuple[str, ...] = (
    MT.SCREEN, "meeting.*", MT.EGRESS, MT.LOG, MT.SCENARIO, MT.ENV, MT.SCENE,
)
# Topics after which the page gets a fresh snapshot (throttled).
SNAPSHOT_TOPICS: frozenset[str] = frozenset({MT.SCREEN, MT.ITEM, MT.PRIVACY, MT.EGRESS, MT.STATUS, MT.ENV})

SNAPSHOT_TOPIC = "ui.snapshot"      # synthetic, stream only, never on the bus
SETTINGS_TOPIC = "ui.settings"      # synthetic, stream only
PING_TOPIC = "ui.ping"              # synthetic heartbeat
RESET_TOPIC = "ui.reset"            # browser cache/queue barrier, not an external-message deletion
TELEMETRY_TOPIC = "sensor.telemetry" # synthetic, allowlisted observations, never on the bus
SNAPSHOT_MIN_INTERVAL = 0.2         # seconds -> at most 5 snapshots per second
TELEMETRY_MIN_INTERVAL = 0.2        # video telemetry at most 5 Hz; completed segments send immediately
HEARTBEAT_SECONDS = 15.0
CLIENT_QUEUE_SIZE = 500
DEFAULT_PORT = 8780                 # docs/PLAN.md: room screen + API on the GB10

# Everything else on the meeting stream can contain speech, model requests/replies, notification
# bodies or free-text diagnostics. Bus.retain controls history, NOT delivery to this subscriber.
PUBLIC_TOPICS = frozenset({MT.PRIVACY, MT.STATUS, MT.CLOCK, MT.EGRESS, MT.ENV,
                           SETTINGS_TOPIC, PING_TOPIC, RESET_TOPIC})


# --------------------------------------------------------------------------- loopback only
def is_loopback(host: str) -> bool:
    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip().strip("[]")).is_loopback
    except ValueError:
        return False


def require_loopback(host: str) -> None:
    if not is_loopback(host):
        raise ValueError(f"refusing to bind {host!r}: the room screen has no auth and serves loopback only")


def target_is_local(target: str) -> bool:
    """True when a RoleHealth.target stays on this box or its LAN: in-process providers, model paths,
    and URLs whose host is loopback, private (RFC 1918) or link-local. Shown as the LOCAL badge."""
    t = (target or "").strip()
    if not t or t.lower() in ("in-process", "none", "disabled"):
        return True
    if "://" not in t:
        return True                      # a model path or a bare label, nothing to connect to
    host = (urlsplit(t).hostname or "").strip("[]").lower()
    if host in ("localhost", ""):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host.endswith((".local", ".lan", ".home", ".internal"))
    return ip.is_loopback or ip.is_private or ip.is_link_local


# --------------------------------------------------------------------------- stream hub
class StreamHub:
    """Fans the streamed bus events out to the connected WebSocket clients.

    The Bus has no unsubscribe, so exactly one "*" handler is registered per app; each client owns a
    bounded queue and the oldest item is dropped when a slow client falls behind.
    """

    def __init__(self, runtime: MeetingRuntimeAPI, *, queue_size: int = CLIENT_QUEUE_SIZE,
                 min_interval: float = SNAPSHOT_MIN_INTERVAL, topics: tuple[str, ...] = STREAM_TOPICS) -> None:
        self._runtime = runtime
        self._clients: set[asyncio.Queue[dict[str, Any]]] = set()
        self._queue_size = queue_size
        self._min_interval = min_interval
        self._topics = topics
        self._last_snapshot = -1e9      # loop time of the previous snapshot send
        self._pending: asyncio.TimerHandle | None = None
        self.last_ts = 0.0              # meeting clock of the newest bus event
        self.dropped = 0
        self._epoch = 0
        self._reset_wall = 0.0
        self._content_after = 0.0
        self._privacy_wall = 0.0
        self._resetting = False
        self._state_object = getattr(runtime, "state", None)
        initial = runtime.snapshot()
        self._runtime_privacy = initial.get("privacy", {}).get("state", "on")
        self._privacy = self._runtime_privacy
        self._caption_after: float | None = None
        # Sensor-capable runtimes opt in through their existing snapshot contract. The standalone
        # screen fake has no sensor pipeline; do not suggest that its scripted story is inference.
        self._telemetry_enabled = "sensors" in initial
        self._sensor_audio: dict[str, Any] | None = None
        self._sensor_vision: dict[str, Any] | None = None
        self._sensor_counts = {"segments": 0, "frames": 0}
        self._last_telemetry = -1e9
        # Tombstones contain identifiers/digests, never copies of wiped card/transcript text.
        self._hidden_items: set[str] = set()
        self._hidden_cards: set[str] = set()
        runtime.bus.subscribe("*", self._on_event, name="web.stream")

    @property
    def clients(self) -> int:
        return len(self._clients)

    def connect(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._queue_size)
        self._clients.add(q)
        return q

    def disconnect(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._clients.discard(q)

    def wants(self, topic: str) -> bool:
        return any(topic_matches(p, topic) for p in self._topics)

    def broadcast(self, payload: dict[str, Any]) -> None:
        if self._epoch and "topic" in payload:
            payload = {**payload, "stream_epoch": self._epoch, "stream_privacy": self._private_state()}
        for q in list(self._clients):
            if q.full():
                try:
                    q.get_nowait()
                    self.dropped += 1
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(payload)

    @staticmethod
    def _card_key(card: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(card, sort_keys=True, default=str).encode()).hexdigest()

    def _hide_content(self, snapshot: dict[str, Any]) -> None:
        self._hidden_items.update(str(i["item_id"]) for i in snapshot.get("items", [])
                                  if i.get("item_id"))
        self._hidden_cards.update(self._card_key(c) for c in snapshot.get("screen", {}).get("cards", []))

    def _barrier(self, *, wall: float, privacy: str, reset: bool = False) -> None:
        """No awaits: purge and advance the epoch atomically with respect to bus callbacks.

        The epoch also invalidates a frame already removed from a queue by the websocket pump.
        It cannot recall bytes already handed to the socket (or messages sent to a notifier).
        """
        self._epoch += 1
        self._content_after = max(self._content_after, wall)
        self._sensor_audio = self._sensor_vision = None
        self._sensor_counts = {"segments": 0, "frames": 0}
        self._last_telemetry = -1e9
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        for q in self._clients:
            keep = []
            while not q.empty():
                payload = q.get_nowait()
                if not reset and payload.get("topic") in PUBLIC_TOPICS - {MT.PRIVACY, RESET_TOPIC}:
                    keep.append(payload)
            for payload in keep:
                q.put_nowait({**payload, "stream_epoch": self._epoch, "stream_privacy": self._private_state()})
        self.broadcast(self.synthetic(RESET_TOPIC, {"privacy": privacy}))

    def begin_reset(self) -> None:
        """Call before awaiting runtime/source teardown, including non-HTTP replay resets."""
        self._resetting = True
        self._reset_wall = time.time()
        self._privacy_wall = self._reset_wall
        self._hidden_items.clear()
        self._hidden_cards.clear()
        self._caption_after = None
        self.last_ts = 0.0
        self._barrier(wall=self._reset_wall, privacy="on", reset=True)

    def end_reset(self) -> None:
        self._state_object = getattr(self._runtime, "state", None)
        self._runtime_privacy = self._runtime.snapshot().get("privacy", {}).get("state", "on")
        self._privacy = self._runtime_privacy
        self._resetting = False
        # Drop content produced during teardown too; a new session must not inherit it.
        self._reset_wall = time.time()
        self._privacy_wall = self._reset_wall
        self._barrier(wall=self._reset_wall, privacy=self._privacy, reset=True)
        self.schedule_snapshot()

    def _sync_runtime(self) -> dict[str, Any]:
        state = getattr(self._runtime, "state", None)
        if state is not self._state_object and not self._resetting:
            # MeetingRuntime replaces its state on reset. This covers resets initiated elsewhere;
            # controllers can use begin_reset/end_reset to install the barrier before teardown.
            self.begin_reset()
            self.end_reset()
        snapshot = self._runtime.snapshot()
        privacy = snapshot.get("privacy", {}).get("state", "on")
        if privacy != self._runtime_privacy and not self._resetting:
            self._runtime_privacy = privacy
            self._hide_content(snapshot)
            self._caption_after = max(self.last_ts, float(snapshot.get("clock") or 0))
            self._barrier(wall=time.time(), privacy=privacy)
        return snapshot

    def _private(self) -> bool:
        # A queued off-record event must not be undone by an already-resumed runtime snapshot;
        # conversely a stale "on" event must not override an already-private runtime.
        return self._resetting or self._privacy != "on" or self._runtime_privacy != "on"

    def _private_state(self) -> str:
        if self._resetting:
            return "paused"
        return self._privacy if self._privacy != "on" else self._runtime_privacy

    def _screen(self, screen: dict[str, Any]) -> dict[str, Any]:
        screen = dict(screen)
        if self._private():
            screen.update(privacy=self._private_state(),
                          captions=[], cards=[], banner=None)
        elif self._caption_after is not None:
            screen["captions"] = [c for c in screen.get("captions", [])
                                  if float(c.get("t", 0)) > self._caption_after]
            screen["cards"] = [c for c in screen.get("cards", [])
                               if self._card_key(c) not in self._hidden_cards
                               and str(c.get("item_id", "")) not in self._hidden_items
                               # Aggregates have no source timestamp; they can re-quote wiped items.
                               and c.get("card") not in {"tasks", "catch_up"}]
        return screen

    def browser_snapshot(self) -> dict[str, Any]:
        snapshot = dict(self._sync_runtime())
        snapshot["privacy"] = {**snapshot.get("privacy", {}), "reason": ""}
        if self._private():
            self._hide_content(snapshot)
            privacy = self._private_state()
            snapshot["privacy"] = {**snapshot.get("privacy", {}), "state": privacy, "reason": ""}
            snapshot["items"] = []
        else:
            snapshot["items"] = [i for i in snapshot.get("items", [])
                                 if str(i.get("item_id", "")) not in self._hidden_items]
        snapshot["screen"] = self._screen(snapshot.get("screen", {}))
        if self._telemetry_enabled:
            snapshot["telemetry"] = None if self._private() else self._telemetry_snapshot()
        return snapshot

    def _telemetry_snapshot(self) -> dict[str, Any]:
        """Copy the last public observations; never copy an original sensor payload wholesale."""
        vision = self._sensor_vision
        return {
            "audio": dict(self._sensor_audio) if self._sensor_audio is not None else None,
            "vision": {**vision, "faces": [{**face, "bbox": list(face["bbox"])}
                                           for face in vision["faces"]]} if vision is not None else None,
            "counts": dict(self._sensor_counts),
        }

    @staticmethod
    def _number(value: Any, *, maximum: float | None = None) -> float | None:
        # Reject booleans and strings too: telemetry is a numeric/identifier projection,
        # not another route for arbitrary model output to enter the browser.
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        return min(maximum, max(0.0, float(value))) if maximum is not None else max(0.0, float(value))

    @staticmethod
    def _sensor_id(value: Any) -> str | None:
        return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.?\-]{1,64}", value) else None

    def _observe_sensor(self, event: Event) -> None:
        """Expose completed model observations, not invented model-start or forward-pass events.

        This uses the same privacy epoch and queue/dequeue barriers as meeting content. No
        telemetry from a private or retired segment is retained for a later reconnect.
        """
        if (not self._telemetry_enabled or self._private() or event.wall <= self._content_after
                or event.topic not in {MT.SEGMENT, MT.SPEAKERS}):
            return
        data = event.data
        if event.topic == MT.SEGMENT:
            start, end = self._number(data.get("start")), self._number(data.get("end"))
            if start is None or end is None or end < start:
                return
            if self._caption_after is not None and start <= self._caption_after:
                return
            self._sensor_audio = {
                "start": start, "end": end,
                "asr_ms": self._number(data.get("asr_ms")), "diar_ms": self._number(data.get("diar_ms")),
                "voice": self._sensor_id(data.get("voice")),
                "voice_conf": self._number(data.get("voice_conf"), maximum=1.0),
            }
            self._sensor_counts["segments"] += 1
        else:
            t = self._number(data.get("t"))
            if t is None or (self._caption_after is not None and t <= self._caption_after):
                return
            faces = []
            # The camera may see many boxes; bound both the payload and retained telemetry.
            observations = data.get("faces")
            for face in observations[:32] if isinstance(observations, (list, tuple)) else []:
                if not isinstance(face, dict):
                    continue
                bbox = face.get("bbox")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                box = [self._number(v, maximum=1.0) for v in bbox]
                if any(v is None for v in box):
                    continue
                faces.append({"track_id": self._sensor_id(face.get("track_id")),
                              "seat": self._sensor_id(face.get("seat")), "bbox": box,
                              "speaking": self._number(face.get("speaking"), maximum=1.0)})
            self._sensor_vision = {
                "t": t, "faces": faces,
                "face_count": sum(face["seat"] != "screen" for face in faces),
                "screen_talking": data.get("screen_talking") is True,
            }
            self._sensor_counts["frames"] += 1
        now = asyncio.get_running_loop().time()
        if event.topic == MT.SPEAKERS and now - self._last_telemetry < TELEMETRY_MIN_INTERVAL:
            return
        self._last_telemetry = now
        payload = self.ready(self.synthetic(TELEMETRY_TOPIC, self._telemetry_snapshot()))
        if payload is not None:
            self.broadcast(payload)

    def ready(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Recheck after dequeue as well as at subscription time; never trust queue age."""
        self._sync_runtime()
        topic = payload.get("topic")
        if payload.get("stream_epoch", 0) != self._epoch:
            return None
        if topic == SNAPSHOT_TOPIC:
            return self.snapshot_event()  # never send a captured, possibly pre-wipe snapshot
        if topic not in PUBLIC_TOPICS:
            if payload.get("wall", 0) <= self._content_after:
                return None
            if topic == MT.SCREEN:
                return {**payload, "data": self._screen(payload["data"])}
            if self._private():
                if topic == MT.SCENARIO:
                    return {**payload, "data": {k: v for k, v in payload["data"].items()
                                               if k in {"scenario_id", "state", "step_index", "passed", "failed"}}}
                if topic == MT.SCENE:
                    return {**payload, "data": {k: v for k, v in payload["data"].items()
                                               if k in {"t", "mode", "conf", "by"}}}
                return None
        if topic == MT.PRIVACY:
            state = self._private_state()
            return {**payload, "data": {**payload["data"], "reason": "", "state": state}}
        return payload

    async def _on_event(self, event: Event) -> None:
        snapshot = self._sync_runtime()
        if event.wall < self._reset_wall:
            return
        if event.topic == MT.PRIVACY:
            if event.wall < self._privacy_wall:
                return
            self._privacy_wall = event.wall
            changed = self._privacy != event.data.get("state", "on")
            self._privacy = event.data.get("state", "on")
            if changed or event.data.get("wiped_from") is not None:
                self._hide_content(snapshot)
                self._caption_after = max(self.last_ts, event.ts, float(snapshot.get("clock") or 0))
                self._barrier(wall=event.wall, privacy=self._privacy)
        self.last_ts = event.ts
        self._observe_sensor(event)
        if self.wants(event.topic):
            payload = event.to_json()
            if self._epoch:
                payload["stream_epoch"] = self._epoch
            payload = self.ready(payload)
            if payload is not None:
                self.broadcast(payload)
        if event.topic in SNAPSHOT_TOPICS:
            self.schedule_snapshot()

    def synthetic(self, topic: str, data: dict[str, Any]) -> dict[str, Any]:
        payload = Event(topic, data, source="web", ts=self.last_ts).to_json()
        if self._epoch:
            payload.update(stream_epoch=self._epoch, stream_privacy=self._private_state())
        return payload

    def snapshot_event(self) -> dict[str, Any]:
        payload = self.synthetic(SNAPSHOT_TOPIC, {"snapshot": self.browser_snapshot(),
                                                 "settings": asdict(self._runtime.settings)})
        return payload

    def schedule_snapshot(self) -> None:
        """Send a snapshot now, or once the throttle window closes (the trailing one is never lost)."""
        if self._pending is not None:
            return
        loop = asyncio.get_running_loop()
        wait = self._last_snapshot + self._min_interval - loop.time()
        if wait <= 0:
            self._send_snapshot()
        else:
            self._pending = loop.call_later(wait, self._send_snapshot)

    def _send_snapshot(self) -> None:
        self._pending = None
        self._last_snapshot = asyncio.get_running_loop().time()
        if self._clients:
            self.broadcast(self.snapshot_event())


def ws_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------- request bodies
ChatCommand = Literal["here", "offrecord", "resume", "recap", "agenda", "undo", "approve", "deny", "ask"]


class ChatBody(BaseModel):
    """A Discord command injected by hand (rehearsal). Same shape as ChatCommandData."""
    model_config = ConfigDict(extra="forbid")
    command: ChatCommand
    user: str = Field(default="rehearsal", min_length=1, max_length=64)
    args: dict[str, Any] = Field(default_factory=dict)
    text: str = Field(default="", max_length=2000)


class SettingsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    act_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    ask_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    escalate: StrictBool | None = None
    wipe_seconds: float | None = Field(default=None, ge=0.0, le=3600.0)
    caption_lines: int | None = Field(default=None, ge=1, le=20)
    speed: float | None = Field(default=None, gt=0.0, le=50.0)

    def changes(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


# --------------------------------------------------------------------------- app
def create_app(runtime: MeetingRuntimeAPI) -> FastAPI:
    app = FastAPI(title="LocoCisco room screen", docs_url=None, redoc_url=None, openapi_url=None)
    hub = StreamHub(runtime)
    app.state.hub = hub
    app.state.runtime = runtime

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html",
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/snapshot")
    async def snapshot(response: Response) -> dict[str, Any]:
        snapshot = hub.browser_snapshot()
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-LocoCisco-Stream-Epoch"] = str(hub._epoch)
        return {"snapshot": snapshot, "settings": asdict(runtime.settings)}

    @app.get("/api/models")
    async def models() -> dict[str, Any]:
        rows = []
        for m in runtime.snapshot().get("models", []) or []:
            row = dict(m)
            row["local"] = target_is_local(str(row.get("target", "")))
            rows.append(row)
        return {"models": rows, "all_local": all(r["local"] for r in rows)}

    @app.get("/api/scenarios")
    async def scenarios() -> list[dict[str, Any]]:
        return runtime.scenarios()

    @app.post("/api/scenario/{scenario_id}")
    async def scenario_run(scenario_id: str) -> dict[str, Any]:
        if scenario_id not in {s["id"] for s in runtime.scenarios()}:
            raise HTTPException(404, f"unknown scenario {scenario_id!r}")
        await runtime.run_scenario(scenario_id)
        return {"ok": True, "scenario_id": scenario_id}

    @app.post("/api/stop")
    async def stop() -> dict[str, Any]:
        await runtime.stop_scenario()
        return {"ok": True}

    @app.post("/api/reset")
    async def reset() -> dict[str, Any]:
        hub.begin_reset()
        try:
            await runtime.stop_scenario()
            await runtime.reset()
        finally:
            hub.end_reset()
        return {"ok": True, "snapshot": hub.browser_snapshot()}

    @app.post("/api/chat")
    async def chat(body: ChatBody) -> dict[str, Any]:
        try:
            result = await runtime.chat(body.model_dump())  # type: ignore[arg-type]
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, f"bad command: {exc}") from exc
        hub.schedule_snapshot()
        return {"ok": bool(result.get("ok", True)), "text": str(result.get("text", ""))}

    @app.post("/api/settings")
    async def settings(body: SettingsBody) -> dict[str, Any]:
        changes = body.changes()
        merged = {**asdict(runtime.settings), **changes}
        if merged["ask_threshold"] > merged["act_threshold"]:
            raise HTTPException(422, "ask_threshold must not exceed act_threshold")
        if changes:
            updated: MeetingSettings = await runtime.update_settings(**changes)
        else:
            updated = runtime.settings
        result = asdict(updated)
        hub.broadcast(hub.synthetic(SETTINGS_TOPIC, {"settings": result}))
        return result

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        q = hub.connect()

        async def pump() -> None:
            await websocket.send_text(ws_text(hub.snapshot_event()))
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    payload = hub.synthetic(PING_TOPIC, {})
                payload = hub.ready(payload)
                if payload is not None:
                    await websocket.send_text(ws_text(payload))

        pump_task = asyncio.create_task(pump(), name="web.ws.pump")
        try:
            while True:                      # the page sends nothing; this only notices the close
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            pump_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await pump_task
            hub.disconnect(q)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


# --------------------------------------------------------------------------- serving
def serve(runtime: MeetingRuntimeAPI, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
    """Blocking: run the room screen on a loopback address. Any other host raises ValueError (no auth!)."""
    require_loopback(host)
    import uvicorn

    uvicorn.run(create_app(runtime), host=host, port=port, log_level="info")


async def serve_async(runtime: MeetingRuntimeAPI, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                      app: FastAPI | None = None) -> None:
    """Same as serve(), for callers that already run an event loop (the integrated runtime). Pass the
    app you created earlier with create_app() so the stream hub was subscribed before the agents started."""
    require_loopback(host)
    import uvicorn

    config = uvicorn.Config(app or create_app(runtime), host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()
