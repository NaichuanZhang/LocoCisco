"""FastAPI app for the demo UI: static page, JSON API and a Server-Sent-Events stream of the bus.

There is NO authentication on purpose: serve() refuses to bind to anything but a loopback
address, so only processes on this machine (or an ssh -L tunnel the user opened) can reach it.
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from ..contracts import Event, RuntimeAPI, Settings, T

log = logging.getLogger("homedemo.web")

STATIC_DIR = Path(__file__).resolve().parent / "static"
HOME_PATH = Path(__file__).resolve().parent.parent / "home.json"

# Scenario-DSL steps the UI may inject by hand (world steps only; runner steps stay with the runner).
WORLD_STEPS: frozenset[str] = frozenset({
    "say", "tv_say", "outside_say", "move", "activity", "set_env", "set_device",
    "set_time", "idle", "phone", "leave", "arrive",
})
SNAPSHOT_TOPIC = "ui.snapshot"      # synthetic, stream only, never on the bus
SETTINGS_TOPIC = "ui.settings"      # synthetic, stream only
SNAPSHOT_MIN_INTERVAL = 0.2         # seconds -> at most 5 snapshots per second
HEARTBEAT_SECONDS = 15.0
CLIENT_QUEUE_SIZE = 500


def load_home() -> dict[str, Any]:
    return json.loads(HOME_PATH.read_text(encoding="utf-8"))


def is_loopback(host: str) -> bool:
    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip().strip("[]")).is_loopback
    except ValueError:
        return False


def require_loopback(host: str) -> None:
    if not is_loopback(host):
        raise ValueError(f"refusing to bind {host!r}: the demo UI has no auth and serves loopback only")


# --------------------------------------------------------------------------- stream hub
class StreamHub:
    """Fans every bus event out to the connected SSE clients.

    The Bus has no unsubscribe, so exactly one "*" handler is registered per app; each client
    owns a bounded queue and the oldest item is dropped when a slow client falls behind.
    """

    def __init__(self, runtime: RuntimeAPI, *, queue_size: int = CLIENT_QUEUE_SIZE,
                 min_interval: float = SNAPSHOT_MIN_INTERVAL) -> None:
        self._runtime = runtime
        self._clients: set[asyncio.Queue[dict[str, Any]]] = set()
        self._queue_size = queue_size
        self._min_interval = min_interval
        self._last_snapshot = -1e9      # loop time of the previous snapshot send
        self._pending: asyncio.TimerHandle | None = None
        self.last_ts = 0.0              # simulated clock of the newest bus event
        self.dropped = 0
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

    def broadcast(self, payload: dict[str, Any]) -> None:
        for q in list(self._clients):
            if q.full():
                try:
                    q.get_nowait()
                    self.dropped += 1
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(payload)

    async def _on_event(self, event: Event) -> None:
        self.last_ts = event.ts
        self.broadcast(event.to_json())
        if event.topic == T.APPLIED or event.topic.startswith("sensor."):
            self.schedule_snapshot()

    def synthetic(self, topic: str, data: dict[str, Any]) -> dict[str, Any]:
        return Event(topic, data, source="web", ts=self.last_ts).to_json()

    def snapshot_event(self) -> dict[str, Any]:
        return self.synthetic(SNAPSHOT_TOPIC, {"snapshot": self._runtime.snapshot()})

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


def sse_frame(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, ensure_ascii=False, default=str)
    return f"id: {payload.get('id', '')}\ndata: {body}\n\n".encode("utf-8")


# --------------------------------------------------------------------------- request bodies
class StepBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    do: Literal[
        "say", "tv_say", "outside_say", "move", "activity", "set_env", "set_device",
        "set_time", "idle", "phone", "leave", "arrive",
    ]


class ReplyBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1, max_length=64)
    answer: str = Field(min_length=1, max_length=64)


class SettingsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decider_mode: Literal["2b", "12b", "cascade", "fake"] | None = None
    escalate_to: Literal["bedrock", "12b", "llm"] | None = None
    act_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    ask_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    speed: float | None = Field(default=None, gt=0.0, le=20.0)
    keyword_shadow: bool | None = None

    def changes(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


# --------------------------------------------------------------------------- app
def create_app(runtime: RuntimeAPI) -> FastAPI:
    app = FastAPI(title="LocoCisco smart-home decider demo", docs_url=None, redoc_url=None, openapi_url=None)
    hub = StreamHub(runtime)
    app.state.hub = hub
    app.state.runtime = runtime
    home: dict[str, Any] = getattr(runtime, "home", None) or load_home()

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html",
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        return {
            "snapshot": runtime.snapshot(),
            "settings": asdict(runtime.settings),
            "agents": runtime.agent_graph(),
            "scenarios": runtime.scenarios(),
            "house_rules": home.get("house_rules", []),
            "home": {k: home.get(k) for k in ("name", "start", "sensors", "thresholds")},
        }

    @app.get("/api/stream")
    async def stream() -> StreamingResponse:
        async def frames() -> AsyncIterator[bytes]:
            q = hub.connect()
            try:
                yield sse_frame(hub.snapshot_event())
                while True:
                    try:
                        payload = await asyncio.wait_for(q.get(), timeout=HEARTBEAT_SECONDS)
                    except asyncio.TimeoutError:
                        yield b": ping\n\n"
                        continue
                    yield sse_frame(payload)
            finally:
                hub.disconnect(q)

        return StreamingResponse(frames(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/scenario/stop")
    async def scenario_stop() -> dict[str, Any]:
        await runtime.stop_scenario()
        return {"ok": True}

    @app.post("/api/scenario/{scenario_id}/run")
    async def scenario_run(scenario_id: str) -> dict[str, Any]:
        if scenario_id not in {s["id"] for s in runtime.scenarios()}:
            raise HTTPException(404, f"unknown scenario {scenario_id!r}")
        await runtime.run_scenario(scenario_id)
        return {"ok": True, "scenario_id": scenario_id}

    @app.post("/api/reset")
    async def reset() -> dict[str, Any]:
        await runtime.stop_scenario()
        await runtime.reset()
        hub.schedule_snapshot()
        return {"ok": True, "snapshot": runtime.snapshot()}

    @app.post("/api/step")
    async def step(body: StepBody) -> dict[str, Any]:
        try:
            await runtime.step(body.model_dump())
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, f"bad step: {exc}") from exc
        hub.schedule_snapshot()
        return {"ok": True}

    @app.post("/api/phone/reply")
    async def phone_reply(body: ReplyBody) -> dict[str, Any]:
        try:
            await runtime.phone_reply(body.request_id, body.answer)
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, f"bad reply: {exc}") from exc
        return {"ok": True}

    @app.post("/api/settings")
    async def settings(body: SettingsBody) -> dict[str, Any]:
        changes = body.changes()
        merged = {**asdict(runtime.settings), **changes}
        if merged["ask_threshold"] > merged["act_threshold"]:
            raise HTTPException(422, "ask_threshold must not exceed act_threshold")
        if changes:
            updated: Settings = await runtime.update_settings(**changes)
        else:
            updated = runtime.settings
        result = asdict(updated)
        hub.broadcast(hub.synthetic(SETTINGS_TOPIC, {"settings": result}))
        return result

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


# --------------------------------------------------------------------------- serving
def serve(runtime: RuntimeAPI, host: str = "127.0.0.1", port: int = 8780) -> None:
    """Blocking: run the UI on a loopback address. Any other host raises ValueError (no auth!)."""
    require_loopback(host)
    import uvicorn

    uvicorn.run(create_app(runtime), host=host, port=port, log_level="info")


async def serve_async(runtime: RuntimeAPI, host: str = "127.0.0.1", port: int = 8780) -> None:
    """Same as serve(), for callers that already run an event loop (e.g. the integrated runtime)."""
    require_loopback(host)
    import uvicorn

    config = uvicorn.Config(create_app(runtime), host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()
