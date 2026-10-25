"""The room's MCP server: what the OpenClaw agent in the NemoClaw sandbox can ask the room.

Streamable HTTP (python `mcp` SDK), bearer auth on every MCP request, loopback only. The sandbox reaches it
through the TLS proxy (unspoken.bridge.tls_proxy) on the docker bridge IP, which checks the same bearer.
Tools are thin: they read MeetingRuntimeAPI or inject Discord commands through `runtime.chat()`; the two
"do something in the room" tools (create_task, set_light) are published as USER proposals on the bus and the
meeting policy still decides.

    serve(runtime, host="127.0.0.1", port=8791, token=...)   # blocking
    create_app(runtime, token) -> ASGI app                       # for tests / embedding
"""
from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import os
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable

from mcp.server.mcpserver import MCPServer

from ..core.events import Event
from ..meeting.contracts import MT, ChatCommandData, ItemKind, MeetingRuntimeAPI, ProposalData
from .config import ENV_MCP_TOKEN, BridgeConfigError

log = logging.getLogger("unspoken.bridge.mcp")

SERVER_NAME = "unspoken-room"
DEFAULT_PORT = 8791            # GB10; the dev desktop uses 8792 (config/bridge.toml, UNSPOKEN_MCP_PORT)
MIN_TOKEN_CHARS = 16
OFF_RECORD = ("Off-record content never exists: anything said while notes were paused or after someone said "
              "'off the record' was never stored, so it cannot be returned by any tool.")
LIGHT_SCENES = ("talk", "slides", "break", "privacy", "off_record", "bright", "dim")
ITEM_KINDS = ("task", "decision", "question", "parking_lot")

_INSTRUCTIONS = f"""You are talking to LocoCisco, the meeting-room agent. All inference is local to the room's box.
{OFF_RECORD}
Prefer room_status before anything else; use transcript sparingly (recent minutes only). create_task and
set_light are proposals: the room's policy decides and may ask the meeting owner first."""


# --------------------------------------------------------------------------- loopback guard
def is_loopback(host: str) -> bool:
    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip().strip("[]")).is_loopback
    except ValueError:
        return False


def require_loopback(host: str) -> None:
    if not is_loopback(host):
        raise ValueError(f"refusing to bind the MCP server to {host!r}: loopback only; the TLS proxy is the way in")


def require_token(token: str | None) -> str:
    if not token or len(token) < MIN_TOKEN_CHARS:
        raise BridgeConfigError(f"MCP bearer token missing or shorter than {MIN_TOKEN_CHARS} chars "
                                f"(set {ENV_MCP_TOKEN}; make one with: openssl rand -hex 32)")
    return token


# --------------------------------------------------------------------------- verdict watcher
class VerdictWatcher:
    """Remembers policy verdicts by proposal id so a tool can answer with the decision when it is quick."""

    def __init__(self, runtime: MeetingRuntimeAPI, *, keep: int = 500) -> None:
        self._seen: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._keep = keep
        self._wakeups: set[asyncio.Event] = set()
        runtime.bus.subscribe(MT.VERDICT, self._on_verdict, name="bridge.mcp.verdicts")

    async def _on_verdict(self, event: Event) -> None:
        pid = event.data.get("proposal_id")
        if not pid:
            return
        self._seen[str(pid)] = dict(event.data)
        while len(self._seen) > self._keep:
            self._seen.popitem(last=False)
        for w in list(self._wakeups):
            w.set()

    async def wait(self, proposal_id: str, timeout: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        while True:
            if proposal_id in self._seen:
                return self._seen[proposal_id]
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            ev = asyncio.Event()
            self._wakeups.add(ev)
            try:
                await asyncio.wait_for(ev.wait(), timeout=left)
            except asyncio.TimeoutError:
                return self._seen.get(proposal_id)
            finally:
                self._wakeups.discard(ev)


# --------------------------------------------------------------------------- tools
def _clip(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def build_server(runtime: MeetingRuntimeAPI, *, verdict_wait: float = 3.0,
                 clock: Callable[[], float] | None = None) -> MCPServer:
    """Register the room tools on an MCPServer. `clock` is the meeting clock (seconds) used by transcript()."""
    server = MCPServer(SERVER_NAME, instructions=_INSTRUCTIONS, version="0.1")
    watcher = VerdictWatcher(runtime)

    def now() -> float:
        if clock is not None:
            return clock()
        tr = runtime.transcript()
        return float(tr[-1].get("end", tr[-1].get("start", 0.0))) if tr else 0.0

    async def chat(command: str, user: str, args: dict[str, Any] | None = None, text: str = "") -> dict[str, Any]:
        cmd: ChatCommandData = {"command": command, "user": _clip(user or "Discord", 80),  # type: ignore[typeddict-item]
                                "args": args or {}, "text": _clip(text, 2000)}
        try:
            result = await runtime.chat(cmd)
        except Exception as exc:  # the agent gets a readable error, the room keeps running
            log.exception("chat %s failed", command)
            return {"ok": False, "text": f"{type(exc).__name__}: {exc}"}
        return {"ok": bool(result.get("ok", True)), "text": str(result.get("text", ""))}

    async def propose(kind: str, action: str, args: dict[str, Any], reason: str, user: str) -> dict[str, Any]:
        data: ProposalData = {  # type: ignore[typeddict-item]
            "by": "user", "mode": "act", "kind": kind, "action": action, "args": args, "confidence": 1.0,
            "probabilities": {}, "reason": reason, "utterance_id": None, "escalated_from": None,
            "route": "user", "model": "discord", "request": {}, "answers": {}, "threshold": 0.0,
            "latency_ms": 0.0, "tokens": None, "tool_calls": [], "reply": None,
        }
        event = Event(MT.PROPOSAL, dict(data), source=f"bridge.mcp:{_clip(user, 40) or 'discord'}", ts=now())
        await runtime.bus.publish(event)
        verdict = await watcher.wait(event.id, verdict_wait)
        out: dict[str, Any] = {"ok": True, "proposal_id": event.id, "action": action, "args": args}
        if verdict is None:
            out.update(verdict="pending", text="Submitted to the room; the policy has not answered yet.")
        else:
            out.update(verdict=verdict.get("verdict"), risk=verdict.get("risk"), rule=verdict.get("rule"),
                       text={"act": "Done.", "ask_owner": "Asked the meeting owner to approve first.",
                             "deny": "The room's policy declined this.", "ignore": "The room ignored this."
                             }.get(str(verdict.get("verdict")), "Decided."))
        return out

    @server.tool(description="Where the meeting is: title, agenda topic, minutes left, who is checked in, "
                             "privacy state (on / paused / off_record), device states, open item counts and the "
                             "egress meter. Call this first. " + OFF_RECORD)
    async def room_status() -> dict[str, Any]:
        snap = runtime.snapshot()
        items = snap.get("items", []) or []
        return {
            "status": snap.get("status", {}),
            "privacy": snap.get("privacy", {}),
            "participants": snap.get("participants", []),
            "devices": snap.get("devices", []),
            "banner": (snap.get("screen") or {}).get("banner"),
            "open_items": {k: sum(1 for i in items if i.get("kind") == k and i.get("status") == "open") for k in ITEM_KINDS},
            "egress": snap.get("egress", {}),
            "models": [m.get("label") or m.get("role") for m in snap.get("models", []) or []],
        }

    @server.tool(description="On-record transcript lines from the last N minutes (default 10, max 120) as "
                             "[{t, name, text}]. Names come from check-ins; unknown voices show as 'voice-A'. "
                             + OFF_RECORD)
    async def transcript(since_minutes: float = 10.0) -> dict[str, Any]:
        minutes = max(0.1, min(float(since_minutes), 120.0))
        since = max(0.0, now() - minutes * 60.0)
        lines = runtime.transcript(since=since)
        out = []
        for u in lines[-200:]:
            sp = u.get("speaker") or {}
            out.append({"t": round(float(u.get("start", 0.0)), 1),
                        "name": sp.get("name") or sp.get("voice") or ("guest" if sp.get("guest") else "unknown"),
                        "text": u.get("text", "")})
        return {"since_minutes": minutes, "count": len(out), "lines": out}

    @server.tool(description="Tasks, decisions, questions and parking-lot ideas captured so far. kind: task | "
                             "decision | question | parking_lot | all. Each has item_id, text, owner, due, status "
                             "(open / done / undone / pending_approval). " + OFF_RECORD)
    async def list_items(kind: str = "all") -> dict[str, Any]:
        k = (kind or "all").strip().lower()
        if k not in ITEM_KINDS and k != "all":
            return {"ok": False, "text": f"kind must be one of {', '.join(ITEM_KINDS)} or all"}
        items = runtime.items(None if k == "all" else k)  # type: ignore[arg-type]
        return {"kind": k, "count": len(items), "items": [dict(i) for i in items]}

    @server.tool(description="Search the room's local documents (the folder in config/meeting.toml) and return the "
                             "best chunks with scores. Nothing leaves the box. " + OFF_RECORD)
    async def search_docs(query: str, k: int = 3) -> dict[str, Any]:
        q = _clip(query, 500)
        if not q:
            return {"ok": False, "text": "query is empty"}
        hits = await runtime.search_docs(q, k=max(1, min(int(k), 10)))
        return {"query": q, "hits": [dict(h) for h in hits]}

    @server.tool(description="Discord-ready markdown recap of the meeting so far: decisions, tasks with owners, "
                             "open questions. Generated by the local model. " + OFF_RECORD)
    async def recap() -> str:
        return await runtime.recap()

    @server.tool(description="Check a person in: '@LocoCisco I'm Priya' -> checkin(name='Priya', user='<Discord "
                             "display name>'). Binds the current unknown voice/seat to that name for this meeting "
                             "only; nothing is stored across meetings.")
    async def checkin(name: str, user: str = "") -> dict[str, Any]:
        n = _clip(name, 60)
        if not n:
            return {"ok": False, "text": "name is empty"}
        return await chat("here", user or n, {"name": n})

    @server.tool(description="Stop taking notes now ('off the record', /offrecord). Always allowed. Recent lines are "
                             "wiped per the room's wipe window; the room screen shows the pause. reason: one short "
                             "human sentence, e.g. 'Sam asked in Discord'.")
    async def pause_notes(reason: str = "asked in Discord", user: str = "") -> dict[str, Any]:
        return await chat("offrecord", user, {"reason": _clip(reason, 200)})

    @server.tool(description="Resume notes after a pause. The policy may ask the meeting owner when the pause came "
                             "from a sensitive hint; the reply says which.")
    async def resume_notes(user: str = "") -> dict[str, Any]:
        return await chat("resume", user)

    @server.tool(description="Propose a task for the room to log and post (owner: a checked-in name or empty; text: "
                             "what to do; due: free text like 'Friday'). This is a USER proposal: the room's policy "
                             "decides (medium risk; a ❌ reaction on the posted task undoes it). Returns the verdict "
                             "when it arrives within a few seconds, else 'pending'.")
    async def create_task(text: str, owner: str = "", due: str = "", user: str = "") -> dict[str, Any]:
        t = _clip(text, 500)
        if not t:
            return {"ok": False, "text": "text is empty"}
        return await propose("task", "create_task", {"owner": _clip(owner, 60) or None, "text": t,
                                                     "due": _clip(due, 60) or None},
                             f"{user or 'Someone'} asked in Discord", user)

    @server.tool(description="Propose a lamp scene: talk | slides | break | privacy | off_record | bright | dim. "
                             "USER proposal, low risk; the policy decides.")
    async def set_light(scene: str, user: str = "") -> dict[str, Any]:
        s = (scene or "").strip().lower()
        if s not in LIGHT_SCENES:
            return {"ok": False, "text": f"scene must be one of {', '.join(LIGHT_SCENES)}"}
        return await propose("room", "set_light", {"scene": s}, f"{user or 'Someone'} asked in Discord", user)

    @server.tool(description="Undo a captured item (a ❌ reaction or 'undo that'): item_id from list_items or the "
                             "posted message. Marks it undone; the Discord post is updated by the room.")
    async def undo_item(item_id: str, user: str = "") -> dict[str, Any]:
        i = _clip(item_id, 64)
        if not i:
            return {"ok": False, "text": "item_id is empty"}
        return await chat("undo", user, {"item_id": i})

    @server.tool(description="Answer a pending approval the room asked the meeting owner for (✅ approve / ❌ deny): "
                             "request_id from the approval message, approve true/false.")
    async def decide_request(request_id: str, approve: bool, user: str = "") -> dict[str, Any]:
        r = _clip(request_id, 64)
        if not r:
            return {"ok": False, "text": "request_id is empty"}
        return await chat("approve" if approve else "deny", user, {"request_id": r})

    return server


# --------------------------------------------------------------------------- ASGI app with bearer auth
class BearerGate:
    """Pure ASGI wrapper: constant-time bearer check on every HTTP request except GET /health."""

    def __init__(self, app: Callable[..., Awaitable[None]], token: str, *, health_path: str = "/health") -> None:
        self._app = app
        self._token = token.encode("utf-8")
        self._health = health_path
        self.rejected = 0

    async def __call__(self, scope: dict[str, Any], receive: Callable[[], Awaitable[dict[str, Any]]],
                       send: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)   # lifespan passes straight through
            return
        if scope.get("path") == self._health and scope.get("method") in ("GET", "HEAD"):
            await _json(send, 200, {"ok": True, "server": SERVER_NAME})
            return
        auth = b""
        for k, v in scope.get("headers", []):
            if k == b"authorization":
                auth = v
                break
        ok = auth[:7].lower() == b"bearer " and hmac.compare_digest(auth[7:].strip(), self._token)
        if not ok:
            self.rejected += 1
            await _json(send, 401, {"error": "unauthorized"},
                        extra=[(b"www-authenticate", b'Bearer realm="unspoken"')])
            return
        await self._app(scope, receive, send)


async def _json(send: Callable[[dict[str, Any]], Awaitable[None]], status: int, body: dict[str, Any],
                extra: list[tuple[bytes, bytes]] | None = None) -> None:
    raw = json.dumps(body).encode("utf-8")
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode()),
                            (b"cache-control", b"no-store")] + (extra or [])})
    await send({"type": "http.response.body", "body": raw})


def create_app(runtime: MeetingRuntimeAPI, token: str, *, host: str = "127.0.0.1", path: str = "/mcp",
               stateless: bool = True, json_response: bool = True, verdict_wait: float = 3.0,
               clock: Callable[[], float] | None = None) -> BearerGate:
    """ASGI app: bearer-gated Streamable HTTP MCP at `path`, GET /health open (returns no room data)."""
    token = require_token(token)
    server = build_server(runtime, verdict_wait=verdict_wait, clock=clock)
    inner = server.streamable_http_app(streamable_http_path=path, json_response=json_response,
                                       stateless_http=stateless, host=host)
    app = BearerGate(inner, token)
    app.server = server  # type: ignore[attr-defined]
    return app


async def serve(runtime: MeetingRuntimeAPI, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                token: str | None = None, **kw: Any) -> None:
    """The fixed interface: `async serve(runtime, host, port, token)`. Loopback only; token from the argument
    or UNSPOKEN_MCP_TOKEN. Runs until cancelled (uvicorn in the caller's event loop)."""
    require_loopback(host)
    import uvicorn

    app = create_app(runtime, require_token(token or os.environ.get(ENV_MCP_TOKEN)), host=host, **kw)
    config = uvicorn.Config(app, host=host, port=port, log_level="info", lifespan="on")
    await uvicorn.Server(config).serve()


def run(runtime: MeetingRuntimeAPI, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
        token: str | None = None, **kw: Any) -> None:
    """Blocking wrapper around serve() for a standalone process."""
    asyncio.run(serve(runtime, host, port, token, **kw))


def _make_runtime(kind: str) -> MeetingRuntimeAPI:
    if kind == "fake":
        from .dev_fake import FakeMeetingRuntime
        return FakeMeetingRuntime()
    try:
        from ..meeting.runtime import MeetingRuntime  # meeting workstream; may not exist in this worktree yet
    except ImportError as exc:  # pragma: no cover - depends on another workstream
        raise SystemExit(f"unspoken.meeting.runtime is not available ({exc}); use --runtime fake") from exc
    return MeetingRuntime()  # type: ignore[call-arg]


def main(argv: list[str] | None = None) -> int:
    """`python -m unspoken.bridge.mcp_server --runtime fake --port 8792` (token from UNSPOKEN_MCP_TOKEN)."""
    import argparse

    from .config import load_bridge_config, mcp_settings

    p = argparse.ArgumentParser(description="LocoCisco room MCP server (Streamable HTTP, bearer, loopback only)")
    p.add_argument("--runtime", choices=("fake", "sim"), default="fake",
                   help="fake: the dev double (no models); sim: unspoken.meeting.runtime.MeetingRuntime")
    p.add_argument("--host", default=None, help="loopback only (default from config/bridge.toml [mcp].host)")
    p.add_argument("--port", type=int, default=None, help="default from config/bridge.toml / UNSPOKEN_MCP_PORT")
    p.add_argument("--config", default=None, help="path to bridge.toml")
    args = p.parse_args(argv)
    cfg = mcp_settings(load_bridge_config(args.config))
    host = args.host or str(cfg["host"])
    port = args.port or int(cfg["port"])
    try:
        require_loopback(host)
        token = require_token(os.environ.get(ENV_MCP_TOKEN))
        run(_make_runtime(args.runtime), host, port, token, path=str(cfg["path"]), stateless=bool(cfg["stateless"]),
            json_response=bool(cfg["json_response"]), verdict_wait=float(cfg["verdict_wait"]))
    except (ValueError, BridgeConfigError) as exc:
        print(f"mcp_server: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
