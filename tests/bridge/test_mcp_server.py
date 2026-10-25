"""MCP server tests: the real `mcp` Streamable HTTP client talks to our server over a loopback socket.
No model servers: the runtime is tests/bridge/fake_runtime.FakeMeetingRuntime."""
from __future__ import annotations

import asyncio
import json
import socket
from typing import Any, AsyncIterator

import httpx
import httpx2
import pytest
import uvicorn
from mcp.client.session import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from unspoken.bridge import mcp_server
from unspoken.bridge.config import BridgeConfigError
from unspoken.bridge.mcp_server import ITEM_KINDS, LIGHT_SCENES, OFF_RECORD, create_app

from unspoken.bridge.dev_fake import FakeMeetingRuntime

TOKEN = "test-token-0123456789abcdef"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def runtime() -> FakeMeetingRuntime:
    return FakeMeetingRuntime()


@pytest.fixture
async def server(runtime: FakeMeetingRuntime) -> AsyncIterator[str]:
    """Runs the bearer-gated MCP app on 127.0.0.1:<free port>; yields the base URL."""
    port = free_port()
    app = create_app(runtime, TOKEN, verdict_wait=2.0, clock=lambda: runtime.clock)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    srv = uvicorn.Server(config)
    task = asyncio.create_task(srv.serve())
    for _ in range(300):
        if srv.started:
            break
        await asyncio.sleep(0.01)
    assert srv.started
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.should_exit = True
        await asyncio.wait_for(task, 10)


class Client:
    """One MCP session with the bearer token."""

    def __init__(self, base: str, token: str = TOKEN) -> None:
        self.url = f"{base}/mcp"
        self.token = token

    async def __aenter__(self) -> "Client":
        self._http = create_mcp_http_client(headers={"Authorization": f"Bearer {self.token}"})
        await self._http.__aenter__()
        self._transport = streamable_http_client(self.url, http_client=self._http)
        read, write = await self._transport.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._session.__aexit__(*exc)
        await self._transport.__aexit__(*exc)
        await self._http.__aexit__(*exc)

    async def tools(self) -> dict[str, Any]:
        return {t.name: t for t in (await self._session.list_tools()).tools}

    async def call(self, tool: str, **args: Any) -> Any:
        result = await self._session.call_tool(tool, args)
        assert not result.is_error, result.content
        if result.structured_content is not None:
            sc = result.structured_content
            return sc["result"] if set(sc) == {"result"} else sc
        return "".join(getattr(c, "text", "") for c in result.content)


# --------------------------------------------------------------------------- auth + health
async def test_health_is_open_and_everything_else_needs_the_bearer(server: str):
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{server}/health")
        assert r.status_code == 200 and r.json() == {"ok": True, "server": "unspoken-room"}
        r = await c.post(f"{server}/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        assert r.status_code == 401 and r.headers["www-authenticate"].startswith("Bearer")
        r = await c.post(f"{server}/mcp", json={}, headers={"Authorization": "Bearer wrong-token-xxxxxxxxxxxx"})
        assert r.status_code == 401
        r = await c.post(f"{server}/mcp", json={}, headers={"Authorization": f"Basic {TOKEN}"})
        assert r.status_code == 401
        r = await c.get(f"{server}/health?x=1", headers={"Authorization": "Bearer nope"})
        assert r.status_code == 200          # health never leaks room data
        assert "status" not in r.json()


async def test_wrong_token_cannot_initialize(server: str):
    with pytest.raises(Exception):
        async with Client(server, token="not-the-token-aaaaaaaaaaaa") as c:
            await c.tools()


def test_token_and_bind_guards(runtime: FakeMeetingRuntime):
    for bad in ("", "short", None):
        with pytest.raises(BridgeConfigError):
            create_app(runtime, bad)  # type: ignore[arg-type]
    for host in ("0.0.0.0", "172.18.0.1", "10.0.0.5", "::", "example.com", ""):
        with pytest.raises(ValueError):
            mcp_server.require_loopback(host)
    for host in ("127.0.0.1", "localhost", "::1", "[::1]"):
        mcp_server.require_loopback(host)


async def test_serve_refuses_non_loopback(runtime: FakeMeetingRuntime):
    with pytest.raises(ValueError):
        await mcp_server.serve(runtime, host="0.0.0.0", port=free_port(), token=TOKEN)


# --------------------------------------------------------------------------- tools
async def test_tool_catalogue_and_descriptions(server: str):
    async with Client(server) as c:
        tools = await c.tools()
    expected = {"room_status", "transcript", "list_items", "search_docs", "recap", "checkin", "pause_notes",
                "resume_notes", "create_task", "set_light", "undo_item", "decide_request"}
    assert set(tools) == expected
    for name in ("room_status", "transcript", "list_items", "search_docs", "recap"):
        assert OFF_RECORD in tools[name].description, name
    assert "proposal" in tools["create_task"].description.lower() and "policy" in tools["set_light"].description.lower()
    assert tools["transcript"].input_schema["properties"]["since_minutes"]["default"] == 10.0
    assert tools["create_task"].input_schema["required"] == ["text"]


async def test_read_tools(server: str, runtime: FakeMeetingRuntime):
    async with Client(server) as c:
        status = await c.call("room_status")
        assert status["status"]["title"] == "Pricing sync" and status["privacy"]["state"] == "on"
        assert status["open_items"] == {"task": 1, "decision": 1, "question": 0, "parking_lot": 0}
        assert status["banner"].startswith("2 min left") and status["models"] == ["Strands Decider 2B (fake)"]
        assert status["egress"]["audio_bytes_out"] == 0

        recent = await c.call("transcript", since_minutes=2)
        assert [l["name"] for l in recent["lines"]] == ["voice-B"] and recent["lines"][0]["t"] == 580.0
        everything = await c.call("transcript", since_minutes=1000)
        assert everything["since_minutes"] == 120.0 and [l["name"] for l in everything["lines"]] == ["Priya", "Priya", "voice-B"]

        items = await c.call("list_items", kind="task")
        assert items["count"] == 1 and items["items"][0]["owner"] == "Sam"
        assert (await c.call("list_items"))["count"] == 2
        bad = await c.call("list_items", kind="emails")
        assert bad["ok"] is False and all(k in bad["text"] for k in ITEM_KINDS)

        docs = await c.call("search_docs", query="pricing tiers", k=50)
        assert docs["hits"][0]["doc"] == "pricing.md" and docs["query"] == "pricing tiers"
        assert (await c.call("search_docs", query="  "))["ok"] is False

        recap = await c.call("recap")
        assert "**Decisions**" in recap and "Sam: send the deck" in recap


async def test_off_record_lines_do_not_exist(server: str, runtime: FakeMeetingRuntime):
    async with Client(server) as c:
        paused = await c.call("pause_notes", reason="Sam asked in Discord", user="Sam")
        assert paused["ok"] is True and "paused" in paused["text"].lower()
        assert runtime.chats[-1]["command"] == "offrecord" and runtime.chats[-1]["args"] == {"reason": "Sam asked in Discord"}
        assert runtime.chats[-1]["user"] == "Sam"
        assert (await c.call("room_status"))["privacy"]["state"] == "off_record"
        assert (await c.call("transcript", since_minutes=120))["lines"] == []
        resumed = await c.call("resume_notes", user="Priya")
        assert resumed["ok"] is True and runtime.chats[-1]["command"] == "resume"
        assert (await c.call("room_status"))["privacy"]["state"] == "on"


async def test_chat_tools_map_to_discord_commands(server: str, runtime: FakeMeetingRuntime):
    async with Client(server) as c:
        r = await c.call("checkin", name="Priya", user="priya_d")
        assert r == {"ok": True, "text": "Welcome, Priya."}
        assert runtime.chats[-1] == {"command": "here", "user": "priya_d", "args": {"name": "Priya"}, "text": ""}
        assert (await c.call("checkin", name="   "))["ok"] is False
        r = await c.call("undo_item", item_id="t1", user="Sam")
        assert r["ok"] is True and runtime.items("task")[0]["status"] == "undone"
        assert (await c.call("undo_item", item_id="zzz"))["ok"] is False
        r = await c.call("decide_request", request_id="req-9", approve=True, user="Priya")
        assert r["text"] == "approve: req-9" and runtime.chats[-1]["command"] == "approve"
        r = await c.call("decide_request", request_id="req-9", approve=False)
        assert runtime.chats[-1]["command"] == "deny" and runtime.chats[-1]["args"] == {"request_id": "req-9"}


async def test_runtime_errors_become_readable_tool_results(server: str, runtime: FakeMeetingRuntime):
    async def broken(command):  # noqa: ANN001
        raise RuntimeError("kaboom")
    runtime.chat = broken  # type: ignore[method-assign]
    async with Client(server) as c:
        r = await c.call("resume_notes")
        assert r == {"ok": False, "text": "RuntimeError: kaboom"}


async def test_create_task_and_set_light_are_user_proposals_the_policy_decides(server: str, runtime: FakeMeetingRuntime):
    runtime.policy["create_task"] = "ask_owner"
    async with Client(server) as c:
        r = await c.call("create_task", text="Send the deck", owner="Sam", due="Friday", user="Priya")
        assert r["ok"] is True and r["verdict"] == "ask_owner" and r["risk"] == "medium"
        assert r["args"] == {"owner": "Sam", "text": "Send the deck", "due": "Friday"}
        assert "owner" in r["text"].lower()
        proposals = runtime.bus.find("meeting.proposal", action="create_task")
        assert len(proposals) == 1
        p = proposals[0]
        assert p.id == r["proposal_id"] and p.source == "bridge.mcp:Priya" and p.ts == runtime.clock
        assert p.data["by"] == "user" and p.data["route"] == "user" and p.data["mode"] == "act"
        assert p.data["kind"] == "task" and p.data["confidence"] == 1.0 and p.data["tokens"] is None
        assert p.data["reason"] == "Priya asked in Discord"

        r = await c.call("set_light", scene="slides")
        assert r["verdict"] == "act" and r["text"] == "Done." and r["risk"] == "low"
        assert runtime.bus.find("meeting.proposal", action="set_light")[0].data["args"] == {"scene": "slides"}
        bad = await c.call("set_light", scene="disco")
        assert bad["ok"] is False and all(s in bad["text"] for s in LIGHT_SCENES)
        assert (await c.call("create_task", text=""))["ok"] is False
        assert not runtime.bus.errors


async def test_verdict_can_be_denied_or_pending(server: str, runtime: FakeMeetingRuntime):
    runtime.policy["set_light"] = "deny"
    runtime.policy["create_task"] = "skip"          # no policy answer at all
    async with Client(server) as c:
        r = await c.call("set_light", scene="dim")
        assert r["verdict"] == "deny" and "declined" in r["text"]
        r = await c.call("create_task", text="Nobody answers")
        assert r["verdict"] == "pending" and r["proposal_id"] and "not answered" in r["text"]


async def test_slow_policy_verdict_is_still_picked_up(runtime: FakeMeetingRuntime):
    """VerdictWatcher waits for the verdict event instead of polling (verdict arrives after 0.3 s)."""
    runtime.verdict_delay = 0.3
    app = create_app(runtime, TOKEN, verdict_wait=2.0, clock=lambda: runtime.clock)
    watcher = mcp_server.VerdictWatcher(runtime)
    from unspoken.core.events import Event
    from unspoken.meeting.contracts import MT
    ev = Event(MT.PROPOSAL, {"action": "set_light", "args": {"scene": "dim"}, "by": "user"}, source="t")
    await runtime.bus.publish(ev)
    verdict = await watcher.wait(ev.id, 2.0)
    assert verdict is not None and verdict["verdict"] == "act"
    assert await watcher.wait("never", 0.05) is None
    assert app.server is not None  # type: ignore[attr-defined]


async def test_plain_http_json_rpc_round_trip_with_bearer(server: str):
    """What the TLS proxy forwards: a bare Streamable HTTP POST with the bearer gets a JSON answer."""
    async with httpx2.AsyncClient() as c:
        r = await c.post(f"{server}/mcp", headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/json, text/event-stream",
                                                   "Content-Type": "application/json"},
                         json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                               "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                          "clientInfo": {"name": "probe", "version": "0"}}})
        assert r.status_code == 200, r.text
        body = r.json() if r.headers["content-type"].startswith("application/json") else json.loads(r.text.split("data: ", 1)[1].split("\n")[0])
        assert body["result"]["serverInfo"]["name"] == "unspoken-room"
        assert "Off-record content never exists" in body["result"]["instructions"]
