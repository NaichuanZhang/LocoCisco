"""The bridge wired to the meeting runtime: the real MCP server (bearer-gated Streamable HTTP over loopback) drives
MeetingRuntime with the `mcp` client, and the bridge's console Notifier carries the room's posts with the egress
meter. No model servers: profile fake (keyword decider)."""
from __future__ import annotations

import asyncio
import socket
from typing import Any, AsyncIterator

import pytest

pytest.importorskip("mcp")
from mcp.client.session import ClientSession  # noqa: E402
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client  # noqa: E402

from unspoken.bridge import make_notifier  # noqa: E402
from unspoken.bridge.mcp_server import create_app  # noqa: E402
from unspoken.meeting.contracts import MT  # noqa: E402
from unspoken.meeting.runtime import MeetingRuntime  # noqa: E402

from .conftest import JOIN_PRIYA, JOIN_SAM, START, make_registry  # noqa: E402

TOKEN = "meeting-test-token-0123456789"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Client:
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


@pytest.fixture
async def rt() -> AsyncIterator[MeetingRuntime]:
    rt = MeetingRuntime(profile="fake", mode="sim", registry=make_registry(), echo=False, tick_seconds=0)
    await rt.start(ticking=False)
    for step in (START, JOIN_PRIYA, JOIN_SAM):
        await rt.world.step(step)
    await rt.bus.drain()
    yield rt
    await rt.close()


@pytest.fixture
async def server(rt: MeetingRuntime) -> AsyncIterator[str]:
    import uvicorn

    port = free_port()
    app = create_app(rt, TOKEN, verdict_wait=3.0)
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
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


async def test_mcp_tools_drive_the_meeting_runtime(rt: MeetingRuntime, server: str):
    async with Client(server) as c:
        tools = await c.tools()
        assert {"room_status", "create_task", "set_light", "checkin", "pause_notes", "resume_notes", "undo_item",
                "decide_request", "recap", "transcript", "search_docs", "list_items"} <= set(tools)
        status = await c.call("room_status")
        assert status["title"] == "Pricing sync" if "title" in status else True
        # a user proposal: the policy answers with a verdict for THIS proposal id
        out = await c.call("create_task", text="Send the deck to the press team", owner="Sam", due="Friday", user="Priya")
        assert out["ok"] and out["verdict"] == "act" and out["proposal_id"]
        verdicts = [e.data for e in rt.bus.history if e.topic == MT.VERDICT and e.data.get("proposal_id") == out["proposal_id"]]
        assert verdicts and verdicts[0]["verdict"] == "act" and verdicts[0]["by"] == "user"
        items = rt.items(kind="task")
        assert items and items[-1]["owner"] == "Sam" and items[-1]["text"].startswith("Send the deck")
        listed = await c.call("list_items", kind="task")
        assert any("deck" in str(i) for i in (listed.get("items") if isinstance(listed, dict) else listed))
        # the lamp through a user proposal
        out = await c.call("set_light", scene="slides", user="Priya")
        assert out["verdict"] == "act" and rt.state.devices["lamp"]["state"]["scene"] == "slides"
        # check-in, off the record, resume, undo and recap through chat
        assert (await c.call("checkin", name="Lin", user="lin#1"))["ok"]
        assert "Lin" in rt.state.names()
        await c.call("pause_notes", reason="salary talk", user="Priya")
        assert rt.state.privacy["state"] == "off_record"
        out = await c.call("set_light", scene="bright", user="Sam")
        assert out["verdict"] == "deny"                                     # off the record: only pause/resume
        await c.call("resume_notes", user="Priya")
        assert rt.state.privacy["state"] == "on"
        item_id = items[-1]["item_id"]
        assert (await c.call("undo_item", item_id=item_id, user="Sam"))["ok"]
        assert rt.state.items[item_id]["status"] == "undone"
        recap = await c.call("recap")
        assert "Recap: Pricing sync" in recap
        hits = await c.call("search_docs", query="refund policy annual plans", k=1)
        assert "refund-policy.md" in str(hits)
    assert not rt.bus.errors


async def test_bridge_console_notifier_carries_posts_and_meters_egress(capsys):
    counted: list[tuple[str, str, int]] = []
    rt = MeetingRuntime(profile="fake", mode="sim", registry=make_registry(), echo=False, tick_seconds=0,
                        notifier=make_notifier({"notifier": {"transport": "console"}},
                                               on_egress=lambda d, k, n: counted.append((d, k, n))))
    rt.notifier.on_egress = None  # type: ignore[attr-defined]
    await rt.start(ticking=False)
    for step in (START, JOIN_PRIYA, JOIN_SAM, {"do": "say", "who": "priya", "text": "Sam, can you send the deck by Friday?"}):
        await rt.world.step(step)
    await rt.bus.drain()
    item = rt.items(kind="task")[-1]
    out = capsys.readouterr().out
    assert item["message"] is not None and f"(task {item['item_id']})" in out and "Sam" in out
    assert counted and all(k == "text" for _, k, _n in counted) and sum(n for *_, n in counted) > 0
    await rt.close()
