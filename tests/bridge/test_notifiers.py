"""Notifier tests. No network: Discord REST uses httpx.MockTransport, OpenClaw uses a fake `nemoclaw` runner."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from unspoken.bridge import BridgeConfigError, make_notifier
from unspoken.bridge.config import load_bridge_config
from unspoken.bridge.notifiers import (ConsoleNotifier, DiscordRestNotifier, OpenClawNotifier, RetryPolicy,
                                       chunk_text, find_id, parse_json_output, thread_name)
from unspoken.contracts import MessageRef, Notifier

CHANNELS = {"notes": "111111111111111111", "tasks": "222222222222222222", "recap": "333333333333333333"}
FAST = RetryPolicy(attempts=3, backoff=0.0, backoff_max=0.0, timeout=2.0)


async def no_sleep(_: float) -> None:
    return None


class Meter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def __call__(self, dest: str, kind: str, nbytes: int) -> None:
        self.calls.append((dest, kind, nbytes))

    @property
    def total(self) -> int:
        return sum(n for _, _, n in self.calls)


# --------------------------------------------------------------------------- helpers
def test_chunk_text_respects_limit_and_boundaries():
    assert chunk_text("", 100) == []
    assert chunk_text("short", 100) == ["short"]
    text = "\n".join(f"line {i} " + "x" * 40 for i in range(20))
    chunks = chunk_text(text, 120)
    assert all(len(c) <= 120 for c in chunks) and len(chunks) > 1
    assert "".join(c.replace("\n", "") for c in chunks).replace(" ", "") == text.replace("\n", "").replace(" ", "")
    long_word = "y" * 500
    assert all(len(c) <= 100 for c in chunk_text(long_word, 100))


def test_thread_name_is_short_and_plain():
    assert thread_name("**Sam**: send the deck by Friday\nsecond line") == "Sam: send the deck by Friday"
    assert len(thread_name("w" * 300)) <= 90 and thread_name("") == "task"


def test_find_id_and_json_output_parsing():
    assert find_id({"ok": True, "result": {"messageId": "123", "channelId": "9"}}) == "123"
    assert find_id({"ok": True, "result": {"id": 42}}) == "42"
    assert find_id({"ok": True}) == ""
    assert parse_json_output('plugin banner line\n{"ok": true, "messageId": "7"}')["messageId"] == "7"
    assert parse_json_output("") == {} and parse_json_output("garbage") == {}


# --------------------------------------------------------------------------- console
async def test_console_notifier_counts_every_byte_and_threads_tasks(capsys):
    meter = Meter()
    n = ConsoleNotifier(CHANNELS, meter, echo=True, retry=FAST)
    assert isinstance(n, Notifier)
    ref = await n.post("notes", "Priya: let's start")
    assert ref == MessageRef(channel="notes", id="console-1", thread=None)
    task = await n.post("tasks", "**Sam**: send the deck by Friday", kind="task")
    assert task.id == "console-2" and task.thread == "thread-3"
    dm = await n.dm("4242", "Recap for you")
    assert dm.channel == "dm:4242" and dm.id
    assert meter.calls == [("console", "text", len("Priya: let's start")),
                           ("console", "text", len("**Sam**: send the deck by Friday")),
                           ("console", "text", len("Recap for you"))]
    assert "[discord:111111111111111111] Priya" in capsys.readouterr().out
    # a long note is chunked and every chunk is counted
    long = "héllo wörld " * 400
    ref = await n.post("notes", long)
    sent = [s for s in n.sent if s.get("target") == CHANNELS["notes"] and "text" in s and s["text"] != "Priya: let's start"]
    assert len(sent) >= 3 and all(len(s["text"]) <= 1900 for s in sent)
    assert meter.total == sum(len(c.encode()) for c in [s["text"] for s in sent]) + len("Priya: let's start") \
        + len("**Sam**: send the deck by Friday") + len("Recap for you")


async def test_console_empty_text_posts_nothing():
    meter = Meter()
    n = ConsoleNotifier(CHANNELS, meter, echo=False)
    ref = await n.post("notes", "   \n ")
    assert ref.id == "" and meter.calls == [] and n.sent == []


# --------------------------------------------------------------------------- Discord REST
class FakeDiscord:
    """Minimal Discord REST: records requests, scripted failures."""

    def __init__(self, fail_first: int = 0, status: int = 500) -> None:
        self.requests: list[tuple[str, str, dict[str, Any] | None]] = []
        self.fail_first = fail_first
        self.status = status
        self.n = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        assert request.url.path.startswith("/api/v10/")
        path = request.url.path[len("/api/v10"):]
        self.requests.append((request.method, path, body))
        assert request.headers["Authorization"] == "Bot tok"
        assert request.headers["User-Agent"].startswith("DiscordBot (")
        if self.fail_first > 0:
            self.fail_first -= 1
            if self.status == 429:
                return httpx.Response(429, json={"retry_after": 0.01, "message": "slow down"})
            return httpx.Response(self.status, json={"message": "nope"})
        self.n += 1
        if path == "/users/@me":
            return httpx.Response(200, json={"id": "bot"})
        if path == "/users/@me/channels":
            return httpx.Response(200, json={"id": f"dm-{body['recipient_id']}"})
        if path.endswith("/threads"):
            return httpx.Response(201, json={"id": f"thread-{self.n}", "name": body["name"]})
        return httpx.Response(200, json={"id": f"msg-{self.n}", "channel_id": path.split("/")[2]})


def rest(fake: FakeDiscord, meter: Meter, **kw: Any) -> DiscordRestNotifier:
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    return DiscordRestNotifier("tok", CHANNELS, meter, client=client, retry=FAST, sleep=no_sleep, **kw)


async def test_discord_rest_posts_threads_and_dms():
    fake, meter = FakeDiscord(), Meter()
    n = rest(fake, meter)
    assert n.transport == "discord-rest" and await n.healthy()
    ref = await n.post("notes", "Priya: let's start")
    assert ref == MessageRef(channel="notes", id="msg-2", thread=None)
    method, path, body = fake.requests[-1]
    assert (method, path) == ("POST", f"/channels/{CHANNELS['notes']}/messages")
    assert body == {"content": "Priya: let's start", "allowed_mentions": {"parse": ["users"]}}
    task = await n.post("tasks", "Sam: send the deck by Friday\nDue Friday.", kind="task", meta={"thread_name": "deck"})
    assert task.id == "msg-3" and task.thread == "thread-4"
    assert fake.requests[-1] == ("POST", f"/channels/{CHANNELS['tasks']}/messages/msg-3/threads",
                                 {"name": "deck", "auto_archive_duration": 1440})
    reply = await n.post("tasks", "✅ to approve", thread=task.thread)
    assert reply.thread == "thread-4" and fake.requests[-1][1] == "/channels/thread-4/messages"
    dm = await n.dm("777", "Here is your recap")
    assert dm.id == "msg-7" and fake.requests[-2] == ("POST", "/users/@me/channels", {"recipient_id": "777"})
    assert [c[0] for c in meter.calls] == ["discord"] * 4
    assert meter.total == sum(len(t.encode()) for t in ["Priya: let's start", "Sam: send the deck by Friday\nDue Friday.",
                                                        "✅ to approve", "Here is your recap"])
    await n.aclose()


async def test_discord_rest_retries_5xx_and_429_then_gives_up_quietly():
    fake, meter = FakeDiscord(fail_first=2, status=500), Meter()
    n = rest(fake, meter)
    ref = await n.post("notes", "retry me")
    assert ref.id == "msg-1" and len(fake.requests) == 3 and meter.total == len("retry me")
    fake2, meter2 = FakeDiscord(fail_first=1, status=429), Meter()
    n2 = rest(fake2, meter2)
    assert (await n2.post("notes", "rate limited once")).id == "msg-1"
    fake3, meter3 = FakeDiscord(fail_first=99, status=503), Meter()
    n3 = rest(fake3, meter3)
    ref = await n3.post("notes", "never")
    assert ref == MessageRef(channel="notes", id="", thread=None) and len(fake3.requests) == 3
    assert meter3.calls == [] and n3.failures == 1       # nothing left the box, nothing counted


async def test_discord_rest_does_not_retry_4xx_and_never_raises():
    fake, meter = FakeDiscord(fail_first=5, status=403), Meter()
    n = rest(fake, meter)
    ref = await n.post("notes", "forbidden")
    assert ref.id == "" and len(fake.requests) == 1 and meter.calls == []
    # unknown logical channel and placeholder ids never raise either
    assert (await n.post("nope", "x")).id == ""
    n.channels["notes"] = "000000000000000000"
    assert (await n.post("notes", "x")).id == "" and len(fake.requests) == 1

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")
    n4 = DiscordRestNotifier("tok", CHANNELS, meter, client=httpx.AsyncClient(transport=httpx.MockTransport(boom)),
                             retry=FAST, sleep=no_sleep)
    assert (await n4.post("notes", "x")).id == "" and not await n4.healthy()


# --------------------------------------------------------------------------- OpenClaw (exec through nemoclaw)
class FakeNemoclaw:
    """Stands in for `nemoclaw <sb> exec -- openclaw message ...`. Records argv, returns the recorded JSON shapes."""

    def __init__(self, fail_first: int = 0, rc: int = 1, stderr: str = "gateway busy") -> None:
        self.calls: list[list[str]] = []
        self.fail_first, self.rc, self.stderr = fail_first, rc, stderr
        self.n = 0

    async def __call__(self, argv: list[str], timeout: float) -> tuple[int, str, str]:
        self.calls.append(argv)
        assert timeout == FAST.timeout
        if self.fail_first > 0:
            self.fail_first -= 1
            return self.rc, "", self.stderr
        self.n += 1
        sub = argv[argv.index("openclaw") + 1:]
        if sub[0] == "channels":
            return 0, json.dumps({"ok": True}), ""
        sub = sub[1:]                        # after "message"
        if sub[:2] == ["thread", "create"]:
            return 0, json.dumps({"ok": True, "result": {"threadId": f"thr-{self.n}", "name": sub[sub.index("--thread-name") + 1]}}), ""
        if sub[0] == "send":
            # the in-sandbox NemoClaw plugin prints a banner on stderr; stdout is the JSON result
            return 0, json.dumps({"ok": True, "result": {"messageId": f"m-{self.n}", "channelId": "x"}}), "[nemoclaw] plugin registered\n"
        return 0, "{}", ""


def openclaw(fake: FakeNemoclaw, meter: Meter, **kw: Any) -> OpenClawNotifier:
    return OpenClawNotifier(CHANNELS, meter, sandbox="unspoken", runner=fake, retry=FAST, sleep=no_sleep, **kw)


async def test_openclaw_exec_sends_verbatim_through_nemoclaw_exec():
    fake, meter = FakeNemoclaw(), Meter()
    n = openclaw(fake, meter)
    assert n.transport == "openclaw"
    text = "Priya: send the deck by Friday — $(rm -rf /) `x` \"quoted\" --json"
    ref = await n.post("notes", text)
    assert ref == MessageRef(channel="notes", id="m-1", thread=None)
    argv = fake.calls[0]
    assert argv[:6] == ["nemoclaw", "unspoken", "exec", "--no-stdin", "--", "openclaw"]
    assert argv[6:8] == ["message", "send"]
    assert argv[argv.index("--target") + 1] == f"channel:{CHANNELS['notes']}"
    assert f"--message={text}" in argv                       # one argv element, no shell, verbatim
    assert argv[-3:] == ["--channel", "discord", "--json"]
    assert meter.calls == [("openclaw", "text", len(text.encode()))]


async def test_openclaw_exec_threads_tasks_and_replies_into_threads():
    fake, meter = FakeNemoclaw(), Meter()
    n = openclaw(fake, meter, account="roombot")
    task = await n.post("tasks", "Sam: send the deck by Friday", kind="task")
    assert task.id == "m-1" and task.thread == "thr-2"
    create = fake.calls[1]
    assert create[6:9] == ["message", "thread", "create"]
    assert create[create.index("--target") + 1] == CHANNELS["tasks"]
    assert create[create.index("--message-id") + 1] == "m-1"
    assert create[create.index("--thread-name") + 1] == "Sam: send the deck by Friday"
    assert "--account" in create and create[create.index("--account") + 1] == "roombot"
    reply = await n.post("tasks", "React ✅ to confirm", thread=task.thread)
    assert reply.thread == "thr-2" and fake.calls[2][fake.calls[2].index("--target") + 1] == "channel:thr-2"
    dm = await n.dm("555", "your recap")
    assert dm.id == "m-4" and fake.calls[3][fake.calls[3].index("--target") + 1] == "user:555"
    assert meter.total == sum(len(t.encode()) for t in ["Sam: send the deck by Friday", "React ✅ to confirm", "your recap"])
    assert await n.healthy() and fake.calls[-1][5:8] == ["openclaw", "channels", "status"]


async def test_openclaw_exec_retries_transient_failures_and_gives_up_quietly():
    fake, meter = FakeNemoclaw(fail_first=2), Meter()
    n = openclaw(fake, meter)
    assert (await n.post("notes", "flaky")).id == "m-1" and len(fake.calls) == 3
    dead = FakeNemoclaw(fail_first=99)
    n2 = openclaw(dead, Meter())
    assert (await n2.post("notes", "down")).id == "" and len(dead.calls) == 3 and n2.failures == 1
    # a config problem is not retried
    cfg = FakeNemoclaw(fail_first=99, stderr="error: channel discord is not configured")
    n3 = openclaw(cfg, Meter())
    assert (await n3.post("notes", "x")).id == "" and len(cfg.calls) == 1

    async def hang(argv: list[str], timeout: float) -> tuple[int, str, str]:
        raise TimeoutError("nemoclaw timed out")
    n4 = OpenClawNotifier(CHANNELS, Meter(), runner=hang, retry=FAST, sleep=no_sleep)
    assert (await n4.post("notes", "x")).id == "" and not await n4.healthy()


async def test_openclaw_hooks_mode_posts_an_agent_turn_with_direct_delivery():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(503, json={"error": "restarting"})
        return httpx.Response(200, json={"ok": True, "runId": "run-1"})
    meter = Meter()
    n = OpenClawNotifier(CHANNELS, meter, mode="hooks", hooks_token="hook-secret-value-123456",
                         client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), retry=FAST, sleep=no_sleep)
    ref = await n.post("tasks", "Sam: send the deck", kind="task")
    assert ref == MessageRef(channel="tasks", id="run-1", thread=None)   # no thread ids come back in hooks mode
    req = seen[-1]
    assert str(req.url) == "http://127.0.0.1:18789/hooks/agent"
    assert req.headers["Authorization"] == "Bearer hook-secret-value-123456" and req.headers["Idempotency-Key"]
    body = json.loads(req.content)
    assert body["channel"] == "discord" and body["to"] == f"channel:{CHANNELS['tasks']}" and body["deliver"] is True
    assert body["agentId"] == "main" and body["sessionMode"] == "isolated" and body["message"].endswith("Sam: send the deck")
    assert meter.calls == [("openclaw", "text", len("Sam: send the deck"))]
    with pytest.raises(BridgeConfigError):
        OpenClawNotifier(CHANNELS, meter, mode="hooks", hooks_token=None)
    with pytest.raises(BridgeConfigError):
        OpenClawNotifier(CHANNELS, meter, mode="carrier-pigeon")
    with pytest.raises(BridgeConfigError):
        OpenClawNotifier(CHANNELS, meter, sandbox="bad name; rm -rf")


# --------------------------------------------------------------------------- factory + config file
def test_bridge_toml_parses_and_make_notifier_picks_transport(monkeypatch):
    cfg = load_bridge_config()
    assert cfg["notifier"]["transport"] == "console"
    assert set(cfg["notifier"]["channels"]) == {"notes", "tasks", "recap"}
    assert cfg["mcp"]["port"] == 8791 and cfg["tls_proxy"]["port"] == 8443 and cfg["tls_proxy"]["bind_ip"] == ""
    for key in ("DISCORD_BOT_TOKEN", "OPENCLAW_HOOKS_TOKEN", "UNSPOKEN_NOTIFIER"):
        monkeypatch.delenv(key, raising=False)
    meter = Meter()
    assert isinstance(make_notifier(cfg, meter), ConsoleNotifier)
    monkeypatch.setenv("UNSPOKEN_NOTIFIER", "discord-rest")
    with pytest.raises(BridgeConfigError):
        make_notifier(cfg, meter)                       # token comes only from the environment
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
    n = make_notifier(cfg, meter)
    assert isinstance(n, DiscordRestNotifier) and n.channels == cfg["notifier"]["channels"]
    assert n.retry == RetryPolicy(attempts=3, backoff=0.5, backoff_max=4.0, timeout=15.0)
    monkeypatch.setenv("UNSPOKEN_NOTIFIER", "openclaw")
    oc = make_notifier(cfg, meter)
    assert isinstance(oc, OpenClawNotifier) and oc.mode == "exec" and oc.sandbox == "unspoken"
    monkeypatch.setenv("UNSPOKEN_NOTIFIER", "fax")
    with pytest.raises(BridgeConfigError):
        make_notifier(cfg, meter)
    # hooks mode needs its token
    hooks_cfg = {"notifier": {**cfg["notifier"], "transport": "openclaw",
                              "openclaw": {**cfg["notifier"]["openclaw"], "mode": "hooks"}}}
    monkeypatch.delenv("UNSPOKEN_NOTIFIER")
    with pytest.raises(BridgeConfigError):
        make_notifier(hooks_cfg, meter)
    monkeypatch.setenv("OPENCLAW_HOOKS_TOKEN", "hook-secret-value-123456")
    assert make_notifier(hooks_cfg, meter).mode == "hooks"


def test_no_secret_values_in_bridge_config_or_repo_files():
    text = (load_bridge_config.__globals__["DEFAULT_PATH"]).read_text(encoding="utf-8")
    for needle in ("DISCORD_BOT_TOKEN", "OPENCLAW_HOOKS_TOKEN", "UNSPOKEN_MCP_TOKEN"):
        assert needle in text                                     # documented by name
        assert f'{needle} = "' not in text and f"{needle}=" not in text  # never with a value
