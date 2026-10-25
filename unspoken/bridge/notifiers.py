"""Notifiers: the only way text leaves the box.

OpenClawNotifier   primary. The OpenClaw bot inside the NemoClaw sandbox posts to Discord. Default mode
                   "exec" runs `openclaw message send` inside the sandbox through `nemoclaw <sb> exec`, so the
                   text is delivered VERBATIM and Discord message ids come back (threads for tasks). Mode
                   "hooks" POSTs /hooks/agent on the NemoClaw-forwarded OpenClaw gateway (127.0.0.1:18789):
                   the agent relays the text, so it is not guaranteed verbatim; use it only when exec is
                   unavailable. Research and day-of checks: docs/NEMOCLAW.md.
DiscordRestNotifier fallback. The host posts straight to the Discord REST API with DISCORD_BOT_TOKEN.
ConsoleNotifier    dev. Prints; nothing leaves the box, the egress meter still counts.

Every post calls on_egress(dest, "text", nbytes) once per message actually sent. Posting never raises
into the pipeline: failures are logged and an empty MessageRef(id="") comes back. Retries with backoff and
per-request timeouts are in RetryPolicy.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from ..contracts import MessageRef, Notifier, Transport
from .config import ENV_DISCORD_TOKEN, ENV_HOOKS_TOKEN, BridgeConfigError, secret

log = logging.getLogger("unspoken.bridge.notifier")

Egress = Callable[[str, str, int], None]
# argv -> (returncode, stdout, stderr); injectable so tests never spawn `nemoclaw`
Runner = Callable[[list[str], float], Awaitable[tuple[int, str, str]]]

DISCORD_LIMIT = 2000
_SNOWFLAKE = re.compile(r"^\d{15,22}$")


def text_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def chunk_text(text: str, limit: int) -> list[str]:
    """Split on paragraph, then line, then space boundaries so each piece fits a Discord message."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    out: list[str] = []
    buf = ""
    for para in text.split("\n"):
        candidate = para if not buf else f"{buf}\n{para}"
        if len(candidate) <= limit:
            buf = candidate
            continue
        if buf:
            out.append(buf)
            buf = ""
        while len(para) > limit:
            cut = para.rfind(" ", 0, limit)
            if cut < limit // 2:
                cut = limit
            out.append(para[:cut].rstrip())
            para = para[cut:].lstrip()
        buf = para
    if buf:
        out.append(buf)
    return out


def thread_name(text: str, limit: int = 90) -> str:
    first = text.strip().splitlines()[0] if text.strip() else "task"
    first = re.sub(r"[*_`#>]+", "", first)
    first = re.sub(r"\s+", " ", first).strip(" -:")
    return (first[: limit - 1] + "…") if len(first) > limit else (first or "task")


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    backoff: float = 0.5
    backoff_max: float = 4.0
    timeout: float = 15.0

    def delay(self, attempt: int) -> float:
        return min(self.backoff_max, self.backoff * (2 ** attempt))


class _Base:
    """Shared plumbing: channel lookup, chunking, egress accounting, retry loop, never-raise post()."""

    transport: Transport
    dest: str
    supports_threads = True

    def __init__(self, channels: dict[str, str], on_egress: Egress | None, *,
                 thread_kinds: tuple[str, ...] | list[str] = ("task",), chunk_chars: int = 1900,
                 retry: RetryPolicy = RetryPolicy(), sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 ) -> None:
        self.channels = {k: str(v) for k, v in channels.items()}
        self._on_egress = on_egress
        self.thread_kinds = tuple(thread_kinds)
        self.chunk_chars = max(100, min(int(chunk_chars), DISCORD_LIMIT))
        self.retry = retry
        self._sleep = sleep
        self.failures = 0

    # ----------------------------------------------------------------- helpers for subclasses
    def resolve(self, channel: str) -> str | None:
        """Logical key ("notes") -> Discord channel id; a raw snowflake passes through."""
        cid = self.channels.get(channel)
        if cid and _SNOWFLAKE.match(cid) and set(cid) != {"0"}:
            return cid
        if _SNOWFLAKE.match(channel) and set(channel) != {"0"}:
            return channel
        return None

    def _count(self, text: str) -> None:
        if self._on_egress is not None:
            self._on_egress(self.dest, "text", text_bytes(text))

    async def _retrying(self, what: str, fn: Callable[[], Awaitable[Any]]) -> Any | None:
        """Run fn with retries; returns None after the last failure (never raises)."""
        for attempt in range(self.retry.attempts):
            try:
                return await asyncio.wait_for(fn(), timeout=self.retry.timeout)
            except _NoRetry as exc:
                log.warning("%s: %s failed, not retrying: %s", self.transport, what, exc)
                break
            except Exception as exc:  # network, timeout, parse: retry
                retry_after = getattr(exc, "retry_after", None)
                wait = retry_after if retry_after is not None else self.retry.delay(attempt)
                if attempt + 1 < self.retry.attempts:
                    log.warning("%s: %s failed (%s: %s); retry in %.1fs", self.transport, what,
                                type(exc).__name__, exc, wait)
                    await self._sleep(wait)
                else:
                    log.error("%s: %s failed after %d attempts: %s", self.transport, what, attempt + 1, exc)
        self.failures += 1
        return None

    # ----------------------------------------------------------------- transport primitives
    async def _send(self, target: str, text: str) -> str:
        """Send one message to a Discord channel/thread id; returns the message id ("" if unknown)."""
        raise NotImplementedError

    async def _create_thread(self, channel_id: str, message_id: str, name: str) -> str | None:
        """Start a thread under message_id; returns the thread id or None when unsupported."""
        return None

    async def _dm_target(self, user: str) -> str:
        """Turn a Discord user id into a sendable target."""
        raise NotImplementedError

    # ----------------------------------------------------------------- Notifier protocol
    async def post(self, channel: str, text: str, *, thread: str | None = None, kind: str = "note",
                   meta: dict[str, Any] | None = None) -> MessageRef:
        try:
            return await self._post(channel, text, thread=thread, kind=kind, meta=meta or {})
        except Exception as exc:  # belt and braces: a notifier bug must not stop the meeting
            log.exception("%s: post to %s failed: %s", self.transport, channel, exc)
            self.failures += 1
            return MessageRef(channel=channel, id="", thread=thread)

    async def _post(self, channel: str, text: str, *, thread: str | None, kind: str,
                    meta: dict[str, Any]) -> MessageRef:
        chunks = chunk_text(text, self.chunk_chars)
        if not chunks:
            return MessageRef(channel=channel, id="", thread=thread)
        channel_id = self.resolve(channel)
        if channel_id is None:
            log.warning("%s: no Discord channel id for %r (fill [notifier.channels] in config/bridge.toml)",
                        self.transport, channel)
            self.failures += 1
            return MessageRef(channel=channel, id="", thread=thread)

        first_id = ""
        if thread:
            for chunk in chunks:
                mid = await self._retrying(f"post to thread {thread}", lambda c=chunk: self._send(thread, c))
                if mid is None:
                    return MessageRef(channel=channel, id=first_id, thread=thread)
                self._count(chunk)
                first_id = first_id or mid
            return MessageRef(channel=channel, id=first_id, thread=thread)

        mid = await self._retrying(f"post to {channel}", lambda: self._send(channel_id, chunks[0]))
        if mid is None:
            return MessageRef(channel=channel, id="", thread=None)
        self._count(chunks[0])
        first_id = mid
        thread_id: str | None = None
        if kind in self.thread_kinds and self.supports_threads and mid:
            name = str(meta.get("thread_name") or thread_name(chunks[0]))
            thread_id = await self._retrying(f"thread for {channel}", lambda: self._create_thread(channel_id, mid, name))
        target = thread_id or channel_id
        for chunk in chunks[1:]:
            extra = await self._retrying(f"post to {channel}", lambda c=chunk: self._send(target, c))
            if extra is None:
                break
            self._count(chunk)
        return MessageRef(channel=channel, id=first_id, thread=thread_id)

    async def dm(self, user: str, text: str, *, meta: dict[str, Any] | None = None) -> MessageRef:
        try:
            chunks = chunk_text(text, self.chunk_chars)
            if not chunks:
                return MessageRef(channel=f"dm:{user}", id="")
            target = await self._retrying(f"dm channel for {user}", lambda: self._dm_target(user))
            if target is None:
                return MessageRef(channel=f"dm:{user}", id="")
            first_id = ""
            for chunk in chunks:
                mid = await self._retrying(f"dm to {user}", lambda c=chunk: self._send(target, c))
                if mid is None:
                    break
                self._count(chunk)
                first_id = first_id or mid
            return MessageRef(channel=f"dm:{user}", id=first_id)
        except Exception as exc:
            log.exception("%s: dm to %s failed: %s", self.transport, user, exc)
            self.failures += 1
            return MessageRef(channel=f"dm:{user}", id="")

    async def healthy(self) -> bool:
        return True


class _NoRetry(Exception):
    """A definite failure (4xx other than 429, bad config): retrying would not help."""


class _RateLimited(Exception):
    def __init__(self, retry_after: float) -> None:
        super().__init__(f"rate limited, retry after {retry_after:.1f}s")
        self.retry_after = retry_after


# =============================================================================== console (dev)
class ConsoleNotifier(_Base):
    transport: Transport = "console"
    dest = "console"

    def __init__(self, channels: dict[str, str] | None = None, on_egress: Egress | None = None,
                 *, echo: bool = True, **kw: Any) -> None:
        super().__init__(channels or {"notes": "100000000000000001", "tasks": "100000000000000002",
                                      "recap": "100000000000000003"}, on_egress, **kw)
        self.echo = echo
        self.sent: list[dict[str, Any]] = []
        self._n = 0

    def resolve(self, channel: str) -> str | None:  # any key works on the console
        return super().resolve(channel) or channel

    async def _send(self, target: str, text: str) -> str:
        self._n += 1
        mid = f"console-{self._n}"
        self.sent.append({"target": target, "text": text, "id": mid})
        if self.echo:
            print(f"[discord:{target}] {text}", flush=True)
        return mid

    async def _create_thread(self, channel_id: str, message_id: str, name: str) -> str | None:
        self._n += 1
        tid = f"thread-{self._n}"
        self.sent.append({"target": channel_id, "thread": tid, "name": name, "under": message_id})
        if self.echo:
            print(f"[discord:{channel_id}] (thread {tid!r} under {message_id}: {name})", flush=True)
        return tid

    async def _dm_target(self, user: str) -> str:
        return f"dm:{user}"


# =============================================================================== Discord REST (fallback)
class DiscordRestNotifier(_Base):
    """Posts with the bot token from the host. Only if NemoClaw is down on the day: the OpenClaw bot and this
    notifier share one bot identity, so do not run both at once."""

    transport: Transport = "discord-rest"
    dest = "discord"

    def __init__(self, token: str, channels: dict[str, str], on_egress: Egress | None = None, *,
                 api_base: str = "https://discord.com/api/v10", auto_archive_minutes: int = 1440,
                 client: httpx.AsyncClient | None = None, **kw: Any) -> None:
        super().__init__(channels, on_egress, **kw)
        if not token:
            raise BridgeConfigError(f"{ENV_DISCORD_TOKEN} is empty")
        self.api_base = api_base.rstrip("/")
        self.auto_archive_minutes = int(auto_archive_minutes)
        self._client = client or httpx.AsyncClient(timeout=self.retry.timeout)
        self._client.headers.update({
            "Authorization": f"Bot {token}",
            "User-Agent": "DiscordBot (https://example.invalid/unspoken, 0.1)",
            "Content-Type": "application/json",
        })

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        r = await self._client.request(method, f"{self.api_base}{path}", json=body)
        if r.status_code == 429:
            try:
                wait = float(r.json().get("retry_after", r.headers.get("Retry-After", 1.0)))
            except (ValueError, json.JSONDecodeError):
                wait = 1.0
            raise _RateLimited(min(wait, self.retry.backoff_max * 4))
        if 400 <= r.status_code < 500:
            raise _NoRetry(f"HTTP {r.status_code} {r.text[:200]}")
        if r.status_code >= 500:
            raise RuntimeError(f"HTTP {r.status_code}")
        if not r.content:
            return {}
        data = r.json()
        return data if isinstance(data, dict) else {"data": data}

    async def _send(self, target: str, text: str) -> str:
        data = await self._call("POST", f"/channels/{target}/messages",
                                {"content": text, "allowed_mentions": {"parse": ["users"]}})
        return str(data.get("id", ""))

    async def _create_thread(self, channel_id: str, message_id: str, name: str) -> str | None:
        data = await self._call("POST", f"/channels/{channel_id}/messages/{message_id}/threads",
                                {"name": name[:100], "auto_archive_duration": self.auto_archive_minutes})
        return str(data.get("id")) if data.get("id") else None

    async def _dm_target(self, user: str) -> str:
        data = await self._call("POST", "/users/@me/channels", {"recipient_id": user})
        cid = data.get("id")
        if not cid:
            raise _NoRetry("no DM channel id in response")
        return str(cid)

    async def healthy(self) -> bool:
        try:
            r = await self._client.get(f"{self.api_base}/users/@me", timeout=self.retry.timeout)
            return r.status_code == 200
        except Exception:
            return False


# =============================================================================== OpenClaw (primary)
async def subprocess_runner(argv: list[str], timeout: float) -> tuple[int, str, str]:
    """Run argv with no shell (the message text is one argv element: nothing is interpreted)."""
    proc = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"{argv[0]} timed out after {timeout:.0f}s")
    return proc.returncode or 0, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


_ID_KEYS = ("messageId", "message_id", "threadId", "thread_id", "id")


def find_id(payload: Any, keys: tuple[str, ...] = _ID_KEYS) -> str:
    """Pull a Discord id out of `openclaw ... --json` output without depending on its exact shape."""
    if isinstance(payload, dict):
        for k in keys:
            v = payload.get(k)
            if isinstance(v, (str, int)) and str(v):
                return str(v)
        for v in payload.values():
            found = find_id(v, keys)
            if found:
                return found
    elif isinstance(payload, list):
        for v in payload:
            found = find_id(v, keys)
            if found:
                return found
    return ""


def parse_json_output(stdout: str) -> Any:
    """The CLI may print a banner before the JSON; take the first JSON object/array in stdout."""
    stdout = stdout.strip()
    if not stdout:
        return {}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        pass
    for opener in ("{", "["):
        i = stdout.find(opener)
        if i >= 0:
            try:
                return json.loads(stdout[i:])
            except json.JSONDecodeError:
                continue
    return {}


class OpenClawNotifier(_Base):
    transport: Transport = "openclaw"
    dest = "openclaw"

    def __init__(self, channels: dict[str, str], on_egress: Egress | None = None, *, sandbox: str = "unspoken",
                 mode: str = "exec", nemoclaw_bin: str = "nemoclaw", account: str = "",
                 hooks_url: str = "http://127.0.0.1:18789/hooks/agent", hooks_token: str | None = None,
                 agent_id: str = "main", runner: Runner | None = None, client: httpx.AsyncClient | None = None,
                 **kw: Any) -> None:
        super().__init__(channels, on_egress, **kw)
        if mode not in ("exec", "hooks"):
            raise BridgeConfigError(f"[notifier.openclaw] mode must be exec or hooks, not {mode!r}")
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$", sandbox):
            raise BridgeConfigError(f"[notifier.openclaw] sandbox name {sandbox!r} is not a NemoClaw sandbox name")
        self.sandbox = sandbox
        self.mode = mode
        self.nemoclaw_bin = nemoclaw_bin
        self.account = account.strip()
        self.hooks_url = hooks_url
        self.hooks_token = hooks_token
        self.agent_id = agent_id
        self._runner: Runner = runner or subprocess_runner
        self._client = client
        if mode == "hooks":
            if not hooks_token:
                raise BridgeConfigError(f"{ENV_HOOKS_TOKEN} is not set (needed for [notifier.openclaw] mode = \"hooks\")")
            self._client = client or httpx.AsyncClient(timeout=self.retry.timeout)
            self.supports_threads = False   # the agent relays; thread ids never come back

    # ---- exec mode: nemoclaw <sb> exec --no-stdin -- openclaw message <...> --json
    def argv(self, *openclaw_args: str) -> list[str]:
        base = [self.nemoclaw_bin, self.sandbox, "exec", "--no-stdin", "--", "openclaw", "message", *openclaw_args,
                "--channel", "discord"]
        if self.account:
            base += ["--account", self.account]
        return base + ["--json"]

    async def _exec_json(self, what: str, *openclaw_args: str) -> Any:
        rc, out, err = await self._runner(self.argv(*openclaw_args), self.retry.timeout)
        if rc != 0:
            msg = (err or out).strip()[:300]
            # policy / config problems do not fix themselves between retries
            if any(s in msg for s in ("not configured", "unknown", "Unknown", "permission", "No such sandbox",
                                      "is not registered", "requires")):
                raise _NoRetry(f"{what}: exit {rc}: {msg}")
            raise RuntimeError(f"{what}: exit {rc}: {msg}")
        data = parse_json_output(out)
        if isinstance(data, dict) and data.get("ok") is False:
            raise RuntimeError(f"{what}: {data.get('error') or data.get('deliveryStatus') or 'ok=false'}")
        return data

    # ---- hooks mode: POST /hooks/agent, the agent posts to channel/to
    async def _hook(self, target: str, text: str) -> str:
        assert self._client is not None
        payload = {
            "message": ("Post the following message to Discord exactly as written, with no commentary, "
                        "no preamble and no formatting changes:\n\n" + text),
            "name": "unspoken",
            "agentId": self.agent_id,
            "deliver": True,
            "channel": "discord",
            "to": target,
            "sessionMode": "isolated",
            "timeoutSeconds": int(self.retry.timeout),
        }
        r = await self._client.post(self.hooks_url, json=payload, headers={
            "Authorization": f"Bearer {self.hooks_token}",
            "Content-Type": "application/json",
            "Idempotency-Key": f"unspoken-{uuid.uuid4().hex}",
        })
        if r.status_code in (401, 404, 400):
            raise _NoRetry(f"hooks HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code == 429:
            raise _RateLimited(float(r.headers.get("Retry-After", "2")))
        if r.status_code >= 500 or r.status_code == 409:
            raise RuntimeError(f"hooks HTTP {r.status_code}")
        try:
            return str(r.json().get("runId", ""))
        except (ValueError, AttributeError):
            return ""

    # ---- primitives
    def _as_target(self, target: str) -> str:
        return target if ":" in target else f"channel:{target}"

    async def _send(self, target: str, text: str) -> str:
        if self.mode == "hooks":
            return await self._hook(self._as_target(target), text)
        data = await self._exec_json("message send", "send", "--target", self._as_target(target), f"--message={text}")
        return find_id(data)

    async def _create_thread(self, channel_id: str, message_id: str, name: str) -> str | None:
        data = await self._exec_json("thread create", "thread", "create", "--target", channel_id,
                                     "--message-id", message_id, "--thread-name", name[:100])
        tid = find_id(data, ("threadId", "thread_id", "id"))
        return tid or None

    async def _dm_target(self, user: str) -> str:
        return f"user:{user}"

    async def healthy(self) -> bool:
        try:
            if self.mode == "hooks":
                assert self._client is not None
                base = self.hooks_url.split("/hooks", 1)[0] or self.hooks_url
                r = await self._client.get(base, timeout=self.retry.timeout)
                return r.status_code < 500
            rc, _out, _err = await self._runner(
                [self.nemoclaw_bin, self.sandbox, "exec", "--no-stdin", "--", "openclaw", "channels", "status",
                 "--channel", "discord", "--json"], self.retry.timeout)
            return rc == 0
        except Exception:
            return False


# =============================================================================== factory
def make_notifier(config: dict[str, Any], on_egress: Egress | None = None) -> Notifier:
    """Build the Notifier named by config/bridge.toml [notifier].transport (env UNSPOKEN_NOTIFIER overrides).

    `config` is the whole bridge.toml document (or just its [notifier] table). Secrets come from the environment:
    DISCORD_BOT_TOKEN (discord-rest), OPENCLAW_HOOKS_TOKEN (openclaw, mode = hooks).
    """
    n = config.get("notifier", config) if isinstance(config, dict) else {}
    transport = (os.environ.get("UNSPOKEN_NOTIFIER") or n.get("transport") or "console").strip().lower()
    channels = {str(k): str(v) for k, v in dict(n.get("channels", {})).items()}
    common: dict[str, Any] = {
        "thread_kinds": tuple(n.get("thread_kinds", ["task"])),
        "chunk_chars": int(n.get("chunk_chars", 1900)),
        "retry": RetryPolicy(attempts=int(n.get("retries", 3)), backoff=float(n.get("backoff", 0.5)),
                             backoff_max=float(n.get("backoff_max", 4.0)), timeout=float(n.get("timeout", 15.0))),
    }
    if transport == "console":
        return ConsoleNotifier(channels or None, on_egress, **common)
    if transport == "discord-rest":
        rest = dict(n.get("discord_rest", {}))
        return DiscordRestNotifier(secret(ENV_DISCORD_TOKEN, required=True) or "", channels, on_egress,
                                   api_base=str(rest.get("api_base", "https://discord.com/api/v10")),
                                   auto_archive_minutes=int(rest.get("auto_archive_minutes", 1440)), **common)
    if transport == "openclaw":
        oc = dict(n.get("openclaw", {}))
        mode = str(oc.get("mode", "exec"))
        return OpenClawNotifier(channels, on_egress, sandbox=str(oc.get("sandbox", "unspoken")), mode=mode,
                                nemoclaw_bin=str(oc.get("nemoclaw_bin", "nemoclaw")), account=str(oc.get("account", "")),
                                hooks_url=str(oc.get("hooks_url", "http://127.0.0.1:18789/hooks/agent")),
                                hooks_token=secret(ENV_HOOKS_TOKEN, required=(mode == "hooks")),
                                agent_id=str(oc.get("agent_id", "main")), **common)
    raise BridgeConfigError(f"unknown notifier transport {transport!r} (openclaw | discord-rest | console)")
