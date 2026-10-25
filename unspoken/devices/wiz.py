"""WiZ colour bulb over its local UDP JSON protocol (port 38899): setPilot / getPilot / getSystemConfig /
registration (discovery). No vendor cloud, no password: anyone on the Wi-Fi can drive the bulb, which is
why docs/DEVICES.md puts the bulbs on our own travel router.

Wire format (one JSON object per datagram, the bulb answers from the same port):
    >> {"method": "setPilot", "params": {"state": true, "r": 255, "g": 0, "b": 0, "dimming": 70}}
    << {"method": "setPilot", "env": "pro", "result": {"success": true}}
    >> {"method": "getPilot", "params": {}}
    << {"method": "getPilot", "env": "pro", "result": {"mac": "...", "rssi": -60, "state": true,
        "sceneId": 0, "r": 255, "g": 0, "b": 0, "c": 0, "w": 0, "dimming": 70}}
    << {"method": "setPilot", "env": "pro", "error": {"code": -32602, "message": "Invalid params"}}
Discovery: broadcast {"method": "registration", "params": {"phoneMac": "AAAAAAAAAAAA", "register": false,
"phoneIp": "1.2.3.4", "id": "1"}} to 255.255.255.255:38899; every bulb answers with its mac.
Reference: https://github.com/sbidy/pywizlight (README "protocol" notes, pywizlight/bulb.py, discovery.py).

WizLight implements unspoken.contracts.Device for kind "light":
- 1 s timeout, 2 retries (so at most 3 datagrams, ~3 s) while the bulb is believed healthy; one try once
  it is known to be down, so a dead lamp costs a command about 1 s, not 3;
- bursts are debounced: the first command goes out at once, later ones inside `debounce_s` are coalesced
  and only the newest is sent when the window closes;
- apply()/state()/healthy() NEVER raise into the caller: on failure they return the last known state with
  "healthy": False and an "error" line.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable

from .scenes import (SCENES, Pilot, merge_pilot, pilot_for_set, pilot_to_state, scene_pilot)

log = logging.getLogger("unspoken.devices.wiz")

WIZ_PORT = 38899
DEFAULT_TIMEOUT_S = 1.0
DEFAULT_RETRIES = 2
DEFAULT_DEBOUNCE_S = 0.15
REGISTRATION_PARAMS = {"phoneMac": "AAAAAAAAAAAA", "register": False, "phoneIp": "1.2.3.4", "id": "1"}


class WizError(Exception):
    """The bulb could not be reached (timeout, network error) after the retries."""


class WizRejected(WizError):
    """The bulb answered with an error object (bad params). Not retried: the same request fails again."""


def encode_request(method: str, params: dict[str, Any] | None = None) -> bytes:
    return json.dumps({"method": method, "params": params or {}}, separators=(",", ":")).encode()


def decode_reply(data: bytes) -> dict[str, Any]:
    """Raises ValueError when the datagram is not a JSON object."""
    try:
        reply = json.loads(data.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        raise ValueError(f"not JSON: {data[:60]!r}") from e
    if not isinstance(reply, dict):
        raise ValueError(f"not a JSON object: {data[:60]!r}")
    return reply


class _DatagramQueue(asyncio.DatagramProtocol):
    """Puts every received datagram (or socket error) on a queue; the client reads with a deadline."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[tuple[bytes | Exception, tuple[str, int] | None]] = asyncio.Queue()
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self.queue.put_nowait((data, addr))

    def error_received(self, exc: Exception) -> None:
        self.queue.put_nowait((exc, None))   # e.g. ICMP port unreachable on a connected UDP socket

    def connection_lost(self, exc: Exception | None) -> None:
        self.queue.put_nowait((exc or ConnectionError("udp transport closed"), None))


class WizClient:
    """Request/response to ONE bulb. One request in flight at a time (the bulb has no request ids)."""

    def __init__(self, host: str, port: int = WIZ_PORT, *, timeout: float = DEFAULT_TIMEOUT_S,
                 retries: int = DEFAULT_RETRIES) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.retries = retries
        self._proto: _DatagramQueue | None = None
        self._lock = asyncio.Lock()
        self.sent = 0           # datagrams sent (tests + X-ray)
        self.failures = 0

    async def request(self, method: str, params: dict[str, Any] | None = None, *,
                      retries: int | None = None) -> dict[str, Any]:
        """The bulb's `result` object. Raises WizRejected (bulb said error) or WizError (unreachable)."""
        attempts = (self.retries if retries is None else retries) + 1
        data = encode_request(method, params)
        last: Exception | None = None
        async with self._lock:
            for attempt in range(attempts):
                try:
                    return await self._once(data, method)
                except WizRejected:
                    self.failures += 1
                    raise
                except (TimeoutError, OSError, ValueError, ConnectionError) as e:
                    last = e if str(e) else TimeoutError(f"no reply within {self.timeout:.2f}s")
                    self.failures += 1
                    await self._close()      # a fresh socket for the retry (ICMP errors poison the old one)
                    log.debug("wiz %s:%s %s attempt %d/%d failed: %r", self.host, self.port, method,
                              attempt + 1, attempts, e)
        raise WizError(f"{method} to {self.host}:{self.port} failed after {attempts} attempt(s): "
                       f"{type(last).__name__}: {last}")

    async def _once(self, data: bytes, method: str) -> dict[str, Any]:
        proto = await self._ensure()
        while not proto.queue.empty():          # stale replies from a timed-out earlier request
            proto.queue.get_nowait()
        assert proto.transport is not None
        proto.transport.sendto(data)
        self.sent += 1
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"no reply within {self.timeout:.2f}s")
            item, _addr = await asyncio.wait_for(proto.queue.get(), remaining)
            if isinstance(item, Exception):
                raise item
            reply = decode_reply(item)
            if reply.get("method") not in (method, None):
                continue                          # answer to an earlier request; keep waiting for ours
            if "error" in reply:
                err = reply["error"] or {}
                raise WizRejected(f"{method}: {err.get('message', err)} (code {err.get('code', '?')})")
            result = reply.get("result", {})
            return result if isinstance(result, dict) else {"value": result}

    async def _ensure(self) -> _DatagramQueue:
        if self._proto is None or self._proto.transport is None or self._proto.transport.is_closing():
            loop = asyncio.get_running_loop()
            _transport, proto = await loop.create_datagram_endpoint(
                _DatagramQueue, remote_addr=(self.host, self.port))
            self._proto = proto
        return self._proto

    async def _close(self) -> None:
        if self._proto is not None and self._proto.transport is not None:
            self._proto.transport.close()
        self._proto = None

    async def aclose(self) -> None:
        await self._close()


class WizLight:
    """A WiZ bulb as a Device (kind "light"). See the module docstring for the failure contract."""

    kind = "light"

    def __init__(self, device_id: str, host: str, port: int = WIZ_PORT, *, timeout: float = DEFAULT_TIMEOUT_S,
                 retries: int = DEFAULT_RETRIES, debounce_s: float = DEFAULT_DEBOUNCE_S,
                 state_ttl_s: float = 2.0, client: WizClient | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.device_id = device_id
        self.host = host
        self.port = port
        self.driver = "wiz"
        self.debounce_s = debounce_s
        self.state_ttl_s = state_ttl_s
        self.client = client or WizClient(host, port, timeout=timeout, retries=retries)
        self.clock = clock
        self._pilot: Pilot = {"state": False}      # last known bulb pilot
        self._healthy: bool | None = None          # None = never contacted
        self._error = ""
        self._last_contact: float | None = None
        self._last_send: float = float("-inf")
        self._desired: Pilot | None = None
        self._desired_seq = 0
        self._sent_seq = 0
        self._send_lock = asyncio.Lock()
        self.info: dict[str, Any] = {}             # getSystemConfig result (mac, moduleName, fwVersion)

    # ------------------------------------------------------------------ Device protocol
    async def apply(self, command: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = args or {}
        try:
            pilot = self.pilot_for(command, args)
        except (KeyError, ValueError, TypeError) as e:
            log.warning("%s: bad command %s %s: %s", self.device_id, command, args, e)
            return {**self._snapshot(), "error": f"bad command {command!r}: {e}"}
        await self._send_debounced(pilot)
        return self._snapshot()

    async def state(self) -> dict[str, Any]:
        if self._last_contact is None or self.clock() - self._last_contact > self.state_ttl_s:
            await self.refresh()
        return self._snapshot()

    async def healthy(self) -> bool:
        if self._healthy is None:
            await self.refresh()
        return bool(self._healthy)

    # ------------------------------------------------------------------ commands -> pilots
    @staticmethod
    def pilot_for(command: str, args: dict[str, Any]) -> Pilot:
        if command == "on":
            return {"state": True}
        if command == "off":
            return {"state": False}
        if command == "scene":
            scene = str(args.get("scene", ""))
            if scene not in SCENES:
                raise KeyError(f"unknown scene {scene!r}; scenes: {', '.join(SCENES)}")
            return scene_pilot(scene)
        if command == "set":
            return pilot_for_set(args.get("brightness"), args.get("color"))
        raise ValueError(f"unknown light command {command!r}; commands: on, off, scene, set")

    # ------------------------------------------------------------------ I/O
    async def refresh(self) -> dict[str, Any]:
        """getPilot -> state. Never raises."""
        try:
            result = await self.client.request("getPilot", {}, retries=self._retries_now())
        except WizError as e:
            self._mark_down(str(e))
        else:
            self._pilot = {k: v for k, v in result.items() if k not in ("mac", "rssi", "src", "env")}
            self._mark_up()
        return self._snapshot()

    async def system_config(self) -> dict[str, Any]:
        """getSystemConfig (mac, moduleName, fwVersion, ...). Never raises; {} when unreachable."""
        try:
            self.info = await self.client.request("getSystemConfig", {}, retries=self._retries_now())
            self._mark_up()
        except WizError as e:
            self._mark_down(str(e))
            return {}
        return dict(self.info)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _send_debounced(self, pilot: Pilot) -> None:
        """Leading + trailing debounce: the first command of a burst goes out now; commands that arrive
        inside `debounce_s` queue behind the lock, and whoever gets the lock next waits out the window and
        sends only the NEWEST desired pilot. Everyone else sees it was already sent and returns."""
        self._desired = pilot
        self._desired_seq += 1
        my_seq = self._desired_seq
        async with self._send_lock:
            if self._sent_seq >= my_seq:
                return                                   # a later caller already flushed our pilot (or newer)
            wait = self.debounce_s - (self.clock() - self._last_send)
            if wait > 0:
                await asyncio.sleep(wait)
            target = self._desired or pilot
            seq = self._desired_seq
            self._last_send = self.clock()
            try:
                await self.client.request("setPilot", target, retries=self._retries_now())
            except WizRejected as e:
                self._error = str(e)                      # our params were bad; the bulb itself is fine
                self._healthy = True
                self._last_contact = self.clock()
                log.warning("%s rejected %s: %s", self.device_id, target, e)
            except WizError as e:
                self._mark_down(str(e))
            else:
                self._pilot = merge_pilot(self._pilot, target)
                self._mark_up()
            self._sent_seq = seq

    def _retries_now(self) -> int:
        return self.client.retries if self._healthy is not False else 0

    def _mark_up(self) -> None:
        self._healthy = True
        self._error = ""
        self._last_contact = self.clock()

    def _mark_down(self, error: str) -> None:
        if self._healthy is not False:
            log.warning("%s (%s:%s) unreachable: %s", self.device_id, self.host, self.port, error)
        self._healthy = False
        self._error = error

    def _snapshot(self) -> dict[str, Any]:
        st = pilot_to_state(self._pilot)
        st.update({"healthy": bool(self._healthy), "error": self._error, "driver": self.driver,
                   "host": self.host})
        return st


# ---------------------------------------------------------------------- discovery
async def discover_wiz(*, broadcast: str = "255.255.255.255", port: int = WIZ_PORT, timeout: float = 3.0,
                       probe_config: bool = True) -> list[dict[str, Any]]:
    """Broadcast a WiZ registration and collect every bulb that answers:
    [{"host", "mac", "moduleName", "fwVersion"}]. `broadcast` may be a unicast address (tests, or one
    known bulb). Never raises; an unusable network gives []."""
    loop = asyncio.get_running_loop()
    found: dict[str, dict[str, Any]] = {}
    try:
        transport, proto = await loop.create_datagram_endpoint(
            _DatagramQueue, local_addr=("0.0.0.0", 0), allow_broadcast=True)
    except OSError as e:
        log.warning("wiz discovery: cannot open a UDP socket: %s", e)
        return []
    try:
        transport.sendto(encode_request("registration", REGISTRATION_PARAMS), (broadcast, port))
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                item, addr = await asyncio.wait_for(proto.queue.get(), remaining)
            except TimeoutError:
                break
            if isinstance(item, Exception) or addr is None:
                continue
            try:
                reply = decode_reply(item)
            except ValueError:
                continue
            if reply.get("method") != "registration":
                continue
            result = reply.get("result") or {}
            found[addr[0]] = {"host": addr[0], "mac": str(result.get("mac", "")), "moduleName": "",
                              "fwVersion": ""}
    finally:
        transport.close()
    if probe_config:
        for host, entry in found.items():
            client = WizClient(host, port, timeout=min(1.0, timeout), retries=0)
            try:
                cfg = await client.request("getSystemConfig", {})
                entry["moduleName"] = str(cfg.get("moduleName", ""))
                entry["fwVersion"] = str(cfg.get("fwVersion", ""))
                entry["mac"] = entry["mac"] or str(cfg.get("mac", ""))
            except WizError:
                pass
            finally:
                await client.aclose()
    return sorted(found.values(), key=lambda d: d["host"])
