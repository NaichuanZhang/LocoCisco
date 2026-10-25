"""Test doubles for the room's appliances.

FakeLight / FakePlug   in-memory Devices (same state dicts as the real drivers). `load_devices` uses them
                       for driver="fake" and for anything missing or misconfigured (flagged unhealthy), so
                       the demo never crashes on a device.
FakeWizBulb            a UDP server on 127.0.0.1 that speaks the real WiZ protocol (setPilot, getPilot,
                       getSystemConfig, registration), keeps a pilot, records every request and can drop
                       datagrams, delay replies or reject params, so the real WizClient/WizLight is tested
                       end to end with no bulb.
FakeKasaDevice         a python-kasa Device look-alike (update / turn_on / turn_off / is_on / alias /
                       model) for KasaPlug's `connect` seam; `fail_after`/`fail_with` make it misbehave.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from .scenes import (DIMMING_MAX, DIMMING_MIN, Pilot, SCENES, merge_pilot, pilot_for_set, pilot_to_state,
                     scene_pilot)
from .wiz import decode_reply


# ---------------------------------------------------------------------- in-memory devices
class FakeLight:
    kind = "light"

    def __init__(self, device_id: str = "lamp", *, healthy: bool = True, error: str = "") -> None:
        self.device_id = device_id
        self.driver = "fake"
        self.host = ""
        self._healthy = healthy
        self._error = error
        self._pilot: Pilot = {"state": False, "dimming": 100, "temp": 4000}
        self.commands: list[tuple[str, dict[str, Any]]] = []

    async def apply(self, command: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(args or {})
        self.commands.append((command, args))
        try:
            if command == "on":
                update: Pilot = {"state": True}
            elif command == "off":
                update = {"state": False}
            elif command == "scene":
                update = scene_pilot(str(args.get("scene", "")))
            elif command == "set":
                update = pilot_for_set(args.get("brightness"), args.get("color"))
            else:
                raise ValueError(f"unknown light command {command!r}")
        except (KeyError, ValueError, TypeError) as e:
            return {**self._snapshot(), "error": f"bad command {command!r}: {e}"}
        if self._healthy:
            self._pilot = merge_pilot(self._pilot, update)
        return self._snapshot()

    async def state(self) -> dict[str, Any]:
        return self._snapshot()

    async def healthy(self) -> bool:
        return self._healthy

    def set_healthy(self, healthy: bool, error: str = "") -> None:
        self._healthy, self._error = healthy, error

    def _snapshot(self) -> dict[str, Any]:
        st = pilot_to_state(self._pilot)
        st.update({"healthy": self._healthy, "error": self._error, "driver": self.driver, "host": self.host})
        return st


class FakePlug:
    kind = "plug"

    def __init__(self, device_id: str = "fan", *, healthy: bool = True, error: str = "") -> None:
        self.device_id = device_id
        self.driver = "fake"
        self.host = ""
        self._healthy = healthy
        self._error = error
        self.on = False
        self.commands: list[tuple[str, dict[str, Any]]] = []

    async def apply(self, command: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        self.commands.append((command, dict(args or {})))
        if command not in ("on", "off"):
            return {**self._snapshot(), "error": f"bad command {command!r}: plug commands are on, off"}
        if self._healthy:
            self.on = command == "on"
        return self._snapshot()

    async def state(self) -> dict[str, Any]:
        return self._snapshot()

    async def healthy(self) -> bool:
        return self._healthy

    def set_healthy(self, healthy: bool, error: str = "") -> None:
        self._healthy, self._error = healthy, error

    def _snapshot(self) -> dict[str, Any]:
        return {"on": self.on, "healthy": self._healthy, "error": self._error, "driver": self.driver,
                "host": self.host}


# ---------------------------------------------------------------------- fake WiZ bulb (UDP)
class FakeWizBulb(asyncio.DatagramProtocol):
    """Start with `await FakeWizBulb.start()`; talk to it at ("127.0.0.1", bulb.port); `await bulb.stop()`.

    Knobs: `drop_next` (silently drop that many requests), `delay_s` (answer late), `reject_next`
    (answer the next request with an "Invalid params" error), `strict_dimming` (reject dimming outside
    10..100 like a real bulb)."""

    def __init__(self, *, mac: str = "a8bb50aabbcc", module: str = "ESP01_SHRGB_03", fw: str = "1.21.4",
                 strict_dimming: bool = True) -> None:
        self.mac = mac
        self.module = module
        self.fw = fw
        self.strict_dimming = strict_dimming
        self.pilot: Pilot = {"state": False, "sceneId": 0, "r": 0, "g": 0, "b": 0, "c": 0, "w": 0, "dimming": 100}
        self.requests: list[dict[str, Any]] = []
        self.drop_next = 0
        self.reject_next = 0
        self.delay_s = 0.0
        self.port = 0
        self._transport: asyncio.DatagramTransport | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    @classmethod
    async def start(cls, host: str = "127.0.0.1", port: int = 0, **kw: Any) -> FakeWizBulb:
        bulb = cls(**kw)
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(lambda: bulb, local_addr=(host, port))
        bulb._transport = transport  # type: ignore[assignment]
        bulb.port = transport.get_extra_info("sockname")[1]
        return bulb

    async def stop(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        if self._transport is not None:
            self._transport.close()
            self._transport = None
        await asyncio.sleep(0)

    @property
    def host(self) -> str:
        return "127.0.0.1"

    def requests_for(self, method: str) -> list[dict[str, Any]]:
        return [r for r in self.requests if r.get("method") == method]

    # ---- protocol
    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            req = decode_reply(data)     # same shape both ways: {"method", "params"}
        except ValueError:
            return
        self.requests.append(req)
        if self.drop_next > 0:
            self.drop_next -= 1
            return
        reply = self.handle(req)
        if self.delay_s > 0:
            task = asyncio.get_running_loop().create_task(self._send_later(reply, addr))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        else:
            self._send(reply, addr)

    async def _send_later(self, reply: dict[str, Any], addr: tuple[str, int]) -> None:
        await asyncio.sleep(self.delay_s)
        self._send(reply, addr)

    def _send(self, reply: dict[str, Any], addr: tuple[str, int]) -> None:
        if self._transport is not None:
            self._transport.sendto(json.dumps(reply, separators=(",", ":")).encode(), addr)

    def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        method = req.get("method")
        params = req.get("params") or {}
        if self.reject_next > 0:
            self.reject_next -= 1
            return {"method": method, "env": "pro", "error": {"code": -32602, "message": "Invalid params"}}
        if method == "setPilot":
            dimming = params.get("dimming")
            if self.strict_dimming and dimming is not None and not (DIMMING_MIN <= int(dimming) <= DIMMING_MAX):
                return {"method": method, "env": "pro", "error": {"code": -32602, "message": "Invalid params"}}
            self.pilot = merge_pilot(self.pilot, params)
            return {"method": method, "env": "pro", "result": {"success": True}}
        if method == "getPilot":
            return {"method": method, "env": "pro", "result": {"mac": self.mac, "rssi": -58, "src": "", **self.pilot}}
        if method == "getSystemConfig":
            return {"method": method, "env": "pro", "result": {"mac": self.mac, "homeId": 1, "roomId": 1,
                                                             "moduleName": self.module, "fwVersion": self.fw,
                                                             "groupId": 0, "drvConf": [20, 2]}}
        if method == "registration":
            return {"method": method, "env": "pro", "result": {"mac": self.mac, "success": True}}
        return {"method": method, "env": "pro", "error": {"code": -32601, "message": "Method not found"}}

    # ---- what a test wants to know
    @property
    def state(self) -> dict[str, Any]:
        return pilot_to_state(self.pilot)

    @property
    def scene(self) -> str | None:
        return self.state["scene"]


def scene_pilots_match(bulb_pilot: Pilot, scene: str) -> bool:
    """True when the bulb holds exactly the scene's pilot values (helper for tests)."""
    return all(bulb_pilot.get(k) == v for k, v in SCENES[scene].items())


# ---------------------------------------------------------------------- fake python-kasa device
class FakeKasaDevice:
    """Shaped like kasa.Device for KasaPlug: await update()/turn_on()/turn_off(); is_on, alias, model."""

    def __init__(self, *, alias: str = "fan", model: str = "HS103(US)", on: bool = False,
                 fail_with: Exception | None = None, fail_after: int | None = None) -> None:
        self.alias = alias
        self.model = model
        self.host = "127.0.0.1"
        self.is_on = on
        self.calls: list[str] = []
        self.fail_with = fail_with
        self.fail_after = fail_after      # start failing after this many calls (None = never)

    def _tick(self, op: str) -> None:
        self.calls.append(op)
        if self.fail_with is not None and (self.fail_after is None or len(self.calls) > self.fail_after):
            raise self.fail_with

    async def update(self) -> None:
        self._tick("update")

    async def turn_on(self) -> None:
        self._tick("turn_on")
        self.is_on = True

    async def turn_off(self) -> None:
        self._tick("turn_off")
        self.is_on = False

    async def disconnect(self) -> None:
        self.calls.append("disconnect")


def fake_kasa_connector(device: FakeKasaDevice, *, fail_connect: Exception | None = None,
                        connects: list[tuple[str, str | None, str | None, float, int | None]] | None = None):
    """A KasaPlug `connect` that hands back `device` (or raises `fail_connect`), logging the call args."""
    async def connect(host: str, username: str | None, password: str | None, timeout_s: float,
                      port: int | None = None) -> Any:
        if connects is not None:
            connects.append((host, username, password, timeout_s, port))
        if fail_connect is not None:
            raise fail_connect
        return device
    return connect


# ---------------------------------------------------------------------- fake Kasa plug (legacy IOT protocol)
XOR_IV = 171


def xor_encrypt(plain: bytes) -> bytes:
    """Kasa legacy "autokey" XOR (TP-Link IOT protocol), no length prefix."""
    key, out = XOR_IV, bytearray()
    for b in plain:
        key ^= b
        out.append(key)
    return bytes(out)


def xor_decrypt(cipher: bytes) -> bytes:
    key, out = XOR_IV, bytearray()
    for c in cipher:
        out.append(key ^ c)
        key = c
    return bytes(out)


class FakeKasaIotPlug:
    """A legacy Kasa plug (HS103-style, IOT protocol, no credentials) on 127.0.0.1: answers the UDP
    discovery query and serves TCP requests (4-byte big-endian length + XOR payload) on the SAME port,
    which is what python-kasa expects for `Discover.discover_single(host, port=...)`. Unknown modules get
    {"err_code": -1, "err_msg": "module not support"} like a real plug without an energy meter.
    `fail_tcp` makes the TCP side drop every connection (an unreachable plug after discovery)."""

    def __init__(self, *, alias: str = "fan", model: str = "HS103(US)", on: bool = False,
                 mac: str = "50:C7:BF:00:00:01") -> None:
        self.alias = alias
        self.model = model
        self.mac = mac
        self.relay = 1 if on else 0
        self.led_off = 0
        self.requests: list[dict[str, Any]] = []
        self.udp_requests = 0
        self.fail_tcp = False
        self.port = 0
        self._udp: asyncio.DatagramTransport | None = None
        self._tcp: asyncio.AbstractServer | None = None

    @property
    def on(self) -> bool:
        return bool(self.relay)

    @classmethod
    async def start(cls, host: str = "127.0.0.1", port: int = 0, **kw: Any) -> FakeKasaIotPlug:
        plug = cls(**kw)
        loop = asyncio.get_running_loop()
        for _ in range(20):          # UDP and TCP are separate namespaces: find a number free in both
            udp, _ = await loop.create_datagram_endpoint(lambda: _UdpRelay(plug), local_addr=(host, port))
            p = udp.get_extra_info("sockname")[1]
            try:
                tcp = await asyncio.start_server(plug._serve_tcp, host, p)
            except OSError:
                udp.close()
                if port:
                    raise
                continue
            plug._udp, plug._tcp, plug.port = udp, tcp, p   # type: ignore[assignment]
            return plug
        raise OSError("no port free for both UDP and TCP")

    async def stop(self) -> None:
        if self._udp is not None:
            self._udp.close()
            self._udp = None
        if self._tcp is not None:
            self._tcp.close()
            await self._tcp.wait_closed()
            self._tcp = None

    # ---- protocol
    def sysinfo(self) -> dict[str, Any]:
        return {"err_code": 0, "sw_ver": "1.1.4 Build 230103 Rel.084402", "hw_ver": "2.1", "model": self.model,
                "deviceId": "8006" + "0" * 36, "oemId": "0" * 32, "hwId": "0" * 32, "rssi": -50,
                "longitude_i": 0, "latitude_i": 0, "alias": self.alias, "status": "new",
                "mic_type": "IOT.SMARTPLUGSWITCH", "feature": "TIM", "mac": self.mac, "updating": 0,
                "led_off": self.led_off, "relay_state": self.relay, "on_time": 0, "icon_hash": "",
                "dev_name": "Smart Wi-Fi Plug Lite", "active_mode": "none", "next_action": {"type": -1},
                "ntc_state": 0}

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        response: dict[str, Any] = {}
        for module, commands in request.items():
            if module != "system" or not isinstance(commands, dict):
                response[module] = {"err_code": -1, "err_msg": "module not support"}
                continue
            out: dict[str, Any] = {}
            for cmd, args in commands.items():
                if cmd == "get_sysinfo":
                    out[cmd] = self.sysinfo()
                elif cmd == "set_relay_state":
                    self.relay = 1 if int((args or {}).get("state", 0)) else 0
                    out[cmd] = {"err_code": 0}
                elif cmd == "set_led_off":
                    self.led_off = int((args or {}).get("off", 0))
                    out[cmd] = {"err_code": 0}
                else:
                    out[cmd] = {"err_code": -2, "err_msg": "member not support"}
            response["system"] = out
        return response

    def _reply_bytes(self, cipher: bytes) -> bytes | None:
        try:
            request = json.loads(xor_decrypt(cipher).decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return xor_encrypt(json.dumps(self.handle(request), separators=(",", ":")).encode())

    async def _serve_tcp(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            if self.fail_tcp:
                return
            while True:
                head = await reader.readexactly(4)
                body = await reader.readexactly(int.from_bytes(head, "big"))
                if self.fail_tcp:           # the plug went away mid-session: drop the connection
                    return
                reply = self._reply_bytes(body)
                if reply is None:
                    return
                writer.write(len(reply).to_bytes(4, "big") + reply)
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()


class _UdpRelay(asyncio.DatagramProtocol):
    def __init__(self, plug: FakeKasaIotPlug) -> None:
        self.plug = plug
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self.plug.udp_requests += 1
        reply = self.plug._reply_bytes(data)
        if reply is not None and self.transport is not None:
            self.transport.sendto(reply, addr)
