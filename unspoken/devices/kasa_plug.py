"""TP-Link Kasa smart plug (the fan that stands in for the AC) through python-kasa, LAN only.

python-kasa speaks both Kasa protocols: the legacy IOT protocol (TCP 9999, XOR, no credentials) and the
newer SMART/KLAP protocol (HTTP 80, encrypted handshake) that newer hardware and some firmware updates use.
KLAP devices authenticate the LOCAL handshake with the TP-Link account the plug was set up with, so for
those two env vars must be set (python-kasa's own names, so the `kasa` CLI agrees):
    KASA_USERNAME=you@example.com  KASA_PASSWORD=...
Credentials never come from a config file. Which plugs need them: docs/DEVICES.md "buyer's list".
Reference: https://python-kasa.readthedocs.io/en/stable/topics.html#protocols-and-transports

KasaPlug implements unspoken.contracts.Device for kind "plug": commands "on" | "off"; state {"on", ...}.
It never raises into the caller: a plug that is off the network returns the last known state with
"healthy": False and an "error" line, and tries again on the next command.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Awaitable, Callable

log = logging.getLogger("unspoken.devices.kasa")

DEFAULT_TIMEOUT_S = 1.0
DEFAULT_RETRIES = 2
USERNAME_ENV = "KASA_USERNAME"
PASSWORD_ENV = "KASA_PASSWORD"

# connect(host, username, password, timeout_s, port) -> a python-kasa Device-like object:
#   async update(), async turn_on(), async turn_off(), is_on, alias, model, host
# port: None = the protocol default (9999 legacy / 80 KLAP); set for a legacy device on another port.
Connector = Callable[[str, str | None, str | None, float, int | None], Awaitable[Any]]


def credentials_from_env(env: dict[str, str] | None = None) -> tuple[str | None, str | None]:
    """(username, password) from KASA_USERNAME / KASA_PASSWORD, or (None, None). Values never logged."""
    e = os.environ if env is None else env
    user, pw = e.get(USERNAME_ENV) or None, e.get(PASSWORD_ENV) or None
    return (user, pw) if user and pw else (None, None)


async def kasa_connect(host: str, username: str | None, password: str | None, timeout_s: float,
                       port: int | None = None) -> Any:
    """Default connector: python-kasa probes the host (UDP 9999 + 20002), picks the protocol and
    authenticates when needed. Imported lazily so tests and the fake driver never need python-kasa."""
    from kasa import Credentials, Discover   # noqa: PLC0415  (lazy: optional at import time)

    creds = Credentials(username, password) if username and password else None
    secs = max(1, int(round(timeout_s)))
    dev = await Discover.discover_single(host, credentials=creds, timeout=secs, discovery_timeout=secs, port=port)
    if dev is None:
        raise ConnectionError(f"no Kasa device answered at {host}")
    await dev.update()
    return dev


class KasaPlug:
    """A Kasa plug as a Device (kind "plug")."""

    kind = "plug"

    def __init__(self, device_id: str, host: str, *, port: int | None = None, username: str | None = None,
                 password: str | None = None, timeout: float = DEFAULT_TIMEOUT_S, retries: int = DEFAULT_RETRIES,
                 state_ttl_s: float = 2.0, connect: Connector | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.device_id = device_id
        self.host = host
        self.port = port
        self.driver = "kasa"
        self.timeout = timeout
        self.retries = retries
        self.state_ttl_s = state_ttl_s
        self.clock = clock
        self._connect: Connector = connect or kasa_connect
        if username is None and password is None:
            username, password = credentials_from_env()
        self._username, self._password = username, password
        self._dev: Any = None
        self._on: bool | None = None
        self._healthy: bool | None = None
        self._error = ""
        self._last_contact: float | None = None
        self._lock = asyncio.Lock()
        self.info: dict[str, Any] = {}

    # ------------------------------------------------------------------ Device protocol
    async def apply(self, command: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if command not in ("on", "off"):
            log.warning("%s: bad command %s %s", self.device_id, command, args)
            return {**self._snapshot(), "error": f"bad command {command!r}: plug commands are on, off"}
        await self._run(command)
        return self._snapshot()

    async def state(self) -> dict[str, Any]:
        if self._last_contact is None or self.clock() - self._last_contact > self.state_ttl_s:
            await self._run("update")
        return self._snapshot()

    async def healthy(self) -> bool:
        if self._healthy is None:
            await self._run("update")
        return bool(self._healthy)

    async def aclose(self) -> None:
        dev, self._dev = self._dev, None
        if dev is not None and hasattr(dev, "disconnect"):
            try:
                await dev.disconnect()
            except Exception:   # noqa: BLE001  closing is best effort
                pass

    # ------------------------------------------------------------------ I/O
    async def _run(self, op: str) -> None:
        """Connect if needed, then update / turn_on / turn_off, with retries. Never raises."""
        attempts = (self.retries if self._healthy is not False else 0) + 1
        async with self._lock:
            last = ""
            for attempt in range(attempts):
                try:
                    await asyncio.wait_for(self._attempt(op), timeout=self.timeout * 2 + 1.0)
                except asyncio.CancelledError:
                    raise
                except Exception as e:   # noqa: BLE001  any python-kasa / network error: retry, then report
                    last = f"{type(e).__name__}: {e}"
                    self._dev = None     # reconnect on the next try
                    log.debug("%s %s attempt %d/%d failed: %s", self.device_id, op, attempt + 1, attempts, last)
                else:
                    self._mark_up()
                    return
            self._mark_down(f"{op} on {self.host} failed after {attempts} attempt(s): {last}")

    async def _attempt(self, op: str) -> None:
        if self._dev is None:
            self._dev = await self._connect(self.host, self._username, self._password, self.timeout, self.port)
            self.info = {"alias": getattr(self._dev, "alias", None), "model": getattr(self._dev, "model", None)}
        dev = self._dev
        if op == "on":
            await dev.turn_on()
        elif op == "off":
            await dev.turn_off()
        await dev.update()
        self._on = bool(dev.is_on)

    def _mark_up(self) -> None:
        self._healthy = True
        self._error = ""
        self._last_contact = self.clock()

    def _mark_down(self, error: str) -> None:
        if self._healthy is not False:
            log.warning("%s (%s) unreachable: %s", self.device_id, self.host, error)
        self._healthy = False
        self._error = error

    def _snapshot(self) -> dict[str, Any]:
        return {"on": bool(self._on), "healthy": bool(self._healthy), "error": self._error, "driver": self.driver,
                "host": self.host, **{k: v for k, v in self.info.items() if v}}


async def discover_kasa(*, broadcast: str = "255.255.255.255", timeout: float = 3.0,
                        username: str | None = None, password: str | None = None) -> list[dict[str, Any]]:
    """Broadcast Kasa discovery: [{"host", "model", "alias", "mac", "needs_credentials"}]. Never raises."""
    try:
        from kasa import Credentials, Discover, UnsupportedDeviceError   # noqa: PLC0415
    except ImportError as e:
        log.warning("kasa discovery: python-kasa not installed (%s)", e)
        return []
    if username is None and password is None:
        username, password = credentials_from_env()
    creds = Credentials(username, password) if username and password else None
    unsupported: list[dict[str, Any]] = []

    async def on_unsupported(err: UnsupportedDeviceError) -> None:
        unsupported.append({"host": getattr(err, "host", "?"), "model": "unsupported", "alias": "",
                            "mac": "", "needs_credentials": None, "detail": str(err)[:120]})

    try:
        devices = await Discover.discover(target=broadcast, discovery_timeout=max(1, int(round(timeout))),
                                          credentials=creds, on_unsupported=on_unsupported)
    except Exception as e:   # noqa: BLE001
        log.warning("kasa discovery failed: %s", e)
        return []
    out: list[dict[str, Any]] = []
    for host, dev in devices.items():
        needs = False
        try:
            await dev.update()
        except Exception as e:   # noqa: BLE001  AuthenticationError and friends: still list the plug
            needs = type(e).__name__ == "AuthenticationError" or needs
        try:
            conn = dev.config.connection_type
            encryption = getattr(getattr(conn, "encryption_type", None), "value", "")
            if encryption and encryption.lower() != "xor":
                needs = True
        except Exception:   # noqa: BLE001
            encryption = ""
        out.append({"host": host, "model": str(getattr(dev, "model", "") or ""),
                    "alias": str(getattr(dev, "alias", "") or ""), "mac": str(getattr(dev, "mac", "") or ""),
                    "encryption": encryption, "needs_credentials": needs})
    return sorted(out + unsupported, key=lambda d: str(d["host"]))
