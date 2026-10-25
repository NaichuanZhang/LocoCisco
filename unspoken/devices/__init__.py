"""Room appliances, driven locally (no vendor cloud): a WiZ colour bulb (lamp) and a Kasa plug (fan).

    from unspoken.devices import load_devices, load_room_config
    devices = load_devices(load_room_config())          # config/room.toml, or $UNSPOKEN_ROOM_CONFIG
    await devices["lamp"].apply("scene", {"scene": "off_record"})
    await devices["fan"].apply("off")

`load_devices` never fails: a device that is missing from the config, has an unknown driver, a bad host or
an import problem becomes a FakeLight/FakePlug flagged unhealthy (state()["healthy"] is False, "error"
says why), so the room brain always has a "lamp" and a "fan" to talk to. Real drivers never raise either.
Device protocol: unspoken.contracts.Device. CLI: `python -m unspoken.devices --help`.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tomllib
from pathlib import Path
from typing import Any

from ..contracts import Device
from .fakes import FakeLight, FakePlug, FakeWizBulb
from .kasa_plug import KasaPlug
from .scenes import SCENE_NAMES, SCENES
from .wiz import WizClient, WizLight

__all__ = ["load_devices", "load_room_config", "probe_devices", "close_devices", "ROOM_CONFIG_PATH",
           "REQUIRED_DEVICES", "SCENES", "SCENE_NAMES", "Device", "WizLight", "WizClient", "KasaPlug",
           "FakeLight", "FakePlug", "FakeWizBulb"]

log = logging.getLogger("unspoken.devices")

ROOM_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "room.toml"
ROOM_CONFIG_ENV = "UNSPOKEN_ROOM_CONFIG"

# The room brain assumes these exist. Anything else in [devices] is loaded too.
REQUIRED_DEVICES: dict[str, str] = {"lamp": "light", "fan": "plug"}
KINDS: dict[str, tuple[str, ...]] = {"light": ("wiz", "fake"), "plug": ("kasa", "fake")}


def load_room_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Parse config/room.toml (or `path`, or $UNSPOKEN_ROOM_CONFIG). A missing or broken file logs a
    warning and returns {} so load_devices falls back to unhealthy fakes instead of crashing."""
    p = Path(path or os.environ.get(ROOM_CONFIG_ENV) or ROOM_CONFIG_PATH)
    try:
        with open(p, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        log.warning("room config %s not found; every device becomes an unhealthy fake", p)
    except (OSError, tomllib.TOMLDecodeError) as e:
        log.warning("room config %s unreadable (%s); every device becomes an unhealthy fake", p, e)
    return {}


def load_devices(config: dict[str, Any] | None, *, required: dict[str, str] | None = None) -> dict[str, Device]:
    """config: the parsed room.toml ({"devices": {...}, ...}) or just its [devices] table.
    Each entry: kind = "light" | "plug", driver = "wiz" | "kasa" | "fake", host = "..." (real drivers),
    optional port / timeout / retries / debounce_s. Returns {device_id: Device}; see the module docstring
    for the fallback rules."""
    required = REQUIRED_DEVICES if required is None else required
    table = _devices_table(config)
    devices: dict[str, Device] = {}
    for device_id, entry in table.items():
        devices[device_id] = _build(str(device_id), entry if isinstance(entry, dict) else {})
    for device_id, kind in required.items():
        if device_id not in devices:
            log.warning("device %r is not in [devices]; using an unhealthy fake %s", device_id, kind)
            devices[device_id] = _fake(device_id, kind, False, f"{device_id} is not configured in room.toml")
        elif devices[device_id].kind != kind:
            log.warning("device %r is kind %r, the room expects %r", device_id, devices[device_id].kind, kind)
    return devices


def _devices_table(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    if "devices" in config and isinstance(config["devices"], dict):
        return config["devices"]
    # already the [devices] table (every value is a table with a kind / driver), or empty
    if all(isinstance(v, dict) for v in config.values()) and any("kind" in v or "driver" in v
                                                                   for v in config.values()):
        return config
    return {}


def _build(device_id: str, entry: dict[str, Any]) -> Device:
    kind = str(entry.get("kind") or "").strip().lower()
    driver = str(entry.get("driver") or "fake").strip().lower()
    if kind not in KINDS:
        return _fake(device_id, "light" if "light" in device_id or "lamp" in device_id else "plug", False,
                     f"{device_id}: unknown kind {kind!r} (light or plug)")
    if driver not in KINDS[kind]:
        return _fake(device_id, kind, False, f"{device_id}: unknown driver {driver!r} for kind {kind!r}")
    if driver == "fake":
        return _fake(device_id, kind, bool(entry.get("healthy", True)), str(entry.get("error", "")))
    host = str(entry.get("host") or "").strip()
    if not host:
        return _fake(device_id, kind, False, f"{device_id}: driver {driver!r} needs host = \"...\"")
    try:
        timeout = float(entry.get("timeout", 1.0))
        retries = int(entry.get("retries", 2))
        if driver == "wiz":
            return WizLight(device_id, host, int(entry.get("port", 38899)), timeout=timeout, retries=retries,
                            debounce_s=float(entry.get("debounce_s", 0.15)))
        port = entry.get("port")
        return KasaPlug(device_id, host, port=int(port) if port is not None else None, timeout=timeout,
                        retries=retries)
    except (TypeError, ValueError) as e:
        return _fake(device_id, kind, False, f"{device_id}: bad config ({e})")


def _fake(device_id: str, kind: str, healthy: bool, error: str) -> Device:
    if error:
        log.warning("%s", error)
    if kind == "light":
        return FakeLight(device_id, healthy=healthy, error=error)
    return FakePlug(device_id, healthy=healthy, error=error)


async def probe_devices(devices: dict[str, Device], timeout: float = 5.0) -> dict[str, dict[str, Any]]:
    """{device_id: state()} for every device, probed in parallel. Never raises."""
    async def one(dev: Device) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(dev.state(), timeout)
        except Exception as e:   # noqa: BLE001  the drivers never raise; this guards foreign Devices
            return {"healthy": False, "error": f"{type(e).__name__}: {e}"}
    ids = list(devices)
    states = await asyncio.gather(*(one(devices[i]) for i in ids))
    return dict(zip(ids, states, strict=False))


async def close_devices(devices: dict[str, Device]) -> None:
    for dev in devices.values():
        close = getattr(dev, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:   # noqa: BLE001
                pass
