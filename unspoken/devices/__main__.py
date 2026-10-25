"""CLI for the room's appliances.

    python -m unspoken.devices discover                 # WiZ broadcast + Kasa discovery -> room.toml block
    python -m unspoken.devices list                     # every device in room.toml with live state + health
    python -m unspoken.devices set lamp scene off_record
    python -m unspoken.devices set lamp set brightness=40 color=#ff8800
    python -m unspoken.devices set fan off
    python -m unspoken.devices scenes                   # the scene table
    python -m unspoken.devices fake-bulb --port 38899   # a WiZ bulb on 127.0.0.1 for end-to-end checks

`--config PATH` (or $UNSPOKEN_ROOM_CONFIG) picks another room.toml; `--json` prints machine-readable
output. Exit status 1 when a command left its device unhealthy or rejected, so scripts can check.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from . import (SCENES, close_devices, load_devices, load_room_config, probe_devices)
from .fakes import FakeWizBulb
from .kasa_plug import discover_kasa
from .wiz import WIZ_PORT, discover_wiz


def _room_toml_block(wiz: list[dict[str, Any]], kasa: list[dict[str, Any]]) -> str:
    lines = ["# paste into config/room.toml (edit the ids: the room expects \"lamp\" and \"fan\")"]
    for i, b in enumerate(wiz):
        did = "lamp" if i == 0 else f"lamp{i + 1}"
        lines += [f"[devices.{did}]", 'kind = "light"', 'driver = "wiz"', f'host = "{b["host"]}"',
                  f"# mac {b.get('mac') or '?'}  module {b.get('moduleName') or '?'}  fw {b.get('fwVersion') or '?'}", ""]
    for i, p in enumerate(kasa):
        did = "fan" if i == 0 else f"plug{i + 1}"
        note = f"# {p.get('model') or '?'} '{p.get('alias') or ''}' mac {p.get('mac') or '?'}"
        if p.get("needs_credentials"):
            note += "  NEEDS KASA_USERNAME/KASA_PASSWORD (KLAP/SMART protocol)"
        elif p.get("needs_credentials") is None:
            note += "  (unsupported by python-kasa: " + str(p.get("detail", ""))[:60] + ")"
        lines += [f"[devices.{did}]", 'kind = "plug"', 'driver = "kasa"', f'host = "{p["host"]}"', note, ""]
    if not wiz and not kasa:
        lines.append("# nothing answered. Same Wi-Fi as the devices? WiZ: UDP 38899; Kasa: UDP 9999/20002.")
    return "\n".join(lines)


async def cmd_discover(a: argparse.Namespace) -> int:
    wiz_task = discover_wiz(broadcast=a.broadcast, port=a.wiz_port, timeout=a.timeout) if not a.no_wiz else None
    kasa_task = discover_kasa(broadcast=a.broadcast, timeout=a.timeout) if not a.no_kasa else None
    results = await asyncio.gather(*(t for t in (wiz_task, kasa_task) if t is not None))
    wiz = results[0] if wiz_task is not None else []
    kasa = results[-1] if kasa_task is not None else []
    if a.json:
        print(json.dumps({"wiz": wiz, "kasa": kasa}, indent=2))
    else:
        print(f"WiZ bulbs: {len(wiz)}   Kasa devices: {len(kasa)}\n")
        print(_room_toml_block(wiz, kasa))
    return 0


def _fmt_state(device_id: str, dev: Any, st: dict[str, Any]) -> str:
    ok = "ok  " if st.get("healthy") else "DOWN"
    bits = [f"{device_id:<8} {dev.kind:<6} {getattr(dev, 'driver', '?'):<5} {ok}"]
    if dev.kind == "light":
        bits.append(f"on={st.get('on')} brightness={st.get('brightness')} scene={st.get('scene')} "
                    f"color={st.get('color')} temp={st.get('temp')}")
    else:
        bits.append(f"on={st.get('on')}")
    if st.get("host"):
        bits.append(f"host={st['host']}")
    if st.get("error"):
        bits.append(f"error: {st['error']}")
    return "  ".join(bits)


async def cmd_list(a: argparse.Namespace) -> int:
    devices = load_devices(load_room_config(a.config))
    try:
        states = await probe_devices(devices, timeout=a.timeout + 3.0)
    finally:
        await close_devices(devices)
    if a.json:
        print(json.dumps(states, indent=2))
    else:
        for did, dev in devices.items():
            print(_fmt_state(did, dev, states[did]))
    return 0 if all(s.get("healthy") for s in states.values()) else 1


def parse_set_args(command: str, raw: list[str]) -> dict[str, Any]:
    """['off_record'] with command scene -> {"scene": "off_record"}; ['brightness=40', 'color=#ff0000'] or
    ['40', '#ff0000'] with command set -> {"brightness": 40, "color": "#ff0000"}."""
    args: dict[str, Any] = {}
    for item in raw:
        if "=" in item:
            k, v = item.split("=", 1)
            args[k.strip()] = _coerce(v.strip())
        elif command == "scene":
            args["scene"] = item
        elif command == "set":
            if item.startswith("#"):
                args["color"] = item
            else:
                args["brightness"] = _coerce(item)
        else:
            raise SystemExit(f"unexpected argument {item!r} for command {command!r}")
    return args


def _coerce(v: str) -> Any:
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


async def cmd_set(a: argparse.Namespace) -> int:
    devices = load_devices(load_room_config(a.config))
    if a.device not in devices:
        print(f"no device {a.device!r}; have: {', '.join(devices)}", file=sys.stderr)
        return 2
    dev = devices[a.device]
    args = parse_set_args(a.command, a.args)
    try:
        st = await dev.apply(a.command, args)
    finally:
        await close_devices(devices)
    if a.json:
        print(json.dumps(st, indent=2))
    else:
        print(_fmt_state(a.device, dev, st))
    return 0 if st.get("healthy") and not st.get("error") else 1


async def cmd_scenes(a: argparse.Namespace) -> int:
    if a.json:
        print(json.dumps(SCENES, indent=2))
        return 0
    for name, pilot in SCENES.items():
        print(f"{name:<11} {json.dumps(pilot)}")
    return 0


async def cmd_fake_bulb(a: argparse.Namespace) -> int:
    bulb = await FakeWizBulb.start(a.host, a.port)
    print(f"fake WiZ bulb on {a.host}:{bulb.port}  (room.toml: [devices.lamp] kind=\"light\" driver=\"wiz\" "
          f"host=\"{a.host}\" port={bulb.port}). Ctrl-C to stop.", flush=True)
    last = 0
    try:
        while True:
            await asyncio.sleep(0.2)
            if len(bulb.requests) > last:
                for req in bulb.requests[last:]:
                    print(f"<< {json.dumps(req)}   -> state {json.dumps(bulb.state)}", flush=True)
                last = len(bulb.requests)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await bulb.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m unspoken.devices", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="room.toml path (default config/room.toml or $UNSPOKEN_ROOM_CONFIG)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="find WiZ bulbs and Kasa plugs on this network")
    d.add_argument("--timeout", type=float, default=3.0)
    d.add_argument("--broadcast", default="255.255.255.255", help="broadcast (or one unicast) address")
    d.add_argument("--wiz-port", type=int, default=WIZ_PORT)
    d.add_argument("--no-kasa", action="store_true")
    d.add_argument("--no-wiz", action="store_true")
    d.set_defaults(func=cmd_discover)

    ls = sub.add_parser("list", help="devices from room.toml with live state and health")
    ls.add_argument("--timeout", type=float, default=3.0)
    ls.set_defaults(func=cmd_list)

    s = sub.add_parser("set", help="send one command: set lamp scene off_record | set fan off")
    s.add_argument("device")
    s.add_argument("command", help="light: on off scene set; plug: on off")
    s.add_argument("args", nargs="*", help="scene name, or key=value (brightness=40 color=#ff8800)")
    s.set_defaults(func=cmd_set)

    sc = sub.add_parser("scenes", help="print the scene table")
    sc.set_defaults(func=cmd_scenes)

    fb = sub.add_parser("fake-bulb", help="run a fake WiZ bulb on 127.0.0.1")
    fb.add_argument("--host", default="127.0.0.1")
    fb.add_argument("--port", type=int, default=WIZ_PORT)
    fb.set_defaults(func=cmd_fake_bulb)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        return asyncio.run(a.func(a))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
