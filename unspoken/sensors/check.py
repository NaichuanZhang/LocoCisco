"""Day-of check for the sensors: devices seen on this box and whether each in-process model loads.

    python -m unspoken.sensors.check            # devices + model health (in-process providers)
    python -m unspoken.sensors.check --no-models
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .audio import default_audio_backend, list_input_devices
from .config import config_path, load_sensors_config, seat_zones


def video_devices() -> list[dict[str, str]]:
    out = []
    base = Path("/sys/class/video4linux")
    if base.exists():
        for dev in sorted(base.iterdir()):
            try:
                out.append({"device": f"/dev/{dev.name}", "name": (dev / "name").read_text().strip()})
            except OSError:
                pass
    return out


async def run(args: argparse.Namespace) -> int:
    cfg = load_sensors_config(args.config)
    print(f"config: {args.config or config_path()}")
    print(f"audio backend: {default_audio_backend()}  configured device: {cfg['audio']['device']!r} "
          f"(arecord fallback: {cfg['audio']['alsa_device']})")
    print(f"audio inputs (sounddevice): {list_input_devices() or 'none visible (no PortAudio or no mic)'}")
    print(f"video devices: {video_devices() or 'none'}  configured: {cfg['video']['device']!r}")
    print(f"seats: {list(seat_zones(cfg))}")
    rc = 0
    if not args.no_models:
        from .local_registry import SensorsRegistry

        reg = SensorsRegistry()
        for h in await reg.check():
            print(f"{'ok ' if h.ok else 'BAD'} {h.role:12s} {h.label:40s} {h.latency_ms or '':>8} ms  {h.detail}")
            rc |= 0 if h.ok else 1
    if args.json:
        print(json.dumps(cfg, default=str))
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-models", action="store_true")
    ap.add_argument("--json", action="store_true")
    return asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
