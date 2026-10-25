"""Replay a recording through the real sensors pipeline and print the bus events as JSON lines.

    python -m unspoken.sensors.replay --audio x.wav [--video y.mp4] [--profile dev] [--fast] [--seats]

--fast runs as fast as the models go (video paced by the audio position); default is real time.
--profile uses unspoken.models.load_registry(profile) when the models package is present; --local (or
when it is not) uses the in-process providers directly (unspoken.sensors.local_registry).
Output: one JSON object per line on stdout ({"topic", "ts", "data"}); a summary on stderr.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Any

from ..core.bus import Bus
from ..core.events import Event
from ..meeting.contracts import MT
from .config import load_sensors_config
from .pipeline import start_sensors


def make_registry(profile: str | None, local: bool, overrides: dict[str, str] | None = None) -> Any:
    if not local:
        try:
            from ..models import load_registry  # type: ignore[import-not-found]  # models workstream
            return load_registry(profile, overrides=overrides)
        except ImportError:
            print("unspoken.models not available; using the in-process sensor providers", file=sys.stderr)
    from .local_registry import SensorsRegistry
    return SensorsRegistry(overrides=overrides)


async def run(args: argparse.Namespace) -> int:
    config = load_sensors_config(args.config)
    config["audio"]["path"] = args.audio
    if args.video:
        config["video"]["path"] = args.video
    config["replay"] = {"realtime": not args.fast, "speed": args.speed}
    if args.no_seats:
        config["seats"] = {}
    registry = make_registry(args.profile, args.local, dict(kv.split("=", 1) for kv in args.set))

    base = time.monotonic()
    clock = lambda: time.monotonic() - base  # noqa: E731  meeting clock starts at 0
    bus = Bus()
    topics = {MT.SEGMENT, MT.SPEAKERS} if not args.all else None

    async def printer(ev: Event) -> None:
        if topics is None or ev.topic in topics:
            sys.stdout.write(json.dumps({"topic": ev.topic, "ts": round(ev.ts, 3), "data": ev.data},
                                        ensure_ascii=False) + "\n")
            sys.stdout.flush()

    bus.subscribe("*", printer, name="replay.printer")
    t0 = time.perf_counter()
    handle = await start_sensors(bus, registry, config, clock)
    print(f"models: {handle.models}", file=sys.stderr)
    try:
        await handle.wait(timeout=args.timeout)
        await bus.drain()
    except asyncio.TimeoutError:
        print("timeout: replay did not finish", file=sys.stderr)
        return 2
    finally:
        await handle.stop()
        aclose = getattr(registry, "aclose", None)
        if aclose:
            await aclose()
    s = handle.stats.summary()
    s["wall_s"] = round(time.perf_counter() - t0, 2)
    print(f"done: {json.dumps(s)}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="WAV (16 kHz mono PCM16 preferred; others decoded via PyAV)")
    ap.add_argument("--video", help="MP4 (or anything OpenCV opens)")
    ap.add_argument("--profile", default=None, help="registry profile (dev | gb10 | fake)")
    ap.add_argument("--local", action="store_true", help="force the in-process providers (no unspoken.models)")
    ap.add_argument("--config", default=None, help="config/sensors.toml path")
    ap.add_argument("--fast", action="store_true", help="do not pace to real time")
    ap.add_argument("--speed", type=float, default=1.0, help="real-time speed factor")
    ap.add_argument("--all", action="store_true", help="print every bus topic, not only segments/speakers")
    ap.add_argument("--no-seats", action="store_true")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--set", action="append", default=[], metavar="ROLE.KEY=VALUE",
                    help='registry override, e.g. --set asr.model=Systran/faster-whisper-small.en')
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
