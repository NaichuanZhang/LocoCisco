"""python -m unspoken.meeting [--profile P] [--mode sim|live|replay] [--port 8781] [--replay trace.jsonl]

Starts the meeting brain and, when unspoken.meeting.web exists (screen workstream), serves its FastAPI app on
127.0.0.1:<port>; otherwise runs headless and prints the trace. Never binds to anything but loopback.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import sys

from .contracts import MT
from .runtime import MeetingRuntime


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m unspoken.meeting")
    ap.add_argument("--profile", default=os.environ.get("UNSPOKEN_PROFILE", "dev"))
    ap.add_argument("--mode", default="sim", choices=["sim", "live", "replay"])
    ap.add_argument("--port", type=int, default=None, help="room screen port (default: [screen] port in config/meeting.toml)")
    ap.add_argument("--replay", help="--mode replay: a .wav/.mp4 recording (through the real sensors) or a bus trace .jsonl")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-escalate", action="store_true", help="decider only, never the LLM")
    ap.add_argument("--fake-io", action="store_true", help="fake devices and console notifier even if real ones import")
    ap.add_argument("--trace", help="write every bus event to this JSONL file")
    ap.add_argument("--scenario", help="sim: run this scenario at start (e.g. demo_three_minutes)")
    ap.add_argument("--headless", action="store_true", help="do not serve the room screen even if it exists")
    ap.add_argument("--log", default="INFO")
    return ap.parse_args(argv)


def start_mcp_server(rt: MeetingRuntime) -> "asyncio.Task | None":
    """The room's MCP server (unspoken.bridge.mcp_server) in this process when UNSPOKEN_MCP_TOKEN is set:
    host/port/path from config/bridge.toml [mcp] (UNSPOKEN_MCP_HOST / UNSPOKEN_MCP_PORT override). Loopback only."""
    token = os.environ.get("UNSPOKEN_MCP_TOKEN", "").strip()
    if not token:
        return None
    try:
        from unspoken.bridge import load_bridge_config
        from unspoken.bridge import mcp_server
        from unspoken.bridge.config import mcp_settings
    except Exception as exc:
        print(f"MCP server not available ({type(exc).__name__}: {exc})")
        return None
    try:
        cfg = mcp_settings(load_bridge_config())
    except Exception as exc:
        print(f"bridge config problem, MCP server not started: {exc}")
        return None
    task = asyncio.create_task(mcp_server.serve(rt, cfg["host"], int(cfg["port"]), token, path=cfg["path"],
                                                verdict_wait=float(cfg["verdict_wait"]), stateless=bool(cfg["stateless"]),
                                                json_response=bool(cfg["json_response"])), name="meeting.mcp")
    print(f"MCP server: http://{cfg['host']}:{cfg['port']}{cfg['path']} (bearer from UNSPOKEN_MCP_TOKEN)")
    return task


async def main_async(a: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, a.log.upper(), logging.INFO), format="%(levelname)s %(name)s: %(message)s")
    rt = MeetingRuntime(profile=a.profile, mode=a.mode, fake_io=a.fake_io, trace_path=a.trace, replay_path=a.replay)
    rt.settings.speed = a.speed
    if a.no_escalate:
        rt.settings.escalate = False

    async def printer(ev) -> None:
        if ev.topic in (MT.LOG, MT.PROPOSAL, MT.VERDICT, MT.PRIVACY, MT.ITEM, MT.COMMAND, MT.SCENARIO):
            d = ev.data
            if ev.topic == MT.PROPOSAL:
                line = f"{d.get('by')} {d.get('mode')} {d.get('route')} {d.get('action')} {d.get('confidence'):.2f} {d.get('latency_ms', 0):.0f}ms: {d.get('reason')}"
            elif ev.topic == MT.SCENARIO:
                line = f"[{d.get('state')}] {d.get('text', '')}"
            else:
                line = " ".join(f"{k}={v}" for k, v in d.items() if k not in ("request", "answers", "probabilities", "kind_probabilities"))
            print(f"{ev.ts:8.1f} {ev.topic:<18} {line}", flush=True)

    rt.bus.subscribe("*", printer, name="printer")
    screen_cfg = rt.config.get("screen") or {}
    host, port = str(screen_cfg.get("host") or "127.0.0.1"), int(a.port or screen_cfg.get("port") or 8780)
    app = web = None
    if not a.headless:
        try:
            web = importlib.import_module("unspoken.meeting.web")
            app = web.create_app(rt)          # BEFORE start(): the page's stream hub must see every bus event
        except Exception as exc:    # the screen workstream's package may not exist yet
            print(f"room screen not available ({type(exc).__name__}: {exc}); running headless")
    await rt.start()
    for h in rt.models_health:
        print(f"model {h.get('role'):<12} {str(h.get('label')):<34} ok={h.get('ok')} {h.get('detail') or ''}")
    mcp_task = start_mcp_server(rt)
    if a.scenario:
        await rt.run_scenario(a.scenario)
    if app is not None:
        print(f"room screen: http://{host}:{port}/  (mode {a.mode}, profile {a.profile})")
        try:
            await web.serve_async(rt, host, port, app=app)
        finally:
            if mcp_task is not None:
                mcp_task.cancel()
            await rt.close()
        return 0
    print(f"headless (mode {a.mode}, profile {a.profile}); Ctrl-C to stop")
    try:
        if rt._scenario_task is not None:
            await rt._scenario_task
            if a.mode == "sim":
                res = rt._runner.last_result
                print(f"scenario {res.scenario_id}: {res.passed} passed, {res.failed} failed ({res.state})")
                return 0 if res.ok else 1
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        if mcp_task is not None:
            mcp_task.cancel()
        await rt.close()
    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(main_async(parse_args())))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
