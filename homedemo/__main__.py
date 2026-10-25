"""Run the demo: python -m homedemo [--port 8780] [--mode cascade|2b|12b|fake] [--escalate bedrock|12b|llm]
                        [--profile dev|dev-local|gb10|fake] [--decider-2b-url http://127.0.0.1:8765]
                        [--trace traces/run.jsonl]

The deciders come from the models registry (config/models.toml, --profile / UNSPOKEN_PROFILE). In cascade
mode the Decider 2B answers first; when it is unsure, --escalate picks who decides: Claude Haiku 4.5 on
Amazon Bedrock (default; AWS profile from Settings.bedrock_profile), the local 12B, or "llm" = the profile's
llm role (a local Qwen on llama.cpp / vLLM in dev-local and gb10). The 12B server may be stopped: it is only
health-checked when it is needed as a fallback.

Binds to 127.0.0.1 only (the UI has no auth by design). View it from a laptop with
`ssh -L 8780:127.0.0.1:8780 <host>` and open http://127.0.0.1:8780.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os

from .cli import add_decider_args, settings_from_args
from .contracts import Settings
from .runtime import Runtime
from .web.server import create_app, require_loopback


async def main_async(host: str, port: int, settings: Settings, trace: str | None, noise: float) -> None:
    require_loopback(host)
    import uvicorn

    if trace:
        os.makedirs(os.path.dirname(trace) or ".", exist_ok=True)
    rt = Runtime(settings, noise=noise, trace_path=trace)
    # The UI subscribes to the bus BEFORE the agents do, so Runtime.reset() can rebuild the agents
    # without dropping the UI stream.
    app = create_app(rt)
    await rt.start()
    health = {k: await d.healthy() for k, d in rt.deciders.items()}
    logging.getLogger("homedemo").warning("deciders healthy: %s; mode=%s escalate=%s; http://%s:%d", health,
                                          settings.decider_mode, settings.escalate_to, host, port)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    try:
        await server.serve()
    finally:
        await rt.close()


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m homedemo")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--mode", default="cascade", choices=["cascade", "2b", "12b", "fake"])
    add_decider_args(ap)
    ap.add_argument("--trace", default=None, help="write every bus event as JSONL")
    ap.add_argument("--noise", type=float, default=0.3)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(main_async(a.host, a.port, settings_from_args(a), a.trace, a.noise))


if __name__ == "__main__":
    main()
