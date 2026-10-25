"""Send one message through the configured notifier (verify.sh's test post).

    UNSPOKEN_NOTIFIER=openclaw python -m unspoken.bridge.send --channel notes --text "bridge check" --health
"""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from .config import BridgeConfigError, load_bridge_config
from .notifiers import make_notifier


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Send one message through the configured LocoCisco notifier")
    p.add_argument("--channel", default="notes", help="logical key from config/bridge.toml or a Discord channel id")
    p.add_argument("--text", required=True)
    p.add_argument("--kind", default="note")
    p.add_argument("--thread", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--health", action="store_true", help="also report healthy()")
    a = p.parse_args(argv)
    meter: list[tuple[str, str, int]] = []
    try:
        notifier = make_notifier(load_bridge_config(a.config), lambda d, k, n: meter.append((d, k, n)))
    except BridgeConfigError as exc:
        print(f"send: {exc}", flush=True)
        return 2

    async def go() -> dict[str, Any]:
        healthy = await notifier.healthy() if a.health else None
        ref = await notifier.post(a.channel, a.text, kind=a.kind, thread=a.thread)
        return {"transport": notifier.transport, "healthy": healthy,
                "message": {"channel": ref.channel, "id": ref.id, "thread": ref.thread},
                "egress": [{"dest": d, "kind": k, "bytes": n} for d, k, n in meter]}

    out = asyncio.run(go())
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["message"]["id"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
