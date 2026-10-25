"""Play library scenarios through the FULL runtime (real sim, real agents, real deciders).

    python -m homedemo.play [--mode cascade|2b|12b|fake] [--escalate bedrock|12b|llm] [--profile dev-local]
                            [--decider-2b-url URL] [--speed 4] [--tag hero] [ids...]

Prints one line per scenario, every decider decision (who, route, action, confidence, latency, tokens)
and the failing expectations, and writes a JSON summary when --out is given. Needs the profile's decider
servers (and Bedrock credentials for --escalate bedrock, or the profile's llm server for --escalate llm)
for modes other than fake.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from .cli import add_decider_args, settings_from_args
from .contracts import Settings, T
from .runtime import Runtime

DECIDER_BY = ("decider-2b", "decider-12b", "bedrock", "llm")


def decision_row(d: dict) -> dict:
    tokens = d.get("tokens")
    return {"by": d["by"], "mode": d["mode"], "route": d.get("route"), "action": d["action"],
            "confidence": round(float(d["confidence"]), 4), "latency_ms": round(float(d.get("latency_ms", 0))),
            "tokens": (tokens["input"] + tokens["output"]) if tokens else None, "kind": d.get("kind"),
            "tool_calls": [c["name"] for c in d.get("tool_calls") or []]}


async def play(ids: list[str], settings: Settings, tag: str | None, out: str | None) -> int:
    mode, speed = settings.decider_mode, settings.speed
    rt = Runtime(settings, tick_seconds=0)
    await rt.start(ticking=False)
    chosen = [s for s in rt.scenarios() if (not ids or s["id"] in ids) and (not tag or tag in s["tags"])]
    results = []
    for sc in chosen:
        t0 = time.time()
        mark = len(rt.bus.history)
        await rt.run_scenario(sc["id"])
        await rt._scenario_task
        events = list(rt.bus.history)[mark:]
        status = [e.data for e in events if e.topic == T.SCENARIO]
        fin = next((s for s in reversed(status) if s.get("state") in ("finished", "error", "stopped")), {})
        fails = [s.get("text", "") for s in status if s.get("state") == "expect_fail"]
        decider = [e.data for e in events if e.topic == T.PROPOSAL and e.data.get("by") in DECIDER_BY]
        warns = [e.data.get("text", "") for e in events if e.topic == T.LOG and e.data.get("level") == "warn"]
        rows = [decision_row(d) for d in decider]
        results.append({"id": sc["id"], "passed": fin.get("passed", 0), "failed": fin.get("failed", 0),
                        "state": fin.get("state"), "fails": fails, "seconds": round(time.time() - t0, 1),
                        "decider_calls": len(decider), "decider": rows, "warnings": warns})
        mark_ok = "PASS" if fin.get("state") == "finished" and not fin.get("failed") else "FAIL"
        print(f"{mark_ok} {sc['id']:<34} {fin.get('passed', 0)}/{fin.get('passed', 0) + fin.get('failed', 0)}"
              f"  {results[-1]['seconds']:>5}s")
        for r in rows:
            tok = f" tokens={r['tokens']}" if r["tokens"] is not None else ""
            calls = f" tools={r['tool_calls']}" if r["tool_calls"] else ""
            print(f"      {r['by']:<11} {r['mode']:<6} {r['route']:<7} {r['kind']:<6} {r['action']:<13} "
                  f"{r['confidence']:.2f}  {r['latency_ms']:>6} ms{tok}{calls}")
        for w in warns:
            print(f"      ! {w}")
        for f in fails:
            print(f"      - {f}")
    errors = rt.bus.errors
    await rt.close()
    if errors:
        print("bus errors:", [(n, repr(e)) for n, _, e in errors])
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"mode": mode, "escalate_to": settings.escalate_to, "speed": speed, "results": results},
                      f, indent=1, ensure_ascii=False)
    bad = sum(1 for r in results if r["failed"] or r["state"] != "finished")
    print(f"\n{len(results) - bad}/{len(results)} scenarios passed (mode={mode})")
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--mode", default="cascade", choices=["cascade", "2b", "12b", "fake"])
    add_decider_args(ap)
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--tag")
    ap.add_argument("--out")
    a = ap.parse_args()
    raise SystemExit(asyncio.run(play(a.ids, settings_from_args(a, speed=a.speed), a.tag, a.out)))


if __name__ == "__main__":
    main()
