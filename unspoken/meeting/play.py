"""Play library scenarios through the FULL runtime and print pass/fail per expect.

    python -m unspoken.meeting.play [ids...] [--profile fake|dev|dev-local|gb10] [--speed N] [--tag demo]
                                    [--no-escalate] [--out results.json] [--real-io]

Prints one line per scenario, every decider / LLM decision (by, route, action, confidence, latency, tokens)
and the failing expectations. Fake devices and the console notifier unless --real-io.
"""
from __future__ import annotations

from typing import Any

import argparse
import asyncio
import json
import os
import time

from .contracts import MT
from .runtime import MeetingRuntime

MODEL_BY = ("decider", "decider_big", "llm")


def decision_row(d: dict) -> dict:
    tokens = d.get("tokens")
    return {"by": d["by"], "mode": d["mode"], "route": d.get("route"), "action": d["action"], "kind": d.get("kind"),
            "confidence": round(float(d["confidence"]), 4), "latency_ms": round(float(d.get("latency_ms", 0))),
            "tokens": (tokens["input"] + tokens["output"]) if tokens else None,
            "tool_calls": [c["name"] for c in d.get("tool_calls") or []], "owner": (d.get("args") or {}).get("owner"),
            "args": d.get("args") or {}, "calls": d.get("tool_calls") or []}


async def play(ids: list[str], *, profile: str, speed: float, tag: str | None, out: str | None, escalate: bool,
               fake_io: bool) -> int:
    rt = MeetingRuntime(profile=profile, mode="sim", fake_io=fake_io, echo=False, tick_seconds=0)
    rt.settings.speed = speed
    rt.settings.escalate = escalate
    # Read what was delivered, not bus.history: the history drops off-record events by design. Subscribing before
    # start() keeps this listener across the per-scenario agent rebuilds.
    stream: list[Any] = []

    async def record(ev: Any) -> None:
        stream.append(ev)

    rt.bus.subscribe("*", record, name="play.stream")
    await rt.start(ticking=False)
    for h in rt.models_health:
        if h.get("role") in ("decider", "decider_big", "llm"):
            print(f"model {h.get('role'):<12} {str(h.get('label')):<34} ok={h.get('ok')} {h.get('detail') or ''}")
    chosen = [s for s in rt.scenarios() if (not ids or s["id"] in ids) and (not tag or tag in s["tags"])]
    results = []
    for sc in chosen:
        t0 = time.time()
        mark = len(stream)
        await rt.run_scenario(sc["id"])
        await rt._scenario_task
        events = stream[mark:]
        status = [e.data for e in events if e.topic == MT.SCENARIO]
        fin = next((s for s in reversed(status) if s.get("state") in ("finished", "error", "stopped")), {})
        checks = [(s.get("state") == "expect_pass", s.get("text", "")) for s in status if s.get("state") in ("expect_pass", "expect_fail")]
        decisions = [e.data for e in events if e.topic == MT.PROPOSAL and e.data.get("by") in MODEL_BY]
        warns = [e.data.get("text", "") for e in events if e.topic == MT.LOG and e.data.get("level") == "warn"]
        rows = [decision_row(d) for d in decisions]
        results.append({"id": sc["id"], "passed": fin.get("passed", 0), "failed": fin.get("failed", 0), "state": fin.get("state"),
                        "checks": checks, "seconds": round(time.time() - t0, 1), "decisions": rows, "warnings": warns})
        ok = "PASS" if fin.get("state") == "finished" and not fin.get("failed") else "FAIL"
        print(f"{ok} {sc['id']:<34} {fin.get('passed', 0)}/{fin.get('passed', 0) + fin.get('failed', 0)}  {results[-1]['seconds']:>6}s")
        for passed, text in checks:
            print(f"      {'ok ' if passed else 'XX '} {text}")
        for r in rows:
            tok = f" tokens={r['tokens']}" if r["tokens"] is not None else ""
            calls = f" tools={r['tool_calls']}" if r["tool_calls"] else ""
            print(f"      {r['by']:<8} {r['mode']:<6} {r['route']:<13} {r['kind'] or '':<10} {r['action']:<15} "
                  f"{r['confidence']:.2f}  {r['latency_ms']:>6} ms{tok}{calls}")
        for w in warns:
            print(f"      ! {w}")
    errors = rt.bus.errors
    await rt.close()
    if errors:
        print("bus errors:", [(n, repr(e)) for n, _, e in errors])
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"profile": profile, "speed": speed, "escalate": escalate, "results": results}, f, indent=1, ensure_ascii=False)
    bad = sum(1 for r in results if r["failed"] or r["state"] != "finished")
    print(f"\n{len(results) - bad}/{len(results)} scenarios passed (profile={profile}, escalate={escalate})")
    return 1 if bad or errors else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--profile", default=os.environ.get("UNSPOKEN_PROFILE", "fake"))
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--tag")
    ap.add_argument("--out")
    ap.add_argument("--no-escalate", action="store_true")
    ap.add_argument("--real-io", action="store_true", help="real devices / notifier when their packages import")
    a = ap.parse_args()
    raise SystemExit(asyncio.run(play(a.ids, profile=a.profile, speed=a.speed, tag=a.tag, out=a.out,
                                      escalate=not a.no_escalate, fake_io=not a.real_io)))


if __name__ == "__main__":
    main()
