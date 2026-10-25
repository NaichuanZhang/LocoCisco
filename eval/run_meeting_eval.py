"""Run eval/meeting_cases.json through build_meeting_query + a decider and report accuracy, coverage and latency.

    .venv/bin/python eval/run_meeting_eval.py --model 2b --version v2
    .venv/bin/python eval/run_meeting_eval.py --model 12b --version v2 --note "final"
    .venv/bin/python eval/run_meeting_eval.py --model keyword             # the plain-code baseline, no server
    .venv/bin/python eval/run_meeting_eval.py --summarize eval/runs/<file>.json

Writes eval/runs/<stamp>-meeting-<version>-<model>.json (gitignored) and appends one row to
eval/MEETING_RESULTS.md. `--set field=value` overrides questions.Spec fields for experiments.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unspoken.core.decider import HttpDecider  # noqa: E402
from unspoken.meeting.fakes import KeywordDecider  # noqa: E402
from unspoken.meeting.questions import build_meeting_query  # noqa: E402

EVAL_DIR = ROOT / "eval"
RUNS_DIR = EVAL_DIR / "runs"
URLS = {"2b": "http://127.0.0.1:8765", "12b": "http://127.0.0.1:8766"}
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
TODAY = dt.date(2026, 10, 26)      # a Monday, so "Friday" / "next week" are stable across runs


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    return json.loads((path or EVAL_DIR / "meeting_cases.json").read_text(encoding="utf-8"))["cases"]


def judge(case: dict[str, Any], it: Any) -> dict[str, Any]:
    """right = action in ok; owner_right (tasks); args_right (room actions / due presence)."""
    right = it.action in case["ok"]
    out: dict[str, Any] = {"right": right}
    if case["expected"] == "create_task":
        out["owner_judged"] = True
        out["owner_right"] = (it.args.get("owner") == case.get("owner")) if it.action == "create_task" else False
        want_due = (case.get("args") or {}).get("due")
        if want_due is not None:
            out["args_judged"] = True
            out["args_right"] = it.action == "create_task" and bool(it.args.get("due")) == bool(want_due)
    elif case["expected"] in ("set_fan", "set_light"):
        out["args_judged"] = True
        want = case.get("args") or {}
        out["args_right"] = it.action == case["expected"] and all(it.args.get(k) == v for k, v in want.items())
    return out


async def run_cases(cases: list[dict[str, Any]], decider: Any, version: str, overrides: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for i, case in enumerate(cases, 1):
        q = build_meeting_query(case["utt"], case["names"], version, today=TODAY, **overrides)
        t0 = time.perf_counter()
        answers = await decider.ask(q.state, q.questions)
        ms = (time.perf_counter() - t0) * 1000
        it = q.interpret(answers)
        row = {"id": case["id"], "category": case["category"], "tags": case.get("tags", []), "expected": case["expected"],
               "ok": case["ok"], "pred": it.action, "args": it.args, "kind": it.kind,
               "confidence": round(it.confidence, 4), "probabilities": {k: round(v, 4) for k, v in it.probabilities.items() if v > 0.001},
               "kind_probabilities": {k: round(v, 4) for k, v in it.kind_probabilities.items()},
               "owner": it.owner, "owner_confidence": round(it.owner_confidence, 3), "latency_ms": round(ms, 1),
               "state": q.state, "detail": it.detail, **judge(case, it)}
        rows.append(row)
        mark = "ok " if row["right"] else "XX "
        extra = ""
        if row.get("owner_judged"):
            extra += f" owner={it.owner}({'ok' if row['owner_right'] else 'XX'})"
        if row.get("args_judged"):
            extra += f" args={'ok' if row['args_right'] else 'XX'}"
        print(f"{mark}{case['id']:4} {case['category']:10} conf={it.confidence:.2f} pred={it.action:16} exp={'/'.join(case['ok']):18}{extra} {ms:6.0f}ms [{i}/{len(cases)}]", flush=True)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    acc = sum(r["right"] for r in rows) / n
    by_t: dict[str, Any] = {}
    for t in THRESHOLDS:
        sure = [r for r in rows if r["confidence"] >= t]
        acting = [r for r in sure if r["pred"] != "none"]
        should_act = [r for r in rows if r["expected"] != "none"]
        by_t[str(t)] = {
            "coverage": round(len(sure) / n, 3),
            "accuracy": round(sum(r["right"] for r in sure) / len(sure), 3) if sure else None,
            "act_n": len(acting),
            "act_precision": round(sum(r["right"] for r in acting) / len(acting), 3) if acting else None,
            "act_recall": round(sum(1 for r in acting if r["right"]) / len(should_act), 3) if should_act else None,
        }
    lat = sorted(r["latency_ms"] for r in rows)
    owners = [r for r in rows if r.get("owner_judged")]
    args = [r for r in rows if r.get("args_judged")]
    out: dict[str, Any] = {
        "n": n, "accuracy": round(acc, 3), "by_threshold": by_t,
        "owner_accuracy": round(sum(r["owner_right"] for r in owners) / len(owners), 3) if owners else None, "owner_n": len(owners),
        "args_accuracy": round(sum(r["args_right"] for r in args) / len(args), 3) if args else None, "args_n": len(args),
        "latency_ms": {"median": round(statistics.median(lat), 0), "p90": round(lat[min(n - 1, int(0.9 * n))], 0), "mean": round(sum(lat) / n, 0)},
        "by_category": {}, "by_tag": {},
        "wrong": [{"id": r["id"], "pred": r["pred"], "ok": r["ok"], "confidence": r["confidence"], "kind": r["kind"]} for r in rows if not r["right"]],
    }
    for r in rows:
        c = out["by_category"].setdefault(r["category"], [0, 0])
        c[0] += r["right"]
        c[1] += 1
        for tag in r.get("tags", []):
            t = out["by_tag"].setdefault(tag, [0, 0])
            t[0] += r["right"]
            t[1] += 1
    bins: dict[str, list[int]] = {}
    for r in rows:
        lo = min(int(r["confidence"] * 10) / 10, 0.9)
        b = bins.setdefault(f"{lo:.1f}-{lo + 0.1:.1f}", [0, 0])
        b[0] += r["right"]
        b[1] += 1
    out["reliability"] = {k: {"n": v[1], "accuracy": round(v[0] / v[1], 3)} for k, v in sorted(bins.items())}
    out["brier"] = round(sum((r["confidence"] - (1.0 if r["right"] else 0.0)) ** 2 for r in rows) / n, 3)
    return out


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def results_row(stamp: str, version: str, model: str, s: dict[str, Any], note: str) -> str:
    t8, t9 = s["by_threshold"]["0.8"], s["by_threshold"]["0.9"]
    return (f"| {stamp} | {version} | {model} | {s['n']} | {pct(s['accuracy'])} | {pct(t8['accuracy'])} ({pct(t8['coverage'])}) | "
            f"{pct(t9['accuracy'])} ({pct(t9['coverage'])}) | {pct(t8['act_precision'])} / {pct(t8['act_recall'])} | "
            f"{pct(s['owner_accuracy'])} | {pct(s['args_accuracy'])} | {s['brier']} | {s['latency_ms']['median']:.0f} / {s['latency_ms']['p90']:.0f} | {note} |")


RESULTS_HEADER = (
    "# Meeting decider eval runs\n\nOne row per run, appended by eval/run_meeting_eval.py (cases: eval/meeting_cases.json). "
    "acc = action in the case's accepted answers; acc@t (cov) = accuracy among cases with confidence >= t and the share of "
    "cases reaching t; act prec / recall @0.8 = of the non-none predictions at >= 0.8, the share that are right, and the share "
    "of cases that needed an action which got the right one at >= 0.8; owner = task owner right (tasks); args = fan/light "
    "argument or due-date presence right.\n\n"
    "| stamp | version | model | n | acc | acc@0.8 (cov) | acc@0.9 (cov) | act prec / recall @0.8 | owner | args | Brier | median / p90 ms | note |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")


def append_results(line: str) -> None:
    path = EVAL_DIR / "MEETING_RESULTS.md"
    if not path.exists():
        path.write_text(RESULTS_HEADER, encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_overrides(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        k, _, v = item.partition("=")
        out[k] = {"true": True, "false": False, "none": None}.get(v.lower(), v)
    return out


def print_summary(name: str, s: dict[str, Any]) -> None:
    print(f"\n== {name}: n={s['n']} acc={s['accuracy']:.3f} owner={s['owner_accuracy']} (n={s['owner_n']}) args={s['args_accuracy']} "
          f"(n={s['args_n']}) brier={s['brier']} latency median={s['latency_ms']['median']:.0f}ms p90={s['latency_ms']['p90']:.0f}ms")
    for t, v in s["by_threshold"].items():
        print(f"   conf>={t}: coverage={v['coverage']:.2f} acc={v['accuracy']} act_n={v['act_n']} act_precision={v['act_precision']} act_recall={v['act_recall']}")
    print("   by category:", {k: f"{v[0]}/{v[1]}" for k, v in s["by_category"].items()})
    print("   by tag:", {k: f"{v[0]}/{v[1]}" for k, v in s["by_tag"].items()})
    print("   wrong:", [(w["id"], w["pred"], w["confidence"]) for w in s["wrong"]])


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(URLS) + ["keyword"], default="2b")
    ap.add_argument("--version", default="v2")
    ap.add_argument("--set", nargs="*", default=[], help="Spec overrides, e.g. sensitive_combine=kind room_combine=kind")
    ap.add_argument("--note", default="")
    ap.add_argument("--tag", default="", help="name used in the run file instead of the version")
    ap.add_argument("--cases", help="alternative cases file")
    ap.add_argument("--no-results", action="store_true")
    ap.add_argument("--summarize", help="re-summarize an existing run file; no server calls")
    args = ap.parse_args()
    if args.summarize:
        run = json.loads(Path(args.summarize).read_text(encoding="utf-8"))
        print_summary(f"{run['version']} {run['model']}", summarize(run["rows"]))
        return
    overrides = parse_overrides(args.set)
    if args.model == "keyword":
        decider: Any = KeywordDecider()
    else:
        decider = HttpDecider(f"decider-{args.model}", URLS[args.model], max_concurrency=1)
        if not await decider.healthy():
            sys.exit(f"decider {args.model} at {URLS[args.model]} is not healthy")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = args.tag or args.version
    cases = load_cases(Path(args.cases) if args.cases else None)
    run: dict[str, Any] = {"stamp": stamp, "version": args.version, "overrides": overrides, "model": args.model, "note": args.note,
                           "summary": {}, "rows": []}
    try:
        rows = await run_cases(cases, decider, args.version, overrides)
        s = summarize(rows)
        run["rows"], run["summary"] = rows, s
        print_summary(f"{args.version} {args.model}", s)
        if not args.no_results:
            note = " ".join(filter(None, [args.note, " ".join(args.set)]))
            append_results(results_row(stamp, name, args.model, s, note))
    finally:
        if run["rows"]:
            RUNS_DIR.mkdir(exist_ok=True)
            out = RUNS_DIR / f"{stamp}-meeting-{name}-{args.model}.json"
            out.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"\nwrote {out}")
        if hasattr(decider, "aclose"):
            await decider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
