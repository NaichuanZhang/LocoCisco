"""Run eval/cases_*.json through build_*_query + HttpDecider and report accuracy and calibration.

    $VENV/bin/python eval/run_eval.py --model 2b --version v1 --kind speech scene --split dev
    $VENV/bin/python eval/run_eval.py --summarize eval/runs/<file>.json     # no server calls

Writes eval/runs/<stamp>-<version>-<model>.json (gitignored) and appends one row per kind to
eval/RESULTS.md. `--set field=value` overrides Spec fields (homedemo.decider.questions.Spec) for
experiments that have no version name yet; the overrides are recorded in the run file.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from homedemo.decider.questions import LIGHT_ACTIONS, build_scene_query, build_speech_query  # noqa: E402
from homedemo.decider_client import HttpDecider  # noqa: E402

EVAL_DIR = ROOT / "eval"
RUNS_DIR = EVAL_DIR / "runs"
URLS = {"2b": "http://127.0.0.1:8765", "12b": "http://127.0.0.1:8766"}
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def load_cases(kind: str, split: str) -> list[dict[str, Any]]:
    cases = json.loads((EVAL_DIR / f"cases_{kind}.json").read_text(encoding="utf-8"))["cases"]
    return [c for c in cases if split == "all" or c["split"] == split]


def build(kind: str, case: dict[str, Any], version: str, overrides: dict[str, Any]):
    if kind == "speech":
        return build_speech_query(case["utt"], version=version, **overrides)
    return build_scene_query(case["scene"], version=version, **overrides)


async def run_cases(kind: str, cases: list[dict[str, Any]], decider: HttpDecider, version: str,
                    overrides: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for i, case in enumerate(cases, 1):
        q = build(kind, case, version, overrides)
        t0 = time.perf_counter()
        answers = await decider.ask(q.state, q.questions)
        ms = (time.perf_counter() - t0) * 1000
        it = q.interpret(answers)
        expected_room = case.get("room")  # None = the speaker's own room
        row = {
            "id": case["id"], "split": case["split"], "category": case["category"], "tags": case.get("tags", []),
            "expected": case["expected"], "ok": case["ok"], "pred": it.action, "right": it.action in case["ok"],
            "confidence": round(it.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in it.probabilities.items()},
            "latency_ms": round(ms, 1), "detail": it.detail, "state": q.state,
            "answers": {k: {"choice": v.choice, "noul": v.noul,
                            "probabilities": {o: round(p, 4) for o, p in v.probabilities.items()}}
                        for k, v in answers.items()},
            "option_actions": q.option_actions,
        }
        if kind == "speech":
            row["expected_room"] = expected_room
            row["pred_room"] = it.room
            # Only light actions have a target room; judged on cases that expect a light action.
            row["room_judged"] = case["expected"] in LIGHT_ACTIONS
            row["room_right"] = row["room_judged"] and it.room == expected_room
        rows.append(row)
        mark = "ok " if row["right"] else "XX "
        print(f"{mark}{case['id']:5} {kind:6} conf={it.confidence:.2f} pred={it.action:12} "
              f"exp={'/'.join(case['ok']):20} room={it.room} {ms:6.0f}ms  [{i}/{len(cases)}]", flush=True)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    acc = sum(r["right"] for r in rows) / n
    by_t = {}
    for t in THRESHOLDS:
        sure = [r for r in rows if r["confidence"] >= t]
        acting = [r for r in sure if r["pred"] != "none"]
        by_t[str(t)] = {
            "coverage": round(len(sure) / n, 3),
            "accuracy": round(sum(r["right"] for r in sure) / len(sure), 3) if sure else None,
            "act_n": len(acting),
            "act_precision": round(sum(r["right"] for r in acting) / len(acting), 3) if acting else None,
        }
    lat = sorted(r["latency_ms"] for r in rows)
    out: dict[str, Any] = {
        "n": n, "accuracy": round(acc, 3), "by_threshold": by_t,
        "latency_ms": {"median": round(statistics.median(lat), 0), "p90": round(lat[min(n - 1, int(0.9 * n))], 0)},
        "by_category": {},
        "wrong": [{"id": r["id"], "pred": r["pred"], "ok": r["ok"], "confidence": r["confidence"]}
                  for r in rows if not r["right"]],
    }
    for r in rows:
        c = out["by_category"].setdefault(r["category"], [0, 0])
        c[0] += r["right"]
        c[1] += 1
    if any("pred_room" in r for r in rows):
        judged = [r for r in rows if r["expected"] in LIGHT_ACTIONS]
        other = [r for r in judged if r.get("expected_room")]
        room_ok = [r["pred_room"] == r["expected_room"] for r in judged]
        out["room_n"] = len(judged)
        out["room_accuracy"] = round(sum(room_ok) / len(judged), 3) if judged else None
        out["room_accuracy_other_room"] = (round(sum(r["pred_room"] == r["expected_room"] for r in other)
                                                 / len(other), 3) if other else None)
        out["room_n_other_room"] = len(other)
        hit = [r for r in judged if r["right"]]
        out["room_accuracy_when_action_right"] = (round(sum(r["pred_room"] == r["expected_room"] for r in hit)
                                                        / len(hit), 3) if hit else None)
        out["room_n_when_action_right"] = len(hit)
    # Calibration: reliability by confidence bin over the chosen action.
    bins: dict[str, list[int]] = {}
    for r in rows:
        lo = min(int(r["confidence"] * 10) / 10, 0.9)
        b = bins.setdefault(f"{lo:.1f}-{lo + 0.1:.1f}", [0, 0])
        b[0] += r["right"]
        b[1] += 1
    out["reliability"] = {k: {"n": v[1], "accuracy": round(v[0] / v[1], 3)} for k, v in sorted(bins.items())}
    out["brier"] = round(sum((r["confidence"] - (1.0 if r["right"] else 0.0)) ** 2 for r in rows) / n, 3)
    return out


def results_row(stamp: str, version: str, model: str, kind: str, split: str, s: dict[str, Any], note: str) -> str:
    t8, t9 = s["by_threshold"]["0.8"], s["by_threshold"]["0.9"]

    def pct(x: float | None) -> str:
        return "-" if x is None else f"{100 * x:.0f}%"

    room = pct(s.get("room_accuracy")) if "room_accuracy" in s else "-"
    return (f"| {stamp} | {version} | {model} | {kind} | {split} | {s['n']} | {pct(s['accuracy'])} | "
            f"{pct(t8['accuracy'])} ({pct(t8['coverage'])}) | {pct(t9['accuracy'])} ({pct(t9['coverage'])}) | "
            f"{pct(t8['act_precision'])} | {room} | {s['brier']} | {s['latency_ms']['median']:.0f} / "
            f"{s['latency_ms']['p90']:.0f} | {note} |")


RESULTS_HEADER = (
    "# Eval runs\n\nOne row per run and kind, appended by eval/run_eval.py. acc@t (cov) = accuracy among cases "
    "with confidence >= t and the share of cases that reach t; act prec@0.8 = share of non-none predictions "
    "at confidence >= 0.8 that are right; room = Interpretation.room matches the expected room (speech).\n\n"
    "| stamp | version | model | kind | split | n | acc | acc@0.8 (cov) | acc@0.9 (cov) | act prec@0.8 | room | "
    "Brier | median / p90 ms | note |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")


def append_results(line: str) -> None:
    path = EVAL_DIR / "RESULTS.md"
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


def print_summary(kind: str, s: dict[str, Any]) -> None:
    print(f"\n== {kind}: n={s['n']} acc={s['accuracy']:.3f} brier={s['brier']} "
          f"latency median={s['latency_ms']['median']:.0f}ms p90={s['latency_ms']['p90']:.0f}ms")
    for t, v in s["by_threshold"].items():
        print(f"   conf>={t}: coverage={v['coverage']:.2f} acc={v['accuracy']} act_n={v['act_n']} "
              f"act_precision={v['act_precision']}")
    if "room_accuracy" in s:
        print(f"   room acc={s['room_accuracy']} (n={s['room_n']} light-action cases) other-room subset="
              f"{s['room_accuracy_other_room']} (n={s['room_n_other_room']}) when action right="
              f"{s['room_accuracy_when_action_right']} (n={s['room_n_when_action_right']})")
    print("   by category:", {k: f"{v[0]}/{v[1]}" for k, v in s["by_category"].items()})
    print("   wrong:", [(w["id"], w["pred"], w["confidence"]) for w in s["wrong"]])


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(URLS), default="2b")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--kind", nargs="+", choices=["speech", "scene"], default=["speech", "scene"])
    ap.add_argument("--split", choices=["dev", "test", "all"], default=None,
                    help="cases to run (default dev); with --summarize, rows to keep (default all)")
    ap.add_argument("--set", nargs="*", default=[], help="Spec overrides, e.g. gate=false facts=state")
    ap.add_argument("--note", default="")
    ap.add_argument("--tag", default="", help="name used in the run file instead of the version")
    ap.add_argument("--no-results", action="store_true", help="do not append to RESULTS.md")
    ap.add_argument("--summarize", help="re-summarize an existing run file; no server calls")
    args = ap.parse_args()

    if args.summarize:
        split = args.split or "all"
        run = json.loads(Path(args.summarize).read_text(encoding="utf-8"))
        for kind, rows in run["rows"].items():
            rows = [r for r in rows if split == "all" or r["split"] == split]
            if rows:
                print_summary(f"{kind} ({split})", summarize(rows))
        return
    args.split = args.split or "dev"

    overrides = parse_overrides(args.set)
    decider = HttpDecider(f"decider-{args.model}", URLS[args.model], max_concurrency=1)
    if not await decider.healthy():
        sys.exit(f"decider {args.model} at {URLS[args.model]} is not healthy")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = args.tag or args.version
    run: dict[str, Any] = {"stamp": stamp, "version": args.version, "overrides": overrides, "model": args.model,
                           "split": args.split, "note": args.note, "summary": {}, "rows": {}}
    try:
        for kind in args.kind:
            cases = load_cases(kind, args.split)
            rows = await run_cases(kind, cases, decider, args.version, overrides)
            s = summarize(rows)
            run["rows"][kind] = rows
            run["summary"][kind] = s
            print_summary(kind, s)
            if not args.no_results:
                note = " ".join(filter(None, [args.note, " ".join(args.set)]))
                append_results(results_row(stamp, name, args.model, kind, args.split, s, note))
    finally:
        if run["rows"]:  # written before any await so an interrupted run still keeps its finished kinds
            RUNS_DIR.mkdir(exist_ok=True)
            out = RUNS_DIR / f"{stamp}-{name}-{args.model}.json"
            out.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"\nwrote {out}")
        await decider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
