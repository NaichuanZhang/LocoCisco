"""Measure the 2B -> Claude-on-Bedrock cascade over eval/cases_speech.json and eval/cases_scene.json.

    $VENV/bin/python eval/run_bedrock.py --decider-2b-url http://127.0.0.1:8767 [--split all|dev|test]
                                         [--kind speech scene] [--baseline] [--out eval/runs/bedrock.json]
    $VENV/bin/python eval/run_bedrock.py --summarize eval/runs/bedrock.json      # no calls

Every case goes through the SAME path as the demo (homedemo.agents.house_agent.HouseAgent with the
DeciderToolRouter and the cascade's accept rule): the 2B answers once; when it is sure the model is
never called (route "local"), otherwise Claude Haiku 4.5 decides with the house tools (route
"bedrock"). With --baseline every case is ALSO sent straight to Claude (router off) for the
"LLM on every sentence" comparison. Results are written after every case, so a stopped run can be
summarized. One request is in flight at a time (the 2B server is not concurrency-safe).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from homedemo.agents.cascade import accept_rule  # noqa: E402
from homedemo.agents.house_agent import HouseAgent, HouseResult, interpretation_of, proposal_from_tool_call  # noqa: E402
from homedemo.contracts import Settings  # noqa: E402
from homedemo.decider.questions import LIGHT_ACTIONS, build_scene_query, build_speech_query  # noqa: E402
from homedemo.decider_client import HttpDecider  # noqa: E402

EVAL_DIR = ROOT / "eval"
RETRIES, RETRY_SLEEP_S = 8, 45.0


def load_cases(kind: str, split: str) -> list[dict[str, Any]]:
    cases = json.loads((EVAL_DIR / f"cases_{kind}.json").read_text(encoding="utf-8"))["cases"]
    return [c for c in cases if split == "all" or c["split"] == split]


def build(kind: str, case: dict[str, Any]):
    if kind == "speech":
        return build_speech_query(case["utt"]), case["utt"]
    return build_scene_query(case["scene"]), case["scene"]


def claude_pred(out: HouseResult) -> tuple[str, list[str]]:
    """Claude's action = the first tool call (ask_resident(X) counts as X); no call = none."""
    if not out.tool_calls:
        return "none", []
    actions = [proposal_from_tool_call(c)[0] for c in out.tool_calls]
    return actions[0], [c["name"] for c in out.tool_calls]


async def with_retries(house: HouseAgent, fn, label: str) -> HouseResult:
    """Bedrock errors (expired credentials, throttling) get a rebuilt model and another try."""
    for attempt in range(1, RETRIES + 1):
        out = await fn()
        if out.error is None:
            return out
        print(f"   ! {label}: {out.error[:140]} (attempt {attempt}/{RETRIES}); retrying in {RETRY_SLEEP_S:.0f} s",
              flush=True)
        house._model = None                       # a fresh boto3 session re-reads the credentials file
        await asyncio.sleep(RETRY_SLEEP_S)
    return out


async def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = Settings(decider_mode="cascade", escalate_to="bedrock", decider_2b_url=args.decider_2b_url,
                        bedrock_profile=args.bedrock_profile or None)
    decider = HttpDecider("decider-2b", settings.decider_2b_url)
    house = HouseAgent(settings)
    run_doc: dict[str, Any] = {"stamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "decider_2b_url": settings.decider_2b_url, "model_id": settings.bedrock_model_id,
                               "act_threshold": settings.act_threshold, "split": args.split, "rows": {}}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for kind in args.kind:
            cases = load_cases(kind, args.split)[: args.limit or None]
            rows: list[dict[str, Any]] = []
            run_doc["rows"][kind] = rows
            for i, case in enumerate(cases, 1):
                query, data = build(kind, case)
                from_person = kind == "scene" or data.get("source", "person") == "person"
                threshold = settings.act_threshold

                def accept(decision, _fp=from_person, _t=threshold):
                    return accept_rule(decision, from_person=_fp, threshold=_t)

                out = await with_retries(house, lambda: house.decide(decider, query, data, accept=accept), case["id"])
                it = interpretation_of(out.decision) if out.decision else None
                row: dict[str, Any] = {
                    "id": case["id"], "kind": kind, "split": case["split"], "category": case["category"],
                    "expected": case["expected"], "ok": case["ok"], "expected_room": case.get("room"),
                    "route": out.route, "error": out.error,
                    "two_b": None if it is None else {"action": it.action, "confidence": round(it.confidence, 4),
                                                        "room": it.room, "latency_ms": round(out.decision.latency_ms, 1),
                                                        "right": it.action in case["ok"]},
                }
                if out.route == "local":
                    row["pred"] = it.action
                    row["pred_room"] = it.room
                    row["latency_ms"] = round(out.decision.latency_ms, 1)
                else:
                    pred, tools = claude_pred(out)
                    row["pred"] = pred
                    row["pred_room"] = next((c["input"].get("room") for c in out.tool_calls if c["input"].get("room")), None)
                    row["claude"] = {"tools": tools, "tool_calls": out.tool_calls, "reply": out.reply,
                                     "tokens": out.tokens, "latency_ms": round(out.latency_ms, 1)}
                    row["latency_ms"] = round(out.total_ms, 1)
                row["right"] = row["pred"] in case["ok"]
                if kind == "speech":
                    row["room_judged"] = case["expected"] in LIGHT_ACTIONS
                    room_pred = row["pred_room"] if row["pred_room"] != data.get("room") else None
                    row["room_right"] = row["room_judged"] and room_pred == case.get("room")
                if args.baseline:
                    base = await with_retries(house, lambda: house.decide(decider, query, data, router=False), case["id"] + " baseline")
                    pred, tools = claude_pred(base)
                    row["baseline"] = {"pred": pred, "right": pred in case["ok"], "tools": tools, "reply": base.reply,
                                       "tokens": base.tokens, "latency_ms": round(base.latency_ms, 1), "error": base.error}
                rows.append(row)
                mark = "ok " if row["right"] else "XX "
                tok = row.get("claude", {}).get("tokens")
                print(f"{mark}{case['id']:5} {kind:6} {row['route']:7} pred={row['pred']:12} exp={'/'.join(case['ok']):20} "
                      f"2b={row['two_b']['action'] if row['two_b'] else '-'}@{row['two_b']['confidence'] if row['two_b'] else '-'} "
                      f"{row['latency_ms']:6.0f}ms tok={tok['input'] + tok['output'] if tok else '-'}"
                      f"{'  base=' + row['baseline']['pred'] if args.baseline else ''}  [{i}/{len(cases)}]", flush=True)
                out_path.write_text(json.dumps(run_doc, indent=1, ensure_ascii=False), encoding="utf-8")
    finally:
        await decider.aclose()
    return run_doc


# ---------------------------------------------------------------------- summary
def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 0) if xs else None


def _p90(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(0.9 * len(s)))], 0)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    local = [r for r in rows if r["route"] == "local"]
    esc = [r for r in rows if r["route"] == "bedrock"]
    acting_local = [r for r in local if r["pred"] != "none"]
    tokens = [r["claude"]["tokens"]["input"] + r["claude"]["tokens"]["output"] for r in esc if r.get("claude", {}).get("tokens")]
    in_tok = [r["claude"]["tokens"]["input"] for r in esc if r.get("claude", {}).get("tokens")]
    out_tok = [r["claude"]["tokens"]["output"] for r in esc if r.get("claude", {}).get("tokens")]
    two_b_all = [r for r in rows if r["two_b"]]
    s: dict[str, Any] = {
        "n": n, "right": sum(r["right"] for r in rows),
        "local": {"n": len(local), "right": sum(r["right"] for r in local),
                  "acting_n": len(acting_local), "acting_right": sum(r["right"] for r in acting_local),
                  "median_ms": _median([r["latency_ms"] for r in local]), "p90_ms": _p90([r["latency_ms"] for r in local])},
        "bedrock": {"n": len(esc), "right": sum(r["right"] for r in esc),
                    "median_total_ms": _median([r["latency_ms"] for r in esc]), "p90_total_ms": _p90([r["latency_ms"] for r in esc]),
                    "median_claude_ms": _median([r["claude"]["latency_ms"] for r in esc]),
                    "median_2b_ms": _median([r["two_b"]["latency_ms"] for r in esc if r["two_b"]]),
                    "tokens_total": sum(tokens), "tokens_median": _median(tokens),
                    "input_tokens_median": _median(in_tok), "output_tokens_median": _median(out_tok),
                    "two_b_would_have_been_right": sum(r["two_b"]["right"] for r in esc if r["two_b"]),
                    "errors": sum(1 for r in esc if r.get("error"))},
        "two_b_alone_right": sum(r["two_b"]["right"] for r in two_b_all),
        "wrong": [{"id": r["id"], "route": r["route"], "pred": r["pred"], "ok": r["ok"],
                   "two_b": r["two_b"]["action"] if r["two_b"] else None,
                   "conf": r["two_b"]["confidence"] if r["two_b"] else None} for r in rows if not r["right"]],
    }
    if any("room_judged" in r for r in rows):
        judged = [r for r in rows if r.get("room_judged")]
        s["room"] = {"n": len(judged), "right": sum(r["room_right"] for r in judged)}
    base = [r["baseline"] for r in rows if r.get("baseline")]
    if base:
        btok = [b["tokens"]["input"] + b["tokens"]["output"] for b in base if b.get("tokens")]
        s["baseline"] = {"n": len(base), "right": sum(b["right"] for b in base),
                         "median_ms": _median([b["latency_ms"] for b in base]), "p90_ms": _p90([b["latency_ms"] for b in base]),
                         "tokens_total": sum(btok), "tokens_median": _median(btok), "errors": sum(1 for b in base if b.get("error")),
                         "wrong": [{"id": r["id"], "pred": r["baseline"]["pred"], "ok": r["ok"]} for r in rows
                                   if r.get("baseline") and not r["baseline"]["right"]]}
    return s


def pct(a: int, b: int) -> str:
    return "-" if not b else f"{a}/{b} ({100 * a / b:.0f}%)"


def markdown(run_doc: dict[str, Any]) -> str:
    lines = [f"Run {run_doc['stamp']}, split `{run_doc['split']}`, 2B at {run_doc['decider_2b_url']}, model "
             f"`{run_doc['model_id']}`, act threshold {run_doc['act_threshold']}.", ""]
    lines += ["| kind | cases | handled locally (2B sure) | right when local | acting locally right | right when escalated to Claude "
              "| cascade right overall | 2B alone right | median latency local | median latency escalated (2B + Claude) | "
              "Claude share | tokens per Claude call (median) | tokens total |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for kind, rows in run_doc["rows"].items():
        s = summarize(rows)
        L, B = s["local"], s["bedrock"]
        lines.append(f"| {kind} | {s['n']} | {pct(L['n'], s['n'])} | {pct(L['right'], L['n'])} | {pct(L['acting_right'], L['acting_n'])} | "
                     f"{pct(B['right'], B['n'])} | {pct(s['right'], s['n'])} | {pct(s['two_b_alone_right'], s['n'])} | "
                     f"{L['median_ms']} ms | {B['median_total_ms']} ms | {B['median_claude_ms']} ms | "
                     f"{B['tokens_median']} (in {B['input_tokens_median']} / out {B['output_tokens_median']}) | {B['tokens_total']} |")
    if any(r.get("baseline") for rows in run_doc["rows"].values() for r in rows):
        lines += ["", "Baseline, Claude on every case (router off):", "",
                  "| kind | cases | right | median latency | p90 latency | tokens per call (median) | tokens total |",
                  "|---|---|---|---|---|---|---|"]
        for kind, rows in run_doc["rows"].items():
            b = summarize(rows).get("baseline")
            if b:
                lines.append(f"| {kind} | {b['n']} | {pct(b['right'], b['n'])} | {b['median_ms']} ms | {b['p90_ms']} ms | "
                             f"{b['tokens_median']} | {b['tokens_total']} |")
    lines += ["", "Wrong answers (cascade):", ""]
    for kind, rows in run_doc["rows"].items():
        for w in summarize(rows)["wrong"]:
            lines.append(f"- {kind} {w['id']}: route {w['route']}, predicted `{w['pred']}`, expected {'/'.join(w['ok'])}"
                         f" (2B said `{w['two_b']}` at {w['conf']})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decider-2b-url", default=Settings().decider_2b_url)
    ap.add_argument("--bedrock-profile", default=Settings().bedrock_profile)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--kind", nargs="+", choices=["speech", "scene"], default=["speech", "scene"])
    ap.add_argument("--baseline", action="store_true", help="also send every case straight to Claude")
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases per kind (smoke test)")
    ap.add_argument("--out", default=str(EVAL_DIR / "runs" / "bedrock.json"))
    ap.add_argument("--summarize", help="print the markdown summary of an existing run file; no calls")
    args = ap.parse_args()
    if args.summarize:
        run_doc = json.loads(Path(args.summarize).read_text(encoding="utf-8"))
    else:
        run_doc = asyncio.run(run(args))
    print()
    print(markdown(run_doc))
    for kind, rows in run_doc["rows"].items():
        print(f"\n{kind}: {json.dumps(summarize(rows), ensure_ascii=False)[:1500]}")


if __name__ == "__main__":
    main()
