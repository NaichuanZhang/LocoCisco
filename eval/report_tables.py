"""Print the markdown tables of eval/REPORT.md from run files, so every number is traceable to a run.

    $VENV/bin/python eval/report_tables.py compare  LABEL=eval/runs/<file>.json[:split] ...
    $VENV/bin/python eval/report_tables.py calib    LABEL=eval/runs/<file>.json[:split] ...
    $VENV/bin/python eval/report_tables.py thresholds LABEL=eval/runs/<file>.json[:split] ...
    $VENV/bin/python eval/report_tables.py failures LABEL=eval/runs/<file>.json[:split] ...

A ":split" suffix keeps only that split's rows of a mixed run. Act precision = share of non-"none"
predictions at or above a confidence threshold that are right (the number the act_threshold must keep
at 0.95 or better).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.run_eval import summarize  # noqa: E402


def load(arg: str) -> tuple[str, dict[str, list[dict]]]:
    label, _, spec = arg.partition("=")
    path, _, split = spec.partition(":")
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {k: [r for r in v if not split or r["split"] == split] for k, v in run["rows"].items()}
    return label, rows


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def compare(args: list[str]) -> None:
    print("| run | kind | n | accuracy | acc@0.8 (coverage) | act precision@0.8 (n acting) | room acc (light-action cases) | "
          "Brier | median / p90 latency |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arg in args:
        label, rows = load(arg)
        for kind, rs in rows.items():
            s = summarize(rs)
            t8 = s["by_threshold"]["0.8"]
            room = (f"{pct(s['room_accuracy'])} ({s['room_n']}); {pct(s['room_accuracy_when_action_right'])} when "
                    f"action right ({s['room_n_when_action_right']})" if s.get("room_n") else "-")
            print(f"| {label} | {kind} | {s['n']} | {pct(s['accuracy'])} | {pct(t8['accuracy'])} ({pct(t8['coverage'])}) | "
                  f"{pct(t8['act_precision'])} ({t8['act_n']}) | {room} | {s['brier']} | "
                  f"{s['latency_ms']['median']:.0f} / {s['latency_ms']['p90']:.0f} ms |")


def calib(args: list[str]) -> None:
    bins = ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9",
            "0.9-1.0"]
    print("| run | kind | " + " | ".join(bins) + " |")
    print("|---|---|" + "---|" * len(bins))
    for arg in args:
        label, rows = load(arg)
        for kind, rs in rows.items():
            rel = summarize(rs)["reliability"]
            cells = [f"{pct(rel[b]['accuracy'])} ({rel[b]['n']})" if b in rel else "-" for b in bins]
            print(f"| {label} | {kind} | " + " | ".join(cells) + " |")
    print("\nCells: accuracy of the chosen action (count) for cases whose confidence falls in the bin.")


def act_precision(rs: list[dict], t: float) -> tuple[float | None, int, float]:
    acting = [r for r in rs if r["confidence"] >= t and r["pred"] != "none"]
    cov = sum(r["confidence"] >= t for r in rs) / len(rs)
    return (sum(r["right"] for r in acting) / len(acting) if acting else None), len(acting), cov


def thresholds(args: list[str]) -> None:
    grid = (0.5, 0.6, 0.7, 0.8, 0.9)
    print("| run | kind | " + " | ".join(f"act precision @{t} (n acting, coverage)" for t in grid) + " |")
    print("|---|---|" + "---|" * len(grid))
    for arg in args:
        label, rows = load(arg)
        for kind, rs in rows.items():
            cells = []
            for t in grid:
                prec, n_act, cov = act_precision(rs, t)
                cells.append(f"{pct(prec)} ({n_act}, {pct(cov)})")
            print(f"| {label} | {kind} | " + " | ".join(cells) + " |")
    print("\nAct precision = share of non-none predictions at or above the threshold that are right; n acting = how "
          "many such predictions; coverage = share of all cases (including none) at or above the threshold.")


def failures(args: list[str]) -> None:
    print("| run | case | said / seen | expected | predicted | confidence |")
    print("|---|---|---|---|---|---|")
    for arg in args:
        label, rows = load(arg)
        for kind, rs in rows.items():
            wrong = sorted((r for r in rs if not r["right"]), key=lambda r: -r["confidence"])
            for r in wrong[:8]:
                text = r["state"].split("Heard: ", 1)[-1].split("Said: ", 1)[-1]
                text = text.split("\n")[0][:90].replace("|", "/")
                print(f"| {label} | {r['id']} | {text} | {'/'.join(r['ok'])} | {r['pred']} | {r['confidence']:.2f} |")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    {"compare": compare, "calib": calib, "thresholds": thresholds, "failures": failures}[cmd](rest)
