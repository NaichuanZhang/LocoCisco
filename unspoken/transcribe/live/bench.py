"""Results table for the report: one row per case from a runs dir (scores.json + latency.json written by
`python -m unspoken.transcribe.live` and `eval.transcribe.score`), next to the offline baseline's scores.

  python -m unspoken.transcribe.live.bench --runs testdata/ami-runs/live/live --baseline testdata/ami-baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _f(v: Any, nd: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and v != v:
        return "nan"
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def _load_line(run: Path) -> str:
    def avg(name: str) -> str:
        p = run / name
        if not p.exists():
            return "?"
        txt = p.read_text().strip()
        return txt.split("average:")[-1].strip().split(",")[0] if "average:" in txt else "?"
    return f"{avg('uptime_before.txt')}->{avg('uptime_after.txt')}"


def rows(runs: Path, baseline: Path | None, cases: list[str] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        case = case_dir.name
        if cases and case not in cases:
            continue
        sc = _load(case_dir / "scores.json") or {}
        lat = _load(case_dir / "latency.json") or {}
        base = _load(baseline / case / "scores.json") if baseline else None
        fl, pl = lat.get("final_latency_s", {}), lat.get("partial_latency_s", {})
        th = lat.get("threads", {})
        out.append({"case": case, "split": "dev" if case.endswith("-dev") else "full",
                    "wer": sc.get("wer"), "cpwer": sc.get("cpwer"), "der": sc.get("der"), "sa_acc": sc.get("sa_acc"),
                    "seat_acc": sc.get("seat_acc"), "hyp_spk": sc.get("hyp_speakers"),
                    "lat_p50": fl.get("p50"), "lat_p95": fl.get("p95"), "part_p50": pl.get("p50"),
                    "partials": lat.get("partials"), "revisions": lat.get("revisions"),
                    "rtf": lat.get("rtf"), "cpu_rtf": lat.get("cpu_rtf"), "busy_rtf": lat.get("busy_rtf"),
                    "threads": f"asr {th.get('asr_threads', '?')} + 3 x 1 (OMP {th.get('OMP_NUM_THREADS', '?')})",
                    "load": _load_line(case_dir),
                    "frames_stale": (lat.get("frames") or {}).get("frames_stale"),
                    "base": base})
    return out


def table(rs: list[dict[str, Any]]) -> str:
    head = ("| case | split | method | WER | cpWER | DER | sa_acc | seat_acc | hyp spk | final lat p50 / p95 s | "
            "partial p50 s | partials / revisions | RTF | CPU RTF | threads | load 1-min before->after |")
    lines = [head, "|" + "---|" * 17]
    for r in rs:
        if r["base"]:
            b = r["base"]
            lines.append(f"| {r['case']} | {r['split']} | baseline (offline replay) | {_f(b.get('wer'))} | {_f(b.get('cpwer'))} | "
                         f"{_f(b.get('der'))} | {_f(b.get('sa_acc'))} | {_f(b.get('seat_acc'))} | {b.get('hyp_speakers')} | "
                         f"- | - | - | - | - | - | - |")
        lines.append(f"| {r['case']} | {r['split']} | live 1x | {_f(r['wer'])} | {_f(r['cpwer'])} | {_f(r['der'])} | "
                     f"{_f(r['sa_acc'])} | {_f(r['seat_acc'])} | {r['hyp_spk']} | {_f(r['lat_p50'], 2)} / {_f(r['lat_p95'], 2)} | "
                     f"{_f(r['part_p50'], 2)} | {r['partials']} / {r['revisions']} | {_f(r['rtf'])} | {_f(r['cpu_rtf'])} | "
                     f"{r['threads']} | {r['load']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rs = rows(a.runs, a.baseline, a.cases)
    if a.json:
        print(json.dumps(rs, indent=1))
    else:
        print(table(rs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
