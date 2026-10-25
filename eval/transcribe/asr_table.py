"""Markdown results table from testdata/ami-runs/asr/<method>/<case>/{scores,timing,asr}.json.

Usage: python -m eval.transcribe.asr_table [--runs testdata/ami-runs/asr] [--full | --dev]
Columns: WER, word-start median |error| (s), share of matched word starts within 0.25 s, match rate (ref words
with an aligned hyp word), RTF (decode wall / audio, model load excluded), threads and the 1-minute load average
at start -> end of the run (six workstreams shared the box, so RTF must be read with its load).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v3-int8", "fw-large-v3-turbo", "fw-distil-large-v3.5", "fw-small.en",
         "fw-base.en", "vad-fw-small.en", "vad-fw-base.en", "vad-fw-large-v3-turbo", "server"]
CASES = ["IS1009a-duo-AC", "IS1009a-duo-DB", "IS1009a-L-room", "IS1009a-R-room"]


def rows(runs: Path, dev: bool) -> list[str]:
    out = ["| method | case | WER | start med err | starts <0.25 s | matched | RTF | threads | load1 |",
           "|---|---|---|---|---|---|---|---|---|"]
    methods = [m for m in ORDER if (runs / m).is_dir()] + sorted(p.name for p in runs.iterdir() if p.is_dir() and p.name not in ORDER)
    for m in methods:
        for c in CASES:
            d = runs / m / (c + ("-dev" if dev else ""))
            if not (d / "scores.json").exists():
                continue
            s, t, a = (json.loads((d / f"{n}.json").read_text()) for n in ("scores", "timing", "asr"))
            tm = a["timing"]
            out.append(f"| {m} | {d.name} | {s['wer']:.3f} | {t['start']['median_abs']:.3f} | {t['start']['within_0.25']:.1%} | "
                       f"{t['match_rate']:.1%} | {tm['rtf']:.3f} | {tm['threads']} | {tm['load1_start']}->{tm['load1_end']} |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("testdata/ami-runs/asr"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--full", action="store_true")
    g.add_argument("--dev", action="store_true")
    a = ap.parse_args()
    if not a.full:
        print("\n".join(rows(a.runs, dev=True)))
    if not a.dev:
        print()
        print("\n".join(rows(a.runs, dev=False)))


if __name__ == "__main__":
    main()
