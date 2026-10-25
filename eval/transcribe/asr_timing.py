"""Word-timing quality of an asr.json against the case's ref.words.json (eval code: it may read references).

Hyp and ref words are normalised (lower case, letters and digits only), aligned as sequences with difflib
(longest matching blocks, no junk heuristic) and every matched pair contributes a start error and an end
error (hyp - ref, seconds). Reported: counts, match rate, median / mean / p90 of |error|, the signed median
(a constant lag the method could correct) and the share of matched words whose start lies within 0.25 s of
the reference (the DER collar). Words of all reference speakers count, on or off camera.

Usage: python -m eval.transcribe.asr_timing --ref testdata/ami/CASE --asr asr.json [--json]
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

_WORD = re.compile(r"[^a-z0-9]+")


def norm(text: str) -> str:
    return _WORD.sub("", text.lower())


def hyp_words(asr: dict[str, Any]) -> list[dict]:
    out = []
    for s in sorted(asr["segments"], key=lambda s: s["start"]):
        out += sorted(s.get("words") or [], key=lambda w: w["start"])
    return out


def align(ref: list[dict], hyp: list[dict]) -> list[tuple[dict, dict]]:
    r = [(norm(w["text"]), w) for w in ref]
    h = [(norm(w["text"]), w) for w in hyp]
    r = [x for x in r if x[0]]
    h = [x for x in h if x[0]]
    sm = difflib.SequenceMatcher(None, [x[0] for x in r], [x[0] for x in h], autojunk=False)
    pairs = []
    for a, b, n in sm.get_matching_blocks():
        pairs += [(r[a + i][1], h[b + i][1]) for i in range(n)]
    return pairs


def _stats(err: np.ndarray) -> dict[str, float]:
    if err.size == 0:
        return {"median_abs": float("nan"), "mean_abs": float("nan"), "p90_abs": float("nan"),
                "median_signed": float("nan"), "within_0.25": float("nan")}
    a = np.abs(err)
    return {"median_abs": round(float(np.median(a)), 3), "mean_abs": round(float(a.mean()), 3),
            "p90_abs": round(float(np.percentile(a, 90)), 3), "median_signed": round(float(np.median(err)), 3),
            "within_0.25": round(float((a <= 0.25).mean()), 3)}


def timing_report(ref_words: list[dict], asr: dict[str, Any]) -> dict[str, Any]:
    hyp = hyp_words(asr)
    pairs = align(ref_words, hyp)
    start = np.array([h["start"] - r["start"] for r, h in pairs])
    end = np.array([h["end"] - r["end"] for r, h in pairs])
    n_ref = sum(1 for w in ref_words if norm(w["text"]))
    return {"ref_words": n_ref, "hyp_words": len(hyp), "matched": len(pairs),
            "match_rate": round(len(pairs) / max(n_ref, 1), 3), "start": _stats(start), "end": _stats(end)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path, help="case dir with ref.words.json")
    ap.add_argument("--asr", required=True, type=Path)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rep = timing_report(json.loads((a.ref / "ref.words.json").read_text()), json.loads(a.asr.read_text()))
    if a.json:
        print(json.dumps(rep))
        return
    print(f"ref {rep['ref_words']}  hyp {rep['hyp_words']}  matched {rep['matched']} ({rep['match_rate']:.1%})")
    for k in ("start", "end"):
        s = rep[k]
        print(f"{k:5s} |err| median {s['median_abs']:.3f}  mean {s['mean_abs']:.3f}  p90 {s['p90_abs']:.3f}  "
              f"signed median {s['median_signed']:+.3f}  within 0.25 s {s['within_0.25']:.1%}")


if __name__ == "__main__":
    main()
