"""Score a per-person transcript against an AMI-style reference (see ami_prepare.py).

Metrics (all lower-is-better except the accuracies):
  wer        speaker-agnostic word error rate (ref segments and hyp utterances each in start-time order)
  cpwer      concatenated minimum-permutation WER: per-speaker text, best hyp->ref speaker mapping
  der        diarization error rate, 10 ms frames, 0.25 s collar, overlap scored, best mapping
  sa_acc     speaker-attributed word accuracy: share of ref words whose time midpoint falls in a hyp word or
             utterance of the hyp speaker mapped (best mapping) to the right ref speaker (during overlap any
             covering hyp speaker counts)
  seat_acc   on-camera seat accuracy, NO mapping help: of ref words said by an on-camera person, the share
             whose hyp speaker carries that person's seat ("left" / "right"); this is the "did vision get who
             was talking" number
Text normalisation (both sides): Whisper's EnglishTextNormalizer (numbers, spelling) and fillers removed.

Inputs: --ref CASE_DIR (ref.words.json, ref.seglst.json, meta.json)
        --hyp transcript.json (unspoken/transcribe/contracts.py Transcript), or
        --asr asr.json (WER only), or --diar diar.json (DER only)
Usage: python -m eval.transcribe.score --ref testdata/ami/IS1009a-L-2p-dev --hyp out/transcript.json [--json]
"""
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
from scipy.optimize import linear_sum_assignment

FILLERS = {"uh", "um", "uhm", "umm", "hmm", "hm", "mm", "mmm", "mhm", "er", "erm", "ah", "eh", "uh-huh",
           "mm-hmm", "huh"}
FRAME = 0.01
COLLAR = 0.25


@lru_cache(maxsize=1)
def _normalizer():
    from whisper_normalizer.english import EnglishTextNormalizer
    return EnglishTextNormalizer()


def normalize(text: str) -> str:
    t = re.sub(r"\b(" + "|".join(re.escape(f) for f in sorted(FILLERS, key=len, reverse=True)) + r")\b", " ",
               text.lower())
    t = _normalizer()(t)
    return " ".join(w for w in t.split() if w not in FILLERS)


def _errors(ref: str, hyp: str) -> tuple[int, int]:
    """(errors, ref word count) for one pair of normalised strings."""
    n = len(ref.split())
    if not ref and not hyp:
        return 0, 0
    if not ref:
        return len(hyp.split()), 0
    if not hyp:
        return n, n
    o = jiwer.process_words(ref, hyp)
    return o.substitutions + o.deletions + o.insertions, n


def wer(ref: str, hyp: str) -> float:
    e, n = _errors(normalize(ref), normalize(hyp))
    return e / max(n, 1)


def cpwer(ref_by_spk: dict[str, str], hyp_by_spk: dict[str, str]) -> tuple[float, dict[str, str]]:
    """Returns (cpWER, hyp speaker -> ref speaker mapping)."""
    rs, hs = list(ref_by_spk), list(hyp_by_spk)
    k = max(len(rs), len(hs))
    rn = [normalize(ref_by_spk[s]) for s in rs] + [""] * (k - len(rs))
    hn = [normalize(hyp_by_spk[s]) for s in hs] + [""] * (k - len(hs))
    cost = np.array([[_errors(r, h)[0] for h in hn] for r in rn], dtype=float)
    ri, hi = linear_sum_assignment(cost)
    total = sum(len(r.split()) for r in rn)
    mapping = {hs[j]: rs[i] for i, j in zip(ri, hi) if i < len(rs) and j < len(hs)}
    return float(cost[ri, hi].sum()) / max(total, 1), mapping


def _frames(turns: list[tuple[float, float, str]], spk: list[str], n: int) -> np.ndarray:
    m = np.zeros((len(spk), n), dtype=bool)
    idx = {s: i for i, s in enumerate(spk)}
    for a, b, s in turns:
        m[idx[s], max(0, int(round(a / FRAME))):min(n, int(round(b / FRAME)))] = True
    return m


def der(ref_turns: list[tuple[float, float, str]], hyp_turns: list[tuple[float, float, str]],
        collar: float = COLLAR) -> dict[str, float]:
    end = max([b for _, b, _ in ref_turns + hyp_turns] + [0.0])
    n = int(np.ceil(end / FRAME)) + 1
    rs, hs = sorted({s for *_, s in ref_turns}), sorted({s for *_, s in hyp_turns})
    R, H = _frames(ref_turns, rs, n), _frames(hyp_turns, hs or ["_"], n)
    keep = np.ones(n, dtype=bool)
    c = int(round(collar / FRAME))
    for a, b, _ in ref_turns:
        for t in (a, b):
            f = int(round(t / FRAME))
            keep[max(0, f - c):min(n, f + c)] = False
    R, H = R[:, keep], H[:, keep]
    co = R.astype(np.int32) @ H.T.astype(np.int32)                # overlap frames per (ref, hyp)
    ri, hi = linear_sum_assignment(-co)
    nr, nh = R.sum(0), H.sum(0)
    total = float(nr.sum())
    miss = float(np.maximum(nr - nh, 0).sum())
    fa = float(np.maximum(nh - nr, 0).sum())
    mapped = np.zeros(R.shape[1], dtype=np.int64)
    for i, j in zip(ri, hi):
        mapped += (R[i] & H[j])
    conf = float((np.minimum(nr, nh) - mapped).sum())
    t = max(total, 1.0)
    return {"der": (miss + fa + conf) / t, "miss": miss / t, "false_alarm": fa / t, "confusion": conf / t}


def _hyp_spans(tr: dict[str, Any]) -> list[tuple[float, float, str]]:
    """Word spans when the transcript has word timings, utterance spans otherwise."""
    spans = []
    for u in tr["utterances"]:
        ws = u.get("words") or []
        if ws:
            spans += [(w["start"], max(w["end"], w["start"] + 0.02), u["speaker"]) for w in ws]
        else:
            spans.append((u["start"], u["end"], u["speaker"]))
    return spans


def _speakers_at(spans: list[tuple[float, float, str]], t: float, slack: float = 0.15) -> set[str]:
    """Hyp speakers whose spans cover t (several during overlap); else the nearest within `slack`."""
    cover = {s for a, b, s in spans if a <= t <= b}
    if cover:
        return cover
    best, gap = None, slack
    for a, b, s in spans:
        d = min(abs(a - t), abs(b - t))
        if d < gap:
            best, gap = s, d
    return {best} if best else set()


def attribution(ref_words: list[dict], tr: dict[str, Any], mapping: dict[str, str]) -> dict[str, float]:
    spans = sorted(_hyp_spans(tr))
    seats = {k: (v or {}).get("seat") for k, v in tr.get("speakers", {}).items()}
    ok = n = seat_ok = seat_n = 0
    for w in ref_words:
        if normalize(w["text"]) == "":
            continue
        here = _speakers_at(spans, (w["start"] + w["end"]) / 2)
        n += 1
        ok += int(any(mapping.get(s) == w["speaker"] for s in here))
        if w.get("on_camera"):
            seat_n += 1
            seat_ok += int(any(seats.get(s) == w["seat"] for s in here))
    return {"sa_acc": ok / max(n, 1), "seat_acc": seat_ok / max(seat_n, 1) if seat_n else float("nan")}


def load_ref(case: Path) -> tuple[list[dict], list[dict], dict]:
    return (json.loads((case / "ref.words.json").read_text()), json.loads((case / "ref.seglst.json").read_text()),
            json.loads((case / "meta.json").read_text()))


def ref_turns(seglst: list[dict]) -> list[tuple[float, float, str]]:
    return [(s["start_time"], s["end_time"], s["speaker"]) for s in seglst]


def score_transcript(case: Path, tr: dict[str, Any]) -> dict[str, Any]:
    words, seglst, meta = load_ref(case)
    ref_text = " ".join(s["words"] for s in sorted(seglst, key=lambda s: s["start_time"]))  # utterance order
    utts = sorted(tr["utterances"], key=lambda u: u["start"])
    hyp_text = " ".join(u["text"] for u in utts)
    by_ref: dict[str, list[str]] = {}
    for w in words:
        by_ref.setdefault(w["speaker"], []).append(w["text"])
    by_hyp: dict[str, list[str]] = {}
    for u in utts:
        by_hyp.setdefault(u["speaker"], []).append(u["text"])
    cp, mapping = cpwer({k: " ".join(v) for k, v in by_ref.items()}, {k: " ".join(v) for k, v in by_hyp.items()})
    hyp_turns = [(u["start"], u["end"], u["speaker"]) for u in utts]
    out = {"case": meta["case"], "wer": wer(ref_text, hyp_text), "cpwer": cp, **der(ref_turns(seglst), hyp_turns),
           **attribution(words, tr, mapping), "mapping": mapping, "ref_speakers": len(by_ref),
           "hyp_speakers": len(by_hyp)}
    return out


def score_asr(case: Path, asr: dict[str, Any]) -> dict[str, Any]:
    _, seglst, meta = load_ref(case)
    hyp = " ".join(s["text"] for s in sorted(asr["segments"], key=lambda s: s["start"]))
    ref = " ".join(s["words"] for s in sorted(seglst, key=lambda s: s["start_time"]))
    return {"case": meta["case"], "wer": wer(ref, hyp)}


def score_diar(case: Path, diar: dict[str, Any]) -> dict[str, Any]:
    _, seglst, meta = load_ref(case)
    hyp = [(t["start"], t["end"], t["voice"]) for t in diar["turns"]]
    return {"case": meta["case"], **der(ref_turns(seglst), hyp), "hyp_speakers": len({h[2] for h in hyp}),
            "ref_speakers": len({s["speaker"] for s in seglst})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--hyp", type=Path)
    g.add_argument("--asr", type=Path)
    g.add_argument("--diar", type=Path)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.hyp:
        r = score_transcript(a.ref, json.loads(a.hyp.read_text()))
    elif a.asr:
        r = score_asr(a.ref, json.loads(a.asr.read_text()))
    else:
        r = score_diar(a.ref, json.loads(a.diar.read_text()))
    if a.json:
        print(json.dumps(r))
    else:
        for k, v in r.items():
            print(f"{k:12s} {v:.3f}" if isinstance(v, float) else f"{k:12s} {v}")


if __name__ == "__main__":
    main()
