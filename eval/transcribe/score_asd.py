"""Score an asd.json (unspoken/transcribe/contracts.py AsdResult) against an AMI-style reference case.

The reference "speaking" signal per ON-CAMERA SEAT comes from ref.words.json (word spans of the person whose
meta.json seat is that seat; gaps shorter than `bridge` seconds inside a run of words are filled, so a pause
between two words of one utterance still counts as speaking, a 0.3 s+ silence does not). The hypothesis per seat
at a frame time t is the highest speaking probability among the hyp faces that carry that seat and are visible at
t (0 when no face of that seat is visible: a missed face is a miss). Seats are fixed by position in both files
and are NEVER learned from the reference.

Per seat:  acc  frame accuracy at p >= 0.5          f1  F1 of speaking frames at p >= 0.5
           ap   average precision of p vs speaking  prior  share of frames in which the seat speaks
           face_cov  share of frames with a visible face of that seat
Overall:   map  mean AP over seats   acc / f1  frame-weighted   which_seat_acc  over frames where exactly one
           on-camera seat speaks (ref), the share whose argmax-scored seat is that seat (a frame where no face
           is visible or none scores above 0 counts as wrong; `which_seat_n` frames, `which_seat_abstain` share
           of such undecided frames)
           talk_f1  F1 of "someone on camera speaks" (max over seats >= 0.5)

Usage: python -m eval.transcribe.score_asd --ref testdata/ami/IS1009a-duo-AC-dev --asd asd.json [--json] [--thr 0.5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

BRIDGE = 0.3


def speech_intervals(words: list[dict], speaker: str, bridge: float = BRIDGE) -> list[tuple[float, float]]:
    spans = sorted((w["start"], w["end"]) for w in words if w["speaker"] == speaker)
    out: list[tuple[float, float]] = []
    for a, b in spans:
        if out and a - out[-1][1] <= bridge:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def active(intervals: list[tuple[float, float]], times: np.ndarray) -> np.ndarray:
    on = np.zeros(len(times), dtype=bool)
    for a, b in intervals:
        on |= (times >= a) & (times <= b)
    return on


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def prf(y: np.ndarray, h: np.ndarray) -> dict[str, float]:
    tp = float((y & h).sum()); fp = float((~y & h).sum()); fn = float((y & ~h).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "acc": float((y == h).mean()) if len(y) else float("nan")}


def seat_scores(asd: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """times, {seat: max p per frame}, {seat: face visible per frame} from an AsdResult dict."""
    seat_of = {f["track_id"]: f.get("seat") for f in asd["faces"]}
    frames = asd["frames"]
    times = np.array([f["t"] for f in frames], dtype=float)
    seats = sorted({s for s in seat_of.values() if s})
    p = {s: np.zeros(len(frames), dtype=float) for s in seats}
    vis = {s: np.zeros(len(frames), dtype=bool) for s in seats}
    for i, f in enumerate(frames):
        for tid, v in f["scores"].items():
            s = seat_of.get(tid)
            if s:
                vis[s][i] = True
                p[s][i] = max(p[s][i], float(v))
    return times, p, vis


def score_asd(case: Path, asd: dict[str, Any], thr: float = 0.5, bridge: float = BRIDGE) -> dict[str, Any]:
    words = json.loads((case / "ref.words.json").read_text())
    meta = json.loads((case / "meta.json").read_text())
    times, p, vis = seat_scores(asd)
    ref_seats = {info["seat"]: spk for spk, info in meta["speakers"].items() if info.get("on_camera") and info.get("seat")}
    out: dict[str, Any] = {"case": meta["case"], "frames": int(len(times)), "fps": asd.get("fps"),
                           "hyp_faces": len(asd["faces"]), "seats": {}}
    ys, hs, ps = [], [], []
    y_by_seat: dict[str, np.ndarray] = {}
    for seat, spk in sorted(ref_seats.items()):
        y = active(speech_intervals(words, spk, bridge), times)
        y_by_seat[seat] = y
        ph = p.get(seat, np.zeros(len(times)))
        h = ph >= thr
        r = prf(y, h)
        r.update({"ap": average_precision(y, ph), "prior": float(y.mean()) if len(y) else float("nan"),
                  "face_cov": float(vis[seat].mean()) if seat in vis else 0.0, "ref_speaker": spk})
        out["seats"][seat] = r
        ys.append(y); hs.append(h); ps.append(ph)
    if ys:
        Y, Hh = np.concatenate(ys), np.concatenate(hs)
        out["acc"] = float((Y == Hh).mean())
        out["f1"] = prf(Y, Hh)["f1"]
        aps = [out["seats"][s]["ap"] for s in out["seats"] if not np.isnan(out["seats"][s]["ap"])]
        out["map"] = float(np.mean(aps)) if aps else float("nan")
        # which seat is talking, over frames where exactly one on-camera seat speaks
        seats = sorted(ref_seats)
        Ym = np.stack([y_by_seat[s] for s in seats])                 # (S, N)
        Pm = np.stack([p.get(s, np.zeros(len(times))) for s in seats])
        Vm = np.stack([vis.get(s, np.zeros(len(times), dtype=bool)) for s in seats])
        one = Ym.sum(0) == 1
        if one.any():
            truth = Ym[:, one].argmax(0)
            pred = Pm[:, one].argmax(0)
            decided = Vm[:, one].any(0) & (Pm[:, one].max(0) > 0)     # some face visible and scored above 0
            out["which_seat_acc"] = float(((pred == truth) & decided).mean())
            out["which_seat_abstain"] = float((~decided).mean())
            out["which_seat_n"] = int(one.sum())
        talk_y, talk_h = Ym.any(0), (Pm.max(0) >= thr)
        out["talk_f1"] = prf(talk_y, talk_h)["f1"]
    out["timing"] = asd.get("timing", {})
    return out


def flat(r: dict[str, Any]) -> dict[str, Any]:
    """One row for a table: overall metrics + per-seat ap/acc/f1/face_cov."""
    row = {k: r.get(k) for k in ("case", "map", "acc", "f1", "which_seat_acc", "which_seat_abstain", "talk_f1", "hyp_faces")}
    for s, v in r.get("seats", {}).items():
        for m in ("ap", "acc", "f1", "prior", "face_cov"):
            row[f"{s}_{m}"] = v[m]
    t = r.get("timing") or {}
    row.update({"rtf": t.get("rtf"), "threads": t.get("threads"), "wall_s": t.get("wall_s")})
    return row


def table(runs: Path, baseline: Path | None = None, cases_dir: Path | None = None, thr: float = 0.5) -> str:
    """Markdown table over testdata/ami-runs/asd/<method>/<case>/scores.json (+ the baseline asd.json rescored)."""
    rows: list[tuple[str, str, dict]] = []
    for sj in sorted(runs.glob("*/*/scores.json")):
        rows.append((sj.parent.parent.name, sj.parent.name, json.loads(sj.read_text())))
    if baseline is not None and cases_dir is not None:
        for asd in sorted(baseline.glob("*/asd.json")):
            case = asd.parent.name
            if (cases_dir / case / "meta.json").exists():
                rows.append(("baseline-live-5hz", case, score_asd(cases_dir / case, json.loads(asd.read_text()), thr)))
    order = {"baseline-live-5hz": 0, "motion": 1, "avcorr": 2, "lightasd-talkset": 3, "lightasd": 4}
    rows.sort(key=lambda r: (r[1].endswith("-dev"), r[1], order.get(r[0], 9), r[0]))
    f = lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"  # noqa: E731
    out = ["| case | set | method | mAP | AP left | AP right | acc | F1 | which-seat acc | talk F1 | face cov L/R | faces | RTF | threads | load start/end |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, c, r in rows:
        s, t = r.get("seats", {}), r.get("timing", {})
        out.append(f"| {c.replace('IS1009a-', '').replace('-dev', '')} | {'dev' if c.endswith('-dev') else 'FULL'} | {m} | "
                   f"{f(r.get('map'))} | {f(s.get('left', {}).get('ap'))} | {f(s.get('right', {}).get('ap'))} | "
                   f"{f(r.get('acc'))} | {f(r.get('f1'))} | {f(r.get('which_seat_acc'))} | {f(r.get('talk_f1'))} | "
                   f"{f(s.get('left', {}).get('face_cov'))}/{f(s.get('right', {}).get('face_cov'))} | {r.get('hyp_faces')} | "
                   f"{f(t.get('rtf'))} | {t.get('threads', '-')} | {t.get('load1_start', '-')}/{t.get('load1_end', '-')} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--asd", type=Path)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--bridge", type=float, default=BRIDGE)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--table", type=Path, help="runs dir (testdata/ami-runs/asd): print a markdown table of every scores.json")
    ap.add_argument("--baseline", type=Path, help="with --table: testdata/ami-baseline, rescored as baseline-live-5hz")
    ap.add_argument("--cases", type=Path, help="with --baseline: testdata/ami (reference cases)")
    a = ap.parse_args()
    if a.table:
        print(table(a.table, a.baseline, a.cases, a.thr))
        return
    if not (a.ref and a.asd):
        ap.error("--ref and --asd are required (or --table)")
    r = score_asd(a.ref, json.loads(a.asd.read_text()), a.thr, a.bridge)
    if a.json:
        print(json.dumps(r))
        return
    for k, v in r.items():
        if k in ("seats", "timing"):
            continue
        print(f"{k:20s} {v:.3f}" if isinstance(v, float) else f"{k:20s} {v}")
    for s, v in r["seats"].items():
        print(f"seat {s:6s} ap {v['ap']:.3f} acc {v['acc']:.3f} f1 {v['f1']:.3f} prec {v['precision']:.3f} "
              f"rec {v['recall']:.3f} prior {v['prior']:.3f} face_cov {v['face_cov']:.3f} (ref {v['ref_speaker']})")
    if r.get("timing"):
        print("timing", json.dumps(r["timing"]))


if __name__ == "__main__":
    main()
