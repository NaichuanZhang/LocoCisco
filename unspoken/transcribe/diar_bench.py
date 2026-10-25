"""Bench + tuner for unspoken.transcribe.diar (workstream tx-diar).

  python -m unspoken.transcribe.diar_bench tune [--embedder campp] [--online]      # grid on the -dev excerpts
  python -m unspoken.transcribe.diar_bench run --methods baseline,window-campp,... [--cases all|dev|full]
  python -m unspoken.transcribe.diar_bench table                                   # markdown from scores.json files

Outputs: testdata/ami-runs/diar/<method>/<case>/diar.json + scores.json (score_diar + timing + speaker-count error).
The L-room and R-room cases share one far-field recording, so the tuner averages over distinct audio only.
Tuning happens on -dev excerpts only; `run` on full cases is the one-shot report number.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from eval.transcribe.score import score_diar
from .contracts import to_json
from . import diar as D

ROOT = Path("/local/home/zhangnai/kirocrew-sessions/2026-10-23-dell-nvidia-hackathon")
TESTDATA = ROOT / "testdata" / "ami"
BASELINE = ROOT / "testdata" / "ami-baseline"
RUNS = ROOT / "testdata" / "ami-runs" / "diar"
CASES_FULL = ["IS1009a-duo-AC", "IS1009a-duo-DB", "IS1009a-L-room", "IS1009a-R-room"]
CASES_DEV = [c + "-dev" for c in CASES_FULL]

# method name -> CLI argv for unspoken.transcribe.diar (tuned values live here; see docs/reports/tx-diar.md)
METHODS: dict[str, list[str]] = {
    # offline, tuned on the -dev excerpts (testdata/ami-runs/diar/tune-*.json):
    "window-campp": ["--method", "window", "--embedder", "campp", "--cluster", "spectral"],
    "window-resnet34": ["--method", "window", "--embedder", "resnet34", "--cluster", "spectral", "--vad-threshold", "0.35"],
    "window-ahc-campp": ["--method", "window", "--embedder", "campp", "--cluster", "ahc", "--threshold", "0.85"],
    # online (causal), tuned on -dev: join >= 0.6, mint < 0.4, at most 4 voices, no lookback (latency = 1.5 s window)
    "online-campp": ["--method", "window", "--online", "--embedder", "campp", "--online-threshold", "0.6",
                     "--online-mint-threshold", "0.4", "--max-speakers", "4", "--vad-threshold", "0.35"],
    # window clustering on pyannote/segmentation-3.0 activity (overlap-aware segmenter, identities ignored) instead of Silero
    "window-campp-pa": ["--method", "window", "--embedder", "campp", "--cluster", "spectral", "--vad", "pyannote"],
    "window-campp-pa-ov": ["--method", "window", "--embedder", "campp", "--cluster", "spectral", "--vad", "pyannote", "--overlap"],
    # neural
    "pyannote-campp": ["--method", "pyannote", "--embedder", "campp", "--threshold", "0.9"],
    "sortformer": ["--latency", "high"],
}


def _cases(which: str) -> list[str]:
    return {"all": CASES_DEV + CASES_FULL, "dev": CASES_DEV, "full": CASES_FULL}.get(which) or which.split(",")


def _score(case: str, res_json: dict, extra: dict | None = None) -> dict:
    sc = score_diar(TESTDATA / case, res_json)
    sc["spk_count_err"] = sc["hyp_speakers"] - sc["ref_speakers"]
    sc["timing"] = res_json.get("timing", {})
    sc["model"] = res_json.get("model")
    if extra:
        sc.update(extra)
    return sc


def run_method(method: str, case: str, extra_args: list[str] | None = None, python_nemo: str | None = None) -> dict:
    out_dir = RUNS / method / case
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "diar.json"
    if method == "baseline":
        src = BASELINE / case / "diar.json"
        out.write_text(src.read_text())
    elif method == "sortformer":
        import subprocess
        py = python_nemo or os.environ.get("UNSPOKEN_NEMO_PYTHON") or sys.executable
        cmd = [py, "-m", "unspoken.transcribe.diar_sortformer", "--in", str(TESTDATA / case / "audio.wav"), "--out", str(out),
               "--threads", os.environ.get("OMP_NUM_THREADS", "6")] + (extra_args or [])
        subprocess.run(cmd, check=True, cwd=str(Path(__file__).resolve().parents[2]))
    else:
        argv = ["--in", str(TESTDATA / case / "audio.wav"), "--out", str(out)] + METHODS[method] + (extra_args or [])
        D.main(argv)
    sc = _score(case, json.loads(out.read_text()), {"method": method, "load1": round(os.getloadavg()[0], 2),
                                                   "threads": int(os.environ.get("OMP_NUM_THREADS", "6"))})
    (out_dir / "scores.json").write_text(json.dumps(sc, indent=1))
    print(f"{method:24s} {case:20s} DER {sc['der']:.3f} (miss {sc['miss']:.3f} fa {sc['false_alarm']:.3f} "
          f"conf {sc['confusion']:.3f}) spk {sc['hyp_speakers']}/{sc['ref_speakers']} rtf {sc['timing'].get('rtf', float('nan')):.4f}")
    return sc


# ----------------------------------------------------------------------------- tuning
def _audio_key(case: str) -> str:
    return hashlib.sha1((TESTDATA / case / "audio.wav").read_bytes()[:4_000_000]).hexdigest()[:10]


def tune(embedder: str, online: bool, cases: list[str], threads: int, grid_json: str | None) -> None:
    seen: dict[str, str] = {}
    uniq = []
    for c in cases:
        k = _audio_key(c)
        if k not in seen:
            seen[k] = c
            uniq.append(c)
    print(f"tuning on distinct audio: {uniq}")
    base = D.WindowSettings()
    vad_grid = [0.35, 0.5]
    if grid_json:
        g = json.loads(grid_json)
    elif online:
        g = {"online_threshold": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75], "lookback": [0, 1, 2], "max_speakers": [4, 8]}
    else:
        g = {"threshold": [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7], "cluster": ["ahc", "spectral"],
             "min_cluster_s": [2.0, 5.0], "reseg": [True, False]}
    keys = list(g)
    combos = [dict(zip(keys, v)) for v in itertools.product(*[g[k] for k in keys])]
    emb = D.OnnxEmbedder(embedder, threads=threads)
    vad = D.SileroProbs(threads=threads)
    results: list[dict] = []
    for vt in vad_grid:
        prepared = {}
        for c in uniq:
            x = D.read_wav(TESTDATA / c / "audio.wav")
            prepared[c] = D.prepare_windows(x, vad, emb, replace(base, vad_threshold=vt), with_reseg=not online)
            print(f"  prepared {c}: {prepared[c].timing}")
        for combo in combos:
            s = replace(base, vad_threshold=vt, **combo)
            ders, errs = [], []
            for c in uniq:
                res, _ = D.finish_windows(prepared[c], s, online)
                sc = score_diar(TESTDATA / c, json.loads(to_json(res)))
                ders.append(sc["der"]); errs.append(sc["hyp_speakers"] - sc["ref_speakers"])
            results.append({"vad_threshold": vt, **combo, "mean_der": float(np.mean(ders)),
                            "der": [round(d, 3) for d in ders], "spk_err": errs})
            print(f"  vad {vt} {combo} -> mean DER {np.mean(ders):.3f} {[round(d, 3) for d in ders]} spk_err {errs}")
    results.sort(key=lambda r: r["mean_der"])
    print("\nBEST 10:")
    for r in results[:10]:
        print(json.dumps(r))
    out = RUNS / f"tune-{'online' if online else 'offline'}-{embedder}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": uniq, "results": results}, indent=1))
    print(f"wrote {out}")


def tune_pyannote(embedder: str, cases: list[str], threads: int) -> None:
    uniq, seen = [], set()
    for c in cases:
        k = _audio_key(c)
        if k not in seen:
            seen.add(k); uniq.append(c)
    results = []
    xs = {c: D.read_wav(TESTDATA / c / "audio.wav") for c in uniq}
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        ders, errs = [], []
        for c in uniq:
            res = D.diarize_pyannote(xs[c], embedder, thr, None, threads)
            sc = score_diar(TESTDATA / c, json.loads(to_json(res)))
            ders.append(sc["der"]); errs.append(sc["hyp_speakers"] - sc["ref_speakers"])
        results.append({"threshold": thr, "mean_der": float(np.mean(ders)), "der": [round(d, 3) for d in ders], "spk_err": errs})
        print(f"  pyannote thr {thr} -> mean DER {np.mean(ders):.3f} {[round(d, 3) for d in ders]} spk_err {errs}")
    results.sort(key=lambda r: r["mean_der"])
    out = RUNS / f"tune-pyannote-{embedder}.json"
    out.write_text(json.dumps({"cases": uniq, "results": results}, indent=1))
    print(f"wrote {out}")


# ----------------------------------------------------------------------------- table
def table(methods: list[str] | None = None, cases: list[str] | None = None) -> str:
    methods = methods or sorted(p.name for p in RUNS.iterdir() if p.is_dir())
    cases = cases or CASES_DEV + CASES_FULL
    rows = ["| method | case | DER | miss | FA | conf | spk hyp/ref | RTF | threads | load1 | latency s |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        for c in cases:
            p = RUNS / m / c / "scores.json"
            if not p.exists():
                continue
            s = json.loads(p.read_text())
            t = s.get("timing", {})
            rtf = t.get("rtf")
            rows.append(f"| {m} | {c}{' (dev)' if c.endswith('-dev') else ' (full)'} | {s['der']:.3f} | {s['miss']:.3f} | "
                        f"{s['false_alarm']:.3f} | {s['confusion']:.3f} | {s['hyp_speakers']}/{s['ref_speakers']} | "
                        f"{rtf:.4f} | {t.get('threads', s.get('threads', '-'))} | {t.get('load1_at_start', s.get('load1', '-'))} | "
                        f"{t.get('label_latency_s', '-')} |" if rtf is not None else
                        f"| {m} | {c} | {s['der']:.3f} | {s['miss']:.3f} | {s['false_alarm']:.3f} | {s['confusion']:.3f} | "
                        f"{s['hyp_speakers']}/{s['ref_speakers']} | n/a (replay) | - | - | - |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--embedder", default="campp")
    t.add_argument("--online", action="store_true")
    t.add_argument("--pyannote", action="store_true")
    t.add_argument("--cases", default="dev")
    t.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "6")))
    t.add_argument("--grid", default=None, help="JSON dict of WindowSettings field -> values")
    r = sub.add_parser("run")
    r.add_argument("--methods", required=True)
    r.add_argument("--cases", default="all")
    r.add_argument("--extra", default="", help="extra CLI args appended to every run")
    r.add_argument("--python-nemo", default=None)
    tb = sub.add_parser("table")
    tb.add_argument("--methods", default=None)
    tb.add_argument("--cases", default=None)
    a = ap.parse_args()
    if a.cmd == "tune":
        if a.pyannote:
            tune_pyannote(a.embedder, _cases(a.cases), a.threads)
        else:
            tune(a.embedder, a.online, _cases(a.cases), a.threads, a.grid)
    elif a.cmd == "run":
        t0 = time.time()
        for m in a.methods.split(","):
            for c in _cases(a.cases):
                run_method(m, c, a.extra.split() if a.extra else None, a.python_nemo)
        print(f"done in {time.time() - t0:.0f}s; load1 {os.getloadavg()[0]:.1f}")
    else:
        print(table(a.methods.split(",") if a.methods else None, _cases(a.cases) if a.cases else None))


if __name__ == "__main__":
    main()
