"""NVIDIA Streaming Sortformer (nvidia/diar_streaming_sortformer_4spk-v2, CC-BY-4.0) -> DiarResult.

    python -m unspoken.transcribe.diar_sortformer --in audio.wav --out diar.json [--latency high] [--threads 6]

End-to-end neural diarizer: 4 sigmoid outputs per 80 ms frame (<= 4 speakers, overlap-aware), streamed over chunks
with an arrival-order speaker cache, so labels are ordered by first appearance and only the past + a short right
context is used. Presets are the model card's streaming configurations (frames of 80 ms; latency = chunk +
right context, compute not included):

  veryhigh  chunk 340 / rc 40 / fifo 40  / update 300 / cache 188  -> 30.4 s
  high      chunk 124 / rc 1  / fifo 124 / update 124 / cache 188  -> 10.0 s
  low       chunk 6   / rc 7  / fifo 188 / update 144 / cache 188  -> 1.04 s
  ultra     chunk 3   / rc 1  / fifo 188 / update 144 / cache 188  -> 0.32 s

Needs `nemo_toolkit[asr]` + torch (NOT in requirements/transcribe-diar.txt: heavy; see docs/reports/tx-diar.md).
On the GB10 the same weights ship as a GGUF for NeMo-Speech.cpp (`nemo-speech diarize`), which is the GPU path.
Model files land in $HF_HOME (default below) on first use; nothing else is written.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .contracts import DiarResult, VoiceTurn, to_json

MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2"
DEFAULT_HF_HOME = "/local/home/zhangnai/kirocrew-sessions/2026-10-23-dell-nvidia-hackathon/dev-models/diar/hf-cache"
FRAME_S = 0.08
PRESETS: dict[str, dict[str, int]] = {
    "veryhigh": dict(chunk_len=340, chunk_right_context=40, fifo_len=40, spkcache_update_period=300, spkcache_len=188),
    "high": dict(chunk_len=124, chunk_right_context=1, fifo_len=124, spkcache_update_period=124, spkcache_len=188),
    "low": dict(chunk_len=6, chunk_right_context=7, fifo_len=188, spkcache_update_period=144, spkcache_len=188),
    "ultra": dict(chunk_len=3, chunk_right_context=1, fifo_len=188, spkcache_update_period=144, spkcache_len=188),
}

_MODEL = None


def load_model(threads: int = 6):
    global _MODEL
    if _MODEL is None:
        os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
        import torch
        torch.set_num_threads(threads)
        from nemo.collections.asr.models import SortformerEncLabelModel
        m = SortformerEncLabelModel.from_pretrained(MODEL_ID, map_location="cpu")
        m.eval()
        _MODEL = m
    return _MODEL


def parse_segments(segs) -> list[tuple[float, float, int]]:
    """NeMo returns strings "begin end speaker_k" (or lists); normalise to (start, end, k)."""
    out = []
    for s in segs:
        if isinstance(s, str):
            a, b, spk = s.split()[:3]
        else:
            a, b, spk = s[0], s[1], s[2]
        k = int(str(spk).replace("speaker_", ""))
        out.append((float(a), float(b), k))
    return out


def to_result(segs: list[tuple[float, float, int]], total_s: float, label: str, wall: float, latency_s: float) -> DiarResult:
    from .diar import rename_by_first_appearance, voice_name
    turns = rename_by_first_appearance([VoiceTurn(round(a, 3), round(b, 3), voice_name(k), 1.0) for a, b, k in segs if b > a])
    return DiarResult(label, turns, {"wall_s": round(wall, 3), "audio_s": round(total_s, 3),
                                     "rtf": round(wall / max(total_s, 1e-6), 4), "label_latency_s": latency_s})


def diarize_sortformer(x: np.ndarray, threads: int = 6, latency: str = "high") -> DiarResult:
    import torch
    m = load_model(threads)
    p = PRESETS[latency]
    for k, v in p.items():
        setattr(m.sortformer_modules, k, v)
    m.sortformer_modules._check_streaming_parameters()
    t0 = time.perf_counter()
    with torch.inference_mode():
        segs = m.diarize(audio=[x.astype(np.float32)], batch_size=1, sample_rate=16000, verbose=False)
    wall = time.perf_counter() - t0
    lat = (p["chunk_len"] + p["chunk_right_context"]) * FRAME_S
    return to_result(parse_segments(segs[0]), len(x) / 16000, f"{MODEL_ID} (NeMo, CPU, streaming preset '{latency}')", wall,
                     round(lat, 2))


def main(argv=None) -> DiarResult:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--latency", default="high", choices=sorted(PRESETS))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "6")))
    a = ap.parse_args(argv)
    from .diar import read_wav
    load1 = os.getloadavg()[0]
    x = read_wav(a.inp)
    res = diarize_sortformer(x, a.threads, a.latency)
    res.timing.update({"threads": a.threads, "load1_at_start": round(load1, 2), "load1_at_end": round(os.getloadavg()[0], 2)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    to_json(res, a.out)
    print(json.dumps({"model": res.model, "turns": len(res.turns), "voices": len({t.voice for t in res.turns}), **res.timing}))
    return res


if __name__ == "__main__":
    main()
