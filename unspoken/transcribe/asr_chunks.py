"""Speech chunks for chunked ASR, cut by the repo's own Silero VAD segmenter (unspoken.sensors.vad).

`speech_chunks` with the defaults reproduces what the live pipeline feeds its ASR (threshold 0.5, 500 ms of
silence closes a segment, 200 ms pad, 20 s cap). `merge_chunks` joins neighbouring chunks into longer
windows (more context for the model, fewer calls) without crossing gaps longer than `gap_s`. Both work on
float32 16 kHz mono audio and return (start, end) seconds from the start of the input.
"""
from __future__ import annotations

import numpy as np

from ..sensors.audio import SAMPLE_RATE, float_to_pcm16
from ..sensors.vad import WINDOW, SileroVAD

# The live pipeline's VAD settings (config/models.toml dev-sensors / gb10 profiles + vad.py defaults).
LIVE_VAD = {"threshold": 0.5, "hysteresis": 0.15, "min_speech_ms": 250, "min_silence_ms": 500, "pad_ms": 200,
            "max_segment_s": 20.0}


def speech_chunks(audio: np.ndarray, *, threads: int = 1, feed_s: float = 1.0, **settings) -> list[tuple[float, float]]:
    """Silero VAD speech segments of `audio` (float32, 16 kHz). `settings` override LIVE_VAD keys."""
    opts = {**LIVE_VAD, **settings}
    vad = SileroVAD(options=opts, threads=threads)
    stream = vad.stream(SAMPLE_RATE)
    pcm = float_to_pcm16(np.asarray(audio, dtype=np.float32))
    step = max(WINDOW * 2, int(feed_s * SAMPLE_RATE) // WINDOW * WINDOW * 2)   # whole windows per feed
    out: list[tuple[float, float]] = []
    for pos in range(0, len(pcm), step):
        out += [(s.start, s.end) for s in stream.feed(pcm[pos: pos + step])]
    out += [(s.start, s.end) for s in stream.flush()]
    total = len(audio) / SAMPLE_RATE
    return [(max(0.0, a), min(total, b)) for a, b in out if b > a]


def merge_chunks(chunks: list[tuple[float, float]], *, max_s: float = 30.0, gap_s: float = 1.0) -> list[tuple[float, float]]:
    """Join consecutive chunks while the joined window stays under `max_s` and the gap under `gap_s`."""
    merged: list[tuple[float, float]] = []
    for a, b in chunks:
        if merged and a - merged[-1][1] <= gap_s and b - merged[-1][0] <= max_s:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def cut(audio: np.ndarray, chunk: tuple[float, float]) -> np.ndarray:
    a, b = chunk
    return np.asarray(audio[int(round(a * SAMPLE_RATE)): int(round(b * SAMPLE_RATE))], dtype=np.float32)


__all__ = ["LIVE_VAD", "speech_chunks", "merge_chunks", "cut"]
