"""NVIDIA Parakeet TDT 0.6B v3 on CPU through onnx-asr (numpy + onnxruntime, no torch), with word timings.

Model: the public ONNX export istupakov/parakeet-tdt-0.6b-v3-onnx (CC-BY-4.0, same licence as the NeMo
checkpoint nvidia/parakeet-tdt-0.6b-v3). fp32 (encoder 2.4 GB) by default: the int8 export (650 MB,
quantization="int8") returned NO tokens for some 10 s windows of clear speech on this x86 box (onnxruntime
1.31 CPU) and fewer tokens for others, so it loses whole sentences while running no faster here (RTF 0.045 vs
0.05 at 6 threads). The export is full-attention, so the recording is cut into speech windows by our Silero
VAD (asr_chunks: live settings, neighbours merged up to `max_s`), each window is decoded greedily and token
emission frames (80 ms) become word timings: a word starts at its first token's frame, ends at the next
token's frame (or +160 ms at a gap). Words are regrouped into segments at pauses > 0.6 s (`split_gap_s`). `offset_s` shifts every time by a constant measured on the -dev cases
(TDT emits a token a frame or two after the sound starts).

GB10: the same model id runs on the GPU through NeMo-Speech.cpp (config/models.toml gb10.asr); this module
is the CPU reference path and the dev-box fallback. onnx-asr also runs it on CUDA/TensorRT providers.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .asr_chunks import cut, merge_chunks, speech_chunks
from .contracts import AsrResult, AsrSegment, Word

MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
FRAME_S = 0.08                                   # encoder frame after 8x subsampling of 10 ms features
_MODELS: dict[tuple, Any] = {}


def models_root() -> Path:
    """Where the ONNX files live: $UNSPOKEN_DEV_MODELS/asr, else the session's dev-models/asr next to the repo
    (or next to the worktrees dir), else ~/.cache/unspoken/dev-models/asr. Missing files are downloaded there."""
    env = os.environ.get("UNSPOKEN_DEV_MODELS")
    if env:
        return Path(env) / "asr"
    repo = Path(__file__).resolve().parents[2]
    for p in (repo.parent, repo.parent.parent):
        if (p / "dev-models").is_dir():
            return p / "dev-models" / "asr"
    return Path.home() / ".cache" / "unspoken" / "dev-models" / "asr"


def model_dir(quantization: str | None) -> Path:
    return models_root() / ("parakeet-tdt-0.6b-v3-int8" if quantization == "int8" else "parakeet-tdt-0.6b-v3")


def load(*, threads: int, quantization: str | None = None, path: str | Path | None = None):
    """A cached onnx-asr Parakeet with token timestamps + log-probs. Downloads into `path` when missing."""
    key = (threads, quantization, str(path))
    if key not in _MODELS:
        import onnx_asr                       # lazy
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        _MODELS[key] = onnx_asr.load_model(MODEL_NAME, path or model_dir(quantization), quantization=quantization,
                                           sess_options=so, providers=["CPUExecutionProvider"]).with_timestamps()
    return _MODELS[key]


def _strip(tok: str) -> str:
    return tok.replace("▁", " ").strip()


def words_from_tokens(tokens: list[str], times: list[float], logprobs: list[float] | None, offset: float = 0.0,
                      *, gap_s: float = 0.5, tail_s: float = 2 * FRAME_S, min_word_s: float = FRAME_S) -> list[Word]:
    """Group SentencePiece tokens (a leading "▁" or " " starts a word) into Words with start/end/conf on the
    recording clock."""
    groups: list[list[int]] = []
    pending_boundary = True
    for i, tok in enumerate(tokens):
        text = _strip(tok)
        if not text:                                     # a lone "▁"/" ": a word boundary without content
            pending_boundary = True
            continue
        if tok.startswith(("▁", " ")) or pending_boundary or not groups:
            groups.append([i])
        else:
            groups[-1].append(i)
        pending_boundary = False
    ends = []
    for i in range(len(tokens)):
        nxt = times[i + 1] if i + 1 < len(tokens) else None
        ends.append(nxt if nxt is not None and nxt - times[i] <= gap_s else times[i] + tail_s)
    words: list[Word] = []
    for g in groups:
        start = times[g[0]] + offset
        end = max(ends[g[-1]] + offset, start + min_word_s)
        text = "".join(_strip(tokens[i]) for i in g)
        conf = math.exp(float(np.mean([logprobs[i] for i in g]))) if logprobs else 1.0
        words.append(Word(round(start, 3), round(end, 3), text, round(min(1.0, conf), 3)))
    return words


def split_segments(words: list[Word], *, split_gap_s: float = 0.6) -> list[AsrSegment]:
    """Cut a window's words into segments at pauses longer than `split_gap_s` (a 30 s VAD window is too coarse
    an utterance for anything that does not look at words). Segment text = its words joined by spaces."""
    out: list[AsrSegment] = []
    group: list[Word] = []
    for w in words:
        if group and w.start - group[-1].end > split_gap_s:
            out.append(_segment(group))
            group = []
        group.append(w)
    if group:
        out.append(_segment(group))
    return out


def _segment(words: list[Word]) -> AsrSegment:
    conf = float(np.mean([w.conf for w in words]))
    return AsrSegment(words[0].start, max(w.end for w in words), " ".join(w.text for w in words), list(words), round(conf, 3))


def transcribe(audio: np.ndarray, *, threads: int = 6, quantization: str | None = None, offset_s: float = 0.0,
               max_s: float = 30.0, gap_s: float = 1.0, batch_size: int = 4, split_gap_s: float = 0.6,
               vad: dict | None = None, path: str | Path | None = None, **_: Any) -> AsrResult:
    model = load(threads=threads, quantization=quantization, path=path)
    t0 = time.perf_counter()
    chunks = merge_chunks(speech_chunks(audio, threads=1, **(vad or {})), max_s=max_s, gap_s=gap_s)
    t1 = time.perf_counter()
    segments: list[AsrSegment] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        results = model.recognize([cut(audio, c) for c in batch], sample_rate=16000)
        for (a, b), r in zip(batch, results):
            words = words_from_tokens(list(r.tokens or []), list(r.timestamps or []), r.logprobs, a + offset_s)
            words = [w for w in words if w.start < b + 1.0]
            if words:
                segments += split_segments(words, split_gap_s=split_gap_s)
            elif r.text.strip():                            # tokens without timestamps: keep the window as one segment
                segments.append(AsrSegment(round(a + offset_s, 3), round(b + offset_s, 3), r.text.strip(), [], 1.0))
    q = quantization or "fp32"
    return AsrResult(f"Parakeet TDT 0.6B v3 (onnx-asr, CPU {q}, Silero VAD windows <= {max_s:g} s, greedy)",
                     segments, {"vad_s": round(t1 - t0, 3), "decode_s": round(time.perf_counter() - t1, 3),
                                "chunks": len(chunks), "offset_s": offset_s})


__all__ = ["MODEL_NAME", "FRAME_S", "models_root", "model_dir", "load", "words_from_tokens", "split_segments", "transcribe"]


class ParakeetASR:
    """Registry ASR provider (unspoken.contracts.ASR) for the live path: one closed VAD segment per call, Parakeet
    fp32 on the CPU via onnx-asr. Config: provider = "python:unspoken.transcribe.asr_parakeet:ParakeetASR",
    options = { threads = 4, quantization = "" }. On the GB10 use the speech server (openai-audio) instead."""

    def __init__(self, spec: Any = None, **options: Any) -> None:
        from ..sensors.config import merge_options
        opts = merge_options(spec, options)
        self.threads = int(opts.get("threads", 4))
        self.quantization = opts.get("quantization") or None
        self.path = opts.get("path")
        self.label = f"Parakeet TDT 0.6B v3 (onnx-asr, CPU {self.quantization or 'fp32'})"
        self._lock = None

    def _model(self):
        return load(threads=self.threads, quantization=self.quantization, path=self.path)

    def _sync(self, audio: np.ndarray, lang: str | None):
        from ..contracts import Transcript as LiveTranscript, Word as LiveWord
        t0 = time.perf_counter()
        r = self._model().recognize([audio], sample_rate=16000)[0]
        words = words_from_tokens(list(r.tokens or []), list(r.timestamps or []), r.logprobs, 0.0)
        text = " ".join(w.text for w in words) if words else (r.text or "").strip()
        conf = float(np.mean([w.conf for w in words])) if words else 0.0
        return LiveTranscript(text=text, lang=lang or "en", conf=round(conf, 3),
                              words=[LiveWord(w.text, w.start, w.end) for w in words],
                              latency_ms=round((time.perf_counter() - t0) * 1000, 1))

    async def transcribe(self, pcm16: bytes, sample_rate: int, lang: str | None = None):
        import asyncio

        from ..contracts import Transcript as LiveTranscript
        if sample_rate != 16000:
            from ..sensors.audio import resample_pcm16
            pcm16 = resample_pcm16(pcm16, sample_rate, 16000)
        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        if len(audio) < 1600:                                  # < 0.1 s: nothing to say
            return LiveTranscript(text="", lang=lang or "en", conf=0.0)
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            return await asyncio.to_thread(self._sync, audio, lang)

    async def warm(self) -> None:
        import asyncio
        await asyncio.to_thread(self._model)

    async def healthy(self) -> bool:
        try:
            await self.warm()
            return True
        except Exception:
            return False
