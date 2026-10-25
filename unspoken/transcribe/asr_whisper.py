"""faster-whisper (CTranslate2) backends with word timings.

whole_file : one `transcribe` over the whole recording with faster-whisper's built-in Silero VAD filter
             (it drops non-speech and packs the rest into <=30 s windows), `word_timestamps=True`.
chunked    : our own Silero VAD chunks (asr_chunks.speech_chunks, the live pipeline's settings) decoded one
             chunk at a time, no VAD filter, no conditioning on previous text; word times shifted onto the
             recording clock. This is what the live path does, so its WER is the live ceiling for whisper.

Models are the public CTranslate2 conversions faster-whisper resolves by short name: base.en / small.en
(Systran), large-v3-turbo (mobiuslabsgmbh), distil-large-v3.5 (distil-whisper). All MIT-licensed weights.
CPU: int8. GPU (GB10): ctranslate2 aarch64 wheels have no CUDA, see docs/reports/tx-asr.md.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .asr_chunks import cut, speech_chunks
from .contracts import AsrResult, AsrSegment, Word

MODEL_IDS = {"base.en": "Systran/faster-whisper-base.en", "small.en": "Systran/faster-whisper-small.en",
             "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
             "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2"}

_MODELS: dict[tuple, Any] = {}


def load(size: str, *, threads: int, device: str = "cpu", compute_type: str | None = None):
    """A cached WhisperModel; `size` is a short name from MODEL_IDS or any HF id / local dir."""
    compute_type = compute_type or ("int8" if device == "cpu" else "float16")
    key = (size, threads, device, compute_type)
    if key not in _MODELS:
        from faster_whisper import WhisperModel  # lazy: slow import

        _MODELS[key] = WhisperModel(MODEL_IDS.get(size, size), device=device, compute_type=compute_type,
                                    cpu_threads=threads)
    return _MODELS[key]


def _segments(fw_segments, offset: float, min_word_s: float = 0.02) -> list[AsrSegment]:
    out: list[AsrSegment] = []
    for s in fw_segments:                              # generator: decoding happens here
        words = []
        for w in s.words or []:
            text = w.word.strip()
            if not text:
                continue
            start = float(w.start) + offset
            words.append(Word(round(start, 3), round(max(float(w.end) + offset, start + min_word_s), 3), text,
                              round(float(w.probability), 3)))
        text = s.text.strip()
        if not text:
            continue
        conf = float(np.clip(np.exp(float(s.avg_logprob)), 0.0, 1.0))
        out.append(AsrSegment(round(float(s.start) + offset, 3), round(float(s.end) + offset, 3), text, words,
                              round(conf, 3)))
    return out


def whole_file(audio: np.ndarray, size: str, *, threads: int = 6, beam_size: int = 5, language: str = "en",
               **_: Any) -> AsrResult:
    model = load(size, threads=threads)
    t0 = time.perf_counter()
    segs, _info = model.transcribe(audio, language=language, beam_size=beam_size, vad_filter=True,
                                   word_timestamps=True, condition_on_previous_text=False)
    segments = _segments(segs, 0.0)
    return AsrResult(f"faster-whisper {size} (CPU int8, whole file + built-in VAD filter, beam {beam_size})",
                     segments, {"decode_s": round(time.perf_counter() - t0, 3)})


def chunked(audio: np.ndarray, size: str, *, threads: int = 6, beam_size: int = 5, language: str = "en",
            vad: dict | None = None, **_: Any) -> AsrResult:
    model = load(size, threads=threads)
    t0 = time.perf_counter()
    chunks = speech_chunks(audio, threads=1, **(vad or {}))
    t1 = time.perf_counter()
    segments: list[AsrSegment] = []
    for a, b in chunks:
        segs, _info = model.transcribe(cut(audio, (a, b)), language=language, beam_size=beam_size,
                                       vad_filter=False, word_timestamps=True, condition_on_previous_text=False)
        segments += _segments(segs, a)
    return AsrResult(f"faster-whisper {size} (CPU int8, Silero VAD chunks as in the live pipeline, beam {beam_size})",
                     segments, {"vad_s": round(t1 - t0, 3), "decode_s": round(time.perf_counter() - t1, 3),
                                "chunks": len(chunks)})


__all__ = ["MODEL_IDS", "load", "whole_file", "chunked"]
