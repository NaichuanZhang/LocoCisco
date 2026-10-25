"""The GPU path: our Silero windows posted to an OpenAI-compatible speech server, words from verbose_json.

On the GB10 the asr role is NeMo-Speech.cpp serving Parakeet TDT 0.6B v3 Q8 on the GPU (deploy/gb10/speech.sh,
127.0.0.1:8770, `POST /v1/audio/transcriptions`, `response_format=verbose_json` adds `words` with start/end).
This method keeps the windowing of asr_parakeet (so the offline transcript on the GB10 is cut the same way as
the CPU reference run here) and only swaps the decoder for the HTTP call. Any server speaking that route works
(whisper.cpp server, vLLM audio, ...). Words come back as {word, start, end[, probability]} relative to the
posted clip; they are shifted onto the recording clock. No model files, no torch; httpx only.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..models.audio import pcm16_to_wav
from ..sensors.audio import float_to_pcm16
from .asr_chunks import cut, merge_chunks, speech_chunks
from .contracts import AsrResult, AsrSegment, Word

DEFAULT_URL = "http://127.0.0.1:8770/v1"       # config/models.toml [profiles.gb10.asr] url


def words_from_body(body: dict[str, Any], offset: float, min_word_s: float = 0.02) -> list[Word]:
    raw = list(body.get("words") or [])
    if not raw:                                  # whisper.cpp style: words nested in segments
        for s in body.get("segments") or []:
            raw += list((s or {}).get("words") or [])
    out: list[Word] = []
    for w in raw:
        if not isinstance(w, dict):
            continue
        text = str(w.get("word", w.get("text", ""))).strip()
        if not text:
            continue
        start = float(w.get("start", 0.0)) + offset
        end = max(float(w.get("end", start)) + offset, start + min_word_s)
        out.append(Word(round(start, 3), round(end, 3), text, round(float(w.get("probability", 1.0)), 3)))
    return out


def transcribe(audio: np.ndarray, *, url: str = DEFAULT_URL, model: str = "parakeet-tdt-0.6b-v3",
               language: str = "en", max_s: float = 30.0, gap_s: float = 1.0, timeout_s: float = 60.0,
               vad: dict | None = None, extra: dict[str, Any] | None = None, **_: Any) -> AsrResult:
    import httpx  # lazy

    t0 = time.perf_counter()
    chunks = merge_chunks(speech_chunks(audio, threads=1, **(vad or {})), max_s=max_s, gap_s=gap_s)
    t1 = time.perf_counter()
    data = {"model": model, "response_format": "verbose_json", "timestamp_granularities[]": "word",
            "language": language, "diarization": "false", **{k: str(v) for k, v in (extra or {}).items()}}
    segments: list[AsrSegment] = []
    with httpx.Client(timeout=timeout_s) as client:
        for a, b in chunks:
            wav = pcm16_to_wav(float_to_pcm16(cut(audio, (a, b))), 16000)
            r = client.post(f"{url.rstrip('/')}/audio/transcriptions", data=data,
                            files={"file": ("audio.wav", wav, "audio/wav")})
            r.raise_for_status()
            body = r.json() if "json" in r.headers.get("content-type", "") else {"text": r.text}
            words = words_from_body(body, a)
            text = str(body.get("text", "")).strip() or " ".join(w.text for w in words)
            if not text:
                continue
            conf = float(np.mean([w.conf for w in words])) if words else 1.0
            segments.append(AsrSegment(round(a, 3), round(b, 3), text, words, round(conf, 3)))
    return AsrResult(f"{model} via speech server {url} (verbose_json words, Silero VAD windows <= {max_s:g} s)",
                     segments, {"vad_s": round(t1 - t0, 3), "decode_s": round(time.perf_counter() - t1, 3),
                                "chunks": len(chunks)})


__all__ = ["DEFAULT_URL", "words_from_body", "transcribe"]
