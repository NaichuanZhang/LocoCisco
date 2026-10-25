"""In-process ASR: faster-whisper (CTranslate2) implementing unspoken.contracts.ASR.

Options (RoleSpec.model / RoleSpec.options): model="Systran/faster-whisper-base.en" (HF id or local dir),
device="cpu"|"cuda"|"auto", compute_type="int8" (CPU) | "float16"/"int8_float16" (GPU), language="en"
(None = detect), beam_size=1, threads=4, vad_filter=False (the pipeline already segments), download_root.

The model loads lazily on the first call (or `warm()`), and `transcribe` runs the blocking decode in a
worker thread so the capture loop is never blocked. Calls are serialised with a lock: CTranslate2 is
thread-safe but one model decoding two segments at once on a CPU only slows both.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import numpy as np

from ..contracts import RoleSpec, Transcript, Word
from .config import merge_options
from .audio import pcm16_to_float

DEFAULT_MODEL = "Systran/faster-whisper-base.en"


class FasterWhisperASR:
    def __init__(self, spec: RoleSpec | None = None, **options: Any) -> None:
        opts = merge_options(spec, options)
        self.model_name = (spec.model if spec is not None and spec.model else None) or opts.get("model") or DEFAULT_MODEL
        self.device = str(opts.get("device", "cpu"))
        self.compute_type = str(opts.get("compute_type", "int8" if self.device == "cpu" else "float16"))
        self.language: str | None = opts.get("language", "en") or None
        self.beam_size = int(opts.get("beam_size", 1))
        self.threads = int(opts.get("threads", 4))
        self.vad_filter = bool(opts.get("vad_filter", False))
        self.download_root = opts.get("download_root")
        self.local_files_only = bool(opts.get("local_files_only", os.environ.get("HF_HUB_OFFLINE") == "1"))
        self._model = None
        self._lock: asyncio.Lock | None = None
        self.label = f"faster-whisper {self.model_name.split('/')[-1]} ({self.device}, {self.compute_type})"

    # ------------------------------------------------------------------ model
    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy: ctranslate2 import is slow

            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type,
                                       cpu_threads=self.threads, download_root=self.download_root,
                                       local_files_only=self.local_files_only)
        return self._model

    async def warm(self) -> None:
        await asyncio.to_thread(self._load)

    def _transcribe_sync(self, audio: np.ndarray, lang: str | None) -> Transcript:
        model = self._load()
        t0 = time.perf_counter()
        segments, info = model.transcribe(audio, language=lang, beam_size=self.beam_size,
                                          vad_filter=self.vad_filter, word_timestamps=True,
                                          condition_on_previous_text=False)
        words: list[Word] = []
        texts: list[str] = []
        logprobs: list[float] = []
        for seg in segments:  # generator: decoding happens here
            texts.append(seg.text.strip())
            logprobs.append(float(seg.avg_logprob))
            for w in seg.words or []:
                words.append(Word(text=w.word.strip(), start=float(w.start), end=float(w.end)))
        text = " ".join(t for t in texts if t)
        conf = float(np.clip(np.exp(np.mean(logprobs)), 0.0, 1.0)) if logprobs else 0.0
        return Transcript(text=text, lang=(info.language or lang or "en"), conf=conf, words=words,
                          latency_ms=(time.perf_counter() - t0) * 1000)

    # ------------------------------------------------------------------ ASR protocol
    async def transcribe(self, pcm16: bytes, sample_rate: int, lang: str | None = None) -> Transcript:
        if sample_rate != 16000:
            from .audio import resample_pcm16
            pcm16 = resample_pcm16(pcm16, sample_rate, 16000)
        audio = pcm16_to_float(pcm16)
        if len(audio) < 1600:  # < 0.1 s: nothing to say
            return Transcript(text="", lang=lang or self.language or "en", conf=0.0)
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            return await asyncio.to_thread(self._transcribe_sync, audio, lang or self.language)

    async def healthy(self) -> bool:
        try:
            await self.warm()
            return True
        except Exception:
            return False


__all__ = ["FasterWhisperASR", "DEFAULT_MODEL"]
