"""Online speaker diarization by embedding clustering (unspoken.contracts.Diarizer / DiarSession).

Model: WeSpeaker VoxCeleb CAM++ (Apache-2.0, 7.2M params, 512-d) exported to ONNX by the sherpa-onnx
project; ResNet34-LM (256-d) is a drop-in alternative. Input: 80-bin Kaldi fbank (25 ms / 10 ms, no dither,
int16-scale samples, per-utterance mean subtraction) from kaldi-native-fbank; runtime onnxruntime CPU
(aarch64 wheels for both). ~40-70 ms per 3 s segment on one x86 core.

Clustering is online and session-only: each segment's embedding is compared (cosine) with the running
centroid of every known voice; >= `threshold` joins that voice (and nudges its centroid), otherwise a new
label "voice-A", "voice-B", ... is minted. `max_voices` caps the number of labels: once reached, a new
embedding joins its nearest voice. reset() forgets everything; nothing is written anywhere.

Options: model_path (default: $UNSPOKEN_MODELS_DIR/wespeaker_en_voxceleb_CAM++.onnx, see fetch_models),
threshold=0.55, max_voices=8, min_seconds=0.6 (shorter audio -> "voice-?" with conf 0), centroid_momentum=0.8,
threads=1.
"""
from __future__ import annotations

import asyncio
import string
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import RoleSpec, VoiceLabel
from .config import merge_options
from .audio import pcm16_to_float
from .assets import resolve_model

DEFAULT_MODEL_FILE = "wespeaker_en_voxceleb_CAM++.onnx"
UNKNOWN = "voice-?"


def voice_name(index: int) -> str:
    """0 -> voice-A ... 25 -> voice-Z, 26 -> voice-AA ..."""
    letters = string.ascii_uppercase
    name = ""
    i = index
    while True:
        name = letters[i % 26] + name
        i = i // 26 - 1
        if i < 0:
            break
    return f"voice-{name}"


def kaldi_fbank(samples: np.ndarray, sample_rate: int = 16000, num_bins: int = 80) -> np.ndarray:
    """[T, 80] log-mel filterbank the WeSpeaker models were trained on. `samples` float in -1..1."""
    import kaldi_native_fbank as knf  # lazy

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = sample_rate
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = num_bins
    opts.mel_opts.debug_mel = False
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sample_rate, (samples * 32768.0).astype(np.float32).tolist())
    fbank.input_finished()
    n = fbank.num_frames_ready
    if n == 0:
        return np.zeros((0, num_bins), dtype=np.float32)
    feats = np.stack([fbank.get_frame(i) for i in range(n)]).astype(np.float32)
    return feats - feats.mean(axis=0, keepdims=True)   # per-utterance CMN, as in wespeaker/cli


class EmbeddingDiarizer:
    def __init__(self, spec: RoleSpec | None = None, **options: Any) -> None:
        opts = merge_options(spec, options)
        self.model_path = Path(opts.get("model_path") or resolve_model(DEFAULT_MODEL_FILE))
        self.threshold = float(opts.get("threshold", 0.55))
        self.max_voices = int(opts.get("max_voices", 8))
        self.min_seconds = float(opts.get("min_seconds", 0.6))
        self.momentum = float(opts.get("centroid_momentum", 0.8))
        self.threads = int(opts.get("threads", 1))
        self._session = None
        self._input = "feats"
        self.label = f"WeSpeaker {self.model_path.stem} (onnxruntime CPU)"

    def _ort(self):
        if self._session is None:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.intra_op_num_threads = self.threads
            so.inter_op_num_threads = 1
            so.log_severity_level = 3
            self._session = ort.InferenceSession(str(self.model_path), sess_options=so,
                                                 providers=["CPUExecutionProvider"])
            self._input = self._session.get_inputs()[0].name
        return self._session

    def embed(self, pcm16: bytes, sample_rate: int = 16000) -> np.ndarray:
        """Unit-norm embedding of one utterance (blocking; ~50 ms per 3 s on CPU)."""
        x = pcm16_to_float(pcm16)
        if sample_rate != 16000:
            from .audio import resample_pcm16
            x = pcm16_to_float(resample_pcm16(pcm16, sample_rate, 16000))
        feats = kaldi_fbank(x, 16000)
        if len(feats) < 10:
            return np.zeros(0, dtype=np.float32)
        emb = self._ort().run(None, {self._input: feats[None, :, :]})[0][0].astype(np.float32)
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 0 else emb

    def session(self) -> "DiarSession":
        return DiarSession(self)

    def healthy(self) -> bool:
        try:
            self._ort()
            return True
        except Exception:
            return False


class DiarSession:
    """One meeting's voices. Labels are stable for the session and meaningless outside it."""

    def __init__(self, diar: EmbeddingDiarizer) -> None:
        self.d = diar
        self.centroids: list[np.ndarray] = []
        self.counts: list[int] = []
        self.last_ms = 0.0

    # ------------------------------------------------------------------ DiarSession protocol
    async def label(self, pcm16: bytes, sample_rate: int) -> VoiceLabel:
        if len(pcm16) / 2 / sample_rate < self.d.min_seconds:
            return VoiceLabel(label=UNKNOWN, conf=0.0)
        t0 = time.perf_counter()
        emb = await asyncio.to_thread(self.d.embed, pcm16, sample_rate)
        out = self.assign(emb)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return out

    def reset(self) -> None:
        self.centroids.clear()
        self.counts.clear()

    # ------------------------------------------------------------------ clustering
    def assign(self, emb: np.ndarray) -> VoiceLabel:
        if emb.size == 0:
            return VoiceLabel(label=UNKNOWN, conf=0.0)
        if not self.centroids:
            self._new(emb)
            return VoiceLabel(label=voice_name(0), conf=1.0)
        sims = np.array([float(np.dot(emb, c) / (np.linalg.norm(c) + 1e-9)) for c in self.centroids])
        best = int(np.argmax(sims))
        if sims[best] >= self.d.threshold or len(self.centroids) >= self.d.max_voices:
            self._update(best, emb)
            # confidence: how clearly the best voice beats the runner-up and the threshold
            runner = float(np.partition(sims, -2)[-2]) if len(sims) > 1 else self.d.threshold - 0.3
            margin = max(sims[best] - max(runner, self.d.threshold), 0.0)
            conf = float(np.clip(0.5 + margin / (2 * max(1 - self.d.threshold, 1e-6)), 0.0, 1.0))
            return VoiceLabel(label=voice_name(best), conf=round(conf, 3))
        self._new(emb)
        conf = float(np.clip(0.5 + (self.d.threshold - sims[best]) / (2 * self.d.threshold), 0.0, 1.0))
        return VoiceLabel(label=voice_name(len(self.centroids) - 1), conf=round(conf, 3))

    def _new(self, emb: np.ndarray) -> None:
        self.centroids.append(emb.astype(np.float32).copy())
        self.counts.append(1)

    def _update(self, i: int, emb: np.ndarray) -> None:
        m = self.d.momentum
        c = m * self.centroids[i] + (1 - m) * emb
        self.centroids[i] = (c / (np.linalg.norm(c) + 1e-9)).astype(np.float32)
        self.counts[i] += 1

    @property
    def voices(self) -> list[str]:
        return [voice_name(i) for i in range(len(self.centroids))]


__all__ = ["EmbeddingDiarizer", "DiarSession", "kaldi_fbank", "voice_name", "UNKNOWN", "DEFAULT_MODEL_FILE"]
