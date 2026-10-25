"""Voice activity detection -> SpeechSegment (unspoken.contracts.VAD / VADStream).

SileroVAD  Silero VAD v5 ONNX (MIT, vendored in models/) on onnxruntime CPU: 512-sample windows at 16 kHz
           with the 64-sample context the v5 graph expects. ~0.3 ms per 32 ms window on one core.
EnergyVAD  RMS gate with an adaptive noise floor; the fallback when onnxruntime is unavailable.

Both share the same segmenter: speech starts after `min_speech_ms` of frames above `threshold`, ends after
`min_silence_ms` below `threshold - hysteresis`, gets `pad_ms` of audio either side, and is cut at
`max_segment_s` so a long monologue still reaches the ASR in pieces. Segment times are seconds from the
first sample fed to the stream; the pipeline shifts them onto the meeting clock.

Options (RoleSpec.options / constructor kwargs): threshold=0.5, hysteresis=0.15, min_speech_ms=250,
min_silence_ms=500, pad_ms=200, max_segment_s=20, model_path (Silero), threads=1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import RoleSpec, SpeechSegment
from .config import merge_options
from .audio import SAMPLE_RATE, pcm16_to_float

MODELS_DIR = Path(__file__).resolve().parent / "models"
SILERO_ONNX = MODELS_DIR / "silero_vad.onnx"
WINDOW = 512          # samples per Silero step at 16 kHz (32 ms)
CONTEXT = 64          # v5 prepends the last 64 samples of the previous window


@dataclass
class SegmenterSettings:
    threshold: float = 0.5
    hysteresis: float = 0.15
    min_speech_ms: int = 250
    min_silence_ms: int = 500
    pad_ms: int = 200
    max_segment_s: float = 20.0

    @classmethod
    def from_options(cls, options: dict[str, Any] | None) -> "SegmenterSettings":
        o = options or {}
        s = cls()
        for k in ("threshold", "hysteresis", "min_speech_ms", "min_silence_ms", "pad_ms", "max_segment_s"):
            if k in o:
                setattr(s, k, type(getattr(s, k))(o[k]))
        return s


class _Segmenter:
    """Turns a per-window speech probability stream into closed SpeechSegments. Keeps only the PCM it may
    still need (current segment + pad); everything else is dropped as soon as the window is classified."""

    def __init__(self, settings: SegmenterSettings, sample_rate: int, window: int) -> None:
        self.s = settings
        self.sr = sample_rate
        self.window = window
        self.window_s = window / sample_rate
        self.min_speech_w = max(1, round(settings.min_speech_ms / 1000 / self.window_s))
        self.min_silence_w = max(1, round(settings.min_silence_ms / 1000 / self.window_s))
        self.pad_w = max(0, round(settings.pad_ms / 1000 / self.window_s))
        self.max_w = max(self.min_speech_w + 1, round(settings.max_segment_s / self.window_s))
        self.n_windows = 0                       # windows seen so far (stream clock)
        self.tail: list[bytes] = []              # last pad_w windows before speech (for the leading pad)
        self.active = False
        self.speech_run = 0
        self.silence_run = 0
        self.buf: list[bytes] = []               # windows of the open segment (incl. leading pad)
        self.start_w = 0
        self.candidate: list[bytes] = []         # windows of a not-yet-confirmed speech onset
        self.candidate_start_w = 0

    def push(self, window_pcm: bytes, p: float) -> list[SpeechSegment]:
        s = self.s
        out: list[SpeechSegment] = []
        w = self.n_windows
        self.n_windows += 1
        if not self.active:
            if p >= s.threshold:
                if not self.candidate:
                    self.candidate_start_w = w
                self.candidate.append(window_pcm)
                self.speech_run += 1
                if self.speech_run >= self.min_speech_w:
                    # confirmed onset: segment starts pad_w windows before the first candidate window
                    lead = self.tail[-self.pad_w:] if self.pad_w else []
                    self.start_w = self.candidate_start_w - len(lead)
                    self.buf = lead + self.candidate
                    self.candidate = []
                    self.active = True
                    self.speech_run = 0
                    self.silence_run = 0
            else:
                # a false start: the candidate windows become plain tail
                for c in self.candidate:
                    self._push_tail(c)
                self.candidate = []
                self.speech_run = 0
                self._push_tail(window_pcm)
            return out
        # active segment
        self.buf.append(window_pcm)
        if p < s.threshold - s.hysteresis:
            self.silence_run += 1
        elif p >= s.threshold:
            self.silence_run = 0
        if self.silence_run >= self.min_silence_w:
            # close: keep pad_w windows of the trailing silence, drop the rest
            drop = self.silence_run - self.pad_w
            if drop > 0:
                dropped = self.buf[len(self.buf) - drop:]
                del self.buf[len(self.buf) - drop:]
                for d in dropped:
                    self._push_tail(d)
            if len(self.buf) - self.pad_w >= self.min_speech_w:
                out.append(self._close())
            else:                      # a sliver left after a max-length split: not worth an ASR call
                self._close()
        elif len(self.buf) >= self.max_w:
            out.append(self._close(keep_going=True))
        return out

    def _push_tail(self, window_pcm: bytes) -> None:
        if self.pad_w:
            self.tail.append(window_pcm)
            if len(self.tail) > self.pad_w:
                del self.tail[0]

    def _close(self, keep_going: bool = False) -> SpeechSegment:
        pcm = b"".join(self.buf)
        start = self.start_w * self.window_s
        seg = SpeechSegment(pcm16=pcm, sample_rate=self.sr, start=start, end=start + len(self.buf) * self.window_s)
        if keep_going:
            # continue the same utterance in a new segment with a tiny overlap for the ASR's sake
            overlap = self.buf[-self.pad_w:] if self.pad_w else []
            self.start_w = self.n_windows - len(overlap)
            self.buf = list(overlap)
            self.silence_run = 0
        else:
            self.active = False
            self.buf = []
            self.silence_run = 0
            self.speech_run = 0
        return seg

    def flush(self) -> list[SpeechSegment]:
        out: list[SpeechSegment] = []
        if self.active and len(self.buf) >= self.min_speech_w:
            out.append(self._close())
        self.active = False
        self.buf = []
        self.candidate = []
        self.tail = []
        return out


class _Stream:
    """VADStream over a probability function; buffers partial windows between feed() calls."""

    def __init__(self, prob, settings: SegmenterSettings, sample_rate: int, window: int) -> None:
        self._prob = prob
        self._seg = _Segmenter(settings, sample_rate, window)
        self._window_bytes = window * 2
        self._pending = b""
        self.last_prob = 0.0

    def feed(self, frame: bytes) -> list[SpeechSegment]:
        out: list[SpeechSegment] = []
        data = self._pending + frame
        pos = 0
        while pos + self._window_bytes <= len(data):
            win = data[pos: pos + self._window_bytes]
            pos += self._window_bytes
            self.last_prob = float(self._prob(win))
            out += self._seg.push(win, self.last_prob)
        self._pending = data[pos:]
        return out

    def flush(self) -> list[SpeechSegment]:
        out: list[SpeechSegment] = []
        if self._pending:
            win = self._pending + bytes(self._window_bytes - len(self._pending))
            self._pending = b""
            out += self._seg.push(win, float(self._prob(win)))
        return out + self._seg.flush()

    @property
    def active(self) -> bool:
        return self._seg.active


# ----------------------------------------------------------------------------- Silero
class SileroVAD:
    """Factory signature for the registry: SileroVAD(spec). Also SileroVAD(options={...}) / kwargs in code."""

    def __init__(self, spec: RoleSpec | None = None, **options: Any) -> None:
        opts = merge_options(spec, options)
        self.settings = SegmenterSettings.from_options(opts)
        self.model_path = Path(opts.get("model_path") or SILERO_ONNX)
        self.threads = int(opts.get("threads", 1))
        self._session = None

    def _ort(self):
        if self._session is None:
            import onnxruntime as ort  # lazy: the registry must not import it until the role is used

            so = ort.SessionOptions()
            so.intra_op_num_threads = self.threads
            so.inter_op_num_threads = 1
            so.log_severity_level = 3
            self._session = ort.InferenceSession(str(self.model_path), sess_options=so,
                                                 providers=["CPUExecutionProvider"])
        return self._session

    def stream(self, sample_rate: int = SAMPLE_RATE) -> "SileroStream":
        if sample_rate != SAMPLE_RATE:
            raise ValueError("SileroVAD runs at 16 kHz in this repo")
        return SileroStream(self._ort(), self.settings, sample_rate)

    def healthy(self) -> bool:
        try:
            self._ort()
            return True
        except Exception:
            return False


class SileroStream(_Stream):
    def __init__(self, session, settings: SegmenterSettings, sample_rate: int) -> None:
        self._sess = session
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)
        self._sr = np.array(sample_rate, dtype=np.int64)
        super().__init__(self._probability, settings, sample_rate, WINDOW)

    def _probability(self, window_pcm: bytes) -> float:
        x = pcm16_to_float(window_pcm)[None, :]
        inp = np.concatenate([self._context, x], axis=1).astype(np.float32)
        out, state = self._sess.run(None, {"input": inp, "state": self._state, "sr": self._sr})
        self._state = state
        self._context = inp[:, -CONTEXT:]
        return float(out[0, 0])


# ----------------------------------------------------------------------------- energy fallback
class EnergyVAD:
    """RMS above an adaptive noise floor. options: floor_db=-50 (initial floor), snr_db=12, adapt=0.02."""

    def __init__(self, spec: RoleSpec | None = None, **options: Any) -> None:
        opts = merge_options(spec, options)
        opts.setdefault("threshold", 0.5)
        opts.setdefault("hysteresis", 0.1)
        self.settings = SegmenterSettings.from_options(opts)
        self.floor_db = float(opts.get("floor_db", -50.0))
        self.snr_db = float(opts.get("snr_db", 12.0))
        self.adapt = float(opts.get("adapt", 0.02))

    def stream(self, sample_rate: int = SAMPLE_RATE) -> _Stream:
        floor = [self.floor_db]

        def prob(window_pcm: bytes) -> float:
            x = pcm16_to_float(window_pcm)
            level = 20 * np.log10(float(np.sqrt(np.mean(x * x)) + 1e-9))
            if level < floor[0]:
                floor[0] = level                                  # the floor drops fast
            else:
                floor[0] += self.adapt * (level - floor[0]) * 0.1  # and rises slowly
            margin = level - floor[0]
            return float(np.clip(margin / self.snr_db, 0.0, 1.0))

        return _Stream(prob, self.settings, sample_rate, WINDOW)

    def healthy(self) -> bool:
        return True


__all__ = ["SileroVAD", "SileroStream", "EnergyVAD", "SegmenterSettings", "SILERO_ONNX", "WINDOW"]
