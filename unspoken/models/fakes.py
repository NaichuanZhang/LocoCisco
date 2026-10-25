"""Deterministic in-process stand-ins for every role (provider "fake"), for unit tests and the simulator.

They need no model, no GPU and no network, give the same answer for the same input on every run, and
implement the protocols in unspoken/contracts.py. Each takes its knobs from RoleSpec.options so a test
or a profile can script them from the TOML:

    [profiles.fake.asr]
    provider = "fake"
    options = { texts = ["hello", "send the deck by friday"] }   # cycles through these
"""
from __future__ import annotations

import hashlib
import math
import re
import struct
from typing import Any

from ..contracts import FaceTrack, PIISpan, RoleSpec, SpeechSegment, Transcript, VoiceLabel
from .audio import mean_abs, pcm16_to_wav


def _opt(spec: RoleSpec | None, key: str, default: Any) -> Any:
    return default if spec is None else spec.options.get(key, default)


# --------------------------------------------------------------------------- speech
class FakeASR:
    """Returns the next scripted text (options.texts), or "" for silence-like input and the fixed
    options.text otherwise. Remembers what it was asked (calls) for assertions."""

    def __init__(self, spec: RoleSpec | None = None) -> None:
        self.texts: list[str] = list(_opt(spec, "texts", []))
        self.text: str = str(_opt(spec, "text", ""))
        self.calls: list[tuple[int, int, str | None]] = []
        self._i = 0

    async def transcribe(self, pcm16: bytes, sample_rate: int, lang: str | None = None) -> Transcript:
        self.calls.append((len(pcm16), sample_rate, lang))
        if self.texts:
            text = self.texts[self._i % len(self.texts)]
            self._i += 1
        else:
            text = self.text if mean_abs(pcm16) > 0 else ""
        return Transcript(text=text, lang=lang or "en", conf=1.0 if text else 0.0, latency_ms=0.0)

    async def healthy(self) -> bool:
        return True


class FakeTTS:
    """A valid 16 kHz mono WAV of a quiet tone, 60 ms per character (so longer text -> longer audio)."""

    def __init__(self, spec: RoleSpec | None = None) -> None:
        self.sample_rate = int(_opt(spec, "sample_rate", 16000))
        self.calls: list[tuple[str, str | None]] = []

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        self.calls.append((text, voice))
        n = max(1, int(0.06 * len(text) * self.sample_rate))
        tone = b"".join(struct.pack("<h", int(2000 * math.sin(2 * math.pi * 220 * i / self.sample_rate)))
                        for i in range(n))
        return pcm16_to_wav(tone, self.sample_rate)

    async def healthy(self) -> bool:
        return True


class FakeVADStream:
    """Energy gate: a frame is speech when its mean |sample| is above `threshold`; a segment closes after
    `hang_ms` of silence. Deterministic, so the simulator can replay the same audio and get the same cuts."""

    def __init__(self, sample_rate: int, threshold: float, hang_ms: float, frame_ms: float) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.hang = hang_ms / 1000.0
        self.frame_bytes = max(2, 2 * int(sample_rate * frame_ms / 1000))
        self._t = 0.0
        self._buf: bytearray = bytearray()
        self._start: float | None = None
        self._quiet_since: float | None = None
        self._pending = bytearray()

    def feed(self, frame: bytes) -> list[SpeechSegment]:
        self._pending.extend(frame)
        closed: list[SpeechSegment] = []
        while len(self._pending) >= self.frame_bytes:
            chunk = bytes(self._pending[:self.frame_bytes])
            del self._pending[:self.frame_bytes]
            closed.extend(self._step(chunk))
        return closed

    def _step(self, chunk: bytes) -> list[SpeechSegment]:
        dt = len(chunk) / 2 / self.sample_rate
        t0, t1 = self._t, self._t + dt
        self._t = t1
        loud = mean_abs(chunk) > self.threshold
        out: list[SpeechSegment] = []
        if loud:
            if self._start is None:
                self._start = t0
                self._buf = bytearray()
            self._buf.extend(chunk)
            self._quiet_since = None
        elif self._start is not None:
            self._buf.extend(chunk)
            self._quiet_since = self._quiet_since if self._quiet_since is not None else t0
            if t1 - self._quiet_since >= self.hang:
                out.append(self._close(self._quiet_since))
        return out

    def _close(self, end: float) -> SpeechSegment:
        seg = SpeechSegment(pcm16=bytes(self._buf), sample_rate=self.sample_rate, start=self._start or 0.0, end=end)
        self._start, self._quiet_since, self._buf = None, None, bytearray()
        return seg

    def flush(self) -> list[SpeechSegment]:
        out: list[SpeechSegment] = []
        if self._pending:
            out.extend(self._step(bytes(self._pending)))
            self._pending = bytearray()
        if self._start is not None:
            out.append(self._close(self._t))
        return out


class FakeVAD:
    def __init__(self, spec: RoleSpec | None = None) -> None:
        self.threshold = float(_opt(spec, "threshold", 300.0))
        self.hang_ms = float(_opt(spec, "hang_ms", 300.0))
        self.frame_ms = float(_opt(spec, "frame_ms", 30.0))

    def stream(self, sample_rate: int = 16000) -> FakeVADStream:
        return FakeVADStream(sample_rate, self.threshold, self.hang_ms, self.frame_ms)


class FakeDiarSession:
    """Labels voices by a coarse loudness signature: audio of about the same level is "the same voice".
    Labels are handed out in order of first appearance (voice-A, voice-B, ...) and forgotten on reset()."""

    def __init__(self, bucket: float) -> None:
        self.bucket = bucket
        self._labels: dict[int, str] = {}

    async def label(self, pcm16: bytes, sample_rate: int) -> VoiceLabel:
        key = int(mean_abs(pcm16) // self.bucket)
        if key not in self._labels:
            self._labels[key] = f"voice-{chr(ord('A') + len(self._labels) % 26)}"
        return VoiceLabel(label=self._labels[key], conf=0.9)

    def reset(self) -> None:
        self._labels.clear()


class FakeDiarizer:
    def __init__(self, spec: RoleSpec | None = None) -> None:
        self.bucket = float(_opt(spec, "bucket", 500.0))

    def session(self) -> FakeDiarSession:
        return FakeDiarSession(self.bucket)


# --------------------------------------------------------------------------- vision
class FakeFaceSession:
    def __init__(self, tracks: list[dict[str, Any]]) -> None:
        self.tracks = tracks

    def update(self, frame: Any, t: float) -> list[FaceTrack]:
        if frame is None:
            return []
        return [FaceTrack(track_id=str(tr.get("track_id", f"face-{i + 1}")),
                          bbox=tuple(float(v) for v in tr.get("bbox", (0.4, 0.3, 0.2, 0.3))),  # type: ignore[arg-type]
                          seat=tr.get("seat")) for i, tr in enumerate(self.tracks)]


class FakeFaces:
    """Reports the faces listed in options.tracks (default: one face in the middle) whenever it gets a frame."""

    def __init__(self, spec: RoleSpec | None = None) -> None:
        self.tracks: list[dict[str, Any]] = list(_opt(spec, "tracks", [{"track_id": "face-1", "bbox": [0.4, 0.3, 0.2, 0.3]}]))

    def session(self) -> FakeFaceSession:
        return FakeFaceSession(self.tracks)


class FakeASDSession:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def update(self, frame: Any, t: float, faces: list[FaceTrack], audio: bytes | None,
               sample_rate: int = 16000) -> list[FaceTrack]:
        talking = audio is not None and mean_abs(audio) > self.threshold
        for i, f in enumerate(faces):
            f.speaking = 1.0 if (talking and i == 0) else 0.0      # the first face talks when there is sound
        return faces


class FakeASD:
    def __init__(self, spec: RoleSpec | None = None) -> None:
        self.threshold = float(_opt(spec, "threshold", 300.0))

    def session(self) -> FakeASDSession:
        return FakeASDSession(self.threshold)


# --------------------------------------------------------------------------- text
_TOKEN = re.compile(r"[\w']+|[\u4e00-\u9fff]")


class FakeEmbedder:
    """Hashed bag of words: each token adds +-1 at a sha1-derived position; the vector is L2-normalised.
    The same text always gives the same vector and texts sharing words are close."""

    def __init__(self, spec: RoleSpec | None = None) -> None:
        self.dim = int(_opt(spec, "dim", 64))

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in _TOKEN.findall(text.lower()):
            h = hashlib.sha1(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            v[idx] += 1.0 if h[4] & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    async def healthy(self) -> bool:
        return True


class FakePII:
    """Flags every occurrence of the words in options.words ({"word": "label"}); nothing by default."""

    def __init__(self, spec: RoleSpec | None = None) -> None:
        self.words: dict[str, str] = dict(_opt(spec, "words", {}))

    async def find(self, text: str) -> list[PIISpan]:
        spans: list[PIISpan] = []
        for word, label in self.words.items():
            for m in re.finditer(re.escape(word), text):
                spans.append(PIISpan(start=m.start(), end=m.end(), label=label, score=1.0))
        return sorted(spans, key=lambda s: s.start)
