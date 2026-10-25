"""Test doubles for the sensors pipeline: no model files, no hardware, deterministic.

FakeASR          text per segment from a script (list, in order), a callable(seg_pcm, start)->str, or a
                 constant; records every call.
FakeDiarizer     labels from a callable(pcm16, sample_rate)->label or round-robin over `labels`.
FakeFaceTracker  tracks from a schedule [(t_from, [FaceTrack...]), ...]: the latest entry with t_from <= t.
FakeASD          speaking=1.0 for `talker` (track id) while the audio has energy, else 0 for everyone.
FakeVideoSource  synthetic frames (uint8 BGR of the given size) at `fps` from t0 for `seconds`, paced by
                 `pace()` (audio media time) when given, otherwise as fast as possible.
FakeRegistry     Registry-shaped: the components you pass; the rest are disabled.
EnergyVAD (vad.py) is the model-free VAD; tests use it directly.
"""
from __future__ import annotations

import asyncio
import itertools
from typing import Any, AsyncIterator, Callable

import numpy as np

from ..contracts import FaceTrack, RoleDisabled, RoleSpec, Transcript, VoiceLabel
from .audio import rms
from .video import VideoFrame


class FakeASR:
    def __init__(self, script: list[str] | Callable[[bytes, float], str] | str | None = None, lang: str = "en",
                 latency_ms: float = 12.0) -> None:
        self.script = script
        self.lang = lang
        self.latency_ms = latency_ms
        self.calls: list[tuple[int, int]] = []     # (pcm bytes, sample_rate)
        self._i = 0

    async def transcribe(self, pcm16: bytes, sample_rate: int, lang: str | None = None) -> Transcript:
        self.calls.append((len(pcm16), sample_rate))
        if callable(self.script):
            text = self.script(pcm16, 0.0)
        elif isinstance(self.script, list):
            text = self.script[self._i] if self._i < len(self.script) else ""
            self._i += 1
        elif isinstance(self.script, str):
            text = self.script
        else:
            text = f"segment {len(self.calls)}"
        await asyncio.sleep(0)
        return Transcript(text=text, lang=lang or self.lang, conf=0.9, latency_ms=self.latency_ms)

    async def healthy(self) -> bool:
        return True


class FakeDiarizer:
    def __init__(self, labels: list[str] | None = None, by: Callable[[bytes, int], str] | None = None) -> None:
        self.labels = labels or ["voice-A", "voice-B", "voice-C"]
        self.by = by
        self.sessions: list["FakeDiarSession"] = []

    def session(self) -> "FakeDiarSession":
        s = FakeDiarSession(self)
        self.sessions.append(s)
        return s


class FakeDiarSession:
    def __init__(self, d: FakeDiarizer) -> None:
        self.d = d
        self._cycle = itertools.cycle(d.labels)
        self.calls = 0
        self.resets = 0
        self.last_ms = 3.0

    async def label(self, pcm16: bytes, sample_rate: int) -> VoiceLabel:
        self.calls += 1
        await asyncio.sleep(0)
        lab = self.d.by(pcm16, sample_rate) if self.d.by else next(self._cycle)
        return VoiceLabel(label=lab, conf=0.8)

    def reset(self) -> None:
        self.resets += 1
        self._cycle = itertools.cycle(self.d.labels)


class FakeFaceTracker:
    def __init__(self, schedule: list[tuple[float, list[FaceTrack]]] | None = None) -> None:
        self.schedule = sorted(schedule or [], key=lambda x: x[0])
        self.seats_configured: dict | None = None
        self.sessions: list["FakeFaceSession"] = []

    def configure_seats(self, zones: dict) -> None:
        self.seats_configured = dict(zones)

    def session(self) -> "FakeFaceSession":
        s = FakeFaceSession(self)
        self.sessions.append(s)
        return s


class FakeFaceSession:
    def __init__(self, f: FakeFaceTracker) -> None:
        self.f = f
        self.calls: list[float] = []
        self.resets = 0

    def update(self, frame: Any, t: float) -> list[FaceTrack]:
        self.calls.append(t)
        tracks: list[FaceTrack] = []
        for t_from, trs in self.f.schedule:
            if t_from <= t:
                tracks = trs
        return [FaceTrack(track_id=tr.track_id, bbox=tr.bbox, seat=tr.seat, speaking=0.0) for tr in tracks]

    def reset(self) -> None:
        self.resets += 1


class FakeASD:
    def __init__(self, talker: str | Callable[[float], str | None] | None = None, energy: float = 0.01) -> None:
        self.talker = talker
        self.energy = energy
        self.sessions: list["FakeASDSession"] = []

    def session(self) -> "FakeASDSession":
        s = FakeASDSession(self)
        self.sessions.append(s)
        return s


class FakeASDSession:
    def __init__(self, a: FakeASD) -> None:
        self.a = a
        self.audio_seen: list[int] = []
        self.resets = 0

    def update(self, frame: Any, t: float, faces: list[FaceTrack], audio: bytes | None,
               sample_rate: int = 16000) -> list[FaceTrack]:
        self.audio_seen.append(len(audio) if audio else 0)
        voice = audio is not None and rms(audio) > self.a.energy
        who = self.a.talker(t) if callable(self.a.talker) else self.a.talker
        return [FaceTrack(track_id=f.track_id, bbox=f.bbox, seat=f.seat,
                          speaking=1.0 if (voice and f.track_id == who) else 0.0) for f in faces]

    def reset(self) -> None:
        self.resets += 1


class FakeVideoSource:
    def __init__(self, *, clock: Callable[[], float], seconds: float, fps: float = 5.0, size: tuple[int, int] = (64, 36),
                 pace: Callable[[], float] | None = None, t0: float | None = None,
                 frame_fn: Callable[[int, float], np.ndarray] | None = None) -> None:
        self.clock = clock
        self.seconds = seconds
        self.fps = fps
        self.size = size
        self.pace = pace
        self._t0 = t0
        self.frame_fn = frame_fn
        self.closed = False
        self.yielded = 0

    async def __aiter__(self) -> AsyncIterator[VideoFrame]:
        t0 = self.clock() if self._t0 is None else self._t0
        n = int(self.seconds * self.fps)
        for i in range(n):
            if self.closed:
                return
            media = i / self.fps
            if self.pace is not None:
                while self.pace() < media and not self.closed:
                    await asyncio.sleep(0.002)
            else:
                await asyncio.sleep(0)
            img = self.frame_fn(i, media) if self.frame_fn else np.full((self.size[1], self.size[0], 3), 80, np.uint8)
            self.yielded += 1
            yield VideoFrame(img, t0 + media, i + 1)

    async def aclose(self) -> None:
        self.closed = True


class FakeRegistry:
    """Only the sensor roles; everything else is disabled. `labels` overrides label(role)."""

    profile = "fake"
    local_only = True

    def __init__(self, *, vad: Any = None, asr: Any = None, diarizer: Any = None, faces: Any = None,
                 asd: Any = None, labels: dict[str, str] | None = None) -> None:
        self._roles = {"vad": vad, "asr": asr, "diarization": diarizer, "faces": faces, "asd": asd}
        self._labels = labels or {}

    def spec(self, role: str) -> RoleSpec:
        return RoleSpec(role=role, provider="fake" if self._roles.get(role) is not None else "none",
                        label=self.label(role))

    def enabled(self, role: str) -> bool:
        return self._roles.get(role) is not None

    def label(self, role: str) -> str:
        return self._labels.get(role, f"fake {role}")

    def _need(self, role: str) -> Any:
        obj = self._roles.get(role)
        if obj is None:
            raise RoleDisabled(role)
        return obj

    def vad(self) -> Any:
        return self._need("vad")

    def asr(self) -> Any:
        return self._need("asr")

    def diarizer(self) -> Any:
        return self._need("diarization")

    def faces(self) -> Any:
        return self._need("faces")

    def asd(self) -> Any:
        return self._need("asd")

    def decider(self, role: str = "decider") -> Any:
        raise RoleDisabled(role)

    def strands_model(self, role: str = "llm") -> Any:
        raise RoleDisabled(role)

    def tts(self) -> Any:
        raise RoleDisabled("tts")

    def embedder(self) -> Any:
        raise RoleDisabled("embed")

    def pii(self) -> Any:
        raise RoleDisabled("pii")

    async def check(self, roles: list[str] | None = None) -> list:
        return []

    async def aclose(self) -> None:
        return None


__all__ = ["FakeASR", "FakeDiarizer", "FakeDiarSession", "FakeFaceTracker", "FakeFaceSession", "FakeASD",
           "FakeASDSession", "FakeVideoSource", "FakeRegistry"]
