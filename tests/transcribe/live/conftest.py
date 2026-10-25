"""Fixtures for the live transcriber tests. No model downloads, no servers: the vendored Silero VAD ONNX, the
sensors fixture WAV (tests/sensors/fixtures/meeting3.wav, 12 scripted lines by 3 piper voices), fake ASR /
diarizer / faces / ASD, and a tiny synthetic mp4 written on the fly."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from unspoken.contracts import FaceTrack, VoiceLabel
from unspoken.sensors.audio import read_wav_pcm16, rms

os.environ.setdefault("HF_HUB_OFFLINE", "1")

FIXTURES = Path(__file__).resolve().parents[2] / "sensors" / "fixtures"
MEETING_WAV = FIXTURES / "meeting3.wav"
MEETING_JSON = FIXTURES / "meeting3.json"
SR = 16000
SEATS = {"left": (0.0, 0.0, 0.5, 1.0), "right": (0.5, 0.0, 0.5, 1.0)}
LEFT_FACE = FaceTrack("f1", (0.10, 0.30, 0.20, 0.40))
RIGHT_FACE = FaceTrack("f2", (0.65, 0.30, 0.20, 0.40))
VOICE_OF = {"Priya": "voice-A", "Sam": "voice-B", "Lee": "voice-C"}
FACE_OF = {"Priya": "f1", "Sam": "f2", "Lee": None}      # Lee is off camera


@pytest.fixture(scope="session")
def truth() -> dict[str, Any]:
    return json.loads(MEETING_JSON.read_text())


@pytest.fixture(scope="session")
def meeting_pcm() -> bytes:
    return read_wav_pcm16(MEETING_WAV)


def speaker_at(truth: dict[str, Any], t: float) -> str | None:
    for line in truth["lines"]:
        if line["start"] - 0.3 <= t <= line["end"] + 0.3:
            return line["speaker"]
    return None


def line_at(truth: dict[str, Any], t: float) -> dict[str, Any] | None:
    return next((l for l in truth["lines"] if l["start"] - 0.3 <= t <= l["end"] + 0.3), None)


def locate(full_pcm: bytes, pcm16: bytes) -> float | None:
    """Media time of the middle of `pcm16` inside the fixture (None when not found)."""
    probe = pcm16[3200:3200 + 6400] if len(pcm16) > 9600 else pcm16[: min(len(pcm16), 6400)]
    off = full_pcm.find(probe)
    if off < 0:
        return None
    return off / 2 / SR + len(pcm16) / 4 / SR


class TruthASR:
    """Fake ASR whose text is the fixture line that really plays in the audio (so partials, which call the
    ASR too, do not eat the script). Partials get the first few words of the line."""

    def __init__(self, full_pcm: bytes, truth: dict[str, Any]) -> None:
        self.full, self.truth = full_pcm, truth
        self.calls: list[int] = []
        self.label = "TruthASR"

    async def transcribe(self, pcm16: bytes, sample_rate: int, lang: str | None = None):
        from unspoken.contracts import Transcript, Word

        self.calls.append(len(pcm16))
        t = locate(self.full, pcm16)
        line = line_at(self.truth, t) if t is not None else None
        if line is None:
            return Transcript(text="", conf=0.0)
        text = line["text"]
        dur = len(pcm16) / 2 / SR
        if dur < line["end"] - line["start"] - 0.4:          # a partial: only what has been said so far
            words = text.split()
            text = " ".join(words[: max(1, int(len(words) * dur / (line["end"] - line["start"])))])
        words = [Word(w, i * dur / max(1, len(text.split())), (i + 1) * dur / max(1, len(text.split())))
                 for i, w in enumerate(text.split())]
        return Transcript(text=text, lang="en", conf=0.9, words=words, latency_ms=5.0)

    async def healthy(self) -> bool:
        return True


class TruthDiarizer:
    """Labels a segment by WHO really speaks there: finds the PCM in the fixture to get its time. Exposes
    nothing the engine's peek needs (`d`/`centroids`), so partials carry voice None, like a server diarizer."""

    def __init__(self, full_pcm: bytes, truth: dict[str, Any]) -> None:
        self.full, self.truth = full_pcm, truth
        self.sessions: list["TruthDiarSession"] = []

    def session(self) -> "TruthDiarSession":
        s = TruthDiarSession(self)
        self.sessions.append(s)
        return s


class TruthDiarSession:
    def __init__(self, d: TruthDiarizer) -> None:
        self.d = d
        self.calls = 0
        self.resets = 0
        self.last_ms = 2.0

    async def label(self, pcm16: bytes, sample_rate: int) -> VoiceLabel:
        self.calls += 1
        probe = pcm16[3200:3200 + 6400] if len(pcm16) > 9600 else pcm16[: min(len(pcm16), 6400)]
        off = self.d.full.find(probe)
        if off < 0:
            return VoiceLabel("voice-?", 0.0)
        t_mid = off / 2 / SR + len(pcm16) / 4 / SR
        who = speaker_at(self.d.truth, t_mid)
        return VoiceLabel(VOICE_OF.get(who or "", "voice-?"), 0.9 if who else 0.0)

    def reset(self) -> None:
        self.resets += 1


class WeakASD:
    """Like sensors.fakes.FakeASD but the talker only scores `level` (below the per-segment thresholds), so
    the speaker has to come from the accumulated voice->seat binding."""

    def __init__(self, talker: Callable[[float], str | None], level: float = 0.2, energy: float = 0.01) -> None:
        self.talker, self.level, self.energy = talker, level, energy
        self.resets = 0

    def session(self) -> "WeakASD":
        return self

    def update(self, frame: Any, t: float, faces: list[FaceTrack], audio: bytes | None, sample_rate: int = SR) -> list[FaceTrack]:
        voice = audio is not None and rms(audio) > self.energy
        who = self.talker(t)
        return [FaceTrack(f.track_id, f.bbox, f.seat, self.level if (voice and f.track_id == who) else 0.0) for f in faces]

    def reset(self) -> None:
        self.resets += 1


def talker_for(truth: dict[str, Any]) -> Callable[[float], str | None]:
    return lambda t: FACE_OF.get(speaker_at(truth, t) or "", None)


def write_tiny_mp4(path: Path, seconds: float = 2.0, fps: float = 10.0, size: tuple[int, int] = (96, 48)) -> Path:
    import cv2

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    n = int(seconds * fps)
    for i in range(n):
        img = np.full((size[1], size[0], 3), 40, np.uint8)
        cv2.rectangle(img, (10 + i, 10), (30 + i, 30), (0, 200, 255), -1)
        w.write(img)
    w.release()
    return path
