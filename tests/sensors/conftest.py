"""Shared fixtures for the sensors tests. Everything runs without model servers or hardware; tests that need
a model FILE skip when it is not cached locally (faster-whisper in ~/.cache/huggingface, the WeSpeaker ONNX
in the unspoken models dir), and nothing here downloads anything."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MEETING_WAV = FIXTURES / "meeting3.wav"
MEETING_JSON = FIXTURES / "meeting3.json"
FACE_JPG = FIXTURES / "face_256.jpg"
SR = 16000

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # never download during tests


def normalize(text: str) -> list[str]:
    text = text.lower().replace("i'll", "i will").replace("let's", "let us")
    return re.sub(r"[^a-z0-9 ]", " ", text).split()


def wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(hyp) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / max(1, len(ref))


def purity(truth: list[str], labels: list[str]) -> tuple[float, int]:
    clusters: dict[str, list[str]] = {}
    for t, l in zip(truth, labels):
        clusters.setdefault(l, []).append(t)
    pure = sum(max(c.count(x) for x in set(c)) for c in clusters.values()) / len(truth)
    return pure, len(clusters)


@pytest.fixture(scope="session")
def truth() -> dict[str, Any]:
    return json.loads(MEETING_JSON.read_text())


@pytest.fixture(scope="session")
def meeting_pcm() -> bytes:
    from unspoken.sensors.audio import read_wav_pcm16
    return read_wav_pcm16(MEETING_WAV)


def line_pcm(pcm: bytes, line: dict[str, Any]) -> bytes:
    return pcm[int(line["start"] * SR) * 2: int(line["end"] * SR) * 2]


def whisper_cached(model: str = "Systran/faster-whisper-base.en") -> bool:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    if "HF_HUB_CACHE" in os.environ:
        hub = Path(os.environ["HF_HUB_CACHE"])
    return (hub / ("models--" + model.replace("/", "--"))).exists()


def speaker_model_present() -> bool:
    from unspoken.sensors.assets import resolve_model
    from unspoken.sensors.diar import DEFAULT_MODEL_FILE
    return resolve_model(DEFAULT_MODEL_FILE).exists()


needs_whisper = pytest.mark.skipif(not whisper_cached(), reason="faster-whisper-base.en not in the HF cache")
needs_speaker_model = pytest.mark.skipif(not speaker_model_present(),
                                         reason="WeSpeaker ONNX not fetched (python -m unspoken.sensors.fetch_models)")
live = pytest.mark.skipif(os.environ.get("UNSPOKEN_LIVE") != "1", reason="set UNSPOKEN_LIVE=1 for live checks")


class FakeClock:
    """Manual clock for tests that must not sleep for real."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt
