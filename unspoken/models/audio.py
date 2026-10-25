"""Tiny PCM16 <-> WAV helpers (stdlib only). Everything in this repo is mono 16-bit little-endian PCM."""
from __future__ import annotations

import io
import struct
import wave


def pcm16_to_wav(pcm16: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()


def wav_to_pcm16(data: bytes) -> tuple[bytes, int]:
    """(pcm16 mono, sample_rate). Multi-channel input keeps the first channel."""
    with wave.open(io.BytesIO(data), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit WAV, got {8 * width}-bit")
    if channels > 1:
        frames = b"".join(frames[i:i + 2] for i in range(0, len(frames), 2 * channels))
    return frames, rate


def silence(seconds: float, sample_rate: int = 16000) -> bytes:
    return bytes(2 * int(seconds * sample_rate))


def mean_abs(pcm16: bytes) -> float:
    """Mean absolute sample value (0..32767); 0 for empty or silent audio."""
    n = len(pcm16) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm16[:2 * n])
    return sum(abs(s) for s in samples) / n


def wav_duration_s(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes() / float(w.getframerate())
