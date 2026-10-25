"""config/sensors.toml loading, seat zones, audio helpers and sources, VAD segmentation on the fixture."""
from __future__ import annotations

import asyncio
import wave

import numpy as np
import pytest

from unspoken.sensors import audio as A
from unspoken.sensors.config import DEFAULT_PATH, load_sensors_config, merge_defaults, seat_for, seat_zones
from unspoken.sensors.vad import WINDOW, EnergyVAD, SegmenterSettings, SileroVAD

from .conftest import MEETING_WAV, SR, FakeClock


# ----------------------------------------------------------------------------- config
def test_repo_config_loads_with_seats_and_screen():
    cfg = load_sensors_config(DEFAULT_PATH)
    zones = seat_zones(cfg)
    assert "screen" in zones and len(zones) >= 4
    assert cfg["audio"]["sample_rate"] == 16000 and cfg["speakers"]["hz"] == 5


def test_missing_file_gives_defaults(tmp_path):
    cfg = load_sensors_config(tmp_path / "nope.toml")
    assert cfg["video"]["fps"] == 5 and cfg["seats"] == {}


def test_merge_defaults_keeps_extra_sections():
    cfg = merge_defaults({"audio": {"device": "Jabra"}, "replay": {"realtime": False}})
    assert cfg["audio"]["device"] == "Jabra" and cfg["audio"]["chunk_ms"] == 32
    assert cfg["replay"] == {"realtime": False}


def test_seat_zones_parse_lists_dicts_and_skip_bad_rows():
    zones = seat_zones({"seats": {"s1": [0, 0, 0.5, 1], "s2": {"x": 0.5, "y": 0, "w": 0.5, "h": 1},
                                  "bad": [0, 0], "zero": [0, 0, 0, 1]}})
    assert set(zones) == {"s1", "s2"}


def test_seat_for_overlap_then_centre_then_none():
    zones = {"s1": (0.0, 0.0, 0.5, 1.0), "screen": (0.7, 0.0, 0.3, 0.5)}
    assert seat_for((0.1, 0.1, 0.2, 0.3), zones) == "s1"
    assert seat_for((0.75, 0.1, 0.1, 0.1), zones) == "screen"
    assert seat_for((0.55, 0.6, 0.1, 0.1), zones) is None     # between the zones
    assert seat_for((0.4, 0.4, 0.2, 0.2), zones) == "s1"       # half inside s1 (share >= 0.3)


# ----------------------------------------------------------------------------- audio helpers
def test_pcm_roundtrip_and_rms():
    x = np.sin(np.linspace(0, 200, 16000)).astype(np.float32) * 0.5
    pcm = A.float_to_pcm16(x)
    assert len(pcm) == 32000
    assert abs(A.rms(pcm) - 0.5 / np.sqrt(2)) < 0.01
    assert np.allclose(A.pcm16_to_float(pcm), x, atol=1e-3)


def test_resample_integer_and_fractional():
    pcm48 = A.float_to_pcm16(np.zeros(4800, np.float32))
    assert len(A.resample_pcm16(pcm48, 48000, 16000)) == 1600 * 2
    pcm44 = A.float_to_pcm16(np.zeros(4410, np.float32))
    assert len(A.resample_pcm16(pcm44, 44100, 16000)) == 1600 * 2
    assert A.resample_pcm16(b"\x00\x00" * 10, 16000, 16000) == b"\x00\x00" * 10


def test_read_wav_handles_stereo_and_other_rates(tmp_path):
    x = (np.sin(np.linspace(0, 100, 48000)) * 10000).astype("<i2")
    p = tmp_path / "s.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(48000)
        w.writeframes(np.stack([x, x], axis=1).tobytes())
    pcm = A.read_wav_pcm16(p)
    assert len(pcm) == 16000 * 2        # 1 s at 16 kHz mono


async def test_wav_source_fast_mode_streams_every_byte_with_media_times():
    clock = FakeClock(100.0)
    src = A.WavFileSource(MEETING_WAV, clock=clock, realtime=False, chunk_ms=32)
    chunks = [c async for c in src]
    assert sum(len(c.pcm16) for c in chunks) == len(src.load())
    assert chunks[0].t == 100.0 and abs(chunks[10].t - (100.0 + 10 * 0.032)) < 1e-6
    assert all(len(c.pcm16) == WINDOW * 2 for c in chunks[:-1])
    assert src.finished.is_set() and abs(src.media_time - src.duration) < 1e-6


async def test_wav_source_realtime_paces_on_the_clock(monkeypatch):
    clock = FakeClock(5.0)
    sleeps: list[float] = []

    async def fake_sleep(dt):
        sleeps.append(dt)
        clock.advance(dt)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    src = A.WavFileSource(MEETING_WAV, clock=clock, realtime=True, chunk_ms=32)
    n = 0
    async for _ in src:
        n += 1
        if n == 50:
            break
    assert sum(sleeps) == pytest.approx(49 * 0.032, abs=1e-6)


async def test_mic_source_reports_missing_backends(monkeypatch):
    monkeypatch.setattr(A.shutil, "which", lambda name: None)
    src = A.MicSource(clock=FakeClock(), backend="arecord")
    with pytest.raises(RuntimeError, match="arecord: not installed"):
        async for _ in src:
            pass


def test_pick_input_device_skips_when_no_name():
    assert A.pick_input_device("") is None
    assert A.default_audio_backend() in ("sounddevice", "arecord", "none")


# ----------------------------------------------------------------------------- VAD
def _segments(vad, pcm: bytes, chunk: int = 1024):
    st = vad.stream(SR)
    segs = []
    for i in range(0, len(pcm), chunk):
        segs += st.feed(pcm[i:i + chunk])
    return segs + st.flush()


def _check_against_truth(segs, lines):
    assert len(segs) == len(lines), [(round(s.start, 2), round(s.end, 2)) for s in segs]
    for s, l in zip(segs, lines):
        overlap = min(s.end, l["end"]) - max(s.start, l["start"])
        assert overlap >= 0.8 * (l["end"] - l["start"]), (s.start, s.end, l)
        assert s.start >= l["start"] - 0.5 and s.end <= l["end"] + 0.7
        assert len(s.pcm16) == int(round((s.end - s.start) * SR)) * 2 and s.sample_rate == SR


@pytest.mark.parametrize("make", [SileroVAD, EnergyVAD], ids=["silero", "energy"])
def test_vad_finds_the_twelve_lines(make, meeting_pcm, truth):
    _check_against_truth(_segments(make(), meeting_pcm), truth["lines"])


def test_silero_is_fast_and_chunk_size_independent(meeting_pcm):
    vad = SileroVAD()
    a = _segments(vad, meeting_pcm, chunk=1024)
    b = _segments(vad, meeting_pcm, chunk=700)       # odd sizes: windows are re-assembled across feeds
    assert [(round(s.start, 2), round(s.end, 2)) for s in a] == [(round(s.start, 2), round(s.end, 2)) for s in b]


def test_silero_settings_from_options():
    vad = SileroVAD(options={"threshold": 0.7, "min_silence_ms": 300})
    assert vad.settings.threshold == 0.7 and vad.settings.min_silence_ms == 300 and vad.settings.pad_ms == 200
    assert SegmenterSettings.from_options(None) == SegmenterSettings()
    assert vad.healthy()


def test_segmenter_ignores_short_blips_and_splits_long_speech():
    vad = EnergyVAD(min_speech_ms=250, min_silence_ms=300, max_segment_s=1.0, pad_ms=0)
    st = vad.stream(SR)
    rng = np.random.default_rng(0)
    noise = A.float_to_pcm16(rng.normal(0, 0.001, SR * 2).astype(np.float32))      # 2 s silence (sets the floor)
    blip = A.float_to_pcm16(rng.normal(0, 0.3, 1024).astype(np.float32))            # 64 ms: too short
    talk = A.float_to_pcm16(rng.normal(0, 0.3, SR * 3).astype(np.float32))          # 3 s: split at 1 s
    segs = st.feed(noise) + st.feed(blip) + st.feed(noise) + st.feed(talk) + st.feed(noise) + st.flush()
    assert len(segs) == 3 and all(0.9 <= s.end - s.start <= 1.1 for s in segs[:2])
    assert segs[0].start >= 4.0


def test_energy_vad_silence_gives_nothing():
    st = EnergyVAD().stream(SR)
    assert st.feed(b"\x00\x00" * SR * 3) == [] and st.flush() == []
