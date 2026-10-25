"""Real PyAV fixtures and deterministic clocks; no ffmpeg binary or hardware."""
from __future__ import annotations

import asyncio
import math
import threading
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

import unspoken.meeting.replay_media as replay
from unspoken.meeting.replay_media import (
    MediaValidationError,
    ReplayAudioSource,
    ReplayClock,
    ReplayVideoSource,
    probe_media,
)


def make_media(
    path: Path, *, audio: bool = True, video: bool = True, fps: int = 25,
    video_start: float = 0, audio_start: float = 0, lossless: bool = False,
) -> Path:
    """One second of genuine stereo tone and distinct tiny video observations."""
    with av.open(str(path), "w") as output:
        vs = output.add_stream("ffv1" if lossless else "mpeg4", rate=fps) if video else None
        if vs is not None:
            vs.width, vs.height, vs.pix_fmt = 64, 48, "yuv420p"
        aus = output.add_stream("pcm_s16le" if lossless else "aac", rate=48000) if audio else None
        if aus is not None:
            aus.layout = "stereo"

        # Interleave by presentation time, rather than buffering a whole file.
        events = []
        if video:
            events.extend((video_start + i / fps, "v", i) for i in range(fps))
        if audio:
            events.extend((audio_start + i / 48000, "a", i) for i in range(0, 48000, 960))
        for _, kind, i in sorted(events):
            if kind == "v":
                image = np.zeros((48, 64, 3), dtype=np.uint8)
                image[:, :, 0] = i * 7 % 256
                image[:, :, 1] = 30
                image[:, :, 2] = 190
                frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                frame.pts = round(video_start * fps) + i
                frame.time_base = Fraction(1, fps)
                stream = vs
            else:
                samples = np.arange(i, i + 960)
                tone = np.rint(6000 * np.sin(2 * np.pi * 1000 * samples / 48000)).astype(np.int16)
                stereo = np.column_stack((tone, tone)).reshape(1, -1)
                frame = av.AudioFrame.from_ndarray(stereo, format="s16", layout="stereo")
                frame.pts = round(audio_start * 48000) + i
                frame.time_base = Fraction(1, 48000)
                frame.sample_rate = 48000
                stream = aus
            for packet in stream.encode(frame):
                output.mux(packet)
        for stream in (vs, aus):
            if stream is not None:
                for packet in stream.encode():
                    output.mux(packet)
    return path


@pytest.fixture
def mp4(tmp_path):
    return make_media(tmp_path / "real-av.mp4", fps=30)


@pytest.fixture
def offset_media(tmp_path):
    return make_media(tmp_path / "offset-av.mkv", lossless=True, video_start=0.4, audio_start=0.52)


class ManualTime:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, clock, seconds):
        self.value += seconds
        if clock.playing:
            # Wake real ReplayClock waits without waiting for real time.
            clock.pause()
            clock.play()


class ObservedClock(ReplayClock):
    def __init__(self):
        self.time = ManualTime()
        super().__init__(monotonic=self.time)
        self.requests = asyncio.Queue()

    async def wait_until(self, target, **kwargs):
        self.requests.put_nowait(target)
        return await super().wait_until(target, **kwargs)

    async def requested(self, target):
        async with asyncio.timeout(5):
            while not math.isclose(await self.requests.get(), target, abs_tol=1e-8):
                pass


class AutomaticClock(ReplayClock):
    """Drive deadlines instantly for sample/PTS tests (not pause/race tests)."""
    def __init__(self):
        self.time = ManualTime()
        super().__init__(monotonic=self.time)
        self.play()

    async def wait_until(self, target, **kwargs):
        self.time.advance(self, max(0, target - self.now()) + 1e-9)
        return await super().wait_until(target, **kwargs)


async def result(task):
    # Only a hang guard; test success depends on events/fake time, not elapsed time.
    return await asyncio.wait_for(task, timeout=5)


def test_probe_real_mp4(mp4):
    info = probe_media(mp4)
    assert info.path == mp4.resolve()
    assert info.duration == pytest.approx(1)
    assert info.origin == pytest.approx(0)
    assert info.audio_start == pytest.approx(0)
    assert info.video_start == pytest.approx(0)
    assert info.audio_end == pytest.approx(1)
    assert (info.sample_rate, info.channels) == (48000, 2)
    assert (info.width, info.height, info.fps) == (64, 48, 30)


@pytest.mark.parametrize("missing", ["audio", "video"])
def test_probe_requires_both_streams(tmp_path, missing):
    path = make_media(tmp_path / "one-stream.mp4", audio=missing != "audio", video=missing != "video")
    with pytest.raises(MediaValidationError, match=f"no {missing} stream"):
        probe_media(path)


def test_probe_missing_directory_corrupt_and_url_paths(tmp_path, monkeypatch):
    with pytest.raises(MediaValidationError):
        probe_media(tmp_path / "missing.mp4")
    with pytest.raises(MediaValidationError, match="regular file"):
        probe_media(tmp_path)
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not an mp4")
    with pytest.raises(MediaValidationError, match="cannot decode"):
        probe_media(corrupt)

    def must_not_open(_):
        pytest.fail("nonlocal/missing paths must never reach PyAV")

    monkeypatch.setattr(replay, "_open", must_not_open)
    with pytest.raises(MediaValidationError):
        probe_media(Path("https://example.invalid/meeting.mp4"))


def test_probe_common_origin_retains_stream_offset(offset_media):
    info = probe_media(offset_media)
    assert info.origin == pytest.approx(0.4)
    assert info.video_start == pytest.approx(0)
    assert info.audio_start == pytest.approx(0.12)
    assert info.duration == pytest.approx(1.12)


async def test_clock_pause_resume_is_idempotent_and_does_not_count_paused_time():
    clock = ObservedClock()
    assert clock.now() == 0 and not clock.playing
    due_now = asyncio.create_task(clock.wait_until(0))
    await clock.requested(0)
    assert not due_now.done()  # even t=0 is blocked initially
    clock.time.advance(clock, 100)
    clock.play()
    clock.play()
    assert await result(due_now)
    clock.time.advance(clock, 0.25)
    assert clock.now() == pytest.approx(0.25)
    clock.pause()
    clock.pause()
    clock.time.advance(clock, 100)
    assert clock.now() == pytest.approx(0.25)
    due = asyncio.create_task(clock.wait_until(0.5))
    await clock.requested(0.5)
    clock.play()
    assert not due.done()
    clock.time.advance(clock, 0.251)
    assert await result(due)
    assert clock.now() == pytest.approx(0.501)


async def test_clock_reset_invalidates_paused_and_timed_waiters_even_after_immediate_play():
    clock = ObservedClock()
    paused = asyncio.create_task(clock.wait_until(20))
    await clock.requested(20)
    assert clock.reset() == 1
    clock.play()
    assert await result(paused) is False
    timed = asyncio.create_task(clock.wait_until(50))
    await clock.requested(50)
    clock.time.advance(clock, 2)
    clock.reset()
    clock.play()
    assert await result(timed) is False
    assert await clock.wait_until(0, generation=0) is False
    assert clock.now() == 0 and clock.generation == 2


async def test_waiter_cancellation_and_stop_are_independent():
    clock = ObservedClock()
    stop = asyncio.Event()
    cancelled = asyncio.create_task(clock.wait_until(4))
    stopped = asyncio.create_task(clock.wait_until(5, stop=stop))
    survivor = asyncio.create_task(clock.wait_until(6))
    await clock.requested(6)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await result(cancelled)
    stop.set()
    assert await result(stopped) is False
    assert not survivor.done()
    clock.reset()
    assert await result(survivor) is False
    with pytest.raises(ValueError, match="finite"):
        await clock.wait_until(float("nan"))


async def test_audio_real_mp4_resampling_sample_count_pts_and_tone(mp4):
    source = ReplayAudioSource(mp4, clock=AutomaticClock())
    chunks = [chunk async for chunk in source]
    pcm = np.frombuffer(b"".join(chunk.pcm16 for chunk in chunks), dtype="<i2")
    # AAC padding is trimmed to the declared stream end, resampler tail flushed.
    assert len(pcm) == 16000
    assert len(chunks) == 32
    assert [c.t for c in chunks] == pytest.approx([i * 512 / 16000 for i in range(32)])
    assert all(c.sample_rate == 16000 for c in chunks)
    assert all(len(c.pcm16) == 1024 for c in chunks[:-1])
    assert len(chunks[-1].pcm16) == 256
    assert chunks[-1].t + chunks[-1].duration == pytest.approx(1)
    assert 3500 < np.std(pcm.astype(float)) < 5000
    frequencies = np.fft.rfftfreq(len(pcm), 1 / 16000)
    assert frequencies[np.argmax(np.abs(np.fft.rfft(pcm)))] == pytest.approx(1000)
    assert source.finished.is_set() and source._decoder is None


async def test_lossless_offset_audio_sample_grid_and_pcm(offset_media):
    info = probe_media(offset_media)
    source = ReplayAudioSource(offset_media, clock=AutomaticClock(), media=info, chunk_ms=37)
    chunks = [chunk async for chunk in source]
    pcm = np.frombuffer(b"".join(chunk.pcm16 for chunk in chunks), dtype="<i2")
    assert len(pcm) == 16000
    assert chunks[0].t == pytest.approx(0.12)  # not independently re-zeroed
    assert chunks[-1].t + chunks[-1].duration == pytest.approx(1.12)
    for a, b in zip(chunks, chunks[1:]):
        assert a.t + a.duration == pytest.approx(b.t, abs=1e-10)
    expected = 6000 * np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000)
    # Ignore filter transients at the two edges, but verify actual sample values.
    assert np.max(np.abs(pcm[32:-32] - expected[32:-32])) < 5


async def test_30fps_video_targets_25_using_real_pts_not_integer_stride(mp4):
    source = ReplayVideoSource(mp4, clock=AutomaticClock(), width=0, height=0)
    frames = [frame async for frame in source]
    indices = [1 + math.ceil(i * 30 / 25 - 1e-8) for i in range(25)]
    assert len(frames) == 25
    assert [f.index for f in frames] == indices
    assert [f.t for f in frames] == pytest.approx([(i - 1) / 30 for i in indices])
    assert len({f.index for f in frames}) == len(frames)
    assert frames[0].image.shape == (48, 64, 3)
    assert frames[0].image.dtype == np.uint8
    assert frames[0].image[20, 20, 2] > frames[0].image[20, 20, 0]  # BGR, not RGB
    assert sum(f.dropped for f in frames) == frames[-1].index - len(frames)
    assert source.finished.is_set() and source._decoder is None


async def test_low_fps_video_never_repeats_observations(tmp_path):
    path = make_media(tmp_path / "slow.mp4", fps=10)
    frames = [f async for f in ReplayVideoSource(path, clock=AutomaticClock(), fps=25)]
    assert len(frames) == 10
    assert [f.t for f in frames] == pytest.approx([i / 10 for i in range(10)])
    assert all(f.dropped == 0 for f in frames)


async def test_shared_clock_pause_resume_stops_both_sources_promptly(offset_media):
    media = probe_media(offset_media)
    clock = ObservedClock()
    audio = ReplayAudioSource(offset_media, clock=clock, media=media)
    video = ReplayVideoSource(offset_media, clock=clock, media=media)
    audio_next = asyncio.create_task(anext(audio))
    video_next = asyncio.create_task(anext(video))
    try:
        await clock.requested(0)
        await clock.requested(0)
        assert not audio_next.done() and not video_next.done()
        clock.play()
        first = await result(video_next)
        assert first.t == pytest.approx(0)
        await clock.requested(0.152)
        video_next = asyncio.create_task(anext(video))
        await clock.requested(0.04)
        clock.pause()
        clock.time.advance(clock, 60)
        assert clock.now() == pytest.approx(0)
        assert not audio_next.done() and not video_next.done()
        clock.play()
        clock.time.advance(clock, 0.040001)
        second = await result(video_next)
        assert second.t == pytest.approx(0.04)
        assert not audio_next.done()  # audio has a real 120 ms start offset
        clock.time.advance(clock, 0.112)
        chunk = await result(audio_next)
        assert chunk.t == pytest.approx(0.12)
        assert chunk.duration == pytest.approx(0.032)
        assert clock.now() == pytest.approx(0.152001)
    finally:
        await asyncio.gather(audio.aclose(), video.aclose())


async def test_late_video_drops_are_counted_not_retimestamped(offset_media):
    clock = ObservedClock()
    clock.play()
    video = ReplayVideoSource(offset_media, clock=clock)
    try:
        clock.time.advance(clock, 0.3)
        first = await result(anext(video))
        assert (first.index, first.dropped) == (8, 7)
        assert first.t == pytest.approx(0.28)
        clock.time.advance(clock, 0.4)
        second = await result(anext(video))
        assert (second.index, second.dropped) == (18, 9)
        assert second.t == pytest.approx(0.68)
    finally:
        await video.aclose()


@pytest.mark.parametrize("action", ["reset", "close"])
async def test_reset_or_close_unblocks_paused_iterators_and_closes_decoders(offset_media, monkeypatch, action):
    clock = ObservedClock()
    clock.play()
    media = probe_media(offset_media)
    original_open = replay._open
    live = []

    @contextmanager
    def tracked_open(path):
        with original_open(path) as container:
            live.append(container)
            try:
                yield container
            finally:
                live.remove(container)

    monkeypatch.setattr(replay, "_open", tracked_open)
    audio = ReplayAudioSource(offset_media, clock=clock, media=media)
    video = ReplayVideoSource(offset_media, clock=clock, media=media)
    pending_audio = asyncio.create_task(anext(audio))
    await result(anext(video))
    await clock.requested(0.152)
    pending_video = asyncio.create_task(anext(video))
    await clock.requested(0.04)
    assert len(live) == 2
    clock.pause()
    if action == "reset":
        clock.reset()
        clock.play()  # waking an old waiter must not make it part of this run
    else:
        await result(asyncio.gather(audio.aclose(), video.aclose()))
    for pending in (pending_audio, pending_video):
        with pytest.raises(StopAsyncIteration):
            await result(pending)
    assert not live
    assert audio.finished.is_set() and video.finished.is_set()
    await asyncio.gather(audio.aclose(), video.aclose())  # idempotent
    clock.play()
    for source in (audio, video):
        with pytest.raises(StopAsyncIteration):
            await anext(source)
    # A fresh source on the new generation starts at media zero.
    if action == "reset":
        fresh = ReplayVideoSource(offset_media, clock=clock, media=media)
        assert (await result(anext(fresh))).t == pytest.approx(0)
        await fresh.aclose()


async def test_close_before_start_and_after_yield_releases_resources(mp4):
    media = probe_media(mp4)
    source = ReplayAudioSource(mp4, clock=ReplayClock(), media=media)
    await source.aclose()
    with pytest.raises(StopAsyncIteration):
        await anext(source)
    assert source._decoder is None
    clock = ObservedClock()
    clock.play()
    source = ReplayVideoSource(mp4, clock=clock, media=media)
    await result(anext(source))
    assert source._decoder is not None
    clock.pause()
    # No outstanding anext: explicit close still releases the suspended generator.
    await source.aclose()
    assert source._decoder is None and source.finished.is_set()


@pytest.mark.parametrize("action", ["cancel", "reset", "close"])
async def test_inflight_decode_is_joined_before_container_closes(mp4, monkeypatch, action):
    media = probe_media(mp4)
    clock = ReplayClock()
    clock.play()
    original_open = replay._open
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release_worker = threading.Event()
    closed = []

    @contextmanager
    def blocked_open(path):
        with original_open(path) as container:
            try:
                loop.call_soon_threadsafe(started.set)
                if not release_worker.wait(5):
                    raise AssertionError("test never released decoder")
                yield container
            finally:
                closed.append(True)

    monkeypatch.setattr(replay, "_open", blocked_open)
    source = ReplayAudioSource(mp4, clock=clock, media=media)
    pending = asyncio.create_task(anext(source))
    closing = None
    try:
        await result(started.wait())
        if action == "cancel":
            pending.cancel()
        elif action == "reset":
            clock.reset()
            clock.play()
        else:
            closing = asyncio.create_task(source.aclose())
        # A scheduling turn, not a timing assertion or wall-time sleep.
        await asyncio.sleep(0)
        assert not closed
        assert not source.finished.is_set()
        release_worker.set()
        error = asyncio.CancelledError if action == "cancel" else StopAsyncIteration
        with pytest.raises(error):
            await result(pending)
        if closing:
            await result(closing)
        assert closed == [True]
        assert source.finished.is_set() and source._decoder is None
    finally:
        release_worker.set()
        await source.aclose()


def test_source_parameters_and_metadata_path_are_validated(mp4, tmp_path):
    media = probe_media(mp4)
    for bad in (0, -1, float("nan"), float("inf"), 1001):
        with pytest.raises(ValueError, match="chunk_ms"):
            ReplayAudioSource(mp4, clock=ReplayClock(), chunk_ms=bad)
    for bad in (0, -1, float("nan"), float("inf"), 241):
        with pytest.raises(ValueError, match="fps"):
            ReplayVideoSource(mp4, clock=ReplayClock(), fps=bad)
    other = make_media(tmp_path / "other.mp4")
    with pytest.raises(ValueError, match="different file"):
        ReplayAudioSource(other, clock=ReplayClock(), media=media)
