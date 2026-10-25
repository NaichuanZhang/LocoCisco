"""Local, real-media inputs for a browser-controlled sensor replay.

Small public API (all clock/source controls belong on the same asyncio loop)::

    media = probe_media(Path("meeting.mp4"))  # synchronous; use to_thread in a server
    clock = ReplayClock()                   # initially paused at zero
    audio = ReplayAudioSource(media.path, clock=clock, media=media)
    video = ReplayVideoSource(media.path, clock=clock, media=media)  # target 25 fps
    # Inject audio/video into config["audio"/"video"]["source"]; pass clock.now
    # as the sensor/runtime clock. No microphone, camera, models, or network here.
    clock.play()
    clock.pause()
    clock.reset()                           # paused at zero; invalidates old sources
    await asyncio.gather(audio.aclose(), video.aclose())
    # For another run, construct NEW sources with the same media and clock.

Sources are single-run async iterators, including after EOF/close/reset. Reset
does not seek old decoders. The controller must also invalidate in-flight model
work; these sources can only prevent further input delivery. ``aclose`` is
idempotent and must be called when abandoning an iterator (also after ``break``).

One common origin is subtracted from both streams, retaining A/V start offsets.
Audio is real decoded, resampled 16 kHz mono little-endian PCM, paced at each
chunk's *end*, with ``AudioChunk.t`` naming its first sample. Video uses actual
presentation timestamps, never duplicates low-fps frames, and drops old frames
when the consumer falls behind. Audio is not dropped. Decoding is demand-driven
in worker threads, holding only decoder buffers, a chunk, and a frame, not the
whole recording. Paths are local regular files only; this module serves nothing.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Iterator, TypeVar

from unspoken.sensors.audio import SAMPLE_RATE, AudioChunk
from unspoken.sensors.video import VideoFrame, resize_to

__all__ = [
    "ReplayClock", "MediaInfo", "MediaValidationError", "probe_media",
    "ReplayAudioSource", "ReplayVideoSource",
]


class ReplayClock:
    """A pauseable media clock, independent of time spent decoding or paused.

    ``wait_until`` returns True only while playing and at/past the target.
    It returns False on generation change or when its optional stop event is set.
    Reset always pauses, rewinds to zero, and increments ``generation``. Even a
    waiter whose deadline is already past remains blocked while paused.

    ``monotonic`` is injectable for deterministic tests; production should use
    the default. Methods are loop-confined, not thread-safe.
    """

    def __init__(self, *, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._monotonic = monotonic
        self._position = 0.0
        self._anchor = monotonic()
        self._playing = False
        self._generation = 0
        self._changed = asyncio.Event()

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def generation(self) -> int:
        return self._generation

    def now(self) -> float:
        return self._position + (self._monotonic() - self._anchor if self._playing else 0.0)

    def _notify(self) -> None:
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    def play(self) -> None:
        if not self._playing:
            self._anchor = self._monotonic()
            self._playing = True
            self._notify()

    def pause(self) -> None:
        if self._playing:
            self._position = self.now()
            self._playing = False
            self._notify()

    def reset(self) -> int:
        self._position = 0.0
        self._anchor = self._monotonic()
        self._playing = False
        self._generation += 1
        self._notify()
        return self._generation

    async def wait_until(
        self, target: float, *, generation: int | None = None,
        stop: asyncio.Event | None = None,
    ) -> bool:
        if not math.isfinite(target):
            raise ValueError("target must be finite")
        generation = self.generation if generation is None else generation
        while True:
            if generation != self.generation or (stop is not None and stop.is_set()):
                return False
            remaining = target - self.now()
            if self.playing and remaining <= 0:
                return True
            # Capture the notification before yielding: pause/play/reset cannot
            # be lost between reading clock state and registering this wait.
            changed = asyncio.create_task(self._changed.wait())
            tasks = [changed]
            if stop is not None:
                tasks.append(asyncio.create_task(stop.wait()))
            try:
                await asyncio.wait(
                    tasks, timeout=max(0.0, remaining) if self.playing else None,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            # Always recheck pause/generation/stop, including after timer expiry.


class MediaValidationError(ValueError):
    """Not a readable local video with timestamped, decodable audio and video."""


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration: float
    origin: float                 # original container/stream seconds
    audio_start: float            # the following times are relative to origin
    video_start: float
    audio_end: float | None
    video_end: float | None
    audio_stream_index: int
    video_stream_index: int
    sample_rate: int              # native rate, not the replay's fixed 16 kHz
    channels: int
    width: int
    height: int
    fps: float | None


def _local_path(path: Path) -> Path:
    try:
        path = Path(path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise MediaValidationError("media must be a local regular file")
        return path
    except OSError as exc:
        raise MediaValidationError(f"media file is not readable: {path}") from exc


def _open(path: Path):
    import av

    # No URL inputs or network subresources (e.g. a misleading playlist file).
    return av.open(str(path), mode="r", options={"protocol_whitelist": "file", "enable_drefs": "0"})


def _timestamp(frame) -> float:
    if frame.pts is None or frame.time_base is None:
        raise MediaValidationError("media frames must have presentation timestamps")
    value = float(frame.pts * frame.time_base)
    if not math.isfinite(value):
        raise MediaValidationError("media has a non-finite presentation timestamp")
    return value


def probe_media(path: Path) -> MediaInfo:
    """Validate both streams by decoding their first frame and return metadata.

    Selects the first audio stream and first non-cover-art video stream.
    Requires a finite positive declared duration; timestamps are not inferred
    from frame counts. Raises MediaValidationError for missing/corrupt/unsupported
    media. This is blocking local I/O; a web controller should use ``to_thread``.
    """
    import av

    path = _local_path(path)
    try:
        with _open(path) as container:
            videos = [s for s in container.streams.video
                      if not s.disposition & av.stream.Disposition.attached_pic]
            if not container.streams.audio:
                raise MediaValidationError("media has no audio stream")
            if not videos:
                raise MediaValidationError("media has no video stream")
            audio = container.streams.audio[0]
            video = videos[0]
            audio_index, video_index = audio.index, video.index
            rate, channels = audio.codec_context.sample_rate, audio.codec_context.channels
            width, height = video.codec_context.width, video.codec_context.height
            fps = float(video.average_rate) if video.average_rate else None
            fallback_start = float(container.start_time or 0) / av.time_base
            # Matroska's Segment Duration is an end timestamp from segment zero,
            # including any initial gap. MOV/MP4 reports the elapsed duration.
            if "matroska" in container.format.name.split(","):
                fallback_start = 0.0
            fallback_end = (fallback_start + float(container.duration) / av.time_base
                            if container.duration is not None else None)
            starts, ends = [], []
            for stream in (audio, video):
                start = (float(stream.start_time * stream.time_base)
                         if stream.start_time is not None else None)
                end = (start + float(stream.duration * stream.time_base)
                       if start is not None and stream.duration is not None else None)
                starts.append(start)
                ends.append(end)

        # Separate readers ensure the first video frame isn't skipped while
        # searching for audio in a file with delayed/interleaved streams.
        for i, index in enumerate((audio_index, video_index)):
            with _open(path) as container:
                first = next(container.decode(container.streams[index]), None)
                if first is None:
                    raise MediaValidationError("media contains an empty audio or video stream")
                first_t = _timestamp(first)
                starts[i] = max(first_t, starts[i]) if starts[i] is not None else first_t
        origin = min(starts)
        available_ends = [end for end in ends if end is not None]
        if any(end is None for end in ends) and fallback_end is not None:
            available_ends.append(fallback_end)
        duration = max(available_ends) - origin if available_ends else 0.0
        if not math.isfinite(duration) or duration <= 0:
            raise MediaValidationError("media must declare a finite positive duration")
        if not rate or not channels or width <= 0 or height <= 0:
            raise MediaValidationError("media has invalid audio/video dimensions")
        return MediaInfo(
            path, duration, origin, starts[0] - origin, starts[1] - origin,
            ends[0] - origin if ends[0] is not None else None,
            ends[1] - origin if ends[1] is not None else None,
            audio_index, video_index, rate, channels, width, height, fps,
        )
    except MediaValidationError:
        raise
    except Exception as exc:
        raise MediaValidationError(f"cannot decode local media: {exc}") from exc


_Item = TypeVar("_Item", AudioChunk, VideoFrame)
_END = object()


class _ReplaySource(Generic[_Item]):
    def __init__(self, path: Path, *, clock: ReplayClock, media: MediaInfo | None) -> None:
        path = _local_path(path)
        self.media = media if media is not None else probe_media(path)
        if self.media.path != path:
            raise ValueError("media metadata belongs to a different file")
        self.path, self.clock = path, clock
        self.generation = clock.generation
        self.finished = asyncio.Event()
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()
        self._decoder: Iterator[_Item] | None = None
        self._inflight: asyncio.Task | None = None
        self._close_task: asyncio.Task | None = None

    def __aiter__(self):
        return self

    def _decode(self) -> Iterator[_Item]:
        raise NotImplementedError

    def _due(self, item: _Item) -> float:
        return item.t

    def _accept(self, item: _Item) -> bool:
        return True

    async def __anext__(self) -> _Item:
        try:
            async with self._lock:
                while True:
                    if not await self.clock.wait_until(0, generation=self.generation, stop=self._stop):
                        raise StopAsyncIteration
                    if self._decoder is None:
                        self._decoder = self._decode()
                    # Shield the *worker*, not the caller: cancelling anext must
                    # never close a PyAV container while its worker is decoding.
                    self._inflight = asyncio.create_task(asyncio.to_thread(next, self._decoder, _END))
                    item = await asyncio.shield(self._inflight)
                    if item is _END:
                        raise StopAsyncIteration
                    if not await self.clock.wait_until(
                        self._due(item), generation=self.generation, stop=self._stop,
                    ):
                        raise StopAsyncIteration
                    if self._accept(item):
                        return item
        except BaseException:
            # Outside the lock: concurrent aclose is allowed to acquire it.
            await self.aclose()
            raise

    async def _release(self) -> None:
        try:
            async with self._lock:
                if self._inflight is not None:
                    try:
                        await asyncio.shield(self._inflight)
                    except Exception:
                        pass  # the original decoding error is raised by anext
                if self._decoder is not None:
                    await asyncio.to_thread(self._decoder.close)
                    self._decoder = None
                self._inflight = None
        finally:
            self.finished.set()

    async def aclose(self) -> None:
        """Stop delivery immediately and await safe decoder teardown.

        Wakes paused anext calls. An in-progress local decode is allowed to
        finish before closing its container; no worker is abandoned. If the
        caller itself is cancelled, shielded teardown still runs to completion.
        """
        self._stop.set()
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._release())
        await asyncio.shield(self._close_task)


class ReplayAudioSource(_ReplaySource[AudioChunk]):
    """Streaming PCM16 source. chunk_ms must be in (0, 1000], default 32 ms."""

    def __init__(
        self, path: Path, *, clock: ReplayClock, media: MediaInfo | None = None,
        chunk_ms: float = 32,
    ) -> None:
        if not math.isfinite(chunk_ms) or not 0 < chunk_ms <= 1000:
            raise ValueError("chunk_ms must be in (0, 1000]")
        self.chunk_samples = max(1, round(SAMPLE_RATE * chunk_ms / 1000))
        self.sample_rate = SAMPLE_RATE
        super().__init__(path, clock=clock, media=media)

    def _due(self, item: AudioChunk) -> float:
        return item.t + item.duration

    def _decode(self) -> Iterator[AudioChunk]:
        import av

        media = self.media
        chunk_bytes = self.chunk_samples * 2
        # Integer output sample positions avoid cumulative float timestamp drift.
        lower = math.ceil((media.origin + media.audio_start) * SAMPLE_RATE - 1e-7)
        upper = (math.ceil((media.origin + media.audio_end) * SAMPLE_RATE - 1e-7)
                 if media.audio_end is not None else None)
        pending = bytearray()
        pending_start = 0
        with _open(self.path) as container:
            stream = container.streams[media.audio_stream_index]
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)

            def converted():
                for frame in container.decode(stream):
                    _timestamp(frame)
                    yield from resampler.resample(frame)
                yield from resampler.resample(None)  # preserve the resampler tail

            for frame in converted():
                first = round(_timestamp(frame) * SAMPLE_RATE)
                last = first + frame.samples
                start, end = max(first, lower), min(last, upper) if upper is not None else last
                if end <= start:
                    continue
                raw = frame.to_ndarray().reshape(-1)[start - first:end - first]
                raw = raw.astype("<i2", copy=False).tobytes()
                if pending and start != pending_start + len(pending) // 2:
                    # A real timestamp gap stays a gap, not invented audio.
                    yield AudioChunk(bytes(pending), SAMPLE_RATE, pending_start / SAMPLE_RATE - media.origin)
                    pending.clear()
                if not pending:
                    pending_start = start
                pending.extend(raw)
                while len(pending) >= chunk_bytes:
                    yield AudioChunk(bytes(pending[:chunk_bytes]), SAMPLE_RATE,
                                     pending_start / SAMPLE_RATE - media.origin)
                    del pending[:chunk_bytes]
                    pending_start += self.chunk_samples
            if pending:
                yield AudioChunk(bytes(pending), SAMPLE_RATE, pending_start / SAMPLE_RATE - media.origin)


class ReplayVideoSource(_ReplaySource[VideoFrame]):
    """Actual BGR observations at up to fps, default 25; never repeats a frame.

    Sampling uses source PTS, not rounded native/target fps ratios (e.g. 30 fps
    input targets 25, not 30). ``index`` is the 1-based decoded-frame index;
    ``dropped`` includes rate limiting and frames older than one target period.
    Set width=height=0 to preserve native dimensions.
    """

    def __init__(
        self, path: Path, *, clock: ReplayClock, media: MediaInfo | None = None,
        fps: float = 25, width: int = 640, height: int = 360,
    ) -> None:
        if not math.isfinite(fps) or not 0 < fps <= 240:
            raise ValueError("fps must be in (0, 240]")
        if width < 0 or height < 0:
            raise ValueError("video dimensions cannot be negative")
        self.fps = float(fps)
        self.width, self.height = int(width), int(height)
        self._taken = 0
        super().__init__(path, clock=clock, media=media)

    def _accept(self, item: VideoFrame) -> bool:
        if self.clock.now() - item.t > 1 / self.fps + 1e-7:
            return False
        item.dropped = item.index - self._taken - 1
        self._taken = item.index
        return True

    def _decode(self) -> Iterator[VideoFrame]:
        next_due = self.media.video_start
        previous = -math.inf
        with _open(self.path) as container:
            for index, frame in enumerate(container.decode(container.streams[self.media.video_stream_index]), 1):
                timestamp = _timestamp(frame) - self.media.origin
                if timestamp <= previous:
                    continue  # duplicate/non-increasing PTS is not a new observation
                previous = timestamp
                if timestamp + 1e-7 < next_due:
                    continue
                # Keep the target grid anchored even for VFR, gaps, or low fps.
                tick = math.floor((timestamp - self.media.video_start) * self.fps + 1e-7) + 1
                next_due = self.media.video_start + tick / self.fps
                image = resize_to(frame.to_ndarray(format="bgr24"), self.width, self.height)
                yield VideoFrame(image, timestamp, index)
