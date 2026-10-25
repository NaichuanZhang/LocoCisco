"""The sensors pipeline: `start_sensors(bus, registry, config, clock) -> SensorHandle`.

    audio source -> VAD stream -> SpeechSegment queue -> [ASR || diarizer] -> MT.SEGMENT, PCM dropped
    video source -> newest frame only -> faces -> ASD (with the PCM since its last call) -> MT.SPEAKERS (~5 Hz)

Both paths stamp times on the one `clock` the caller passes (the meeting clock). Event.ts is the moment of
the observation on that clock (segment end, frame time), so replayed and live events look the same.

The capture coroutines never wait on a model: the audio loop only feeds the VAD (sub-millisecond) and
enqueues closed segments; ASR and diarization run in worker threads via the providers; face detection runs
in a thread too and the video loop skips frames while one is in flight (frames are dropped, audio never).

Sources come from `config` (see config.py): a live mic/camera by default, a WAV/MP4 when
config["audio"]["path"] / config["video"]["path"] are set (replay; config["replay"]["realtime"] = False for a
fast run), or ready-made source objects in config["audio"]["source"] / config["video"]["source"] (tests).
Roles the registry reports as disabled are skipped: no diarizer -> voice None; no faces -> no video path;
no ASD -> speaking 0.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from ..contracts import ASR, VAD, ActiveSpeaker, Diarizer, FaceTracker, Registry, RoleDisabled, SpeechSegment
from ..core.events import BusAPI, Event
from ..meeting.contracts import MT, FaceData, SegmentData, SpeakersData
from .audio import SAMPLE_RATE, AudioChunk, AudioSource, MicSource, WavFileSource
from .config import merge_defaults, seat_for, seat_zones
from .video import CameraSource, Mp4FileSource, VideoFrame, VideoSource

log = logging.getLogger("unspoken.sensors")
Clock = Callable[[], float]
SOURCE_AUDIO = "sensors.audio"
SOURCE_VIDEO = "sensors.video"


@dataclass
class SensorStats:
    audio_chunks: int = 0
    segments: int = 0            # closed by the VAD
    segments_published: int = 0
    segments_empty: int = 0      # ASR returned no text
    frames: int = 0              # frames taken from the source
    frames_skipped: int = 0      # taken but not processed (rate limit / detector busy)
    frames_dropped: int = 0      # dropped by the source before we saw them
    speakers_published: int = 0
    asr_ms: list[float] = field(default_factory=list)
    diar_ms: list[float] = field(default_factory=list)
    faces_ms: list[float] = field(default_factory=list)
    segment_publish_ms: list[float] = field(default_factory=list)
    errors: int = 0

    def summary(self) -> dict[str, Any]:
        avg = lambda xs: round(sum(xs) / len(xs), 1) if xs else None  # noqa: E731
        p95 = lambda xs: round(sorted(xs)[max(0, math.ceil(len(xs) * .95) - 1)], 1) if xs else None
        return {"audio_chunks": self.audio_chunks, "segments": self.segments,
                "segments_published": self.segments_published, "segments_empty": self.segments_empty,
                "frames": self.frames, "frames_skipped": self.frames_skipped, "frames_dropped": self.frames_dropped,
                "speakers_published": self.speakers_published, "asr_ms_avg": avg(self.asr_ms),
                "diar_ms_avg": avg(self.diar_ms), "faces_ms_avg": avg(self.faces_ms),
                "asr_ms_p95": p95(self.asr_ms), "diar_ms_p95": p95(self.diar_ms),
                "segment_publish_ms_p95": p95(self.segment_publish_ms), "errors": self.errors}


class SensorHandle:
    """What start_sensors returns. `await stop()` cancels capture, forgets every voice/face and drops buffers.
    `await wait()` returns when file sources are exhausted and every segment is published (replay)."""

    def __init__(self) -> None:
        self.tasks: list[asyncio.Task] = []
        self.sources: list[Any] = []
        self.sessions: list[Any] = []
        self.stats = SensorStats()
        self.audio_done = asyncio.Event()
        self.video_done = asyncio.Event()
        self.audio_media_time = 0.0       # seconds of audio consumed (relative to the first chunk)
        self.video_media_time = 0.0       # seconds of video consumed (relative to the first frame)
        self.models: dict[str, str] = {}
        self._stopped = False

    async def wait(self, timeout: float | None = None) -> None:
        await asyncio.wait_for(asyncio.gather(self.audio_done.wait(), self.video_done.wait()), timeout)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        for src in self.sources:
            try:
                await src.aclose()
            except Exception:
                log.debug("source close failed", exc_info=True)
        for t in self.tasks:
            t.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        for s in self.sessions:                  # voices and faces are session-only: wipe them
            reset = getattr(s, "reset", None)
            if callable(reset):
                try:
                    reset()
                except Exception:
                    pass
        self.sessions.clear()
        self.audio_done.set()
        self.video_done.set()


def _get(registry: Registry, role: str, getter: str) -> Any | None:
    enabled = getattr(registry, "enabled", None)
    if callable(enabled):
        try:
            if not enabled(role):
                return None
        except Exception:
            return None
    try:
        return getattr(registry, getter)()
    except RoleDisabled:
        return None
    except AttributeError:
        return None


def _label(registry: Registry, role: str, obj: Any) -> str:
    lab = getattr(registry, "label", None)
    if callable(lab):
        try:
            v = lab(role)
            if v:
                return str(v)
        except Exception:
            pass
    return str(getattr(obj, "label", type(obj).__name__))


def _audio_source(config: dict[str, Any], clock: Clock, pace: Callable[[], float] | None) -> AudioSource | None:
    a = config["audio"]
    if a.get("source") is not None:
        return a["source"]
    replay = config.get("replay") or {}
    if a.get("path"):
        realtime = bool(replay.get("realtime", True))
        both_files = bool(config["video"].get("path")) and not config["video"].get("source")
        return WavFileSource(a["path"], clock=clock, realtime=realtime, chunk_ms=int(a.get("chunk_ms", 32)),
                             speed=float(replay.get("speed", 1.0)),
                             pace=pace if (not realtime and both_files) else None)
    if a.get("enabled", True) is False:
        return None
    return MicSource(clock=clock, device=str(a.get("device", "")), alsa_device=str(a.get("alsa_device", "default")),
                     chunk_ms=int(a.get("chunk_ms", 32)), backend=str(a.get("backend", "auto")))


def _video_source(config: dict[str, Any], clock: Clock, pace: Callable[[], float] | None) -> VideoSource | None:
    v = config["video"]
    if v.get("source") is not None:
        return v["source"]
    replay = config.get("replay") or {}
    if v.get("path"):
        realtime = bool(replay.get("realtime", True))
        return Mp4FileSource(v["path"], clock=clock, realtime=realtime, fps=float(v.get("fps", 5)),
                             width=int(v.get("width", 640)), height=int(v.get("height", 360)),
                             speed=float(replay.get("speed", 1.0)), pace=None if realtime else pace)
    if v.get("enabled", True) is False or config["audio"].get("path"):
        return None   # audio-only replay: no camera
    return CameraSource(clock=clock, device=v.get("device", 0), fps=float(v.get("fps", 5)),
                        width=int(v.get("width", 640)), height=int(v.get("height", 360)))


async def start_sensors(bus: BusAPI, registry: Registry, config: dict[str, Any], clock: Clock) -> SensorHandle:
    cfg = merge_defaults(dict(config or {}))
    handle = SensorHandle()
    zones = seat_zones(cfg)
    speaking_threshold = float(cfg["speakers"].get("speaking_threshold", 0.5))
    hz = float(cfg["speakers"].get("hz", 5.0))

    vad: VAD | None = _get(registry, "vad", "vad")
    asr: ASR | None = _get(registry, "asr", "asr")
    diar: Diarizer | None = _get(registry, "diarization", "diarizer")
    faces: FaceTracker | None = _get(registry, "faces", "faces")
    asd: ActiveSpeaker | None = _get(registry, "asd", "asd")
    for role, obj in (("vad", vad), ("asr", asr), ("diarization", diar), ("faces", faces), ("asd", asd)):
        handle.models[role] = _label(registry, role, obj) if obj is not None else "disabled"
    if faces is not None and zones and hasattr(faces, "configure_seats"):
        faces.configure_seats(zones)

    ring: deque[AudioChunk] = deque()       # ~2 s of recent PCM for the ASD; nothing older is kept
    ring_seconds = 2.0

    # fast replay of two files: audio may run at most 1 s ahead of video, video never ahead of audio
    def audio_pace() -> float:
        return float("inf") if handle.video_done.is_set() or faces is None else handle.video_media_time + 1.0

    def video_pace() -> float:
        return float("inf") if handle.audio_done.is_set() else handle.audio_media_time

    # ------------------------------------------------------------------ audio
    audio_src = _audio_source(cfg, clock, audio_pace) if (vad is not None and asr is not None) else None
    if audio_src is None:
        handle.audio_done.set()
        if vad is None or asr is None:
            log.warning("audio path off: vad=%s asr=%s", handle.models["vad"], handle.models["asr"])
    else:
        handle.sources.append(audio_src)
        seg_queue: asyncio.Queue[SpeechSegment | None] = asyncio.Queue()
        diar_session = diar.session() if diar is not None else None
        if diar_session is not None:
            handle.sessions.append(diar_session)

        async def audio_loop() -> None:
            stream = vad.stream(SAMPLE_RATE)
            t_first: float | None = None
            fed = 0
            try:
                async for chunk in audio_src:
                    if t_first is None:
                        t_first = chunk.t
                    handle.stats.audio_chunks += 1
                    fed += len(chunk.pcm16)
                    handle.audio_media_time = fed / 2 / SAMPLE_RATE
                    ring.append(chunk)
                    while ring and chunk.t - ring[0].t > ring_seconds:
                        ring.popleft()
                    for seg in stream.feed(chunk.pcm16):
                        seg.start += t_first
                        seg.end += t_first
                        handle.stats.segments += 1
                        seg_queue.put_nowait(seg)
                for seg in stream.flush():
                    seg.start += t_first or 0.0
                    seg.end += t_first or 0.0
                    handle.stats.segments += 1
                    seg_queue.put_nowait(seg)
            finally:
                seg_queue.put_nowait(None)

        async def segment_worker() -> None:
            try:
                while True:
                    seg = await seg_queue.get()
                    if seg is None:
                        return
                    try:
                        await publish_segment(seg)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        handle.stats.errors += 1
                        log.exception("segment failed")
                    finally:
                        del seg          # the PCM is gone with the segment object
            finally:
                handle.audio_done.set()

        async def publish_segment(seg: SpeechSegment) -> None:
            t0 = time.perf_counter()
            if diar_session is not None:
                tr, vl = await asyncio.gather(asr.transcribe(seg.pcm16, seg.sample_rate),
                                              diar_session.label(seg.pcm16, seg.sample_rate))
            else:
                tr, vl = await asr.transcribe(seg.pcm16, seg.sample_rate), None
            asr_ms = float(getattr(tr, "latency_ms", 0.0) or 0.0)
            diar_ms = float(getattr(diar_session, "last_ms", 0.0) or 0.0) if diar_session is not None else 0.0
            if asr_ms <= 0:
                asr_ms = (time.perf_counter() - t0) * 1000
            handle.stats.asr_ms.append(asr_ms)
            if vl is not None:
                handle.stats.diar_ms.append(diar_ms)
            text = (tr.text or "").strip()
            if not text:
                handle.stats.segments_empty += 1
                return
            data: SegmentData = {
                "segment_id": uuid.uuid4().hex[:10],
                "start": round(seg.start, 3), "end": round(seg.end, 3),
                "text": text, "lang": tr.lang or "en", "asr_conf": round(float(tr.conf), 3),
                "voice": vl.label if vl is not None else None,
                "voice_conf": round(float(vl.conf), 3) if vl is not None else 0.0,
                "asr_ms": round(asr_ms, 1), "diar_ms": round(diar_ms, 1),
            }
            await bus.publish(Event(MT.SEGMENT, dict(data), source=SOURCE_AUDIO, ts=round(seg.end, 3)))
            handle.stats.segment_publish_ms.append(max(0, (clock() - seg.end) * 1000))
            handle.stats.segments_published += 1

        handle.tasks.append(asyncio.create_task(audio_loop(), name="sensors.audio"))
        handle.tasks.append(asyncio.create_task(segment_worker(), name="sensors.segments"))

    # ------------------------------------------------------------------ video
    video_src = _video_source(cfg, clock, video_pace) if faces is not None else None
    if video_src is None:
        handle.video_done.set()
    else:
        handle.sources.append(video_src)
        face_session = faces.session()
        asd_session = asd.session() if asd is not None else None
        handle.sessions += [s for s in (face_session, asd_session) if s is not None]

        def audio_since(t_from: float | None, t_to: float) -> bytes | None:
            if not ring:
                return None
            parts = [c.pcm16 for c in ring if (t_from is None or c.t > t_from) and c.t <= t_to]
            return b"".join(parts) if parts else None

        async def video_loop() -> None:
            period = 1.0 / max(0.5, hz)
            last_t: float | None = None
            first_t: float | None = None
            try:
                async for frame in video_src:
                    if first_t is None:
                        first_t = frame.t
                    handle.video_media_time = frame.t - first_t
                    handle.stats.frames += 1
                    handle.stats.frames_dropped += max(0, frame.dropped)
                    if last_t is not None and frame.t - last_t < period * 0.9:
                        handle.stats.frames_skipped += 1
                        continue
                    t0 = time.perf_counter()
                    tracks = await asyncio.to_thread(face_session.update, frame.image, frame.t)
                    handle.stats.faces_ms.append((time.perf_counter() - t0) * 1000)
                    if zones:
                        for tr in tracks:
                            if tr.seat is None:
                                tr.seat = seat_for(tr.bbox, zones)
                    if asd_session is not None:
                        audio = audio_since(last_t, frame.t)
                        tracks = asd_session.update(frame.image, frame.t, tracks, audio, SAMPLE_RATE)
                    last_t = frame.t
                    await publish_speakers(frame, tracks)
            finally:
                handle.video_done.set()

        async def publish_speakers(frame: VideoFrame, tracks: list) -> None:
            faces_data: list[FaceData] = [
                {"track_id": tr.track_id, "seat": tr.seat, "bbox": [round(float(v), 4) for v in tr.bbox],
                 "speaking": round(float(tr.speaking), 3)} for tr in tracks]
            data: SpeakersData = {
                "t": round(frame.t, 3), "faces": faces_data,
                "face_count": len([f for f in faces_data if f.get("seat") != "screen"]),
                "screen_talking": any(f.get("seat") == "screen" and f["speaking"] >= speaking_threshold
                                      for f in faces_data),
            }
            await bus.publish(Event(MT.SPEAKERS, dict(data), source=SOURCE_VIDEO, ts=round(frame.t, 3)))
            handle.stats.speakers_published += 1

        handle.tasks.append(asyncio.create_task(video_loop(), name="sensors.video"))

    log.info("sensors started: %s", handle.models)
    return handle


__all__ = ["start_sensors", "SensorHandle", "SensorStats", "SOURCE_AUDIO", "SOURCE_VIDEO"]
