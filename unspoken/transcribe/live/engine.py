"""The streaming per-person transcriber (the live counterpart of the offline asr/diar/asd/fuse stages).

    audio source -> VAD -> [open segment: PARTIAL ASR every ~1 s (latest wins)]
                       -> [closed segment: FINAL ASR || voice label] -> speaker decision -> caption (final)
    video source -> faces + ASD at the rate the detector sustains (frames that arrive while it is busy or
                    that are already stale are skipped, never queued) -> per-seat speaking evidence
    voice <-> seat binding: duration-weighted ASD evidence per voice (binding.py); re-evaluated after every
                    final; a changed binding re-emits earlier captions of that voice with the new speaker

Differences from the baseline live path (unspoken/sensors/pipeline.py + meeting/fusion.py): segments are cut
at pauses and capped at `max_segment_s` (8 s, the baseline's 20 s segments mix speakers), partial captions
appear while someone is still talking, several diarizer voices may merge into one on-camera person, and
speaker labels are revised when later evidence disagrees. Everything here is session-only: voices, faces and
PCM live in memory and are dropped when `run()` returns; nothing but captions.jsonl / faces.jsonl (times,
boxes, probabilities, text) is written.

Threads: the ASR provider owns its CTranslate2 threads (4 by default); the VAD, speaker model and face
detector run on one thread each in worker threads, so the whole engine fits the 6-thread budget of the dev box.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ...contracts import FaceTrack, SpeechSegment
from ...sensors.audio import SAMPLE_RATE, AudioChunk
from ...sensors.config import Zone, seat_for
from ...sensors.diar import UNKNOWN as DIAR_UNKNOWN
from ...sensors.diar import voice_name
from ..contracts import Transcript
from .binding import SeatBinder
from .captions import UNKNOWN_VOICE, Caption, CaptionLog, build_transcript

log = logging.getLogger("unspoken.transcribe.live")
Clock = Callable[[], float]

DEFAULT_SEATS: dict[str, Zone] = {"left": (0.0, 0.0, 0.5, 1.0), "right": (0.5, 0.0, 0.5, 1.0)}


@dataclass
class LiveSettings:
    speed: float = 1.0               # playback speed of file sources (1.0 = like a camera and a mic)
    realtime: bool = True            # False: as fast as the models go (tests, accuracy-only runs)
    partials: bool = True
    partial_every_s: float = 1.0     # one partial per open segment per this much NEW audio
    partial_min_s: float = 1.2       # no partial before the open segment holds this much audio
    partial_window_s: float = 5.0    # a partial transcribes at most the last N seconds of the open segment
    partial_max_rtf: float = 0.5     # no partials while the ASR runs slower than this (seconds per audio second):
                                     # on a loaded host finals come first, partials are the first thing to go
    asd_hz: float = 10.0             # ceiling for face detection + ASD; runs only when the detector is free
    stale_s: float = 0.25            # a frame older than this (behind the media clock) is skipped
    pad_s: float = 0.35              # ASD frames this close to a segment's edges count (meeting/fusion.PAD_S)
    ring_s: float = 2.0              # PCM kept for the ASD's audio gate
    seats: dict[str, Zone] = field(default_factory=lambda: dict(DEFAULT_SEATS))
    speaking_threshold: float = 0.5  # viewer: a face at or above this is drawn as talking
    frames_keep: int = 1200          # ASD frames kept for segment evidence (~2 min at 10 Hz)


@dataclass
class LiveStats:
    audio_chunks: int = 0
    segments: int = 0
    segments_empty: int = 0
    finals: int = 0
    partials: int = 0
    partials_skipped: int = 0        # superseded before the worker got to them
    partials_throttled: int = 0      # not requested because the ASR was running too slowly
    revisions: int = 0
    frames: int = 0                  # frames taken from the source
    frames_stale: int = 0            # skipped: already behind the media clock when we got them
    frames_detected: int = 0         # ran faces (+ASD)
    frames_dropped: int = 0          # dropped by the source before we saw them
    asr_ms: list[float] = field(default_factory=list)          # finals
    partial_asr_ms: list[float] = field(default_factory=list)
    diar_ms: list[float] = field(default_factory=list)
    faces_ms: list[float] = field(default_factory=list)
    asd_ms: list[float] = field(default_factory=list)
    errors: int = 0
    wall_s: float = 0.0
    cpu_s: float = 0.0
    audio_s: float = 0.0
    warm_s: float = 0.0
    max_rss_mb: float = 0.0
    asr_rtf: float = 0.0             # EMA of ASR seconds per audio second (finals and partials)

    def note_asr(self, ms: float, audio_s: float) -> None:
        if audio_s > 0:
            r = ms / 1000.0 / audio_s
            self.asr_rtf = r if self.asr_rtf == 0.0 else 0.7 * self.asr_rtf + 0.3 * r

    def summary(self) -> dict[str, Any]:
        def avg(xs: list[float]) -> float | None:
            return round(sum(xs) / len(xs), 1) if xs else None
        return {"audio_chunks": self.audio_chunks, "segments": self.segments, "segments_empty": self.segments_empty,
                "finals": self.finals, "partials": self.partials, "partials_skipped": self.partials_skipped,
                "partials_throttled": self.partials_throttled, "asr_rtf_last": round(self.asr_rtf, 3),
                "revisions": self.revisions, "frames": self.frames, "frames_stale": self.frames_stale,
                "frames_detected": self.frames_detected, "frames_dropped": self.frames_dropped,
                "asr_ms_avg": avg(self.asr_ms), "partial_asr_ms_avg": avg(self.partial_asr_ms),
                "diar_ms_avg": avg(self.diar_ms), "faces_ms_avg": avg(self.faces_ms), "asd_ms_avg": avg(self.asd_ms),
                "errors": self.errors, "wall_s": round(self.wall_s, 2), "cpu_s": round(self.cpu_s, 2),
                "audio_s": round(self.audio_s, 2), "warm_s": round(self.warm_s, 2),
                "max_rss_mb": round(self.max_rss_mb, 1)}


@dataclass
class _SegRecord:
    cap: Caption
    voice: str | None
    voice_conf: float
    seat_probs: dict[str, float]
    from_binding: bool


class LiveTranscriber:
    """Feed it an audio source (and a video source) via `run()`; captions come out through `on_caption` and
    the CaptionLog; `transcript()` builds the final Transcript afterwards."""

    def __init__(self, *, vad: Any, asr: Any, diar: Any = None, faces: Any = None, asd: Any = None,
                 settings: LiveSettings | None = None, clock: Clock | None = None,
                 log: CaptionLog | None = None, faces_log: str | Path | None = None,
                 on_caption: Callable[[Caption], None] | None = None,
                 on_frame: Callable[[Any, list[FaceTrack], float], None] | None = None,
                 binder: SeatBinder | None = None) -> None:
        self.vad, self.asr, self.diar, self.faces, self.asd = vad, asr, diar, faces, asd
        self.s = settings or LiveSettings()
        base = time.monotonic()
        self.clock: Clock = clock or (lambda: time.monotonic() - base)
        self.log = log or CaptionLog(None)
        self.on_caption = on_caption
        self.on_frame = on_frame
        self.binder = binder or SeatBinder()
        self.stats = LiveStats()
        self.models: dict[str, str] = {}
        self.audio_media_time = 0.0          # seconds of audio consumed
        self.video_media_time = 0.0
        self.audio_done = asyncio.Event()
        self.video_done = asyncio.Event()
        self._t_start = 0.0
        self._audio_end = 0.0                # media time of the last audio sample fed
        self._ring: deque[AudioChunk] = deque()
        self._frames: deque[tuple[float, dict[str, float]]] = deque(maxlen=self.s.frames_keep)
        self._seat_tracks: dict[str, Counter] = {}
        self._tracks: list[FaceTrack] = []
        self._finals: deque[tuple[str, SpeechSegment] | None] = deque()
        self._partial: tuple[str, float, float, bytes] | None = None
        self._wake = asyncio.Event()
        self._seg_ids: dict[float, str] = {}
        self._partial_mark: dict[str, float] = {}      # seg id -> audio end of its last requested partial
        self._partial_count: Counter = Counter()
        self._segments: dict[str, _SegRecord] = {}
        self._diar_session: Any = None
        self._sessions: list[Any] = []
        self._faces_fh = None
        if faces_log:
            Path(faces_log).parent.mkdir(parents=True, exist_ok=True)
            self._faces_fh = open(faces_log, "w", encoding="utf-8")
        for role, obj in (("vad", vad), ("asr", asr), ("diarization", diar), ("faces", faces), ("asd", asd)):
            self.models[role] = str(getattr(obj, "label", type(obj).__name__)) if obj is not None else "disabled"

    # ------------------------------------------------------------------ clock
    def now(self) -> float:
        """Media seconds now: wall time since start x speed when real time, else the audio position."""
        if self.s.realtime:
            return (self.clock() - self._t_start) * self.s.speed
        return self._audio_end

    # ------------------------------------------------------------------ lifecycle
    async def warm(self) -> None:
        """Load every model before the first segment so the first caption's latency is not a model load."""
        t0 = time.perf_counter()
        for obj in (self.asr, self.diar, self.vad, self.faces):
            if obj is None:
                continue
            w = getattr(obj, "warm", None)
            h = getattr(obj, "healthy", None)
            try:
                if callable(w):
                    r = w()
                    if asyncio.iscoroutine(r):
                        await r
                elif callable(h):
                    r = h()
                    if asyncio.iscoroutine(r):
                        await r
            except Exception:
                log.exception("warm-up of %s failed", type(obj).__name__)
        self.stats.warm_s = time.perf_counter() - t0

    async def run(self, audio_src: Any, video_src: Any = None) -> None:
        self._t_start = self.clock()
        cpu0, wall0 = time.process_time(), time.perf_counter()
        tasks = [asyncio.create_task(self._audio_loop(audio_src), name="live.audio"),
                 asyncio.create_task(self._worker(), name="live.asr")]
        if video_src is not None and self.faces is not None:
            tasks.append(asyncio.create_task(self._video_loop(video_src), name="live.video"))
        else:
            self.video_done.set()
        try:
            await asyncio.gather(*tasks)
        finally:
            for src in (audio_src, video_src):
                aclose = getattr(src, "aclose", None)
                if callable(aclose):
                    try:
                        await aclose()
                    except Exception:
                        pass
            for s in self._sessions:                 # voices and faces are session-only
                reset = getattr(s, "reset", None)
                if callable(reset):
                    try:
                        reset()
                    except Exception:
                        pass
            self._ring.clear()
            self.stats.wall_s = time.perf_counter() - wall0
            self.stats.cpu_s = time.process_time() - cpu0
            self.stats.audio_s = self.audio_media_time
            try:
                import resource
                self.stats.max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            except Exception:
                pass
            if self._faces_fh is not None:
                self._faces_fh.close()
                self._faces_fh = None
            self.log.close()

    # ------------------------------------------------------------------ audio
    async def _audio_loop(self, src: Any) -> None:
        stream = self.vad.stream(SAMPLE_RATE)
        t_first: float | None = None
        fed = 0
        try:
            async for chunk in src:
                if t_first is None:
                    t_first = chunk.t
                self.stats.audio_chunks += 1
                fed += len(chunk.pcm16)
                self.audio_media_time = fed / 2 / SAMPLE_RATE
                self._audio_end = chunk.t + chunk.duration
                self._ring.append(chunk)
                while self._ring and chunk.t - self._ring[0].t > self.s.ring_s:
                    self._ring.popleft()
                for seg in stream.feed(chunk.pcm16):
                    self._enqueue_final(seg, t_first)
                self._maybe_partial(stream, t_first)
            for seg in stream.flush():
                self._enqueue_final(seg, t_first or 0.0)
        finally:
            self._finals.append(None)
            self._wake.set()

    def _seg_id(self, start: float) -> str:
        key = round(start, 3)
        sid = self._seg_ids.get(key)
        if sid is None:
            sid = self._seg_ids[key] = uuid.uuid4().hex[:8]
        return sid

    def _enqueue_final(self, seg: SpeechSegment, t_first: float) -> None:
        seg.start += t_first
        seg.end += t_first
        self.stats.segments += 1
        sid = self._seg_id(seg.start)
        self._seg_ids.pop(round(seg.start, 3), None)
        if self._partial is not None and self._partial[0] == sid:
            self._partial = None                     # the final supersedes a pending partial
            self.stats.partials_skipped += 1
        self._finals.append((sid, seg))
        self._wake.set()

    @staticmethod
    def open_segment(stream: Any, t_first: float) -> tuple[float, float, bytes] | None:
        """(start, end, pcm) of the segment the VAD currently has open, read from the sensors segmenter;
        None for VADs that do not expose one (then no partials are produced)."""
        seg = getattr(stream, "_seg", None)
        if seg is None or not getattr(seg, "active", False):
            return None
        buf, start_w, window_s = getattr(seg, "buf", None), getattr(seg, "start_w", None), getattr(seg, "window_s", None)
        if not buf or start_w is None or window_s is None:
            return None
        start = start_w * window_s + t_first
        return start, start + len(buf) * window_s, b"".join(buf)

    def _maybe_partial(self, stream: Any, t_first: float) -> None:
        if not self.s.partials:
            return
        op = self.open_segment(stream, t_first)
        if op is None:
            return
        start, end, pcm = op
        if end - start < self.s.partial_min_s:
            return
        sid = self._seg_id(start)
        if end - self._partial_mark.get(sid, -math.inf) < self.s.partial_every_s:
            return
        self._partial_mark[sid] = end
        if self.s.realtime and self.stats.asr_rtf > self.s.partial_max_rtf:
            self.stats.partials_throttled += 1
            return
        window = int(self.s.partial_window_s * SAMPLE_RATE) * 2
        if len(pcm) > window:
            pcm = pcm[-window:]
        if self._partial is not None:
            self.stats.partials_skipped += 1
        self._partial = (sid, start, end, pcm)
        self._wake.set()

    # ------------------------------------------------------------------ ASR worker
    async def _worker(self) -> None:
        if self.diar is not None:
            self._diar_session = self.diar.session()
            self._sessions.append(self._diar_session)
            if self.binder.similarity is None:
                self.binder.similarity = self.voice_similarity
        try:
            while True:
                if self._finals:
                    item = self._finals.popleft()
                    if item is None:
                        return
                    try:
                        await self._do_final(*item)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        self.stats.errors += 1
                        log.exception("final segment failed")
                    continue
                if self._partial is not None:
                    p, self._partial = self._partial, None
                    try:
                        await self._do_partial(*p)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        self.stats.errors += 1
                        log.exception("partial failed")
                    continue
                self._wake.clear()
                if self._finals or self._partial is not None:
                    continue
                await self._wake.wait()
        finally:
            self.audio_done.set()

    async def _do_final(self, sid: str, seg: SpeechSegment) -> None:
        t0 = time.perf_counter()
        if self._diar_session is not None:
            tr, vl = await asyncio.gather(self.asr.transcribe(seg.pcm16, seg.sample_rate),
                                          self._diar_session.label(seg.pcm16, seg.sample_rate))
        else:
            tr, vl = await self.asr.transcribe(seg.pcm16, seg.sample_rate), None
        asr_ms = float(getattr(tr, "latency_ms", 0.0) or 0.0) or (time.perf_counter() - t0) * 1000
        self.stats.asr_ms.append(asr_ms)
        self.stats.note_asr(asr_ms, seg.end - seg.start)
        voice: str | None = None
        voice_conf = 0.0
        if vl is not None:
            self.stats.diar_ms.append(float(getattr(self._diar_session, "last_ms", 0.0) or 0.0))
            if vl.label and vl.label not in (DIAR_UNKNOWN, UNKNOWN_VOICE):
                voice, voice_conf = vl.label, float(vl.conf)
        seat_probs = self.seat_probs(seg.start, seg.end)
        self.binder.add(voice, seg.end - seg.start, seat_probs)
        changes = self.binder.recompute()
        dec = self.binder.decide(voice, voice_conf, seat_probs)
        text = (tr.text or "").strip()
        del seg.pcm16                                   # the PCM is not needed past this point
        if text:
            words = [{"start": round(seg.start + float(w.start), 3), "end": round(seg.start + float(w.end), 3),
                      "text": str(w.text), "conf": 1.0} for w in (getattr(tr, "words", None) or [])]
            now = self.now()
            cap = Caption(sid, round(now, 3), round(seg.start, 3), round(seg.end, 3), dec.speaker, text, True, 0,
                          voice, dec.conf, dec.source, words, round(now - seg.end, 3))
            self._segments[sid] = _SegRecord(cap, voice, voice_conf, seat_probs, dec.from_binding)
            self._emit(cap)
            self.stats.finals += 1
        else:
            self.stats.segments_empty += 1
        if changes:
            self._revise(changes, exclude=sid)

    def _revise(self, changes: dict[str, tuple[str | None, str | None]], exclude: str) -> None:
        for voice in changes:
            for sid, rec in self._segments.items():
                if sid == exclude or rec.voice != voice or not rec.from_binding:
                    continue
                dec = self.binder.decide(rec.voice, rec.voice_conf, rec.seat_probs)
                if dec.speaker == rec.cap.speaker:
                    continue
                cap = Caption(sid, round(self.now(), 3), rec.cap.start, rec.cap.end, dec.speaker, rec.cap.text, True,
                              rec.cap.revision + 1, rec.voice, dec.conf, dec.source, rec.cap.words, None)
                rec.cap, rec.from_binding = cap, dec.from_binding
                self.stats.revisions += 1
                self._emit(cap)

    async def _do_partial(self, sid: str, start: float, end: float, pcm: bytes) -> None:
        if any(f is not None and f[0] == sid for f in self._finals):
            self.stats.partials_skipped += 1
            return
        t0 = time.perf_counter()
        tr = await self.asr.transcribe(pcm, SAMPLE_RATE)
        ms = float(getattr(tr, "latency_ms", 0.0) or 0.0) or (time.perf_counter() - t0) * 1000
        self.stats.partial_asr_ms.append(ms)
        self.stats.note_asr(ms, len(pcm) / 2 / SAMPLE_RATE)
        text = (tr.text or "").strip()
        if not text:
            return
        voice = await asyncio.to_thread(self.peek_voice, pcm) if self._diar_session is not None else None
        dec = self.binder.decide(voice, 0.5, self.seat_probs(start, end))
        n = self._partial_count[sid]
        self._partial_count[sid] += 1
        self._emit(Caption(sid, round(self.now(), 3), round(start, 3), round(end, 3), dec.speaker, text, False, n,
                           voice, dec.conf, dec.source, [], None))
        self.stats.partials += 1

    def voice_similarity(self, a: str, b: str) -> float | None:
        """Cosine between two voices' running centroids (sensors.diar sessions); None when unavailable."""
        cents = getattr(self._diar_session, "centroids", None)
        if not cents:
            return None
        try:
            ia, ib = [voice_name(i) for i in range(len(cents))].index(a), [voice_name(i) for i in range(len(cents))].index(b)
        except ValueError:
            return None
        ca, cb = cents[ia], cents[ib]
        return float(np.dot(ca, cb) / ((np.linalg.norm(ca) * np.linalg.norm(cb)) + 1e-9))

    def peek_voice(self, pcm: bytes) -> str | None:
        """Nearest known voice for a partial WITHOUT touching the clustering (sensors.diar sessions only)."""
        s = self._diar_session
        d = getattr(s, "d", None)
        cents = getattr(s, "centroids", None)
        if d is None or not cents or not hasattr(d, "embed"):
            return None
        if len(pcm) / 2 / SAMPLE_RATE < float(getattr(d, "min_seconds", 0.6)):
            return None
        emb = d.embed(pcm, SAMPLE_RATE)
        if getattr(emb, "size", 0) == 0:
            return None
        sims = [float(np.dot(emb, c) / (np.linalg.norm(c) + 1e-9)) for c in cents]
        best = int(np.argmax(sims))
        return voice_name(best) if sims[best] >= float(getattr(d, "threshold", 0.55)) else None

    def _emit(self, cap: Caption) -> None:
        self.log.append(cap)
        if self.on_caption is not None:
            try:
                self.on_caption(cap)
            except Exception:
                log.exception("on_caption failed")

    # ------------------------------------------------------------------ video
    def _audio_since(self, t_from: float | None, t_to: float) -> bytes | None:
        parts = [c.pcm16 for c in self._ring if (t_from is None or c.t > t_from) and c.t <= t_to]
        return b"".join(parts) if parts else None

    async def _video_loop(self, src: Any) -> None:
        zones = dict(self.s.seats or {})
        if zones and hasattr(self.faces, "configure_seats"):
            self.faces.configure_seats(zones)
        face_session = self.faces.session()
        asd_session = self.asd.session() if self.asd is not None else None
        self._sessions += [s for s in (face_session, asd_session) if s is not None]
        period = 1.0 / max(0.5, self.s.asd_hz)
        last_det: float | None = None
        first_t: float | None = None
        try:
            async for frame in src:
                if first_t is None:
                    first_t = frame.t
                self.video_media_time = frame.t - first_t
                self.stats.frames += 1
                self.stats.frames_dropped += max(0, int(getattr(frame, "dropped", 0)))
                if self.s.realtime and frame.t < self.now() - self.s.stale_s:
                    self.stats.frames_stale += 1
                    continue
                tracks = self._tracks
                if last_det is None or frame.t - last_det >= period * 0.9:
                    t0 = time.perf_counter()
                    tracks = await asyncio.to_thread(face_session.update, frame.image, frame.t)
                    self.stats.faces_ms.append((time.perf_counter() - t0) * 1000)
                    if zones:
                        for tr in tracks:
                            if tr.seat is None:
                                tr.seat = seat_for(tr.bbox, zones)
                    if asd_session is not None:
                        t1 = time.perf_counter()
                        tracks = asd_session.update(frame.image, frame.t, tracks, self._audio_since(last_det, frame.t),
                                                    SAMPLE_RATE)
                        self.stats.asd_ms.append((time.perf_counter() - t1) * 1000)
                    last_det = frame.t
                    self._record_frame(frame.t, tracks)
                    self.stats.frames_detected += 1
                self._tracks = tracks
                if self.on_frame is not None:
                    try:
                        self.on_frame(frame, tracks, self.now())
                    except Exception:
                        log.exception("on_frame failed")
        finally:
            self.video_done.set()

    def _record_frame(self, t: float, tracks: list[FaceTrack]) -> None:
        seats: dict[str, float] = {}
        for tr in tracks:
            if tr.seat and tr.seat != "screen":
                seats[tr.seat] = max(seats.get(tr.seat, 0.0), float(tr.speaking))
                self._seat_tracks.setdefault(tr.seat, Counter())[tr.track_id] += 1
        self._frames.append((t, seats))
        if self._faces_fh is not None:
            self._faces_fh.write(json.dumps({"t": round(t, 3), "faces": [
                {"track_id": tr.track_id, "seat": tr.seat, "bbox": [round(float(v), 4) for v in tr.bbox],
                 "speaking": round(float(tr.speaking), 3)} for tr in tracks]}) + "\n")

    def seat_probs(self, start: float, end: float) -> dict[str, float]:
        """Mean speaking probability per seat over the ASD frames inside [start - pad, end + pad]."""
        window = [seats for t, seats in self._frames if start - self.s.pad_s <= t <= end + self.s.pad_s]
        if not window:
            return {}
        totals: dict[str, float] = {}
        for seats in window:
            for seat, p in seats.items():
                totals[seat] = totals.get(seat, 0.0) + p
        return {s: v / len(window) for s, v in totals.items()}

    # ------------------------------------------------------------------ outputs
    def tracks_by_seat(self) -> dict[str, str]:
        return {seat: c.most_common(1)[0][0] for seat, c in self._seat_tracks.items() if c}

    def transcript(self, session_id: str, duration: float | None = None, meta: dict[str, Any] | None = None) -> Transcript:
        m = {"method": "live: VAD-cut segments (<= max_segment_s), partial + final ASR, online voice labels, "
                       "ASD evidence per segment, revisable voice->seat binding",
             "models": dict(self.models), "stats": self.stats.summary(),
             "bindings": dict(self.binder.bound), "settings": {k: v for k, v in self.s.__dict__.items() if k != "seats"},
             "seats": {k: list(v) for k, v in (self.s.seats or {}).items()}}
        m.update(meta or {})
        return build_transcript(self.log.captions, session_id, duration if duration is not None else self.audio_media_time,
                                m, self.tracks_by_seat())


def thread_budget() -> dict[str, Any]:
    """What this process was allowed to use (for the report's 'threads' column)."""
    return {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "cpu_count": os.cpu_count()}


__all__ = ["LiveTranscriber", "LiveSettings", "LiveStats", "DEFAULT_SEATS", "thread_budget"]
