"""Who is talking on camera: video (+ its audio track) -> asd.json (unspoken.transcribe.contracts.AsdResult).

    python -m unspoken.transcribe.asd --in CASE/video.mp4 --out asd.json --method lightasd [--fps 25]

Every method shares one front end (faces.py): frames are decoded in order at the video's own rate (or
subsampled to --fps), YuNet finds faces, an IoU tracker gives them stable ids ("f1", "f2", ...) and a seat
from fixed zones (left / right halves of the frame, or [seats] of a sensors.toml), and lost tracks coast for
up to --coast seconds so a model sees unbroken crop sequences. Methods:

  motion     MotionASD from the live pipeline (unspoken/sensors/asd.py) at full frame rate: mouth-region motion
             in excess of the rest of the face x an audio-energy gate. No model. The "today" reference point.
  avcorr     cheap lip-sync proxy: per face, the correlation over a sliding window between mouth motion and the
             audio energy envelope, gated by audio energy. No model.
  lightasd   Light-ASD (CVPR 2023, 1.03 M params, MIT) pretrained on AVA-ActiveSpeaker: 112x112 grayscale
             face crops at 25 fps + 13-dim MFCC at 100 Hz -> P(speaking) per frame; logits averaged over
             clip durations {1..6} s as the authors do (asd_lightasd.py). Needs torch and the fetched weights.
  lightasd-talkset   the same network with the authors' TalkSet fine-tuned weights (research-licence data).

Scores are written for every processed frame for every face visible (detected or coasted) at that time. The
audio comes from the video's own audio track (--audio overrides, e.g. the case's audio.wav, same signal).
Nothing is written to disk except the JSON. Threads: --threads (default 6) caps OpenCV and torch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import FaceTrack
from .contracts import AsdFrame, AsdResult, FaceInfo, to_json
from .faces import BBox, Detector, FaceTracks, TrackBox, crop_square, iter_frames, video_info, zones_from_config

SR = 16000
METHODS = ("motion", "avcorr", "lightasd", "lightasd-talkset")


# ----------------------------------------------------------------------------------------------- audio
def decode_audio(video: str | Path, sample_rate: int = SR) -> np.ndarray:
    """The video's audio track as int16 mono at `sample_rate` (PyAV); empty array when there is none."""
    import av

    out: list[np.ndarray] = []
    with av.open(str(video)) as c:
        if not c.streams.audio:
            return np.zeros(0, dtype=np.int16)
        stream = c.streams.audio[0]
        rs = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        for frame in c.decode(stream):
            for f in rs.resample(frame):
                out.append(f.to_ndarray().reshape(-1))
        for f in rs.resample(None):
            out.append(f.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.int16) if out else np.zeros(0, dtype=np.int16)


def read_wav(path: str | Path, sample_rate: int = SR) -> np.ndarray:
    import wave

    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16-bit PCM")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels())[:, 0]
        if w.getframerate() != sample_rate:
            from ..sensors.audio import resample_pcm16
            x = np.frombuffer(resample_pcm16(x.tobytes(), w.getframerate(), sample_rate), dtype="<i2")
    return np.ascontiguousarray(x)


def energy_db(x: np.ndarray) -> float:
    if len(x) == 0:
        return -100.0
    f = x.astype(np.float32) / 32768.0
    return float(20 * np.log10(np.sqrt(np.mean(f * f)) + 1e-9))


# ----------------------------------------------------------------------------------------------- methods
class Method:
    label = "?"

    def on_frame(self, k: int, t: float, bgr: np.ndarray, gray: np.ndarray, boxes: list[TrackBox],
                 audio: np.ndarray, alive: set[str]) -> None: ...

    def finish(self) -> dict[str, dict[int, float]]:
        """track_id -> {frame k -> P(speaking)}"""
        return {}


class MotionMethod(Method):
    def __init__(self, *, motion_gain: float = 6.0, smooth: float = 0.5) -> None:
        from ..sensors.asd import MotionASD

        self.session = MotionASD(motion_gain=motion_gain, smooth=smooth).session()
        self.label = f"MotionASD (mouth motion x voice energy, no model; gain {motion_gain:g}) at full frame rate"
        self.scores: dict[str, dict[int, float]] = defaultdict(dict)

    def on_frame(self, k, t, bgr, gray, boxes, audio, alive):
        faces = [FaceTrack(b.track_id, b.box, b.seat, 0.0) for b in boxes]
        for f in self.session.update(gray, t, faces, audio.tobytes() if len(audio) else None):
            self.scores[f.track_id][k] = float(f.speaking)

    def finish(self):
        return dict(self.scores)


class AVCorrMethod(Method):
    """Mouth-motion / audio-energy correlation in a sliding window (`window_s`), gated by audio energy above an
    adaptive floor. Motion is MotionASD's mouth-minus-upper-face difference per frame."""

    def __init__(self, *, window_s: float = 2.0, floor_db: float = -50.0, snr_db: float = 10.0) -> None:
        from ..sensors.asd import mouth_box, upper_box, _crop

        self._crop, self._mouth, self._upper = _crop, mouth_box, upper_box
        self.window_s, self.floor_db, self.snr_db = window_s, floor_db, snr_db
        self.label = f"AV-corr (mouth motion x audio energy correlation, {window_s:g} s window, no model)"
        self.prev: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.motion: dict[str, dict[int, float]] = defaultdict(dict)
        self.energy: dict[int, float] = {}
        self.times: dict[int, float] = {}

    def on_frame(self, k, t, bgr, gray, boxes, audio, alive):
        self.energy[k] = energy_db(audio)
        self.times[k] = t
        g = gray.astype(np.float32)
        for b in boxes:
            m = self._crop(g, self._mouth(b.box), (32, 20))
            u = self._crop(g, self._upper(b.box), (32, 20))
            prev = self.prev.get(b.track_id)
            if m is not None and u is not None:
                self.prev[b.track_id] = (m, u)
                if prev is not None and prev[0].shape == m.shape:
                    self.motion[b.track_id][k] = max(0.0, float(np.mean(np.abs(m - prev[0])) - np.mean(np.abs(u - prev[1])))) / 255.0
        for tid in list(self.prev):
            if tid not in alive:
                self.prev.pop(tid, None)

    def finish(self):
        ks = sorted(self.energy)
        if not ks:
            return {}
        e = np.array([self.energy[k] for k in ks], dtype=np.float32)
        fps = 1.0 / max(1e-6, (self.times[ks[-1]] - self.times[ks[0]]) / max(1, len(ks) - 1))
        floor = max(float(np.percentile(e, 10)), self.floor_db)      # one noise floor for the file
        gate = np.clip((e - floor) / self.snr_db, 0.0, 1.0)
        w = max(5, int(round(self.window_s * fps)))
        out: dict[str, dict[int, float]] = {}
        pos = {k: i for i, k in enumerate(ks)}
        for tid, mk in self.motion.items():
            m = np.zeros(len(ks), dtype=np.float32)
            have = np.zeros(len(ks), dtype=bool)
            for k, v in mk.items():
                m[pos[k]] = v
                have[pos[k]] = True
            p = np.zeros(len(ks), dtype=np.float32)
            for i in np.flatnonzero(have):
                a, b = max(0, i - w // 2), min(len(ks), i + w // 2 + 1)
                mm, ee = m[a:b], e[a:b]
                if mm.std() < 1e-6 or ee.std() < 1e-6:
                    continue
                c = float(np.corrcoef(mm, ee)[0, 1])
                p[i] = max(0.0, c) * gate[i]
            out[tid] = {k: round(float(p[pos[k]]), 3) for k in mk}
        return out


class LightASDMethod(Method):
    """Crops every visible face (detected or coasted) to 112x112, keeps them per track on a 25 fps grid and
    scores each track in chunks of `chunk_s` seconds (or when the track ends) against the MFCC of the whole
    audio. Coasted crops at the END of a track (the face never came back) are discarded. A video faster than
    25 fps maps several frames to one grid tick (one crop scored, the score shared); a slower one repeats crops."""

    def __init__(self, scorer: Any, mfcc: np.ndarray, *, chunk_s: float = 60.0, grid_fps: int = 25) -> None:
        self.scorer = scorer
        self.mfcc = mfcc
        self.grid_fps = grid_fps
        self.chunk = int(chunk_s * grid_fps)
        self.label = scorer.label
        self.buf: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)       # tid -> [(grid g, crop)]
        self.pending: dict[str, list[tuple[int, int, np.ndarray]]] = defaultdict(list)  # coasted, unconfirmed
        self.frame_grid: dict[str, dict[int, int]] = defaultdict(dict)              # tid -> {frame k: grid g}
        self.score_g: dict[str, dict[int, float]] = defaultdict(dict)               # tid -> {grid g: p}
        self.last_g: dict[str, int] = {}
        self.model_s = 0.0

    def _flush(self, tid: str) -> None:
        items = self.buf.pop(tid, [])
        if not items:
            return
        t0 = time.perf_counter()
        g0, g1 = items[0][0], items[-1][0] + 1
        crops = np.stack([c for _, c in items])
        mf = self.mfcc[g0 * 4:g1 * 4] if g0 * 4 < len(self.mfcc) else np.zeros((0, 13), dtype=np.float32)
        for (g, _), v in zip(items, self.scorer.score(crops, mf)):
            self.score_g[tid][g] = float(v)
        self.model_s += time.perf_counter() - t0

    def _push(self, tid: str, g: int, k: int, crop: np.ndarray) -> None:
        self.frame_grid[tid][k] = g
        last = self.last_g.get(tid)
        if last is not None and g <= last:
            return                                                # same grid tick as the previous frame
        if last is not None:
            while g > last + 1:                                   # video slower than 25 fps: repeat the crop
                last += 1
                self.buf[tid].append((last, crop))
        self.buf[tid].append((g, crop))
        self.last_g[tid] = g
        if len(self.buf[tid]) >= self.chunk:
            self._flush(tid)

    def on_frame(self, k, t, bgr, gray, boxes, audio, alive):
        g = int(round(t * self.grid_fps))
        for b in boxes:
            crop = crop_square(gray, b.box)
            if b.detected:
                for item in self.pending.pop(b.track_id, []):
                    self._push(b.track_id, *item)
                self._push(b.track_id, g, k, crop)
            else:
                self.pending[b.track_id].append((g, k, crop))
        for tid in list(self.last_g):
            if tid not in alive:
                self.pending.pop(tid, None)
                self._flush(tid)
                self.last_g.pop(tid, None)

    def finish(self):
        self.pending.clear()
        for tid in list(self.buf):
            self._flush(tid)
        return {tid: {k: round(self.score_g[tid].get(g, 0.0), 3) for k, g in fg.items()}
                for tid, fg in self.frame_grid.items()}


# ----------------------------------------------------------------------------------------------- driver
@dataclass
class Settings:
    method: str = "lightasd"
    fps: float | None = None            # processing / output rate; None = the video's own
    scale: float = 1.0                  # detector upscale
    seats: str | None = None            # sensors.toml with [seats]; None = left/right halves
    threads: int = 6
    device: str = "cpu"
    weights: str | None = None
    durations: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    audio: str | None = None
    motion_gain: float = 6.0
    max_lost_s: float = 3.0
    coast_s: float = 1.0
    min_track_s: float = 0.5
    chunk_s: float = 60.0
    score_threshold: float = 0.6
    extra: dict[str, Any] = field(default_factory=dict)


def _load1() -> float:
    try:
        return os.getloadavg()[0]
    except OSError:
        return float("nan")


def make_method(s: Settings, audio: np.ndarray, *, scorer: Any = None) -> Method:
    if s.method == "motion":
        return MotionMethod(motion_gain=s.motion_gain)
    if s.method == "avcorr":
        return AVCorrMethod()
    if s.method in ("lightasd", "lightasd-talkset"):
        from .asd_lightasd import DEFAULT_WEIGHTS, LightASDScorer, mfcc_features

        if scorer is None:
            w = s.weights or (DEFAULT_WEIGHTS if s.method == "lightasd" else "lightasd_finetuning_TalkSet.model")
            scorer = LightASDScorer(w, device=s.device, threads=s.threads, durations=s.durations)
        return LightASDMethod(scorer, mfcc_features(audio) if len(audio) else np.zeros((0, 13), dtype=np.float32),
                              chunk_s=s.chunk_s)
    raise ValueError(f"unknown method {s.method!r}; known: {', '.join(METHODS)}")


def run_asd(video: str | Path, s: Settings, *, detector: Detector | None = None, scorer: Any = None,
            audio: np.ndarray | None = None, log=None) -> AsdResult:
    import cv2

    cv2.setNumThreads(int(s.threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(s.threads))
    t_start, load0 = time.perf_counter(), _load1()
    info = video_info(video)
    native = float(info["fps"])
    every = max(1, int(round(native / s.fps))) if s.fps else 1
    out_fps = native / every
    if audio is None:
        audio = read_wav(s.audio) if s.audio else decode_audio(video)
    if detector is None:
        from .faces import YuNetDetector
        detector = YuNetDetector(scale=s.scale, score_threshold=s.score_threshold, threads=s.threads)
    tracks = FaceTracks(detector, zones=zones_from_config(s.seats), max_lost_s=s.max_lost_s, coast_s=s.coast_s)
    method = make_method(s, audio, scorer=scorer)
    hop = SR / out_fps
    times: dict[int, float] = {}
    visible: dict[int, list[TrackBox]] = {}
    seat_votes: dict[str, Counter] = defaultdict(Counter)
    boxes_px: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    W, H = info["width"], info["height"]
    n = 0
    t_detect = 0.0
    coasting: dict[str, list[int]] = defaultdict(list)     # tid -> frames coasted since its last detection
    dead: set[tuple[str, int]] = set()                      # coasted (tid, k) whose face never came back
    for k, (idx, t, bgr) in enumerate(iter_frames(video, every=every)):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        boxes = tracks.update(bgr, t)
        t_detect += tracks.last_detect_ms / 1000
        a0 = int(round(k * hop))
        a1 = int(round((k + 1) * hop))
        alive = tracks.alive()
        method.on_frame(k, t, bgr, gray, boxes, audio[a0:a1], alive)
        times[k] = t
        visible[k] = boxes
        for b in boxes:
            if b.detected:
                coasting.pop(b.track_id, None)
                seat_votes[b.track_id][b.seat] += 1
                boxes_px[b.track_id].append((b.box[0] * W, b.box[1] * H, b.box[2] * W, b.box[3] * H))
            else:
                coasting[b.track_id].append(k)
        for tid in list(coasting):
            if tid not in alive:
                dead.update((tid, kk) for kk in coasting.pop(tid))
        n += 1
        if log and k % (int(out_fps) * 60) == 0:
            log(f"  t={t:7.1f}s faces={len(boxes)} load={_load1():.1f}")
    for tid, ks in coasting.items():
        dead.update((tid, kk) for kk in ks)
    scores = method.finish()
    keep = {tid for tid, bx in boxes_px.items() if len(bx) >= s.min_track_s * out_fps}
    faces = []
    for tid in sorted(keep, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
        bx = np.array(boxes_px[tid])
        faces.append(FaceInfo(tid, seat_votes[tid].most_common(1)[0][0], tuple(round(float(v), 1) for v in bx.mean(0)),
                              len(bx)))
    frames = []
    for k in range(n):
        sc = {}
        for b in visible[k]:
            if b.track_id in keep and (b.track_id, k) not in dead:
                sc[b.track_id] = round(float(scores.get(b.track_id, {}).get(k, 0.0)), 3)
        frames.append(AsdFrame(round(times[k], 3), sc))
    wall = time.perf_counter() - t_start
    dur = n / out_fps
    timing = {"wall_s": round(wall, 2), "audio_s": round(dur, 2), "rtf": round(wall / max(dur, 1e-6), 4),
              "threads": s.threads, "load1_start": round(load0, 2), "load1_end": round(_load1(), 2),
              "detect_s": round(t_detect, 2), "model_s": round(getattr(method, "model_s", 0.0), 2), "frames": n,
              "video_fps": native, "every": every}
    label = f"{detector.label} + {method.label}; boxes in px of {W}x{H}"
    return AsdResult(label, round(out_fps, 4), faces, frames, timing)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--method", default="lightasd", choices=METHODS)
    ap.add_argument("--fps", type=float, default=None, help="process/output rate (default: the video's own)")
    ap.add_argument("--scale", type=float, default=1.0, help="detector upscale factor (1.0 = native)")
    ap.add_argument("--seats", type=Path, default=None, help="sensors.toml with [seats] zones (default: halves)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--device", default="cpu", help="torch device for lightasd (cpu / cuda)")
    ap.add_argument("--weights", default=None, help="Light-ASD weight file (name in the models dir or a path)")
    ap.add_argument("--durations", default="1,2,3,4,5,6", help="Light-ASD clip durations to average, seconds")
    ap.add_argument("--audio", type=Path, default=None, help="WAV to use instead of the video's audio track")
    ap.add_argument("--motion-gain", type=float, default=6.0)
    ap.add_argument("--max-lost", type=float, default=3.0, help="s a lost track keeps its id")
    ap.add_argument("--coast", type=float, default=1.0, help="s a lost track is still scored at its last box")
    ap.add_argument("--min-track", type=float, default=0.5, help="drop faces seen less than this many seconds")
    ap.add_argument("--chunk", type=float, default=60.0, help="Light-ASD scoring chunk, seconds")
    ap.add_argument("--det-threshold", type=float, default=0.6)
    ap.add_argument("--quiet", action="store_true")
    return ap


def settings_from_args(a: argparse.Namespace) -> Settings:
    return Settings(method=a.method, fps=a.fps, scale=a.scale, seats=str(a.seats) if a.seats else None,
                    threads=a.threads, device=a.device, weights=a.weights,
                    durations=tuple(int(x) for x in str(a.durations).split(",") if x.strip()),
                    audio=str(a.audio) if a.audio else None, motion_gain=a.motion_gain, max_lost_s=a.max_lost,
                    coast_s=a.coast, min_track_s=a.min_track, chunk_s=a.chunk, score_threshold=a.det_threshold)


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    s = settings_from_args(a)
    log = None if a.quiet else (lambda m: print(m, file=sys.stderr, flush=True))
    if log:
        log(f"asd {s.method} on {a.video} (threads {s.threads}, load {_load1():.1f})")
    res = run_asd(a.video, s, log=log)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    to_json(res, a.out)
    if log:
        log(f"wrote {a.out}: {len(res.faces)} faces, {len(res.frames)} frames @ {res.fps:g} fps, "
            f"{json.dumps(res.timing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
