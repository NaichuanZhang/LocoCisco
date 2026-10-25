"""Active speaker detection: which face is talking (unspoken.contracts.ActiveSpeaker / ASDSession).

MotionASD  Heuristic, no model: for every tracked face, the mouth region (lower 40 % of the box, central
           60 %) is compared with the previous frame of the same track; mouth motion in excess of the rest
           of the face (so a nodding head does not count) is smoothed per track and multiplied by an audio
           gate (RMS of the PCM since the last call against an adaptive noise floor; when the pipeline passes
           no audio the gate is 0.5 = unknown). With exactly one face in the room and voice present, that face
           gets at least `single_face_floor`. It is cheap (<1 ms per face) and good enough to pick the
           talker among seated people who do not all move at once.

Light-ASD  (https://github.com/Junhua-Liao/Light-ASD, MIT): a learned audio-visual model that is markedly
           more robust. Not wired in this iteration (torch + its MFCC front end did not fit the time box);
           docs/SENSORS.md lists the exact steps. The pipeline only needs an ActiveSpeaker factory, so it is
           a config change once `unspoken.sensors.asd:LightASD` exists.

Options: motion_gain=6.0 (mouth-minus-face motion that counts as 1.0), smooth=0.5 (EMA on the per-track
score), floor_db=-50, snr_db=10, single_face_floor=0.6, patch=(32, 20) (mouth patch size in px).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import FaceTrack, RoleSpec
from .config import merge_options
from .audio import pcm16_to_float


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float32)
    # BGR -> luma without importing cv2 (works on any HxWx3 uint8 array)
    return (0.114 * frame[..., 0] + 0.587 * frame[..., 1] + 0.299 * frame[..., 2]).astype(np.float32)


def _crop(gray: np.ndarray, box: tuple[float, float, float, float], size: tuple[int, int]) -> np.ndarray | None:
    h, w = gray.shape[:2]
    x0, y0 = int(round(box[0] * w)), int(round(box[1] * h))
    x1, y1 = int(round((box[0] + box[2]) * w)), int(round((box[1] + box[3]) * h))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    patch = gray[y0:y1, x0:x1]
    # resize by sampling (no cv2 dependency, tiny patches)
    ys = np.linspace(0, patch.shape[0] - 1, size[1]).astype(int)
    xs = np.linspace(0, patch.shape[1] - 1, size[0]).astype(int)
    return patch[np.ix_(ys, xs)]


def mouth_box(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return (x + 0.2 * w, y + 0.6 * h, 0.6 * w, 0.4 * h)


def upper_box(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return (x + 0.2 * w, y + 0.1 * h, 0.6 * w, 0.4 * h)


class MotionASD:
    def __init__(self, spec: RoleSpec | None = None, **options: Any) -> None:
        opts = merge_options(spec, options)
        self.motion_gain = float(opts.get("motion_gain", 6.0))
        self.smooth = float(opts.get("smooth", 0.5))
        self.floor_db = float(opts.get("floor_db", -50.0))
        self.snr_db = float(opts.get("snr_db", 10.0))
        self.single_face_floor = float(opts.get("single_face_floor", 0.6))
        patch = opts.get("patch", (32, 20))
        self.patch = (int(patch[0]), int(patch[1]))
        self.label = "MotionASD (mouth motion x voice energy, no model)"

    def session(self) -> "MotionASDSession":
        return MotionASDSession(self)

    def healthy(self) -> bool:
        return True


class MotionASDSession:
    def __init__(self, asd: MotionASD) -> None:
        self.a = asd
        self._prev: dict[str, tuple[np.ndarray, np.ndarray]] = {}   # track_id -> (mouth, upper) patches
        self._score: dict[str, float] = {}
        self._floor_db = asd.floor_db
        self.voice = 0.0            # last audio gate (0..1)

    # ------------------------------------------------------------------ audio gate
    def audio_gate(self, audio: bytes | None) -> float:
        if audio is None or len(audio) < 320:
            return 0.5
        x = pcm16_to_float(audio)
        level = 20 * np.log10(float(np.sqrt(np.mean(x * x)) + 1e-9))
        if level < self._floor_db:
            self._floor_db = level
        else:
            self._floor_db += 0.002 * (level - self._floor_db)
        gate = float(np.clip((level - self._floor_db) / self.a.snr_db, 0.0, 1.0))
        self.voice = gate
        return gate

    # ------------------------------------------------------------------ ASDSession protocol
    def update(self, frame: Any, t: float, faces: list[FaceTrack], audio: bytes | None,
               sample_rate: int = 16000) -> list[FaceTrack]:
        gate = self.audio_gate(audio)
        gray = _gray(np.asarray(frame)) if frame is not None else None
        seen: set[str] = set()
        motions: dict[str, float] = {}
        for face in faces:
            seen.add(face.track_id)
            if gray is None:
                continue
            mouth = _crop(gray, mouth_box(face.bbox), self.a.patch)
            upper = _crop(gray, upper_box(face.bbox), self.a.patch)
            if mouth is None or upper is None:
                continue
            prev = self._prev.get(face.track_id)
            self._prev[face.track_id] = (mouth, upper)
            if prev is None or prev[0].shape != mouth.shape or prev[1].shape != upper.shape:
                continue
            m = float(np.mean(np.abs(mouth - prev[0]))) / 255.0
            u = float(np.mean(np.abs(upper - prev[1]))) / 255.0
            raw = float(np.clip((m - u) * self.a.motion_gain, 0.0, 1.0))
            s = self.a.smooth
            motions[face.track_id] = s * self._score.get(face.track_id, 0.0) + (1 - s) * raw
            self._score[face.track_id] = motions[face.track_id]
        for tid in list(self._prev):
            if tid not in seen:
                self._prev.pop(tid, None)
                self._score.pop(tid, None)
        people = [f for f in faces if f.seat != "screen"]
        out: list[FaceTrack] = []
        for face in faces:
            p = motions.get(face.track_id, 0.0) * gate
            if len(people) == 1 and face is people[0] and gate > 0.5:
                p = max(p, self.a.single_face_floor * gate)
            out.append(FaceTrack(track_id=face.track_id, bbox=face.bbox, seat=face.seat, speaking=round(p, 3)))
        return out

    def reset(self) -> None:
        self._prev.clear()
        self._score.clear()
        self._floor_db = self.a.floor_db


__all__ = ["MotionASD", "MotionASDSession", "mouth_box", "upper_box"]
