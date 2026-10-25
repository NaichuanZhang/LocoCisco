"""Face detection + tracking + seat zones (unspoken.contracts.FaceTracker / FaceSession).

YuNetFaces  OpenCV FaceDetectorYN with the YuNet 2023mar ONNX (MIT, 232 KB, vendored). ~10-20 ms per
            640x360 frame on one CPU core. Detections are matched to existing tracks by IoU (greedy), so a
            face keeps its `track_id` ("f1", "f2", ...) while it stays in view; a track unseen for
            `max_lost_s` is forgotten. Boxes are EMA-smoothed. Each track gets the seat whose zone holds
            it (config/sensors.toml [seats]); the seat is sticky: a face that drifts out of every zone keeps
            its last seat. The zone named "screen" marks the room display, so a talking face there is media,
            not a person in the room.

No face recognition, no identity: the ids are per-session counters and are gone when the session ends.

Options: score_threshold=0.6, nms_threshold=0.3, iou_match=0.3, max_lost_s=1.0, smooth=0.6, model_path,
seats={...} (zones; otherwise read from config/sensors.toml), min_overlap=0.3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import FaceTrack, RoleSpec
from .config import Zone, load_sensors_config, merge_options, seat_for, seat_zones

MODELS_DIR = Path(__file__).resolve().parent / "models"
YUNET_ONNX = MODELS_DIR / "face_detection_yunet_2023mar.onnx"

BBox = tuple[float, float, float, float]


def iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: str
    bbox: BBox
    last_seen: float
    hits: int = 1
    seat: str | None = None
    score: float = 0.0
    landmarks: list[tuple[float, float]] = field(default_factory=list)


class IoUTracker:
    """Detections (normalised boxes) -> tracks with stable ids. Pure python, model-agnostic."""

    def __init__(self, *, iou_match: float = 0.3, max_lost_s: float = 1.0, smooth: float = 0.6,
                 zones: dict[str, Zone] | None = None, min_overlap: float = 0.3) -> None:
        self.iou_match = iou_match
        self.max_lost_s = max_lost_s
        self.smooth = smooth
        self.zones: dict[str, Zone] = dict(zones or {})
        self.min_overlap = min_overlap
        self.tracks: list[_Track] = []
        self._next = 1

    def update(self, detections: list[tuple[BBox, float, list[tuple[float, float]]]], t: float) -> list[FaceTrack]:
        # greedy IoU matching, best pairs first
        pairs: list[tuple[float, int, int]] = []
        for di, (box, _, _) in enumerate(detections):
            for ti, tr in enumerate(self.tracks):
                v = iou(box, tr.bbox)
                if v >= self.iou_match:
                    pairs.append((v, di, ti))
        pairs.sort(reverse=True)
        used_d: set[int] = set()
        used_t: set[int] = set()
        for _, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            used_d.add(di); used_t.add(ti)
            tr = self.tracks[ti]
            box, score, lm = detections[di]
            a = self.smooth
            tr.bbox = tuple(a * o + (1 - a) * n for o, n in zip(tr.bbox, box))  # type: ignore[assignment]
            tr.last_seen = t
            tr.hits += 1
            tr.score = score
            tr.landmarks = lm
        for di, (box, score, lm) in enumerate(detections):
            if di in used_d:
                continue
            self.tracks.append(_Track(f"f{self._next}", box, t, score=score, landmarks=lm))
            self._next += 1
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= self.max_lost_s]
        out: list[FaceTrack] = []
        for tr in self.tracks:
            if tr.last_seen != t:
                continue  # briefly lost: keep the track alive but do not report a face we do not see
            seat = seat_for(tr.bbox, self.zones, self.min_overlap)
            if seat is not None:
                tr.seat = seat
            out.append(FaceTrack(track_id=tr.track_id, bbox=tuple(round(v, 4) for v in tr.bbox),  # type: ignore[arg-type]
                                 seat=tr.seat, speaking=0.0))
        return out

    def reset(self) -> None:
        self.tracks.clear()
        self._next = 1


class YuNetFaces:
    def __init__(self, spec: RoleSpec | None = None, **options: Any) -> None:
        opts = merge_options(spec, options)
        self.model_path = Path(opts.get("model_path") or YUNET_ONNX)
        self.score_threshold = float(opts.get("score_threshold", 0.6))
        self.nms_threshold = float(opts.get("nms_threshold", 0.3))
        self.iou_match = float(opts.get("iou_match", 0.3))
        self.max_lost_s = float(opts.get("max_lost_s", 1.0))
        self.smooth = float(opts.get("smooth", 0.6))
        self.min_overlap = float(opts.get("min_overlap", 0.3))
        self.zones: dict[str, Zone] = seat_zones({"seats": opts["seats"]}) if "seats" in opts \
            else seat_zones(load_sensors_config(opts.get("sensors_config")))
        self.label = "YuNet 2023mar (OpenCV, CPU)"

    def configure_seats(self, zones: dict[str, Zone]) -> None:
        """The pipeline calls this with the zones from its own config (wins over the TOML read here)."""
        self.zones = dict(zones)

    def session(self) -> "YuNetSession":
        return YuNetSession(self)

    def healthy(self) -> bool:
        try:
            YuNetSession(self)._detector((64, 64))
            return True
        except Exception:
            return False


class YuNetSession:
    def __init__(self, faces: YuNetFaces) -> None:
        self.f = faces
        self.tracker = IoUTracker(iou_match=faces.iou_match, max_lost_s=faces.max_lost_s, smooth=faces.smooth,
                                  zones=faces.zones, min_overlap=faces.min_overlap)
        self._det = None
        self._size: tuple[int, int] | None = None
        self.last_ms = 0.0

    def _detector(self, size: tuple[int, int]):
        import cv2  # lazy

        if self._det is None:
            self._det = cv2.FaceDetectorYN.create(str(self.f.model_path), "", size, self.f.score_threshold,
                                                  self.f.nms_threshold, 5000)
            self._size = size
        if size != self._size:
            self._det.setInputSize(size)
            self._size = size
        return self._det

    def detect(self, frame: np.ndarray) -> list[tuple[BBox, float, list[tuple[float, float]]]]:
        """Normalised (x, y, w, h), score, 5 landmarks (right eye, left eye, nose, right mouth, left mouth)."""
        h, w = frame.shape[:2]
        det = self._detector((w, h))
        _, faces = det.detect(frame)
        out: list[tuple[BBox, float, list[tuple[float, float]]]] = []
        if faces is None:
            return out
        for row in faces:
            x, y, bw, bh = (float(v) for v in row[:4])
            if bw <= 0 or bh <= 0:
                continue
            box: BBox = (max(0.0, x / w), max(0.0, y / h), min(1.0, bw / w), min(1.0, bh / h))
            lm = [(float(row[4 + 2 * i]) / w, float(row[5 + 2 * i]) / h) for i in range(5)]
            out.append((box, float(row[14]), lm))
        return out

    def update(self, frame: np.ndarray, t: float) -> list[FaceTrack]:
        import time

        t0 = time.perf_counter()
        # tracker.zones follows the factory (configure_seats may run after the session was made)
        self.tracker.zones = self.f.zones
        tracks = self.tracker.update(self.detect(frame), t)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return tracks

    def reset(self) -> None:
        self.tracker.reset()


__all__ = ["YuNetFaces", "YuNetSession", "IoUTracker", "iou", "YUNET_ONNX"]
