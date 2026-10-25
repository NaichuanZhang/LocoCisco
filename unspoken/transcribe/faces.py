"""Face tracks for the per-person transcriber: detector + tracker + seat, frame by frame, offline or live.

Detector   YuNet 2023mar (OpenCV FaceDetectorYN, MIT, vendored in unspoken/sensors/models) on the frame at its
           native size, optionally upscaled by `scale` (INTER_CUBIC) for very small faces. Measured on the AMI
           dev excerpts (tests never run this): at 1.0x YuNet finds both ~24 px faces of the 352x288 room view
           in >99 % of frames (4 ms/frame, 6 threads) and both ~64 px faces of the 704x288 duo view in 100 %
           (7 ms/frame); 2x upscaling only costs time. The live pipeline's low face coverage on the room view
           came from stretching 352x288 to 704x288 (config), not from the detector.
Tracker    unspoken.sensors.faces.IoUTracker (greedy IoU match, EMA-smoothed boxes, sticky seat from the seat
           zones). Offline the tracker is allowed to keep a lost track alive for `max_lost_s` (people stay in
           their chairs) and this module COASTS it: while a track is unseen for at most `coast_s` its last box
           is still reported with detected=False, so a model sees an unbroken crop sequence. Coasted frames
           that lead nowhere (the track never comes back) are dropped by the caller (see asd.py).
Seats      left / right halves of the frame unless a sensors.toml with a [seats] table is given; the zone named
           "screen" marks a display (a face there is media, not a person).

No face recognition and no identity: track ids are per-run counters ("f1", "f2", ...).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import numpy as np

from ..sensors.config import Zone, load_sensors_config, seat_zones
from ..sensors.faces import YUNET_ONNX, IoUTracker

BBox = tuple[float, float, float, float]          # x, y, w, h normalised to the frame (0..1)
Detection = tuple[BBox, float, list[tuple[float, float]]]   # box, score, 5 landmarks (may be empty)

HALVES: dict[str, Zone] = {"left": (0.0, 0.0, 0.5, 1.0), "right": (0.5, 0.0, 0.5, 1.0)}


class Detector(Protocol):
    label: str

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...


@dataclass
class TrackBox:
    track_id: str
    box: BBox                 # normalised x, y, w, h (EMA-smoothed by the tracker)
    seat: str | None
    detected: bool            # False = coasted (last known box while the detector missed the face)
    score: float = 0.0        # detector confidence of the last detection


class YuNetDetector:
    """YuNet on the (optionally upscaled) frame. Normalised boxes, so `scale` does not change the output space."""

    def __init__(self, *, scale: float = 1.0, score_threshold: float = 0.6, nms_threshold: float = 0.3,
                 model_path: str | Path | None = None, threads: int = 6) -> None:
        import cv2  # lazy: unit tests inject a fake detector

        cv2.setNumThreads(int(threads))
        self.cv2 = cv2
        self.scale = float(scale)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.model_path = Path(model_path or YUNET_ONNX)
        self._det = None
        self._size: tuple[int, int] | None = None
        self.label = f"YuNet 2023mar @{self.scale:g}x (thr {self.score_threshold:g})"

    def _detector(self, size: tuple[int, int]):
        if self._det is None:
            self._det = self.cv2.FaceDetectorYN.create(str(self.model_path), "", size, self.score_threshold,
                                                       self.nms_threshold, 5000)
            self._size = size
        elif size != self._size:
            self._det.setInputSize(size)
            self._size = size
        return self._det

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        img = frame_bgr
        if self.scale != 1.0:
            img = self.cv2.resize(frame_bgr, (int(round(w * self.scale)), int(round(h * self.scale))),
                                  interpolation=self.cv2.INTER_CUBIC)
        H, W = img.shape[:2]
        _, faces = self._detector((W, H)).detect(img)
        out: list[Detection] = []
        if faces is None:
            return out
        for row in faces:
            x, y, bw, bh = (float(v) for v in row[:4])
            if bw <= 0 or bh <= 0:
                continue
            box: BBox = (max(0.0, x / W), max(0.0, y / H), min(1.0, bw / W), min(1.0, bh / H))
            lm = [(float(row[4 + 2 * i]) / W, float(row[5 + 2 * i]) / H) for i in range(5)]
            out.append((box, float(row[14]), lm))
        return out


class FakeDetector:
    """Scripted detections for tests: `boxes_at(frame_index) -> list[BBox]`."""

    label = "fake detector"

    def __init__(self, boxes_at: Callable[[int], list[BBox]]) -> None:
        self.boxes_at = boxes_at
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        boxes = self.boxes_at(self.calls)
        self.calls += 1
        return [(b, 0.9, []) for b in boxes]


def zones_from_config(path: str | Path | None) -> dict[str, Zone]:
    """[seats] of a sensors.toml, else the left/right halves."""
    if path is None:
        return dict(HALVES)
    z = seat_zones(load_sensors_config(path))
    return z or dict(HALVES)


class FaceTracks:
    """Per-frame face tracks with coasting. `update(frame, t)` -> list[TrackBox] for that frame."""

    def __init__(self, detector: Detector, *, zones: dict[str, Zone] | None = None, iou_match: float = 0.3,
                 max_lost_s: float = 3.0, coast_s: float = 1.0, smooth: float = 0.6, min_overlap: float = 0.3) -> None:
        self.detector = detector
        self.coast_s = float(coast_s)
        self.tracker = IoUTracker(iou_match=iou_match, max_lost_s=max(max_lost_s, coast_s), smooth=smooth,
                                  zones=zones if zones is not None else dict(HALVES), min_overlap=min_overlap)
        self.last_detect_ms = 0.0

    def update(self, frame_bgr: np.ndarray, t: float) -> list[TrackBox]:
        import time

        t0 = time.perf_counter()
        dets = self.detector.detect(frame_bgr)
        self.last_detect_ms = (time.perf_counter() - t0) * 1000
        seen = self.tracker.update(dets, t)
        by_id = {tr.track_id: tr for tr in self.tracker.tracks}
        out = [TrackBox(f.track_id, f.bbox, f.seat, True, by_id[f.track_id].score) for f in seen]
        seen_ids = {f.track_id for f in seen}
        for tr in self.tracker.tracks:            # coast briefly lost tracks at their last box
            if tr.track_id in seen_ids or t - tr.last_seen > self.coast_s:
                continue
            out.append(TrackBox(tr.track_id, tuple(round(v, 4) for v in tr.bbox), tr.seat, False, tr.score))  # type: ignore[arg-type]
        return out

    def alive(self) -> set[str]:
        return {tr.track_id for tr in self.tracker.tracks}


def crop_square(gray: np.ndarray, box: BBox, *, out: int = 112, scale: float = 0.7, down: float = 0.2,
                fill: int = 110) -> np.ndarray:
    """Light-ASD's face crop (Columbia_test.py: cropScale 0.40, 224 px, centre 112): a square of side
    `scale` x max(w, h) centred on the box centre shifted DOWN by `down` x max(w, h) (mouth-heavy), padded with
    grey where it leaves the frame, resized to `out` x `out`. Input is a grayscale HxW uint8 image."""
    import cv2

    h, w = gray.shape[:2]
    x, y, bw, bh = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    side = max(bw, bh) * scale
    cx, cy = x + bw / 2, y + bh / 2 + max(bw, bh) * down
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.full((out, out), fill, dtype=np.uint8)
    px0, py0, px1, py1 = max(0, -x0), max(0, -y0), max(0, x1 - w), max(0, y1 - h)
    patch = gray[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if px0 or py0 or px1 or py1:
        patch = np.pad(patch, ((py0, py1), (px0, px1)), constant_values=fill)
    return cv2.resize(patch, (out, out), interpolation=cv2.INTER_LINEAR if patch.shape[0] < out else cv2.INTER_AREA)


def iter_frames(path: str | Path, *, every: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
    """(frame index, media time, BGR frame) for every `every`-th frame of a video file, in order."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                return
            if idx % every == 0:
                ok, img = cap.retrieve()
                if ok:
                    yield idx, idx / fps, img
            idx += 1
    finally:
        cap.release()


def video_info(path: str | Path) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    try:
        return {"fps": cap.get(cv2.CAP_PROP_FPS) or 25.0, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    finally:
        cap.release()


__all__ = ["BBox", "Detection", "Detector", "TrackBox", "YuNetDetector", "FakeDetector", "FaceTracks", "HALVES",
           "zones_from_config", "crop_square", "iter_frames", "video_info"]
