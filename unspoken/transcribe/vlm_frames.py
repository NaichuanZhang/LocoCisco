"""Frames for the VLM judge: sample K frames from a time unit, show them as one picture the model can read.

Views (what the model is shown for one unit of time):
  full        the whole frame(s), downscaled to `max_side`; K frames = K images in the request
  faces       one tiled picture: a row per sampled frame (time runs top to bottom), two columns = the LEFT
              and the RIGHT person's face crop (square around the detected face, `zoom` x the face box,
              resized to `tile` px), labelled "LEFT" / "RIGHT" in the header row
  faces-diff  like `faces` plus, next to every crop, |frame - frame 0.2 s earlier| in the same crop, so
              lip motion shows as bright pixels (grey tiles = still)

Faces come from YuNet (unspoken.sensors.faces, the same detector the live pipeline uses) and are assigned to
the seat whose zone holds them (left / right halves by default). When a seat has no detection in a frame the
crop falls back to the last box seen for that seat, then to the centre of the seat zone. No recognition, no
identity: a seat is a region of the picture.

Everything here is pure numpy / OpenCV so tests can run on synthetic frames with a fake face finder.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

BBox = tuple[float, float, float, float]            # normalised x, y, w, h (0..1 of the frame)
DEFAULT_SEATS: dict[str, BBox] = {"left": (0.0, 0.0, 0.5, 1.0), "right": (0.5, 0.0, 0.5, 1.0)}
SEATS = ("left", "right")


def sample_times(start: float, end: float, k: int) -> list[float]:
    """K times evenly spread inside [start, end]: the midpoint for k=1, bin centres otherwise."""
    k = max(1, int(k))
    d = max(end - start, 0.0)
    return [start + (i + 0.5) * d / k for i in range(k)]


def windows(start: float, end: float, length: float) -> list[tuple[float, float]]:
    """Cut [start, end] into windows of `length` s; a short tail (< length/2) joins the previous window."""
    if end - start <= length * 1.5:
        return [(start, end)]
    n = int(round((end - start) / length))
    edges = [start + i * (end - start) / n for i in range(n + 1)]
    return [(edges[i], edges[i + 1]) for i in range(n)]


class FrameSource(Protocol):
    width: int
    height: int
    fps: float
    duration: float

    def read(self, t: float) -> np.ndarray | None: ...        # BGR uint8 frame nearest to t, None past the end


class VideoFrames:
    """cv2.VideoCapture with seek-by-time. Sequential reads of nearby times reuse the decoder position."""

    def __init__(self, path: str | Path) -> None:
        import cv2  # lazy: tests with synthetic frames do not need it

        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video {self.path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.n_frames / self.fps if self.n_frames > 0 else 0.0
        self._pos = -1          # index of the frame the next read() returns
        self._cv2 = cv2

    def read(self, t: float) -> np.ndarray | None:
        idx = int(round(t * self.fps))
        if idx < 0 or (self.n_frames > 0 and idx >= self.n_frames):
            return None
        if idx != self._pos:
            if 0 <= idx - self._pos <= 12:                  # close ahead: decode forward instead of seeking
                for _ in range(idx - self._pos - 1):
                    if not self.cap.grab():
                        return None
            else:
                self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        self._pos = idx + 1 if ok else -1
        return frame if ok else None

    def close(self) -> None:
        self.cap.release()


class ArrayFrames:
    """A FrameSource over an in-memory list of frames (tests, fixtures)."""

    def __init__(self, frames: list[np.ndarray], fps: float = 25.0) -> None:
        self.frames = frames
        self.fps = fps
        self.height, self.width = frames[0].shape[:2]
        self.duration = len(frames) / fps

    def read(self, t: float) -> np.ndarray | None:
        idx = int(round(t * self.fps))
        return self.frames[idx] if 0 <= idx < len(self.frames) else None


FaceFinder = Callable[[np.ndarray], list[BBox]]      # frame -> normalised face boxes


def yunet_finder(score_threshold: float = 0.5) -> FaceFinder:
    """The live pipeline's YuNet detector as a plain function (OpenCV, CPU, ~10 ms per small frame)."""
    from ..sensors.faces import YuNetFaces

    sess = YuNetFaces(seats={s: list(z) for s, z in DEFAULT_SEATS.items()}, score_threshold=score_threshold).session()
    return lambda frame: [box for box, _, _ in sess.detect(frame)]


def seat_of(box: BBox, seats: dict[str, BBox] = DEFAULT_SEATS) -> str | None:
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    for seat, (x, y, w, h) in seats.items():
        if x <= cx < x + w and y <= cy < y + h:
            return seat
    return None


@dataclass
class SeatBoxes:
    """Per-seat face box with memory: the last detection wins, then the seat zone centre."""
    seats: dict[str, BBox] = field(default_factory=lambda: dict(DEFAULT_SEATS))
    last: dict[str, BBox] = field(default_factory=dict)
    seen: dict[str, int] = field(default_factory=dict)
    sum: dict[str, np.ndarray] = field(default_factory=dict)

    def update(self, boxes: list[BBox]) -> dict[str, BBox]:
        best: dict[str, BBox] = {}
        for b in boxes:
            s = seat_of(b, self.seats)
            if s and (s not in best or b[2] * b[3] > best[s][2] * best[s][3]):
                best[s] = b
        for s, b in best.items():
            self.last[s] = b
            self.seen[s] = self.seen.get(s, 0) + 1
            self.sum[s] = self.sum.get(s, np.zeros(4)) + np.array(b)
        return best

    def box(self, seat: str) -> BBox:
        if seat in self.last:
            return self.last[seat]
        x, y, w, h = self.seats.get(seat, DEFAULT_SEATS["left"])
        side = min(w, h) * 0.5
        return (x + w / 2 - side / 2, y + h / 2 - side / 2, side, side)

    def mean_box(self, seat: str) -> BBox | None:
        n = self.seen.get(seat, 0)
        return tuple(float(v) for v in self.sum[seat] / n) if n else None  # type: ignore[return-value]


def crop_square(frame: np.ndarray, box: BBox, zoom: float, size: int) -> np.ndarray:
    """Square crop centred on the face box, side = zoom * max(w, h) of the box, clipped and resized."""
    import cv2

    h, w = frame.shape[:2]
    bx, by, bw, bh = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    side = max(8.0, zoom * max(bw, bh))
    cx, cy = bx + bw / 2, by + bh / 2 + 0.1 * bh          # a little below centre: mouth matters more than hair
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        crop = np.zeros((8, 8, 3), dtype=np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC if side < size else cv2.INTER_AREA)


def downscale(frame: np.ndarray, max_side: int) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return frame
    f = max_side / s
    return cv2.resize(frame, (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=cv2.INTER_AREA)


def _label(img: np.ndarray, text: str, color=(255, 255, 255)) -> None:
    import cv2

    cv2.putText(img, text, (6, int(img.shape[0] * 0.75)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, int(img.shape[0] * 0.75)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def faces_grid(frames: list[np.ndarray], boxes: list[dict[str, BBox]], *, tile: int = 224, zoom: float = 2.2,
               diffs: list[np.ndarray] | None = None, header: bool = True) -> np.ndarray:
    """Rows = frames (time order), columns = LEFT person, RIGHT person (each optionally followed by its
    motion tile). `boxes[i]` holds the seat -> face box for frame i; `diffs[i]` the earlier frame for the
    motion tile."""
    import cv2

    rows = []
    for i, (fr, bx) in enumerate(zip(frames, boxes)):
        cells = []
        for seat in SEATS:
            b = bx[seat]
            cells.append(crop_square(fr, b, zoom, tile))
            if diffs is not None:
                prev = crop_square(diffs[i], b, zoom, tile)
                d = cv2.absdiff(cells[-1], prev)
                d = cv2.cvtColor(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                cells.append(np.clip(d.astype(np.int32) * 3, 0, 255).astype(np.uint8))
        rows.append(np.concatenate(cells, axis=1))
    grid = np.concatenate(rows, axis=0)
    if header:
        hdr = np.full((28, grid.shape[1], 3), 40, dtype=np.uint8)
        ncol = 2 if diffs is None else 4
        cw = grid.shape[1] // ncol
        for j, seat in enumerate(SEATS):
            x = j * (ncol // 2) * cw
            cell = hdr[:, x:x + cw]
            _label(cell, f"{seat.upper()} person")
            if diffs is not None:
                _label(hdr[:, x + cw:x + 2 * cw], f"{seat.upper()} motion", (180, 220, 255))
        grid = np.concatenate([hdr, grid], axis=0)
    return grid


def jpeg_data_url(img: np.ndarray, quality: int = 85) -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


@dataclass
class UnitPictures:
    images: list[np.ndarray]                 # what the model sees (1 grid, or K full frames)
    times: list[float]                       # sampled frame times
    boxes: list[dict[str, BBox]]             # per frame seat -> face box (for asd.json)
    frames_read: int = 0


class UnitSampler:
    """Turns a time unit into the picture(s) for the model, keeping per-seat face boxes across units."""

    def __init__(self, source: FrameSource, *, view: str = "faces", k: int = 3, tile: int = 224, zoom: float = 2.2,
                 max_side: int = 704, diff_dt: float = 0.2, finder: FaceFinder | None = None,
                 seats: dict[str, BBox] | None = None) -> None:
        if view not in ("full", "faces", "faces-diff"):
            raise ValueError(f"unknown view {view!r}")
        self.src = source
        self.view, self.k, self.tile, self.zoom, self.max_side, self.diff_dt = view, k, tile, zoom, max_side, diff_dt
        self.finder = finder
        self.boxes = SeatBoxes(dict(seats or DEFAULT_SEATS))
        self.frames_read = 0

    def _frame(self, t: float) -> np.ndarray | None:
        f = self.src.read(min(max(t, 0.0), max(self.src.duration - 1.0 / self.src.fps, 0.0)))
        if f is not None:
            self.frames_read += 1
        return f

    def sample(self, start: float, end: float) -> UnitPictures | None:
        times = sample_times(start, end, self.k)
        frames, boxes, prevs, used = [], [], [], []
        for t in times:
            prev = self._frame(t - self.diff_dt) if self.view == "faces-diff" else None
            f = self._frame(t)
            if f is None:
                continue
            found = self.boxes.update(self.finder(f)) if self.finder else {}
            boxes.append({s: found.get(s, self.boxes.box(s)) for s in SEATS})
            frames.append(f)
            prevs.append(prev if prev is not None else f)
            used.append(t)
        if not frames:
            return None
        if self.view == "full":
            imgs = [downscale(f, self.max_side) for f in frames]
        else:
            imgs = [faces_grid(frames, boxes, tile=self.tile, zoom=self.zoom,
                               diffs=prevs if self.view == "faces-diff" else None)]
        return UnitPictures(imgs, used, boxes, len(frames))


__all__ = ["sample_times", "windows", "VideoFrames", "ArrayFrames", "UnitSampler", "UnitPictures", "SeatBoxes",
           "faces_grid", "crop_square", "downscale", "jpeg_data_url", "yunet_finder", "seat_of", "DEFAULT_SEATS",
           "SEATS"]
