"""Video capture: BGR frames stamped on the meeting clock. Frames live in memory only and are dropped when
the consumer is slower than the camera; nothing here writes an image to disk.

VideoSource (async iterator of VideoFrame):
  CameraSource   USB webcam via OpenCV/V4L2 in a reader thread; device by index, /dev path or name substring
                 (matched against /sys/class/video4linux/*/name). Downscaled to width x height, paced to fps.
                 A size-1 mailbox keeps only the newest frame: the capture thread never blocks on us.
  Mp4FileSource  replay a video file. realtime=True paces by the clock; realtime=False yields frames whose
                 media time <= pace() (an audio replay's position) so a fast replay stays in sync with the
                 WAV, or as fast as possible when there is no pace.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Protocol

import numpy as np

log = logging.getLogger("unspoken.sensors.video")

Clock = Callable[[], float]


@dataclass
class VideoFrame:
    image: Any            # HxWx3 uint8 BGR numpy array
    t: float              # meeting-clock seconds
    index: int = 0        # capture index (gaps = dropped frames)
    dropped: int = 0      # frames dropped since the previous yielded frame


class VideoSource(Protocol):
    def __aiter__(self) -> AsyncIterator[VideoFrame]: ...
    async def aclose(self) -> None: ...


def resize_to(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0 or image.shape[1] == width:
        return image
    import cv2
    scale = width / image.shape[1]
    h = height if height > 0 else int(round(image.shape[0] * scale))
    return cv2.resize(image, (width, h), interpolation=cv2.INTER_AREA)


def find_v4l2_device(name: str) -> int | str:
    """Index for an int-like string, the path for /dev/videoN, else the first /dev/videoN whose sysfs name
    contains `name` (case-insensitive)."""
    if isinstance(name, int) or (isinstance(name, str) and name.isdigit()):
        return int(name)
    if str(name).startswith("/dev/"):
        return str(name)
    base = Path("/sys/class/video4linux")
    if base.exists():
        for dev in sorted(base.iterdir()):
            try:
                label = (dev / "name").read_text().strip()
            except OSError:
                continue
            if str(name).lower() in label.lower():
                return f"/dev/{dev.name}"
    raise LookupError(f"no camera matching {name!r}")


class CameraSource:
    def __init__(self, *, clock: Clock, device: int | str = 0, fps: float = 5.0, width: int = 640,
                 height: int = 360) -> None:
        self.clock = clock
        self.device = device
        self.fps = max(0.5, float(fps))
        self.width, self.height = int(width), int(height)
        self._latest: VideoFrame | None = None
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._index = 0
        self._taken = 0
        self.error: str | None = None

    def _reader(self) -> None:
        import cv2

        target = find_v4l2_device(self.device)
        cap = cv2.VideoCapture(target, cv2.CAP_V4L2) if isinstance(target, (int, str)) else cv2.VideoCapture(target)
        if not cap.isOpened():
            self.error = f"camera {self.device!r} did not open"
            with self._cond:
                self._cond.notify_all()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        period = 1.0 / self.fps
        next_due = self.clock()
        try:
            while not self._stop.is_set():
                ok, img = cap.read()      # grab at the camera's own rate so its buffer never fills
                if not ok:
                    self.error = "camera read failed"
                    break
                now = self.clock()
                if now < next_due:
                    continue              # this frame is dropped on purpose (pacing to fps)
                next_due = now + period
                img = resize_to(img, self.width, self.height)
                self._index += 1
                with self._cond:
                    self._latest = VideoFrame(img, now, self._index)
                    self._cond.notify_all()
        finally:
            cap.release()
            with self._cond:
                self._cond.notify_all()

    async def __aiter__(self) -> AsyncIterator[VideoFrame]:
        self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
        self._thread.start()
        loop = asyncio.get_running_loop()

        def take() -> VideoFrame | None:
            with self._cond:
                while self._latest is None and not self._stop.is_set() and self.error is None \
                        and self._thread is not None and self._thread.is_alive():
                    self._cond.wait(0.5)
                fr, self._latest = self._latest, None
                return fr

        try:
            while not self._stop.is_set():
                fr = await loop.run_in_executor(None, take)
                if fr is None:
                    if self.error:
                        raise RuntimeError(self.error)
                    return
                fr.dropped = fr.index - self._taken - 1
                self._taken = fr.index
                yield fr
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()


class Mp4FileSource:
    def __init__(self, path: str | Path, *, clock: Clock, realtime: bool = True, fps: float = 5.0,
                 width: int = 640, height: int = 360, t0: float | None = None, speed: float = 1.0,
                 pace: Callable[[], float] | None = None) -> None:
        self.path = Path(path)
        self.clock = clock
        self.realtime = realtime
        self.fps = max(0.5, float(fps))
        self.width, self.height = int(width), int(height)
        self._t0 = t0
        self.speed = max(0.1, speed)
        self.pace = pace
        self._stop = False
        self.finished = asyncio.Event()

    @property
    def t0(self) -> float:
        if self._t0 is None:
            self._t0 = self.clock()
        return self._t0

    async def __aiter__(self) -> AsyncIterator[VideoFrame]:
        import cv2

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video {self.path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(src_fps / self.fps)))
        t0 = self.t0
        start_wall = self.clock()
        index = 0
        taken = 0
        try:
            while not self._stop:
                ok = await asyncio.to_thread(cap.grab)
                if not ok:
                    return
                index += 1
                if (index - 1) % step:
                    continue
                media = (index - 1) / src_fps
                if self.realtime:
                    wait = start_wall + media / self.speed - self.clock()
                    if wait > 0:
                        await asyncio.sleep(wait)
                elif self.pace is not None:
                    while self.pace() < media and not self._stop:
                        await asyncio.sleep(0.005)
                else:
                    await asyncio.sleep(0)
                ok, img = cap.retrieve()
                if not ok:
                    continue
                img = resize_to(img, self.width, self.height)
                fr = VideoFrame(img, t0 + media, index, dropped=index - taken - 1)
                taken = index
                yield fr
        finally:
            cap.release()
            self.finished.set()

    async def aclose(self) -> None:
        self._stop = True
        self.finished.set()


__all__ = ["VideoFrame", "VideoSource", "CameraSource", "Mp4FileSource", "find_v4l2_device", "resize_to"]
