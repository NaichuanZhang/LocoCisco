"""Light-ASD for the LIVE sensors pipeline: an ActiveSpeaker provider (unspoken.contracts.ASDSession) that
scores each tracked face from a sliding window of its recent crops and the recent audio.

    [profiles.gb10.asd]
    provider = "python:unspoken.transcribe.asd_live:LightASDLive"
    label = "Light-ASD (AVA, CUDA)"
    options = { device = "cuda", window_s = 1.0, every_s = 0.2, weights = "lightasd_pretrain_AVA_CVPR.model" }

How it works: every pipeline frame (`update(frame, t, faces, audio_since_last_call)`) each visible face is
cropped (faces.crop_square, 112x112 grey) and appended to its own buffer on the model's 25 fps grid (a frame
arriving late repeats the previous crop, two frames on one tick keep the first); the PCM is appended to a ring
of `window_s` + 1 s. Every `every_s` seconds all tracks with at least `min_s` of crops are scored on their last
`window_s` of crops against the MFCC of the same span (one forward per face, duration = window_s), and the mean
of the last `tail_s` of the clip's per-frame probabilities becomes `FaceTrack.speaking`; between scorings the
last value is held. The pipeline must deliver frames at ~25 fps ([video] fps = 25 in sensors.toml): at 5 fps
(the MotionASD default) the model sees 5 real crops per second repeated 5x and is much weaker.

Cost per scoring (1 s clip, one face): ~55 ms on the shared x86 CPU at 6 threads under load ~50 (measured in
asd_lightasd benchmarks), so 2 faces at 5 Hz is ~0.55 s compute per second on CPU; on the GB10 GPU it is a
few ms. Nothing here writes audio or video to disk.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ..contracts import FaceTrack, RoleSpec
from ..sensors.config import merge_options
from .asd_lightasd import AUDIO_HZ, VIDEO_FPS, DEFAULT_WEIGHTS
from .faces import crop_square


class LightASDLive:
    def __init__(self, spec: RoleSpec | None = None, **options: Any) -> None:
        opts = merge_options(spec, options)
        self.weights = opts.get("weights", DEFAULT_WEIGHTS)
        self.device = str(opts.get("device", "cpu"))
        self.threads = int(opts.get("threads", 6))
        self.window_s = float(opts.get("window_s", 1.0))
        self.every_s = float(opts.get("every_s", 0.2))
        self.min_s = float(opts.get("min_s", 0.4))
        self.tail_s = float(opts.get("tail_s", 0.3))
        self.scorer = opts.get("scorer")              # tests inject a fake; otherwise built lazily
        self.label = f"Light-ASD live ({self.weights}, {self.window_s:g} s window, {self.device})"

    def _scorer(self):
        if self.scorer is None:
            from .asd_lightasd import LightASDScorer
            self.scorer = LightASDScorer(self.weights, device=self.device, threads=self.threads,
                                         durations=(max(1, int(round(self.window_s))),))
        return self.scorer

    def session(self) -> "LightASDLiveSession":
        return LightASDLiveSession(self)

    def healthy(self) -> bool:
        try:
            self._scorer()
            return True
        except Exception:
            return False


class LightASDLiveSession:
    def __init__(self, asd: LightASDLive) -> None:
        self.a = asd
        self.win = int(round(asd.window_s * VIDEO_FPS))
        self.crops: dict[str, deque[np.ndarray]] = {}
        self.last_g: dict[str, int] = {}
        self.speaking: dict[str, float] = {}
        self.pcm: deque[np.ndarray] = deque()
        self.pcm_len = 0
        self.pcm_max = int((asd.window_s + 1.0) * 16000)
        self.last_scored = -1e9
        self.last_ms = 0.0

    def _push_audio(self, audio: bytes | None, sample_rate: int) -> None:
        if not audio:
            return
        x = np.frombuffer(audio, dtype="<i2")
        if sample_rate != 16000:
            from ..sensors.audio import resample_pcm16
            x = np.frombuffer(resample_pcm16(x.tobytes(), sample_rate, 16000), dtype="<i2")
        self.pcm.append(x)
        self.pcm_len += len(x)
        while self.pcm_len - len(self.pcm[0]) >= self.pcm_max and len(self.pcm) > 1:
            self.pcm_len -= len(self.pcm.popleft())

    def _push_crop(self, tid: str, g: int, crop: np.ndarray) -> None:
        buf = self.crops.setdefault(tid, deque(maxlen=self.win))
        last = self.last_g.get(tid)
        if last is not None and g <= last:
            return
        if last is not None:
            for _ in range(min(self.win, g - last - 1)):
                buf.append(crop)
        buf.append(crop)
        self.last_g[tid] = g

    def _score(self, faces: list[FaceTrack]) -> None:
        import time

        from .asd_lightasd import mfcc_features

        t0 = time.perf_counter()
        pcm = np.concatenate(list(self.pcm)) if self.pcm else np.zeros(0, dtype=np.int16)
        need = int(self.a.window_s * 16000)
        pcm = pcm[-need:] if len(pcm) >= need else np.pad(pcm, (need - len(pcm), 0))
        mf = mfcc_features(pcm)                                   # (100 * window_s, 13)
        scorer = self.a._scorer()
        tail = max(1, int(round(self.a.tail_s * VIDEO_FPS)))
        for f in faces:
            buf = self.crops.get(f.track_id)
            if not buf or len(buf) < self.a.min_s * VIDEO_FPS:
                continue
            crops = np.stack(buf)
            rows = mf[-crops.shape[0] * (AUDIO_HZ // VIDEO_FPS):]    # the audio span matching the crops
            p = scorer.score(crops, rows)
            self.speaking[f.track_id] = float(np.mean(p[-tail:]))
        self.last_ms = (time.perf_counter() - t0) * 1000

    def update(self, frame: Any, t: float, faces: list[FaceTrack], audio: bytes | None,
               sample_rate: int = 16000) -> list[FaceTrack]:
        self._push_audio(audio, sample_rate)
        if frame is not None and faces:
            img = np.asarray(frame)
            gray = (0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]).astype(np.uint8) if img.ndim == 3 else img
            g = int(round(t * VIDEO_FPS))
            for f in faces:
                self._push_crop(f.track_id, g, crop_square(gray, f.bbox))
        alive = {f.track_id for f in faces}
        for tid in list(self.crops):
            if tid not in alive:
                self.crops.pop(tid, None); self.last_g.pop(tid, None); self.speaking.pop(tid, None)
        if faces and t - self.last_scored >= self.a.every_s:
            self._score(faces)
            self.last_scored = t
        return [FaceTrack(f.track_id, f.bbox, f.seat, round(self.speaking.get(f.track_id, 0.0), 3)) for f in faces]

    def reset(self) -> None:
        self.crops.clear(); self.last_g.clear(); self.speaking.clear(); self.pcm.clear()
        self.pcm_len = 0
        self.last_scored = -1e9


__all__ = ["LightASDLive", "LightASDLiveSession"]
