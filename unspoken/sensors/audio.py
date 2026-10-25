"""Audio capture: 16 kHz mono PCM16 chunks stamped on the meeting clock. Nothing here writes audio to disk.

AudioSource (async iterator of AudioChunk):
  MicSource      USB speakerphone via sounddevice/PortAudio; falls back to an `arecord` subprocess when
                 PortAudio is missing (DGX OS minimal installs). Device picked by name substring.
  WavFileSource  replay a WAV (16 kHz mono PCM16 read with the stdlib; anything else decoded via PyAV),
                 in real time (paced by the clock) or as fast as possible.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable, Protocol

import numpy as np

log = logging.getLogger("unspoken.sensors.audio")

SAMPLE_RATE = 16000
Clock = Callable[[], float]


@dataclass
class AudioChunk:
    pcm16: bytes          # mono 16-bit little-endian
    sample_rate: int
    t: float              # meeting-clock seconds of the first sample

    @property
    def duration(self) -> float:
        return len(self.pcm16) / 2 / self.sample_rate


class AudioSource(Protocol):
    def __aiter__(self) -> AsyncIterator[AudioChunk]: ...
    async def aclose(self) -> None: ...


# ----------------------------------------------------------------------------- PCM helpers
def pcm16_to_float(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def resample_pcm16(pcm16: bytes, src_rate: int, dst_rate: int = SAMPLE_RATE) -> bytes:
    """Cheap resampling for capture fallbacks (box-filter decimation for integer ratios, else linear).
    Good enough for speech; file replay of odd rates goes through PyAV instead."""
    if src_rate == dst_rate:
        return pcm16
    x = pcm16_to_float(pcm16)
    if src_rate % dst_rate == 0:
        k = src_rate // dst_rate
        n = (len(x) // k) * k
        y = x[:n].reshape(-1, k).mean(axis=1)
    else:
        n = int(round(len(x) * dst_rate / src_rate))
        y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x) if len(x) > 1 else x
    return float_to_pcm16(y)


def rms(pcm16: bytes) -> float:
    if not pcm16:
        return 0.0
    x = pcm16_to_float(pcm16)
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


# ----------------------------------------------------------------------------- WAV replay
def read_wav_pcm16(path: str | Path, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Whole file as 16 kHz mono PCM16. Stdlib for plain PCM16 WAVs; PyAV (a faster-whisper dependency)
    for anything else (other rates, stereo, mp3 ...)."""
    p = Path(path)
    try:
        with wave.open(str(p), "rb") as w:
            if w.getsampwidth() == 2 and w.getnchannels() == 1 and w.getframerate() == sample_rate:
                return w.readframes(w.getnframes())
            if w.getsampwidth() == 2:
                raw = w.readframes(w.getnframes())
                x = np.frombuffer(raw, dtype="<i2").reshape(-1, w.getnchannels()).mean(axis=1) / 32768.0
                return resample_pcm16(float_to_pcm16(x), w.getframerate(), sample_rate)
    except wave.Error:
        pass
    from faster_whisper.audio import decode_audio  # lazy: PyAV

    return float_to_pcm16(np.asarray(decode_audio(str(p), sampling_rate=sample_rate), dtype=np.float32))


class WavFileSource:
    """Yields chunk_ms chunks. realtime=True sleeps so chunks arrive at playback pace (clock = meeting clock);
    realtime=False yields as fast as the consumer takes them. `t` of each chunk = t0 + media time, where t0
    is the clock value at the first chunk (so audio and video replay share one origin when started together)."""

    def __init__(self, path: str | Path, *, clock: Clock, realtime: bool = True, chunk_ms: int = 32,
                 sample_rate: int = SAMPLE_RATE, t0: float | None = None, speed: float = 1.0,
                 pace: Callable[[], float] | None = None) -> None:
        self.path = Path(path)
        self.clock = clock
        self.realtime = realtime
        self.pace = pace                 # fast mode: do not hand out audio past pace() seconds of media
        self.chunk = int(sample_rate * chunk_ms / 1000) * 2
        self.sample_rate = sample_rate
        self.speed = max(0.1, speed)
        self._t0 = t0
        self._pcm: bytes | None = None
        self.media_time = 0.0            # seconds of audio handed out so far
        self.finished = asyncio.Event()

    @property
    def t0(self) -> float:
        if self._t0 is None:
            self._t0 = self.clock()
        return self._t0

    def load(self) -> bytes:
        if self._pcm is None:
            self._pcm = read_wav_pcm16(self.path, self.sample_rate)
        return self._pcm

    @property
    def duration(self) -> float:
        return len(self.load()) / 2 / self.sample_rate

    async def __aiter__(self) -> AsyncIterator[AudioChunk]:
        pcm = self.load()
        t0 = self.t0
        start_wall = self.clock()
        pos = 0
        try:
            while pos < len(pcm):
                piece = pcm[pos: pos + self.chunk]
                media = pos / 2 / self.sample_rate
                if self.realtime:
                    due = start_wall + media / self.speed
                    wait = due - self.clock()
                    if wait > 0:
                        await asyncio.sleep(wait)
                elif self.pace is not None:
                    while self.pace() < media and not self.finished.is_set():
                        await asyncio.sleep(0.002)
                else:
                    await asyncio.sleep(0)   # let the consumers run
                pos += len(piece)
                self.media_time = pos / 2 / self.sample_rate
                yield AudioChunk(piece, self.sample_rate, t0 + media)
        finally:
            self.finished.set()

    async def aclose(self) -> None:
        self.finished.set()


# ----------------------------------------------------------------------------- microphone
def pick_input_device(name: str) -> int | None:
    """sounddevice input device index whose name contains `name` (case-insensitive); None = default."""
    if not name:
        return None
    import sounddevice as sd  # lazy: needs the PortAudio shared library

    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0 and name.lower() in str(dev.get("name", "")).lower():
            return i
    raise LookupError(f"no input device matching {name!r}; devices: "
                      f"{[d['name'] for d in sd.query_devices() if d.get('max_input_channels', 0) > 0]}")


class MicSource:
    """Live microphone. backend: "auto" (sounddevice, then arecord), "sounddevice", "arecord".
    Chunks are never dropped: the callback thread hands every block to the asyncio queue."""

    def __init__(self, *, clock: Clock, device: str = "", alsa_device: str = "default", chunk_ms: int = 32,
                 sample_rate: int = SAMPLE_RATE, backend: str = "auto") -> None:
        self.clock = clock
        self.device = device
        self.alsa_device = alsa_device
        self.chunk_ms = chunk_ms
        self.sample_rate = sample_rate
        self.backend = backend
        self._queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
        self._stop = threading.Event()
        self._stream = None
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task | None = None
        self.used_backend = ""

    async def __aiter__(self) -> AsyncIterator[AudioChunk]:
        await self._start()
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    return
                yield item
        finally:
            await self.aclose()

    async def _start(self) -> None:
        loop = asyncio.get_running_loop()
        errors: list[str] = []
        if self.backend in ("auto", "sounddevice"):
            try:
                self._start_sounddevice(loop)
                self.used_backend = "sounddevice"
                return
            except Exception as exc:  # PortAudio missing, device busy, unsupported rate ...
                errors.append(f"sounddevice: {type(exc).__name__}: {exc}")
                if self.backend == "sounddevice":
                    raise
        if self.backend in ("auto", "arecord"):
            if shutil.which("arecord") is None:
                errors.append("arecord: not installed")
            else:
                await self._start_arecord(loop)
                self.used_backend = "arecord"
                return
        raise RuntimeError("no audio capture backend: " + "; ".join(errors))

    def _start_sounddevice(self, loop: asyncio.AbstractEventLoop) -> None:
        import sounddevice as sd

        index = pick_input_device(self.device)
        info = sd.query_devices(index, "input") if index is not None else sd.query_devices(kind="input")
        native = int(info["default_samplerate"])
        rate = self.sample_rate
        try:
            sd.check_input_settings(device=index, samplerate=rate, channels=1, dtype="int16")
        except Exception:
            rate = native  # open at the native rate and resample in software
        block = int(rate * self.chunk_ms / 1000)
        chunk_s = block / rate

        def callback(indata, frames, time_info, status):  # PortAudio thread
            if status:
                log.debug("portaudio status: %s", status)
            raw = bytes(indata[:, 0].tobytes()) if indata.ndim == 2 else bytes(indata.tobytes())
            if rate != self.sample_rate:
                raw = resample_pcm16(raw, rate, self.sample_rate)
            t = self.clock() - chunk_s
            loop.call_soon_threadsafe(self._queue.put_nowait, AudioChunk(raw, self.sample_rate, t))

        self._stream = sd.InputStream(samplerate=rate, channels=1, dtype="int16", device=index,
                                      blocksize=block, callback=callback)
        self._stream.start()
        log.info("mic: sounddevice device=%s rate=%d (%s)", info.get("name"), rate,
                 "native" if rate == self.sample_rate else "resampled")

    async def _start_arecord(self, loop: asyncio.AbstractEventLoop) -> None:
        cmd = ["arecord", "-D", self.alsa_device, "-f", "S16_LE", "-r", str(self.sample_rate), "-c", "1",
               "-t", "raw", "-q"]
        self._proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                          stderr=asyncio.subprocess.DEVNULL)
        block = int(self.sample_rate * self.chunk_ms / 1000) * 2
        chunk_s = block / 2 / self.sample_rate
        assert self._proc.stdout is not None
        stdout = self._proc.stdout

        async def reader() -> None:
            try:
                while True:
                    raw = await stdout.readexactly(block)
                    self._queue.put_nowait(AudioChunk(raw, self.sample_rate, self.clock() - chunk_s))
            except (asyncio.IncompleteReadError, asyncio.CancelledError):
                pass
            finally:
                self._queue.put_nowait(None)

        self._reader = asyncio.create_task(reader(), name="arecord-reader")
        log.info("mic: arecord -D %s", self.alsa_device)

    async def aclose(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), 2)
            except asyncio.TimeoutError:
                self._proc.kill()
        if self._reader is not None:
            self._reader.cancel()
        self._queue.put_nowait(None)


def default_audio_backend() -> str:
    """"sounddevice" when PortAudio loads, else "arecord" when present, else "none". For `check` CLIs."""
    try:
        import sounddevice
        sounddevice.get_portaudio_version()
        return "sounddevice"
    except Exception:
        return "arecord" if shutil.which("arecord") else "none"


def list_input_devices() -> list[str]:
    try:
        import sounddevice as sd
        return [d["name"] for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
    except Exception:
        return []


__all__ = ["AudioChunk", "AudioSource", "MicSource", "WavFileSource", "SAMPLE_RATE", "pcm16_to_float",
           "float_to_pcm16", "resample_pcm16", "rms", "read_wav_pcm16", "pick_input_device",
           "default_audio_backend", "list_input_devices"]
