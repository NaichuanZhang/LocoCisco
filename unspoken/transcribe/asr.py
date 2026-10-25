"""Transcription with word timings -> AsrResult (unspoken/transcribe/contracts.py), offline, CPU or GPU.

    python -m unspoken.transcribe.asr --in CASE/video.mp4|audio.wav --out asr.json --method NAME [--threads 6]
                                      [--opt key=value ...] [--list]

Methods (CPU int8 for whisper; Parakeet fp32 because its int8 ONNX export loses whole windows here):
  fw-<size>          faster-whisper whole file + its built-in Silero VAD filter, word_timestamps=True
  vad-fw-<size>      our Silero VAD chunks (the live pipeline's settings) + faster-whisper per chunk
                     sizes: base.en, small.en, large-v3-turbo, distil-large-v3.5
  parakeet-tdt-0.6b-v3[-int8]   NVIDIA Parakeet TDT 0.6B v3 via onnx-asr, Silero VAD windows, token timings
  server             the same Silero windows posted to an OpenAI-compatible speech server (GB10: NeMo-Speech.cpp
                     Parakeet on the GPU, --opt url=http://127.0.0.1:8770/v1), words from verbose_json

Python API: transcribe(audio_f32_16k, method, threads=6, **options) -> AsrResult. `timing` carries wall_s,
audio_s, rtf (= wall / audio, model load excluded and reported separately), threads and the 1-minute load
average before and after, so every number can be read with its conditions. Word times are seconds from the
start of the input; `conf` is the model's word probability (whisper) or exp(mean token log-prob) (Parakeet).
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .contracts import AsrResult, to_json

SAMPLE_RATE = 16000
WHISPER_SIZES = ("base.en", "small.en", "large-v3-turbo", "distil-large-v3.5")


def _fw(size: str) -> Callable[..., AsrResult]:
    def run(audio: np.ndarray, **o: Any) -> AsrResult:
        from . import asr_whisper
        return asr_whisper.whole_file(audio, size, **o)
    return run


def _vad_fw(size: str) -> Callable[..., AsrResult]:
    def run(audio: np.ndarray, **o: Any) -> AsrResult:
        from . import asr_whisper
        return asr_whisper.chunked(audio, size, **o)
    return run


def _parakeet(quantization: str | None) -> Callable[..., AsrResult]:
    def run(audio: np.ndarray, **o: Any) -> AsrResult:
        from . import asr_parakeet
        o.setdefault("quantization", quantization)
        return asr_parakeet.transcribe(audio, **o)
    return run


def _server(audio: np.ndarray, **o: Any) -> AsrResult:
    from . import asr_server
    return asr_server.transcribe(audio, **o)


METHODS: dict[str, Callable[..., AsrResult]] = {
    **{f"fw-{s}": _fw(s) for s in WHISPER_SIZES},
    **{f"vad-fw-{s}": _vad_fw(s) for s in WHISPER_SIZES},
    "parakeet-tdt-0.6b-v3": _parakeet(None),          # fp32: the int8 export drops whole windows (see asr_parakeet)
    "parakeet-tdt-0.6b-v3-int8": _parakeet("int8"),
    "server": _server,
}


def load_audio(path: str | Path) -> np.ndarray:
    """float32 mono 16 kHz from a WAV (stdlib when it already is 16 kHz mono PCM16) or any container PyAV
    decodes (mp4 video: the audio track)."""
    p = Path(path)
    if p.suffix.lower() == ".wav":
        from ..sensors.audio import pcm16_to_float, read_wav_pcm16
        return pcm16_to_float(read_wav_pcm16(p, SAMPLE_RATE))
    from faster_whisper.audio import decode_audio  # PyAV
    return np.asarray(decode_audio(str(p), sampling_rate=SAMPLE_RATE), dtype=np.float32)


def _load1() -> float:
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:
        return float("nan")


def transcribe(audio: np.ndarray, method: str, *, threads: int = 6, **options: Any) -> AsrResult:
    """Run `method` on float32 16 kHz mono audio. Model loading (first call) is timed separately from decoding."""
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; one of {sorted(METHODS)}")
    audio = np.asarray(audio, dtype=np.float32)
    load1 = _load1()
    t0 = time.perf_counter()
    res = METHODS[method](audio, threads=threads, **options)
    wall = time.perf_counter() - t0
    audio_s = len(audio) / SAMPLE_RATE
    inner = res.timing.get("decode_s", 0.0) + res.timing.get("vad_s", 0.0)   # time spent after the model was ready
    res.timing.update({"wall_s": round(wall, 3), "audio_s": round(audio_s, 3),
                       "rtf": round(inner / audio_s, 4) if audio_s and inner else round(wall / audio_s, 4),
                       "model_load_s": round(max(0.0, wall - inner), 3) if inner else 0.0,
                       "threads": threads, "load1_start": load1, "load1_end": _load1(),
                       "host": f"{platform.machine()} {platform.system()} CPU", "method": method})
    return res


def _parse_opts(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items:
        k, _, v = it.partition("=")
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        elif v.lower() in ("none", "null"):
            out[k] = None
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, help="video.mp4 (audio track) or audio.wav")
    ap.add_argument("--out", type=Path, help="asr.json to write")
    ap.add_argument("--method", default="fw-base.en", help="see --list")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                    help="method option, e.g. beam_size=1, offset_s=-0.08, max_s=30, quantization=none")
    ap.add_argument("--list", action="store_true", help="list methods and exit")
    a = ap.parse_args(argv)
    if a.list:
        print("\n".join(sorted(METHODS)))
        return 0
    if not a.inp or not a.out:
        ap.error("--in and --out are required")
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    audio = load_audio(a.inp)
    res = transcribe(audio, a.method, threads=a.threads, **_parse_opts(a.opt))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    to_json(res, a.out)
    t = res.timing
    print(f"{a.method}: {len(res.segments)} segments, {sum(len(s.words) for s in res.segments)} words, "
          f"audio {t['audio_s']:.0f} s, decode {t['wall_s'] - t['model_load_s']:.1f} s, RTF {t['rtf']:.3f}, "
          f"{a.threads} threads, load1 {t['load1_start']}->{t['load1_end']} -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
