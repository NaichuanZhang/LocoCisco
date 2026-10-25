"""One command: video in, per-person transcript out (the winners of the AMI real test, see docs/TRANSCRIBE.md).

    python -m unspoken.transcribe --video meeting.mp4 --out DIR [--profile cpu|gb10] [--ref CASE_DIR]

Stages (each writes its JSON into DIR; asr, diar and asd run in parallel):
  asr.json         Parakeet TDT 0.6B v3 with word timings (cpu: fp32 ONNX via onnx-asr; gb10: NeMo-Speech.cpp on :8770)
  diar.json        WeSpeaker CAM++ on 1.5 s windows over Silero speech + spectral clustering (CPU on both)
  asd.json         YuNet faces + Light-ASD (AVA weights, 6-duration ensemble; gb10: on the GPU)
  transcript.json  fuse `bind`: words -> voice turns, voices <-> seats on whole-session evidence
  transcript.txt / .srt / .md  readable renderings
With --ref, scores.json is written too (eval.transcribe.score). Audio is read from the video's own track.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROFILES = {
    "cpu": {"asr": ["--method", "parakeet-tdt-0.6b-v3"],
            "diar": ["--method", "window"],
            "asd": ["--method", "lightasd", "--device", "cpu"]},
    "gb10": {"asr": ["--method", "server", "--opt", "url=http://127.0.0.1:8770/v1"],
             "diar": ["--method", "window"],
             "asd": ["--method", "lightasd", "--device", "cuda"]},
}


def _audio(video: Path, out: Path) -> Path:
    wav = out / "audio.wav"           # scratch for the audio stages; deleted at the end (privacy: no audio kept)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(wav)], check=True)
    return wav


def _stage(name: str, argv: list[str], log: Path) -> tuple[str, float]:
    t = time.perf_counter()
    with log.open("w") as f:
        subprocess.run([sys.executable, "-m", f"unspoken.transcribe.{name}", *argv], check=True, stdout=f,
                       stderr=subprocess.STDOUT)
    return name, time.perf_counter() - t


def run(video: Path, out: Path, profile: str, threads: int, ref: Path | None, keep_audio: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    p = PROFILES[profile]
    t0 = time.perf_counter()
    wav = _audio(video, out)
    th = ["--threads", str(threads)]
    jobs = {"asr": ["--in", str(wav), "--out", str(out / "asr.json"), *p["asr"], *th],
            "diar": ["--in", str(wav), "--out", str(out / "diar.json"), *p["diar"], *th],
            "asd": ["--in", str(video), "--audio", str(wav), "--out", str(out / "asd.json"), *p["asd"], *th,
                    "--quiet"]}
    timing: dict[str, float] = {}
    try:
        with ThreadPoolExecutor(3) as ex:
            for name, secs in ex.map(lambda kv: _stage(kv[0], kv[1], out / f"{kv[0]}.log"), jobs.items()):
                timing[f"{name}_s"] = round(secs, 1)
        t = time.perf_counter()
        subprocess.run([sys.executable, "-m", "unspoken.transcribe.fuse", "--asr", str(out / "asr.json"),
                        "--diar", str(out / "diar.json"), "--asd", str(out / "asd.json"),
                        "--out", str(out / "transcript.json"), "--method", "bind", "--render"], check=True,
                       stdout=subprocess.DEVNULL)
        timing["fuse_s"] = round(time.perf_counter() - t, 2)
    finally:
        if not keep_audio:
            wav.unlink(missing_ok=True)
    timing["total_s"] = round(time.perf_counter() - t0, 1)
    result: dict = {"video": str(video), "profile": profile, "threads": threads, "timing": timing}
    if ref is not None:
        r = subprocess.run([sys.executable, "-m", "eval.transcribe.score", "--ref", str(ref), "--hyp",
                            str(out / "transcript.json"), "--json"], check=True, capture_output=True, text=True)
        result["scores"] = json.loads(r.stdout)
        (out / "scores.json").write_text(json.dumps(result["scores"], indent=1))
    (out / "run.json").write_text(json.dumps(result, indent=1))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--profile", default="cpu", choices=sorted(PROFILES))
    ap.add_argument("--threads", type=int, default=6, help="per stage")
    ap.add_argument("--ref", type=Path, help="AMI-style case dir: also score the transcript")
    ap.add_argument("--keep-audio", action="store_true", help="keep DIR/audio.wav (default: deleted)")
    a = ap.parse_args(argv)
    r = run(a.video, a.out, a.profile, a.threads, a.ref, a.keep_audio)
    print(json.dumps({k: r[k] for k in ("timing",)} | ({"scores": {k: round(v, 3) for k, v in r["scores"].items()
                                                                  if isinstance(v, float)}} if "scores" in r else {})))
    print(f"transcript: {a.out / 'transcript.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
