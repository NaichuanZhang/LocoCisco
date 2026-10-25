"""Start a llama.cpp `llama-server` with a vision projector for the VLM judge (127.0.0.1 only).

  python -m unspoken.transcribe.vlm_server --model dev-models/vlm/qwen3-vl-2b/Qwen3-VL-2B-Instruct-Q8_0.gguf \
      --mmproj dev-models/vlm/qwen3-vl-2b/mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf --port 8795 --threads 6

Runs in the foreground until Ctrl-C (or use `start()` / `stop()` from code). The build dir comes from
$LLAMA_CPP_DIR or the hack-day dev-models path; on the GB10 the same flags work with a CUDA build plus
`-ngl 99` (`--gpu-layers`), see docs/reports/tx-vlm.md.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_DIR = Path(os.environ.get("LLAMA_CPP_DIR") or
                   "/local/home/zhangnai/kirocrew-sessions/2026-10-23-dell-nvidia-hackathon/dev-models/llama.cpp/llama-b11869")


def command(model: Path, mmproj: Path, port: int, threads: int, ctx: int = 8192, gpu_layers: int = 0,
            llama_dir: Path = DEFAULT_DIR, extra: list[str] | None = None) -> list[str]:
    cmd = [str(llama_dir / "llama-server"), "-m", str(model), "--mmproj", str(mmproj), "--host", "127.0.0.1",
           "--port", str(port), "-t", str(threads), "-tb", str(threads), "-c", str(ctx), "-np", "1",
           "--no-warmup", "--log-disable"]
    if gpu_layers:
        cmd += ["-ngl", str(gpu_layers)]
    return cmd + list(extra or [])


def wait_healthy(url: str, timeout_s: float = 300.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url + "/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def start(model: Path, mmproj: Path, port: int = 8795, threads: int = 6, ctx: int = 8192, gpu_layers: int = 0,
          llama_dir: Path = DEFAULT_DIR, log: Path | None = None, extra: list[str] | None = None,
          timeout_s: float = 300.0) -> subprocess.Popen:
    env = dict(os.environ, LD_LIBRARY_PATH=f"{llama_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}",
               OMP_NUM_THREADS=str(threads))
    out = open(log, "ab") if log else subprocess.DEVNULL
    p = subprocess.Popen(command(model, mmproj, port, threads, ctx, gpu_layers, llama_dir, extra), env=env,
                         stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if not wait_healthy(f"http://127.0.0.1:{port}", timeout_s):
        stop(p)
        raise RuntimeError(f"llama-server on :{port} did not become healthy (log: {log})")
    return p


def stop(p: subprocess.Popen, grace_s: float = 10.0) -> None:
    if p.poll() is not None:
        return
    p.send_signal(signal.SIGINT)
    try:
        p.wait(grace_s)
    except subprocess.TimeoutExpired:
        p.kill()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--mmproj", required=True, type=Path)
    ap.add_argument("--port", type=int, default=8795)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--gpu-layers", type=int, default=0)
    ap.add_argument("--llama-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("extra", nargs="*", help="extra llama-server flags after --")
    a = ap.parse_args(argv)
    p = start(a.model, a.mmproj, a.port, a.threads, a.ctx, a.gpu_layers, a.llama_dir, a.log, a.extra)
    print(f"llama-server ready on http://127.0.0.1:{a.port} (pid {p.pid}, {a.threads} threads)", flush=True)
    try:
        p.wait()
    except KeyboardInterrupt:
        stop(p)
        sys.exit(0)


if __name__ == "__main__":
    main()
