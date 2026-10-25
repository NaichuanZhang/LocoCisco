"""Model file locations for the in-process sensor providers.

Vendored (ship in the repo, see models/SOURCES.md): Silero VAD, YuNet.
Downloaded once (too big for git): the WeSpeaker speaker-embedding ONNX. Lookup order for a file name:
  1. $UNSPOKEN_MODELS_DIR/<name>
  2. ~/.cache/unspoken/models/<name>
  3. unspoken/sensors/models/<name> (vendored)
`python -m unspoken.sensors.fetch_models` downloads the pinned files into 1. (or 2.) and checks sha256.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

VENDORED_DIR = Path(__file__).resolve().parent / "models"
SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"

PINNED: dict[str, tuple[str, str]] = {   # name -> (url, sha256)
    "wespeaker_en_voxceleb_CAM++.onnx": (
        SHERPA + "wespeaker_en_voxceleb_CAM++.onnx",
        "c46fad10b5f81e1aa4a60c162714208577093655076c5450f8c469e522ec54ef"),
    "wespeaker_en_voxceleb_resnet34_LM.onnx": (
        SHERPA + "wespeaker_en_voxceleb_resnet34_LM.onnx",
        "e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012"),
    "silero_vad.onnx": (
        "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx",
        "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"),
    "face_detection_yunet_2023mar.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
}


def models_dir() -> Path:
    env = os.environ.get("UNSPOKEN_MODELS_DIR")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "unspoken" / "models"


def resolve_model(name: str) -> Path:
    """First existing location; falls back to the download dir path (may not exist yet) so error messages
    name the place to put the file."""
    for base in (models_dir(), Path.home() / ".cache" / "unspoken" / "models", VENDORED_DIR):
        p = base / name
        if p.exists():
            return p
    return models_dir() / name


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, dest_dir: Path | None = None, force: bool = False) -> Path:
    """Download a pinned file (sha256 verified). Returns the local path."""
    url, digest = PINNED[name]
    dest_dir = dest_dir or models_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if dest.exists() and not force and sha256_of(dest) == digest:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:  # noqa: S310 (pinned https URL)
        for block in iter(lambda: r.read(1 << 20), b""):
            f.write(block)
    got = sha256_of(tmp)
    if got != digest:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: sha256 mismatch (got {got}, want {digest}); refusing to use it")
    tmp.replace(dest)
    return dest


__all__ = ["PINNED", "VENDORED_DIR", "models_dir", "resolve_model", "fetch", "sha256_of"]
