"""Light-ASD (Liao et al., CVPR 2023, "A Light Weight Model for Active Speaker Detection") on CPU or GPU.

Model code ported from https://github.com/Junhua-Liao/Light-ASD (MIT, Copyright (c) 2023 Liao Junhua; commit
ed38c232, 2025-03-23): model/Encoder.py, model/Classifier.py, model/Model.py and the FC head of loss.py. 1.03 M
parameters. Inputs per clip: grayscale face crops (T, 112, 112) at 25 fps and 13-dim MFCC (4T, 13) at 100 Hz
(python_speech_features, 25 ms window, 10 ms step, as in the authors' Columbia_test.py). Output: one logit pair
per video frame; P(speaking) = softmax[:, 1].

Weights (same repo, `weight/`, MIT): `pretrain_AVA_CVPR.model` trained on AVA-ActiveSpeaker (the paper's model,
94.1 mAP on AVA val) and `finetuning_TalkSet.model` (fine-tuned on TalkSet, the authors' choice for in-the-wild
video). Both are fetched once, sha256-checked, into $UNSPOKEN_MODELS_DIR (default ~/.cache/unspoken/models):
    python -m unspoken.transcribe.asd_lightasd fetch [--all]
AVA-ActiveSpeaker itself is CC BY 4.0 annotations over YouTube films; the released weights carry the repo's MIT
licence. TalkSet is built from VoxCeleb2 + LRS3 (research licences): prefer the AVA weights for anything shipped.

Inference follows Columbia_test.py: a track is cut into clips of `durations` seconds (the authors average the
scores of durations {1,2,3,4,5,6}; each is one pass over the track), the logits are averaged over durations and
the softmax taken last. Clips of one duration are batched. Everything runs on `device` (cpu / cuda).
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

REPO_RAW = "https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/ed38c232de5efe0261dbd68627c0ade7cdfe14eb/weight/"
WEIGHTS: dict[str, tuple[str, str]] = {   # name -> (url, sha256)
    "lightasd_pretrain_AVA_CVPR.model": (
        REPO_RAW + "pretrain_AVA_CVPR.model", "d44bc3ea7baa8e0946fa3921311714a630ed8b90a1928fab0dbe30d918909317"),
    "lightasd_finetuning_TalkSet.model": (
        REPO_RAW + "finetuning_TalkSet.model", "efc375833887eefa9d209dc92810e18519b04c3c73ea35a549f2a7f40b7d94d5"),
}
DEFAULT_WEIGHTS = "lightasd_pretrain_AVA_CVPR.model"
VIDEO_FPS = 25          # the model's frame rate
AUDIO_HZ = 100          # MFCC frames per second (10 ms step)
CROP = 112
DEFAULT_DURATIONS = (1, 2, 3, 4, 5, 6)


# ----------------------------------------------------------------------------------------------- weights
def models_dir() -> Path:
    env = os.environ.get("UNSPOKEN_MODELS_DIR")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "unspoken" / "models"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_weights(name: str = DEFAULT_WEIGHTS) -> Path:
    """Existing weight file (models dir, then any path given as `name`), else the download target path."""
    p = Path(name).expanduser()
    if p.exists():
        return p
    return models_dir() / name


def fetch_weights(name: str = DEFAULT_WEIGHTS, dest_dir: Path | None = None, force: bool = False) -> Path:
    url, digest = WEIGHTS[name]
    dest_dir = dest_dir or models_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if dest.exists() and not force and sha256_of(dest) == digest:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:  # noqa: S310 (pinned https URL)
        for block in iter(lambda: r.read(1 << 20), b""):
            f.write(block)
    got = sha256_of(tmp)
    if got != digest:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: sha256 mismatch (got {got}, want {digest}); refusing to use it")
    tmp.replace(dest)
    return dest


# ----------------------------------------------------------------------------------------------- features
def mfcc_features(audio_int16: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """(N, 13) float32 MFCC at 100 Hz exactly as the authors compute it (python_speech_features defaults
    except numcep=13, winlen=0.025, winstep=0.010). Pads/trims to AUDIO_HZ * seconds rows."""
    import python_speech_features

    x = np.asarray(audio_int16)
    if x.dtype != np.int16:
        x = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    if x.ndim > 1:
        x = x[:, 0]
    want = int(round(len(x) / sample_rate * AUDIO_HZ))
    if len(x) < int(0.025 * sample_rate) + 1:
        return np.zeros((want, 13), dtype=np.float32)
    m = python_speech_features.mfcc(x, sample_rate, numcep=13, winlen=0.025, winstep=0.010).astype(np.float32)
    if m.shape[0] < want:
        m = np.pad(m, ((0, want - m.shape[0]), (0, 0)), mode="edge")
    return m[:want]


# ----------------------------------------------------------------------------------------------- model
def build_model():
    """The network (torch imported lazily so the rest of the package works without it)."""
    import torch
    import torch.nn as nn

    class AudioBlock(nn.Module):
        def __init__(self, cin: int, cout: int) -> None:
            super().__init__()
            bn = lambda: nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)  # noqa: E731
            self.relu = nn.ReLU()
            self.m_3 = nn.Conv2d(cin, cout, (3, 1), padding=(1, 0), bias=False); self.bn_m_3 = bn()
            self.t_3 = nn.Conv2d(cout, cout, (1, 3), padding=(0, 1), bias=False); self.bn_t_3 = bn()
            self.m_5 = nn.Conv2d(cin, cout, (5, 1), padding=(2, 0), bias=False); self.bn_m_5 = bn()
            self.t_5 = nn.Conv2d(cout, cout, (1, 5), padding=(0, 2), bias=False); self.bn_t_5 = bn()
            self.last = nn.Conv2d(cout, cout, 1, bias=False); self.bn_last = bn()

        def forward(self, x):
            x3 = self.relu(self.bn_t_3(self.t_3(self.relu(self.bn_m_3(self.m_3(x))))))
            x5 = self.relu(self.bn_t_5(self.t_5(self.relu(self.bn_m_5(self.m_5(x))))))
            return self.relu(self.bn_last(self.last(x3 + x5)))

    class VisualBlock(nn.Module):
        def __init__(self, cin: int, cout: int, is_down: bool = False) -> None:
            super().__init__()
            bn = lambda: nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)  # noqa: E731
            st = (1, 2, 2) if is_down else (1, 1, 1)
            self.relu = nn.ReLU()
            self.s_3 = nn.Conv3d(cin, cout, (1, 3, 3), stride=st, padding=(0, 1, 1), bias=False); self.bn_s_3 = bn()
            self.t_3 = nn.Conv3d(cout, cout, (3, 1, 1), padding=(1, 0, 0), bias=False); self.bn_t_3 = bn()
            self.s_5 = nn.Conv3d(cin, cout, (1, 5, 5), stride=st, padding=(0, 2, 2), bias=False); self.bn_s_5 = bn()
            self.t_5 = nn.Conv3d(cout, cout, (5, 1, 1), padding=(2, 0, 0), bias=False); self.bn_t_5 = bn()
            self.last = nn.Conv3d(cout, cout, 1, bias=False); self.bn_last = bn()

        def forward(self, x):
            x3 = self.relu(self.bn_t_3(self.t_3(self.relu(self.bn_s_3(self.s_3(x))))))
            x5 = self.relu(self.bn_t_5(self.t_5(self.relu(self.bn_s_5(self.s_5(x))))))
            return self.relu(self.bn_last(self.last(x3 + x5)))

    class VisualEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block1 = VisualBlock(1, 32, is_down=True)
            self.pool1 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            self.block2 = VisualBlock(32, 64)
            self.pool2 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            self.block3 = VisualBlock(64, 128)
            self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        def forward(self, x):                      # (B, 1, T, H, W) -> (B, T, 128)
            x = self.block3(self.pool2(self.block2(self.pool1(self.block1(x)))))
            x = x.transpose(1, 2)
            B, T, C, H, W = x.shape
            return self.maxpool(x.reshape(B * T, C, H, W)).view(B, T, C)

    class AudioEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block1 = AudioBlock(1, 32)
            self.pool1 = nn.MaxPool3d((1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
            self.block2 = AudioBlock(32, 64)
            self.pool2 = nn.MaxPool3d((1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
            self.block3 = AudioBlock(64, 128)

        def forward(self, x):                      # (B, 1, 13, 4T) -> (B, T, 128)
            x = self.block3(self.pool2(self.block2(self.pool1(self.block1(x)))))
            return torch.mean(x, dim=2, keepdim=True).squeeze(2).transpose(1, 2)

    class BGRU(nn.Module):
        def __init__(self, channel: int) -> None:
            super().__init__()
            self.gru_forward = nn.GRU(channel, channel, num_layers=1, batch_first=True)
            self.gru_backward = nn.GRU(channel, channel, num_layers=1, batch_first=True)
            self.gelu = nn.GELU()

        def forward(self, x):
            x, _ = self.gru_forward(x)
            x = torch.flip(self.gelu(x), dims=[1])
            x, _ = self.gru_backward(x)
            return self.gelu(torch.flip(x, dims=[1]))

    class ASDModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visualEncoder = VisualEncoder()
            self.audioEncoder = AudioEncoder()
            self.GRU = BGRU(128)

        def forward_visual_frontend(self, x):      # (B, T, H, W) uint8-range floats
            B, T, H, W = x.shape
            x = (x.view(B, 1, T, H, W) / 255 - 0.4161) / 0.1688
            return self.visualEncoder(x)

        def forward_audio_frontend(self, x):       # (B, 4T, 13)
            return self.audioEncoder(x.unsqueeze(1).transpose(2, 3))

        def forward_audio_visual_backend(self, a, v):
            return self.GRU(a + v)                 # (B, T, 128)

    class LightASD(nn.Module):
        """ASD_Model + the lossAV FC head; `forward` returns logits (B, T, 2)."""

        def __init__(self) -> None:
            super().__init__()
            self.model = ASDModel()
            self.fc = nn.Linear(128, 2)

        def forward(self, audio, video):
            a = self.model.forward_audio_frontend(audio)
            v = self.model.forward_visual_frontend(video)
            return self.fc(self.model.forward_audio_visual_backend(a, v))

        def load_repo_weights(self, path: str | Path) -> None:
            sd = torch.load(str(path), map_location="cpu", weights_only=True)
            out = {}
            for k, v in sd.items():
                k = k.replace("module.", "")
                if k.startswith("lossAV.FC."):
                    out["fc." + k[len("lossAV.FC."):]] = v
                elif k.startswith("model."):
                    out[k] = v
            missing, unexpected = self.load_state_dict(out, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"Light-ASD weights do not match the model: missing {missing}, unexpected {unexpected}")

    return LightASD()


class LightASDScorer:
    """Scores (T, 112, 112) crop sequences against (4T, 13) MFCC rows: P(speaking) per frame."""

    def __init__(self, weights: str | Path | None = None, *, device: str = "cpu", threads: int = 6,
                 durations: tuple[int, ...] = DEFAULT_DURATIONS, batch: int = 1, model: Any = None) -> None:
        # batch=1 (one clip per forward) measured 3x faster than batch 8 on the shared x86 CPU (conv3d path);
        # on a GPU raise it (16-32).
        import torch

        torch.set_num_threads(int(threads))
        self.torch = torch
        self.device = device
        self.durations = tuple(int(d) for d in durations) or (1,)
        self.batch = max(1, int(batch))
        self.model = model if model is not None else build_model()
        self.weights = None
        if model is None:
            p = resolve_weights(str(weights) if weights else DEFAULT_WEIGHTS)
            if not p.exists():
                raise FileNotFoundError(f"Light-ASD weights not found at {p}; run "
                                        f"`python -m unspoken.transcribe.asd_lightasd fetch` (or set UNSPOKEN_MODELS_DIR)")
            self.model.load_repo_weights(p)
            self.weights = p
        self.model.to(device).eval()
        self.label = f"Light-ASD ({Path(self.weights).name if self.weights else 'untrained'}, durations {','.join(map(str, self.durations))}, {device})"

    def logits(self, crops: np.ndarray, mfcc: np.ndarray, duration: int) -> np.ndarray:
        """Mean class-1-minus-class-0 logit per frame for one clip duration (the softmax is taken by `score`)."""
        torch = self.torch
        T = crops.shape[0]
        n = int(math.ceil(T / (duration * VIDEO_FPS)))
        out = np.zeros(T, dtype=np.float32)
        clips: list[tuple[int, int]] = [(i * duration * VIDEO_FPS, min(T, (i + 1) * duration * VIDEO_FPS)) for i in range(n)]
        # batch clips of equal length (all but the last are full length)
        groups: dict[int, list[tuple[int, int]]] = {}
        for a, b in clips:
            groups.setdefault(b - a, []).append((a, b))
        with torch.no_grad():
            for length, items in groups.items():
                for s in range(0, len(items), self.batch):
                    chunk = items[s:s + self.batch]
                    v = torch.from_numpy(np.stack([crops[a:b] for a, b in chunk]).astype(np.float32)).to(self.device)
                    a_ = torch.from_numpy(np.stack([mfcc[a * 4:b * 4] for a, b in chunk]).astype(np.float32)).to(self.device)
                    lg = self.model(a_, v).float().cpu().numpy()          # (B, length, 2)
                    for (a, b), l in zip(chunk, lg):
                        out[a:b] = l[:, 1] - l[:, 0]
        return out

    def score(self, crops: np.ndarray, mfcc: np.ndarray) -> np.ndarray:
        """(T,) P(speaking). `mfcc` must have at least 4T rows (extra rows are ignored, short input is edge-padded)."""
        T = int(crops.shape[0])
        if T == 0:
            return np.zeros(0, dtype=np.float32)
        if mfcc.shape[0] < 4 * T:
            mfcc = np.pad(mfcc, ((0, 4 * T - mfcc.shape[0]), (0, 0)), mode="edge") if mfcc.shape[0] else \
                np.zeros((4 * T, 13), dtype=np.float32)
        mfcc = mfcc[:4 * T]
        acc = np.zeros(T, dtype=np.float32)
        for d in self.durations:
            acc += self.logits(crops, mfcc, d)
        acc /= len(self.durations)
        return (1.0 / (1.0 + np.exp(-acc))).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the pinned weights (sha256-checked)")
    f.add_argument("names", nargs="*", help=f"default: {DEFAULT_WEIGHTS}; known: {', '.join(WEIGHTS)}")
    f.add_argument("--all", action="store_true")
    f.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    names = list(WEIGHTS) if a.all else (a.names or [DEFAULT_WEIGHTS])
    rc = 0
    for name in names:
        if name not in WEIGHTS:
            print(f"unknown weights {name}; known: {', '.join(WEIGHTS)}", file=sys.stderr)
            rc = 2
            continue
        p = fetch_weights(name, force=a.force)
        print(f"ok  {p} ({p.stat().st_size / 1e6:.1f} MB)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
