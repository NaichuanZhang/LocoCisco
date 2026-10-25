"""Who spoke when, from audio only -> DiarResult (unspoken/transcribe/contracts.py). Workstream tx-diar.

    python -m unspoken.transcribe.diar --in CASE/audio.wav --out diar.json --method window [--online]

Methods (all local, CPU; every model file is Apache-2.0 / MIT, see docs/reports/tx-diar.md):
  window      Silero VAD speech regions -> sliding windows (1.5 s / 0.75 s hop) -> WeSpeaker embedding
              (CAM++ 512-d or ResNet34-LM 256-d, ONNX) -> offline clustering with automatic speaker count
              (AHC on cosine distance, or spectral with eigengap) -> resegmentation with shorter windows against
              the cluster centroids -> frame labels -> voice turns. Splits a long VAD segment at speaker changes.
  window --online
              Same front end, but causal: each window is labelled when its audio is complete, against running
              centroids of the voices heard so far (new voice when nothing is close, `max_voices` cap). Labels
              never change afterwards. Label latency = window + lookback * hop (+ compute), reported in timing.
  pyannote    sherpa-onnx OfflineSpeakerDiarization: pyannote/segmentation-3.0 (neural, overlap-aware local
              segmentation, MIT) + WeSpeaker embedding + sherpa's agglomerative clustering (threshold or count).
  sortformer  NVIDIA Streaming Sortformer (NeMo, CC-BY-4.0), see diar_sortformer.py; needs nemo_toolkit.

Only the audio is read. Nothing is written except the requested JSON. Labels are session-only (voice-A, voice-B,
...), in order of first appearance. Times are seconds from the start of the file (= video start for the AMI cases).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .contracts import DiarResult, VoiceTurn, to_json

SR = 16000
FRAME = 0.01                       # label frame (s); also the fbank hop
DEV_MODELS = Path("/local/home/zhangnai/kirocrew-sessions/2026-10-23-dell-nvidia-hackathon/dev-models")
EMBEDDERS = {"campp": "wespeaker_en_voxceleb_CAM++.onnx", "resnet34": "wespeaker_en_voxceleb_resnet34_LM.onnx"}
PYANNOTE_SEG = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"


def voice_name(index: int) -> str:
    """0 -> voice-A ... 25 -> voice-Z, 26 -> voice-AA ..."""
    from ..sensors.diar import voice_name as vn
    return vn(index)


# ----------------------------------------------------------------------------- audio
def read_wav(path: str | Path) -> np.ndarray:
    """16 kHz mono float32 in -1..1. Anything else is decoded with PyAV (sensors dep) and resampled."""
    with wave.open(str(path), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        if sr == SR and ch == 1 and sw == 2:
            return np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    import av  # lazy

    with av.open(str(path)) as c:
        rs = av.AudioResampler(format="s16", layout="mono", rate=SR)
        out = [np.frombuffer(f.to_ndarray().tobytes(), dtype="<i2") for fr in c.decode(audio=0) for f in rs.resample(fr)]
    return np.concatenate(out).astype(np.float32) / 32768.0 if out else np.zeros(0, np.float32)


def models_dir(ws: str = "sensors") -> Path:
    env = os.environ.get("UNSPOKEN_MODELS_DIR")
    if env and ws == "sensors":
        return Path(env).expanduser()
    return DEV_MODELS / ws


# ----------------------------------------------------------------------------- VAD
class SileroProbs:
    """Per-32 ms speech probability over a whole signal (stateful Silero v5 ONNX, same file the live VAD uses)."""

    def __init__(self, model_path: Path | None = None, threads: int = 6) -> None:
        from ..sensors.vad import SILERO_ONNX
        self.model_path = Path(model_path or (models_dir() / "silero_vad.onnx"))
        if not self.model_path.exists():
            self.model_path = SILERO_ONNX
        self.threads = threads
        self._sess = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        import onnxruntime as ort
        from ..sensors.vad import CONTEXT, WINDOW

        if self._sess is None:
            so = ort.SessionOptions()
            so.intra_op_num_threads, so.inter_op_num_threads, so.log_severity_level = self.threads, 1, 3
            self._sess = ort.InferenceSession(str(self.model_path), sess_options=so, providers=["CPUExecutionProvider"])
        n = int(np.ceil(len(x) / WINDOW))
        pad = np.zeros(n * WINDOW, np.float32)
        pad[:len(x)] = x
        state = np.zeros((2, 1, 128), np.float32)
        ctx = np.zeros((1, CONTEXT), np.float32)
        sr = np.array(SR, dtype=np.int64)
        probs = np.zeros(n, np.float32)
        for i in range(n):
            w = pad[i * WINDOW:(i + 1) * WINDOW][None, :]
            inp = np.concatenate([ctx, w], axis=1)
            out, state = self._sess.run(None, {"input": inp, "state": state, "sr": sr})
            ctx = inp[:, -CONTEXT:]
            probs[i] = out[0, 0]
        return probs


def energy_probs(x: np.ndarray, window: int = 512) -> np.ndarray:
    """Model-free stand-in for tests: RMS per window scaled against the 5th-percentile noise floor."""
    n = int(np.ceil(len(x) / window))
    pad = np.zeros(n * window, np.float32)
    pad[:len(x)] = x
    rms = np.sqrt((pad.reshape(n, window) ** 2).mean(axis=1)) + 1e-9
    db = 20 * np.log10(rms)
    floor = np.percentile(db, 5)
    return np.clip((db - floor) / 12.0, 0.0, 1.0).astype(np.float32)


def speech_regions(probs: np.ndarray, step_s: float = 512 / SR, threshold: float = 0.5, hysteresis: float = 0.15,
                   min_speech_s: float = 0.2, min_silence_s: float = 0.3, pad_s: float = 0.1) -> list[tuple[float, float]]:
    """Hysteresis segmenter over per-step probabilities -> [(start, end)] in seconds, padded, non-overlapping."""
    regions: list[tuple[float, float]] = []
    active, start, run_sil = False, 0.0, 0
    min_sil = max(1, int(round(min_silence_s / step_s)))
    for i, p in enumerate(probs):
        t = i * step_s
        if not active:
            if p >= threshold:
                active, start, run_sil = True, t, 0
        else:
            if p < threshold - hysteresis:
                run_sil += 1
                if run_sil >= min_sil:
                    end = t - (run_sil - 1) * step_s
                    if end - start >= min_speech_s:
                        regions.append((start, end))
                    active = False
            elif p >= threshold:
                run_sil = 0
    if active:
        end = len(probs) * step_s - run_sil * step_s
        if end - start >= min_speech_s:
            regions.append((start, end))
    total = len(probs) * step_s
    out: list[tuple[float, float]] = []
    for a, b in regions:
        a, b = max(0.0, a - pad_s), min(total, b + pad_s)
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


# ----------------------------------------------------------------------------- windows + embeddings
@dataclass
class Window:
    start: float
    end: float
    region: int                        # index of the speech region it came from


def sliding_windows(regions: Sequence[tuple[float, float]], win_s: float = 1.5, hop_s: float = 0.75,
                    min_s: float = 0.3) -> list[Window]:
    """Windows of win_s every hop_s inside each region; a region shorter than win_s is one window (>= min_s)."""
    out: list[Window] = []
    for r, (a, b) in enumerate(regions):
        if b - a < min_s:
            continue
        if b - a <= win_s:
            out.append(Window(a, b, r))
            continue
        t = a
        while t + win_s <= b + 1e-6:
            out.append(Window(t, t + win_s, r))
            t += hop_s
        if out[-1].end < b - 1e-3:                   # cover the tail so every speech frame has a window
            out.append(Window(b - win_s, b, r))
    return out


def whole_fbank(x: np.ndarray) -> np.ndarray:
    """[T, 80] Kaldi fbank of the whole signal (no CMN; applied per window)."""
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = SR
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = 80
    opts.mel_opts.debug_mel = False
    fb = knf.OnlineFbank(opts)
    step = SR * 60
    for i in range(0, len(x), step):
        fb.accept_waveform(SR, (x[i:i + step] * 32768.0).astype(np.float32).tolist())
    fb.input_finished()
    n = fb.num_frames_ready
    return np.stack([fb.get_frame(i) for i in range(n)]).astype(np.float32) if n else np.zeros((0, 80), np.float32)


class OnnxEmbedder:
    """WeSpeaker ONNX speaker embedding (unit norm) for windows of one signal, batched by equal frame count."""

    def __init__(self, name: str = "campp", model_path: Path | None = None, threads: int = 6) -> None:
        self.name = name
        self.model_path = Path(model_path or (models_dir() / EMBEDDERS[name]))
        self.threads = threads
        self._sess = None
        self.label = f"WeSpeaker {self.model_path.stem} (onnxruntime CPU)"

    def _ort(self):
        if self._sess is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads, so.inter_op_num_threads, so.log_severity_level = self.threads, 1, 3
            self._sess = ort.InferenceSession(str(self.model_path), sess_options=so, providers=["CPUExecutionProvider"])
        return self._sess

    def embed_windows(self, x: np.ndarray, windows: Sequence[Window], fbank: np.ndarray | None = None) -> np.ndarray:
        fb = whole_fbank(x) if fbank is None else fbank
        sess = self._ort()
        name = sess.get_inputs()[0].name
        dim = int(sess.get_outputs()[0].shape[-1]) if isinstance(sess.get_outputs()[0].shape[-1], int) else 512
        out = np.zeros((len(windows), dim), np.float32)
        slices: dict[int, list[int]] = {}
        feats: list[np.ndarray] = []
        for i, w in enumerate(windows):
            a, b = int(round(w.start / FRAME)), int(round(w.end / FRAME)) - 2
            f = fb[max(0, a):max(0, b)]
            if len(f) < 10:
                f = np.zeros((10, 80), np.float32)
            f = f - f.mean(axis=0, keepdims=True)
            feats.append(f)
            slices.setdefault(len(f), []).append(i)
        for n, idx in slices.items():
            for j in range(0, len(idx), 64):
                chunk = idx[j:j + 64]
                batch = np.stack([feats[i] for i in chunk])
                e = sess.run(None, {name: batch})[0].astype(np.float32)
                out[chunk] = e
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


# ----------------------------------------------------------------------------- clustering
def cluster_ahc(emb: np.ndarray, threshold: float = 0.45, num_speakers: int | None = None, max_speakers: int = 8,
                linkage: str = "average") -> np.ndarray:
    """Agglomerative clustering on cosine distance; cut at `threshold` (auto count) unless num_speakers is given."""
    from scipy.cluster.hierarchy import fcluster, linkage as _linkage
    from scipy.spatial.distance import pdist

    n = len(emb)
    if n == 0:
        return np.zeros(0, int)
    if n == 1:
        return np.zeros(1, int)
    d = np.clip(pdist(emb, "cosine"), 0.0, 2.0)
    Z = _linkage(d, linkage)
    if num_speakers:
        lab = fcluster(Z, min(num_speakers, n), "maxclust")
    else:
        lab = fcluster(Z, threshold, "distance")
        if lab.max() > max_speakers:
            lab = fcluster(Z, max_speakers, "maxclust")
    return _relabel_by_first(lab - 1)


def cluster_spectral(emb: np.ndarray, num_speakers: int | None = None, max_speakers: int = 8, max_rp: float = 0.25,
                     sparse_search: int = 30, seed: int = 0) -> np.ndarray:
    """Spectral clustering with the speaker count from the eigengap, NME-SC style (Park et al., 2019,
    https://arxiv.org/abs/2003.02405): cosine affinity -> for each candidate neighbour count p, a binarised
    symmetric kNN graph, unnormalised Laplacian eigenvalues, eigengap -> keep the p with the smallest
    (p/n) / normalised-max-gap ratio -> k = argmax gap at that p -> k-means on the first k eigenvectors of the
    normalised Laplacian of that graph."""
    from scipy.cluster.vq import kmeans2

    n = len(emb)
    if n <= 2:
        return np.zeros(n, int)
    S = np.clip(emb @ emb.T, -1.0, 1.0)
    np.fill_diagonal(S, 1.0)
    kmax = max(1, min(max_speakers, n - 1))
    p_max = max(int(round(n * max_rp)), min(n - 1, 10))        # small n: still allow up to 10 neighbours
    cands = np.unique(np.linspace(2, min(p_max, n - 1), sparse_search).astype(int))
    best = None
    for p in cands:
        idx = np.argsort(-S, axis=1)[:, :p]                       # top-p per row (self included)
        A = np.zeros((n, n))
        np.put_along_axis(A, idx, 1.0, axis=1)
        A = 0.5 * (A + A.T)
        L = np.diag(A.sum(axis=1)) - A
        vals = np.linalg.eigvalsh(L)
        gaps = np.diff(vals[:kmax + 1])
        if not len(gaps) or gaps.max() <= 1e-9 or vals[min(kmax, n - 1)] < 1e-8:
            continue                                               # graph in more pieces than max_speakers: too sparse
        k_hat = int(np.argmax(gaps)) + 1
        g = gaps.max() / (vals.max() + 1e-9)
        ratio = (p / n) / max(g, 1e-9)
        if best is None or ratio < best[0]:
            best = (ratio, k_hat, A)
    if best is None:
        return np.zeros(n, int)
    _, k_hat, A = best
    k = min(num_speakers, n) if num_speakers else k_hat
    if k <= 1:
        return np.zeros(n, int)
    deg = A.sum(axis=1)
    Dm = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    Lsym = np.eye(n) - (Dm[:, None] * A * Dm[None, :])
    _, vecs = np.linalg.eigh(Lsym)
    X = vecs[:, :k]
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    rng = np.random.default_rng(seed)
    best_lab, best_inertia = np.zeros(n, int), None
    for _ in range(5):
        cents, lab = kmeans2(X, k, minit="++", seed=rng.integers(1 << 30))
        inertia = float(((X - cents[lab]) ** 2).sum())
        if best_inertia is None or inertia < best_inertia:
            best_inertia, best_lab = inertia, lab
    return _relabel_by_first(best_lab)


def _relabel_by_first(lab: np.ndarray) -> np.ndarray:
    """Dense labels 0..k-1 in order of first appearance."""
    order: dict[int, int] = {}
    out = np.zeros(len(lab), int)
    for i, l in enumerate(lab):
        out[i] = order.setdefault(int(l), len(order))
    return out


def centroids_of(emb: np.ndarray, lab: np.ndarray) -> np.ndarray:
    k = int(lab.max()) + 1 if len(lab) else 0
    c = np.stack([emb[lab == i].mean(axis=0) for i in range(k)]) if k else np.zeros((0, emb.shape[1]))
    return c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-9)


def absorb_small_clusters(emb: np.ndarray, lab: np.ndarray, windows: Sequence[Window], min_s: float) -> np.ndarray:
    """A cluster with less than `min_s` seconds of windows is re-assigned to the nearest big centroid."""
    if len(lab) == 0:
        return lab
    k = int(lab.max()) + 1
    dur = np.array([sum(w.end - w.start for w, l in zip(windows, lab) if l == i) for i in range(k)])
    big = [i for i in range(k) if dur[i] >= min_s] or [int(np.argmax(dur))]
    if len(big) == k:
        return lab
    cents = centroids_of(emb, lab)[big]
    out = lab.copy()
    for i in range(k):
        if i not in big:
            m = lab == i
            out[m] = np.array(big)[np.argmax(emb[m] @ cents.T, axis=1)]
    return _relabel_by_first(out)


# ----------------------------------------------------------------------------- labels -> turns
def frame_labels(windows: Sequence[Window], lab: np.ndarray, regions: Sequence[tuple[float, float]],
                 total_s: float) -> np.ndarray:
    """Per 10 ms frame: the label of the window whose centre is nearest (−1 = no speech)."""
    n = int(np.ceil(total_s / FRAME)) + 1
    out = np.full(n, -1, int)
    dist = np.full(n, np.inf)
    for w, l in zip(windows, lab):
        a, b = int(round(w.start / FRAME)), int(round(w.end / FRAME))
        c = (a + b) / 2
        idx = np.arange(max(0, a), min(n, b))
        d = np.abs(idx - c)
        m = d < dist[idx]
        out[idx[m]] = l
        dist[idx[m]] = d[m]
    speech = np.zeros(n, bool)
    for a, b in regions:
        speech[int(round(a / FRAME)):int(round(b / FRAME))] = True
    out[~speech] = -1
    holes = np.flatnonzero(speech & (out < 0))
    if len(holes) and len(windows):
        centers = np.array([(w.start + w.end) / 2 / FRAME for w in windows])
        order = np.argsort(centers)
        pos = np.clip(np.searchsorted(centers[order], holes), 1, len(order) - 1)
        left, right = order[pos - 1], order[pos]
        nearest = np.where(np.abs(centers[left] - holes) <= np.abs(centers[right] - holes), left, right)
        out[holes] = np.asarray(lab)[nearest]
    return out


def _runs(fl: np.ndarray) -> list[list[int]]:
    """Run-length encoding: [[label, start_frame, end_frame], ...]."""
    runs: list[list[int]] = []
    for i, l in enumerate(fl):
        if runs and runs[-1][0] == l:
            runs[-1][2] = i + 1
        else:
            runs.append([int(l), i, i + 1])
    return runs


def smooth_labels(fl: np.ndarray, min_turn_s: float = 0.25) -> np.ndarray:
    """A speech run shorter than min_turn_s that touches another speech run takes the label of the longer
    speech neighbour (repeated until stable). Short runs surrounded by silence are kept: they are real
    backchannels, and the collar forgives their edges."""
    min_f = int(round(min_turn_s / FRAME))
    runs = _runs(fl)
    changed = True
    while changed:
        changed = False
        for k, (lab, a, b) in enumerate(runs):
            if lab < 0 or b - a >= min_f:
                continue
            left = runs[k - 1] if k > 0 and runs[k - 1][0] >= 0 else None
            right = runs[k + 1] if k + 1 < len(runs) and runs[k + 1][0] >= 0 else None
            if left is None and right is None:
                continue
            pick = left if right is None or (left is not None and left[2] - left[1] >= right[2] - right[1]) else right
            runs[k][0] = pick[0]
            changed = True
            break
        if changed:
            merged: list[list[int]] = []
            for r in runs:
                if merged and merged[-1][0] == r[0]:
                    merged[-1][2] = r[2]
                else:
                    merged.append(r)
            runs = merged
    out = fl.copy()
    for lab, a, b in runs:
        out[a:b] = lab
    return out


def labels_to_turns(fl: np.ndarray, conf: np.ndarray | None = None) -> list[VoiceTurn]:
    turns: list[VoiceTurn] = []
    n = len(fl)
    i = 0
    while i < n:
        j = i
        while j < n and fl[j] == fl[i]:
            j += 1
        if fl[i] >= 0:
            c = float(np.mean(conf[i:j])) if conf is not None else 1.0
            turns.append(VoiceTurn(round(i * FRAME, 3), round(j * FRAME, 3), voice_name(int(fl[i])), round(c, 3)))
        i = j
    return turns


def rename_by_first_appearance(turns: list[VoiceTurn]) -> list[VoiceTurn]:
    order: dict[str, str] = {}
    out = []
    for t in sorted(turns, key=lambda t: (t.start, t.end)):
        v = order.setdefault(t.voice, voice_name(len(order)))
        out.append(VoiceTurn(t.start, t.end, v, t.conf))
    return out


# ----------------------------------------------------------------------------- online labelling
class OnlineLabeler:
    """Causal voice labelling: cosine similarity to running centroids (mean of all embeddings of the voice).
    sim >= threshold: join and update the centroid. mint_threshold <= sim < threshold: assign to the nearest voice
    with low confidence but leave its centroid alone (keeps centroids pure). sim < mint_threshold: new voice until
    max_voices. `lookback` delays each decision by that many windows and takes the majority of the raw labels over
    the lookback span (latency = window + lookback * hop)."""

    def __init__(self, threshold: float = 0.6, max_voices: int = 8, lookback: int = 0,
                 mint_threshold: float | None = None) -> None:
        self.threshold, self.max_voices, self.lookback = threshold, max_voices, lookback
        self.mint_threshold = threshold if mint_threshold is None else min(mint_threshold, threshold)
        self.sums: list[np.ndarray] = []
        self.counts: list[int] = []
        self.raw: list[int] = []
        self.confs: list[float] = []

    def push(self, emb: np.ndarray) -> tuple[int, float]:
        if not self.sums:
            self.sums.append(emb.copy()); self.counts.append(1)
            self.raw.append(0); self.confs.append(1.0)
            return 0, 1.0
        cents = np.stack(self.sums)
        cents = cents / np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-9)
        sims = cents @ emb
        best = int(np.argmax(sims))
        runner = float(np.partition(sims, -2)[-2]) if len(sims) > 1 else self.threshold - 0.3
        if sims[best] >= self.threshold or len(self.sums) >= self.max_voices:
            self.sums[best] += emb; self.counts[best] += 1
            conf = float(np.clip(0.5 + max(sims[best] - max(runner, self.threshold), 0.0), 0.0, 1.0))
            self.raw.append(best); self.confs.append(conf)
            return best, conf
        if sims[best] >= self.mint_threshold:
            conf = float(np.clip(0.25 + (sims[best] - self.mint_threshold), 0.0, 0.5))
            self.raw.append(best); self.confs.append(conf)
            return best, conf
        self.sums.append(emb.copy()); self.counts.append(1)
        conf = float(np.clip(0.5 + (self.mint_threshold - sims[best]), 0.0, 1.0))
        self.raw.append(len(self.sums) - 1); self.confs.append(conf)
        return len(self.sums) - 1, conf

    def decided(self, i: int) -> tuple[int, float]:
        """Final label of window i once windows i..i+lookback have been pushed (majority vote)."""
        span = self.raw[i:i + self.lookback + 1]
        vals, counts = np.unique(span, return_counts=True)
        return int(vals[np.argmax(counts)]), float(np.mean(self.confs[i:i + self.lookback + 1]))


# ----------------------------------------------------------------------------- pipeline
@dataclass
class WindowSettings:
    win_s: float = 1.5
    hop_s: float = 0.75
    reseg_win_s: float = 1.0
    reseg_hop_s: float = 0.25
    threshold: float = 0.45           # AHC cosine-distance cut (offline) / 1 - online similarity threshold
    cluster: str = "spectral"         # "spectral" (NME-SC, auto count; dev-best) | "ahc" (cosine-distance cut)
    num_speakers: int | None = None
    max_speakers: int = 8
    min_cluster_s: float = 2.0
    min_turn_s: float = 0.25
    reseg: bool = True
    vad_threshold: float = 0.5
    vad_min_silence_s: float = 0.3
    vad_pad_s: float = 0.1
    overlap_second: bool = False      # offline: emit a 2nd voice where the neural segmenter heard overlap
    min_overlap_s: float = 0.1
    lookback: int = 0                 # online only
    online_threshold: float = 0.6     # online cosine similarity to join a voice (and update its centroid)
    online_mint_threshold: float | None = None   # below this a new voice is minted (default: = online_threshold)


@dataclass
class Prepared:
    """Front-end output that does not depend on clustering settings (cached by the tuner)."""
    total_s: float
    regions: list[tuple[float, float]]
    windows: list[Window]
    emb: np.ndarray
    reseg_windows: list[Window]
    reseg_emb: np.ndarray
    timing: dict[str, float]
    label: str
    overlap: np.ndarray | None = None      # per 10 ms frame: >= 2 people talking (from a neural segmenter), if known


def pyannote_activity(x: np.ndarray, threads: int = 6, seg_model: Path | None = None,
                      embedder_name: str = "campp") -> tuple[list[tuple[float, float]], np.ndarray]:
    """Speech regions + overlap frames from pyannote/segmentation-3.0 (sherpa-onnx), identities ignored:
    the segmenter is overlap-aware (up to 3 concurrent speakers per 10 s chunk), the VAD is not."""
    res = diarize_pyannote(x, embedder_name, 0.5, None, threads, seg_model, min_on=0.1, min_off=0.1)
    n = int(np.ceil(len(x) / SR / FRAME)) + 1
    count = np.zeros(n, np.int16)
    for t in res.turns:
        count[int(round(t.start / FRAME)):int(round(t.end / FRAME))] += 1
    speech = count >= 1
    regions: list[tuple[float, float]] = []
    i = 0
    while i < n:
        if speech[i]:
            j = i
            while j < n and speech[j]:
                j += 1
            regions.append((round(i * FRAME, 3), round(j * FRAME, 3)))
            i = j
        else:
            i += 1
    return regions, count >= 2


def prepare_windows(x: np.ndarray, probs_fn: Callable[[np.ndarray], np.ndarray] | None, embedder, s: WindowSettings,
                    with_reseg: bool = True, activity: tuple[list[tuple[float, float]], np.ndarray] | None = None) -> Prepared:
    t0 = time.perf_counter()
    overlap = None
    if activity is not None:
        regions, overlap = activity
    else:
        probs = probs_fn(x)
        regions = speech_regions(probs, threshold=s.vad_threshold, min_silence_s=s.vad_min_silence_s, pad_s=s.vad_pad_s)
    t1 = time.perf_counter()
    total_s = len(x) / SR
    fb = whole_fbank(x) if getattr(embedder, "needs_fbank", True) else None
    windows = sliding_windows(regions, s.win_s, s.hop_s)
    emb = embedder.embed_windows(x, windows, fb) if windows else np.zeros((0, 1), np.float32)
    rw = sliding_windows(regions, s.reseg_win_s, s.reseg_hop_s) if (with_reseg and s.reseg and windows) else []
    re = embedder.embed_windows(x, rw, fb) if rw else np.zeros((0, emb.shape[1]), np.float32)
    t2 = time.perf_counter()
    name = getattr(embedder, "label", type(embedder).__name__)
    return Prepared(total_s, regions, windows, emb, rw, re,
                    {"vad_s": round(t1 - t0, 3), "embed_s": round(t2 - t1, 3), "front_s": round(t2 - t0, 3),
                     "regions": len(regions), "windows": len(windows), "speech_s": round(sum(b - a for a, b in regions), 2)},
                    name, overlap)


def finish_windows(p: Prepared, s: WindowSettings, online: bool = False) -> tuple[DiarResult, dict]:
    t0 = time.perf_counter()
    info: dict = {}
    if not p.windows:
        return DiarResult(_label(p.label, s, online), [], dict(p.timing)), info
    if online:
        ol = OnlineLabeler(s.online_threshold, s.max_speakers, s.lookback, s.online_mint_threshold)
        for e in p.emb:
            ol.push(e)
        dec = [ol.decided(i) for i in range(len(p.windows))]
        lab = np.array([d[0] for d in dec]); conf_w = np.array([d[1] for d in dec])
        fl = frame_labels(p.windows, lab, p.regions, p.total_s)
        cf = np.ones(len(fl))
        for w, c in zip(p.windows, conf_w):
            cf[int(round(w.start / FRAME)):int(round(w.end / FRAME))] = c
        turns = labels_to_turns(fl, cf)
        info["label_latency_s"] = round(s.win_s + s.lookback * s.hop_s, 3)
        info["voices"] = len(ol.sums)
    else:
        lab = cluster_ahc(p.emb, s.threshold, s.num_speakers, s.max_speakers) if s.cluster == "ahc" \
            else cluster_spectral(p.emb, s.num_speakers, s.max_speakers)
        lab = absorb_small_clusters(p.emb, lab, p.windows, s.min_cluster_s)
        cents = centroids_of(p.emb, lab)
        if s.reseg and len(cents) > 1 and len(p.reseg_windows):
            sims = p.reseg_emb @ cents.T
            rlab = np.argmax(sims, axis=1)
            fl = frame_labels(p.reseg_windows, rlab, p.regions, p.total_s)
            srt = np.sort(sims, axis=1)
            margin = (srt[:, -1] - srt[:, -2]) if sims.shape[1] > 1 else np.ones(len(p.reseg_windows))
            cf = np.ones(len(fl))
            for w, m in zip(p.reseg_windows, margin):
                cf[int(round(w.start / FRAME)):int(round(w.end / FRAME))] = float(np.clip(0.5 + m, 0.0, 1.0))
        else:
            fl = frame_labels(p.windows, lab, p.regions, p.total_s)
            cf = np.ones(len(fl))
        fl = smooth_labels(fl, s.min_turn_s)
        turns = labels_to_turns(fl, cf)
        if s.overlap_second and p.overlap is not None and len(cents) > 1 and len(p.reseg_windows):
            # where the segmenter heard two people, add the second-nearest voice of the covering window
            order = np.argsort(-sims, axis=1)
            second = frame_labels(p.reseg_windows, order[:, 1], p.regions, p.total_s)
            primary_second = frame_labels(p.reseg_windows, order[:, 0], p.regions, p.total_s)
            fl2 = np.where(p.overlap[:len(fl)] & (fl >= 0), second[:len(fl)], -1)
            fl2 = np.where(fl2 == fl, np.where(fl >= 0, primary_second[:len(fl)], -1), fl2)   # never duplicate the primary
            fl2[fl2 == fl] = -1
            extra = [t for t in labels_to_turns(fl2) if t.end - t.start >= s.min_overlap_s]
            info["overlap_turns"] = len(extra)
            turns += extra
        info["voices"] = len(cents)
    turns = rename_by_first_appearance(turns)
    back = time.perf_counter() - t0
    wall = p.timing["front_s"] + back
    timing = {"wall_s": round(wall, 3), "audio_s": round(p.total_s, 3), "rtf": round(wall / max(p.total_s, 1e-6), 4),
              **p.timing, "cluster_s": round(back, 3)}
    if online:
        timing["label_latency_s"] = info["label_latency_s"]
    return DiarResult(_label(p.label, s, online), turns, timing), info


def diarize_windows(x: np.ndarray, probs_fn: Callable[[np.ndarray], np.ndarray],
                    embedder, s: WindowSettings, online: bool = False) -> tuple[DiarResult, dict]:
    """Front end + back end in one go (what the CLI does)."""
    return finish_windows(prepare_windows(x, probs_fn, embedder, s, with_reseg=not online), s, online)


def _label(name: str, s: WindowSettings, online: bool) -> str:
    if online:
        mint = s.online_threshold if s.online_mint_threshold is None else s.online_mint_threshold
        return (f"{name}, online centroid labelling on {s.win_s}s/{s.hop_s}s windows (join>={s.online_threshold}, "
                f"mint<{mint}, lookback {s.lookback}, max {s.max_speakers} voices)")
    cut = f" (cut {s.threshold})" if s.cluster == "ahc" else ""
    return (f"{name}, {s.cluster.upper()}{cut} on {s.win_s}s/{s.hop_s}s windows, reseg {s.reseg_win_s}s/{s.reseg_hop_s}s"
            + (", 2nd voice in overlap" if s.overlap_second else ""))


# ----------------------------------------------------------------------------- sherpa-onnx pyannote
def diarize_pyannote(x: np.ndarray, embedder_name: str = "campp", threshold: float = 0.5, num_speakers: int | None = None,
                     threads: int = 6, seg_model: Path | None = None, min_on: float = 0.3, min_off: float = 0.5) -> DiarResult:
    import sherpa_onnx

    seg = Path(seg_model or (models_dir("diar") / PYANNOTE_SEG))
    emb = models_dir() / EMBEDDERS[embedder_name]
    t0 = time.perf_counter()
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg)), num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers or -1, threshold=threshold),
        min_duration_on=min_on, min_duration_off=min_off)
    if not cfg.validate():
        raise RuntimeError(f"sherpa-onnx diarization config invalid (seg={seg}, emb={emb})")
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    res = sd.process(x.astype(np.float32)).sort_by_start_time()
    turns = rename_by_first_appearance([VoiceTurn(round(r.start, 3), round(r.end, 3), voice_name(int(r.speaker)), 1.0) for r in res])
    wall = time.perf_counter() - t0
    total_s = len(x) / SR
    return DiarResult(f"pyannote/segmentation-3.0 (sherpa-onnx) + WeSpeaker {Path(EMBEDDERS[embedder_name]).stem}, "
                      f"AHC {'k=' + str(num_speakers) if num_speakers else 'thr=' + str(threshold)}", turns,
                      {"wall_s": round(wall, 3), "audio_s": round(total_s, 3), "rtf": round(wall / max(total_s, 1e-6), 4)})


# ----------------------------------------------------------------------------- live provider (sensors contract)
class OnlineWindowDiarizer:
    """Drop-in for the `diarization` role (unspoken.contracts.Diarizer): the tuned online labeller applied to each
    live VAD segment. The segment is cut into win_s windows (hop_s), each embedded with the WeSpeaker ONNX model
    and pushed through one OnlineLabeler for the session; the segment's label is the majority window label.
    Causal, session-only, nothing stored. models.toml:
        provider = "python:unspoken.transcribe.diar:OnlineWindowDiarizer"
        options = { join = 0.6, mint = 0.4, max_voices = 4, win_s = 1.5, hop_s = 0.75, threads = 2 }
    Options also accept model_path / embedder ("campp" | "resnet34") / min_seconds (shorter -> "voice-?")."""

    def __init__(self, spec=None, **options) -> None:
        from ..sensors.config import merge_options
        from ..sensors.diar import EmbeddingDiarizer
        o = merge_options(spec, options)
        self.join = float(o.get("join", 0.6))
        self.mint = float(o.get("mint", 0.4))
        self.max_voices = int(o.get("max_voices", 4))
        self.win_s = float(o.get("win_s", 1.5))
        self.hop_s = float(o.get("hop_s", 0.75))
        self.min_seconds = float(o.get("min_seconds", 0.3))
        emb_opts = {"threads": int(o.get("threads", 2))}
        if o.get("model_path"):
            emb_opts["model_path"] = o["model_path"]
        elif o.get("embedder") in EMBEDDERS:
            emb_opts["model_path"] = str(models_dir() / EMBEDDERS[o["embedder"]])
        self.embedder = o.get("_embedder") or EmbeddingDiarizer(**emb_opts)
        self.label = f"{getattr(self.embedder, 'label', 'embedding')} + online window labelling"

    def session(self) -> "OnlineWindowSession":
        return OnlineWindowSession(self)

    def healthy(self) -> bool:
        h = getattr(self.embedder, "healthy", None)
        return bool(h()) if h else True


class OnlineWindowSession:
    def __init__(self, d: OnlineWindowDiarizer) -> None:
        self.d = d
        self.ol = OnlineLabeler(d.join, d.max_voices, 0, d.mint)
        self.last_ms = 0.0

    async def label(self, pcm16: bytes, sample_rate: int):
        import asyncio
        return await asyncio.to_thread(self.label_sync, pcm16, sample_rate)

    def label_sync(self, pcm16: bytes, sample_rate: int):
        from ..contracts import VoiceLabel
        from ..sensors.diar import UNKNOWN
        t0 = time.perf_counter()
        n_samples = len(pcm16) // 2
        dur = n_samples / sample_rate
        if dur < self.d.min_seconds:
            return VoiceLabel(label=UNKNOWN, conf=0.0)
        step = int(self.d.hop_s * sample_rate) * 2
        win = int(self.d.win_s * sample_rate) * 2
        chunks = [pcm16[i:i + win] for i in range(0, max(len(pcm16) - win, 0) + 1, step)] if len(pcm16) > win else [pcm16]
        if len(pcm16) > win and (len(pcm16) - win) % step:
            chunks.append(pcm16[-win:])
        labels, confs = [], []
        for c in chunks:
            emb = self.d.embedder.embed(c, sample_rate)
            if emb.size == 0:
                continue
            lab, conf = self.ol.push(emb)
            labels.append(lab); confs.append(conf)
        self.last_ms = (time.perf_counter() - t0) * 1000
        if not labels:
            return VoiceLabel(label=UNKNOWN, conf=0.0)
        vals, counts = np.unique(labels, return_counts=True)
        best = int(vals[np.argmax(counts)])
        conf = float(np.mean([c for l, c in zip(labels, confs) if l == best])) * (counts.max() / len(labels))
        return VoiceLabel(label=voice_name(best), conf=round(conf, 3))

    def reset(self) -> None:
        self.ol = OnlineLabeler(self.d.join, self.d.max_voices, 0, self.d.mint)

    @property
    def voices(self) -> list[str]:
        return [voice_name(i) for i in range(len(self.ol.sums))]


# ----------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> DiarResult:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="audio.wav (16 kHz mono) or anything PyAV decodes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="window", choices=["window", "pyannote", "sortformer"])
    ap.add_argument("--online", action="store_true", help="window method: causal labelling (past only)")
    ap.add_argument("--embedder", default="campp", choices=sorted(EMBEDDERS))
    ap.add_argument("--cluster", default="spectral", choices=["spectral", "ahc"])
    ap.add_argument("--threshold", type=float, default=None, help="AHC cosine-distance cut (window) / sherpa threshold (pyannote)")
    ap.add_argument("--online-threshold", type=float, default=0.6, help="cosine similarity to join a known voice")
    ap.add_argument("--online-mint-threshold", type=float, default=None,
                    help="below this similarity a new voice is minted; between it and --online-threshold the window is "
                         "assigned to the nearest voice without updating its centroid (default: same as --online-threshold)")
    ap.add_argument("--lookback", type=int, default=0, help="online: decide a window after this many more windows")
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=8)
    ap.add_argument("--window", type=float, default=1.5)
    ap.add_argument("--hop", type=float, default=0.75)
    ap.add_argument("--no-reseg", action="store_true")
    ap.add_argument("--min-silence", type=float, default=0.3)
    ap.add_argument("--vad-threshold", type=float, default=0.5)
    ap.add_argument("--min-cluster", type=float, default=2.0, help="seconds; smaller clusters are absorbed")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "6")))
    ap.add_argument("--vad", default="silero", choices=["silero", "energy", "pyannote"],
                    help="speech regions: Silero VAD (default), energy gate (tests), or pyannote/segmentation-3.0 via "
                         "sherpa-onnx (overlap-aware; enables --overlap)")
    ap.add_argument("--overlap", action="store_true", help="with --vad pyannote: add the 2nd-nearest voice where 2 people talk")
    a = ap.parse_args(argv)
    load1 = os.getloadavg()[0]
    x = read_wav(a.inp)
    if a.method == "window":
        s = WindowSettings(win_s=a.window, hop_s=a.hop, cluster=a.cluster, num_speakers=a.num_speakers,
                           max_speakers=a.max_speakers, reseg=not a.no_reseg, vad_min_silence_s=a.min_silence,
                           vad_threshold=a.vad_threshold, min_cluster_s=a.min_cluster,
                           lookback=a.lookback, online_threshold=a.online_threshold,
                           online_mint_threshold=a.online_mint_threshold)
        if a.threshold is not None:
            s.threshold = a.threshold
        s.overlap_second = a.overlap and a.vad == "pyannote"
        emb = OnnxEmbedder(a.embedder, threads=a.threads)
        if a.vad == "pyannote":
            activity = pyannote_activity(x, a.threads, embedder_name=a.embedder)
            res, info = finish_windows(prepare_windows(x, None, emb, s, with_reseg=not a.online, activity=activity), s, a.online)
        else:
            probs_fn = SileroProbs(threads=a.threads) if a.vad == "silero" else energy_probs
            res, info = diarize_windows(x, probs_fn, emb, s, online=a.online)
        res.timing.update({k: v for k, v in info.items() if k in ("regions", "windows", "speech_s", "voices", "overlap_turns")})
        res.timing["vad"] = a.vad
    elif a.method == "pyannote":
        res = diarize_pyannote(x, a.embedder, a.threshold if a.threshold is not None else 0.5, a.num_speakers, a.threads)
    else:
        from .diar_sortformer import diarize_sortformer
        res = diarize_sortformer(x, threads=a.threads)
    res.timing.update({"threads": a.threads, "load1_at_start": round(load1, 2), "load1_at_end": round(os.getloadavg()[0], 2)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    to_json(res, a.out)
    print(json.dumps({"model": res.model, "turns": len(res.turns), "voices": len({t.voice for t in res.turns}), **res.timing}))
    return res


if __name__ == "__main__":
    main()
