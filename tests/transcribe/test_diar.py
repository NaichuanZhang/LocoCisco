"""unspoken.transcribe.diar without models: synthetic two-"speaker" signal (two tones), an embedder that reads the
tone, the energy VAD. Checks regions, windows, clustering (AHC + spectral), resegmentation, smoothing, the online
labeller, the CLI contract (DiarResult JSON) and the Sortformer output parsing. Nothing is downloaded."""
from __future__ import annotations

import json

import numpy as np
import pytest

from eval.transcribe.score import der
from unspoken.transcribe import diar as D
from unspoken.transcribe import diar_sortformer as SF
from unspoken.transcribe.contracts import load_diar

SR = D.SR
TONES = {"A": 220.0, "B": 660.0, "C": 1500.0}
# (speaker, start, end) in seconds; silence in between
SCHEDULE = [("A", 0.5, 5.0), ("B", 6.0, 9.0), ("A", 9.0, 14.0), ("B", 15.0, 18.0), ("A", 18.0, 19.5)]


def synth(schedule=SCHEDULE, total=20.5, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.002, int(total * SR)).astype(np.float32)
    t = np.arange(int(total * SR)) / SR
    for spk, a, b in schedule:
        m = (t >= a) & (t < b)
        x[m] += 0.3 * np.sin(2 * np.pi * TONES[spk] * t[m]).astype(np.float32)
    return x


DIM = 16


class ToneEmbedder:
    """Embedding = normalised band energies around the three tones in the first 3 dims, plus small noise in
    the other 13 (real embeddings vary in many directions; a 3-d fixture makes kNN graphs that are degenerate
    for spectral clustering, which is a property of the fixture, not of speech embeddings)."""
    needs_fbank = False
    label = "tone embedder (test fake)"

    def __init__(self, noise: float = 0.05, seed: int = 0) -> None:
        self.noise, self.rng = noise, np.random.default_rng(seed)

    def embed_windows(self, x, windows, fbank=None):
        out = np.zeros((len(windows), DIM), np.float32)
        for i, w in enumerate(windows):
            seg = x[int(w.start * SR):int(w.end * SR)]
            spec = np.abs(np.fft.rfft(seg))
            freqs = np.fft.rfftfreq(len(seg), 1 / SR)
            for j, f in enumerate(TONES.values()):
                out[i, j] = spec[(freqs > f - 30) & (freqs < f + 30)].sum()
            out[i] /= max(out[i].max(), 1e-6)
            out[i] += self.rng.normal(0, self.noise, DIM)
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def ref_turns():
    return [(a, b, s) for s, a, b in SCHEDULE]


# ----------------------------------------------------------------------------- front end
def test_speech_regions_hysteresis_and_padding():
    probs = np.zeros(100, np.float32)       # 100 steps of 32 ms = 3.2 s
    probs[10:40] = 0.9                       # speech 0.32..1.28
    probs[41:43] = 0.9                       # a 2-step blip after a 1-step dip (shorter than min silence) joins
    probs[80:84] = 0.9                       # 0.128 s burst < min_speech: dropped
    r = D.speech_regions(probs, min_speech_s=0.2, min_silence_s=0.3, pad_s=0.1)
    assert len(r) == 1
    a, b = r[0]
    assert a == pytest.approx(0.32 - 0.1, abs=1e-6) and b == pytest.approx(43 * 0.032 + 0.1, abs=0.04)


def test_sliding_windows_cover_regions():
    ws = D.sliding_windows([(0.0, 4.0), (10.0, 10.8), (20.0, 20.1)], win_s=1.5, hop_s=0.75, min_s=0.3)
    assert [w.region for w in ws].count(0) == 5 and ws[0].start == 0.0 and ws[4].end == pytest.approx(4.0)
    assert [(w.start, w.end) for w in ws if w.region == 1] == [(10.0, 10.8)]     # short region = one window
    assert not [w for w in ws if w.region == 2]                                  # below min_s: skipped


def test_energy_probs_marks_tone_as_speech():
    x = synth()
    p = D.energy_probs(x)
    step = 512 / SR
    assert p[int(2.0 / step)] > 0.8 and p[int(5.5 / step)] < 0.2


# ----------------------------------------------------------------------------- clustering
def _clouds(sizes=(30, 30), seed=1, noise=0.05):
    """Unit vectors around distinct axes with isotropic noise (16-d, like real embeddings in the ways that
    matter for kNN graphs)."""
    rng = np.random.default_rng(seed)
    parts, truth = [], []
    for i, m in enumerate(sizes):
        c = np.zeros(DIM); c[i] = 1.0
        parts.append(c + rng.normal(0, noise, (m, DIM)))
        truth += [i] * m
    e = np.vstack(parts)
    return e / np.linalg.norm(e, axis=1, keepdims=True), np.array(truth)


def _two_clouds(n=30, seed=1):
    return _clouds((n, n), seed)


def test_ahc_and_spectral_find_two_speakers():
    emb, truth = _two_clouds()
    for lab in (D.cluster_ahc(emb, threshold=0.5), D.cluster_spectral(emb)):
        assert lab.max() == 1 and lab[0] == 0 and (lab == truth).all()          # first appearance -> 0


def test_spectral_speaker_count_one_three_four_unequal():
    for sizes in [(50,), (40, 15, 5), (60, 20, 10, 6)]:
        for seed in range(3):
            emb, truth = _clouds(sizes, seed)
            lab = D.cluster_spectral(emb)
            assert lab.max() + 1 == len(sizes), (sizes, seed, lab.max() + 1)
            assert all(len(set(lab[truth == i])) == 1 for i in range(len(sizes)))


def test_ahc_respects_num_and_max_speakers():
    emb, _ = _two_clouds()
    assert D.cluster_ahc(emb, threshold=0.01, num_speakers=1).max() == 0
    assert D.cluster_ahc(emb, threshold=0.01, max_speakers=3).max() <= 2
    assert D.cluster_spectral(emb, num_speakers=3).max() <= 2


def test_absorb_small_clusters_merges_tiny_voice():
    emb, truth = _two_clouds()
    lab = truth.copy()
    lab[5] = 2                                                                   # a one-window third voice
    ws = [D.Window(i * 1.0, i * 1.0 + 1.0, 0) for i in range(len(lab))]
    out = D.absorb_small_clusters(emb, lab, ws, min_s=2.0)
    assert out.max() == 1 and out[5] == 0


# ----------------------------------------------------------------------------- labels -> turns
def test_frame_labels_smoothing_turns_and_naming():
    ws = [D.Window(0.0, 1.0, 0), D.Window(0.5, 1.5, 0), D.Window(1.0, 2.0, 0)]
    fl = D.frame_labels(ws, np.array([1, 1, 0]), [(0.0, 2.0)], 2.5)
    assert fl[10] == 1 and fl[190] == 0 and fl[220] == -1
    assert fl[124] == 1 and fl[126] == 0                   # boundary = midpoint between window centres (1.25 s)
    fl[60:70] = 0                                                                # a 0.1 s blip inside voice 1
    sm = D.smooth_labels(fl, min_turn_s=0.25)
    assert (sm[0:125] == 1).all() and (sm[126:200] == 0).all()
    turns = D.rename_by_first_appearance(D.labels_to_turns(sm))
    assert [t.voice for t in turns] == ["voice-A", "voice-B"] and turns[0].start == 0.0
    assert turns[0].end == pytest.approx(turns[1].start)


def test_short_run_between_silences_is_kept():
    fl = np.full(100, -1)
    fl[40:50] = 2                                                                # 0.1 s backchannel alone
    assert (D.smooth_labels(fl, 0.25) == fl).all()


# ----------------------------------------------------------------------------- online labeller
def test_online_labeler_joins_mints_caps_and_lookback():
    a = np.array([1.0, 0, 0]); b = np.array([0, 1.0, 0])
    c = np.array([0, 0.8, 0.6]); d = np.array([0, 0, 1.0])
    ol = D.OnlineLabeler(threshold=0.6, max_voices=2, lookback=1)
    assert ol.push(a)[0] == 0 and ol.push(a)[0] == 0
    assert ol.push(b)[0] == 1                                                    # unlike anything: new voice
    assert ol.push(c)[0] == 1                                                    # sim 0.8 to b: joins, no mint
    assert ol.push(d)[0] == 1 and len(ol.sums) == 2                              # cap reached: joins nearest
    assert ol.raw == [0, 0, 1, 1, 1]
    assert ol.decided(0)[0] == 0 and ol.decided(2)[0] == 1                       # majority over windows i..i+1


def test_online_mint_band_assigns_without_updating_centroid():
    a = np.array([1.0, 0, 0])
    mid = np.array([0.7, 0.714, 0]); mid /= np.linalg.norm(mid)                  # sim 0.7 to a
    ol = D.OnlineLabeler(threshold=0.9, max_voices=8, mint_threshold=0.5)
    ol.push(a)
    lab, conf = ol.push(mid)
    assert lab == 0 and conf < 0.5 and len(ol.sums) == 1 and np.allclose(ol.sums[0], a)   # no centroid drift
    assert ol.push(np.array([0, 0, 1.0]))[0] == 1                                # below mint threshold: new voice


# ----------------------------------------------------------------------------- end to end (fakes)
def test_offline_pipeline_recovers_two_voices_from_synthetic_signal():
    x = synth()
    s = D.WindowSettings(cluster="spectral", vad_min_silence_s=0.3)
    res, info = D.diarize_windows(x, D.energy_probs, ToneEmbedder(), s)
    voices = {t.voice for t in res.turns}
    assert voices == {"voice-A", "voice-B"} and info["voices"] == 2
    hyp = [(t.start, t.end, t.voice) for t in res.turns]
    assert der(ref_turns(), hyp)["der"] < 0.08
    # resegmentation splits the 9.0 s change inside one continuous speech region (6.0..14.0); the 1.0 s
    # resegmentation windows place the boundary within about half a window
    change = [t for t in res.turns if abs(t.start - 9.0) < 0.6]
    assert change and change[0].voice == "voice-A"
    assert res.timing["rtf"] > 0 and res.timing["audio_s"] == pytest.approx(20.5)


def test_offline_pipeline_ahc_matches_spectral_on_synthetic():
    x = synth()
    res, _ = D.diarize_windows(x, D.energy_probs, ToneEmbedder(), D.WindowSettings(cluster="ahc", threshold=0.5))
    assert {t.voice for t in res.turns} == {"voice-A", "voice-B"}
    assert der(ref_turns(), [(t.start, t.end, t.voice) for t in res.turns])["der"] < 0.08


def test_online_pipeline_is_causal_and_reports_latency():
    x = synth()
    s = D.WindowSettings(online_threshold=0.7, lookback=1)
    res, info = D.diarize_windows(x, D.energy_probs, ToneEmbedder(), s, online=True)
    assert {t.voice for t in res.turns} == {"voice-A", "voice-B"}
    assert res.timing["label_latency_s"] == pytest.approx(1.5 + 0.75) and info["label_latency_s"] == res.timing["label_latency_s"]
    assert der(ref_turns(), [(t.start, t.end, t.voice) for t in res.turns])["der"] < 0.1


def test_cli_writes_contract_json(tmp_path, monkeypatch):
    import wave
    x = synth()
    wav = tmp_path / "audio.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    monkeypatch.setattr(D, "OnnxEmbedder", lambda name, threads=6: ToneEmbedder())
    out = tmp_path / "diar.json"
    res = D.main(["--in", str(wav), "--out", str(out), "--method", "window", "--vad", "energy", "--cluster", "spectral",
                  "--threads", "2"])
    loaded = load_diar(out)
    assert loaded.model == res.model and len(loaded.turns) == len(res.turns) >= 5
    assert loaded.timing["threads"] == 2 and "rtf" in loaded.timing
    d = json.loads(out.read_text())
    assert set(d) == {"model", "turns", "timing"} and set(d["turns"][0]) == {"start", "end", "voice", "conf"}
    assert all(t["voice"].startswith("voice-") and t["end"] > t["start"] for t in d["turns"])


def test_cli_online_flag(tmp_path, monkeypatch):
    import wave
    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((synth() * 32767).astype("<i2").tobytes())
    monkeypatch.setattr(D, "OnnxEmbedder", lambda name, threads=6: ToneEmbedder())
    res = D.main(["--in", str(wav), "--out", str(tmp_path / "o.json"), "--online", "--vad", "energy",
                  "--online-threshold", "0.8", "--online-mint-threshold", "0.5", "--lookback", "1"])
    assert res.timing["label_latency_s"] == pytest.approx(2.25) and "mint<0.5" in res.model


# ----------------------------------------------------------------------------- sortformer glue (no nemo)
def test_sortformer_parse_and_result():
    segs = SF.parse_segments(["0.000 1.500 speaker_1", "1.400 3.000 speaker_0", [3.0, 3.5, "speaker_1"]])
    assert segs == [(0.0, 1.5, 1), (1.4, 3.0, 0), (3.0, 3.5, 1)]
    res = SF.to_result(segs, total_s=4.0, label="sf", wall=0.4, latency_s=10.0)
    assert [t.voice for t in res.turns] == ["voice-A", "voice-B", "voice-A"]    # first appearance wins the name
    assert res.timing["rtf"] == pytest.approx(0.1) and res.timing["label_latency_s"] == 10.0
    assert all((p["chunk_len"] + p["chunk_right_context"]) * SF.FRAME_S > 0 for p in SF.PRESETS.values())
    assert SF.PRESETS["high"]["chunk_len"] + SF.PRESETS["high"]["chunk_right_context"] == 125   # 10.0 s


# ----------------------------------------------------------------------------- live provider adapter
class _PcmToneEmbedder:
    """EmbeddingDiarizer stand-in: embed(pcm16, sr) -> 16-d tone embedding of that chunk."""
    label = "fake pcm tone embedder"

    def __init__(self) -> None:
        self.inner = ToneEmbedder()

    def embed(self, pcm16: bytes, sample_rate: int = 16000):
        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        return self.inner.embed_windows(x, [D.Window(0.0, len(x) / SR, 0)])[0]

    def healthy(self) -> bool:
        return True


async def test_online_window_diarizer_live_protocol():
    from unspoken.contracts import Diarizer
    d = D.OnlineWindowDiarizer(options={"join": 0.7, "mint": 0.5, "max_voices": 4, "_embedder": _PcmToneEmbedder()})
    assert isinstance(d, Diarizer) and d.healthy()
    x = synth()
    pcm = lambda a, b: (np.clip(x[int(a * SR):int(b * SR)], -1, 1) * 32767).astype("<i2").tobytes()
    s = d.session()
    first = await s.label(pcm(0.5, 5.0), SR)                 # A, 4.5 s -> several windows
    second = await s.label(pcm(6.0, 9.0), SR)                # B
    third = await s.label(pcm(9.0, 14.0), SR)                # A again
    assert first.label == "voice-A" and second.label == "voice-B" and third.label == "voice-A"
    assert 0 < third.conf <= 1 and s.last_ms >= 0 and s.voices == ["voice-A", "voice-B"]
    assert (await s.label(pcm(6.0, 6.1), SR)).label == "voice-?"   # too short to embed
    s.reset()
    assert s.voices == [] and (await s.label(pcm(6.0, 9.0), SR)).label == "voice-A"   # B is the first voice now
