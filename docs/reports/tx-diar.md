# tx-diar: who spoke when, from audio only

Real test on AMI IS1009a (duo-AC, duo-DB, L-room, R-room; `-dev` = 180 s excerpts for tuning, full = ~830 s reported
once). Scorer: `eval/transcribe/score.py --diar` (DER, 10 ms frames, 0.25 s collar, overlap scored, best mapping).
Everything below runs locally on CPU; every model is Apache-2.0 / MIT except Sortformer (CC-BY-4.0, commercial use OK
with attribution). The L-room and R-room cases share one far-field recording, so their diarization numbers are identical
by construction (the video differs, not the audio).

## 1. What I built

- `unspoken/transcribe/diar.py` — the diarizer CLI and library: Silero VAD (or pyannote segmentation) speech regions ->
  sliding-window WeSpeaker embeddings (CAM++ 512-d / ResNet34-LM 256-d, ONNX, the same files the live sensors use) ->
  offline clustering with automatic speaker count (NME-SC spectral, or AHC) -> resegmentation with 1.0 s / 0.25 s windows
  against the cluster centroids -> smoothing -> `DiarResult`. `--online`: causal centroid labelling (past only).
  `--vad pyannote [--overlap]`: speech + overlap frames from pyannote/segmentation-3.0 (sherpa-onnx), second voice in
  overlap. `--method pyannote`: sherpa-onnx's own OfflineSpeakerDiarization. `OnlineWindowDiarizer`: drop-in provider for
  the live `diarization` role (`unspoken.contracts.Diarizer`), the tuned online labeller per VAD segment.
- `unspoken/transcribe/diar_sortformer.py` — NVIDIA Streaming Sortformer (`nvidia/diar_streaming_sortformer_4spk-v2`,
  NeMo) -> `DiarResult`, with the model card's four streaming presets (latency 0.32 s .. 30.4 s).
- `unspoken/transcribe/diar_bench.py` — tuner (grid on the -dev excerpts with cached embeddings), runner (method x case ->
  `testdata/ami-runs/diar/<method>/<case>/{diar,scores}.json`) and the markdown table.
- `tests/transcribe/test_diar.py` — 18 tests, no models: synthetic two-tone signal + a fake embedder; regions, windows,
  AHC/spectral speaker count (1/2/3/4 speakers, unequal sizes), resegmentation change point, smoothing, online labeller
  (join / mint band / cap / lookback), CLI contract JSON, live-provider protocol, Sortformer output parsing.
- `requirements/transcribe-diar.txt` — one new pin (`sherpa-onnx==2.0.0`); NeMo documented as optional.
- Models: `dev-models/diar/sherpa-onnx-pyannote-segmentation-3-0/` (6 MB, MIT), `dev-models/diar/hf-cache/`
  (Sortformer .nemo, downloaded by NeMo on first use). WeSpeaker + Silero come from `dev-models/sensors/`.
- Run outputs + `BEST`: `testdata/ami-runs/diar/` (not in the repo); tuning sweeps in `testdata/ami-runs/diar/tune-*.json`.

## 2. How to run

```bash
cd unspoken-wt/tx-diar
export OMP_NUM_THREADS=6 UNSPOKEN_MODELS_DIR=/local/home/zhangnai/kirocrew-sessions/2026-10-23-dell-nvidia-hackathon/dev-models/sensors
T=../../testdata/ami/IS1009a-L-room
# Sortformer only: a separate venv (uv venv --python 3.12 nemo-venv; uv pip install torch==2.9.0 torchaudio==2.9.0
#   --index-url https://download.pytorch.org/whl/cpu; uv pip install 'nemo_toolkit[asr]==3.0.0'); ~4 min with the uv cache
NEMO_PY=/path/to/nemo-venv/bin/python

# offline recommendation (pyannote activity + CAM++ windows + NME-SC spectral); add --overlap for a 2nd voice in overlap
.venv/bin/python -m unspoken.transcribe.diar --in $T/audio.wav --out diar.json --method window --vad pyannote
# Silero-only variant (no sherpa-onnx needed)
.venv/bin/python -m unspoken.transcribe.diar --in $T/audio.wav --out diar.json --method window
# online (causal) recommendation
.venv/bin/python -m unspoken.transcribe.diar --in $T/audio.wav --out diar.json --method window --online \
    --online-threshold 0.6 --online-mint-threshold 0.4 --max-speakers 4 --vad-threshold 0.35
# sherpa-onnx pyannote pipeline; Sortformer (needs the NeMo venv, see section 5)
.venv/bin/python -m unspoken.transcribe.diar --in $T/audio.wav --out diar.json --method pyannote --threshold 0.9
$NEMO_PY -m unspoken.transcribe.diar_sortformer --in $T/audio.wav --out diar.json --latency high
# score
.venv/bin/python -m eval.transcribe.score --ref $T --diar diar.json
# everything: tune on -dev, run all methods on all cases, print the table
.venv/bin/python -m unspoken.transcribe.diar_bench tune --embedder campp            # offline grid
.venv/bin/python -m unspoken.transcribe.diar_bench tune --embedder campp --online   # online grid
.venv/bin/python -m unspoken.transcribe.diar_bench run --methods baseline,window-campp,window-campp-pa,window-campp-pa-ov,window-resnet34,window-ahc-campp,online-campp,pyannote-campp --cases all
.venv/bin/python -m unspoken.transcribe.diar_bench run --methods sortformer --cases all --python-nemo $NEMO_PY
.venv/bin/python -m unspoken.transcribe.diar_bench table
.venv/bin/python -m pytest -q tests/transcribe/test_diar.py
```

## 3. Results

All numbers: DER (hyp speakers / ref speakers). "dev" = 180 s tuning excerpts, "full" = ~830 s, run once after
tuning. R-room is omitted from the matrix (same audio as L-room; its rows are in the detail table and identical).
RTF = wall / audio seconds at 6 threads (`OMP_NUM_THREADS=6`, onnxruntime intra-op 6, torch 6) while five other
workstreams ran: 1-min load average 76-115 on 32 cores for every row (`load1_at_start` in each scores.json); on the
idle box the window method measured RTF 0.030 (duo-AC-dev, load 2.6). Baseline rows are the live-pipeline replay
converted by the integrator (no RTF of its own).

### DER matrix

| method | duo-AC-dev (dev) | duo-DB-dev (dev) | L-room-dev (dev) | duo-AC (full) | duo-DB (full) | L-room (full) | mean full | RTF (6 thr) | latency |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.096 (7/2) | 0.162 (8/2) | 0.151 (7/4) | 0.165 (8/2) | 0.176 (8/2) | 0.272 (8/4) | 0.204 | n/a (replay) | - |
| window-campp | 0.056 (2/2) | 0.066 (2/2) | 0.083 (4/4) | 0.126 (2/2) | 0.141 (2/2) | 0.226 (5/4) | 0.164 | 0.021-0.051 | - |
| window-campp-pa | 0.039 (2/2) | 0.078 (2/2) | 0.052 (4/4) | 0.089 (2/2) | 0.135 (3/2) | 0.203 (5/4) | 0.142 | 0.016-0.058 | - |
| window-campp-pa-ov | 0.042 (2/2) | 0.107 (2/2) | 0.047 (4/4) | 0.061 (2/2) | 0.255 (3/2) | 0.201 (5/4) | 0.172 | 0.011-0.043 | - |
| window-resnet34 | 0.052 (2/2) | 0.082 (2/2) | 0.070 (4/4) | 0.121 (2/2) | 0.205 (3/2) | 0.232 (5/4) | 0.186 | 0.019-0.055 | - |
| window-ahc-campp | 0.056 (2/2) | 0.066 (2/2) | 0.196 (3/4) | 0.130 (3/2) | 0.160 (5/2) | 0.268 (7/4) | 0.186 | 0.019-0.056 | - |
| online-campp | 0.079 (4/2) | 0.082 (4/2) | 0.090 (4/4) | 0.135 (4/2) | 0.152 (4/2) | 0.268 (4/4) | 0.185 | 0.015-0.027 | 1.5 s |
| pyannote-campp | 0.212 (2/2) | 0.588 (3/2) | 0.332 (2/4) | 0.423 (4/2) | 0.646 (6/2) | 0.588 (5/4) | 0.553 | 0.060-0.157 | - |
| sortformer | 0.110 (4/2) | 0.539 (3/2) | 0.299 (4/4) | 0.131 (4/2) | 0.294 (4/2) | 0.251 (4/4) | 0.225 | 0.068-1.921 | 10.0 s |

### What was tuned (dev excerpts only; sweeps in testdata/ami-runs/diar/tune-*.json, mean over the 3 distinct audios)

| method | grid | picked | dev mean DER | speaker-count error |
|---|---|---|---|---|
| window (CAM++) | cluster ahc/spectral x AHC cut 0.3-0.9 x min_cluster 2/5 s x reseg on/off x VAD thr 0.35/0.5 | spectral (NME-SC, auto count), VAD 0.5, reseg on | 0.069 (0.067 reseg off) | 0 / 0 / 0 |
| window (CAM++), AHC | cut 0.3-0.9 | cut 0.85 | 0.107 | 0 / 0 / -1 (room merges 2 voices) |
| window (ResNet34-LM) | as above | spectral, VAD 0.35 | 0.068 | 0 / 0 / 0 |
| window (CAM++) + pyannote activity | front end only | no overlap (mean); overlap for the room | 0.056 (overlap 0.065) | 0 / 0 / 0 |
| online (CAM++) | join 0.5-0.7 x mint 0.35-0.55 x lookback 0-2 x max voices 4/6/8 x VAD 0.35/0.5 | join 0.6, mint 0.4, lookback 0, max 4, VAD 0.35 | 0.084 | +2 / +2 / 0 |
| sherpa-onnx pyannote pipeline | clustering threshold 0.3-0.9 | 0.9 | 0.378 | 0 / +1 / -2 |
| Sortformer | presets veryhigh / high / low on L-room-dev | high (10 s) | room-dev 0.310 / 0.299 / 0.272 at RTF 0.25 / 0.09 / 1.79 | 0 |

### Winner and why

- Offline: **window-campp-pa** (pyannote/segmentation-3.0 activity -> CAM++ 1.5 s/0.75 s windows -> NME-SC spectral ->
  resegmentation). Full-case mean DER 0.142 vs 0.204 for today's pipeline (-30 %), 0.164 for the same clustering on
  Silero activity, and the right speaker count on every dev case (the live pipeline mints 7-8 voices for 2-4 people).
  Confusion is 0.001-0.04 on the duos; what is left is **miss**: 0.065 / 0.036 / 0.094 on the full cases, mostly
  overlapped speech the single-label system cannot score (13.6 % of the room's speech time is overlap). The
  pyannote segmenter hears overlap, so `--overlap` adds the second-nearest voice there: best single numbers of the
  whole table on duo-AC (0.061) and the room (0.201 full, 0.047 dev), but on the artificial gated headset mix
  duo-DB it adds false alarms (0.255), so it stays an option for the far-field room microphone, not the default.
- Online: **online-campp** (causal centroid labelling, join 0.6 / mint 0.4 / 4 voices, 1.5 s label latency). Full
  mean 0.185 vs 0.204 baseline, dev mean 0.084 vs 0.136; on the full room it only matches the baseline (0.268 vs
  0.272; confusion 0.091): far-field embeddings drift and the causal labeller cannot revisit a decision. It is the
  cheapest method here (RTF 0.015-0.027) and already fits the live `diarization` role via `OnlineWindowDiarizer`.
- Neural end-to-end: Sortformer (streaming, 10 s preset) got the right count on the room but confused speakers
  (0.251 full room, 0.294 duo-DB, mean 0.225, worse than baseline) and is slow on this CPU (10 s preset: RTF 0.07-0.53
  on the full cases, 1.1-1.9 on the 180 s excerpts where the per-run setup dominates, load 76-113; the 1.04 s
  preset ran at RTF 1.8 on a dev case). sherpa-onnx's complete pyannote pipeline (its own
  segment embeddings + clustering) was the worst method (0.55): the segmenter is good, its clustering is not, which is
  exactly why the winner keeps the segmenter and replaces the clustering.
- ResNet34-LM embeddings tie CAM++ on dev (0.068 vs 0.069) and lose on full (0.186 vs 0.164): CAM++ it is.
- AHC needs a cut tuned per channel (0.85 here) and still merged two room voices; NME-SC needs no threshold.

### Detail (every method x case, from scores.json)

| method | case | DER | miss | FA | conf | spk hyp/ref | RTF | threads | load1 | latency s |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | IS1009a-duo-AC-dev | 0.096 | 0.031 | 0.020 | 0.046 | 7/2 | n/a (replay) | - | - | - |
| baseline | IS1009a-duo-DB-dev | 0.162 | 0.007 | 0.130 | 0.025 | 8/2 | n/a (replay) | - | - | - |
| baseline | IS1009a-L-room-dev | 0.151 | 0.043 | 0.010 | 0.098 | 7/4 | n/a (replay) | - | - | - |
| baseline | IS1009a-R-room-dev | 0.153 | 0.044 | 0.011 | 0.098 | 7/4 | n/a (replay) | - | - | - |
| baseline | IS1009a-duo-AC | 0.165 | 0.095 | 0.018 | 0.052 | 8/2 | n/a (replay) | - | - | - |
| baseline | IS1009a-duo-DB | 0.176 | 0.090 | 0.067 | 0.018 | 8/2 | n/a (replay) | - | - | - |
| baseline | IS1009a-L-room | 0.272 | 0.151 | 0.022 | 0.099 | 8/4 | n/a (replay) | - | - | - |
| baseline | IS1009a-R-room | 0.272 | 0.151 | 0.022 | 0.099 | 8/4 | n/a (replay) | - | - | - |
| window-campp | IS1009a-duo-AC-dev (dev) | 0.056 | 0.050 | 0.006 | 0.000 | 2/2 | 0.0389 | 6 | 76.04 | - |
| window-campp | IS1009a-duo-DB-dev (dev) | 0.066 | 0.016 | 0.051 | 0.000 | 2/2 | 0.0207 | 6 | 80.31 | - |
| window-campp | IS1009a-L-room-dev (dev) | 0.083 | 0.071 | 0.005 | 0.007 | 4/4 | 0.0512 | 6 | 84.29 | - |
| window-campp | IS1009a-R-room-dev (dev) | 0.083 | 0.071 | 0.005 | 0.007 | 4/4 | 0.0663 | 6 | 86.51 | - |
| window-campp | IS1009a-duo-AC (full) | 0.126 | 0.118 | 0.006 | 0.001 | 2/2 | 0.0487 | 6 | 92.69 | - |
| window-campp | IS1009a-duo-DB (full) | 0.141 | 0.101 | 0.033 | 0.007 | 2/2 | 0.0233 | 6 | 99.63 | - |
| window-campp | IS1009a-L-room (full) | 0.226 | 0.170 | 0.017 | 0.039 | 5/4 | 0.0404 | 6 | 97.52 | - |
| window-campp | IS1009a-R-room (full) | 0.226 | 0.170 | 0.017 | 0.039 | 5/4 | 0.0421 | 6 | 93.96 | - |
| window-campp-pa | IS1009a-duo-AC-dev (dev) | 0.039 | 0.015 | 0.024 | 0.000 | 2/2 | 0.0301 | 6 | 91.04 | - |
| window-campp-pa | IS1009a-duo-DB-dev (dev) | 0.078 | 0.000 | 0.078 | 0.000 | 2/2 | 0.0162 | 6 | 89.82 | - |
| window-campp-pa | IS1009a-L-room-dev (dev) | 0.052 | 0.012 | 0.014 | 0.026 | 4/4 | 0.0504 | 6 | 90.23 | - |
| window-campp-pa | IS1009a-R-room-dev (dev) | 0.052 | 0.012 | 0.014 | 0.026 | 4/4 | 0.0360 | 6 | 93.68 | - |
| window-campp-pa | IS1009a-duo-AC (full) | 0.089 | 0.065 | 0.015 | 0.009 | 2/2 | 0.0397 | 6 | 92.6 | - |
| window-campp-pa | IS1009a-duo-DB (full) | 0.135 | 0.036 | 0.065 | 0.035 | 3/2 | 0.0184 | 6 | 108.41 | - |
| window-campp-pa | IS1009a-L-room (full) | 0.203 | 0.094 | 0.030 | 0.079 | 5/4 | 0.0580 | 6 | 114.72 | - |
| window-campp-pa | IS1009a-R-room (full) | 0.203 | 0.094 | 0.030 | 0.079 | 5/4 | 0.0384 | 6 | 108.05 | - |
| window-campp-pa-ov | IS1009a-duo-AC-dev (dev) | 0.042 | 0.015 | 0.027 | 0.000 | 2/2 | 0.0357 | 6 | 106.38 | - |
| window-campp-pa-ov | IS1009a-duo-DB-dev (dev) | 0.107 | 0.000 | 0.107 | 0.000 | 2/2 | 0.0108 | 6 | 104.26 | - |
| window-campp-pa-ov | IS1009a-L-room-dev (dev) | 0.047 | 0.008 | 0.014 | 0.026 | 4/4 | 0.0427 | 6 | 103.83 | - |
| window-campp-pa-ov | IS1009a-R-room-dev (dev) | 0.047 | 0.008 | 0.014 | 0.026 | 4/4 | 0.0487 | 6 | 102.59 | - |
| window-campp-pa-ov | IS1009a-duo-AC (full) | 0.061 | 0.027 | 0.025 | 0.008 | 2/2 | 0.0286 | 6 | 106.72 | - |
| window-campp-pa-ov | IS1009a-duo-DB (full) | 0.255 | 0.033 | 0.193 | 0.028 | 3/2 | 0.0156 | 6 | 103.31 | - |
| window-campp-pa-ov | IS1009a-L-room (full) | 0.201 | 0.051 | 0.043 | 0.107 | 5/4 | 0.0407 | 6 | 99.29 | - |
| window-campp-pa-ov | IS1009a-R-room (full) | 0.201 | 0.051 | 0.043 | 0.107 | 5/4 | 0.0459 | 6 | 104.38 | - |
| window-resnet34 | IS1009a-duo-AC-dev (dev) | 0.052 | 0.044 | 0.009 | 0.000 | 2/2 | 0.0469 | 6 | 88.57 | - |
| window-resnet34 | IS1009a-duo-DB-dev (dev) | 0.082 | 0.015 | 0.068 | 0.000 | 2/2 | 0.0187 | 6 | 86.37 | - |
| window-resnet34 | IS1009a-L-room-dev (dev) | 0.070 | 0.059 | 0.008 | 0.004 | 4/4 | 0.0465 | 6 | 85.78 | - |
| window-resnet34 | IS1009a-R-room-dev (dev) | 0.070 | 0.059 | 0.008 | 0.004 | 4/4 | 0.0509 | 6 | 84.26 | - |
| window-resnet34 | IS1009a-duo-AC (full) | 0.121 | 0.108 | 0.008 | 0.006 | 2/2 | 0.0446 | 6 | 83.04 | - |
| window-resnet34 | IS1009a-duo-DB (full) | 0.205 | 0.085 | 0.037 | 0.082 | 3/2 | 0.0282 | 6 | 81.74 | - |
| window-resnet34 | IS1009a-L-room (full) | 0.232 | 0.159 | 0.018 | 0.055 | 5/4 | 0.0552 | 6 | 84.27 | - |
| window-resnet34 | IS1009a-R-room (full) | 0.232 | 0.159 | 0.018 | 0.055 | 5/4 | 0.0581 | 6 | 81.0 | - |
| window-ahc-campp | IS1009a-duo-AC-dev (dev) | 0.056 | 0.050 | 0.006 | 0.000 | 2/2 | 0.0555 | 6 | 89.1 | - |
| window-ahc-campp | IS1009a-duo-DB-dev (dev) | 0.066 | 0.016 | 0.051 | 0.000 | 2/2 | 0.0363 | 6 | 93.61 | - |
| window-ahc-campp | IS1009a-L-room-dev (dev) | 0.196 | 0.071 | 0.005 | 0.119 | 3/4 | 0.0557 | 6 | 96.44 | - |
| window-ahc-campp | IS1009a-R-room-dev (dev) | 0.196 | 0.071 | 0.005 | 0.119 | 3/4 | 0.0380 | 6 | 97.6 | - |
| window-ahc-campp | IS1009a-duo-AC (full) | 0.130 | 0.118 | 0.006 | 0.006 | 3/2 | 0.0417 | 6 | 97.63 | - |
| window-ahc-campp | IS1009a-duo-DB (full) | 0.160 | 0.101 | 0.033 | 0.026 | 5/2 | 0.0187 | 6 | 95.84 | - |
| window-ahc-campp | IS1009a-L-room (full) | 0.268 | 0.170 | 0.017 | 0.081 | 7/4 | 0.0313 | 6 | 93.94 | - |
| window-ahc-campp | IS1009a-R-room (full) | 0.268 | 0.170 | 0.017 | 0.081 | 7/4 | 0.0456 | 6 | 91.81 | - |
| online-campp | IS1009a-duo-AC-dev (dev) | 0.079 | 0.044 | 0.009 | 0.027 | 4/2 | 0.0250 | 6 | 106.97 | 1.5 |
| online-campp | IS1009a-duo-DB-dev (dev) | 0.082 | 0.015 | 0.068 | 0.000 | 4/2 | 0.0152 | 6 | 108.41 | 1.5 |
| online-campp | IS1009a-L-room-dev (dev) | 0.090 | 0.059 | 0.008 | 0.024 | 4/4 | 0.0227 | 6 | 108.41 | 1.5 |
| online-campp | IS1009a-R-room-dev (dev) | 0.090 | 0.059 | 0.008 | 0.024 | 4/4 | 0.0255 | 6 | 109.1 | 1.5 |
| online-campp | IS1009a-duo-AC (full) | 0.135 | 0.108 | 0.008 | 0.019 | 4/2 | 0.0233 | 6 | 108.29 | 1.5 |
| online-campp | IS1009a-duo-DB (full) | 0.152 | 0.085 | 0.037 | 0.030 | 4/2 | 0.0152 | 6 | 109.51 | 1.5 |
| online-campp | IS1009a-L-room (full) | 0.268 | 0.159 | 0.018 | 0.091 | 4/4 | 0.0270 | 6 | 110.46 | 1.5 |
| online-campp | IS1009a-R-room (full) | 0.268 | 0.159 | 0.018 | 0.091 | 4/4 | 0.0267 | 6 | 115.33 | 1.5 |
| pyannote-campp | IS1009a-duo-AC-dev (dev) | 0.212 | 0.006 | 0.029 | 0.178 | 2/2 | 0.1117 | 6 | 90.6 | - |
| pyannote-campp | IS1009a-duo-DB-dev (dev) | 0.588 | 0.000 | 0.159 | 0.430 | 3/2 | 0.0604 | 6 | 90.23 | - |
| pyannote-campp | IS1009a-L-room-dev (dev) | 0.332 | 0.003 | 0.012 | 0.318 | 2/4 | 0.1104 | 6 | 90.68 | - |
| pyannote-campp | IS1009a-R-room-dev (dev) | 0.332 | 0.003 | 0.012 | 0.318 | 2/4 | 0.1216 | 6 | 93.39 | - |
| pyannote-campp | IS1009a-duo-AC (full) | 0.423 | 0.020 | 0.031 | 0.373 | 4/2 | 0.1110 | 6 | 93.25 | - |
| pyannote-campp | IS1009a-duo-DB (full) | 0.646 | 0.026 | 0.212 | 0.408 | 6/2 | 0.1041 | 6 | 99.08 | - |
| pyannote-campp | IS1009a-L-room (full) | 0.588 | 0.043 | 0.055 | 0.490 | 5/4 | 0.1569 | 6 | 109.98 | - |
| pyannote-campp | IS1009a-R-room (full) | 0.588 | 0.043 | 0.055 | 0.490 | 5/4 | 0.1652 | 6 | 113.73 | - |
| sortformer | IS1009a-duo-AC-dev (dev) | 0.110 | 0.086 | 0.012 | 0.011 | 4/2 | 1.1417 | 6 | 76.04 | 10.0 |
| sortformer | IS1009a-duo-DB-dev (dev) | 0.539 | 0.060 | 0.059 | 0.419 | 3/2 | 1.5694 | 6 | 80.67 | 10.0 |
| sortformer | IS1009a-L-room-dev (dev) | 0.299 | 0.071 | 0.001 | 0.227 | 4/4 | 1.9215 | 6 | 95.28 | 10.0 |
| sortformer | IS1009a-R-room-dev (dev) | 0.299 | 0.071 | 0.001 | 0.227 | 4/4 | 1.5170 | 6 | 113.46 | 10.0 |
| sortformer | IS1009a-duo-AC (full) | 0.131 | 0.087 | 0.032 | 0.013 | 4/2 | 0.5288 | 6 | 106.83 | 10.0 |
| sortformer | IS1009a-duo-DB (full) | 0.294 | 0.117 | 0.144 | 0.033 | 4/2 | 0.1791 | 6 | 103.64 | 10.0 |
| sortformer | IS1009a-L-room (full) | 0.251 | 0.097 | 0.096 | 0.058 | 4/4 | 0.0681 | 6 | 83.66 | 10.0 |
| sortformer | IS1009a-R-room (full) | 0.251 | 0.097 | 0.096 | 0.058 | 4/4 | 0.0718 | 6 | 77.86 | 10.0 |


## 4. What runs on the GB10 and how

Nothing in the recommended path needs a GPU: Silero VAD (2.3 MB), WeSpeaker CAM++ (29 MB, 7.2 M params) and
pyannote/segmentation-3.0 (6 MB) run on onnxruntime CPU with aarch64 wheels (onnxruntime 1.31.0, sherpa-onnx 2.0.0,
kaldi-native-fbank 1.22.4, scipy 1.18.1 all have cp312 manylinux aarch64 wheels on PyPI). The whole offline path is
RTF 0.02-0.07 at 6 x86 threads under a 90+ load average, so the GB10's 20 Arm cores are more than enough; the GPU is
free for ASR / VLM. The only GPU candidate is Sortformer: NVIDIA ships the same weights as a GGUF for NeMo-Speech.cpp
(`nemo-speech diarize meeting.wav --model diar_streaming_sortformer_4spk-v2.q8_0.gguf`, CUDA build), which is the way to
run it on the GB10 — NeMo + torch cu13 on aarch64 was NOT tried here (CPU-only box). Given its DER here (section 3) it is
not the recommendation for this meeting; keep it as the overlap-aware fallback to re-check on the GB10 GPU.

`config/models.toml` block for the room brain (live path; replaces the per-segment momentum clustering with the tuned
causal labeller; the registry builds `python:` providers from `RoleSpec`, options go to the constructor):

```toml
[profiles.gb10.diarization]                  # who is talking (voice-A, voice-B, ... session-only), causal, CPU
provider = "python:unspoken.transcribe.diar:OnlineWindowDiarizer"
label = "WeSpeaker CAM++ (ONNX, CPU) + online window labelling"
options = { embedder = "campp", join = 0.6, mint = 0.4, max_voices = 4, win_s = 1.5, hop_s = 0.75, threads = 2 }
```

Offline (post-meeting) transcript: call `python -m unspoken.transcribe.diar --method window --vad pyannote` on the
recording's audio track (`--overlap` for the far-field room mic), or import `prepare_windows` / `finish_windows`.

## 5. Not verified, contract requests, deps

Not verified here:
- Any GB10 / aarch64 execution. Wheel availability was checked on PyPI listings only (sherpa-onnx 2.0.0 aarch64 cp312:
  yes; torch 2.9.0 manylinux_2_28 aarch64: yes; nemo_toolkit 3.0.0: pure python).
- Sortformer on GPU, and its low-latency presets on the full cases (CPU: RTF 0.07-0.53 on the full cases and 1.1-1.9 on
  the 180 s excerpts for the 10 s preset, 1.8 for the 1.04 s preset on a dev case; load 76-113).
- Sortformer's training set lists "AMI Meeting Corpus" without the split; IS1009a belongs to the standard AMI test
  partition, but a trained-on-this-meeting bias cannot be excluded from the card.
- `read_wav`'s PyAV fallback (non-16 kHz input); every test input here is 16 kHz mono PCM.
- Timings: every number was taken with 6 threads while five other workstreams ran (1-min load 76-115 on 32 cores, in
  `scores.json` as `load1_at_start`); on an idle box the window method ran at RTF 0.03 (duo-AC-dev, load 2.6).
- The online labeller's label latency is reported as window + lookback x hop (1.5 s); compute per window is ~15 ms at
  6 threads, not included.

Contract change requests (integrator-owned files, not edited):
- `contracts.DiarResult`: none needed. Overlapping `VoiceTurn`s are produced by `--overlap`; the scorer handles them.
- `eval/transcribe/score.py::score_diar`: please add `spk_count_err = hyp_speakers - ref_speakers` to the JSON (the bench
  computes it itself today) and expose `der()`'s per-speaker mapping so the fuse workstream can see which voice was
  dropped when `hyp_speakers > ref_speakers`.
- `unspoken/sensors/pipeline.py`: the live VAD closes segments after 500 ms of silence and cuts at 20 s; the online
  labeller would see speaker changes inside a 20 s segment only through `OnlineWindowDiarizer`'s per-segment windows
  (one label per segment is still emitted). To publish a label per 0.75 s hop, the pipeline needs a
  `audio.voice` sub-segment event (request for the live workstream).

Deps added (`requirements/transcribe-diar.txt`): `sherpa-onnx==2.0.0` (aarch64 wheel: yes). Optional, not pinned in the
file: `nemo_toolkit[asr]==3.0.0`, `torch==2.9.0`, `torchaudio==2.9.0` (CPU index) in a separate venv
(`$KIROCREW_SCRATCH/diar/nemo-venv`); installs in ~4 min with the shared uv cache.

Licences: Silero VAD MIT; WeSpeaker models Apache-2.0; pyannote/segmentation-3.0 MIT (the converted ONNX from sherpa-onnx
keeps the MIT notice); sherpa-onnx Apache-2.0; NVIDIA Streaming Sortformer CC-BY-4.0 (commercial OK, attribution);
`pyannote/speaker-diarization-community-1` was NOT used (gated on Hugging Face, no token on this box). AMI corpus CC BY 4.0.

## 6. Open risks

- Miss dominates the remaining DER of the winner on the full cases (0.065 / 0.036 / 0.094 of 0.089 / 0.135 / 0.203;
  with Silero activity 0.10-0.17 of 0.13-0.23): the reference counts overlapped speech, laughter and very short
  backchannels; a single-label window system cannot score overlap. The `--overlap` second voice
  (pyannote activity) is the lever: it cut the room's miss, but adds false alarms on the artificial gated duo mixes.
- The online labeller is sensitive to its thresholds on far-field audio (dev vs full and sweep-to-sweep swings of
  several DER points); the 4-voice cap keeps it honest for fusion but confusion stays ~0.09 on the room. Sortformer's
  low-latency preset on the GB10 GPU is the better live candidate if its GPU DER is confirmed.
- Spectral speaker count is right on every dev case (2/2, 4/4); on the full cases it adds one voice on the room (both
  front ends) and, with the pyannote front end, one on duo-DB. The extra voice holds little time (confusion stays
  0.035-0.079), but fusion should expect a spare `voice-X`. NME-SC on fewer than ~20 windows (sub-minute input) works
  on small kNN graphs — guarded against over-fragmented graphs, but AHC (`--cluster ahc --threshold 0.85`) is the
  safer choice there.
- Timings were taken on a saturated shared box; RTFs are upper bounds.
