# tx-asr: transcription with word timings

Real test on AMI IS1009a (CC BY 4.0), x86_64 dev box, CPU only, 6 threads per process, six workstreams sharing
32 cores (1-minute load average 3 to 100 during these runs; every row says which). References were read only by
the scorer and by `eval/transcribe/asr_timing.py`. Tuning (window length, int8 vs fp32, timing offset) was done
on the `-dev` excerpts; full-case numbers were produced once.

## 1. What I built

| Path | What |
|---|---|
| `unspoken/transcribe/asr.py` | Entry point: `python -m unspoken.transcribe.asr --in CASE/video.mp4|audio.wav --out asr.json --method NAME [--threads 6] [--opt k=v]`, Python API `transcribe(audio_f32_16k, method, threads=6, **opts) -> AsrResult`, method registry, mp4/wav loading (PyAV for mp4), timing capture (wall, audio s, RTF, threads, load average before/after). |
| `unspoken/transcribe/asr_chunks.py` | Speech windows from the repo's own Silero VAD segmenter (`unspoken.sensors.vad`, the live pipeline's settings) and `merge_chunks` (join neighbours up to 30 s, gaps <= 1 s). |
| `unspoken/transcribe/asr_whisper.py` | faster-whisper (CTranslate2 int8) with `word_timestamps=True`: `whole_file` (built-in VAD filter) and `chunked` (our Silero chunks, what the live path does). Sizes base.en, small.en, large-v3-turbo, distil-large-v3.5. |
| `unspoken/transcribe/asr_parakeet.py` | NVIDIA Parakeet TDT 0.6B v3 on CPU via onnx-asr (numpy + onnxruntime, no torch), fp32 by default; token emission frames (80 ms) grouped into words with start/end/conf. |
| `unspoken/transcribe/asr_server.py` | `--method server`: the same Silero windows posted to an OpenAI-compatible `/v1/audio/transcriptions` server (GB10: NeMo-Speech.cpp Parakeet on the GPU), words from `verbose_json`. |
| `eval/transcribe/asr_timing.py` | Word-timing quality vs `ref.words.json`: difflib alignment on normalised words; median / mean / p90 / signed median of start and end errors, share of starts within the 0.25 s DER collar, match rate. |
| `eval/transcribe/run_asr.sh` | One method x N cases -> `testdata/ami-runs/asr/<method>/<case>/{asr,scores,timing}.json` with the load average logged. |
| `eval/transcribe/asr_table.py` | The results tables below, straight from those JSON files. |
| `tests/transcribe/test_asr.py` | 14 unit tests (fakes for the model backends, the vendored Silero on the synthetic 3-voice fixture, a loopback fake speech server, CLI round trip, timing eval) + 1 live test (`UNSPOKEN_LIVE=1`). |
| `requirements/transcribe-asr.txt` | `onnx-asr==0.12.0` (pure python). Everything else was already pinned in `requirements/sensors.txt` / `transcribe-eval.txt`. |
| `testdata/ami-runs/asr/<method>/<case>/` (not in git) | `asr.json` for every method x case run, plus `scores.json` and `timing.json`; `testdata/ami-runs/asr/BEST` names the recommendation. |

## 2. How to run

```bash
cd /local/home/zhangnai/kirocrew-sessions/2026-10-23-dell-nvidia-hackathon/unspoken-wt/tx-asr
UV_CACHE_DIR=/local/home/zhangnai/kirocrew-sessions/2026-10-15-strands-decider-research/hosting/uv-cache scripts/dev-venv.sh
uv pip install --python .venv/bin/python -r requirements/sensors.txt -r requirements/transcribe-eval.txt -r requirements/transcribe-asr.txt

T=/local/home/zhangnai/kirocrew-sessions/2026-10-23-dell-nvidia-hackathon/testdata
export OMP_NUM_THREADS=6
# one case, one method (first use downloads the model: whisper into ~/.cache/huggingface, Parakeet ONNX into
# <session>/dev-models/asr/parakeet-tdt-0.6b-v3[-int8]; override with UNSPOKEN_DEV_MODELS=/dir)
.venv/bin/python -m unspoken.transcribe.asr --in $T/ami/IS1009a-duo-AC-dev/audio.wav --out out/asr.json --method parakeet-tdt-0.6b-v3 --threads 6
.venv/bin/python -m eval.transcribe.score --ref $T/ami/IS1009a-duo-AC-dev --asr out/asr.json          # WER
.venv/bin/python -m eval.transcribe.asr_timing --ref $T/ami/IS1009a-duo-AC-dev --asr out/asr.json     # word timing
# all cases of one method, scored, into testdata/ami-runs/asr/<method>/<case>/
eval/transcribe/run_asr.sh parakeet-tdt-0.6b-v3                                   # the four -dev excerpts
eval/transcribe/run_asr.sh parakeet-tdt-0.6b-v3 IS1009a-duo-AC IS1009a-duo-DB IS1009a-L-room IS1009a-R-room
.venv/bin/python -m eval.transcribe.asr_table --runs $T/ami-runs/asr              # the tables below
.venv/bin/python -m unspoken.transcribe.asr --list                                  # methods
# options: --opt beam_size=1 (whisper), --opt max_s=20 --opt gap_s=0.5 --opt offset_s=-0.04 --opt batch_size=4 (Parakeet),
#          --opt quantization=int8 is the same as -int8; --method server --opt url=http://127.0.0.1:8770/v1
.venv/bin/python -m pytest -q                                                       # whole suite
```

Python: `from unspoken.transcribe.asr import load_audio, transcribe; res = transcribe(load_audio(path), "parakeet-tdt-0.6b-v3", threads=6)`;
`res.segments[i].words[j]` has `start`, `end`, `text`, `conf`; `res.timing` has `wall_s, audio_s, rtf, model_load_s, threads, load1_start, load1_end`.

## 3. Results

Columns: WER from `eval.transcribe.score --asr`; word-start median |error| and the share of aligned word starts
within 0.25 s (`asr_timing.py`); matched = share of reference words with an aligned hypothesis word (all
speakers, on and off camera); RTF = decode wall / audio seconds with model loading excluded, 6 threads, with the
1-minute load average at start -> end (32 cores shared by six workstreams; rows at load > 32 are slowed by the
box, not by the method). Whisper methods: int8, beam 5. Parakeet: greedy, fp32 unless `-int8`. L-room and R-room
share the same far-field audio, so their rows are identical by construction.

Live baseline for reference (`testdata/ami-baseline`, faster-whisper base.en per live VAD segment, no word
timings): WER dev 0.091 / 0.046 / 0.126 / 0.126 (duo-AC / duo-DB / L-room / R-room), full 0.229 / 0.208 / 0.338 / 0.335.

### Dev excerpts (180 s each; tuning data)

| method | case | WER | start med err | starts <0.25 s | matched | RTF | threads | load1 |
|---|---|---|---|---|---|---|---|---|
| parakeet-tdt-0.6b-v3 | IS1009a-duo-AC-dev | 0.072 | 0.088 | 94.7% | 90.1% | 0.055 | 6 | 12.46->14.39 |
| parakeet-tdt-0.6b-v3 | IS1009a-duo-DB-dev | 0.038 | 0.082 | 95.9% | 91.2% | 0.029 | 6 | 14.39->18.06 |
| parakeet-tdt-0.6b-v3 | IS1009a-L-room-dev | 0.125 | 0.092 | 94.9% | 88.1% | 0.053 | 6 | 18.06->20.46 |
| parakeet-tdt-0.6b-v3 | IS1009a-R-room-dev | 0.125 | 0.092 | 94.9% | 88.1% | 0.051 | 6 | 20.46->21.03 |
| parakeet-tdt-0.6b-v3-int8 | IS1009a-duo-AC-dev | 0.053 | 0.076 | 99.2% | 91.2% | 0.042 | 6 | 3.12->4.99 |
| parakeet-tdt-0.6b-v3-int8 | IS1009a-duo-DB-dev | 0.076 | 0.064 | 97.2% | 88.7% | 0.021 | 6 | 4.99->6.03 |
| parakeet-tdt-0.6b-v3-int8 | IS1009a-L-room-dev | 0.140 | 0.074 | 97.0% | 84.4% | 0.048 | 6 | 6.03->8.67 |
| parakeet-tdt-0.6b-v3-int8 | IS1009a-R-room-dev | 0.140 | 0.074 | 97.0% | 84.4% | 0.046 | 6 | 10.14->12.46 |
| fw-large-v3-turbo | IS1009a-duo-AC-dev | 0.042 | 0.060 | 80.9% | 86.1% | 0.141 | 6 | 3.12->10.14 |
| fw-large-v3-turbo | IS1009a-duo-DB-dev | 0.051 | 0.080 | 74.6% | 83.8% | 0.054 | 6 | 10.14->12.82 |
| fw-large-v3-turbo | IS1009a-L-room-dev | 0.120 | 0.060 | 79.5% | 81.6% | 0.160 | 6 | 12.82->21.03 |
| fw-large-v3-turbo | IS1009a-R-room-dev | 0.120 | 0.060 | 79.5% | 81.6% | 0.130 | 6 | 20.46->20.61 |
| fw-distil-large-v3.5 | IS1009a-duo-AC-dev | 0.083 | 0.030 | 89.5% | 83.5% | 0.232 | 6 | 36.34->36.27 |
| fw-distil-large-v3.5 | IS1009a-duo-DB-dev | 0.114 | 0.040 | 88.1% | 83.8% | 0.354 | 6 | 36.25->49.55 |
| fw-distil-large-v3.5 | IS1009a-L-room-dev | 0.143 | 0.040 | 86.1% | 79.3% | 0.598 | 6 | 49.55->51.08 |
| fw-distil-large-v3.5 | IS1009a-R-room-dev | 0.143 | 0.040 | 86.1% | 79.3% | 0.586 | 6 | 51.08->51.42 |
| fw-small.en | IS1009a-duo-AC-dev | 0.057 | 0.050 | 85.8% | 85.3% | 0.052 | 6 | 3.12->4.99 |
| fw-small.en | IS1009a-duo-DB-dev | 0.089 | 0.065 | 73.4% | 80.0% | 0.022 | 6 | 4.99->6.03 |
| fw-small.en | IS1009a-L-room-dev | 0.184 | 0.050 | 84.6% | 75.4% | 0.068 | 6 | 6.99->10.14 |
| fw-small.en | IS1009a-R-room-dev | 0.184 | 0.050 | 84.6% | 75.4% | 0.063 | 6 | 10.14->12.82 |
| fw-base.en | IS1009a-duo-AC-dev | 0.129 | 0.110 | 72.0% | 79.9% | 0.057 | 6 | 32.66->36.34 |
| fw-base.en | IS1009a-duo-DB-dev | 0.152 | 0.130 | 70.8% | 81.2% | 0.031 | 6 | 36.34->35.51 |
| fw-base.en | IS1009a-L-room-dev | 0.219 | 0.110 | 73.0% | 73.4% | 0.044 | 6 | 35.51->35.36 |
| fw-base.en | IS1009a-R-room-dev | 0.219 | 0.110 | 73.0% | 73.4% | 0.043 | 6 | 35.36->35.09 |
| vad-fw-small.en | IS1009a-duo-AC-dev | 0.087 | 0.050 | 90.4% | 83.9% | 0.160 | 6 | 12.82->21.03 |
| vad-fw-small.en | IS1009a-duo-DB-dev | 0.051 | 0.073 | 86.4% | 82.5% | 0.080 | 6 | 21.03->21.03 |
| vad-fw-small.en | IS1009a-L-room-dev | 0.181 | 0.046 | 89.2% | 75.9% | 0.145 | 6 | 21.03->24.1 |
| vad-fw-small.en | IS1009a-R-room-dev | 0.181 | 0.046 | 89.2% | 75.9% | 0.231 | 6 | 24.1->36.34 |
| vad-fw-base.en | IS1009a-duo-AC-dev | 0.080 | 0.105 | 84.1% | 85.0% | 0.061 | 6 | 21.03->20.61 |
| vad-fw-base.en | IS1009a-duo-DB-dev | 0.101 | 0.112 | 81.2% | 80.0% | 0.048 | 6 | 20.61->23.16 |
| vad-fw-base.en | IS1009a-L-room-dev | 0.201 | 0.094 | 80.8% | 75.1% | 0.066 | 6 | 23.16->24.58 |
| vad-fw-base.en | IS1009a-R-room-dev | 0.201 | 0.094 | 80.8% | 75.1% | 0.125 | 6 | 24.61->32.66 |
| vad-fw-large-v3-turbo | IS1009a-duo-AC-dev | 0.057 | 0.057 | 86.4% | 86.4% | 1.265 | 6 | 20.61->49.68 |
| vad-fw-large-v3-turbo | IS1009a-duo-DB-dev | 0.076 | 0.086 | 81.8% | 82.5% | 1.312 | 6 | 49.68->45.28 |

### Full cases (~830 s each; run once, no tuning)

| method | case | WER | start med err | starts <0.25 s | matched | RTF | threads | load1 |
|---|---|---|---|---|---|---|---|---|
| parakeet-tdt-0.6b-v3 | IS1009a-duo-AC | 0.181 | 0.074 | 95.6% | 82.7% | 0.073 | 6 | 47.62->44.4 |
| parakeet-tdt-0.6b-v3 | IS1009a-duo-DB | 0.165 | 0.074 | 94.7% | 83.1% | 0.043 | 6 | 44.4->43.2 |
| parakeet-tdt-0.6b-v3 | IS1009a-L-room | 0.262 | 0.078 | 94.8% | 75.0% | 0.070 | 6 | 43.2->36.86 |
| parakeet-tdt-0.6b-v3 | IS1009a-R-room | 0.262 | 0.078 | 94.8% | 75.0% | 0.093 | 6 | 36.86->55.88 |
| parakeet-tdt-0.6b-v3-int8 | IS1009a-duo-AC | 0.254 | 0.068 | 97.8% | 75.0% | 0.066 | 6 | 37.36->49.55 |
| parakeet-tdt-0.6b-v3-int8 | IS1009a-duo-DB | 0.166 | 0.070 | 97.0% | 82.6% | 0.032 | 6 | 49.55->50.88 |
| parakeet-tdt-0.6b-v3-int8 | IS1009a-L-room | 0.282 | 0.068 | 97.4% | 72.4% | 0.063 | 6 | 50.81->49.62 |
| parakeet-tdt-0.6b-v3-int8 | IS1009a-R-room | 0.282 | 0.068 | 97.4% | 72.4% | 0.073 | 6 | 49.62->52.6 |
| fw-large-v3-turbo | IS1009a-duo-AC | 0.174 | 0.060 | 80.7% | 79.5% | 0.369 | 6 | 51.91->42.61 |
| fw-large-v3-turbo | IS1009a-duo-DB | 0.163 | 0.060 | 78.8% | 80.6% | 0.184 | 6 | 40.88->42.43 |
| fw-large-v3-turbo | IS1009a-L-room | 0.270 | 0.050 | 85.3% | 70.2% | 1.868 | 6 | 42.43->99.29 |
| fw-large-v3-turbo | IS1009a-R-room | 0.270 | 0.050 | 85.3% | 70.2% | 0.579 | 6 | 99.29->60.48 |
| fw-small.en | IS1009a-duo-AC | 0.202 | 0.050 | 82.0% | 76.2% | 0.179 | 6 | 51.42->48.83 |
| fw-small.en | IS1009a-duo-DB | 0.170 | 0.050 | 81.8% | 78.6% | 0.059 | 6 | 48.83->44.02 |
| fw-small.en | IS1009a-L-room | 0.276 | 0.050 | 87.0% | 69.8% | 0.145 | 6 | 42.98->37.8 |
| fw-small.en | IS1009a-R-room | 0.276 | 0.050 | 87.0% | 69.8% | 0.819 | 6 | 37.8->93.14 |
| fw-base.en | IS1009a-duo-AC | 0.235 | 0.100 | 76.3% | 73.6% | 1.170 | 6 | 93.14->98.74 |
| fw-base.en | IS1009a-duo-DB | 0.203 | 0.090 | 74.1% | 76.2% | 0.161 | 6 | 98.74->101.17 |
| fw-base.en | IS1009a-L-room | 0.325 | 0.090 | 79.1% | 65.6% | 0.189 | 6 | 101.17->83.66 |
| fw-base.en | IS1009a-R-room | 0.328 | 0.090 | 79.1% | 65.6% | 0.181 | 6 | 83.66->63.04 |
| vad-fw-small.en | IS1009a-duo-AC | 0.200 | 0.054 | 89.1% | 77.5% | 0.278 | 6 | 41.64->66.69 |
| vad-fw-small.en | IS1009a-duo-DB | 0.178 | 0.048 | 91.6% | 78.0% | 0.195 | 6 | 66.69->55.55 |
| vad-fw-small.en | IS1009a-L-room | 0.311 | 0.052 | 87.4% | 67.0% | 0.248 | 6 | 55.55->36.27 |
| vad-fw-small.en | IS1009a-R-room | 0.311 | 0.052 | 87.4% | 67.0% | 0.152 | 6 | 36.27->13.66 |
| vad-fw-base.en | IS1009a-duo-AC | 0.241 | 0.092 | 82.2% | 75.0% | 0.120 | 6 | 61.91->48.83 |
| vad-fw-base.en | IS1009a-duo-DB | 0.208 | 0.084 | 85.2% | 76.8% | 0.070 | 6 | 48.83->46.79 |
| vad-fw-base.en | IS1009a-L-room | 0.325 | 0.088 | 81.5% | 66.4% | 0.106 | 6 | 46.79->47.44 |
| vad-fw-base.en | IS1009a-R-room | 0.325 | 0.088 | 81.5% | 66.4% | 0.127 | 6 | 47.44->49.67 |

Not run on the full cases: `fw-distil-large-v3.5` (lost to large-v3-turbo on every dev case) and
`vad-fw-large-v3-turbo` (chunked whisper pays a 30 s encoder pass per VAD segment; RTF > 1 on dev). The box sat
at load 40-116 for the second hour, so these were cut to leave CPU to the other workstreams. `vad-fw-small.en`
on the full cases (0.200 / 0.178 / 0.311 / 0.311) sits between base.en and whole-file small.en, as on dev.

### What the numbers say

- **Parakeet TDT 0.6B v3 (fp32, onnx-asr) wins on word timing by a wide margin and ties whisper large-v3-turbo
  on WER at a fraction of the CPU.** 95-99% of its word starts are within the 0.25 s collar (median |error|
  0.07-0.09 s, p90 0.17-0.21 s, signed median +0.03 s so no offset correction was applied); the best whisper
  (large-v3-turbo) puts 75-85% within the collar with a long tail (p90 0.43-0.48 s, up to 1.4 s for base.en).
  Fusion splits segments at word boundaries, so the tail is what matters, not the median.
- Full cases, WER: Parakeet fp32 0.181 / 0.165 / 0.262 / 0.262 (duo-AC / duo-DB / L-room / R-room),
  large-v3-turbo 0.174 / 0.163 / 0.270 / 0.270, small.en 0.202 / 0.170 / 0.276 / 0.276, base.en 0.235 / 0.203 /
  0.325 / 0.328 (the live baseline, base.en per live segment: 0.229 / 0.208 / 0.338 / 0.335). Parakeet beats the
  live baseline by 0.05-0.08 absolute WER on every case and carries word timings the baseline does not have.
- WER on the dev excerpts: Parakeet 0.04-0.13, large-v3-turbo 0.04-0.12, small.en 0.06-0.18, base.en 0.13-0.22.
  On the full cases every method lands at 0.16-0.28: the full recordings contain overlapping speech that a
  single-stream ASR cannot transcribe (14.9% of duo-AC reference words and 29.9% of room-case words overlap
  another speaker; the duo-AC dev excerpt has 0.7%), and the match rates (70-83%) show it as deletions. That
  floor is the same for all methods and is where diarization/fusion, not ASR, has to earn the words back.
- **int8 Parakeet is not usable here**: the int8 ONNX export returned zero tokens for a 10 s window of clear
  speech (duo-AC 552-563 s) and 125 instead of 159 tokens for the next one; WER 0.254 vs 0.181 (fp32) on full
  duo-AC, 0.282 vs 0.262 on the room cases, for no decisive speed win (dev RTF 0.042 at load 3-5 vs fp32 0.055
  at load 12-14). fp32 needs 2.4 GB of RAM for the encoder; fine on both boxes.
- RTF at 6 threads and load < 20 (dev rows): Parakeet 0.03-0.055, whisper base.en 0.03-0.06, small.en
  0.02-0.07, large-v3-turbo 0.05-0.16, distil-large-v3.5 0.23+ (measured only at load 36-51). On the full cases
  at load 37-56 Parakeet stayed at 0.04-0.09 while large-v3-turbo went 0.18-0.37 and then 0.58-1.87 when the load
  passed 90: it is the model that suffers first when the box is busy. Chunked whisper (the live path) is 2-4x
  slower than whole-file whisper of the same size because every VAD segment costs a full 30 s encoder pass;
  chunked large-v3-turbo hit RTF 1.3 at load 20-50, i.e. not real time on this CPU. Parakeet's cost is
  proportional to the audio in the window, so it keeps its RTF on short live segments.
- `vad-fw-base.en` (our Silero chunks + base.en, i.e. the live path with word timings added) reproduces the live
  baseline on the full cases (0.241 / 0.208 / 0.325 / 0.325 vs 0.229 / 0.208 / 0.338 / 0.335), which confirms the
  chunker matches what the room brain feeds its ASR today; whole-file base.en is no better (0.235-0.328).
- distil-large-v3.5 was worse than large-v3-turbo on every dev case (0.08-0.14) and slower under the load it
  ran at; it was dropped from the full runs.

### Recommendation (`testdata/ami-runs/asr/BEST`)

- (a) offline accuracy: `parakeet-tdt-0.6b-v3` (fp32). Best timing, WER within noise of large-v3-turbo on dev
  (0.072 / 0.038 / 0.125 vs 0.042 / 0.051 / 0.120) and on full (0.181 / 0.165 / 0.262 vs 0.174 / 0.163 / 0.270),
  at a third to a tenth of the CPU. The fusion workstream should consume
  `testdata/ami-runs/asr/parakeet-tdt-0.6b-v3/<case>/asr.json` (segments cut at pauses > 0.6 s, words inside).
- (b) live on this CPU: `parakeet-tdt-0.6b-v3` as well, fed one closed VAD segment at a time (the API accepts
  any float32 16 kHz clip; `asr_parakeet.transcribe` with `max_s=20` reproduces the live segment sizes). Its
  RTF is 0.03-0.07 at 6 threads, and unlike whisper it does not pay a 30 s encoder pass per short segment.
  If RAM mattered, `vad-fw-small.en` is the whisper fallback (dev WER 0.05-0.18, 86-90% of starts within 0.25 s).

## 4. On the GB10

GPU path: NeMo-Speech.cpp (NVIDIA's ggml runtime, arm64 + CUDA 13 build, `deploy/gb10/speech.sh asr`) serves
Parakeet TDT 0.6B v3 Q8 on `127.0.0.1:8770`, OpenAI route `POST /v1/audio/transcriptions`,
`response_format=verbose_json` returning `words` with start/end. The live `asr` role already points there
(`config/models.toml [profiles.gb10.asr]`, provider `openai-audio`); the offline transcriber reaches the same
server with `--method server`:

```bash
python -m unspoken.transcribe.asr --in CASE/video.mp4 --out asr.json --method server \
    --opt url=http://127.0.0.1:8770/v1 --opt model=parakeet-tdt-0.6b-v3
```

so the GB10 transcript is windowed and timed exactly like the CPU reference runs here. Config block (what the
registry needs; the live block already in `config/models.toml` is identical in content):

```toml
[profiles.gb10.asr]                          # NeMo-Speech.cpp: Parakeet TDT 0.6B v3 Q8 on the GPU
provider = "openai-audio"                    # POST /v1/audio/transcriptions (multipart WAV), verbose_json = words + timestamps
url = "http://127.0.0.1:8770/v1"             # health GET /ready (200 loaded, 503 loading); deploy/gb10/speech.sh asr
model = "parakeet-tdt-0.6b-v3"
label = "Parakeet TDT 0.6B v3 (GPU, NeMo-Speech.cpp)"
options = { response_format = "verbose_json", language = "en", diarization = false, timeout_s = 20, health = "/ready" }
# offline stage: python -m unspoken.transcribe.asr --method server --opt url=http://127.0.0.1:8770/v1
# CPU fallback on the GB10 (no server): --method parakeet-tdt-0.6b-v3 (onnx-asr fp32 on the 20 ARM cores,
#   ~2.4 GB RAM; expect RTF a few times the 0.05 measured here per 6 x86 threads) or --method vad-fw-small.en.
```

What runs where: faster-whisper (ctranslate2 4.8.3) has an aarch64 wheel but CPU-only; onnx-asr is pure python
and onnxruntime 1.31.0 has an aarch64 CPU wheel, so the fp32 Parakeet CPU path works on the GB10 unchanged (not
measured there). The GPU work is in NeMo-Speech.cpp. onnx-asr can also run the same ONNX on CUDA/TensorRT
providers, but that needs an aarch64 onnxruntime-gpu build, which the sensors workstream found is not on PyPI.

## 5. Not verified, contract requests, dependencies

Tests: `.venv/bin/python -m pytest -q` in this worktree: 857 passed, 9 skipped (the other workstreams' pinned
requirements installed into the venv so their tests import). One pre-existing sensors test
(`test_faster_whisper_gets_most_words`, asserts < 10 s latency per segment) failed once at load 94 with
`OMP_NUM_THREADS=1` and passed on rerun; it is a timing assertion, not a code change.

Not verified here (no GB10, no GPU, no aarch64):
- NeMo-Speech.cpp's `verbose_json` word fields. `asr_server.py` and the existing `openai_compat.py` both assume
  the OpenAI shape `words: [{word, start, end}]` (nested `segments[].words` also accepted); if the server names
  them differently, `words_from_body` is the one place to change. The method was tested against a loopback fake.
- Parakeet's word END semantics: onnx-asr exposes the emission frame of each token but not the TDT duration, so a
  word ends at the next token's frame (or +160 ms at a gap). Measured end error median 0.09 s, 86-92% within
  0.25 s; exposing the duration (a 30-line custom decode loop over onnx-asr's `_decode`) would tighten it.
- GB10 timings for any method; aarch64 installs of the pins.
- Whether the int8 Parakeet failure is the export or onnxruntime 1.31 x86 int8 kernels (not tested elsewhere).

Contract change requests (none blocking):
- `AsrResult.timing`: please document the keys this stage writes (`wall_s, audio_s, rtf, model_load_s, threads,
  load1_start, load1_end, host, method`) in `contracts.py` so fusion/live can rely on `rtf` and `threads`.
- `Word.conf`: documented as "model confidence"; here it is whisper's word probability or exp(mean token log-prob)
  for Parakeet (0..1, not calibrated across models). A one-line note in the contract would avoid misuse.
- `eval/transcribe/score.py --asr` gives WER only; `asr_timing.py` covers timing. If the integrator wants one
  call, `timing_report()` can be imported into `score_asr`.

Dependencies added: `onnx-asr==0.12.0` (MIT; py3-none-any, aarch64 yes). Model files (public, downloaded on
first use): `istupakov/parakeet-tdt-0.6b-v3-onnx` (CC-BY-4.0, export of `nvidia/parakeet-tdt-0.6b-v3`
CC-BY-4.0: commercial use OK with attribution), `Systran/faster-whisper-base.en|small.en`,
`mobiuslabsgmbh/faster-whisper-large-v3-turbo`, `distil-whisper/distil-large-v3.5-ct2` (all MIT). Silero VAD:
the vendored MIT ONNX. Nothing non-commercial. No torch.

## 6. Open risks

- Overlapping speech is the WER floor (15-30% of words on the full cases); ASR alone cannot recover it. Fusion
  should expect deletions during overlap, not misattributions.
- Parakeet TDT is offline per window: live latency = VAD close (500 ms) + decode (RTF x segment). For true
  streaming output NeMo-Speech.cpp offers Nemotron 3.5 ASR 0.6B streaming (`ASR_MODEL=nemotron35` in
  `deploy/gb10/speech.sh`); not measured here, for the live workstream.
- RTF figures above were taken on a box at load 3-100; absolute numbers at the day's load will differ, the
  ranking will not.
- `server` method and the GB10 config are untested against the real NeMo-Speech.cpp; first thing to run on the day.
