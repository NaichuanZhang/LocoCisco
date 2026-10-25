# tx-fuse: words + voices + faces -> per-person utterances (report, 2026-10-25)

Workstream `ws/tx-fuse`. Post-processing that turns `asr.json` + `diar.json` + `asd.json` into a `transcript.json`
where every line says who said it. Five named ablations, scored with `eval.transcribe.score` on all 8 AMI cases,
on the baseline inputs and on the best upstream inputs available at the time of writing.

## 1. What I built

| path | one line |
|---|---|
| `unspoken/transcribe/fuse.py` | CLI + library. Stage 1 `align` (word/segment -> most-overlapping voice turn, islands, merges), stage 2 `bind` (voice<->seat on whole-session rank-based co-occurrence, Hungarian, margin threshold, off-camera voices), stage 3 `av` (per-utterance audio x video decision with a self-calibrated video weight). Every decision and its evidence lands in `Transcript.meta`. |
| `unspoken/transcribe/fuse_decider.py` | Stage 4 `decider`: low-confidence utterances -> Strands Decider 2B, then 12B when unsure, as one multiple-choice "who said this line?" with the lines before/after, the voice's history and the evidence numbers. |
| `unspoken/transcribe/fuse_llm.py` | Stage 5 `llm`: DiarizationLM-style zero-shot boundary correction by a local llama-server (Qwen3-4B), words never change, only the boundary moves; replies that are not a permutation of the shown words are ignored. |
| `unspoken/transcribe/render.py` | `transcript.json` -> `[00:12.3] Left person: ...` text, SRT, markdown (CLI and `render_files`). |
| `tests/transcribe/test_fuse.py` | 21 unit tests on a synthetic 2-seat scene (no models, no servers, no data files): alignment, islands, Hungarian binding, full-room elimination, room-mode positive-evidence rule, seat-bias invariance, video override of a mislabelled turn, self-calibration, decider with `FakeDecider`, llm with a fake, render, CLI round trip. |
| `requirements/transcribe-fuse.txt` | numpy, scipy, httpx (all already in the eval/sensors sets; aarch64 wheels: yes). |
| `testdata/ami-runs/fuse/<method>-<inputs>/<case>/` (not in the repo) | `transcript.json`, `.txt`, `.srt`, `.md`, `scores.json` for every cell of the matrix below; `testdata/ami-runs/fuse/BEST` names the best method per case. |

Key design points (all inputs-only, no reference ever read):
- Alignment works with or without word timings: segments without words get character-proportional pseudo-words
  (`interp_words`), so a VAD-segment ASR and a word-timed ASR go through the same code. Groups are split at voice
  changes AND at ASR segment boundaries, so the per-utterance decision can re-attribute one sentence of a turn.
- Binding evidence `E[voice, seat]` = Mann-Whitney z of the seat's ASD score during the voice's turns against the
  same seat during the other voices' turns (ranks per seat over speech frames). Ranks remove a face's own bias (the
  failure that swapped duo-DB in the baseline: the right face scores higher *everywhere*); the two-sample form asks
  the right question ("is this face more active when THIS voice speaks than when the others do?"). Hungarian over
  the major voices (>= 5 % of speech time), accepted on a z margin >= 1; in a "full room" (no more major voices than
  seats) elimination is allowed (the inactive face identifies the speaker), otherwise a seat needs positive
  evidence; minor voices take a free seat only on strong evidence (z >= 3) so a 3-second over-clustered voice
  cannot steal a seat from the dominant speaker (this happened on duo-AC full before tiering). Unbound voices get
  an off-camera confidence from the same table.
- `av`: audio evidence = alignment confidence x binding confidence, mapped so that "no confidence" means
  uniform, never a vote against the voice's own pick (a bug that cost 2 cpWER points before I found it); video
  evidence = softmax of the seats' z during the utterance, used only when the camera itself is sure (top-two
  margin >= 0.6); the video weight is self-calibrated per session = how often those confident video votes agree with
  confidently bound voices, rescaled above chance and shrunk toward 0 for few samples. Surely-off-camera voices
  damp the video weight further.

## 2. How to run

```bash
cd /local/home/zhangnai/kirocrew-sessions/2026-10-23-dell-nvidia-hackathon/unspoken-wt/tx-fuse
UV_CACHE_DIR=/local/home/zhangnai/kirocrew-sessions/2026-10-15-strands-decider-research/hosting/uv-cache scripts/dev-venv.sh
uv pip install --python .venv/bin/python -r requirements/sensors.txt -r requirements/transcribe-eval.txt \
    -r requirements/transcribe-fuse.txt -r requirements/models.txt      # models.txt: the existing tests need `openai`
T=../../testdata
# one case, best method, readable output next to the json
.venv/bin/python -m unspoken.transcribe.fuse --asr $T/ami-runs/asr/fw-large-v3-turbo/IS1009a-duo-AC/asr.json \
    --diar $T/ami-runs/diar/window-campp/IS1009a-duo-AC/diar.json --asd $T/ami-baseline/IS1009a-duo-AC/asd.json \
    --out out/duo-AC/transcript.json --method bind --session IS1009a-duo-AC --render
.venv/bin/python -m eval.transcribe.score --ref $T/ami/IS1009a-duo-AC --hyp out/duo-AC/transcript.json
.venv/bin/python -m unspoken.transcribe.render out/duo-AC/transcript.json --format srt --out out/duo-AC/transcript.srt
# the other methods
.venv/bin/python -m unspoken.transcribe.fuse ... --method av                        # + per-utterance audio-visual
.venv/bin/python -m unspoken.transcribe.fuse ... --method decider --decider-url http://127.0.0.1:8765 --decider-big-url http://127.0.0.1:8766
dev-models/llama.cpp/llama-b11869/llama-server -m dev-models/gguf/Qwen3-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8798 -t 6 -c 2048 --parallel 1 -n 200 &
.venv/bin/python -m unspoken.transcribe.fuse ... --method llm --llm-url http://127.0.0.1:8798 --set llm_max_calls=40
# any Settings field: --set merge_gap=0.3 --set bind_min_margin=1.0 --set av_video_weight=0.8 (fixed instead of calibrated)
.venv/bin/python -m pytest -q                                                       # 862 passed, 10 skipped
```

## 3. Results

Inputs: `-base` = the live pipeline's baseline stage outputs (`testdata/ami-baseline`: ASR without word timings, one
voice per VAD segment, MotionASD). `-up` = the best upstream outputs present at 04:45 UTC: `asr/fw-large-v3-turbo`
(R-room full: `asr/parakeet-tdt-0.6b-v3`, turbo not run there yet), `diar/window-campp`, `asd/avcorr` on the dev
cases and the baseline MotionASD `asd.json` on the full cases (no upstream ASD had full-case outputs yet). All
numbers are the scorer's on the written transcripts. Dev rows were used for tuning; FULL rows were run once at the
end. RTF = fuse wall / audio seconds, in-process, 6 threads, machine load 1-min average 82-92 (six workstreams);
align/bind/av cost 0.02-0.4 s per case, i.e. RTF < 0.002. Decider/llm RTF is dominated by model latency under that
load (2B 7-35 s, 12B 25-40 s, Qwen3-4B 3-55 s per call; the shared 2B is quoted at 2.4 s idle).

| case | method | WER | cpWER | DER | sa_acc | seat_acc | RTF (fuse) | notes |
|---|---|---|---|---|---|---|---|---|
| duo-AC-dev (dev) | baseline | 0.091 | 0.235 | 0.096 | 0.913 | 0.913 | - |  |
| duo-AC-dev (dev) | align-base | 0.091 | 0.235 | 0.089 | 0.913 | 0.000 | 0.00001 | voices only, no seats |
| duo-AC-dev (dev) | bind-base | 0.091 | 0.235 | 0.089 | 0.913 | 0.913 | 0.00170 | voice-A left, voice-E right |
| duo-AC-dev (dev) | av-base | 0.091 | 0.235 | 0.089 | 0.913 | 0.913 | 0.00004 | video weight 0.21 |
| duo-AC-dev (dev) | decider-base | 0.091 | 0.242 | 0.089 | 0.909 | 0.909 | 0.6 (107 s wall) | asked 3, changed 1 (wrong), all 3 escalated to 12B, 35 s/call |
| duo-AC-dev (dev) | llm-base | 0.091 | 0.235 | 0.089 | 0.913 | 0.913 | 1.3 (235 s wall) | 9 boundaries, asked 7, moved 0, 33 s/call |
| duo-AC-dev (dev) | align-up | 0.042 | 0.053 | 0.156 | 0.972 | 0.000 | 0.00001 |  |
| duo-AC-dev (dev) | **bind-up** | 0.042 | 0.053 | 0.156 | 0.972 | 0.972 | 0.00159 | voice-A left, voice-B right |
| duo-AC-dev (dev) | av-up | 0.042 | 0.053 | 0.156 | 0.972 | 0.972 | 0.00009 | video weight 0.06 |
| duo-AC-dev (dev) | decider-up | 0.042 | 0.053 | 0.156 | 0.972 | 0.972 | 0.104 | asked 1, changed 0, 18.5 s/call |
| duo-AC-dev (dev) | llm-up | 0.042 | 0.053 | 0.156 | 0.972 | 0.972 | 0.207 | 2 boundaries, asked 2, moved 0, 18.5 s/call |
| duo-DB-dev (dev) | baseline | 0.165 | 0.266 | 0.162 | 0.918 | 0.918 | - |  |
| duo-DB-dev (dev) | align-base | 0.165 | 0.266 | 0.154 | 0.918 | 0.000 | 0.00001 |  |
| duo-DB-dev (dev) | bind-base | 0.165 | 0.266 | 0.154 | 0.918 | 0.918 | 0.00002 | voice-B left, voice-C right |
| duo-DB-dev (dev) | av-base | 0.165 | 0.266 | 0.154 | 0.918 | 0.918 | 0.00003 | video weight 0.44 |
| duo-DB-dev (dev) | decider-base | 0.165 | 0.266 | 0.154 | 0.918 | 0.918 | 0.5 (94 s wall) | asked 3, changed 0, 3 escalated, 31 s/call |
| duo-DB-dev (dev) | llm-base | 0.165 | 0.266 | 0.154 | 0.918 | 0.918 | 1.0 (184 s wall) | 9 boundaries, asked 6, moved 0, 30 s/call |
| duo-DB-dev (dev) | align-up | 0.051 | 0.051 | 0.110 | 0.959 | 0.000 | 0.00001 |  |
| duo-DB-dev (dev) | **bind-up** | 0.051 | 0.051 | 0.110 | 0.959 | 0.959 | 0.00006 | voice-A left, voice-B right |
| duo-DB-dev (dev) | av-up | 0.051 | 0.051 | 0.110 | 0.959 | 0.959 | 0.00006 | video weight 0.22 |
| duo-DB-dev (dev) | decider-up | 0.051 | 0.051 | 0.110 | 0.959 | 0.959 | 0.085 | asked 2, changed 0, 7.6 s/call |
| duo-DB-dev (dev) | llm-up | 0.051 | 0.051 | 0.110 | 0.959 | 0.959 | 0.030 | 2 boundaries, asked 2, moved 0, 1 unparsed, 2.7 s/call |
| L-room-dev (dev) | baseline | 0.207 | 0.455 | 0.151 | 0.791 | 0.675 | - |  |
| L-room-dev (dev) | align-base | 0.207 | 0.455 | 0.140 | 0.791 | 0.000 | 0.00002 |  |
| L-room-dev (dev) | bind-base | 0.207 | 0.455 | 0.140 | 0.791 | 0.056 | 0.00003 | wrong: two off-camera voices bound |
| L-room-dev (dev) | av-base | 0.207 | 0.455 | 0.140 | 0.791 | 0.056 | 0.00004 | video weight 0.0 |
| L-room-dev (dev) | decider-base | 0.207 | 0.455 | 0.140 | 0.791 | 0.056 | - | nothing below the threshold |
| L-room-dev (dev) | align-up | 0.120 | 0.117 | 0.080 | 0.948 | 0.000 | 0.00002 |  |
| L-room-dev (dev) | **bind-up** | 0.120 | 0.117 | 0.080 | 0.948 | 0.119 | 0.00008 | wrong binding (avcorr, 30 px faces) |
| L-room-dev (dev) | av-up | 0.120 | 0.117 | 0.080 | 0.948 | 0.119 | 0.00009 | video weight 0.31 |
| R-room-dev (dev) | baseline | 0.207 | 0.455 | 0.153 | 0.791 | 0.068 | - |  |
| R-room-dev (dev) | align-base | 0.207 | 0.455 | 0.142 | 0.791 | 0.000 | 0.00002 |  |
| R-room-dev (dev) | bind-base | 0.207 | 0.455 | 0.142 | 0.791 | 0.000 | 0.00003 | A (off camera) bound right: MotionASD z = +6 |
| R-room-dev (dev) | av-base | 0.207 | 0.455 | 0.142 | 0.791 | 0.000 | 0.00004 | video weight 0.3 |
| R-room-dev (dev) | decider-base | 0.207 | 0.455 | 0.142 | 0.791 | 0.000 | - | nothing below the threshold |
| R-room-dev (dev) | align-up | 0.120 | 0.117 | 0.080 | 0.948 | 0.000 | 0.00002 |  |
| R-room-dev (dev) | **bind-up** | 0.120 | 0.117 | 0.080 | 0.948 | 0.068 | 0.00008 | wrong binding (avcorr); with `asd/motion` instead: seat 0.890 |
| R-room-dev (dev) | av-up | 0.120 | 0.117 | 0.080 | 0.948 | 0.068 | 0.00010 | video weight 0.24 |
| duo-AC (FULL) | baseline | 0.229 | 0.321 | 0.165 | 0.857 | 0.857 | - |  |
| duo-AC (FULL) | align-base | 0.229 | 0.321 | 0.158 | 0.857 | 0.000 | 0.00001 |  |
| duo-AC (FULL) | bind-base | 0.229 | 0.321 | 0.158 | 0.857 | 0.857 | 0.00003 | voice-A left, voice-D right (= baseline) |
| duo-AC (FULL) | av-base | 0.229 | 0.321 | 0.158 | 0.857 | 0.857 | 0.00003 | video weight 0.17 |
| duo-AC (FULL) | align-up | 0.174 | 0.192 | 0.195 | 0.902 | 0.000 | 0.00001 |  |
| duo-AC (FULL) | **bind-up** | 0.174 | 0.192 | 0.195 | 0.902 | 0.902 | 0.00003 | voice-A left, voice-B right |
| duo-AC (FULL) | av-up | 0.174 | 0.192 | 0.194 | 0.903 | 0.903 | 0.00004 | video weight 0.20 |
| duo-DB (FULL) | baseline | 0.208 | 0.247 | 0.176 | 0.911 | **0.033** | - | voice->seat binding swapped |
| duo-DB (FULL) | align-base | 0.208 | 0.247 | 0.173 | 0.911 | 0.000 | 0.00001 |  |
| duo-DB (FULL) | bind-base | 0.208 | 0.247 | 0.173 | 0.911 | **0.911** | 0.00002 | swap fixed: voice-D left, voice-C right |
| duo-DB (FULL) | av-base | 0.208 | 0.247 | 0.173 | 0.911 | 0.911 | 0.00002 | video weight 0.15 |
| duo-DB (FULL) | align-up | 0.165 | 0.160 | 0.276 | 0.920 | 0.000 | 0.00001 |  |
| duo-DB (FULL) | **bind-up** | 0.165 | 0.160 | 0.276 | 0.920 | 0.920 | 0.00002 | voice-A left, voice-B right |
| duo-DB (FULL) | av-up | 0.165 | 0.224 | 0.367 | 0.875 | 0.875 | 0.00003 | video weight 0.46: MotionASD votes hurt |
| L-room (FULL) | baseline | 0.338 | 0.497 | 0.272 | 0.726 | 0.808 | - |  |
| L-room (FULL) | align-base | 0.338 | 0.497 | 0.265 | 0.726 | 0.000 | 0.00002 |  |
| L-room (FULL) | bind-base | 0.338 | 0.497 | 0.265 | 0.726 | 0.808 | 0.00003 | voice-C left (= A, correct), voice-F right (= D, off camera: wrong) |
| L-room (FULL) | av-base | 0.338 | 0.497 | 0.265 | 0.726 | 0.808 | 0.00004 | video weight 0.34 |
| L-room (FULL) | align-up | 0.262 | 0.287 | 0.265 | 0.782 | 0.000 | 0.00002 |  |
| L-room (FULL) | **bind-up** | 0.262 | 0.287 | 0.265 | 0.782 | 0.819 | 0.00003 | voice-B left, voice-E right |
| L-room (FULL) | av-up | 0.262 | 0.287 | 0.265 | 0.782 | 0.819 | 0.00005 | video weight 0.36 |
| R-room (FULL) | baseline | 0.335 | 0.488 | 0.272 | 0.726 | 0.517 | - |  |
| R-room (FULL) | align-base | 0.335 | 0.488 | 0.265 | 0.726 | 0.000 | 0.00002 |  |
| R-room (FULL) | bind-base | 0.335 | 0.488 | 0.265 | 0.726 | 0.402 | 0.00003 | voice-F left, voice-G right |
| R-room (FULL) | av-base | 0.335 | 0.488 | 0.265 | 0.726 | 0.402 | 0.00004 | video weight 0.12 |
| R-room (FULL) | align-up | 0.262 | 0.287 | 0.265 | 0.782 | 0.000 | 0.00002 | parakeet-tdt-0.6b-v3 ASR |
| R-room (FULL) | **bind-up** | 0.262 | 0.287 | 0.265 | 0.782 | 0.697 | 0.00003 | voice-A left, voice-D right |
| R-room (FULL) | av-up | 0.262 | 0.287 | 0.265 | 0.782 | 0.697 | 0.00005 | video weight 0.24 |

Winner: `bind` (alignment + whole-session binding) on the upstream inputs, for every case (`testdata/ami-runs/fuse/BEST`).
- Against today's live pipeline on the FULL cases: cpWER 0.321 -> 0.192 (duo-AC), 0.247 -> 0.160 (duo-DB),
  0.497 -> 0.287 (L-room), 0.488 -> 0.287 (R-room); speaker-attributed accuracy 0.857 -> 0.902, 0.911 -> 0.920,
  0.726 -> 0.782 (x2); seat accuracy 0.857 -> 0.902, 0.033 -> 0.920, 0.808 -> 0.819, 0.517 -> 0.697. Most of the
  cpWER gain is the upstream ASR (word timings, large-v3-turbo) and diarizer (window-campp, 2 clean voices instead
  of 7-8 over-clustered ones); fusion's own contribution is (a) carrying that quality through word-level alignment
  (align-up cpWER == bind-up cpWER: the alignment loses nothing) and (b) the binding.
- On the baseline inputs alone, the binding fixes the swapped duo-DB (seat 0.033 -> 0.911) and ties the baseline on
  duo-AC and L-room; R-room drops 0.517 -> 0.402 (the baseline's greedy raw-sum vote happened to pick the right
  dominant seat there; the rank statistic sees no on-camera evidence for the dominant voice because MotionASD at
  30 px is noise: see risks). DER improves by ~0.007 everywhere from the island/merge rules.
- `av` (per-utterance audio-visual) never beat `bind`. The self-calibration measures why: confident per-utterance
  video votes agree with confidently bound voices only 55-70 % of the time (chance 50 %) with MotionASD and avcorr,
  so the weight lands at 0.06-0.46 and, where it is highest (duo-DB full, 0.46), it costs 6 cpWER points. The stage
  is correct in the unit tests (a reliable ASD fixes a mislabelled turn, a noisy one gets weight ~0) and becomes
  useful the day an ASD is reliable per utterance (Light-ASD / TalkNet on the GB10); until then run `bind`.
- `decider`: with the baseline inputs 3 utterances per duo dev case fell below the 0.35 confidence threshold, all
  went 2B -> 12B (2B confidence 0.45-0.55 on a 2-way choice), 1 of 6 answers changed a speaker and it was wrong
  (cpWER 0.235 -> 0.242). With the upstream inputs 1-2 utterances per case were asked and nothing changed.
  Accuracy gained: none measurable; cost 7-35 s per call under the shared load. The question is sound (the state
  carries the neighbouring lines, voice history and both evidence vectors) but the decider sees only text about
  the evidence, and the hard utterances here are 1-2 word backchannels ("Okay.", "Mm-hmm") that no context decides.
- `llm` (Qwen3-4B zero-shot DiarizationLM): 13 boundaries asked on the baseline dev inputs, 4 on the upstream, 0
  moved (every reply reproduced the input tags; 1 unparsed). Neutral: it neither helps nor hurts; the words are
  preserved by construction. The published DiarizationLM gains come from a fine-tuned model; a 4B base model with
  a prompt does not reproduce them.

15 lines of the best duo-AC transcript (`testdata/ami-runs/fuse/bind-up/IS1009a-duo-AC/transcript.txt`, FULL case,
speech starts at 54.8 s; A = Left person, C = Right person):

```
[00:54.8] Left person: Okay.
[00:55.7] Left person: Everybody ready?
[00:57.5] Left person: I think the first thing we do is introduce ourselves.
[01:01.7] Left person: And miss name and what your function is.
[01:05.6] Left person: So maybe we start with you.
[01:21.2] Left person: Okay.
[01:27.5] Left person: And your name is?
[01:29.8] Right person: Okay.
[01:31.3] Right person: I'm Janavi.
[01:33.4] Right person: My role is industrial designer. And my responsibilities are
[01:39.9] Right person: deal with the technical functional designs and specifications of user interface and dealing with user interface design
[01:47.4] Left person: very good and as you already know I'm Betsy I'm the project manager for today so why don't we look at the presentation to see what we really are supposed to do yes
[02:02.7] Left person: opening acquaintance tool training well the tools I think we already I guess the tool is really our the computer as far as I can see we get in each of us will get instructions and we'll take it from their project plan that falls under the same heading pretty much I don't think we have any great discussion at this point here
[02:29.7] Left person: is what this thing should be this thing we are gonna design is a new remote control should be original trendy and of course user -friendly so
[02:45.3] Left person: maybe you want to make some notes of that okay
```

## 4. What runs on the GB10 and how

Fusion is pure numpy/scipy, CPU, < 0.5 s per 14-minute meeting: it runs on the GB10's CPU cores unchanged (numpy
2.5.4 and scipy 1.18.1 ship manylinux aarch64 cp312 wheels; httpx is pure Python). Nothing in this workstream
needs the GPU; the GPU is for the upstream ASR / diarization / ASD stages and for the decider and llm servers.
Methods `decider` and `llm` only need loopback URLs:

```toml
# config/models.toml additions (proposed; the registry's role names are the integrator's call)
[fuse]
method = "bind"                 # align | bind | av | decider | llm | all; "av" once an ASD is reliable per utterance
merge_gap = 0.3
bind_min_margin = 1.0
bind_major_share = 0.05
decider_threshold = 0.35        # only for method = "decider" / "all"
llm_max_calls = 400             # only for method = "llm" / "all"

[roles.decider]                 # Strands Decider 2B, GPU on the GB10 (vLLM / strands-decider serve), loopback only
provider = "strands-decider"
url = "http://127.0.0.1:8765"
model = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"

[roles.decider_big]
provider = "strands-decider"
url = "http://127.0.0.1:8766"
model = "StrandsAgents/strands-decider-12B-gemma4-v1-2610"

[roles.fuse_llm]                # DiarizationLM-style correction; llama.cpp built with CUDA on aarch64, or vLLM
provider = "openai"
url = "http://127.0.0.1:8798/v1"
model = "Qwen3-4B-Q4_K_M.gguf"  # dev-models/gguf; Apache-2.0 (Qwen3), commercial use OK
```

The CLI flags map 1:1 (`--method`, `--set key=value`, `--decider-url`, `--decider-big-url`, `--llm-url`). On the
GB10 the 2B decider at GPU speed (~0.1-0.3 s/call) makes `decider` cheap enough to run on every low-confidence line
in the live path; on this CPU box it was 7-35 s/call under load, which is why I capped it.

## 5. Not verified, contract requests, deps

Not verified:
- No GB10 run (x86 CPU only); aarch64 wheel status is from the pins' PyPI listings via the sensors/eval files, not
  from an install on the device.
- `decider` and `llm` were run only on the dev cases (baseline inputs: 4 and 2 cases; upstream inputs: the 2 duo
  cases) with call caps (20 / 40 and 8) because of 7-55 s per call on the saturated box; the full-case effect is
  extrapolated from "0 changes on dev".
- Full-case ASD: no upstream ASD had full-case outputs, so every FULL `-up` row uses the baseline MotionASD
  `asd.json`; `asd/avcorr` and `asd/motion` (25 fps) were only available on dev. L-room/R-room dev with `avcorr`
  bind wrongly, with `asd/motion` R-room-dev binds right (seat 0.890): the room-case seat numbers will move with
  whatever ASD the asd workstream picks as BEST; the fuser does not need to change for that.
- Timing numbers were taken at load average 82-92 (six workstreams); RTFs for align/bind/av are so small that this
  does not matter; the decider/llm latencies are not representative of an idle box.

Contract change requests (integrator-owned files, not edited):
1. `contracts.py` `Utterance`: add `evidence: dict[str, Any] | None = None` (or bless `Transcript.meta["evidence"]`,
   a list aligned with `utterances`, which is what I write now). The decider/llm stages and the live X-ray need the
   per-utterance audio/video evidence; a parallel list in `meta` is fragile under re-sorting.
2. `AsdResult.faces[].box` says "pixels of the input video" but the baseline writes 0..1 fractions; the fuser does
   not use boxes, but the live seat zones will. Pick one (fractions, since `sensors.toml` seats are fractions).
3. `score.py`: `seat_acc` is reported as 0.000 for a transcript with no on-camera speaker (method `align`), which
   reads like "all wrong"; `nan`/"n/a" when `hyp` has no seats would make the ablation table honest. Also consider
   reporting the voice->ref mapping for `voice-X` speakers in `mapping` (it does already) plus the number of
   hypothesised on-camera speakers.
4. Upstream `diar.json` producers: please write `conf` per turn (window-campp writes 1.0 everywhere); the alignment
   confidence uses it and the decider threshold would select better.

Deps added: `requirements/transcribe-fuse.txt` = numpy 2.5.4, scipy 1.18.1 (both manylinux aarch64 cp312 wheels),
httpx (pure Python). No model downloads, no torch. The `llm` method needs a llama-server binary (the x86 build in
dev-models; the GB10 needs an aarch64 CUDA build of llama.cpp or vLLM) and Qwen3-4B-Q4_K_M.gguf (Apache-2.0).
Note for whoever builds the venv: `tests/models/test_openai_compat.py` imports `openai`, which only
`requirements/models.txt` pins; the dev-venv + sensors + eval sets alone fail that one unrelated test.

## 6. Open risks

- MotionASD is not a speaking signal at 30 px (room cases) and is inverted on duo-DB full (the listening face
  moves more). Any co-occurrence binding is only as good as the ASD; the rank statistic fixes seat bias, not a
  signal that points at the listener. The elimination rule (full room) is what makes the 2-person demo robust.
- Dominant-speaker dilution: when one person talks 80 % of the time, their own face's "more active than usual"
  z is near 0 by construction and the other face's silence carries the decision. Fine for 2 seats; with 3+ seats
  and a dominant speaker the binding leans on the margin threshold.
- `av` self-calibration cannot detect a consistently wrong binding (agreement with a wrong binding is still
  agreement). Keep `bind` as the default until the ASD workstream shows per-utterance accuracy > 85 %.
- Decider questions are 2-3 way choices with short backchannels; a 2B answers at ~0.5 confidence and the 12B is
  then always consulted. If the live path adopts it, gate on utterance length (>= 3 words) as well as confidence.
- The llm stage trusts the model to return the exact words; it does (0 words changed) but at 3-55 s a call on CPU.
  It is off in BEST.

Servers I started: a private llama-server on 127.0.0.1:8798 for the `llm` runs; stopped at the end of this
workstream. The shared decider services were only queried.
