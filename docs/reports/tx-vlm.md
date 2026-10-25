# tx-vlm: a vision LLM as the judge of who is talking

Question: can a small local vision-language model, shown a few frames from an utterance, say who is talking
(left person / right person / neither on camera)? Measured on the AMI IS1009a cases against the live
pipeline's MotionASD heuristic (baseline) and against the trivial floor "the left person always talks".

## 1. What was built

- `unspoken/transcribe/vlm_frames.py`: K frames per time unit; views `faces` (grid: a row per frame, columns = LEFT /
  RIGHT face crop from YuNet, seat zones as fallback), `faces-diff` (plus |frame − frame 0.2 s earlier| per crop),
  `full` (whole frames). Pure numpy/OpenCV; `ArrayFrames` for tests, `VideoFrames` (cv2) for mp4.
- `unspoken/transcribe/vlm_judge.py`: the judge. GBNF grammar `left | right | neither | both` on llama-server, answer
  probability from token logprobs at the answer position (pre-grammar softmax over the vocabulary, renormalised
  over the four words); prompts `plain`, `cues` (what a talking mouth looks like, off-camera = neither), `json` (must
  describe each mouth before naming the speaker). Units = baseline ASR segments or 1.5 s windows inside them.
  Writes `units.json`, `asd.json` (faces `seat-left`/`seat-right`, one frame per unit with P(seat speaking)),
  `transcript.json` (baseline text; speaker `person-left`/`person-right`, or the segment's voice label on
  `neither`) for four binds (`none` = VLM alone, `voice` = voice-label affinity when the VLM is below
  `--min-conf`, `fuse` = ½ VLM + ½ the voice label's duration-weighted affinity, `voice-seat` = the audio voice
  per segment, the VLM only decides which seat each voice sits in), and `scores*.json`. `--rescore` rebuilds
  transcripts and scores from an existing `units.json` (no model calls). `--constraint guided_choice --model-id
  ...` sends the same request to vLLM (GB10). `--fake left` = the floor.
- `unspoken/transcribe/vlm_server.py`: starts `llama-server --mmproj` on 127.0.0.1 (port 8795; 8790 was taken) with
  `-t/-tb` capped, `-np 1`, 8k context.
- `tests/transcribe/test_vlm.py`: 13 tests on synthetic frames and canned server responses (no model, no server);
  one live test behind `UNSPOKEN_LIVE=1`.
- `requirements/transcribe-vlm.txt`: no new Python packages (stdlib client; OpenCV/numpy from sensors.txt); the
  model list with licences.
- Models (public GGUFs under `dev-models/vlm/`): Qwen3-VL-2B-Instruct Q8_0 + mmproj Q8_0
  ([ggml-org/Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF), Apache-2.0),
  SmolVLM2-500M-Video-Instruct Q8_0 + mmproj f16
  ([ggml-org/SmolVLM2-500M-Video-Instruct-GGUF](https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF),
  Apache-2.0), Qwen3-VL-4B-Instruct Q4_K_M + mmproj Q8_0
  ([Qwen/Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF), Apache-2.0; downloaded,
  not measured: no CPU time left on the shared box). Not used: Qwen2.5-VL-3B-Instruct (Qwen Research licence,
  non-commercial), gemma-3-4b-it (Gemma terms allow commercial use with restrictions; skipped for time).
- Runs: `testdata/ami-runs/vlm/<method>/<case>/` (`units.json`, `asd.json`, `transcript*.json`, `scores*.json`);
  `testdata/ami-runs/vlm/BEST/<case>/` = copies of the recommended run per case with the `voice-seat` bind as
  `transcript.json` / `scores.json` (README.md there lists them).

## 2. How to run

```bash
cd unspoken-wt/tx-vlm
UV_CACHE_DIR=.../2026-10-15-strands-decider-research/hosting/uv-cache scripts/dev-venv.sh
uv pip install --python .venv/bin/python -r requirements/sensors.txt -r requirements/transcribe-eval.txt \
    -r requirements/transcribe-vlm.txt
M=../../dev-models/vlm
# model files (once):
#   hf download ggml-org/Qwen3-VL-2B-Instruct-GGUF Qwen3-VL-2B-Instruct-Q8_0.gguf mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf --local-dir $M/qwen3-vl-2b
# server (CPU, 6 threads; foreground):
.venv/bin/python -m unspoken.transcribe.vlm_server --model $M/qwen3-vl-2b/Qwen3-VL-2B-Instruct-Q8_0.gguf \
    --mmproj $M/qwen3-vl-2b/mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf --port 8795 --threads 6
# one case (baseline asr.json / diar.json are picked up from testdata/ami-baseline/<case>/):
T=../../testdata
OMP_NUM_THREADS=6 .venv/bin/python -m unspoken.transcribe.vlm_judge --case $T/ami/IS1009a-duo-AC-dev \
    --out $T/ami-runs/vlm/q3vl2b-faces-diff-k3-cues-seg/IS1009a-duo-AC-dev --server http://127.0.0.1:8795 \
    --view faces-diff --frames 3 --prompt cues --unit segments --bind voice-seat,none,voice,fuse --threads 6
# floors:  --fake left   (no server)        windows:  --unit windows --window 1.5
# other binds from a finished run, no model:  ... --out <same OUT> --rescore --bind voice-seat
# scores:  python -m eval.transcribe.score --ref $T/ami/IS1009a-duo-AC-dev --hyp .../transcript.json
# tests:   .venv/bin/python -m pytest -q   (whole suite: 854 passed, 11 skipped here; tests/models also need
#          `uv pip install -r requirements/models.txt`)   live check: UNSPOKEN_LIVE=1 UNSPOKEN_VLM_SERVER=http://127.0.0.1:8795
```

Method names: `<model>-<view>-k<frames>-<prompt>-<seg|win1.5>`; `transcript.json`/`scores.json` are the first bind
on the command line, `transcript.<bind>.json` / `scores.<bind>.json` the others (the runs in the table were made
with `voice` first, so their recommended output is `transcript.voice-seat.json`; `BEST/` has it as
`transcript.json`).

## 3. Results

All VLM rows: Qwen3-VL-2B-Instruct Q8_0 + mmproj Q8_0 on llama.cpp b11869 (x86 CPU), `-t 6 -tb 6`, one request per
unit, grammar `left|right|neither|both`, probabilities from logprobs. Baseline text and voices from
`testdata/ami-baseline`. `seat_acc VLM alone` = bind `none`; the other metric columns are for bind `voice-seat`
(audio decides which voice, the VLM decides which seat each voice sits in). `judge acc / balanced` = per-unit
accuracy of the VLM's answer against the reference majority speaker of that unit (analysis script, not a
method; balanced = mean recall over left / right / neither). `s/unit` and `RTF` (wall / case audio seconds)
were measured while the shared box ran at the `load` shown (32 cores; six workstreams); the clean reference
latency at load 6 is in section 5. Dev = 180 s excerpt (tuning), full = whole meeting (reported once).

| method | case | split | seat_acc VLM alone | seat_acc voice-seat | sa_acc | cpWER | DER | judge acc / balanced | s/unit | RTF | thr | load |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (live pipeline, MotionASD votes) | duo-AC-dev | dev | - | 0.913 | 0.913 | 0.235 | 0.096 | - | - | - | - | - |
| baseline (live pipeline, MotionASD votes) | duo-DB-dev | dev | - | 0.918 | 0.918 | 0.266 | 0.162 | - | - | - | - | - |
| baseline (live pipeline, MotionASD votes) | L-room-dev | dev | - | 0.675 | 0.791 | 0.455 | 0.151 | - | - | - | - | - |
| baseline (live pipeline, MotionASD votes) | R-room-dev | dev | - | 0.068 | 0.791 | 0.455 | 0.153 | - | - | - | - | - |
| baseline (live pipeline, MotionASD votes) | duo-AC | full | - | 0.857 | 0.857 | 0.321 | 0.165 | - | - | - | - | - |
| baseline (live pipeline, MotionASD votes) | duo-DB | full | - | 0.033 | 0.911 | 0.247 | 0.176 | - | - | - | - | - |
| baseline (live pipeline, MotionASD votes) | L-room | full | - | 0.808 | 0.726 | 0.497 | 0.272 | - | - | - | - | - |
| baseline (live pipeline, MotionASD votes) | R-room | full | - | 0.517 | 0.726 | 0.488 | 0.272 | - | - | - | - | - |
| const-left (floor: left always talks) | duo-AC-dev | dev | 0.881 | - | 0.881 | 0.303 | 0.213 | - | - | - | - | - |
| const-left (floor: left always talks) | duo-DB-dev | dev | 0.575 | - | 0.575 | 0.861 | 0.496 | - | - | - | - | - |
| const-left (floor: left always talks) | L-room-dev | dev | 0.857 | - | 0.665 | 0.735 | 0.364 | - | - | - | - | - |
| const-left (floor: left always talks) | R-room-dev | dev | 0.575 | - | 0.665 | 0.735 | 0.366 | - | - | - | - | - |
| const-left (floor: left always talks) | duo-AC | full | 0.849 | - | 0.849 | 0.334 | 0.179 | - | - | - | - | - |
| const-left (floor: left always talks) | duo-DB | full | 0.683 | - | 0.683 | 0.634 | 0.469 | - | - | - | - | - |
| const-left (floor: left always talks) | L-room | full | 0.841 | - | 0.585 | 0.777 | 0.429 | - | - | - | - | - |
| const-left (floor: left always talks) | R-room | full | 0.653 | - | 0.585 | 0.774 | 0.429 | - | - | - | - | - |
| q3vl2b-faces-diff-k3-cues-seg | duo-AC-dev | dev | 0.893 | 0.940 | 0.940 | 0.186 | 0.082 | 0.95 / 0.97 | 14.0 | 1.65 | 6 | 47 |
| q3vl2b-faces-diff-k3-cues-seg | duo-DB-dev | dev | 0.986 | 0.986 | 0.986 | 0.165 | 0.137 | 0.92 / 0.95 | 15.0 | 1.09 | 6 | 58 |
| q3vl2b-faces-diff-k3-cues-seg | L-room-dev | dev | 0.917 | 0.917 | 0.711 | 0.592 | 0.248 | 0.86 / 0.67 | 42.2 | 5.16 | 6 | 113 |
| q3vl2b-faces-diff-k3-cues-seg | R-room-dev | dev | 0.562 | 0.575 | 0.708 | 0.589 | 0.255 | 0.09 / 0.33 | 19.1 | 2.34 | 6 | 34 |
| q3vl2b-faces-k1-cues-seg | duo-AC-dev | dev | 0.933 | 0.940 | 0.940 | 0.189 | 0.084 | 0.86 / 0.93 | 6.8 | 0.79 | 6 | 52 |
| q3vl2b-faces-k1-cues-seg | duo-DB-dev | dev | 0.945 | 0.945 | 0.945 | 0.215 | 0.162 | 0.77 / 0.86 | 9.3 | 0.67 | 6 | 97 |
| q3vl2b-faces-k1-cues-seg | L-room-dev | dev | 0.857 | 0.857 | 0.665 | 0.735 | 0.364 | 0.82 / 0.33 | 10.1 | 1.24 | 6 | 38 |
| q3vl2b-faces-k1-cues-seg | R-room-dev | dev | 0.562 | 0.575 | 0.662 | 0.732 | 0.371 | 0.14 / 0.35 | 3.7 | 0.45 | 6 | 76 |
| q3vl2b-faces-k1-cues-seg | duo-AC | full | 0.845 | 0.867 | 0.867 | 0.306 | 0.158 | 0.94 / 0.97 | 3.4 | 0.37 | 6 | 51 |
| q3vl2b-faces-k1-cues-seg | duo-DB | full | 0.699 | 0.932 | 0.932 | 0.207 | 0.172 | 0.61 / 0.61 | 4.8 | 0.39 | 6 | 42 |
| q3vl2b-faces-k1-cues-win1.5 | duo-DB-dev | dev | 0.945 | 0.945 | 0.945 | 0.215 | 0.162 | 0.80 / 0.82 | 1.8 | 0.26 | 6 | 59 |
| q3vl2b-faces-k1-plain-seg | duo-AC-dev | dev | 0.933 | 0.940 | 0.940 | 0.186 | 0.082 | 0.95 / 0.97 | 0.6 | 0.07 | 6 | 63 |
| q3vl2b-faces-k3-cues-seg | duo-AC-dev | dev | 0.885 | 0.940 | 0.940 | 0.186 | 0.082 | 0.86 / 0.93 | 12.1 | 1.42 | 6 | 35 |
| q3vl2b-faces-k3-cues-seg | duo-DB-dev | dev | 0.767 | 0.986 | 0.986 | 0.165 | 0.137 | 0.77 / 0.86 | 6.1 | 0.45 | 6 | 54 |
| q3vl2b-faces-k3-cues-seg | L-room-dev | dev | 0.857 | 0.857 | 0.665 | 0.735 | 0.364 | 0.82 / 0.33 | 33.3 | 4.08 | 6 | 100 |
| q3vl2b-faces-k3-cues-seg | duo-AC | full | 0.834 | 0.867 | 0.867 | 0.306 | 0.158 | 0.89 / 0.86 | 14.6 | 1.60 | 6 | 36 |
| q3vl2b-faces-k3-cues-seg | duo-DB | full | 0.742 | 0.939 | 0.939 | 0.197 | 0.163 | 0.73 / 0.77 | 22.4 | 1.81 | 6 | 93 |
| q3vl2b-full-k3-cues-seg | duo-AC-dev | dev | 0.893 | 0.940 | 0.940 | 0.186 | 0.082 | 0.86 / 0.93 | 13.1 | 1.54 | 6 | 48 |
| smol500m-faces-k3-cues-seg | duo-AC-dev | dev | 0.881 | 0.881 | 0.881 | 0.303 | 0.213 | 0.95 / 0.50 | 8.1 | 0.96 | 6 | 36 |

Smol = SmolVLM2-500M (answers "left" with ~0.42 / 0.36 / 0.08 / 0.13 probabilities whatever the picture: it
does not read the task at this size; dropped after one case). `faces-k1-plain` shares its images with
`faces-k1-cues`, so the server's prompt cache made it 0.6 s/unit: the accuracy is a fair comparison, the latency
is not.

**What the numbers say**
1. The floor matters: "the left person always talks" scores seat_acc 0.849 on duo-AC and 0.841 on L-room, within
   0.01 of the baseline. The baseline's seat_acc is speaker A; its recall of the right person is 0.14 (duo-AC
   full) and its duo-DB binding is swapped (0.033).
2. With ~100 px faces (duo cases) the 2B reads who is talking: per-segment accuracy 0.89-0.95 (balanced
   0.86-0.97) on duo-AC, 0.73-0.77 on the harder duo-DB. Three frames beat one frame on duo-DB full (0.73 vs
   0.61) and tie on duo-AC; the full-frame view (0.893) lost to the face grid (0.933, duo-AC-dev, VLM alone).
3. At ~30 px faces (room views) the plain face grid is blind: "left" for every unit, 0.75-1.0 confidence, both
   k=1 and k=3. The motion tiles (`faces-diff`) fix the left/right question there: L-room-dev right-person word
   recall 0.97 (34/35), seat_acc 0.917 vs 0.857 (floor, k1, k3) and 0.675 (baseline). They also leave duo
   accuracy intact (duo-DB-dev 0.92/0.95, duo-AC-dev 0.95/0.97), at ~3x the prompt tokens (767 vs 249).
4. "neither" is never chosen (0 of 110 room units), although 20-77 % of room words are off camera: R-room-dev
   (A and C off camera, 77 % of words) stays at the floor whatever the view, and sa_acc there (0.71) comes only
   from the voice labels. The 2B cannot say "nobody on camera is talking"; that decision must come from the
   audio side (a voice no seat claims) or a trained ASD.
5. Binding beats per-segment judging. `voice-seat` lifts duo-DB full from 0.74 (VLM alone) to 0.939 and
   duo-AC full from 0.83 to 0.867, with sa_acc and cpWER better than the baseline on both (duo-DB: 0.939 /
   0.197 vs 0.911 / 0.247; duo-AC: 0.867 / 0.306 vs 0.857 / 0.321). Per-segment word attribution is capped by
   the baseline VAD segmentation (segments up to 20 s mix speakers: ceiling ~0.86 on duo-AC full, ~0.90 on
   duo-DB), which is why a judge that is right on 94 % of segments scores 0.845 alone.
6. Cost on this CPU: 1.9 s (k=1, 249 tokens) to 9 s (faces-diff k=3, 767 tokens) per unit unloaded, i.e. RTF
   0.2-1.0 against speech time at 6 threads; prompt processing (~100 tok/s) is the whole cost, the answer is
   1-2 tokens.

**Winner**: `faces-diff k=3 cues` + `voice-seat` (BEST for the dev cases; the full duo cases carry the same
`voice-seat` numbers with the face grid alone, run before the motion result was in). It is the only setup that
works on both face sizes; when faces are >= 100 px, `faces k=3` (or k=1 at half the cost) is enough.

## 4. On the GB10

**Which setup, plainly.** On the GB10 use the main model (Qwen3.6-35B-A3B on vLLM, the registry's `vlm` role) as the
judge with `view=faces-diff, frames=3, prompt=cues, unit=segments` (or 1.5 s windows once the ASR has word
timings), `bind=voice-seat`, through `--constraint guided_choice` (vLLM's `guided_choice: ["left","right","neither","both"]`
replaces the llama.cpp GBNF grammar; `logprobs`/`top_logprobs` work the same, so `asd.json` keeps real
probabilities). One request is ~770 prompt tokens (896x700 px grid; ~250 for the plain one-frame face grid).
Three frames beat one on the harder duo case; `full` frames lost to the face grid; the motion tiles are what
made the 30 px room faces readable. Do not add a 2B/4B VLM on the GPU for this: it is the same picture-reading
task the 35B already does better, and a second model competes for the same memory bandwidth.

**Is it worth it vs a dedicated ASD model?** Not as the primary ASD. Accuracy: with faces ~100 px the 2B judges
segments at 0.86-0.97 balanced accuracy (full duo-AC), which beats MotionASD (right-person recall 0.14) but a
trained ASD (TalkNet / Light-ASD: AVA-ActiveSpeaker mAP ~92-94, ~1 M params, <1 ms per face per frame on a GPU)
is better per frame and gives 25 fps timing the VLM cannot (one call per segment or 1.5 s window at best). At
30 px faces (room view) the 2B needs the motion tiles to see anything and still never says "neither"; a trained
ASD degrades too but is built for that scale. Latency: Qwen3.6-35B-A3B prefill of ~300 tokens plus the vision tower is of the order of
0.1-0.3 s per call on the GB10 (3 B active params, NVFP4; estimate, not measured here), i.e. fine at one call
per utterance (~0.3-1 calls/s in a meeting) and too slow/expensive at 5 Hz. GPU contention: every call takes
the shared vLLM instance for that long and delays the room brain's own replies; at 1.5 s windows that is
10-20 % of the GPU, at one call per utterance a few per cent. Verdict: run a dedicated ASD at frame rate
(tx-asd), and call the VLM judge **per utterance as the voice->seat binder and tie-breaker**: accumulate its answers per voice label
(`bind=voice-seat`), (re)bind a voice to a seat when the affinity is clear, and ask it on single utterances only
when the ASD and the binding disagree. That is exactly what fixes the baseline's swapped binding on duo-DB
(seat_acc 0.033 -> 0.939). Do not rely on its "neither": a voice whose affinity never clears a margin is the
off-camera signal, not the model's word.

**Config block** (proposal for `config/models.toml`; the registry has no `vlm_judge` role yet, so these are
options on the `vlm` role that `vlm_judge.LlamaServerJudge` reads: `constraint`, `view`, `frames`, `prompt`,
`unit`, `bind`, `min_conf`):

```toml
[profiles.gb10.vlm]                          # Qwen3.6 on vLLM: reads the whiteboard AND judges who is talking
provider = "openai"
url = "http://127.0.0.1:8000/v1"
model = "nvidia/Qwen3.6-35B-A3B-NVFP4"
label = "Qwen3.6 35B-A3B NVFP4 on vLLM (vision, GPU)"
options = { api_key = "none", max_tokens = 400, temperature = 0, timeout_s = 120,
            judge = { constraint = "guided_choice", view = "faces-diff", frames = 3, prompt = "cues",
                      unit = "segments", bind = "voice-seat", min_conf = 0.6, tile = 224, zoom = 2.2,
                      max_calls_per_s = 1.0, use = "tie-breaker" } }   # use = "tie-breaker" | "primary" | "off"
```

Fallback if the main model must not be shared: llama.cpp built with CUDA on the GB10 (aarch64, sm_121) serving
`Qwen3-VL-4B-Instruct-Q4_K_M.gguf` + `mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf` with `-ngl 99` on another port:
`python -m unspoken.transcribe.vlm_server --model ... --mmproj ... --port 8795 --gpu-layers 99`, then
`provider = "openai", url = "http://127.0.0.1:8795/v1"` with `constraint = "grammar"`. GGUF files are
architecture-independent; only the binary is rebuilt.

## 5. Not verified, contract requests, deps

**Not verified here**
- Qwen3.6-35B-A3B as the judge (no GPU on this box): accuracy and latency above are extrapolations from the 2B.
  The `guided_choice` request path is unit-tested for shape only; it never hit a vLLM.
- Qwen3-VL-4B: downloaded (Apache-2.0) but not run: the shared box was at load 90-110 on 32 cores for the whole
  window and the 2B alone took 15-33 s per unit (4.6 s at load 6).
- Full-case numbers for the room views were not run (no server time left at ~40 s per unit; 92 units each).
  Without motion tiles the dev runs equal `const-left`; with them L-room-dev reached 0.917, so the full L-room
  number is unknown rather than predictable.
- `faces-diff` on the full cases, the `json` prompt, `k=6`, a mirrored-grid control for the left bias, SmolVLM2
  beyond one dev case, 1.5 s windows on more than one case (and windows only reach the transcript once the ASR
  has word timings: here they are averaged back to the segment).
- Timings: every per-unit latency in the table was measured with the box at load 35-113 (uptime), 6 threads
  per server; the clean reference numbers (load 6, 6 threads, Qwen3-VL-2B Q8_0, this build) are: faces k=1
  1.9 s (249 prompt tokens), faces k=3 4.6 s (492), full k=1 2.9 s (326), faces-diff k=3 9.0 s (767).
  Prompt processing dominates (~100 tokens/s); the answer is 1-2 tokens.

**Contract change requests (integrator)**
1. `eval/transcribe/score.py`: add a per-ref-seat recall / confusion to the output (`seat_recall: {left, right,
   off}`) and a balanced seat accuracy. `seat_acc` alone is dominated by speaker A: "the left person always
   talks" scores 0.849 on duo-AC and 0.841 on L-room, within 0.01 of the baseline.
2. Document (or score) the per-segment ceiling: with the baseline VAD segmentation (segments up to 20 s that
   mix speakers) the best per-segment speaker choice caps seat_acc at ~0.86 (duo-AC full), ~0.85 (L-room),
   ~0.90 (duo-DB). Any per-segment judge, VLM or ASD, is bounded by it; word-timed ASR (tx-asr) + window units
   is what lifts the ceiling. A `--unit windows` run already writes per-window probabilities to `asd.json`.
3. `contracts.FaceInfo.box` says pixels, the baseline writes 0..1 fractions; this workstream writes pixels. Pick
   one (I suggest fractions, like the live `video.speakers` events) and fix the baseline converter.
4. `AsdResult.fps` is a poor fit for irregular units (one frame per utterance); allow `fps = 0` meaning
   "irregular, use frame times".

**Deps added**: none in Python (stdlib client; OpenCV + numpy from `requirements/sensors.txt`, both with aarch64
wheels). Runtime: llama.cpp `llama-server` (C++, builds on aarch64 with CUDA) or vLLM on the GB10.

## 6. Open risks

- At ~30 px faces (wide room view) the 2B answers "left" with 0.8-1.0 confidence from the face crops alone, so
  its probabilities are not trustworthy there; the motion tiles rescue left-vs-right (L-room-dev 0.917) but
  not off-camera speech (R-room-dev at the floor). Camera placement (close-ups, or a higher-resolution room
  camera) is the real fix.
- Position bias: with the LEFT column first in the grid and the prompt, the unsure answer is "left" (every room
  unit without motion tiles). A control (mirror the grid, swap the labels, average) would measure and cancel
  it; not done.
- Long VAD segments mix speakers; the judge answers for the dominant face and the minority speaker's words are
  lost at segment granularity (duo-AC full, VLM alone: right-person word recall 0.45 while 83-100 % of the
  units it answered "right" for were right).
- Per-call cost on the shared GPU model; a burst of short utterances could queue behind a long chat reply.
  Rate-limit (`max_calls_per_s`) and skip the judge when audio diarization is confident.
- `neither` is never chosen by the 2B (0 of 110 room units although 20-77 % of those words are off camera);
  the fusion must not trust a seat answer when the audio says a new voice or both faces are still.
