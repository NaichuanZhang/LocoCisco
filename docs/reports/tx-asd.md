# tx-asd: who is talking on camera (vision + audio-visual ASD)

Workstream report for the per-person transcriber real test (branch `ws/tx-asd`). Everything below was run on the
x86_64 dev box (32 cores, no GPU) with each process capped at 6 threads while six workstreams shared the machine
(load average 40-100 during the runs; every timing row says which).

## 1. What was built

| Path | What |
|---|---|
| `unspoken/transcribe/faces.py` | YuNet at native frame size (optional `--scale`), IoU tracker with stable ids, seats from fixed zones (left/right halves or a `sensors.toml`), coasting of briefly lost tracks, `FakeDetector` for tests, `crop_square` (Light-ASD's crop geometry), frame iterator. |
| `unspoken/transcribe/asd_lightasd.py` | Light-ASD (CVPR 2023, MIT) ported from the authors' repo: model, sha256-pinned weight fetch (`fetch` subcommand), MFCC front end identical to theirs, `LightASDScorer` (duration ensemble, sequential clips, cpu/cuda). |
| `unspoken/transcribe/asd.py` | CLI/driver -> `AsdResult` JSON. Methods `motion` (MotionASD at full frame rate), `avcorr` (mouth-motion x audio-energy correlation), `lightasd` (AVA weights), `lightasd-talkset`. Drops sub-0.5 s tracks and dead-end coasted frames; records RTF, threads, load. |
| `unspoken/transcribe/asd_live.py` | `LightASDLive`: an `ActiveSpeaker` provider for the existing live pipeline (sliding 1 s window, 5 Hz), a `config/models.toml` one-liner away. |
| `eval/transcribe/score_asd.py` | Scorer: per-seat frame acc / F1 / AP, mAP, which-seat accuracy over single-talker frames, talk F1, face coverage; `--table` builds the markdown table below. |
| `eval/transcribe/run_asd.sh` | Runs + scores every case for one method into `testdata/ami-runs/asd/<method>/<case>/`. |
| `tests/transcribe/test_asd.py` | 13 tests: tracker/coasting, crop, chunk plumbing with a fake scorer, model shapes (random weights), live session, scorer oracle/swap/abstain, CLI. No downloads, no real faces. |
| `requirements/transcribe-asd.txt` | torch 2.14.1, python_speech_features 0.6 (pinned). |
| `testdata/ami-runs/asd/<method>/<case>/{asd.json,scores.json,run.err}` and `testdata/ami-runs/asd/BEST` | Outputs and the recommendation (not in the repo). |

Weights live outside the repo: `dev-models/asd/lightasd_pretrain_AVA_CVPR.model` and `lightasd_finetuning_TalkSet.model`
(4.2 MB each, the authors' originals under `dev-models/asd/light-asd/weight/`), sha256 in `asd_lightasd.WEIGHTS`.

## 2. How to run

```bash
cd unspoken-wt/tx-asd
UV_CACHE_DIR=.../uv-cache scripts/dev-venv.sh
uv pip install --python .venv/bin/python -r requirements/sensors.txt -r requirements/transcribe-eval.txt
uv pip install --python .venv/bin/python "torch==2.14.1+cpu" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python python_speech_features==0.6
export UNSPOKEN_MODELS_DIR=$PWD/../../dev-models/asd          # where the weights are (default ~/.cache/unspoken/models)
.venv/bin/python -m unspoken.transcribe.asd_lightasd fetch --all   # once; sha256-checked, 2 x 4.2 MB from GitHub

# one case, one method -> asd.json (AsdResult)
OMP_NUM_THREADS=6 .venv/bin/python -m unspoken.transcribe.asd --in ../../testdata/ami/IS1009a-duo-AC-dev/video.mp4 \
    --out asd.json --method lightasd            # motion | avcorr | lightasd | lightasd-talkset ; --durations 1 for the fast variant
.venv/bin/python -m eval.transcribe.score_asd --ref ../../testdata/ami/IS1009a-duo-AC-dev --asd asd.json [--json]

# every case for a method (parallel, 6 threads each) -> testdata/ami-runs/asd/<method>/<case>/{asd.json,scores.json}
eval/transcribe/run_asd.sh lightasd -- IS1009a-duo-AC IS1009a-duo-DB IS1009a-L-room IS1009a-R-room
# the table below
.venv/bin/python -m eval.transcribe.score_asd --table ../../testdata/ami-runs/asd --baseline ../../testdata/ami-baseline --cases ../../testdata/ami
.venv/bin/python -m pytest -q                   # whole suite
```

## 3. Results

All metrics from `eval/transcribe/score_asd.py` against the AMI reference words of the person in each seat (seats
fixed by frame position, never learned from the reference). mAP = mean over the two on-camera seats of the AP of the
per-frame speaking probability; acc / F1 at p >= 0.5; which-seat acc = over frames where exactly one on-camera
person talks, how often the higher-scored seat is that person (no visible/scored face = wrong); talk F1 = "someone on
camera is talking" at p >= 0.5; face cov = share of frames with a visible face of that seat. `baseline-live-5hz` is
`testdata/ami-baseline/<case>/asd.json` (today's pipeline: YuNet on a stretched frame + MotionASD at 5 Hz) rescored
with the same scorer. Methods: `motion` = MotionASD at 25 fps on the new front end; `avcorr` = mouth-motion x audio-
energy correlation; `lightasd` = Light-ASD AVA weights, 6-duration ensemble; `lightasd-d1` = same, single 1 s
duration (5-6x cheaper); `lightasd-talkset-d1` = the TalkSet fine-tuned weights, single duration.
"set" = dev (180 s excerpt, used for every tuning decision) or FULL (~830 s, run once). RTF = wall / audio seconds
with the thread cap and the 1-min load average at start/end of the run (32-core box shared by six workstreams;
load > 32 means each process had less than its 6 threads' worth of CPU). The baseline rows have no timing: the
baseline JSON carries none (its pipeline ran at 5 Hz in real time).

| case | set | method | mAP | AP left | AP right | acc | F1 | which-seat acc | talk F1 | face cov L/R | faces | RTF | threads | load start/end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L-room | FULL | baseline-live-5hz | 0.389 | 0.648 | 0.131 | 0.720 | 0.160 | 0.713 | 0.198 | 0.926/0.794 | 151 | - | - | -/- |
| L-room | FULL | motion | 0.358 | 0.585 | 0.130 | 0.710 | 0.032 | 0.563 | 0.040 | 0.975/0.910 | 26 | 0.354 | 6 | 91.73/117.23 |
| L-room | FULL | avcorr | 0.324 | 0.551 | 0.097 | 0.709 | 0.000 | 0.375 | 0.000 | 0.975/0.910 | 26 | 0.276 | 6 | 29.34/63.04 |
| L-room | FULL | lightasd | 0.751 | 0.891 | 0.610 | 0.784 | 0.423 | 0.915 | 0.444 | 0.975/0.910 | 26 | 1.263 | 6 | 1.74/23.7 |
| L-room | FULL | lightasd-d1 | 0.661 | 0.844 | 0.479 | 0.744 | 0.228 | 0.858 | 0.241 | 0.975/0.910 | 26 | 1.484 | 6 | 100.23/50.08 |
| R-room | FULL | baseline-live-5hz | 0.171 | 0.192 | 0.150 | 0.817 | 0.130 | 0.464 | 0.253 | 0.703/0.844 | 156 | - | - | -/- |
| R-room | FULL | motion | 0.184 | 0.222 | 0.145 | 0.864 | 0.049 | 0.528 | 0.110 | 0.926/0.941 | 36 | 0.377 | 6 | 91.73/114.43 |
| R-room | FULL | avcorr | 0.150 | 0.194 | 0.107 | 0.869 | 0.000 | 0.389 | 0.007 | 0.926/0.941 | 36 | 0.288 | 6 | 29.34/62.31 |
| R-room | FULL | lightasd | 0.409 | 0.536 | 0.281 | 0.875 | 0.241 | 0.716 | 0.264 | 0.926/0.941 | 36 | 1.280 | 6 | 1.74/22.04 |
| R-room | FULL | lightasd-d1 | 0.359 | 0.453 | 0.265 | 0.873 | 0.174 | 0.650 | 0.190 | 0.926/0.941 | 36 | 1.514 | 6 | 100.23/46.93 |
| duo-AC | FULL | baseline-live-5hz | 0.437 | 0.705 | 0.169 | 0.712 | 0.016 | 0.441 | 0.017 | 0.986/0.959 | 99 | - | - | -/- |
| duo-AC | FULL | motion | 0.458 | 0.737 | 0.179 | 0.710 | 0.004 | 0.465 | 0.005 | 0.987/0.961 | 13 | 0.536 | 6 | 91.73/108.37 |
| duo-AC | FULL | avcorr | 0.422 | 0.717 | 0.127 | 0.710 | 0.000 | 0.336 | 0.000 | 0.987/0.961 | 13 | 0.411 | 6 | 29.34/54.72 |
| duo-AC | FULL | lightasd | 0.893 | 0.968 | 0.819 | 0.890 | 0.777 | 0.946 | 0.808 | 0.987/0.961 | 13 | 1.365 | 6 | 1.74/11.66 |
| duo-AC | FULL | lightasd-d1 | 0.860 | 0.965 | 0.755 | 0.851 | 0.667 | 0.910 | 0.696 | 0.987/0.961 | 13 | 1.510 | 6 | 100.23/48.93 |
| duo-DB | FULL | baseline-live-5hz | 0.305 | 0.390 | 0.220 | 0.859 | 0.041 | 0.466 | 0.197 | 0.932/0.960 | 123 | - | - | -/- |
| duo-DB | FULL | motion | 0.336 | 0.419 | 0.253 | 0.861 | 0.021 | 0.499 | 0.143 | 0.931/0.964 | 18 | 0.600 | 6 | 91.73/108.02 |
| duo-DB | FULL | avcorr | 0.380 | 0.422 | 0.338 | 0.870 | 0.002 | 0.411 | 0.002 | 0.931/0.964 | 18 | 0.410 | 6 | 29.34/54.72 |
| duo-DB | FULL | lightasd | 0.782 | 0.802 | 0.763 | 0.934 | 0.693 | 0.833 | 0.713 | 0.931/0.964 | 18 | 1.291 | 6 | 1.74/20.49 |
| duo-DB | FULL | lightasd-d1 | 0.733 | 0.763 | 0.703 | 0.916 | 0.547 | 0.783 | 0.558 | 0.931/0.964 | 18 | 1.517 | 6 | 100.23/46.93 |
| L-room | dev | baseline-live-5hz | 0.339 | 0.542 | 0.136 | 0.718 | 0.153 | 0.661 | 0.167 | 0.962/0.687 | 40 | - | - | -/- |
| L-room | dev | motion | 0.344 | 0.530 | 0.158 | 0.720 | 0.093 | 0.601 | 0.094 | 1.000/0.772 | 5 | 0.348 | 6 | 40.64/75.11 |
| L-room | dev | avcorr | 0.348 | 0.510 | 0.186 | 0.710 | 0.000 | 0.379 | 0.000 | 1.000/0.772 | 5 | 0.534 | 6 | 76.22/98.61 |
| L-room | dev | lightasd | 0.853 | 0.875 | 0.831 | 0.774 | 0.378 | 0.930 | 0.381 | 1.000/0.772 | 5 | 3.106 | 6 | 29.34/34.04 |
| L-room | dev | lightasd-d1 | 0.732 | 0.811 | 0.653 | 0.739 | 0.192 | 0.878 | 0.197 | 1.000/0.772 | 5 | 4.374 | 6 | 100.23/77.86 |
| R-room | dev | baseline-live-5hz | 0.232 | 0.128 | 0.336 | 0.906 | 0.325 | 0.629 | 0.357 | 0.732/0.894 | 30 | - | - | -/- |
| R-room | dev | motion | 0.137 | 0.105 | 0.170 | 0.924 | 0.000 | 0.570 | 0.006 | 1.000/0.955 | 7 | 0.304 | 6 | 40.64/73.73 |
| R-room | dev | avcorr | 0.145 | 0.094 | 0.196 | 0.926 | 0.012 | 0.390 | 0.012 | 1.000/0.955 | 7 | 0.418 | 6 | 76.22/101.26 |
| R-room | dev | lightasd | 0.553 | 0.531 | 0.576 | 0.936 | 0.285 | 0.853 | 0.285 | 1.000/0.955 | 7 | 3.188 | 6 | 29.34/30.05 |
| R-room | dev | lightasd-d1 | 0.381 | 0.342 | 0.420 | 0.930 | 0.149 | 0.721 | 0.149 | 1.000/0.955 | 7 | 4.409 | 6 | 100.23/76.11 |
| duo-AC | dev | baseline-live-5hz | 0.452 | 0.609 | 0.294 | 0.716 | 0.059 | 0.397 | 0.059 | 1.000/0.852 | 14 | - | - | -/- |
| duo-AC | dev | motion | 0.437 | 0.623 | 0.252 | 0.710 | 0.014 | 0.561 | 0.014 | 1.000/0.861 | 5 | 0.470 | 6 | 40.64/76.22 |
| duo-AC | dev | avcorr | 0.448 | 0.740 | 0.157 | 0.710 | 0.000 | 0.438 | 0.000 | 1.000/0.861 | 5 | 0.656 | 6 | 76.22/94.24 |
| duo-AC | dev | lightasd | 0.941 | 0.966 | 0.916 | 0.873 | 0.729 | 0.940 | 0.741 | 1.000/0.861 | 5 | 2.357 | 6 | 49.09/43.73 |
| duo-AC | dev | lightasd-d1 | 0.909 | 0.963 | 0.855 | 0.839 | 0.621 | 0.917 | 0.627 | 1.000/0.861 | 5 | 0.846 | 6 | 49.09/50.47 |
| duo-AC | dev | lightasd-talkset-d1 | 0.902 | 0.864 | 0.940 | 0.829 | 0.714 | 0.862 | 0.818 | 1.000/0.861 | 5 | 2.823 | 6 | 103.44/71.79 |
| duo-DB | dev | baseline-live-5hz | 0.400 | 0.484 | 0.317 | 0.926 | 0.000 | 0.519 | 0.000 | 1.000/0.999 | 5 | - | - | -/- |
| duo-DB | dev | motion | 0.383 | 0.384 | 0.381 | 0.925 | 0.000 | 0.582 | 0.000 | 1.000/1.000 | 3 | 0.458 | 6 | 40.64/75.19 |
| duo-DB | dev | avcorr | 0.463 | 0.266 | 0.660 | 0.925 | 0.000 | 0.519 | 0.000 | 1.000/1.000 | 3 | 0.618 | 6 | 76.22/95.3 |
| duo-DB | dev | lightasd | 0.910 | 0.950 | 0.870 | 0.975 | 0.828 | 0.948 | 0.861 | 1.000/1.000 | 3 | 3.332 | 6 | 29.34/24.09 |
| duo-DB | dev | lightasd-d1 | 0.849 | 0.902 | 0.796 | 0.963 | 0.695 | 0.915 | 0.727 | 1.000/1.000 | 3 | 4.903 | 6 | 100.23/60.48 |

Winner: `lightasd` (Light-ASD, AVA-ActiveSpeaker weights, 6-duration ensemble). On every case it is the only method
that is not at chance: FULL-case mAP 0.89 / 0.78 (duo-AC / duo-DB) and 0.75 / 0.41 (L-room / R-room) vs 0.44 / 0.31
and 0.39 / 0.17 for today's baseline; which-seat accuracy 0.95 / 0.83 / 0.92 / 0.72 vs 0.44 / 0.47 / 0.71 / 0.46;
F1 at p >= 0.5 0.78 / 0.69 / 0.42 / 0.24 vs 0.02 / 0.04 / 0.16 / 0.13. RTF 1.26-1.36 at 6 threads with the box
nearly idle (load 2-24); the single-duration variant is 0.85 at load 49. The two no-model methods
(`motion`, `avcorr`) are statistically at the speaking prior everywhere (AP within +-0.05 of the baseline, F1 ~0,
which-seat 0.34-0.60): mouth-pixel motion at 24-65 px face size does not separate talking from everything else
people do at a table. The authors' 6-duration ensemble (`lightasd`) beats the single 1 s duration (`lightasd-d1`) on every dev
case, and most where it matters: mAP +0.03 (duo-AC) / +0.06 (duo-DB) / +0.12 (L-room) / +0.17 (R-room), which-seat
+0.02 / +0.03 / +0.05 / +0.13, at 2.8-3.7x the compute (longer clips give the GRU more audio context, which is what
the far-field room cases lack). On the FULL cases the ensemble adds +0.03 / +0.05 / +0.09 / +0.05 mAP and +0.04 / +0.05 / +0.06 / +0.07
which-seat over `lightasd-d1`. The TalkSet weights are a wash on duo-AC-dev (mAP 0.902
vs 0.909, better F1 at 0.5 because they are more confident, worse which-seat 0.862 vs 0.917) and carry a licence
problem, so AVA it is.

Why the room cases are lower: the camera sees 24 px faces at 352x288, and in R-room the two people on camera
(D, B) speak 15 % of the time while the loudest voice in the far-field mic is A, who is off camera. Light-ASD
still ranks the right seat first in 65-72 % of single-talker frames (prior-matched chance ~50 %) and its AP is 5x
the speaking prior, but it cannot say "this voice belongs to nobody in view": that is for the fusion stage, using
diarization. L-room (A and C on camera, 58 % of the speech) is at 0.66 mAP / 0.86 which-seat with the same faces.

Threshold: the model is under-confident on this footage. Sweep on the FULL `lightasd-d1` outputs (the hint was
chosen on dev, this is the one full-case report): F1 / talk F1 at p >= 0.5 -> 0.2: duo-AC 0.667 / 0.696 ->
0.797 / 0.878; duo-DB 0.547 / 0.558 -> 0.657 / 0.786; L-room 0.228 / 0.241 -> 0.618 / 0.685; R-room 0.174 / 0.190
-> 0.366 / 0.456. The ensemble is more confident (its F1 at 0.5 is already 0.78 / 0.69 / 0.42 / 0.24) but the
same direction holds. Fusion should
consume the probabilities (argmax over visible faces when the audio says someone talks) or threshold at 0.2-0.3.

Face front end (dev excerpts, all cases): YuNet at native size finds both faces in 99-100 % of frames (room ~24 px,
duo ~64 px) at 4 ms (352x288) / 7 ms (704x288) per frame; upscaling 1.5-3x adds no detections and costs 2-9x the
time, so the default is 1.0x. Coverage in the table ("face cov") is lower than that for the right seat in some
cases because the person turns away from the camera for stretches (tracks end; no face = no score).

## 4. What runs on the GB10 and how

- Face detection + tracking: unchanged YuNet ONNX (OpenCV, CPU; 4 ms per 352x288 frame, 7 ms per 704x288 frame on
  one x86 core at 6 threads under load). Run it on the frame at its native size: upscaling did not add a single
  detection on the dev excerpts (both ~24 px room faces found in >99 % of frames at 1.0x) and costs 4-9x the time.
  The live config's 704x288 stretch of the room view is what starved the baseline of faces (35 % per-frame coverage
  at 5 Hz); `[video] width/height` should keep the camera's aspect ratio.
- Light-ASD: torch, `device = "cuda"` (torch 2.14.1+cu130 has a cp312 manylinux_2_28 aarch64 wheel on the PyTorch
  cu130 index; the CPU-only aarch64 wheel exists too). 1.03 M parameters, 4.2 MB weights; CPU here is ~55 ms per
  1 s clip per face at 6 threads under load, so the GPU is a convenience (a few ms per clip, lets the 6-duration
  ensemble run offline at well under real time), not a requirement. MFCC stays on CPU (python_speech_features).
- Offline (`transcript.json` path): `python -m unspoken.transcribe.asd --method lightasd --device cuda`.
- Live (room brain): the pipeline's `[video] fps` must go from 5 to 25 (the model is a 25 fps model; crops at 5 fps
  repeated 5x are a different input distribution). Detection can stay cheap: YuNet every frame is 4-7 ms.

```toml
# config/models.toml
[profiles.gb10.faces]                        # in-process YuNet face detector (OpenCV), native frame size, no recognition
provider = "python:unspoken.sensors.faces:YuNetFaces"
label = "YuNet 2023mar (OpenCV, CPU)"

[profiles.gb10.asd]                          # Light-ASD (AVA-ActiveSpeaker pretrained, MIT) on the GPU, sliding 1 s window
provider = "python:unspoken.transcribe.asd_live:LightASDLive"
label = "Light-ASD (AVA, CUDA)"
options = { device = "cuda", threads = 4, window_s = 1.0, every_s = 0.2, tail_s = 0.3, weights = "lightasd_pretrain_AVA_CVPR.model" }

# config/sensors.toml  -- the camera must deliver 25 fps frames to the ASD
[video]
fps = 25
width = 640        # keep the camera's aspect ratio (640x360 for 16:9); do not stretch
height = 360
[speakers]
hz = 5
speaking_threshold = 0.3      # Light-ASD is under-confident on far-field faces (see results); 0.3 on dev, re-check on the room camera
```

Weights on the box: `UNSPOKEN_MODELS_DIR=/data/unspoken/models python -m unspoken.transcribe.asd_lightasd fetch`
(or copy `dev-models/asd/lightasd_pretrain_AVA_CVPR.model` there; the file name is the registry key).

## 5. Not verified / contract requests / deps

Not verified here (no GPU, no GB10, no camera):
- The `device = "cuda"` path and the `torch==2.14.1+cu130` aarch64 wheel install. The model is plain torch
  (conv3d, conv2d, GRU, linear); nothing is x86-specific. Verify on the box with
  `UNSPOKEN_MODELS_DIR=... python -m unspoken.transcribe.asd --in CASE/video.mp4 --out o.json --method lightasd --device cuda`.
- `LightASDLive` inside the running pipeline with a real camera at 25 fps. It is unit-tested through the
  `ASDSession` protocol with a fake scorer, and instantiated + health-checked through `local_registry.build_provider`
  with the real weights on CPU (231 ms per scoring of 2 faces at 2 threads under load ~55). Not run end to end
  through `unspoken.sensors.pipeline` (the replay path fixes fps at 5 via the config; see risk 1).
- `lightasd-talkset` beyond one dev case (duo-AC-dev): not run; the box was shared by six workstreams (load
  90-117) for most of the time box. The AVA weights are the recommendation anyway (licence + which-seat).
- Timings: every RTF above was measured with 6 threads per process on the shared x86 box at the load shown.
  Under load ~100 each process got about one core, so the 6 torch threads over-subscribed it (dev RTF 4.4-4.9
  for a model that runs at RTF 0.85 at load 49). Treat the RTFs as upper bounds; on the GB10 GPU the model is
  not the bottleneck.

Contract change requests (`unspoken/transcribe/contracts.py`): none required. Two optional additions the fusion
stage might want (I did NOT add them):
- `FaceInfo.detected_frames` vs `frames`: today `frames` counts detected frames only; coasted frames (face
  briefly missed, box held) are scored and present in `frames[*].scores` but not counted. Harmless, but a
  consumer dividing scores by `frames` would be off by a few percent.
- `AsdResult.timing` is free-form; I write `wall_s, audio_s, rtf, threads, load1_start, load1_end, detect_s,
  model_s, frames, video_fps, every`. If the integrator wants `threads`/`load` standardised across stages, say so.

Deps added (`requirements/transcribe-asd.txt`): `torch==2.14.1` (CPU wheel here from the PyTorch CPU index;
`torch-2.14.1+cu130-cp312-cp312-manylinux_2_28_aarch64.whl` exists on the cu130 index for the GB10, and the
PyPI/CPU aarch64 wheels exist too), `python_speech_features==0.6` (pure python sdist, MIT). Everything else is
already in `requirements/sensors.txt` (numpy, opencv-python-headless, av). Model weights are not a pip dep:
`python -m unspoken.transcribe.asd_lightasd fetch` (sha256-pinned, MIT, 2 x 4.2 MB).

Licences: Light-ASD code and weights MIT (https://github.com/Junhua-Liao/Light-ASD/blob/main/LICENSE). The AVA
weights were trained on AVA-ActiveSpeaker (annotations CC BY 4.0 over film clips) -> fine to ship. The TalkSet
weights were fine-tuned on VoxCeleb2 + LRS3 derived data (research/non-commercial terms on the source data) ->
flagged; do not ship `lightasd_finetuning_TalkSet.model`. YuNet MIT. python_speech_features MIT. AMI test
data CC BY 4.0 (evaluation only, never shipped).

## 6. Open risks

1. Frame rate. The live pipeline keeps 5 fps (`[video] fps`); Light-ASD is a 25 fps model. `LightASDLive` repeats
   crops onto the 25 fps grid, so it runs at 5 fps, but the input is then far from the training distribution and
   the gain over MotionASD is unmeasured at 5 fps. Raise `fps` to 25 (YuNet is 4-7 ms per frame, fine) and keep
   the 5 Hz MT.SPEAKERS publish rate; the sensors workstream should confirm the newest-frame mailbox copes.
2. Threshold. The AVA-trained model is under-confident on these far-field, low-resolution faces: at p >= 0.5
   recall is 0.4-0.6 on the duo cases and 0.1-0.2 on the room cases while precision stays 0.8-1.0. AP (threshold
   free) is high, so fusion should use the probabilities (argmax over faces, or p >= 0.2-0.3; see the sweep above)
   rather than the pipeline's `speaking_threshold = 0.5`. Chosen on dev; re-check on the real room camera.
3. Room cases: the on-camera people in R-room (D, B) talk 15 % of the time while the loudest voice (A) is off
   camera; a vision-only model cannot know the audio belongs to nobody in view. mAP 0.36-0.38 there is 5x above
   the speaking prior but which-seat is only 0.65-0.72. Fusion with diarization (voice -> face binding over time)
   is what fixes this, not a better ASD.
4. Track fragmentation on the full cases (13-36 faces for 2 people): tracks break when a person turns away or
   leaves the chair; seat is per track so the fusion must key on seat, not track id. A re-identification-free fix
   (longer `--max-lost`, box-prior matching) is cheap if needed.
5. Compute on CPU: the 6-duration ensemble is ~2.4x real time at 6 threads (RTF 2.36 at load 49), the single
   duration 0.85. Offline on the GB10 GPU both are far under real time; the live path scores 1 s clips at 5 Hz.
6. The `faces` role still runs YuNet; `LightASDLive` only consumes its boxes. If the faces provider's `fps`/size
   config stretches the frame (the 704x288 stretch of the 352x288 room view cost the baseline 30 % face coverage),
   the ASD inherits the loss. Keep the camera aspect ratio.
