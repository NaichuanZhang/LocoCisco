# HACKDAY.md: LocoCisco on the Mac Studio (Apple M1 Max)

Everything except model serving was built and tested before today. On the Mac we: install LM Studio and load the
LLM, create the Python venv, start the decider, run one check, rehearse. Budget: **~45 minutes** from a clean Mac
to the first rehearsal, most of it one-time downloads (LM Studio pulls the Qwen GGUF; the sensing models download
into `~/.cache` on first use, or come from the staged kit).

No containers, Docker, or managed vLLM on the Mac: the LLM is served by **LM Studio** (OpenAI-compatible on
`http://localhost:1234/v1`), the decider by `strands-decider serve`, and the sensing models in process on the CPU.
The demo launcher is `deploy/demo/up.sh`; the staging kit is built by `deploy/usb/prepare.sh` (see
[DOWNLOAD.md](DOWNLOAD.md)). Every local service binds `127.0.0.1`. Logs: `~/unspoken-logs/`.

## 0. The night before, on your laptop (internet)

| Step | Command | Expect | Time |
|---|---|---|---|
| 0.1 Stage location | a plain folder `~/unspoken-usb` (or an exFAT USB labelled `UNSPOKEN`, works on macOS + Linux) | `~/unspoken-usb` (mac) or `/Volumes/UNSPOKEN/unspoken-usb` | 1 min |
| 0.1b What you get | every model, app and wheel with links and pins: [DOWNLOAD.md](DOWNLOAD.md) | | |
| 0.2 Dry run | `deploy/usb/prepare.sh ~/unspoken-usb --dry-run` | every item with its size from the HF API / PyPI, `TOTAL ~13 GiB` | 1 min |
| 0.3 Fill it | `deploy/usb/prepare.sh ~/unspoken-usb` | `finished: N done, 0 skipped, 0 failed`, `MANIFEST.sha256`, `README.txt` | 20–40 min (network bound) |
| 0.4 Resume | re-run the same command after any failure; finished items are skipped (`.done/`) | `done already: …` | — |
| 0.5 Verify | `cd ~/unspoken-usb && shasum -a 256 -c MANIFEST.sha256` | no output = good | 2 min |

No model here is gated: the LLM is a small open Qwen and the decider base is `Qwen/Qwen3.5-2B-Base`, so no
Hugging Face licence or `HF_TOKEN` is needed. If it fails: `pip download … no matching distribution` on a repo
requirement means a teammate pinned a package without a macOS arm64 wheel; note the name, carry on (`--only=`
reruns one item), and tell that workstream. The LM Studio model itself is easiest to pull from inside LM Studio
(1.1); the kit copy is the offline fallback.

## 1. On the Mac Studio

### 1.1 Install LM Studio + load the LLM (10 min, network for the model)
Install LM Studio (from the kit, or https://lmstudio.ai), open it once, then:
- In the model search, download **Qwen3-8B** (4-bit, e.g. a `Q4_K_M` GGUF). On 32 GB this is the comfortable
  ceiling; if memory is tight, load **Qwen3-4B** (~3 GB) instead.
- Developer tab → **Start Server** (port 1234). Enable tool / function calling.
```bash
curl -s http://localhost:1234/v1/models | grep -i qwen3     # the server lists the loaded model
```
Expect one model id (e.g. `qwen3-8b`); the `mac` profile's `[profiles.mac.llm]` already points at
`http://localhost:1234/v1`. If you loaded the 4B, `export UNSPOKEN_LLM_MODEL=qwen3-4b` before 1.4.

### 1.2 Python environment (5 min)
```bash
cd ~/lococisco                      # the repo checkout
scripts/dev-venv.sh                 # uv venv .venv (python 3.12) + the pinned wheels
# offline from the kit instead:  uv pip install --python .venv/bin/python --find-links ~/unspoken-usb/wheels -r requirements-dev.txt
```
Expect `venv ready: .venv`. If `uv` is missing: `brew install uv` (or copy it from the kit). This installs torch
for Apple Silicon (CPU / MPS), onnxruntime, onnx-asr and OpenCV — no CUDA wheels.

### 1.3 Start the decider (2 min)
```bash
strands-decider serve --model decider-2b --port 8765        # from the venv; first start loads the 2B
curl -s http://127.0.0.1:8765/health                        # expect {"status":"ok"}
```
The 2B is the first opinion on the Mac (there is no 12B at 32 GB). It runs on the CPU / Metal and the runtime
serializes its calls. It loads its weights from `~/.cache/huggingface`; unpack the staged kit there first for an
offline Mac (§3).

### 1.4 One check (2 min)
```bash
cd ~/lococisco
UNSPOKEN_PROFILE=mac .venv/bin/python -m unspoken.models check      # one tiny probe per enabled role, with latency
```
Expect every enabled role `ok`: `decider` (2B on :8765), `llm` (Qwen on LM Studio :1234), `asr` (Parakeet via
onnx-asr), `vad`, `diarization`, `faces`, `asd` in process, `pii` regex; `vlm`, `tts`, `decider_big` are `none`
on this profile. If `llm` fails: LM Studio's server is not running or the model name differs (`UNSPOKEN_LLM_MODEL`,
or edit `[profiles.mac.llm]`). If `decider` fails: 1.3 is not up. `.venv/bin/python -m unspoken.models show
--profile mac` prints the resolved table with no network.

### 1.5 Sensors: mic, camera, in-process models (5 min)
```bash
cd ~/lococisco && .venv/bin/python -m unspoken.sensors.check            # --no-models for devices only
```
This is the sensors-workstream sanity check; it uses that workstream's **defaults** (faster-whisper for ASR,
MotionASD), which are not the `mac` profile's demo choices — the profile's Parakeet (onnx-asr) and Light-ASD are
exercised by §1.4's `models check --profile mac`. Expect: `audio backend: sounddevice`, the speakerphone under
`audio inputs`, then `ok vad Silero VAD v5`, `ok diarization WeSpeaker CAM++`, `ok faces YuNet 2023mar`, and
`ok asd MotionASD`. Notes on macOS: `video devices` reads a Linux path and prints `none` on the Mac (the browser
demo processes a local video, so a webcam is optional); the `asr` row needs the optional faster-whisper weights
(DOWNLOAD.md) and reads `BAD` without them — harmless, since the demo transcribes with Parakeet. If audio is
missing → grant Terminal **Microphone** access in System Settings → Privacy & Security, then re-run; set
`[audio] device` (a substring of the speakerphone's name) in `config/sensors.toml`. `BAD diarization` → CAM++
missing (re-run the kit stage, or let it download into `~/.cache`).

### 1.6 Run the demo (5 min)
```bash
cd ~/lococisco
export UNSPOKEN_PROFILE=mac
deploy/demo/up.sh                 # reuses LM Studio on :1234 and the decider on :8765; starts the room + screen
```
Open the room screen (`http://127.0.0.1:8780/?present=1`) on the display, then run the 3-minute demo checklist
below twice. Startup reuses already-running local services; it does not install packages or download weights.

### 1.7 Tear down / restart
Stop the room + screen process (Ctrl-C in its terminal, or kill the `unspoken.meeting` / demo process); LM Studio
and the decider can be left running between takes. `deploy/demo/up.sh` is idempotent: re-running it leaves already-
running services alone and only restarts what is down.

## 2. Memory budget (32 GB unified, shared by CPU and GPU via Metal)

| Component | Approx |
|---|---:|
| macOS + apps | ~8 GB |
| LM Studio: Qwen3-8B 4-bit (or Qwen3-4B ~3 GB) | ~6 GB |
| Strands Decider 2B | ~5 GB |
| Sensing (ONNX / torch CPU: VAD, CAM++, YuNet, Parakeet, Light-ASD) | ~2–3 GB |
| App process + video buffers | ~2–3 GB |
| **Total** | **~23–25 GB** (~7–9 GB headroom) |

The 35B-class model is intentionally out of scope on 32 GB: the first-opinion decider is the 2B, and the LLM
escalation is a small Qwen on LM Studio. Keep the LLM context modest. The M1 Max shares ~400 GB/s of memory
bandwidth across CPU and GPU; the runtime serializes decider calls, so a decider answer and a streaming Qwen reply
do not compete for long. TTS and the whiteboard VLM are disabled on this target (`provider = "none"`).

## 3. Staging kit

Sizes from `deploy/usb/prepare.sh --dry-run` (HF API, PyPI); approximate for the GGUF and ONNX artifacts. A plain
folder or any ~32 GB drive holds it with room to spare.

| Item | Group | Download | What |
|---|---|---:|---|
| LM Studio (macOS, Apple Silicon) | core | ~0.5 GiB | the LLM server (easiest pulled from inside LM Studio) |
| Qwen3-8B 4-bit GGUF (for LM Studio) | core | ~4.9 GiB | main LLM (or Qwen3-4B ~3 GiB) |
| `Qwen/Qwen3.5-2B-Base` @b1485b2f + `strands-decider-2B-qwen3.5-v1-2610` @216d1ccb | core | ~4.5 GiB | decider 2B (base + adapter) |
| wheels: decider (strands-decider @930e6d5 + deps) + unspoken (every `requirements/*.txt`) | core | ~0.5 GiB | macOS arm64 cp312, incl. torch for Apple Silicon, onnxruntime, onnx-asr, OpenCV |
| tools: uv 0.13.5, repo git bundle | core | ~25 MiB | |
| `istupakov/parakeet-tdt-0.6b-v3-onnx` (fp32 ONNX export, CC-BY-4.0) | sensors | ~2.4 GiB | ASR via onnx-asr (CPU), into `~/.cache` |
| WeSpeaker CAM++ ONNX (sha256-pinned, → `~/.cache/unspoken/models`) | sensors | 28 MiB | diarization voices |
| Light-ASD AVA weights (MIT, sha256-pinned, → `~/.cache/unspoken/models`) + `python_speech_features` wheel | sensors | 4 MiB | who is talking on camera |
| **Total** | | **~13 GiB** | required only |

Silero VAD v5 and the YuNet face detector are vendored in the repo (`unspoken/sensors/models/`): nothing to
download. HF repos travel as tar files of their cache directories, unpacked into `~/.cache/huggingface/hub`;
onnx-asr and the sensors loader look there first, so a staged kit means the day is offline-safe.

## 4. Models: `config/models.toml`, profile `mac`

The `mac` profile in `config/models.toml` is the single source of truth. Print it and check it:
```bash
.venv/bin/python -m unspoken.models show  --profile mac      # role, provider, URL, model, label (no network)
.venv/bin/python -m unspoken.models check --profile mac      # one tiny probe per role; exit 1 if any enabled role fails
```

| Role | Provider | Where |
|---|---|---|
| decider | `strands-decider` | :8765, 2B on the CPU / Metal (`strands-decider serve`) |
| decider_big | `none` | no second decider at 32 GB |
| llm | `openai` | `http://localhost:1234/v1`, Qwen3-8B 4-bit on LM Studio (Chat Completions, tools) |
| vlm | `none` | whiteboard VLM disabled on this target |
| asr | `python:unspoken.transcribe.asr_parakeet:ParakeetASR` | in process, Parakeet TDT 0.6B v3 via onnx-asr (CPU) |
| tts | `none` | disabled on this target |
| vad, diarization, faces, asd | `python:unspoken.sensors.*` / `unspoken.transcribe.*` | in process: Silero, CAM++, YuNet, Light-ASD |
| embed | `fake` | swap to an embedding server if time allows |
| pii | `regex` | in process |

Switch a model with one edit in that file, or one variable without editing it: `UNSPOKEN_LLM_MODEL=qwen3-4b`
(the smaller LLM), `UNSPOKEN_LLM_URL=...` (a different LM Studio / OpenAI endpoint). `local_only = true` refuses
any model that is not on this box or its LAN. Details: [MODELS.md](MODELS.md). Health probes: decider
`GET /health`, LM Studio `GET /v1/models`.

## 5. Pinned versions (why)

| Thing | Pin | Why |
|---|---|---|
| LLM server | LM Studio (OpenAI-compatible server, port 1234) | local, Metal-backed, no containers; serves any GGUF/MLX Qwen |
| LLM model | Qwen3-8B, 4-bit GGUF (`Q4_K_M`); fallback Qwen3-4B | fits the 32 GB budget with the decider and sensing resident |
| Decider code | strands-decider `930e6d5` (v0.1.0-8) | the commit serving the 2B on the dev desktop; Gemma 4 support is on main, not on PyPI |
| Decider deps | `deploy/usb/requirements-decider.txt` (transformers 5.20.2, peft 0.21.2, …) | `uv pip freeze` of that working venv |
| Decider base | `Qwen/Qwen3.5-2B-Base` @b1485b2f + `strands-decider-2B-qwen3.5-v1-2610` @216d1ccb | the 2B is the only decider at 32 GB |
| ASR model | `istupakov/parakeet-tdt-0.6b-v3-onnx` (fp32), via onnx-asr | the public ONNX export of `nvidia/parakeet-tdt-0.6b-v3` (6.34% WER); runs on the CPU with no torch — the 282x real-time figure was measured on the original GB10 GPU path, not the Mac |
| uv | 0.13.5 | reproducible venvs |
| Sensors models | CAM++ sha256 `c46fad10…` (= `unspoken/sensors/assets.py`) | what `unspoken.sensors` loads; `tests/deploy` fails if the kit pins drift from the code |

## 6. The 3-minute demo checklist

Before (T-10 min): `UNSPOKEN_PROFILE=mac .venv/bin/python -m unspoken.models check` all `ok`; room screen on the
display; LM Studio server running; the decider up; phone timer.

1. (0:00) "LocoCisco hears the room and writes named captions": two people talk; captions appear with names/seats on the screen. Say the latency out loud (ASR and decider).
2. (0:40) "Sam, can you send the deck by Friday?" "Sure." → a task card appears as a **local output preview** (decider said `commitment`, policy allowed); the egress meter shows that nothing left the box. Live Discord delivery is rehearsed locally (see §note).
3. (1:20) "It's warm in here" → fan on, lamp to `talk` scene (device actions are local previews). "Slides please" → lamp dims to `slides`.
4. (1:50) "Off the record." → captions stop, the screen shows the privacy state, nothing retained. "Back on the record." → resumes.
5. (2:20) Ask a question of the room notes → the runtime calls Qwen on LM Studio and answers from the local notes.
6. (2:45) Point at the X-ray panel: every decision shows which local model made it and in how many ms; open Activity Monitor in a corner: the Mac doing all of it, nothing left the room.

Recovery during the demo: if captions stall → re-run `unspoken.models check`; LLM down → LM Studio **Start Server**
again (meanwhile decider-only mode still does captions, tasks, devices); decider down → `strands-decider serve
--model decider-2b --port 8765` (meanwhile the runtime falls through to the LLM).

Note: the Discord bridge (OpenClaw / OpenShell via NemoClaw) is legacy NVIDIA-only tooling and does not run on the
Mac; see [NEMOCLAW.md](NEMOCLAW.md). The Mac demo rehearses notifications and device actions as local output
previews.

## Sources
- LM Studio local server (OpenAI-compatible API): https://lmstudio.ai/docs/app/api/endpoints/openai
- Qwen3 model cards: https://huggingface.co/Qwen
- Parakeet ONNX export used by onnx-asr: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx (checkpoint: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- onnx-asr: https://github.com/istupakov/onnx-asr
- Strands Decider serving and memory: https://github.com/strands-labs/strands-decider/blob/main/docs/inference.md
- Light-ASD (active speaker detection, MIT): https://github.com/Junhua-Liao/Light-ASD
