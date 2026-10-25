# HACKDAY.md: LocoCisco on the Dell Pro Max GB10, Sun 2026-10-25

Everything except model deployment was built and tested before today. On the box we: restore the USB, let
NemoClaw bring up vLLM, start the decider and the speech server, run one check, rehearse. Budget: **90 minutes**
from unboxing to the first rehearsal, of which ~25 minutes is the NemoClaw installer on venue Wi-Fi.

Scripts live in `deploy/gb10/` and `deploy/usb/`; every one takes `--dry-run` (prints each command, runs nothing)
and `-h`. Every server binds `127.0.0.1`. Logs: `~/unspoken-logs/`, `docker logs unspoken-*`.

## 0. The night before, on your laptop (internet)

| Step | Command | Expect | Time |
|---|---|---|---|
| 0.1 Format the USB | exFAT (works on macOS + Linux; nothing on the drive needs symlinks), label `UNSPOKEN` | `/Volumes/UNSPOKEN` (mac) or `/media/$USER/UNSPOKEN` | 1 min |
| 0.2 Gemma licence | accept https://huggingface.co/google/gemma-4-12B-it on huggingface.co, `export HF_TOKEN=hf_…` (never into a file in the repo) | token in your shell only | 2 min |
| 0.2b What you get | every model, image and installer with links and pins: [DOWNLOAD.md](DOWNLOAD.md) | | |
| 0.3 Dry run | `deploy/usb/prepare.sh /Volumes/UNSPOKEN/unspoken-usb --dry-run` | every item with its size from the HF API / registry / PyPI / Ubuntu index, `TOTAL 77.6 GiB`, `drive: 128 GB` | 1 min |
| 0.4 Fill it | `deploy/usb/prepare.sh /Volumes/UNSPOKEN/unspoken-usb` (`--skip-optional` drops the optional fallback image and TTS: 14 GiB less) | `finished: 28 done, 0 skipped, 0 failed`, `MANIFEST.sha256`, `README.txt` | 1–3 h (network bound: ~78 GB download) |
| 0.5 Resume | re-run the same command after any failure; finished items are skipped (`.done/`) | `done already: …` | — |
| 0.6 Verify | `cd /Volumes/UNSPOKEN/unspoken-usb && shasum -a 256 -c MANIFEST.sha256` | no output = good | 10 min |

If it fails: `hf download … 401` means HF_TOKEN missing or Gemma not accepted (0.2). `docker pull … no matching
manifest` means Docker Desktop is not set to pull foreign platforms: the images are linux/arm64 and the flag is
already passed; update Docker Desktop. `pip download … no matching distribution` on a repo requirement means a
teammate pinned a package without an aarch64 wheel: note the name, carry on (`--only=` reruns one item), and tell
that workstream. Prefer `--no-zstd` only when `zstd` is missing on the laptop (`brew install zstd`): plain
`docker save` tars are ~2.2x the compressed size (see the USB table).

## 1. On the GB10

### 1.1 Unbox and log in (10 min)
- Plug: power, HDMI (room screen), USB speakerphone, USB webcam, the USB drive, the travel router (WiZ lamp + Kasa fan live on it).
- DGX OS first-boot wizard: user `unspoken`, **no** automatic updates, join venue Wi-Fi.
- Terminal: `id -nG | grep docker || sudo usermod -aG docker $USER && newgrp docker`.

### 1.2 Preflight (2 min)
```bash
USB=/media/$USER/UNSPOKEN/unspoken-usb
bash $USB/bin/gb10/00-preflight.sh          # read-only; exit code = number of FAIL lines
```
Expect: `OK arch aarch64`, `OK nvidia-smi present` with `NVIDIA GB10`, `CUDA Version: 13.x`, `OK RAM 1xx GB`,
`OK home has >150 GB free`, `OK docker daemon reachable`, `OK nvidia container runtime registered`, every port
`free`, `OK USB at …`, `OK every manifest item is present`, `== result: 0 FAIL`.
If it fails: no nvidia runtime → `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
Ports 8000/18789 in use → NemoClaw from a previous owner is already installed; fine, skip its install later.
`--verify` adds the full sha256 check (10 min) if the drive was dropped.

### 1.3 Restore the USB (15 min, no network)
```bash
bash $USB/bin/restore.sh $USB            # add --native-decider to also build the no-docker decider venv
```
Expect, in order: `docker load` of `nvcr.io/nvidia/vllm:26.05.post1-py3` and `nemo-speech.cpp:0.2.0` (3–6 min),
`restored nvidia/Qwen3.6-35B-A3B-NVFP4` + the two deciders + two base models into `~/.cache/huggingface/hub`
(5 min) plus `Systran/faster-whisper-base.en` and `small.en` (each with `refs/main` pointing at the pinned snapshot, so
the offline loaders find them), speech GGUFs + `Magpie tokenizer` + wheelhouses into `~/unspoken-models`,
`OK sensors model wespeaker_en_voxceleb_CAM++.onnx (sha256 ok)` in `~/.cache/unspoken/models`, a sudo prompt for
`== 3b. apt packages for the microphone` (installs `libportaudio2` + `alsa-utils` from the USB .debs, only the packages
DGX OS lacks, never downgrading; `already installed` if present), `uv` +
NeMo-Speech native build into `~/unspoken-models/tools`, NemoClaw source into `~/nemoclaw-src`, the repo cloned to
`~/unspoken` (branch `hack/main`) with an offline venv, `OK restore done in …s`.
If it fails: `docker load: no space` → `df -h ~`; the box needs ~120 GB free. apt step fails → it retries from
the network itself; if both fail the sensors use `arecord` if present, else no mic (`--skip-apt` skips the step).
CAM++ `sha256 mismatch` → `~/unspoken/.venv/bin/python -m unspoken.sensors.fetch_models` (29 MB, Wi-Fi). `tar: … hf/` corrupt → re-copy that
one tar from the laptop. `uv venv … python 3.12 not found` → `sudo apt install python3.12-venv` is NOT needed with
uv; check `python3 --version` (DGX OS 7 ships 3.12); otherwise `UV_PYTHON_PREFERENCE=only-system uv venv --python python3 ~/unspoken/.venv`.

### 1.4 NemoClaw + OpenClaw + managed vLLM (25 min, venue Wi-Fi)
Order matters: the sandbox image bakes in our private CA at onboarding, so make the CA and export every setting
BEFORE the installer runs (the express install onboards right away; `--defer-onboarding` does not apply to OpenClaw).
The CA does not depend on the bridge IP; `connect.sh` (1.6) re-issues only the server cert for the real
`openshell-docker` gateway and keeps the CA.
```bash
export DISCORD_BOT_TOKEN=…  DISCORD_SERVER_ID=…  DISCORD_REQUIRE_MENTION=1     # from the team password manager
export UNSPOKEN_MCP_TOKEN="$(openssl rand -hex 32)"     # one bearer: MCP server, TLS proxy, sandbox (keep this shell open)
~/unspoken/deploy/nemoclaw/make-ca.sh                   # private CA in ~/.unspoken/tls (0700, outside the repo)
export NEMOCLAW_CORPORATE_CA_BUNDLE=~/.unspoken/tls/ca.pem
export NEMOCLAW_AGENT=openclaw NEMOCLAW_SANDBOX_NAME=unspoken NEMOCLAW_POLICY_TIER=restricted
cd ~/nemoclaw-src && bash install.sh          # == curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash, but from the USB copy of the lkg tag
nemoclaw unspoken status                      # sandbox "unspoken" Ready; if it is missing or was built without the CA:
# cd ~/unspoken && deploy/nemoclaw/onboard.sh  # same env; idempotent (see docs/NEMOCLAW.md §4)
```
Press Enter at `Run express install with these settings? [Y/n]:` (DGX preset: OpenClaw + local vLLM). Express
picks the Spark profile `nvidia/Qwen3.6-35B-A3B-NVFP4`; the weights are already in `~/.cache/huggingface`, so it
skips the 23 GB download; the vLLM image is already loaded. Expect `Application startup complete` in
`docker logs -f nemoclaw-vllm`, then `curl -s 127.0.0.1:8000/v1/models | grep Qwen3.6`.
Still on Wi-Fi: NemoClaw's CLI (Node via nvm), OpenShell and the sandbox image (~1 GB, not on the USB: the
installer resolves them at run time). If Wi-Fi is hopeless: `deploy/gb10/vllm.sh` starts the same vLLM from the USB
image without NemoClaw, and the bridge workstream's fallback (`discord-rest` Notifier) replaces OpenClaw.
If the managed vLLM dies: `docker logs nemoclaw-vllm | tail -50`. `EngineDeadError` after a few requests = memory
pressure → stop the decider 2B / TTS first (`deploy/gb10/speech.sh tts --stop`). NemoClaw's doc says the Spark profile
uses `--gpu-memory-utilization 0.4`, 4 seqs, 262K context.

### 1.5 Our servers (5 min)
```bash
cd ~/unspoken
deploy/gb10/up.sh                 # vLLM reused if :8000 answers; decider 12B; ASR; TTS; then check.sh
deploy/gb10/status.sh             # one line per server with probe latency
```
Expect: `vLLM already answering on :8000`, decider: `torch 2.x cuda=True NVIDIA GB10` then `OK decider 12b up after
60–180s` (first start also pip-installs the wheelhouse, +60 s), `OK speech asr up after <30s`, `OK speech tts up`,
check: four `OK` probes, one decider answer like `{"has_task":{"noul":0.86}}`, `unspoken.models check` all `ok`.
If it fails:
- decider `cuda=False` → the container did not get the GPU: `docker run --rm --gpus all nvcr.io/nvidia/vllm:26.05.post1-py3 nvidia-smi`; if that fails, redo 1.2's nvidia-ctk step.
- decider import error on `transformers`/`peft` → the wheelhouse install failed; `docker logs unspoken-decider-12b`; retry with the other torso: `DECIDER_IMAGE=nvcr.io/nvidia/pytorch:26.09-py3 deploy/gb10/decider.sh 12b` (image on the USB unless `--skip-optional`).
- decider CUDA kernel error (sm_121) → no-docker path: `bash $USB/bin/restore.sh $USB --native-decider` then `DECIDER_RUNTIME=native deploy/gb10/decider.sh 12b` (torch 2.14.1+cu130 aarch64 wheels from the USB).
- decider OOM → `deploy/gb10/decider.sh 12b --stop && deploy/gb10/decider.sh 2b` and point `[profiles.gb10.decider]` at :8766 (or `UNSPOKEN_DECIDER_URL=http://127.0.0.1:8766`).
- speech `/ready` 503 → `docker logs unspoken-asr`; a GGUF path error means restore.sh did not copy `speech/`; a CUDA error → `SPEECH_RUNTIME=native deploy/gb10/speech.sh asr` (the 70 MB cuda13 build from the USB), last resort CPU: in-process faster-whisper (`[profiles.gb10.asr] provider = "python:unspoken.sensors.asr_local:FasterWhisperASR"`).
- TTS fails → it is optional: `deploy/gb10/up.sh --no-tts`, set `[profiles.gb10.tts] provider = "none"`.

### 1.5b Sensors: mic, camera, in-process models (5 min)
```bash
cd ~/unspoken && HF_HUB_OFFLINE=1 .venv/bin/python -m unspoken.sensors.check
```
Expect: `audio backend: sounddevice`, the speakerphone under `audio inputs`, `/dev/video0` under `video devices`,
then five rows: `ok vad Silero VAD v5`, `ok asr faster-whisper base.en (CPU int8)`, `ok diarization WeSpeaker CAM++`,
`ok faces YuNet 2023mar`, `ok asd MotionASD`. (`deploy/gb10/check.sh` runs this too.)
If it fails: no audio inputs → `arecord -l`; `libportaudio2` missing → `sudo apt-get install -y libportaudio2`. Set
`[audio] device` (substring of the speakerphone's name) and `[video] device` in `config/sensors.toml`. `BAD asr` with
`LocalEntryNotFoundError` → the faster-whisper cache has no `refs/main`: re-run `restore.sh` (it backfills it).
`BAD diarization` → CAM++ missing (see 1.3). Seat zones: record 20 s of the chairs and adjust `[seats]` (sensors docs).

### 1.6 Bridge, screen, rehearsal (15 min)
Same shell as 1.4 (needs `UNSPOKEN_MCP_TOKEN` and the Discord env). Fill the three channel ids in
`config/bridge.toml [notifier.channels]` first.
```bash
cd ~/unspoken
UNSPOKEN_PROFILE=gb10 UNSPOKEN_NOTIFIER=openclaw .venv/bin/python -m unspoken.meeting --mode live &   # room brain; serves MCP on 127.0.0.1:8791 and the screen on :8780
deploy/nemoclaw/connect.sh --firewall           # TLS proxy on the openshell-docker gateway :8443, mcp add, policy, persona
deploy/nemoclaw/verify.sh --post --agent        # 401 without the token, tools through the proxy, a test post, the agent calls room_status
```
If `nemoclaw` is broken on the day: `UNSPOKEN_NOTIFIER=discord-rest` posts with the bot token from the box
(`connect.sh --fallback-rest`; docs/NEMOCLAW.md §6). Open the room screen (`http://127.0.0.1:8780/?present=1`) on the
HDMI display, then run the 3-minute demo checklist below twice.

### 1.7 Tear down / restart
```bash
deploy/gb10/down.sh            # ours only; --all also stops nemoclaw-vllm (restart that one with nemoclaw, not vllm.sh)
deploy/gb10/up.sh              # idempotent: running servers are left alone
```

## 2. Memory budget (128 GB unified; ~120 GiB visible to CUDA, shared by CPU and GPU)

| What | Resident | Note |
|---|---:|---|
| vLLM Qwen3.6-35B-A3B NVFP4 (`--gpu-memory-utilization 0.40`) | ~48 GiB cap | 23.5 GB weights + KV cache (fp8) + activations; NemoClaw's own Spark profile value |
| Strands Decider 12B (gemma-4-12B-it, bf16, text decoder only) | ~27 GB | 23 GB weights (strands-decider docs) + activations at the 4K window, `--max-batch 8` |
| NeMo-Speech.cpp ASR Parakeet TDT v3 Q8 + Nemotron 3 Diarization Q8 | ~3 GB | 0.7 + 0.1 GB of weights, ggml buffers |
| MagpieTTS 357M Q8 + NanoCodec (optional) | ~1 GB | |
| Decider 2B (optional) | ~5 GB | 4.5 GB bf16 |
| DGX OS, Docker, NemoClaw/OpenShell sandbox, OpenClaw (Node), LocoCisco host process, camera/mic buffers | ~12 GB | |
| **Total with everything** | **~96 GB** | **~24 GB headroom**; without TTS and 2B: ~90 GB |

Rules: vLLM first, then the decider (up.sh does this). Two models decoding at once share 273 GB/s: expect decider
answers ~2x slower while Qwen streams (forum 386489); the runtime serializes decider calls anyway. Never run image
or video generation on this box during the demo (power-off reports, forum 385991); if the box reboots under load,
`00-preflight.sh --cap-clocks` and a desk fan.

## 3. USB drive

Sizes from `deploy/usb/prepare.sh --dry-run` on 2026-10-24 (HF API, nvcr.io manifests, PyPI). Buy a **256 GB**
USB 3.x drive (usable ~238 GB); 128 GB fits only `--skip-optional` with zstd (72 GiB).

| Item | Group | Download | On USB (plain tar) | What |
|---|---|---:|---:|---|
| `nvcr.io/nvidia/vllm:26.05.post1-py3` (arm64 `sha256:9204569b…`) | core | 8.9 GiB | 19.7 GiB (8.9 with zstd) | NemoClaw's vLLM image; decider torso |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | core | 21.9 GiB | 21.9 GiB | main LLM |
| `google/gemma-4-12B-it` @707f0a3b (gated) | core | 22.3 GiB | 22.3 GiB | base of the 12B decider |
| `StrandsAgents/strands-decider-12B-gemma4-v1-2610` | core | 0.8 GiB | 0.8 GiB | decider adapter |
| `Qwen/Qwen3.5-2B-Base` @b1485b2f + `strands-decider-2B-qwen3.5-v1-2610` @216d1ccb | core | 4.5 GiB | 4.5 GiB | decider 2B |
| wheels: decider (51 + strands-decider, no torch), unspoken (98: every `requirements/*.txt` incl. the transcriber) | core | 280 MiB | 280 MiB | aarch64 cp312 (all resolve, checked 2026-10-25) |
| tools: uv 0.13.5, NemoClaw bootstrap + shallow source @lkg, repo bundle | core | ~0.2 GiB | ~0.2 GiB | |
| `nvcr.io/nvidia/nemo-speech.cpp:0.2.0` (arm64) + Parakeet v3 Q8 + Nemotron diar Q8 | speech | 1.0 GiB | 1.2 GiB | ASR |
| Nemotron 3.5 ASR streaming Q8, Magpie TTS Q8 + .nemo + NanoCodec | speech (optional) | 2.4 GiB | 2.4 GiB | alt ASR, TTS |
| WeSpeaker CAM++ ONNX (sha256-pinned, → `~/.cache/unspoken/models`) | sensors | 28 MiB | 28 MiB | diarization voices |
| `Systran/faster-whisper-base.en` @3d3d5dee + `small.en` @d1d751a5 | sensors | 605 MiB | 605 MiB | in-process CPU ASR (fallback) |
| Ubuntu noble arm64 .debs: libportaudio2, alsa-utils + 8 deps | sensors | 4 MiB | 4 MiB | mic access (Silero + YuNet are vendored in the repo) |
| Light-ASD AVA weights (MIT, sha256-pinned, → `~/.cache/unspoken/models`) + `python_speech_features` wheel | sensors | 4 MiB | 4 MiB | who is talking on camera |
| torch 2.14.1+cu130 aarch64 + CUDA libs (29 wheels) → repo venv | sensors | 2.9 GiB | 2.9 GiB | Light-ASD on the GPU; also the no-docker decider |
| `nvcr.io/nvidia/pytorch:26.09-py3` (arm64) | fallback (optional) | 11.4 GiB | 25.1 GiB (11.4 zstd) | second decider torso |
| NeMo-Speech native cuda13 | fallback (optional) | 67 MiB | 67 MiB | speech without docker |
| `hexgrad/Kokoro-82M` | extras (optional) | 0.3 GiB | 0.3 GiB | alt TTS, not scripted |
| **Total** | | **77.6 GiB** | **102 GiB plain / 78 GiB zstd** | required only: 63.3 GiB |

Layout on the drive: `images/ hf/ speech/ sensors/ apt/ wheels/ tools/ repo/ bin/ MANIFEST.sha256 README.txt .done/`.
HF repos travel as tar files of their cache directories (`models--org--repo`), unpacked into
`~/.cache/huggingface/hub` on the box, which is exactly where NemoClaw's downloader looks, so it finds Qwen3.6
and downloads nothing.

## 4. Models: `config/models.toml`, profile `gb10`

The `gb10` profile in `config/models.toml` is the single source of truth (it was reconciled with these scripts).
Print it and check it:
```bash
.venv/bin/python -m unspoken.models show  --profile gb10      # role, provider, URL, model, label (no network)
.venv/bin/python -m unspoken.models check --profile gb10      # one tiny probe per role; exit 1 if any enabled role fails
```

| Role | Provider | Where |
|---|---|---|
| decider | `strands-decider` | :8765, 12B on the GPU |
| decider_big | `none` | uncomment the 2B :8766 block to add it |
| llm, vlm | `openai` | :8000/v1, `nvidia/Qwen3.6-35B-A3B-NVFP4` (Chat Completions, tools) |
| asr | `openai-audio` | :8770/v1, Parakeet TDT 0.6B v3 (NeMo-Speech.cpp), `verbose_json` |
| tts | `openai-speech` | :8771/v1, MagpieTTS 357M (optional) |
| vad, diarization, faces, asd | `python:unspoken.sensors.*` | in-process: Silero, CAM++, YuNet, MotionASD |
| embed | `fake` | swap to an embedding server if time allows |
| pii | `regex` | in-process |

Switch a model with one edit in that file, or one variable without editing it:
`UNSPOKEN_DECIDER_URL=http://127.0.0.1:8766` (2B instead of the 12B), `UNSPOKEN_LLM_MODEL=...`,
`UNSPOKEN_ASR_PROVIDER=python:unspoken.sensors.asr_local:FasterWhisperASR` (CPU ASR, no speech server),
`UNSPOKEN_TTS_PROVIDER=none`. `local_only = true` refuses any model that is not on this box. Details: docs/MODELS.md.
Health probes: decider `GET /health`, vLLM `GET /v1/models`, speech `GET /ready` (200 loaded, 503 loading).

## 5. Pinned versions (why)

| Thing | Pin | Why |
|---|---|---|
| vLLM image | `nvcr.io/nvidia/vllm:26.05.post1-py3`, arm64 `sha256:9204569b17ee4c0eff75194b8e6e458479c8aee18953b5ab9cf359fcdac659e2` | the image NemoClaw's Spark profile documents; CUDA 13.2, Python 3.12, torch + triton built for `12.0+PTX` (runs on GB10 sm_121) |
| Decider code | strands-decider `930e6d5` (v0.1.0-8) | the commit serving the 2B and 12B on the dev desktop today; Gemma 4 support is on main, not on PyPI |
| Decider deps | `deploy/usb/requirements-decider.txt` (transformers 5.20.2, peft 0.21.2, …) | `uv pip freeze` of that working venv |
| Speech runtime | NeMo-Speech.cpp 0.2.0 (image arm64 `sha256:3ad7c3f4…`, native `linux-aarch64-cuda13`) | NVIDIA's own C++ speech runtime with a DGX Spark build; OpenAI-compatible routes |
| ASR model | `nvidia/parakeet-tdt-0.6b-v3` Q8 GGUF | 282x real time measured on GB10 (NVIDIA forum 356353); 6.34% WER |
| Fallback torso | `nvcr.io/nvidia/pytorch:26.09-py3` arm64 `sha256:307c449d…` | current NGC PyTorch (2.14.0a0, CUDA 13.4) |
| Fallback torch | `torch==2.14.1+cu130` aarch64 from download.pytorch.org | matches the dev venv's 2.14.1; cu130 wheels run on Spark per community reports |
| NemoClaw | `lkg` tag (8e63c869 on 2026-10-24) | NVIDIA's last-known-good; the installer defaults to it |
| uv | 0.13.5 | offline venvs on the box |
| Sensors models | CAM++ sha256 `c46fad10…` (= `unspoken/sensors/assets.py`), faster-whisper base.en @3d3d5dee / small.en @d1d751a5 | what `unspoken.sensors` loads; `tests/deploy` fails if the USB pins drift from the code |
| apt | noble / noble-updates / noble-security arm64, resolved at prepare time, sha256 per .deb from the index | DGX OS 7 is Ubuntu 24.04; restore installs only missing packages |

## 6. The 3-minute demo checklist

Before (T-10 min): `deploy/gb10/status.sh` all `ok`; room screen on HDMI; lamp + fan reachable (devices workstream
`python -m unspoken.devices ping`); Discord channel open on the laptop; phone timer.

1. (0:00) "LocoCisco hears the room and writes named captions": two people talk; captions appear with names/seats on the screen. Say the latency out loud (ASR ~0.3 s, decider ~0.2 s).
2. (0:40) "Sam, can you send the deck by Friday?" "Sure." → a task card lands in Discord within 3 s (decider said `commitment`, policy allowed, Notifier posted; the egress meter shows the bytes that left the box: text only).
3. (1:20) "It's warm in here" → fan on, lamp to `talk` scene. "Slides please" → lamp dims to `slides`.
4. (1:50) "Off the record." → captions stop, screen shows the privacy state, nothing posted. "Back on the record." → resumes.
5. (2:20) In Discord: `@LocoCisco what did we decide?` → OpenClaw calls the MCP tool, Qwen3.6 answers from the local notes.
6. (2:45) Point at the X-ray panel: every decision shows which local model made it and in how many ms; `nvidia-smi` in a corner: the GB10 doing all of it, nothing left the room but the Discord text.

Recovery during the demo: if captions stall → `deploy/gb10/status.sh`; ASR down → `deploy/gb10/speech.sh asr`
(20 s); decider down → `deploy/gb10/decider.sh 12b` (90 s; meanwhile the runtime falls through to the LLM);
vLLM down → `docker restart nemoclaw-vllm` (3 min; meanwhile decider-only mode still does captions, tasks, devices).

## Sources
- NemoClaw vLLM setup (image, Spark profile, parser flags): https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/inference/local-inference/set-up-vllm
- Qwen3.6-35B-A3B-NVFP4 model card and Spark serving recipe: https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
- NemoClaw installer: https://github.com/NVIDIA/NemoClaw (install.sh, `NEMOCLAW_INSTALL_TAG`)
- NeMo-Speech.cpp: https://github.com/NVIDIA/NeMo-Speech.cpp (docs/server.md, docs/api.md, docs/asr/models.md, docs/tts/models.md); container: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-speech.cpp
- Parakeet on GB10: https://forums.developer.nvidia.com/t/running-parakeet-speech-to-text-on-spark/356353/13
- Strands Decider serving and memory: https://github.com/strands-labs/strands-decider/blob/main/docs/inference.md
- DGX Spark power-off reports and clock lock: https://forums.developer.nvidia.com/t/385991 ; two models share bandwidth: https://forums.developer.nvidia.com/t/386489
- PyTorch cu130 aarch64 wheels: https://download.pytorch.org/whl/cu130/torch/
- NGC PyTorch container: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
