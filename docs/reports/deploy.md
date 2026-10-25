# DEPLOY workstream report (ws/deploy, 2026-10-24)

## 1. What I built

- `deploy/gb10/lib.sh`: shared paths, pinned image tags + arm64 digests, ports, `run` (echo + execute, print-only under `--dry-run`), health wait, HF-cache and docker helpers.
- `deploy/gb10/decider.sh 12b|2b [--dry-run|--stop|--status]`: Strands Decider on the GPU, loopback. Default runtime `docker`: the pinned NVIDIA vLLM image's own CUDA-13/aarch64 torch (`nvcr.io/nvidia/vllm:26.05.post1-py3`, on the box anyway because NemoClaw runs it) plus strands-decider and its pure-Python deps installed from the USB wheelhouse into a per-container user site on first start; `DECIDER_IMAGE=nvcr.io/nvidia/pytorch:26.09-py3` is the second torso; `DECIDER_RUNTIME=native` runs a host venv (torch 2.14.1+cu130 aarch64 wheels from the USB, or the dev-desktop CPU proof). 12B on :8765, 2B on :8766.
- `deploy/gb10/speech.sh asr|tts|all`: NVIDIA NeMo-Speech.cpp 0.2.0 (C++ ggml runtime, arm64 CUDA 13 build for DGX Spark) serving the OpenAI routes: ASR `POST /v1/audio/transcriptions` on :8770 with Parakeet TDT 0.6B v3 Q8 + Nemotron 3 Diarization Q8 (`ASR_MODEL=nemotron35` for the streaming multilingual model), TTS `POST /v1/audio/speech` on :8771 with MagpieTTS 357M Q8 + NanoCodec. Docker (186 MB image) or native tarball runtime.
- `deploy/gb10/vllm.sh`: standalone vLLM for `nvidia/Qwen3.6-35B-A3B-NVFP4` on :8000 from the model card's Spark recipe (Chat Completions, `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3`, `--gpu-memory-utilization 0.40`, fp8 KV, MTP optional), only when `nemoclaw-vllm` is not running; the memory budget is in its header.
- `deploy/gb10/00-preflight.sh`: arch/OS, nvidia-smi, free, df (home + USB), docker + nvidia runtime, port owners, USB manifest presence (`--verify` = full sha256), HF cache, clocks/thermals, opt-in `--cap-clocks`/`--reset-clocks` for the power-off issue.
- `deploy/gb10/up.sh`, `down.sh`, `status.sh`, `check.sh`: ordered start with health waits and logs under `~/unspoken-logs`, stop (never touches `nemoclaw-vllm` without `--all`), one-line-per-server status with probe latency, raw probes + one typed decider question + `UNSPOKEN_PROFILE=gb10 python -m unspoken.models check`.
- `deploy/usb/usbkit.py`: the single list of USB items (images, HF repos, GGUFs, wheels, tools, repos) with pins; `plan` (TSV for bash), `sizes` (HF API, nvcr.io manifests for linux/arm64, PyPI JSON, pip `--dry-run --report` + HEAD; `--offline` reads `sizes.lock.json`), `pip-flags`.
- `deploy/usb/sizes.lock.json`: the sizes measured online on 2026-10-24 (used by tests and `--offline`).
- `deploy/usb/requirements-decider.txt`: the pinned decider stack without torch (52 wheels, all with aarch64/pure wheels).
- `deploy/usb/prepare.sh USB_DIR [--dry-run] [--offline] [--group=…] [--only=…] [--skip-optional] [--no-zstd]`: laptop side (macOS bash 3.2 safe), `docker pull --platform linux/arm64` + `docker save` (zstd when present), `hf download` into a local staging cache then tar per repo (exFAT-safe), GGUFs by filename, `pip download` for aarch64 cp312 (full manylinux tag list), strands-decider wheel built from the pinned git commit, git bundle of the repo, shallow tar of NemoClaw @ `lkg`, `MANIFEST.sha256`, `.done/` markers (resume-safe), `bin/` copy of restore + gb10 scripts.
- `deploy/usb/restore.sh [USB_DIR] [--verify] [--skip-images] [--native-decider]`: GB10 side: docker load, HF cache tars into `~/.cache/huggingface/hub` (where NemoClaw looks), speech/sensors/wheels to `~/unspoken-models`, Magpie tokenizer extraction from the `.nemo`, uv + NeMo-Speech native + NemoClaw source, repo clone from the bundle, offline venv with uv, optional native decider venv.
- `docs/HACKDAY.md`: the runbook (laptop night-before, box steps 1.1–1.7 with commands, expected output, times, failure handling), memory budget table, USB size table and drive choice, the exact `config/models.toml` gb10 blocks, pin rationale, the 3-minute demo checklist.
- `tests/deploy/test_scripts.py`: 34 tests (bash -n, shellcheck when available, loopback-only and no-"hackathon" greps, dry-runs of every script with asserted command lines, prepare/restore dry-runs on a fake USB, usbkit plan↔lock consistency and size bounds, HACKDAY toml blocks parse and point at the fixed ports; one live test under `UNSPOKEN_LIVE=1`).

## 2. How to run it

Laptop (night before, internet, `HF_TOKEN` with Gemma access in the shell):
```bash
deploy/usb/prepare.sh /Volumes/UNSPOKEN/unspoken-usb --dry-run       # sizes + every command
deploy/usb/prepare.sh /Volumes/UNSPOKEN/unspoken-usb                 # ~77 GB download; re-run to resume
```
GB10:
```bash
USB=/media/$USER/UNSPOKEN/unspoken-usb
bash $USB/bin/gb10/00-preflight.sh
bash $USB/bin/restore.sh $USB
cd ~/nemoclaw-src && bash install.sh                                  # NemoClaw express (Wi-Fi), brings up nemoclaw-vllm on :8000
cd ~/unspoken && deploy/gb10/up.sh && deploy/gb10/status.sh         # decider 12B :8765, ASR :8770, TTS :8771, check
deploy/gb10/down.sh
```
Dev desktop proof (CPU): `DECIDER_RUNTIME=native DECIDER_DEVICE=cpu DECIDER_PORT=18765 DECIDER_BIN=<hosting>/model-venv/bin/strands-decider UNSPOKEN_HF_HOME=<hosting>/hf-home deploy/gb10/decider.sh 2b`.

## 3. Verification (this desktop, x86_64, no GPU)

- `bash -n` on all 11 scripts: clean. shellcheck 0.10.0 `-x -S warning` on all: clean (one documented `disable=SC2034` for the shared variable file).
- `--dry-run` of every script (`00-preflight`, `decider 12b/2b`, `speech all`, `vllm`, `up --with-2b`, `down`, `status`, `check`, `prepare --offline`, `restore --native-decider`): exit 0, commands printed as shown in HACKDAY.md.
- `prepare.sh --dry-run` ONLINE: every item sized from the HF API / nvcr.io manifests / PyPI / pip resolution / Ubuntu index: **download 77.5 GiB, on USB 102.1 GiB with plain docker-save tars (≈78 GiB with zstd), required items only 71.2 GiB → 256 GB drive** (after the sensors update, see 7). arm64 digests confirmed: vLLM `sha256:9204569b…` (8.9 GiB compressed, matches NemoClaw's doc), pytorch 26.09 `sha256:307c449d…` (11.4 GiB), nemo-speech.cpp 0.2.0 `sha256:3ad7c3f4…` (178 MiB).
- `prepare.sh` FOR REAL on the small items (`--only=yunet,light-asd,silero-vad,uv,nemoclaw-installer,unspoken-repo,wheels-decider`): 7 done, 83 MB; wheelhouse holds 51 pinned wheels, all `py3-none-any` or `cp312/abi3 manylinux aarch64`, plus `strands_decider-0.1.1.dev8+g930e6d5-py3-none-any.whl` built from the pinned commit; `MANIFEST.sha256` verifies (`sha256sum -c`), the second run skips the finished items.
- Decider CPU proof through `decider.sh` (native runtime, 2B, :18765): up in 8–22 s, `/health` ok, one `/v1/systemone` request with a `noul` + a `choice` question answered in 1.87 s (`has_task 0.86`, `who = Ana 0.75`), `--status` ok, `--stop` closes the port. This also caught and fixed a bug in `run()` that dropped the command name when executing.
- Test suite (branch fast-forwarded to `hack/main` 4f37da6, i.e. with sensors/screen/devices): `.venv/bin/python -m pytest -q` → **566 passed, 4 skipped** (deploy: 36 + 1 live skip; with `SHELLCHECK=…` shellcheck runs inside the suite, `UNSPOKEN_LIVE=1` adds the online sizes test).

## 4. Not verifiable here, and how to verify on the GB10

- The decider on sm_121 inside the vLLM image (`torch.cuda.is_available()`, Gemma 4 12B bf16 load, ~27 GB). Verify: `deploy/gb10/decider.sh 12b` prints `torch … cuda=True NVIDIA GB10`, then `deploy/gb10/check.sh`. Fallbacks are scripted (pytorch:26.09 image; native torch cu130 wheels).
- NeMo-Speech.cpp CUDA on GB10 and its real latency (only RTX 4090 numbers published for the runtime; Parakeet via NeMo measured 282x on GB10). Verify: `curl -F file=@tests/fixtures/x.wav 127.0.0.1:8770/v1/audio/transcriptions`.
- Magpie TTS with `--tokenizer-dir` pointing at the extracted `.nemo` (documented manual route; untested). Optional.
- `pip install --user --break-system-packages` inside the NVIDIA image as a non-root `--user` (PEP 668 marker presence unknown). If it fails, the log says so; `DECIDER_IMAGE=nvcr.io/nvidia/pytorch:26.09-py3` or the native path.
- `docker save` sizes (plain tars assumed 2.2x the compressed size), USB write speed, NemoClaw express install picking up the pre-staged HF cache and the pre-loaded image (documented behaviour, not run).
- `--cap-clocks` (`nvidia-smi -lgc`) support on GB10.
- The running fleet of the other workstreams (`unspoken.models check`, `python -m unspoken.meeting`): check.sh falls back to raw probes when `unspoken/models` is absent.

## 5. Contract changes, deps, config blocks

- Contract changes: none.
- Repo deps: none added (`requirements/deploy.txt` not needed; tests use pytest + stdlib `tomllib`). Deployment-only pins in `deploy/usb/requirements-decider.txt` (all have aarch64 wheels) and the torch-cu130 fallback set (29 wheels resolved by pip for aarch64, 2.9 GiB). Repo requirements of all workstreams so far resolve to 55 aarch64 wheels; `prepare.sh` reports any future pin without an aarch64 wheel by name.
- `config/models.toml` gb10 blocks for the models owner: section 4 of `docs/HACKDAY.md` (decider :8765 `strands-decider`; decider_big `none`/:8766; llm + vlm `openai` :8000/v1 `nvidia/Qwen3.6-35B-A3B-NVFP4`; asr `openai-audio` :8770/v1 `verbose_json`; tts `openai-audio` :8771/v1). The `openai-audio` provider must send multipart WAV to `/v1/audio/transcriptions` and read `text` (+ `words` with `verbose_json`); health = `GET /ready` → 200.
- Sensors owner: the USB carries `sensors/face_detection_yunet_2023mar.onnx`, `sensors/light-asd/{finetuning_TalkSet,pretrain_AVA_CVPR}.model`, `sensors/silero_vad.onnx` under `~/unspoken-models/sensors` on the box; confirm these are the files `unspoken.sensors` loads (edit `ITEMS` in `deploy/usb/usbkit.py` if not).
- Bridge owner: NemoClaw source is restored to `~/nemoclaw-src` at the `lkg` tag; `deploy/nemoclaw/connect.sh` is referenced by HACKDAY.md step 1.6.

## 6. Open risks

- NemoClaw's installer still needs venue Wi-Fi for its CLI (Node/nvm), OpenShell and the sandbox image (~1 GB); no documented offline install exists. Mitigation: `vllm.sh` + the bridge's `discord-rest` fallback run the demo without NemoClaw.
- `google/gemma-4-12B-it` is gated: `prepare.sh` needs an `HF_TOKEN` whose account accepted the licence, or the 12B decider has no base model (the 2B path is fully open).
- Images are built for `12.0+PTX`, not `12.1` explicitly; NemoClaw runs the same vLLM image on Spark, so the decider's torch should follow, but it is the one untested hop in the chain (two fallbacks scripted).
- The `.nemo` tokenizer route for Magpie TTS is untested; TTS is optional.
- The 256 GB drive assumes plain `docker save` tars; with zstd on the laptop a 128 GB drive fits the required set (71 GiB) but leaves no room for the fallback images.

## 7. Sensors update (after ws/sensors merged into hack/main)

USB list (`deploy/usb/usbkit.py`, lock regenerated online):
- added `sensors/wespeaker_en_voxceleb_CAM++.onnx` (29 MB, URL + sha256 = `unspoken/sensors/assets.py`; prepare refuses a wrong hash; restore copies it to `${UNSPOKEN_MODELS_DIR:-~/.cache/unspoken/models}` and re-checks it);
- added `Systran/faster-whisper-base.en` @3d3d5dee (default, 141 MiB) and `small.en` @d1d751a5 (464 MiB) to the HF cache tars;
- added kind `apt`: `libportaudio2` + `alsa-utils` resolved from the noble/noble-updates/noble-security arm64 indexes on ports.ubuntu.com (no docker, no emulation; 10 .debs, 4 MiB, sha256 per .deb from the index). `restore.sh` step 3b installs, with sudo, only the .debs whose package DGX OS lacks (never downgrades), falling back to `apt-get install` from the network; `--skip-apt` skips it;
- dropped YuNet + Silero (vendored in the repo) and both Light-ASD weights (not used);
- the unspoken wheelhouse now resolves 82 aarch64 cp312 wheels from every `requirements*` file (sensors, screen, devices included): every pin has an aarch64 wheel, as the sensors report said.

faster-whisper for the Arm CPU: keep **base.en** as the fallback default. Measured here (x86, CPU int8, 12 fixture lines): base.en 300 ms/line at 8 threads, WER 3.3 %; small.en 885 ms, WER 3.3 %; distil-small.en 800 ms, WER 5.5 %. small.en is on the USB for a noisy room only (HACKDAY §4). Primary ASR stays the GPU speech server.

Day-of: `deploy/gb10/check.sh` now also runs `HF_HUB_OFFLINE=1 python -m unspoken.sensors.check`; `00-preflight.sh` section 9 checks the two apt packages, capture devices (`arecord -l`), `/dev/video*`, the CAM++ sha256 and both faster-whisper caches; HACKDAY step 1.5b; gb10 TOML blocks for vad/diarization/faces/asd and the faster-whisper ASR fallback.

Verified with real files: `prepare.sh --group=sensors` (4 items, manifest verifies), `restore.sh --skip-images` into a scratch HOME, then the repo's own `HF_HUB_OFFLINE=1 python -m unspoken.sensors.check` on exactly that layout: **five `ok` rows** (vad 66 ms, asr 457 ms, diarization 369 ms, faces 24 ms, asd 1 ms); small.en also loads offline (1.0 s).

Bugs this caught and fixed (they would have broken the box offline):
- huggingface_hub 2.x keeps file content in hub-level `blobs/<xx>/<hash>` and the repo's `blobs/` entries are symlinks: the HF tars held 2.5 MB of dangling links. `prepare.sh` now tars the repo dir plus the shared blobs it points to.
- a `--revision` download writes no `refs/main`, and `HF_HUB_OFFLINE=1` / `local_files_only` loaders resolve `main`: faster-whisper failed with `LocalEntryNotFoundError`. prepare writes `refs/main` for pinned repos, restore backfills it for single-snapshot repos. The same gap applied to the pinned `google/gemma-4-12B-it` and `Qwen/Qwen3.5-2B-Base` decider bases.
- rsync 3.5 refuses paths through a symlinked home (`/home -> /local/home`): restore copies with `cp -R`.
- a redone item left stale hashes in `MANIFEST.sha256`; mark_done now replaces its lines.

Still to verify on the GB10: the apt step under sudo on DGX OS 7, `sensors.check` listing the speakerphone and the webcam, base.en latency on the X925/A725 cores.
