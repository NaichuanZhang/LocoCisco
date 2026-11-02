# DOWNLOAD.md: every model, app and wheel the Mac Studio needs

## On a fresh computer

The download script lives in the LocoCisco repo, so the fresh computer needs the repo first. It comes as one
file, `unspoken-handoff.tar` (the repo as a git bundle + this page + checksums).

1. Install the tools.
   - macOS (Apple Silicon): `xcode-select --install` (git, curl, python3), then `brew install uv zstd`.
     Install **LM Studio** (https://lmstudio.ai) and open it once.
   - (Staging from a Linux/x86 laptop also works: `sudo apt install -y git curl python3 python3-venv zstd`
     and a recent `uv`; the files are platform-neutral except the Python wheels, which resolve for macOS arm64.)
   - Either: ~20 GB free for staging, plus a plain folder or any drive (exFAT if shared with Linux), label `UNSPOKEN`.
2. Unpack and clone:
   ```bash
   tar -xf unspoken-handoff.tar && cd unspoken-handoff && shasum -a 256 -c SHA256SUMS
   git clone unspoken.bundle lococisco && cd lococisco
   ```
3. Run the commands below from inside `lococisco/`. The staging path is `~/unspoken-usb` (or `/Volumes/UNSPOKEN/unspoken-usb`).

## Get everything

One command gets all of it (the list, pins and sizes live in `deploy/usb/usbkit.py`; this page mirrors it).
Run it the night before, on a Mac (or laptop) with internet:

```bash
deploy/usb/prepare.sh ~/unspoken-usb --dry-run             # sizes, downloads nothing
deploy/usb/prepare.sh ~/unspoken-usb                       # everything: ~13 GiB
```

No model here is gated, so no Hugging Face licence or `HF_TOKEN` is needed. Re-run the same command after any
failure: finished items are skipped. `--only=<key>` fetches one item. The script needs `python3` (3.9+), `git`,
`curl`, `tar`, `zstd` and `uv` (step 1), plus internet to Hugging Face and PyPI. It takes 20–40 minutes. The
LLM GGUF is easiest pulled from inside LM Studio (see below); the kit copy is the offline fallback.

The tables below are for fetching an item by hand when the script cannot.

## The language model (LM Studio)

The LLM is served by LM Studio on the Mac; there is no vLLM or container. Load the model from inside LM Studio
(model search → download → Developer → Start Server on port 1234), or stage the GGUF for an offline Mac.

| Key | Model | Size | Link |
|---|---|---|---|
| llm-qwen3-8b | Qwen3-8B, 4-bit GGUF (`Q4_K_M`), main LLM (LM Studio :1234) | ~4.9 GiB | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) (GGUF builds under `lmstudio-community` / `Qwen`) |
| llm-qwen3-4b (fallback) | Qwen3-4B, 4-bit GGUF, for a tighter memory budget | ~3 GiB | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |

In LM Studio, enable tool / function calling and keep the context modest (the `mac` profile caps replies at 400
tokens). An MLX build works too if you prefer Apple's runtime; the OpenAI endpoint is the same.

## Models: must have

| Key | Model | Size | Link | Pin |
|---|---|---|---|---|
| hf-qwen35-2b | Qwen3.5 2B Base, base of Decider 2B | 4.3 GiB | [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` |
| hf-decider-2b | Strands Decider 2B | 0.2 GiB | [StrandsAgents/strands-decider-2B-qwen3.5-v1-2610](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610) | `216d1ccb46a9e9b036eb9187b248db98480f77d4` |
| onnx-parakeet | Parakeet TDT 0.6B v3, ONNX export (speech to text, onnx-asr, CPU) | ~2.4 GiB | [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) | main (CC-BY-4.0; export of [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)) |
| lightasd-ava | Light-ASD AVA weights (who is talking on camera, MIT) | 4 MiB | [pretrain_AVA_CVPR.model](https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/ed38c232de5efe0261dbd68627c0ade7cdfe14eb/weight/pretrain_AVA_CVPR.model) | sha256 `d44bc3ea7baa8e0946fa3921311714a630ed8b90a1928fab0dbe30d918909317` |
| wespeaker-campp | WeSpeaker CAM++ (voice labels) | 28 MiB | [wespeaker_en_voxceleb_CAM++.onnx](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx) | sha256 `c46fad10b5f81e1aa4a60c162714208577093655076c5450f8c469e522ec54ef` |

Silero VAD and the YuNet face detector ship inside the repo (`unspoken/sensors/models/`): nothing to download.
onnx-asr downloads the Parakeet export into `~/.cache` on first use if you did not stage it.

## Models: optional

| Key | Model | Size | Link |
|---|---|---|---|
| hf-fw-base-en | faster-whisper base.en (alternate CPU speech-to-text) | 0.1 GiB | [Systran/faster-whisper-base.en](https://huggingface.co/Systran/faster-whisper-base.en) |
| hf-fw-small-en | faster-whisper small.en (alternate CPU speech-to-text) | 0.5 GiB | [Systran/faster-whisper-small.en](https://huggingface.co/Systran/faster-whisper-small.en) |

The `mac` profile uses Parakeet via onnx-asr; faster-whisper is kept only as an in-process CPU alternative
(`UNSPOKEN_ASR_PROVIDER=python:unspoken.sensors.asr_local:FasterWhisperASR`). TTS and the whiteboard VLM are
disabled on the Mac target, so no TTS or codec weights are staged.

## Installers and tools

| Key | Item | Size | Link |
|---|---|---|---|
| lm-studio | LM Studio for macOS (Apple Silicon) | ~0.5 GiB | [lmstudio.ai](https://lmstudio.ai) |
| uv | uv 0.13.5 for macOS arm64 | 15 MiB | [uv-aarch64-apple-darwin.tar.gz](https://github.com/astral-sh/uv/releases/download/0.13.5/uv-aarch64-apple-darwin.tar.gz) |
| unspoken-repo | this repo as a git bundle | 10 MiB | built from the local checkout |

## Python packages (macOS arm64, cp312)

Installed into the repo venv by `scripts/dev-venv.sh` (`uv venv` + the pinned wheels). Stage them for an offline
Mac with `--find-links`; torch resolves to the Apple Silicon build (CPU / MPS, no CUDA).

| Key | Item | Size | Link |
|---|---|---|---|
| wheels-decider | `strands-decider` @ `930e6d5de16309be32579f5631a7369a3322140c` + pinned deps (incl. torch for Apple Silicon) | ~0.4 GiB | [strands-labs/strands-decider](https://github.com/strands-labs/strands-decider) |
| wheels-sdist | `python_speech_features==0.6`, built into a wheel | 146 KiB | [PyPI](https://pypi.org/project/python_speech_features/0.6/) |
| wheels-unspoken | every `requirements/*.txt` of this repo (onnxruntime, onnx-asr, OpenCV, …) | ~0.25 GiB | resolved from PyPI by the script |

## By hand

```bash
# LLM: easiest inside LM Studio; to stage the GGUF directly:
hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir lmstudio/Qwen3-8B    # repo/name may differ by GGUF publisher
hf download Qwen/Qwen3.5-2B-Base --revision b1485b2fa6dfa1287294f269f5fb618e03d52d7c
hf download StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --revision 216d1ccb46a9e9b036eb9187b248db98480f77d4
hf download istupakov/parakeet-tdt-0.6b-v3-onnx --local-dir ~/.cache/unspoken/dev-models/asr/parakeet-tdt-0.6b-v3
curl -fL -o lightasd_pretrain_AVA_CVPR.model https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/ed38c232de5efe0261dbd68627c0ade7cdfe14eb/weight/pretrain_AVA_CVPR.model
shasum -a 256 lightasd_pretrain_AVA_CVPR.model                              # must match the pin above
```

Then check the drive: `cd ~/unspoken-usb && shasum -a 256 -c MANIFEST.sha256` (no output = good).

Note: GGUF publishers and file names vary (e.g. `lmstudio-community`, `Qwen`, `bartowski`); pick a 4-bit
`Q4_K_M` build of Qwen3-8B. Loading it from inside LM Studio avoids guessing the exact repo path.
