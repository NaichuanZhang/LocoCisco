# DOWNLOAD.md: every model, image and installer the GB10 needs

## On a fresh computer

The download script lives in the LocoCisco repo, so the fresh computer needs the repo first. It comes as one
file, `unspoken-handoff.tar` (the repo as a git bundle + this page + checksums).

1. Install the tools.
   - macOS (Apple silicon or Intel): Docker Desktop (running), then `xcode-select --install` (git, curl,
     python3) and `brew install zstd`.
   - Ubuntu/Debian: `sudo apt install -y docker.io git curl python3 python3-venv zstd`, then
     `sudo usermod -aG docker $USER` and log in again.
   - Either: ~60 GB free on the internal disk for staging, plus a 128 GB USB drive formatted exFAT, label `UNSPOKEN`.
2. Unpack and clone:
   ```bash
   tar -xf unspoken-handoff.tar && cd unspoken-handoff && shasum -a 256 -c SHA256SUMS
   git clone --branch hack/main unspoken.bundle unspoken && cd unspoken
   ```
3. Run the commands below from inside `unspoken/`. On Linux the USB path is `/media/$USER/UNSPOKEN/unspoken-usb`.

## Get everything

One command gets all of it (the list, pins and sizes live in `deploy/usb/usbkit.py`; this page mirrors it).
Run it the night before, on a laptop with internet and Docker:

```bash
# 1. accept the Gemma licence on https://huggingface.co/google/gemma-4-12B-it, then:
export HF_TOKEN=hf_...                                                     # shell only, never in a file
deploy/usb/prepare.sh /Volumes/UNSPOKEN/unspoken-usb --dry-run             # sizes, downloads nothing
deploy/usb/prepare.sh /Volumes/UNSPOKEN/unspoken-usb                       # everything: 77.6 GiB
deploy/usb/prepare.sh /Volumes/UNSPOKEN/unspoken-usb --skip-optional       # must-have only: 63.3 GiB
```

Re-run the same command after any failure: finished items are skipped. `--only=<key>` fetches one item.
The script needs `python3` (3.9+), `docker`, `git`, `curl`, `tar` and `zstd` (step 1 above), plus internet to
Hugging Face, nvcr.io, GitHub, PyPI, download.pytorch.org and ports.ubuntu.com. It takes 1-3 hours.

The tables below are for fetching an item by hand when the script cannot.

## Models: must have

| Key | Model | Size | Link | Pin |
|---|---|---|---|---|
| hf-qwen36 | Qwen3.6 35B-A3B NVFP4 (main model, vLLM :8000) | 21.9 GiB | [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | main |
| hf-gemma4-12b | Gemma 4 12B, base of Decider 12B (GATED) | 22.3 GiB | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| hf-decider-12b | Strands Decider 12B | 0.8 GiB | [StrandsAgents/strands-decider-12B-gemma4-v1-2610](https://huggingface.co/StrandsAgents/strands-decider-12B-gemma4-v1-2610) | main |
| hf-qwen35-2b | Qwen3.5 2B Base, base of Decider 2B | 4.3 GiB | [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` |
| hf-decider-2b | Strands Decider 2B | 0.2 GiB | [StrandsAgents/strands-decider-2B-qwen3.5-v1-2610](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610) | `216d1ccb46a9e9b036eb9187b248db98480f77d4` |
| speech-parakeet | Parakeet TDT 0.6B v3, `parakeet-tdt-0.6b-v3.q8_0.gguf` (speech to text) | 0.7 GiB | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | main |
| speech-diar | Nemotron 3 Diarization, `Nemotron-3-Diarization.q8_0.gguf` | 0.1 GiB | [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) | main |
| lightasd-ava | Light-ASD AVA weights (who is talking on camera, MIT) | 4 MiB | [pretrain_AVA_CVPR.model](https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/ed38c232de5efe0261dbd68627c0ade7cdfe14eb/weight/pretrain_AVA_CVPR.model) | sha256 `d44bc3ea7baa8e0946fa3921311714a630ed8b90a1928fab0dbe30d918909317` |
| wespeaker-campp | WeSpeaker CAM++ (voice labels) | 28 MiB | [wespeaker_en_voxceleb_CAM++.onnx](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx) | sha256 `c46fad10b5f81e1aa4a60c162714208577093655076c5450f8c469e522ec54ef` |
| hf-fw-base-en | faster-whisper base.en (CPU speech-to-text backup) | 0.1 GiB | [Systran/faster-whisper-base.en](https://huggingface.co/Systran/faster-whisper-base.en) | `3d3d5dee26484f91867d81cb899cfcf72b96be6c` |
| hf-fw-small-en | faster-whisper small.en (CPU speech-to-text backup) | 0.5 GiB | [Systran/faster-whisper-small.en](https://huggingface.co/Systran/faster-whisper-small.en) | `d1d751a5f8271d482d14ca55d9e2deeebbae577f` |

Silero VAD and the YuNet face detector ship inside the repo (`unspoken/sensors/models/`): nothing to download.

## Models: optional

| Key | Model | Size | Link |
|---|---|---|---|
| speech-nemotron35 | Nemotron 3.5 streaming ASR, `nemotron-3.5-asr-streaming-0.6b.q8_0.gguf` | 0.7 GiB | [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) |
| speech-magpie | Magpie TTS, `magpie_tts_multilingual_357m.v2607.q8_0.gguf` + `magpie_tts_multilingual_357m.nemo` | 1.7 GiB | [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) |
| speech-nanocodec | NanoCodec decoder for Magpie, `nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf` | 75 MiB | [nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps) |
| hf-kokoro | Kokoro TTS | 0.3 GiB | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) |

## Containers (linux/arm64)

| Key | Image | Size | Link |
|---|---|---|---|
| image-vllm | `nvcr.io/nvidia/vllm:26.05.post1-py3` | 8.9 GiB | [NGC vLLM](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm) |
| image-speech | `nvcr.io/nvidia/nemo-speech.cpp:0.2.0` | 0.2 GiB | [NGC NeMo-Speech.cpp](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-speech.cpp) |
| image-pytorch (optional) | `nvcr.io/nvidia/pytorch:26.09-py3` | 11.4 GiB | [NGC PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) |

## Installers and tools

| Key | Item | Size | Link |
|---|---|---|---|
| nemoclaw-installer | NemoClaw bootstrap installer | 16 KiB | [nemoclaw.sh](https://www.nvidia.com/nemoclaw.sh) |
| nemoclaw-repo | NemoClaw source at the `lkg` tag | 0.1 GiB | [NVIDIA/NemoClaw](https://github.com/NVIDIA/NemoClaw/tree/lkg) |
| uv | uv 0.13.5 for aarch64 | 15 MiB | [uv-aarch64-unknown-linux-gnu.tar.gz](https://github.com/astral-sh/uv/releases/download/0.13.5/uv-aarch64-unknown-linux-gnu.tar.gz) |
| nemo-speech-native (optional) | NeMo-Speech.cpp native build, CUDA 13 | 67 MiB | [nemo-speech-0.2.0-linux-aarch64-cuda13.tar.gz](https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.2.0/nemo-speech-0.2.0-linux-aarch64-cuda13.tar.gz) |
| apt-audio | `libportaudio2`, `alsa-utils` + 8 deps (Ubuntu noble arm64 .debs) | 4 MiB | [ports.ubuntu.com](http://ports.ubuntu.com/ubuntu-ports) |
| unspoken-repo | this repo as a git bundle | 10 MiB | built from the local checkout |

## Python packages (linux aarch64, cp312)

| Key | Item | Size | Link |
|---|---|---|---|
| wheels-torch-cu130 | `torch==2.14.1+cu130` + CUDA libs (Light-ASD on the GPU) | 2.9 GiB | [PyTorch cu130 index](https://download.pytorch.org/whl/cu130) |
| wheels-decider | `strands-decider` @ `930e6d5de16309be32579f5631a7369a3322140c` + 51 pinned wheels | 60 MiB | [strands-labs/strands-decider](https://github.com/strands-labs/strands-decider) |
| wheels-sdist | `python_speech_features==0.6`, built into a wheel | 146 KiB | [PyPI](https://pypi.org/project/python_speech_features/0.6/) |
| wheels-unspoken | every `requirements/*.txt` of this repo (98 wheels) | 220 MiB | resolved from PyPI by the script |

## By hand

```bash
hf download nvidia/Qwen3.6-35B-A3B-NVFP4                                    # whole repo into the HF cache
hf download google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7   # needs HF_TOKEN
hf download nvidia/parakeet-tdt-0.6b-v3 parakeet-tdt-0.6b-v3.q8_0.gguf --local-dir speech/nvidia/parakeet-tdt-0.6b-v3
docker pull --platform linux/arm64 nvcr.io/nvidia/vllm:26.05.post1-py3
curl -fL -o lightasd_pretrain_AVA_CVPR.model https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/ed38c232de5efe0261dbd68627c0ade7cdfe14eb/weight/pretrain_AVA_CVPR.model
sha256sum lightasd_pretrain_AVA_CVPR.model                                  # must match the pin above
pip download --platform manylinux_2_28_aarch64 --python-version 3.12 --only-binary=:all: \
    --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple -d wheels/torch-cu130 torch==2.14.1+cu130
```

Then check the drive: `cd /Volumes/UNSPOKEN/unspoken-usb && shasum -a 256 -c MANIFEST.sha256` (no output = good).

Risk: Gemma is gated. Without the licence accepted on your account and `HF_TOKEN` set, Decider 12B cannot download.
