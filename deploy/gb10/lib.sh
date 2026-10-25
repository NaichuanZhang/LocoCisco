#!/usr/bin/env bash
# Shared helpers for deploy/gb10/*.sh. Source it; do not run it.
#
#   source "$(dirname "$0")/lib.sh"
#   parse_common_flags "$@"      # sets DRY_RUN=1 for --dry-run / -n, drops the flag from ARGS
#   run docker ps                # prints "+ docker ps" and runs it (prints only, under --dry-run)
#
# Every server binds 127.0.0.1. Every script accepts --dry-run and prints each command it would run.
# Paths (override with the environment):
#   UNSPOKEN_HOME    repo checkout on the box             default ~/unspoken
#   UNSPOKEN_HF_HOME Hugging Face cache (HF_HOME)         default ~/.cache/huggingface
#   UNSPOKEN_MODELS  speech GGUFs, wheels, tools          default ~/unspoken-models
#   UNSPOKEN_LOGS    one log per server                   default ~/unspoken-logs
#   UNSPOKEN_STATE   per-server scratch (decider venv)    default ~/unspoken-state
#   UNSPOKEN_USB     the USB mount (restore.sh reads it)  default: first /media/*/*/unspoken-usb found

# shellcheck disable=SC2034  # the variables below are used by the scripts that source this file
set -euo pipefail

UNSPOKEN_HOME="${UNSPOKEN_HOME:-$HOME/unspoken}"
UNSPOKEN_HF_HOME="${UNSPOKEN_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}"
UNSPOKEN_MODELS="${UNSPOKEN_MODELS:-$HOME/unspoken-models}"
UNSPOKEN_LOGS="${UNSPOKEN_LOGS:-$HOME/unspoken-logs}"
UNSPOKEN_STATE="${UNSPOKEN_STATE:-$HOME/unspoken-state}"

# Pinned images (arm64 digests verified against nvcr.io on 2026-10-24; see docs/HACKDAY.md).
IMAGE_VLLM="${IMAGE_VLLM:-nvcr.io/nvidia/vllm:26.05.post1-py3}"              # NemoClaw's managed vLLM image
IMAGE_VLLM_DIGEST_ARM64="sha256:9204569b17ee4c0eff75194b8e6e458479c8aee18953b5ab9cf359fcdac659e2"
IMAGE_PYTORCH="${IMAGE_PYTORCH:-nvcr.io/nvidia/pytorch:26.09-py3}"            # fallback torso for the decider
IMAGE_PYTORCH_DIGEST_ARM64="sha256:307c449df7dabed6b36e017a2b2f2e45bdec9dc203b81507111315f1e971de6e"
IMAGE_SPEECH="${IMAGE_SPEECH:-nvcr.io/nvidia/nemo-speech.cpp:0.2.0}"          # ASR + diarization + TTS runtime
IMAGE_SPEECH_DIGEST_ARM64="sha256:3ad7c3f4c600e02f896da577a7ed3c3628996cc97b072f8ea21c282c4fe38098"

# Ports (fixed by docs/PLAN.md; all 127.0.0.1).
PORT_VLLM=8000
PORT_DECIDER_12B=8765
PORT_DECIDER_2B=8766
PORT_ASR=8770
PORT_TTS=8771

# Models.
LLM_MODEL="${LLM_MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}"
DECIDER_12B_REPO="StrandsAgents/strands-decider-12B-gemma4-v1-2610"
DECIDER_12B_BASE="google/gemma-4-12B-it"
DECIDER_2B_REPO="StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
DECIDER_2B_BASE="Qwen/Qwen3.5-2B-Base"
ASR_GGUF="${ASR_GGUF:-nvidia/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.q8_0.gguf}"
ASR_STREAMING_GGUF="nvidia/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
DIAR_GGUF="${DIAR_GGUF:-nvidia/Nemotron-3-Diarization/Nemotron-3-Diarization.q8_0.gguf}"
TTS_GGUF="${TTS_GGUF:-nvidia/magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.v2607.q8_0.gguf}"
TTS_CODEC_GGUF="${TTS_CODEC_GGUF:-nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf}"
TTS_TOKENIZER_DIR="${TTS_TOKENIZER_DIR:-nvidia/magpie_tts_multilingual_357m/tokenizer}"

DRY_RUN="${DRY_RUN:-0}"
ARGS=()

if [ -t 2 ]; then
  C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else
  C_RED=""; C_GRN=""; C_YEL=""; C_DIM=""; C_OFF=""
fi

log()  { printf '%s[%s]%s %s\n' "$C_DIM" "$(date +%H:%M:%S)" "$C_OFF" "$*" >&2; }
ok()   { printf '%sOK%s   %s\n' "$C_GRN" "$C_OFF" "$*" >&2; }
warn() { printf '%sWARN%s %s\n' "$C_YEL" "$C_OFF" "$*" >&2; }
fail() { printf '%sFAIL%s %s\n' "$C_RED" "$C_OFF" "$*" >&2; }
die()  { fail "$*"; exit 1; }

# parse_common_flags "$@": strips --dry-run/-n (sets DRY_RUN=1) and -h/--help (prints the script header).
parse_common_flags() {
  ARGS=()
  for a in "$@"; do
    case "$a" in
      --dry-run|-n) DRY_RUN=1 ;;
      -h|--help) usage; exit 0 ;;
      *) ARGS+=("$a") ;;
    esac
  done
}

# usage: prints the leading comment block of the calling script.
usage() { sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; }

# run CMD...: echo the command, then execute it unless DRY_RUN=1.
run() {
  local a first=1
  for a in "$@"; do
    if [ "$first" = 1 ]; then printf '+ %q' "$a" >&2; first=0; else printf ' %q' "$a" >&2; fi
  done
  printf '\n' >&2
  [ "$DRY_RUN" = 1 ] && return 0
  "$@"
}

# run_sh 'shell snippet': same for a snippet that needs a pipe or redirection.
run_sh() {
  printf '+ %s\n' "$1" >&2
  [ "$DRY_RUN" = 1 ] && return 0
  bash -c "$1"
}

need() { command -v "$1" >/dev/null 2>&1 || die "missing tool: $1"; }
have() { command -v "$1" >/dev/null 2>&1; }

# docker_running NAME: 0 when a container with that exact name is running.
docker_running() { have docker && [ "$(docker ps -q -f "name=^/$1\$" 2>/dev/null | wc -l)" -gt 0 ]; }

# docker_exists NAME: 0 when a container with that name exists in any state.
docker_exists() { have docker && [ "$(docker ps -aq -f "name=^/$1\$" 2>/dev/null | wc -l)" -gt 0 ]; }

# docker_rm_if_exists NAME
docker_rm_if_exists() { if docker_exists "$1"; then run docker rm -f "$1"; fi; }

# port_open PORT: 0 when something listens on 127.0.0.1:PORT.
port_open() {
  if have ss; then ss -ltn 2>/dev/null | grep -q ":$1 "; else (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; fi
}

# wait_http URL SECONDS [LABEL]: poll until the URL answers 2xx (prints the time it took).
wait_http() {
  local url="$1" secs="$2" label="${3:-$1}" t=0
  if [ "$DRY_RUN" = 1 ]; then printf '+ (wait up to %ss for %s)\n' "$secs" "$url" >&2; return 0; fi
  while [ "$t" -lt "$secs" ]; do
    if curl -fsS -m 3 -o /dev/null "$url" 2>/dev/null; then ok "$label up after ${t}s"; return 0; fi
    sleep 2; t=$((t + 2))
  done
  fail "$label not up after ${secs}s ($url)"; return 1
}

# hf_snapshot REPO: newest snapshot dir of REPO in the HF cache (empty when absent).
hf_snapshot() {
  local d="$UNSPOKEN_HF_HOME/hub/models--${1//\//--}/snapshots"
  [ -d "$d" ] || { echo ""; return 0; }
  # shellcheck disable=SC2012
  ls -1dt "$d"/* 2>/dev/null | head -1
}

# hf_cached REPO: 0 when the repo has a snapshot with at least one file.
hf_cached() { local s; s="$(hf_snapshot "$1")"; [ -n "$s" ] && [ -n "$(ls -A "$s" 2>/dev/null)" ]; }

# gpu_flag: "--gpus all" normally; empty with DECIDER_DEVICE=cpu / NO_GPU=1 (dev desktop tests).
gpu_flag() { if [ "${NO_GPU:-0}" = 1 ]; then echo ""; else echo "--gpus all"; fi; }

mkdirs() { [ "$DRY_RUN" = 1 ] || mkdir -p "$UNSPOKEN_LOGS" "$UNSPOKEN_STATE"; }

# bytes_h N: human size.
bytes_h() { awk -v b="$1" 'BEGIN{ s="B KB MB GB TB"; split(s,u," "); i=1; while (b>=1024 && i<5){ b/=1024; i++ } printf "%.1f %s", b, u[i] }'; }
