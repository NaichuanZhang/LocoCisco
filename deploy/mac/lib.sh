#!/usr/bin/env bash
# Shared helpers for deploy/mac/*.sh. Source it; do not run it.
#
#   source "$(dirname "$0")/lib.sh"
#   parse_common_flags "$@"      # sets DRY_RUN=1 for --dry-run / -n, drops the flag from ARGS
#   run strands-decider --help   # prints "+ strands-decider --help" and runs it (prints only, under --dry-run)
#
# The Mac Studio (Apple M1 Max, 32 GB unified memory) path: all inference is local and native.
#   - the LLM is a small Qwen served by LM Studio (OpenAI-compatible server at http://localhost:1234/v1)
#   - the decider runs via `strands-decider serve` (native, CPU / Metal)
#   - sensing (VAD, ASR, diarization, faces, ASD) is in-process Python on the CPU
# No Docker, no vLLM, no NeMo-Speech.cpp, no nvidia-smi, no GPU clocks. Every server binds 127.0.0.1 /
# localhost. Every script accepts --dry-run and prints each command it would run.
#
# Paths (override with the environment):
#   UNSPOKEN_HOME    repo checkout on the Mac               default ~/unspoken
#   UNSPOKEN_HF_HOME Hugging Face cache (HF_HOME)           default ~/.cache/huggingface
#   UNSPOKEN_LOGS    one log per server                     default ~/unspoken-logs
#   UNSPOKEN_STATE   per-server scratch (decider pid file)  default ~/unspoken-state

# shellcheck disable=SC2034  # the variables below are used by the scripts that source this file
set -euo pipefail

UNSPOKEN_HOME="${UNSPOKEN_HOME:-$HOME/unspoken}"
UNSPOKEN_HF_HOME="${UNSPOKEN_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}"
UNSPOKEN_LOGS="${UNSPOKEN_LOGS:-$HOME/unspoken-logs}"
UNSPOKEN_STATE="${UNSPOKEN_STATE:-$HOME/unspoken-state}"

# Ports / endpoints (match config/models.toml [profiles.mac]; all loopback).
PORT_DECIDER_2B=8765                                       # `strands-decider serve` (decider-2b)
PORT_APP=8780                                              # the browser demo app
LLM_URL="${LLM_URL:-http://localhost:1234/v1}"             # LM Studio's OpenAI-compatible server
LLM_MODEL="${LLM_MODEL:-qwen3-8b}"                         # the model id LM Studio serves (Qwen3-8B 4-bit)

# The decider served natively on the Mac (CPU / Metal).
DECIDER_MODEL="${DECIDER_MODEL:-decider-2b}"
DECIDER_BIN="${DECIDER_BIN:-strands-decider}"

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

# port_open PORT: 0 when something listens on 127.0.0.1:PORT (lsof on macOS; /dev/tcp fallback).
port_open() {
  if have lsof; then lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
  else (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; fi
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

mkdirs() { [ "$DRY_RUN" = 1 ] || mkdir -p "$UNSPOKEN_LOGS" "$UNSPOKEN_STATE"; }

# bytes_h N: human size.
bytes_h() { awk -v b="$1" 'BEGIN{ s="B KB MB GB TB"; split(s,u," "); i=1; while (b>=1024 && i<5){ b/=1024; i++ } printf "%.1f %s", b, u[i] }'; }
