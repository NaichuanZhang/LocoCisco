#!/bin/bash
# Start / stop the local llama.cpp test server behind the "dev-local" profile (config/models.toml):
# an OpenAI-compatible Chat Completions server with tool calls, on 127.0.0.1:8010, CPU only.
#
#   scripts/dev-llm.sh start     # downloads nothing: expects the pinned files below under DEV_MODELS
#   scripts/dev-llm.sh stop
#   scripts/dev-llm.sh status
#
# Pinned artefacts (public sources, fetched once into DEV_MODELS):
#   llama.cpp build b11869, llama-b11869-bin-ubuntu-x64.tar.gz
#     https://github.com/ggml-org/llama.cpp/releases/download/b11869/llama-b11869-bin-ubuntu-x64.tar.gz
#     sha256 529ebb30c8aafcfd7d2ae0606aef346a3f3209d0a49addf382070c0c78fa2a6e
#     (the same release ships llama-b11869-bin-ubuntu-arm64.tar.gz and -cuda-13.4-arm64 for the GB10)
#   Qwen/Qwen3-4B-GGUF  Qwen3-4B-Q4_K_M.gguf  (Apache-2.0)
#     https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf
#     sha256 7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5
#
# Why these flags: --jinja turns on the model's own chat template, which is what makes OpenAI-style
# `tools` and streamed `tool_calls` work; --reasoning-format deepseek keeps Qwen3's <think> out of the
# visible text (the dev-local profile also sends enable_thinking=false); -c 8192 fits the house prompt
# with 12 tools several times over; --alias is the model id the profile sends.
set -euo pipefail

DEV_MODELS="${DEV_MODELS:-/local/home/zhangnai/kirocrew-sessions/2026-10-23-dell-nvidia-hackathon/dev-models}"
LLAMA_BUILD="${LLAMA_BUILD:-b11869}"
LLAMA_DIR="${LLAMA_DIR:-$DEV_MODELS/llama.cpp/llama-$LLAMA_BUILD}"
GGUF="${GGUF:-$DEV_MODELS/gguf/Qwen3-4B-Q4_K_M.gguf}"
ALIAS="${ALIAS:-qwen3-4b}"
HOST=127.0.0.1
PORT="${PORT:-8010}"
THREADS="${THREADS:-16}"
RUN_DIR="${KIROCREW_SCRATCH:-${TMPDIR:-/tmp}}/unspoken-dev-llm"
PID_FILE="$RUN_DIR/llama-server.pid"
LOG_FILE="$RUN_DIR/llama-server.log"

status() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "llama-server running (pid $(cat "$PID_FILE")) on http://$HOST:$PORT/v1, log $LOG_FILE"
    curl -sS "http://$HOST:$PORT/health" && echo
    return 0
  fi
  echo "llama-server not running"
  return 1
}

start() {
  mkdir -p "$RUN_DIR"
  if status >/dev/null 2>&1; then status; return 0; fi
  [ -x "$LLAMA_DIR/llama-server" ] || { echo "missing $LLAMA_DIR/llama-server (see the pinned URL in this script)"; exit 2; }
  [ -f "$GGUF" ] || { echo "missing $GGUF (see the pinned URL in this script)"; exit 2; }
  export LD_LIBRARY_PATH="$LLAMA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  nohup "$LLAMA_DIR/llama-server" -m "$GGUF" --alias "$ALIAS" --host "$HOST" --port "$PORT" \
      --jinja --reasoning-format deepseek -c 8192 -t "$THREADS" --parallel 1 --no-webui \
      > "$LOG_FILE" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  echo -n "starting llama-server (pid $!) "
  for _ in $(seq 1 120); do
    if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then echo; status; return 0; fi
    kill -0 "$(cat "$PID_FILE")" 2>/dev/null || { echo; echo "llama-server died; last log lines:"; tail -20 "$LOG_FILE"; exit 1; }
    echo -n "."; sleep 1
  done
  echo; echo "llama-server did not become healthy in 120 s; log: $LOG_FILE"; exit 1
}

stop() {
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
      kill "$PID"; for _ in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
      kill -0 "$PID" 2>/dev/null && kill -9 "$PID" || true
      echo "stopped llama-server (pid $PID)"
    fi
    rm -f "$PID_FILE"
  else
    echo "llama-server not running (no pid file)"
  fi
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  restart) stop; start ;;
  *) echo "usage: $0 start|stop|status|restart"; exit 64 ;;
esac
