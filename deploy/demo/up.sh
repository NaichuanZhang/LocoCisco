#!/usr/bin/env bash
set -euo pipefail

# Bring up the browser demo on the Mac Studio target (M1 Max, 32 GB), all native, loopback only:
#   1. LM Studio (LLM) must already answer at http://localhost:1234/v1 -- it is a GUI app, not launched here.
#   2. Strands Decider 2B on :8765, native (deploy/mac/decider.sh 2b).
#   3. The demo app on :8780 (deploy/demo/run.sh), native Python from the repo venv.
# No Docker, no vLLM, no GPU image, no USB wheelhouse. Sensing (VAD/ASR/diarization/faces/ASD) is in-process.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LOCOCISCO_RUNTIME="${LOCOCISCO_RUNTIME:-$ROOT/../lococisco-runtime}"
export UNSPOKEN_HOME="$ROOT"
export UNSPOKEN_HF_HOME="${UNSPOKEN_HF_HOME:-$LOCOCISCO_RUNTIME/hf}"
export UNSPOKEN_STATE="${UNSPOKEN_STATE:-$LOCOCISCO_RUNTIME/state}"
export UNSPOKEN_LOGS="${UNSPOKEN_LOGS:-$LOCOCISCO_RUNTIME/logs}"
export UNSPOKEN_PROFILE="${UNSPOKEN_PROFILE:-mac}"
export HF_HOME="$UNSPOKEN_HF_HOME"
LLM_URL="${LLM_URL:-http://localhost:1234/v1}"
LLM_MODEL="${LLM_MODEL:-qwen3-8b}"

test -d "$ROOT/.venv" || { echo "Prepare the application venv first (uv venv && uv pip install -r requirements/...)." >&2; exit 1; }
mkdir -p "$UNSPOKEN_STATE" "$UNSPOKEN_LOGS"

wait_ready() {
  local endpoint="$1"
  for _ in {1..120}; do
    if curl -fsS --max-time 2 "$endpoint" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  echo "Local service did not become ready: $endpoint" >&2
  return 1
}

# 1. LM Studio (LLM). It is a GUI app; this script never launches it.
if ! curl -fsS --max-time 2 "$LLM_URL/models" >/dev/null 2>&1; then
  echo "LM Studio not reachable at $LLM_URL. Open LM Studio, load $LLM_MODEL, and start its Local Server (Developer tab -> Start Server, port 1234), then re-run." >&2
  exit 1
fi

# 2. Strands Decider 2B on :8765 (native).
if ! curl -fsS --max-time 2 http://127.0.0.1:8765/health >/dev/null 2>&1; then
  "$ROOT/deploy/mac/decider.sh" 2b
fi

# 3. The demo app on :8780 (native).
if curl -fsS --max-time 2 http://127.0.0.1:8780/api/replay >/dev/null 2>&1; then
  echo "LocoCisco demo already answering on :8780; leaving it."
else
  "$ROOT/deploy/demo/run.sh"
fi
wait_ready http://127.0.0.1:8780/api/replay
echo "LocoCisco: http://127.0.0.1:8780/ — check the model readiness indicator before playback."
