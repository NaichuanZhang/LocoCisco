#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export UNSPOKEN_KIT="${UNSPOKEN_KIT:-$ROOT/../unspoken-usb}"
export LOCOCISCO_RUNTIME="${LOCOCISCO_RUNTIME:-$ROOT/../lococisco-runtime}"
export UNSPOKEN_HOME="$ROOT"
export UNSPOKEN_HF_HOME="$LOCOCISCO_RUNTIME/hf"
export UNSPOKEN_MODELS="$UNSPOKEN_KIT"
export UNSPOKEN_STATE="$LOCOCISCO_RUNTIME/state"
export UNSPOKEN_LOGS="$LOCOCISCO_RUNTIME/logs"
export DECIDER_WHEELS="$UNSPOKEN_KIT/wheels/decider"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export LOCOCISCO_GPU_IMAGE="${LOCOCISCO_GPU_IMAGE:-$(docker inspect --format '{{.Image}}' "${LOCOCISCO_VLLM_CONTAINER:-nemoclaw-vllm}")}"
export DECIDER_IMAGE="$LOCOCISCO_GPU_IMAGE"
: "${UNSPOKEN_LOCAL_API_KEY_FILE:?Set this to the existing local vLLM credential file}"

SPEECH="$LOCOCISCO_RUNTIME/tools/nemo-speech-0.2.0-linux-aarch64-cuda13/bin/nemo-speech"
ASR_MODEL="$UNSPOKEN_KIT/speech/nvidia/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.q8_0.gguf"
test -x "$SPEECH" && test -f "$ASR_MODEL" || { echo "Stage the local speech runtime and model first." >&2; exit 1; }
test -x "$UNSPOKEN_STATE/decider/pyuser/bin/strands-decider" &&
  test -f "$UNSPOKEN_STATE/decider/pyuser/.installed" &&
  ! test "$DECIDER_WHEELS/requirements.txt" -nt "$UNSPOKEN_STATE/decider/pyuser/.installed" ||
  { echo "Prepare the Decider's offline wheel environment first. Startup does not install packages." >&2; exit 1; }
test -d "$ROOT/.venv/lib/python3.12/site-packages" || { echo "Prepare the application environment from local wheels first." >&2; exit 1; }
docker image inspect "$LOCOCISCO_GPU_IMAGE" >/dev/null
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

if ! curl -fsS --max-time 2 http://127.0.0.1:8770/ready >/dev/null 2>&1; then
  if ss -ltn '( sport = :8770 )' | grep -q LISTEN; then
    echo "Waiting for the existing speech listener on :8770."
  else
    nohup "$SPEECH" serve --host 127.0.0.1 --port 8770 --threads 4 \
      --asr-model "$ASR_MODEL" --no-ui >>"$UNSPOKEN_LOGS/asr.log" 2>&1 </dev/null &
    echo "$!" >"$UNSPOKEN_STATE/asr.pid"
  fi
  wait_ready http://127.0.0.1:8770/ready
fi

if ! curl -fsS --max-time 2 http://127.0.0.1:8765/health >/dev/null 2>&1; then
  "$ROOT/deploy/gb10/decider.sh" 12b
fi
NAME="${LOCOCISCO_DEMO_CONTAINER:-lococisco-demo}"
if docker container inspect "$NAME" >/dev/null 2>&1; then
  OWNER="$(docker inspect --format '{{index .Config.Labels "io.lococisco.component"}}' "$NAME")"
  test "$OWNER" = browser-demo || { echo "Container $NAME has no LocoCisco ownership label; inspect it manually." >&2; exit 1; }
  if test "$(docker inspect --format '{{.State.Running}}' "$NAME")" != true; then docker start "$NAME"; fi
else
  "$ROOT/deploy/demo/run.sh"
fi
wait_ready http://127.0.0.1:8780/api/replay
echo "LocoCisco: http://127.0.0.1:8780/ — check the model readiness indicator before playback."
