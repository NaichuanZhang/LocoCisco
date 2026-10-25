#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KIT="${UNSPOKEN_KIT:-$ROOT/../unspoken-usb}"
RUNTIME="${LOCOCISCO_RUNTIME:-$ROOT/../lococisco-runtime}"
NAME="${LOCOCISCO_DEMO_CONTAINER:-lococisco-demo}"
DEFAULT_VIDEO="$RUNTIME/media/lococisco-official-demo.mp4"
VIDEO="${LOCOCISCO_VIDEO:-$DEFAULT_VIDEO}"
if [ "$VIDEO" = "$DEFAULT_VIDEO" ]; then
  ATTRIBUTION="LocoCisco official demo · Presentation/Demo.mp4 · presentation branch"
else
  ATTRIBUTION="Operator-provided local meeting video"
fi
KEY="${UNSPOKEN_LOCAL_API_KEY_FILE:?Set UNSPOKEN_LOCAL_API_KEY_FILE to the existing local vLLM credential file}"
IMAGE="${LOCOCISCO_GPU_IMAGE:-$(docker inspect --format '{{.Image}}' "${LOCOCISCO_VLLM_CONTAINER:-nemoclaw-vllm}")}"

docker image inspect "$IMAGE" >/dev/null
test -f "$KEY" || { echo "Local vLLM credential file is missing." >&2; exit 1; }
test -d "$ROOT/.venv/lib/python3.12/site-packages" || { echo "Install the local application wheel environment first." >&2; exit 1; }
test -f "$VIDEO" || { echo "Prepare the local meeting-video fixture first." >&2; exit 1; }
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "Container $NAME already exists. Inspect it before restarting; this command will not replace it." >&2
  exit 1
fi
mkdir -p "$RUNTIME/state/demo"
exec docker run -d --pull=never --name "$NAME" \
  --label io.lococisco.component=browser-demo --network host --gpus all --ipc host \
  --user "$(id -u):$(id -g)" --workdir /app \
  --mount "type=bind,src=$ROOT,dst=/app,readonly" \
  --mount "type=bind,src=$KIT/sensors,dst=/models,readonly" \
  --mount "type=bind,src=$RUNTIME,dst=/runtime" \
  --mount "type=bind,src=$VIDEO,dst=/input/meeting.mp4,readonly" \
  --mount "type=bind,src=$KEY,dst=/run/secrets/vllm-api-key,readonly" \
  -e PYTHONPATH=/app:/app/.venv/lib/python3.12/site-packages \
  -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/runtime/state/demo \
  -e HF_HOME=/runtime/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_DISABLE_TELEMETRY=1 -e UNSPOKEN_MODELS_DIR=/models \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=2 \
  --entrypoint python "$IMAGE" -m unspoken.meeting.demo \
  --media-dir /runtime/media/uploads --video /input/meeting.mp4 \
  --api-key-file /run/secrets/vllm-api-key --port 8780 \
  --attribution "${LOCOCISCO_ATTRIBUTION:-$ATTRIBUTION}"
