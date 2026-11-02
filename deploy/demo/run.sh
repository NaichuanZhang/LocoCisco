#!/usr/bin/env bash
set -euo pipefail

# Start the browser demo natively on the Mac Studio target (no container, no GPU image).
# It reuses the already-running LM Studio (LLM) + Strands Decider 2B and does its sensing in-process.
# Staged-video selection (LOCOCISCO_VIDEO, else the official staged demo) is preserved.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="${LOCOCISCO_RUNTIME:-$ROOT/../lococisco-runtime}"
DEFAULT_VIDEO="$RUNTIME/media/lococisco-official-demo.mp4"
VIDEO="${LOCOCISCO_VIDEO:-$DEFAULT_VIDEO}"
if [ "$VIDEO" = "$DEFAULT_VIDEO" ]; then
  ATTRIBUTION="LocoCisco official demo · Presentation/Demo.mp4 · presentation branch"
else
  ATTRIBUTION="Operator-provided local meeting video"
fi

export UNSPOKEN_PROFILE="${UNSPOKEN_PROFILE:-mac}"
export UNSPOKEN_HF_HOME="${UNSPOKEN_HF_HOME:-$RUNTIME/hf}"
export HF_HOME="$UNSPOKEN_HF_HOME"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

test -d "$ROOT/.venv" || { echo "Install the local application venv first." >&2; exit 1; }
test -f "$VIDEO" || { echo "Prepare the local meeting-video fixture first." >&2; exit 1; }
if curl -fsS --max-time 2 http://127.0.0.1:8780/api/replay >/dev/null 2>&1; then
  echo "A demo is already answering on :8780. Stop it before starting another." >&2
  exit 1
fi

MEDIA_DIR="$RUNTIME/media/uploads"
STATE="$RUNTIME/state/demo"
LOGS="${UNSPOKEN_LOGS:-$RUNTIME/logs}"
mkdir -p "$MEDIA_DIR" "$STATE" "$LOGS"

cd "$ROOT"
nohup "$PY" -m unspoken.meeting.demo \
  --media-dir "$MEDIA_DIR" --video "$VIDEO" --host 127.0.0.1 --port 8780 \
  --attribution "${LOCOCISCO_ATTRIBUTION:-$ATTRIBUTION}" \
  >>"$LOGS/demo.log" 2>&1 </dev/null &
echo "$!" >"$STATE/demo.pid"
echo "demo started (pid $(cat "$STATE/demo.pid")); log: $LOGS/demo.log"
