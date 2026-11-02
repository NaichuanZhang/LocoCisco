#!/usr/bin/env bash
# deploy/mac/00-preflight.sh [--dry-run]
#
# First thing on the Mac Studio. Read-only. Prints one line per check:
#   arch + OS (macOS, Apple Silicon arm64), memory + disk, LM Studio reachable at http://localhost:1234/v1,
#   strands-decider on PATH, who holds our ports (8765 decider, 8780 app), Python / repo venv, sensors
#   (CoreAudio input device and a camera seen by the in-process models).
# No GPU clocks, no nvidia-smi, no Docker: the Mac path is native (LM Studio + strands-decider + in-process
# CPU sensing). Exit code: number of FAIL lines (0 = go).

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_common_flags "$@"
for a in "${ARGS[@]+"${ARGS[@]}"}"; do case "$a" in *) die "unknown flag $a" ;; esac; done
FAILS=0
f() { fail "$*"; FAILS=$((FAILS + 1)); }

log "== 1. machine"
OS="$(uname -s)"
if [ "$OS" = Darwin ]; then ok "OS macOS ($(sw_vers -productVersion 2>/dev/null || uname -r))"; else f "OS is $OS, expected Darwin (macOS). Dry-run on another box is fine."; fi
ARCH="$(uname -m)"
if [ "$ARCH" = arm64 ]; then ok "arch arm64 (Apple Silicon)"; else f "arch is $ARCH, expected arm64 (Apple Silicon M1 Max). Dry-run on an x86 box is fine."; fi

log "== 2. memory and disk"
if have sysctl; then
  MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
  MEM_GB=$((MEM_BYTES / 1024 / 1024 / 1024))
  if [ "$MEM_GB" -ge 30 ]; then ok "RAM ${MEM_GB} GB unified (budget assumes 32 GB)"; else warn "RAM ${MEM_GB} GB: below the 32 GB budget; keep the LLM context modest"; fi
fi
{ df -h "$HOME" 2>/dev/null | sed 's/^/     /' >&2; } || true
HOME_FREE_GB="$(df -g "$HOME" 2>/dev/null | awk 'NR==2 {print $4}')"
if [ "${HOME_FREE_GB:-0}" -ge 20 ]; then ok "home has ${HOME_FREE_GB} GB free"; else warn "home has only ${HOME_FREE_GB:-?} GB free (models + caches want ~20 GB)"; fi

log "== 3. LM Studio (LLM) at $LLM_URL"
if [ "$DRY_RUN" = 1 ]; then printf '+ curl -fsS -m 5 %s/models\n' "$LLM_URL" >&2
elif curl -fsS -m 5 "$LLM_URL/models" >/dev/null 2>&1; then ok "LM Studio answering at $LLM_URL (expected model: $LLM_MODEL)"
else f "LM Studio not reachable at $LLM_URL. Start LM Studio, load $LLM_MODEL, and turn on its Local Server (OpenAI-compatible, port 1234). It is a GUI app; this script does not launch it."; fi

log "== 4. decider binary"
if have "$DECIDER_BIN" || [ -x "$DECIDER_BIN" ]; then ok "strands-decider found ($(command -v "$DECIDER_BIN" 2>/dev/null || echo "$DECIDER_BIN"))"
else f "strands-decider not found (pip install it into the repo venv, or set DECIDER_BIN=/path/to/strands-decider)"; fi

log "== 5. ports (127.0.0.1)"
for p in "$PORT_DECIDER_2B" "$PORT_APP"; do
  if port_open "$p"; then
    who="$( (lsof -nP -iTCP:"$p" -sTCP:LISTEN 2>/dev/null | awk 'NR==2 {print $1" (pid "$2")"}') || true)"
    warn "port $p in use ${who:+by $who}"
  else ok "port $p free"; fi
done

log "== 6. python and repo venv ($UNSPOKEN_HOME)"
PY="$UNSPOKEN_HOME/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || true)"
if [ -n "$PY" ] && [ -x "$PY" ]; then ok "python: $PY ($("$PY" --version 2>&1))"; else f "no python3 (install the repo venv: uv venv && uv pip install -r requirements/...)"; fi
if [ -d "$UNSPOKEN_HOME/unspoken" ]; then ok "repo at $UNSPOKEN_HOME"; else warn "repo not at $UNSPOKEN_HOME (set UNSPOKEN_HOME); fine before the checkout is staged"; fi

log "== 7. sensors (mic, camera)"
if have system_profiler; then
  NMIC="$(system_profiler SPAudioDataType 2>/dev/null | grep -c -i 'Input' || true)"
  if [ "${NMIC:-0}" -gt 0 ]; then ok "audio input device(s) seen"; else warn "no audio input device seen (plug the USB speakerphone)"; fi
  NCAM="$(system_profiler SPCameraDataType 2>/dev/null | grep -c -i 'Model ID\|Unique ID' || true)"
  if [ "${NCAM:-0}" -gt 0 ]; then ok "camera(s) seen"; else warn "no camera seen (plug the USB webcam or enable the built-in camera)"; fi
fi

log "== result: $FAILS FAIL"
[ "$DRY_RUN" = 1 ] && exit 0
exit "$FAILS"
