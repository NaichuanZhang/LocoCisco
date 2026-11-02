#!/usr/bin/env bash
# deploy/mac/decider.sh 2b [--dry-run] [--stop] [--status]
#
# The Strands Decider on the Mac Studio, native, loopback only (the 2B is the only decider at 32 GB):
#   2b -> 127.0.0.1:8765  decider-2b, served by `strands-decider serve` on the CPU / Apple Silicon (Metal)
#
# Runtime: native only. No Docker, no vLLM image, no --gpus. `strands-decider` must be on PATH (or set
# DECIDER_BIN=/path/to/strands-decider); deploy/mac/00-preflight.sh checks it. The model `decider-2b` is
# resolved by strands-decider itself (no checkpoint path, no HF snapshot staging needed here).
#
# Env: DECIDER_PORT (override the port)   DECIDER_BIN (override the binary)
#      DECIDER_MODEL (default decider-2b) DECIDER_EXTRA="..." extra `strands-decider serve` flags
# Health: curl -s 127.0.0.1:8765/health ; ask: POST /v1/systemone {"state":..., "questions":{...}}

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_common_flags "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"

SIZE="${1:-}"; shift || true
ACTION=start
for a in "$@"; do case "$a" in --stop) ACTION=stop ;; --status) ACTION=status ;; *) die "unknown flag $a" ;; esac; done

case "$SIZE" in
  2b) MODEL="$DECIDER_MODEL"; PORT="${DECIDER_PORT:-$PORT_DECIDER_2B}" ;;
  *) usage; exit 2 ;;                        # no 12B decider on the Mac target (32 GB)
esac

BIN="$DECIDER_BIN"
NAME="unspoken-decider-$SIZE"
LOG="$UNSPOKEN_LOGS/decider-$SIZE.log"
STATE="$UNSPOKEN_STATE/decider"
PIDFILE="$STATE/$NAME.pid"
HEALTH="http://127.0.0.1:$PORT/health"

status() {
  if [ "$DRY_RUN" = 1 ]; then printf '+ curl -fsS %s\n' "$HEALTH" >&2; return 0; fi
  if curl -fsS -m 3 "$HEALTH" 2>/dev/null; then echo; ok "decider $SIZE answers on :$PORT"; return 0; fi
  fail "decider $SIZE not answering on :$PORT"; return 1
}

stop() {
  if [ -f "$PIDFILE" ]; then run kill "$(cat "$PIDFILE")" || true; [ "$DRY_RUN" = 1 ] || rm -f "$PIDFILE"
  else warn "no pid file $PIDFILE"; fi
}

start() {
  if [ "$DRY_RUN" != 1 ] && ! command -v "$BIN" >/dev/null 2>&1 && [ ! -x "$BIN" ]; then
    die "strands-decider not found (set DECIDER_BIN or add it to PATH; see deploy/mac/00-preflight.sh)"
  fi
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then ok "$NAME already running (pid $(cat "$PIDFILE"))"; status; return; fi
  mkdirs; [ "$DRY_RUN" = 1 ] || mkdir -p "$STATE"
  export HF_HOME="$UNSPOKEN_HF_HOME" HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
  # shellcheck disable=SC2086
  run_sh "nohup $(printf '%q' "$BIN") serve --model $(printf '%q' "$MODEL") --host 127.0.0.1 --port $PORT ${DECIDER_EXTRA:-} > $(printf '%q' "$LOG") 2>&1 </dev/null & echo \$! > $(printf '%q' "$PIDFILE")"
  wait_http "$HEALTH" "${DECIDER_WAIT:-300}" "decider $SIZE" || { tail -40 "$LOG" >&2 || true; exit 1; }
  status
}

case "$ACTION" in
  status) status ;;
  stop) stop ;;
  start)
    log "decider $SIZE: $MODEL native -> 127.0.0.1:$PORT"
    start ;;
esac
