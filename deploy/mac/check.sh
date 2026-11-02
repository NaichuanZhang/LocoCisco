#!/usr/bin/env bash
# deploy/mac/check.sh [--dry-run] [--roles decider,llm,asr] [--no-sensors]
#
# The one check command of the day: raw HTTP probes of the two servers (LM Studio + decider), then
#   UNSPOKEN_PROFILE=mac python -m unspoken.models check     (every role; models workstream)
#   UNSPOKEN_PROFILE=mac python -m unspoken.sensors.check     (mic + camera seen, in-process roles load)
# from the repo venv ($UNSPOKEN_HOME/.venv). Exit 0 when all green. --no-sensors skips the second one.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_common_flags "$@"
ROLES=""; SENSORS=1
prev=""
for a in "${ARGS[@]+"${ARGS[@]}"}"; do
  if [ "$prev" = --roles ]; then ROLES="$a"; prev=""; continue; fi
  case "$a" in --roles) prev=--roles ;; --no-sensors) SENSORS=0 ;; *) die "unknown flag $a" ;; esac
done
RC=0
probe() { # probe LABEL URL
  if [ "$DRY_RUN" = 1 ]; then printf '+ curl -fsS -m 5 %s\n' "$2" >&2; return 0; fi
  local t; t="$(curl -sS -m 5 -o /dev/null -w '%{http_code} %{time_total}' "$2" 2>/dev/null || echo "000 0")"
  if [ "${t%% *}" = 200 ]; then ok "$(printf '%-12s %s (%d ms)' "$1" "$2" "$(awk -v s="${t##* }" 'BEGIN{print int(s*1000)}')")"
  else fail "$(printf '%-12s %s -> HTTP %s' "$1" "$2" "${t%% *}")"; RC=1; fi
}
log "== raw probes"
probe llm      "$LLM_URL/models"
probe decider  "http://127.0.0.1:$PORT_DECIDER_2B/health"

log "== decider: one typed question"
if [ "$DRY_RUN" != 1 ]; then
  curl -sS -m 30 "http://127.0.0.1:$PORT_DECIDER_2B/v1/systemone" -H 'content-type: application/json' -d '{
    "state": "Sam: can you send the deck by Friday? Ana: sure.",
    "questions": {"has_task": {"type": "noul", "instructions": "Does this contain a commitment to do something?"}}
  }' | head -c 300 >&2; echo >&2
else printf '+ curl -X POST http://127.0.0.1:%s/v1/systemone -d {state, questions}\n' "$PORT_DECIDER_2B" >&2; fi

log "== unspoken.models check (profile mac)"
PY="$UNSPOKEN_HOME/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || true)"
CMD=(env UNSPOKEN_PROFILE=mac "$PY" -m unspoken.models check)
[ -n "$ROLES" ] && CMD+=(--roles "$ROLES")
if [ "$DRY_RUN" = 1 ]; then printf '+ cd %q && ' "$UNSPOKEN_HOME" >&2; run "${CMD[@]}"
elif [ -d "$UNSPOKEN_HOME/unspoken/models" ]; then (cd "$UNSPOKEN_HOME" && run "${CMD[@]}") || RC=1
else warn "$UNSPOKEN_HOME/unspoken/models not present (models workstream not merged yet); raw probes above are the check"; fi

if [ "$SENSORS" = 1 ]; then
  log "== unspoken.sensors.check (mic, camera, vad/asr/diarization/faces/asd load)"
  SCMD=(env HF_HUB_OFFLINE=1 UNSPOKEN_PROFILE=mac "$PY" -m unspoken.sensors.check)
  if [ "$DRY_RUN" = 1 ]; then printf '+ cd %q && ' "$UNSPOKEN_HOME" >&2; run "${SCMD[@]}"
  elif [ -d "$UNSPOKEN_HOME/unspoken/sensors" ]; then (cd "$UNSPOKEN_HOME" && run "${SCMD[@]}") || RC=1
  else warn "$UNSPOKEN_HOME/unspoken/sensors not present"; fi
fi
exit $RC
