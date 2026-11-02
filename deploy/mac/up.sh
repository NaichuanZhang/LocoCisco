#!/usr/bin/env bash
# deploy/mac/up.sh [--dry-run] [--no-check]
#
# Bring the Mac Studio target up, in order, waiting for each health check:
#   1. 00-preflight.sh         (macOS + Apple Silicon + ports + python; read-only)
#   2. LM Studio on :1234      -- must already be reachable; LM Studio is a GUI app, this never launches it.
#                                 If it is down: open LM Studio, load qwen3-8b, start its Local Server.
#   3. Strands Decider 2B on :8765 (native)          decider.sh 2b
#   4. check.sh (UNSPOKEN_PROFILE=mac python -m unspoken.models check + sensors check)
# Logs: ~/unspoken-logs/*.log. Everything binds 127.0.0.1 / localhost. No Docker, no vLLM, no GPU.
# Typical cold start: decider 2B ~30-90 s; sensing models load lazily on first use.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_common_flags "$@"
CHECK=1
for a in "${ARGS[@]+"${ARGS[@]}"}"; do
  case "$a" in --no-check) CHECK=0 ;; *) die "unknown flag $a" ;; esac
done
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DR=(); [ "$DRY_RUN" = 1 ] && DR=(--dry-run)
T0=$(date +%s)
step() { log "---- $* ----"; }

mkdirs
step "1/4 preflight"
"$D/00-preflight.sh" "${DR[@]+"${DR[@]}"}" || warn "preflight reported problems (see above)"

step "2/4 LM Studio :1234"
if [ "$DRY_RUN" = 1 ]; then
  printf '+ curl -fsS -m 5 %s/models\n' "$LLM_URL" >&2
elif curl -fsS -m 5 "$LLM_URL/models" >/dev/null 2>&1; then
  ok "LM Studio answering at $LLM_URL (leaving it; expected model: $LLM_MODEL)"
else
  die "LM Studio not reachable at $LLM_URL. It is a GUI app: open LM Studio, load $LLM_MODEL, and start its Local Server (Developer tab -> Start Server, port 1234). Then re-run up.sh."
fi

step "3/4 decider 2B :$PORT_DECIDER_2B"
"$D/decider.sh" 2b "${DR[@]+"${DR[@]}"}"

step "4/4 check"
if [ "$CHECK" = 1 ]; then "$D/check.sh" "${DR[@]+"${DR[@]}"}" || warn "check reported problems (see above)"; else warn "--no-check: skipping"; fi

ok "up.sh done in $(( $(date +%s) - T0 ))s; logs: $UNSPOKEN_LOGS ; stop with deploy/mac/down.sh"
