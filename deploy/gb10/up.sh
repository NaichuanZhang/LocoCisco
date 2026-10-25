#!/usr/bin/env bash
# deploy/gb10/up.sh [--dry-run] [--no-vllm] [--no-tts] [--with-2b] [--no-check]
#
# Bring every model server up, in memory order, waiting for each health check:
#   1. vLLM Qwen3.6 on :8000   -- reuses the NemoClaw-managed `nemoclaw-vllm` if it is running, else vllm.sh
#   2. Strands Decider 12B on :8765 (GPU)            decider.sh 12b
#   3. ASR + diarization on :8770                    speech.sh asr
#   4. TTS on :8771 (skip with --no-tts)             speech.sh tts
#   5. Decider 2B on :8766 (only with --with-2b)     decider.sh 2b
#   6. check.sh (UNSPOKEN_PROFILE=gb10 python -m unspoken.models check)
# Logs: ~/unspoken-logs/*.log and `docker logs <unspoken-*>`. Everything binds 127.0.0.1.
# Typical cold start: vLLM 2-4 min (first start longer: kernel compile), decider 12B 1-3 min, speech < 30 s.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_common_flags "$@"
VLLM=1; TTS=1; TWO_B=0; CHECK=1
for a in "${ARGS[@]+"${ARGS[@]}"}"; do
  case "$a" in --no-vllm) VLLM=0 ;; --no-tts) TTS=0 ;; --with-2b) TWO_B=1 ;; --no-check) CHECK=0 ;; *) die "unknown flag $a" ;; esac
done
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DR=(); [ "$DRY_RUN" = 1 ] && DR=(--dry-run)
T0=$(date +%s)
step() { log "---- $* ----"; }

mkdirs
step "1/6 vLLM :$PORT_VLLM"
if [ "$VLLM" = 1 ]; then
  if [ "$DRY_RUN" != 1 ] && curl -fsS -m 3 "http://127.0.0.1:$PORT_VLLM/v1/models" >/dev/null 2>&1; then
    ok "vLLM already answering on :$PORT_VLLM (NemoClaw-managed or ours); leaving it"
  else
    "$D/vllm.sh" "${DR[@]+"${DR[@]}"}"
  fi
else warn "--no-vllm: skipping"; fi

step "2/6 decider 12B :$PORT_DECIDER_12B"
"$D/decider.sh" 12b "${DR[@]+"${DR[@]}"}"

step "3/6 ASR :$PORT_ASR"
"$D/speech.sh" asr "${DR[@]+"${DR[@]}"}"

step "4/6 TTS :$PORT_TTS"
if [ "$TTS" = 1 ]; then "$D/speech.sh" tts "${DR[@]+"${DR[@]}"}" || warn "TTS failed to start; it is optional, carrying on"; else warn "--no-tts: skipping"; fi

step "5/6 decider 2B :$PORT_DECIDER_2B"
if [ "$TWO_B" = 1 ]; then "$D/decider.sh" 2b "${DR[@]+"${DR[@]}"}"; else log "not requested (--with-2b)"; fi

step "6/6 check"
if [ "$CHECK" = 1 ]; then "$D/check.sh" "${DR[@]+"${DR[@]}"}" || warn "check reported problems (see above)"; fi

ok "up.sh done in $(( $(date +%s) - T0 ))s; status: deploy/gb10/status.sh ; logs: $UNSPOKEN_LOGS"
