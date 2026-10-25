#!/usr/bin/env bash
# deploy/gb10/down.sh [--dry-run] [--all]
#
# Stop the servers up.sh started: unspoken-decider-12b, unspoken-decider-2b, unspoken-asr, unspoken-tts,
# unspoken-vllm (ours). Never touches the NemoClaw-managed `nemoclaw-vllm` unless --all
# (then: `nemoclaw` owns it; restart it with `nemoclaw` afterwards, not with vllm.sh).
# Native-runtime servers (pid files under $UNSPOKEN_STATE) are killed too.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_common_flags "$@"
ALL=0
for a in "${ARGS[@]+"${ARGS[@]}"}"; do case "$a" in --all) ALL=1 ;; *) die "unknown flag $a" ;; esac; done
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DR=(); [ "$DRY_RUN" = 1 ] && DR=(--dry-run)

"$D/decider.sh" 2b --stop "${DR[@]+"${DR[@]}"}" || true
"$D/decider.sh" 12b --stop "${DR[@]+"${DR[@]}"}" || true
"$D/speech.sh" all --stop "${DR[@]+"${DR[@]}"}" || true
"$D/vllm.sh" --stop "${DR[@]+"${DR[@]}"}" || true
# native-runtime leftovers
for pf in "$UNSPOKEN_STATE"/*.pid "$UNSPOKEN_STATE"/decider/*.pid; do
  [ -f "$pf" ] || continue
  run kill "$(cat "$pf")" || true; [ "$DRY_RUN" = 1 ] || rm -f "$pf"
done
if [ "$ALL" = 1 ] && docker_exists nemoclaw-vllm; then warn "--all: stopping the NemoClaw-managed vLLM"; run docker stop nemoclaw-vllm; fi
ok "down"
