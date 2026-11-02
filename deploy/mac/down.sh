#!/usr/bin/env bash
# deploy/mac/down.sh [--dry-run]
#
# Stop what up.sh started on the Mac: the native Strands Decider 2B (:8765). LM Studio is a GUI app that
# up.sh never launched, so down.sh never stops it -- quit it from the LM Studio window if you want it off.
# Native-runtime servers (pid files under $UNSPOKEN_STATE) are killed too.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_common_flags "$@"
for a in "${ARGS[@]+"${ARGS[@]}"}"; do case "$a" in *) die "unknown flag $a" ;; esac; done
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DR=(); [ "$DRY_RUN" = 1 ] && DR=(--dry-run)

"$D/decider.sh" 2b --stop "${DR[@]+"${DR[@]}"}" || true
# native-runtime leftovers
for pf in "$UNSPOKEN_STATE"/*.pid "$UNSPOKEN_STATE"/decider/*.pid; do
  [ -f "$pf" ] || continue
  run kill "$(cat "$pf")" || true; [ "$DRY_RUN" = 1 ] || rm -f "$pf"
done
ok "down (LM Studio, if running, was left alone — quit it from its window)"
