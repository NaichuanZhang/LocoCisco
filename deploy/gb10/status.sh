#!/usr/bin/env bash
# deploy/gb10/status.sh [--dry-run] [--logs]
#
# One line per server: port, container/pid state, health, probe latency. --logs tails the last 15 lines of each.
# Exit code: number of servers that are down (TTS and decider 2B are optional and do not count).

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_common_flags "$@"
LOGS=0
for a in "${ARGS[@]+"${ARGS[@]}"}"; do case "$a" in --logs) LOGS=1 ;; *) die "unknown flag $a" ;; esac; done
DOWN=0

row() { # row NAME PORT URL OPTIONAL
  local name="$1" port="$2" url="$3" optional="${4:-0}" state="-" health="down" ms="-"
  if docker_running "$name"; then state="container"; elif docker_running nemoclaw-vllm && [ "$port" = "$PORT_VLLM" ]; then state="nemoclaw-vllm"
  elif [ -f "$UNSPOKEN_STATE/$name.pid" ] || [ -f "$UNSPOKEN_STATE/decider/$name.pid" ]; then state="pid"; fi
  if [ "$DRY_RUN" != 1 ]; then
    local t; t="$(curl -sS -m 3 -o /dev/null -w '%{http_code} %{time_total}' "$url" 2>/dev/null || echo "000 0")"
    if [ "${t%% *}" = 200 ]; then health="ok"; ms="$(awk -v s="${t##* }" 'BEGIN{printf "%d ms", s*1000}')"; fi
  fi
  printf '%-22s :%-5s %-14s %-5s %s\n' "$name" "$port" "$state" "$health" "$ms"
  if [ "$health" != ok ] && [ "$optional" != 1 ]; then DOWN=$((DOWN + 1)); fi
  if [ "$LOGS" = 1 ] && docker_running "$name"; then docker logs --tail 15 "$name" 2>&1 | sed 's/^/     | /'; fi
}

printf '%-22s %-6s %-14s %-5s %s\n' server port runtime health probe
row unspoken-vllm        "$PORT_VLLM"        "http://127.0.0.1:$PORT_VLLM/v1/models"
row unspoken-decider-12b "$PORT_DECIDER_12B" "http://127.0.0.1:$PORT_DECIDER_12B/health"
row unspoken-asr         "$PORT_ASR"         "http://127.0.0.1:$PORT_ASR/ready"
row unspoken-tts         "$PORT_TTS"         "http://127.0.0.1:$PORT_TTS/ready" 1
row unspoken-decider-2b  "$PORT_DECIDER_2B"  "http://127.0.0.1:$PORT_DECIDER_2B/health" 1
if have nvidia-smi; then
  nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,power.draw,clocks.sm --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /'
fi
if have free; then free -g | awk 'NR==2 {printf "ram: %d/%d GB used, %d GB available\n", $3, $2, $7}'; fi
[ "$DRY_RUN" = 1 ] && exit 0
exit "$DOWN"
