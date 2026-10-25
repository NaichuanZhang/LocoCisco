#!/usr/bin/env bash
# Shared helpers for deploy/nemoclaw/*.sh. Source it; do not run it.
#   run <cmd...>      run a command, or print it when DRY_RUN=1 (every state-changing command goes through run)
#   need <bin>...     fail unless each binary is on PATH (in dry-run: warn)
#   need_env <VAR>... fail unless each variable is set and not a placeholder (in dry-run: warn)
#   bridge_ip         private IPv4 gateway of the openshell-docker bridge (what the sandbox can route to)
#   log/warn/die      messages on stderr
set -o pipefail

DRY_RUN="${DRY_RUN:-0}"
for _arg in "$@"; do
  case "$_arg" in --dry-run) DRY_RUN=1 ;; esac
done
export DRY_RUN

# Paths. Scripts live in <repo>/deploy/nemoclaw; config and nemoclaw/ assets are relative to the repo root.
UNSPOKEN_ROOT="${UNSPOKEN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export UNSPOKEN_ROOT
UNSPOKEN_SANDBOX="${UNSPOKEN_SANDBOX:-unspoken}"           # NemoClaw sandbox name (matches config/bridge.toml)
UNSPOKEN_TLS_DIR="${UNSPOKEN_TLS_DIR:-$HOME/.unspoken/tls}" # key material, 0700, outside the repo
UNSPOKEN_TLS_PORT="${UNSPOKEN_TLS_PORT:-8443}"
UNSPOKEN_MCP_PORT="${UNSPOKEN_MCP_PORT:-8791}"
UNSPOKEN_MCP_PATH="${UNSPOKEN_MCP_PATH:-/mcp}"
UNSPOKEN_RUN_DIR="${UNSPOKEN_RUN_DIR:-$HOME/.unspoken/run}" # pid files + logs for the MCP server and proxy
export UNSPOKEN_SANDBOX UNSPOKEN_TLS_DIR UNSPOKEN_TLS_PORT UNSPOKEN_MCP_PORT UNSPOKEN_MCP_PATH UNSPOKEN_RUN_DIR

log()  { printf '\033[1;34m[unspoken]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m[unspoken] WARN\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[unspoken] ERROR\033[0m %s\n' "$*" >&2; exit 1; }

# Print a command with shell quoting so a dry-run line can be pasted back into a shell.
quote_cmd() {
  local out="" a
  for a in "$@"; do out+=" $(printf '%q' "$a")"; done
  printf '%s' "${out# }"
}

run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '+ %s\n' "$(quote_cmd "$@")"
    return 0
  fi
  log "+ $(quote_cmd "$@")"
  "$@"
}

need() {
  local b
  for b in "$@"; do
    if ! command -v "$b" >/dev/null 2>&1; then
      if [ "$DRY_RUN" = "1" ]; then warn "$b is not on PATH (dry-run continues)"; else die "$b is not on PATH"; fi
    fi
  done
}

is_placeholder() {
  case "$1" in ""|\<*\>|000000000000000000|replace-me|changeme|CHANGE_ME) return 0 ;; esac
  return 1
}

need_env() {
  local v
  for v in "$@"; do
    if is_placeholder "${!v:-}"; then
      if [ "$DRY_RUN" = "1" ]; then warn "$v is not set (dry-run continues)"; else die "$v is not set or still a placeholder"; fi
    fi
  done
}

# The sandbox reaches host services only on an address the OpenShell gateway can route to. For the default
# Docker-driver topology that is the private IPv4 gateway of the `openshell-docker` bridge (the same address
# NemoClaw publishes managed vLLM on). UNSPOKEN_BRIDGE_IP overrides; the docker0 gateway is the last resort.
bridge_ip() {
  if [ -n "${UNSPOKEN_BRIDGE_IP:-}" ]; then printf '%s\n' "$UNSPOKEN_BRIDGE_IP"; return 0; fi
  local ip=""
  if command -v docker >/dev/null 2>&1; then
    ip="$(docker network inspect openshell-docker -f '{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null || true)"
    [ -n "$ip" ] || ip="$(docker network inspect bridge -f '{{(index .IPAM.Config 0).Gateway}}' 2>/dev/null || true)"
  fi
  [ -n "$ip" ] || { command -v ip >/dev/null 2>&1 && ip="$(ip -4 -o addr show dev docker0 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n1)"; }
  if [ -z "$ip" ]; then
    if [ "$DRY_RUN" = "1" ]; then printf '%s\n' "172.17.0.1"; warn "no docker bridge found; dry-run assumes 172.17.0.1"; return 0; fi
    return 1
  fi
  printf '%s\n' "$ip"
}

# Subnet of the sandbox network, for the firewall rule. UNSPOKEN_BRIDGE_SUBNET overrides; otherwise the
# openshell-docker (then default bridge) network, else a /16 around the bridge IP.
bridge_subnet() {
  if [ -n "${UNSPOKEN_BRIDGE_SUBNET:-}" ]; then printf '%s\n' "$UNSPOKEN_BRIDGE_SUBNET"; return 0; fi
  local s="" n
  if command -v docker >/dev/null 2>&1 && [ -z "${UNSPOKEN_BRIDGE_IP:-}" ]; then
    for n in openshell-docker bridge; do
      s="$(docker network inspect "$n" -f '{{(index .IPAM.Config 0).Subnet}}' 2>/dev/null | tr -d '[:space:]' || true)"
      case "$s" in *.*.*.*/*) break ;; *) s="" ;; esac
    done
  fi
  if [ -z "$s" ]; then
    local ip; ip="$(bridge_ip 2>/dev/null || printf '172.17.0.1')"
    s="$(printf '%s' "$ip" | awk -F. '{printf "%s.%s.0.0/16", $1, $2}')"
  fi
  printf '%s\n' "$s"
}

# The venv python of this worktree (scripts/dev-venv.sh) or whatever python3 is on PATH.
py() {
  if [ -x "$UNSPOKEN_ROOT/.venv/bin/python" ]; then printf '%s\n' "$UNSPOKEN_ROOT/.venv/bin/python"; else printf '%s\n' "python3"; fi
}

# A pidfile-based background service: start_bg <name> <cmd...>; stop_bg <name>; alive_bg <name>
start_bg() {
  local name="$1"; shift
  mkdir -p "$UNSPOKEN_RUN_DIR"
  if alive_bg "$name"; then log "$name already running (pid $(cat "$UNSPOKEN_RUN_DIR/$name.pid"))"; return 0; fi
  if [ "$DRY_RUN" = "1" ]; then printf '+ nohup %s >%s 2>&1 &  # pidfile %s\n' "$(quote_cmd "$@")" "$UNSPOKEN_RUN_DIR/$name.log" "$UNSPOKEN_RUN_DIR/$name.pid"; return 0; fi
  log "+ $(quote_cmd "$@") &"
  nohup "$@" >"$UNSPOKEN_RUN_DIR/$name.log" 2>&1 </dev/null &
  echo $! >"$UNSPOKEN_RUN_DIR/$name.pid"
}

alive_bg() {
  local f="$UNSPOKEN_RUN_DIR/$1.pid"
  [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null
}

stop_bg() {
  local f="$UNSPOKEN_RUN_DIR/$1.pid"
  if [ -f "$f" ]; then run kill "$(cat "$f")" || true; [ "$DRY_RUN" = "1" ] || rm -f "$f"; fi
}
