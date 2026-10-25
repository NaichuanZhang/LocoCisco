#!/usr/bin/env bash
# verify.sh [--dry-run] [--post] [--agent]
#
# Day-of step 3: prove the plug-in works, in the order that matters.
#   1. MCP server /health on loopback; TLS proxy /health through the CA; anonymous POST -> 401 (denial first)
#   2. a real MCP tool call from the host through the proxy, exactly like the sandbox: tools, room_status
#   3. nemoclaw <sb> mcp status unspoken --json  -> trustedPrivateTarget.state == match
#   4. Discord channel status; with --post: one test message through the configured notifier
#      (UNSPOKEN_NOTIFIER=openclaw|discord-rest|console; needs the channel ids in config/bridge.toml)
#   5. --agent: one OpenClaw turn that must call the room's tool ("reply with the meeting title")
# Exit non-zero at the first failed check. --dry-run prints every command.
set -euo pipefail
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh" "$@"

POST=0; AGENT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) ;;
    --post) POST=1 ;;
    --agent) AGENT=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

need curl
need_env UNSPOKEN_MCP_TOKEN
PY="$(py)"
BRIDGE_IP="$(bridge_ip)" || die "cannot find the docker bridge IP; set UNSPOKEN_BRIDGE_IP"
MCP_URL="https://$BRIDGE_IP:$UNSPOKEN_TLS_PORT$UNSPOKEN_MCP_PATH"
CA="$UNSPOKEN_TLS_DIR/ca.pem"
FAILED=0
check() {  # check <label> <cmd...> : run, report PASS/FAIL, keep going
  local label="$1"; shift
  if [ "$DRY_RUN" = "1" ]; then printf '+ %s\n' "$(quote_cmd "$@")"; return 0; fi
  if "$@" >"$UNSPOKEN_RUN_DIR/verify.out" 2>&1; then
    log "PASS  $label"
  else
    FAILED=1; warn "FAIL  $label"; sed 's/^/      /' "$UNSPOKEN_RUN_DIR/verify.out" | head -n 20 >&2
  fi
}
mkdir -p "$UNSPOKEN_RUN_DIR"

log "== 1. listeners"
check "MCP server /health on 127.0.0.1:$UNSPOKEN_MCP_PORT" curl -fsS --max-time 3 "http://127.0.0.1:$UNSPOKEN_MCP_PORT/health"
check "TLS proxy /health on $BRIDGE_IP:$UNSPOKEN_TLS_PORT (CA $CA)" curl -fsS --max-time 3 --cacert "$CA" "https://$BRIDGE_IP:$UNSPOKEN_TLS_PORT/health"
deny401() { [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 --cacert "$CA" -X POST "$MCP_URL" -H 'Content-Type: application/json' -d '{}')" = "401" ]; }
check "anonymous POST $MCP_URL is denied with 401" deny401
notpublic() { ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "^(0\.0\.0\.0|\*|\[::\]):($UNSPOKEN_TLS_PORT|$UNSPOKEN_MCP_PORT)$"; }
check "no listener on 0.0.0.0/:: for ports $UNSPOKEN_TLS_PORT,$UNSPOKEN_MCP_PORT" notpublic

log "== 2. MCP tool calls through the proxy (bearer + TLS, like the sandbox)"
check "tools/list via $MCP_URL" "$PY" -m unspoken.bridge.probe --url "$MCP_URL" --ca "$CA" tools
check "room_status via $MCP_URL" "$PY" -m unspoken.bridge.probe --url "$MCP_URL" --ca "$CA" room_status
check "list_items(kind=task)" "$PY" -m unspoken.bridge.probe --url "$MCP_URL" --ca "$CA" list_items kind=task

log "== 3. NemoClaw managed MCP registration"
if command -v nemoclaw >/dev/null 2>&1 || [ "$DRY_RUN" = "1" ]; then
  mcpmatch() {
    nemoclaw "$UNSPOKEN_SANDBOX" mcp status unspoken --json | "$PY" -c 'import json,sys; d=json.load(sys.stdin); s=(d.get("trustedPrivateTarget") or {}).get("state"); print(json.dumps(d, indent=1)[:1500]); sys.exit(0 if s=="match" else 1)'
  }
  check "nemoclaw $UNSPOKEN_SANDBOX mcp status unspoken: trustedPrivateTarget.state == match" mcpmatch
  check "nemoclaw $UNSPOKEN_SANDBOX skill list shows unspoken-room" bash -c "nemoclaw $UNSPOKEN_SANDBOX skill list | grep -q unspoken-room"
else
  warn "nemoclaw not on PATH: skipping sandbox checks (fine on the dev desktop)"
fi

log "== 4. Discord"
if command -v nemoclaw >/dev/null 2>&1 || [ "$DRY_RUN" = "1" ]; then
  check "nemoclaw $UNSPOKEN_SANDBOX channels status --channel discord" nemoclaw "$UNSPOKEN_SANDBOX" channels status --channel discord
fi
if [ "$POST" = "1" ]; then
  check "test post through UNSPOKEN_NOTIFIER=${UNSPOKEN_NOTIFIER:-<config>}" "$PY" -m unspoken.bridge.send --channel notes --health \
    --text "LocoCisco bridge check ✅ $(date -u +%H:%M:%SZ) — nothing but this text left the box."
fi

if [ "$AGENT" = "1" ]; then
  log "== 5. one agent turn that must use the room's tools"
  check "openclaw agent calls room_status" nemoclaw "$UNSPOKEN_SANDBOX" agent --agent main --timeout 120 \
    -m "Call the unspoken room_status tool and reply with only the meeting title it returns."
fi

if [ "$DRY_RUN" = "1" ]; then exit 0; fi
if [ "$FAILED" = "0" ]; then log "all checks passed"; else die "some checks failed (see above)"; fi
