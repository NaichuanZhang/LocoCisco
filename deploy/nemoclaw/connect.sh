#!/usr/bin/env bash
# connect.sh [--dry-run] [--runtime fake|sim] [--firewall] [--fallback-rest] [--hooks] [--skip-persona]
#
# Day-of step 2: plug the running room into the NemoClaw sandbox.
#   a) MCP server on 127.0.0.1:$UNSPOKEN_MCP_PORT (started here with --runtime fake|sim if nothing answers; in the
#      real run the meeting process serves it itself and this step only checks /health)
#   b) TLS proxy on <bridge-ip>:8443 -> 127.0.0.1:$UNSPOKEN_MCP_PORT, cert from make-ca.sh
#   c) host firewall: :8443 only from the sandbox subnet (printed; applied with --firewall, needs sudo)
#   d) nemoclaw <sb> mcp add unspoken --url https://<bridge-ip>:8443/mcp --env UNSPOKEN_MCP_TOKEN --trusted-private-host <bridge-ip>
#      (--fallback-rest instead: custom policy preset + agent-native mcp.servers entry)
#   e) nemoclaw <sb> policy add discord --yes   (idempotent)
#   f) persona: upload SOUL/AGENTS/TOOLS/IDENTITY/USER.md to /sandbox/.openclaw/workspace, skill install unspoken-room
#   g) --hooks: enable the OpenClaw gateway /hooks routes with OPENCLAW_HOOKS_TOKEN (only for notifier mode = "hooks")
# Idempotent; --dry-run prints every command.
# Needs: UNSPOKEN_MCP_TOKEN (make one: export UNSPOKEN_MCP_TOKEN=$(openssl rand -hex 32)); --hooks needs OPENCLAW_HOOKS_TOKEN.
# Sources: https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/manage-sandboxes/mcp-servers/add-an-mcp-server.md
#          https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/network-policy/configure-policies/create-custom-policy-presets.md
#          https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/reference/troubleshooting.md#agent-cannot-reach-a-host-side-http-service
set -euo pipefail
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh" "$@"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNTIME="${UNSPOKEN_MCP_RUNTIME:-fake}"; FIREWALL=0; FALLBACK=0; HOOKS=0; PERSONA=1
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) ;;
    --runtime) RUNTIME="$2"; shift ;;
    --firewall) FIREWALL=1 ;;
    --fallback-rest) FALLBACK=1 ;;
    --hooks) HOOKS=1 ;;
    --skip-persona) PERSONA=0 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

need nemoclaw docker curl openssl
need_env UNSPOKEN_MCP_TOKEN
[ "$HOOKS" = "0" ] || need_env OPENCLAW_HOOKS_TOKEN
PY="$(py)"
BRIDGE_IP="$(bridge_ip)" || die "cannot find the docker bridge IP; set UNSPOKEN_BRIDGE_IP"
SUBNET="$(bridge_subnet)"
MCP_URL="https://$BRIDGE_IP:$UNSPOKEN_TLS_PORT$UNSPOKEN_MCP_PATH"
CA="$UNSPOKEN_TLS_DIR/ca.pem"
log "bridge ip $BRIDGE_IP (subnet $SUBNET); MCP url for the sandbox: $MCP_URL"

log "== a) MCP server on 127.0.0.1:$UNSPOKEN_MCP_PORT"
if curl -fsS --max-time 3 "http://127.0.0.1:$UNSPOKEN_MCP_PORT/health" >/dev/null 2>&1; then
  log "MCP server already answering on 127.0.0.1:$UNSPOKEN_MCP_PORT"
else
  start_bg unspoken-mcp "$PY" -m unspoken.bridge.mcp_server --runtime "$RUNTIME" --host 127.0.0.1 --port "$UNSPOKEN_MCP_PORT"
  if [ "$DRY_RUN" = "0" ]; then
    for _ in $(seq 1 50); do curl -fsS --max-time 1 "http://127.0.0.1:$UNSPOKEN_MCP_PORT/health" >/dev/null 2>&1 && break; sleep 0.2; done
    curl -fsS --max-time 3 "http://127.0.0.1:$UNSPOKEN_MCP_PORT/health" >/dev/null || die "MCP server did not come up: $UNSPOKEN_RUN_DIR/unspoken-mcp.log"
  fi
fi

log "== b) TLS proxy on $BRIDGE_IP:$UNSPOKEN_TLS_PORT -> http://127.0.0.1:$UNSPOKEN_MCP_PORT"
"$HERE/make-ca.sh" --ip "$BRIDGE_IP" --dir "$UNSPOKEN_TLS_DIR"
start_bg unspoken-tls "$PY" -m unspoken.bridge.tls_proxy --bind-ip "$BRIDGE_IP" --port "$UNSPOKEN_TLS_PORT" \
  --upstream "http://127.0.0.1:$UNSPOKEN_MCP_PORT" --mcp-path "$UNSPOKEN_MCP_PATH" \
  --cert "$UNSPOKEN_TLS_DIR/server.crt" --key "$UNSPOKEN_TLS_DIR/server.key"
if [ "$DRY_RUN" = "0" ]; then
  for _ in $(seq 1 50); do curl -fsS --max-time 1 --cacert "$CA" "https://$BRIDGE_IP:$UNSPOKEN_TLS_PORT/health" >/dev/null 2>&1 && break; sleep 0.2; done
  curl -fsS --max-time 3 --cacert "$CA" "https://$BRIDGE_IP:$UNSPOKEN_TLS_PORT/health" >/dev/null || die "TLS proxy did not come up: $UNSPOKEN_RUN_DIR/unspoken-tls.log"
  # verify denial BEFORE anything else depends on it: no bearer -> 401, wrong bearer -> 401
  code="$(curl -s -o /dev/null -w '%{http_code}' --cacert "$CA" -X POST "$MCP_URL" -H 'Content-Type: application/json' -d '{}')"
  [ "$code" = "401" ] || die "proxy answered $code without a bearer; expected 401. Stop."
  log "proxy up; anonymous POST $MCP_URL -> 401 (good)"
fi

log "== c) firewall: allow $UNSPOKEN_TLS_PORT/tcp from $SUBNET to $BRIDGE_IP only"
FW_UFW=(sudo ufw allow proto tcp from "$SUBNET" to "$BRIDGE_IP" port "$UNSPOKEN_TLS_PORT" comment unspoken-mcp-proxy)
FW_UFW_DENY=(sudo ufw deny "$UNSPOKEN_TLS_PORT/tcp" comment unspoken-mcp-proxy-deny-rest)
if [ "$FIREWALL" = "1" ]; then
  need sudo ufw
  run "${FW_UFW[@]}"
  run "${FW_UFW_DENY[@]}"
  run sudo ufw status numbered
else
  log "not applied (pass --firewall). Commands:"
  printf '  %s\n  %s\n' "$(quote_cmd "${FW_UFW[@]}")" "$(quote_cmd "${FW_UFW_DENY[@]}")" >&2
  log "nftables alternative: sudo nft add rule inet filter input ip saddr $SUBNET tcp dport $UNSPOKEN_TLS_PORT accept; sudo nft add rule inet filter input tcp dport $UNSPOKEN_TLS_PORT drop"
fi

if [ "$FALLBACK" = "0" ]; then
  log "== d) managed MCP registration (nemoclaw mcp add; the bearer is read from UNSPOKEN_MCP_TOKEN on this host)"
  if [ "$DRY_RUN" = "0" ] && nemoclaw "$UNSPOKEN_SANDBOX" mcp list 2>/dev/null | grep -qw unspoken; then
    log "mcp server 'unspoken' already registered; refreshing status"
  else
    run nemoclaw "$UNSPOKEN_SANDBOX" mcp add unspoken --url "$MCP_URL" --env UNSPOKEN_MCP_TOKEN --trusted-private-host "$BRIDGE_IP"
  fi
  run nemoclaw "$UNSPOKEN_SANDBOX" mcp status unspoken
  log "expect: trustedPrivateTarget.state = match (nemoclaw $UNSPOKEN_SANDBOX mcp status unspoken --json)"
else
  log "== d) FALLBACK A: custom policy preset + agent-native mcp.servers entry (token becomes readable in the sandbox)"
  PRESET="$UNSPOKEN_RUN_DIR/unspoken-mcp-host.yaml"
  run mkdir -p "$UNSPOKEN_RUN_DIR"
  if [ "$DRY_RUN" = "1" ]; then
    printf '+ sed s/BRIDGE_IP_PLACEHOLDER/%s/ %s > %s\n' "$BRIDGE_IP" "$UNSPOKEN_ROOT/nemoclaw/policy/unspoken-mcp-host.yaml" "$PRESET"
  else
    sed "s/BRIDGE_IP_PLACEHOLDER/$BRIDGE_IP/" "$UNSPOKEN_ROOT/nemoclaw/policy/unspoken-mcp-host.yaml" >"$PRESET"
  fi
  run nemoclaw "$UNSPOKEN_SANDBOX" exec -- which node
  run nemoclaw "$UNSPOKEN_SANDBOX" policy add --from-file "$PRESET" --trusted-private-host "$BRIDGE_IP" --dry-run
  run nemoclaw "$UNSPOKEN_SANDBOX" policy add --from-file "$PRESET" --trusted-private-host "$BRIDGE_IP" --yes
  run nemoclaw "$UNSPOKEN_SANDBOX" exec -- openclaw config set mcp.servers.unspoken.url "$MCP_URL"
  run nemoclaw "$UNSPOKEN_SANDBOX" exec -- openclaw config set mcp.servers.unspoken.headers.Authorization "Bearer <UNSPOKEN_MCP_TOKEN>"
  warn "replace <UNSPOKEN_MCP_TOKEN> by hand inside the sandbox (openclaw config set ...) so the value never appears in host logs"
  run nemoclaw "$UNSPOKEN_SANDBOX" gateway restart
fi

log "== e) Discord egress preset (idempotent; the channel was enabled at onboarding)"
run nemoclaw "$UNSPOKEN_SANDBOX" policy add discord --yes
run nemoclaw "$UNSPOKEN_SANDBOX" policy list

if [ "$PERSONA" = "1" ]; then
  log "== f) persona + skill"
  for f in SOUL.md AGENTS.md TOOLS.md IDENTITY.md USER.md; do
    run nemoclaw "$UNSPOKEN_SANDBOX" upload "$UNSPOKEN_ROOT/nemoclaw/workspace/$f" "/sandbox/.openclaw/workspace/$f"
  done
  run nemoclaw "$UNSPOKEN_SANDBOX" skill install "$UNSPOKEN_ROOT/nemoclaw/skills/unspoken-room/"
  run nemoclaw "$UNSPOKEN_SANDBOX" skill list
fi

if [ "$HOOKS" = "1" ]; then
  log "== g) OpenClaw gateway hooks (for config/bridge.toml [notifier.openclaw] mode = \"hooks\")"
  PATCH="$UNSPOKEN_RUN_DIR/hooks.patch.json5"
  run mkdir -p "$UNSPOKEN_RUN_DIR"
  if [ "$DRY_RUN" = "1" ]; then
    # shellcheck disable=SC2016  # the dry-run line shows the variable NAME on purpose (never the secret)
    printf '+ umask 077; printf {hooks:{enabled:true,token:"$OPENCLAW_HOOKS_TOKEN",path:"/hooks",allowedAgentIds:["main"],allowRequestSessionKey:false}} > %s\n' "$PATCH"
  else
    ( umask 077; printf '{ hooks: { enabled: true, token: "%s", path: "/hooks", allowedAgentIds: ["main"], allowRequestSessionKey: false } }\n' "$OPENCLAW_HOOKS_TOKEN" >"$PATCH" )
  fi
  run nemoclaw "$UNSPOKEN_SANDBOX" upload "$PATCH" /sandbox/hooks.patch.json5
  run nemoclaw "$UNSPOKEN_SANDBOX" exec -- openclaw config patch --file /sandbox/hooks.patch.json5
  run nemoclaw "$UNSPOKEN_SANDBOX" exec -- rm -f /sandbox/hooks.patch.json5
  [ "$DRY_RUN" = "1" ] || rm -f "$PATCH"
  run nemoclaw "$UNSPOKEN_SANDBOX" gateway restart
  log "smoke: curl -i http://127.0.0.1:18789/hooks/agent -H 'Authorization: Bearer \$OPENCLAW_HOOKS_TOKEN' -H 'Content-Type: application/json' -d '{\"message\":\"ping\",\"deliver\":false,\"agentId\":\"main\"}'"
fi

log "connected. next: deploy/nemoclaw/verify.sh"
