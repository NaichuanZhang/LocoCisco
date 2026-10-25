#!/usr/bin/env bash
# onboard.sh [--dry-run] [--profile <nemoclaw-profile-id>] [--model <slug>] [--tier restricted|balanced|open]
#
# Day-of step 1 on the GB10: create the NemoClaw sandbox "unspoken" with Discord enabled and inference on the
# NemoClaw-managed local vLLM (Qwen3.6-35B-A3B-NVFP4 on the GB10 profile). Idempotent: rerunning re-onboards the
# same sandbox name (NemoClaw reconciles). --dry-run prints every command and runs nothing.
#
# Needs in the environment (never written to disk by us):
#   DISCORD_BOT_TOKEN   bot token from the Discord Developer Portal (Bot -> Reset Token)
#   DISCORD_SERVER_ID   the server (guild) id (Developer Mode -> right-click server -> Copy Server ID)
# Optional: DISCORD_REQUIRE_MENTION (default 1), DISCORD_USER_ID (restrict to one user), NEMOCLAW_DISABLE_TELEMETRY=1,
#           UNSPOKEN_SANDBOX (default unspoken), UNSPOKEN_TLS_DIR (default ~/.unspoken/tls), HF_TOKEN (gated models only)
# Sources: https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/manage-sandboxes/messaging-channels/set-up-discord.md
#          https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/inference/local-inference/set-up-vllm.md
#          https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/security/configure-corporate-ca-trust.md
#          https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/reference/commands.md (onboard, env vars)
set -euo pipefail
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh" "$@"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROFILE="${UNSPOKEN_NEMOCLAW_PROFILE:-}"; MODEL="${UNSPOKEN_VLLM_MODEL:-qwen3.6-35b-a3b-nvfp4}"; TIER="${NEMOCLAW_POLICY_TIER:-restricted}"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) ;;
    --profile) PROFILE="$2"; shift ;;
    --model) MODEL="$2"; shift ;;
    --tier) TIER="$2"; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done
case "$TIER" in restricted|balanced|open) ;; *) die "--tier must be restricted, balanced or open (not personal: it opens the internet)";; esac

log "== 1/5 host checks"
need nemoclaw docker openssl
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >&2 || warn "nvidia-smi failed"
else
  warn "nvidia-smi not found: managed vLLM needs the GPU; on this dev desktop only --dry-run makes sense"
fi
run nemoclaw host probe

log "== 2/5 Discord environment (token + server id must be exported; mention-only by default)"
need_env DISCORD_BOT_TOKEN DISCORD_SERVER_ID
export DISCORD_REQUIRE_MENTION="${DISCORD_REQUIRE_MENTION:-1}"
log "DISCORD_SERVER_ID=${DISCORD_SERVER_ID:-<unset>} DISCORD_REQUIRE_MENTION=$DISCORD_REQUIRE_MENTION DISCORD_USER_ID=${DISCORD_USER_ID:-<any member>}"
log "Developer Portal checklist: Message Content Intent ON; invite URL scopes bot + applications.commands;"
log "  permissions View Channels, Send Messages, Send Messages in Threads, Read Message History, Embed Links, Add Reactions"

log "== 3/5 private CA for the MCP TLS proxy (baked into the sandbox image at onboarding)"
BRIDGE_IP="$(bridge_ip)" || die "cannot find the docker bridge IP (is Docker running?); set UNSPOKEN_BRIDGE_IP"
"$HERE/make-ca.sh" --ip "$BRIDGE_IP" --dir "$UNSPOKEN_TLS_DIR"       # honours DRY_RUN; idempotent
export NEMOCLAW_CORPORATE_CA_BUNDLE="$UNSPOKEN_TLS_DIR/ca.pem"
log "NEMOCLAW_CORPORATE_CA_BUNDLE=$NEMOCLAW_CORPORATE_CA_BUNDLE (CA-only PEM, regular file, not group/world writable)"
[ "$DRY_RUN" = "1" ] || [ -f "$NEMOCLAW_CORPORATE_CA_BUNDLE" ] || die "CA bundle missing"

log "== 4/5 NemoClaw onboarding: agent openclaw, sandbox $UNSPOKEN_SANDBOX, tier $TIER, managed local vLLM"
export NEMOCLAW_AGENT=openclaw NEMOCLAW_NON_INTERACTIVE=1 NEMOCLAW_SANDBOX_NAME="$UNSPOKEN_SANDBOX" NEMOCLAW_POLICY_TIER="$TIER"
export NEMOCLAW_MINIMAL_BOOTSTRAP=1          # we upload our own SOUL/AGENTS/TOOLS in connect.sh
ONBOARD=(nemoclaw onboard --name "$UNSPOKEN_SANDBOX" --non-interactive --yes --yes-i-accept-third-party-software)
if [ -n "$PROFILE" ]; then
  # --profile is mutually exclusive with NEMOCLAW_PROVIDER / NEMOCLAW_VLLM_MODEL
  unset NEMOCLAW_PROVIDER NEMOCLAW_VLLM_MODEL NEMOCLAW_MODEL
  log "serving profile: $PROFILE (from: nemoclaw profiles list --json)"
  run nemoclaw profiles list
  ONBOARD+=(--profile "$PROFILE")
else
  export NEMOCLAW_PROVIDER=install-vllm NEMOCLAW_VLLM_MODEL="$MODEL"
  log "NEMOCLAW_PROVIDER=install-vllm NEMOCLAW_VLLM_MODEL=$MODEL (GB10 = DGX Spark/N1x host profile; on N1x this is the Deferred preview)"
fi
log "env: NEMOCLAW_AGENT=openclaw NEMOCLAW_NON_INTERACTIVE=1 NEMOCLAW_SANDBOX_NAME=$UNSPOKEN_SANDBOX NEMOCLAW_POLICY_TIER=$TIER NEMOCLAW_MINIMAL_BOOTSTRAP=1 NEMOCLAW_CORPORATE_CA_BUNDLE=... DISCORD_*=..."
run "${ONBOARD[@]}"

log "== 5/5 post-onboarding checks"
run nemoclaw "$UNSPOKEN_SANDBOX" status
run nemoclaw "$UNSPOKEN_SANDBOX" channels status
run nemoclaw "$UNSPOKEN_SANDBOX" policy list
run nemoclaw inference get
log "next: deploy/nemoclaw/connect.sh  (MCP server + TLS proxy + mcp add + persona), then deploy/nemoclaw/verify.sh"
