#!/usr/bin/env bash
# make-ca.sh [--dry-run] [--force] [--ip <bridge-ip>] [--dir <cert-dir>] [--extra-ip <ip>]...
#
# Creates the private CA and the server certificate the TLS proxy presents to the NemoClaw sandbox:
#   <dir>/ca.key      CA private key (0600)              <dir>/ca.pem      CA certificate  -> NEMOCLAW_CORPORATE_CA_BUNDLE
#   <dir>/server.key  proxy private key (0600)           <dir>/server.crt  leaf cert with SAN IP:<bridge-ip>
# NemoClaw rules this satisfies (docs/NEMOCLAW.md §MCP): the MCP URL is an IPv4 literal, so the leaf needs a
# matching IP SAN; the chain must reach a CA the sandbox trusts, which NEMOCLAW_CORPORATE_CA_BUNDLE bakes into
# the image at onboarding (rebuild after changing it). The CA file must contain ONLY CA certs (basicConstraints
# CA:TRUE), be a regular non-symlink file and not group/world writable: make-ca.sh writes it that way.
# Idempotent: existing files are kept unless --force. The dir is created 0700 OUTSIDE the repo (default
# ~/.unspoken/tls). Nothing here is ever committed.
set -euo pipefail
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh" "$@"

FORCE=0; IP=""; DIR="$UNSPOKEN_TLS_DIR"; EXTRA_IPS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) ;;
    --force) FORCE=1 ;;
    --ip) IP="$2"; shift ;;
    --dir) DIR="$2"; shift ;;
    --extra-ip) EXTRA_IPS+=("$2"); shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

need openssl
[ -n "$IP" ] || IP="$(bridge_ip)" || die "cannot find the docker bridge IP; pass --ip <ip> or set UNSPOKEN_BRIDGE_IP"
case "$IP" in
  10.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*|192.168.*|127.*) ;;
  *) die "refusing to issue a cert for $IP: the proxy serves the private sandbox network only (RFC1918). Set UNSPOKEN_BRIDGE_IP to the openshell-docker gateway (docker network inspect openshell-docker)" ;;
esac

log "cert dir $DIR, server SAN IP:$IP${EXTRA_IPS[*]:+ (+ ${EXTRA_IPS[*]})}"
run mkdir -p "$DIR"
run chmod 700 "$DIR"

if [ "$FORCE" = "0" ] && [ -f "$DIR/server.crt" ] && [ -f "$DIR/ca.pem" ] && [ "$DRY_RUN" = "0" ]; then
  if openssl x509 -in "$DIR/server.crt" -noout -ext subjectAltName 2>/dev/null | grep -q "IP Address:$IP\b"; then
    log "certificates already exist for $IP (use --force to regenerate)"
    openssl x509 -in "$DIR/server.crt" -noout -subject -enddate >&2
    exit 0
  fi
  warn "existing server.crt does not carry IP:$IP; regenerating the leaf (CA kept)"
fi

SAN="IP:$IP"
for x in "${EXTRA_IPS[@]:-}"; do [ -n "$x" ] && SAN="$SAN,IP:$x"; done
DAYS_CA=825; DAYS_LEAF=398   # browsers' max leaf lifetime; the sandbox does not care, but keep it conventional

if [ "$FORCE" = "1" ] || [ ! -f "$DIR/ca.key" ] || [ ! -f "$DIR/ca.pem" ]; then
  run openssl ecparam -name prime256v1 -genkey -noout -out "$DIR/ca.key"
  run chmod 600 "$DIR/ca.key"
  run openssl req -x509 -new -key "$DIR/ca.key" -sha256 -days "$DAYS_CA" -subj "/CN=LocoCisco room CA/O=LocoCisco" \
      -addext "basicConstraints=critical,CA:TRUE,pathlen:0" -addext "keyUsage=critical,keyCertSign,cRLSign" \
      -addext "subjectKeyIdentifier=hash" -out "$DIR/ca.pem"
  run chmod 644 "$DIR/ca.pem"
fi

run openssl ecparam -name prime256v1 -genkey -noout -out "$DIR/server.key"
run chmod 600 "$DIR/server.key"
run openssl req -new -key "$DIR/server.key" -subj "/CN=unspoken-mcp/O=LocoCisco" -out "$DIR/server.csr"
EXT="$DIR/server.ext"
if [ "$DRY_RUN" = "1" ]; then
  printf '+ printf ... > %s   # subjectAltName=%s, extendedKeyUsage=serverAuth, basicConstraints=CA:FALSE\n' "$EXT" "$SAN"
else
  printf 'basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=%s\nauthorityKeyIdentifier=keyid,issuer\nsubjectKeyIdentifier=hash\n' "$SAN" >"$EXT"
fi
run openssl x509 -req -in "$DIR/server.csr" -CA "$DIR/ca.pem" -CAkey "$DIR/ca.key" -CAcreateserial -sha256 \
    -days "$DAYS_LEAF" -extfile "$EXT" -out "$DIR/server.crt"
run chmod 644 "$DIR/server.crt"
run rm -f "$DIR/server.csr" "$EXT"

if [ "$DRY_RUN" = "0" ]; then
  openssl verify -CAfile "$DIR/ca.pem" "$DIR/server.crt" >&2
  openssl x509 -in "$DIR/server.crt" -noout -subject -enddate -ext subjectAltName >&2
  # NemoClaw rejects CA files that are not plain regular files or are group/world writable
  [ -L "$DIR/ca.pem" ] && die "$DIR/ca.pem must not be a symlink"
fi
log "done. Export before onboarding:  export NEMOCLAW_CORPORATE_CA_BUNDLE=$DIR/ca.pem"
