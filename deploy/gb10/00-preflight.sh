#!/usr/bin/env bash
# deploy/gb10/00-preflight.sh [--dry-run] [--verify] [--cap-clocks] [--reset-clocks]
#
# First thing on the box. Read-only unless you pass a --cap/--reset flag. Prints one line per check:
#   arch + OS, NVIDIA driver/GPU/memory, free -g, df -h (home + USB), Docker + NVIDIA runtime, who holds our
#   ports, USB contents vs MANIFEST.sha256 (--verify runs the full sha256 check, ~10 min for 100 GB), HF cache,
#   GPU clocks / temperature / power, sensors (libportaudio2 + alsa-utils, capture devices, cameras, CAM++ model,
#   faster-whisper caches).
# --cap-clocks   caps the SM clock at 85% of max with `sudo nvidia-smi -lgc` (reported to help the DGX Spark
#                power-off-under-long-prefill issue, NVIDIA forum 385991). Only if the box reboots under load.
# --reset-clocks undoes it (`sudo nvidia-smi -rgc`).
# Exit code: number of FAIL lines (0 = go).

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_common_flags "$@"
VERIFY=0; CAP=0; RESET=0
for a in "${ARGS[@]+"${ARGS[@]}"}"; do
  case "$a" in --verify) VERIFY=1 ;; --cap-clocks) CAP=1 ;; --reset-clocks) RESET=1 ;; *) die "unknown flag $a" ;; esac
done
FAILS=0
f() { fail "$*"; FAILS=$((FAILS + 1)); }

find_usb() {
  if [ -n "${UNSPOKEN_USB:-}" ]; then echo "$UNSPOKEN_USB"; return; fi
  local d
  for d in /media/*/*/unspoken-usb /media/*/unspoken-usb /mnt/*/unspoken-usb /Volumes/*/unspoken-usb "$HOME/unspoken-usb"; do
    [ -d "$d" ] && { echo "$d"; return; }
  done
  echo ""
}

log "== 1. machine"
ARCH="$(uname -m)"
if [ "$ARCH" = aarch64 ]; then ok "arch aarch64 ($(uname -r))"; else f "arch is $ARCH, expected aarch64 (GB10). Dry-run on a dev box is fine."; fi
if [ -r /etc/os-release ]; then . /etc/os-release; ok "OS ${PRETTY_NAME:-?}"; fi
[ -r /etc/dgx-release ] && ok "DGX OS: $(tr '\n' ' ' < /etc/dgx-release | cut -c1-120)"

log "== 2. GPU"
if have nvidia-smi; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.max.sm --format=csv,noheader 2>/dev/null \
    | sed 's/^/     /' >&2 || f "nvidia-smi query failed"
  ok "nvidia-smi present"
  CUDA_V="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | head -1)"; [ -n "$CUDA_V" ] && ok "$CUDA_V (need 13.x for the pinned images)"
else
  f "nvidia-smi missing"
fi

log "== 3. memory and disk"
if have free; then free -g | sed 's/^/     /' >&2; fi
MEM_GB="$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)"
if [ "$MEM_GB" -ge 100 ]; then ok "RAM ${MEM_GB} GB (budget in docs/HACKDAY.md assumes 128 GB unified)"; else warn "RAM ${MEM_GB} GB: below the 128 GB budget; drop the decider 12B to 2B"; fi
{ df -h "$HOME" 2>/dev/null | sed 's/^/     /' >&2; } || true
HOME_FREE_GB="$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc 0-9)"
if [ "${HOME_FREE_GB:-0}" -ge 150 ]; then ok "home has ${HOME_FREE_GB} GB free (restore needs ~120 GB)"; else f "home has only ${HOME_FREE_GB:-?} GB free; restore needs ~120 GB"; fi

log "== 4. docker"
if have docker; then
  if docker info >/dev/null 2>&1; then
    ok "docker daemon reachable ($(docker --version))"
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then ok "nvidia container runtime registered"; else f "no nvidia runtime in docker (nvidia-container-toolkit)"; fi
    docker ps --format '     {{.Names}}  {{.Image}}  {{.Status}}' >&2 || true
    docker_running nemoclaw-vllm && ok "nemoclaw-vllm is running (NemoClaw owns :8000)"
    for img in "$IMAGE_VLLM" "$IMAGE_SPEECH"; do
      if docker image inspect "$img" >/dev/null 2>&1; then ok "image present: $img"; else warn "image not loaded yet: $img (restore.sh loads it)"; fi
    done
  else
    f "docker daemon not reachable (is your user in the docker group? newgrp docker)"
  fi
else
  f "docker missing"
fi

log "== 5. ports (127.0.0.1)"
for p in $PORT_VLLM $PORT_DECIDER_12B $PORT_DECIDER_2B $PORT_ASR $PORT_TTS 8780 8781 8791 8792 18789; do
  if port_open "$p"; then
    who="$(ss -ltnp 2>/dev/null | awk -v p=":$p" '$4 ~ p"$" {print $NF}' | head -1)"
    warn "port $p in use ${who:+by $who}"
  else ok "port $p free"; fi
done

log "== 6. USB"
USB="$(find_usb)"
if [ -n "$USB" ] && [ -d "$USB" ]; then
  ok "USB at $USB"
  df -h "$USB" 2>/dev/null | tail -1 | sed 's/^/     /' >&2
  if [ -f "$USB/MANIFEST.sha256" ]; then
    ok "MANIFEST.sha256: $(wc -l < "$USB/MANIFEST.sha256") items"
    missing=0
    while read -r _ path; do [ -n "$path" ] && [ ! -e "$USB/${path#\*}" ] && { f "missing on USB: $path"; missing=1; }; done < "$USB/MANIFEST.sha256"
    [ "$missing" = 0 ] && ok "every manifest item is present"
    if [ "$VERIFY" = 1 ]; then
      log "sha256 verify (long)…"
      if (cd "$USB" && run sha256sum -c --quiet MANIFEST.sha256); then ok "sha256 verified"; else f "sha256 mismatch on the USB"; fi
    fi
  else
    f "no MANIFEST.sha256 on $USB (prepare.sh writes it)"
  fi
  du -sh "$USB"/* 2>/dev/null | sed 's/^/     /' >&2
else
  warn "no USB found (set UNSPOKEN_USB=/media/<user>/<label>/unspoken-usb); fine after restore.sh has run"
fi

log "== 7. HF cache ($UNSPOKEN_HF_HOME)"
for repo in "$LLM_MODEL" "$DECIDER_12B_REPO" "$DECIDER_12B_BASE" "$DECIDER_2B_REPO" "$DECIDER_2B_BASE"; do
  if hf_cached "$repo"; then ok "cached: $repo"; else warn "not cached: $repo (restore.sh)"; fi
done
for fl in "$ASR_GGUF" "$DIAR_GGUF" "$TTS_GGUF"; do
  if [ -f "$UNSPOKEN_MODELS/speech/$fl" ]; then ok "speech model: $fl"; else warn "speech model missing: $UNSPOKEN_MODELS/speech/$fl (restore.sh)"; fi
done

log "== 8. clocks and thermals"
if have nvidia-smi; then
  nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,power.limit,clocks_throttle_reasons.active --format=csv 2>/dev/null | sed 's/^/     /' >&2 || true
  if [ "$CAP" = 1 ]; then
    MAX="$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)"
    if [ -n "$MAX" ]; then CAPV=$((MAX * 85 / 100)); run sudo nvidia-smi -lgc "0,$CAPV" && ok "SM clock capped at $CAPV MHz (85% of $MAX)"; else f "could not read clocks.max.sm"; fi
  fi
  if [ "$RESET" = 1 ]; then run sudo nvidia-smi -rgc && ok "GPU clocks reset"; fi
  warn "if the box powers off under load: a desk fan on the chassis + --cap-clocks (forum 385991); never run video gen during the demo"
fi
if have sensors; then { sensors 2>/dev/null | grep -i -E "temp|cpu" | head -5 | sed 's/^/     /' >&2; } || true; fi

log "== 9. sensors (mic, camera, in-process models)"
if have dpkg-query; then
  for p in libportaudio2 alsa-utils; do
    if dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q 'install ok installed'; then ok "apt: $p"; else warn "apt: $p missing (restore.sh installs it)"; fi
  done
fi
if have arecord; then
  NCAP="$(arecord -l 2>/dev/null | grep -c '^card' || true)"
  if [ "${NCAP:-0}" -gt 0 ]; then ok "capture devices: $NCAP"; { arecord -l 2>/dev/null | grep '^card' | sed 's/^/     /' >&2; } || true; else warn "no capture device (plug the USB speakerphone)"; fi
fi
VIDS="$( (ls /dev/video* 2>/dev/null || true) | tr '\n' ' ')"
if [ -n "$VIDS" ]; then ok "cameras: $VIDS"; else warn "no /dev/video* (plug the USB webcam)"; fi
SMD="${UNSPOKEN_MODELS_DIR:-$HOME/.cache/unspoken/models}"
CAMPP="$SMD/wespeaker_en_voxceleb_CAM++.onnx"
if [ -f "$CAMPP" ] && [ "$(sha256sum "$CAMPP" | awk '{print $1}')" = "c46fad10b5f81e1aa4a60c162714208577093655076c5450f8c469e522ec54ef" ]; then ok "CAM++ speaker model (sha256 ok)"
else warn "CAM++ speaker model missing or wrong at $CAMPP (restore.sh, or python -m unspoken.sensors.fetch_models)"; fi
for repo in Systran/faster-whisper-base.en Systran/faster-whisper-small.en; do
  if hf_cached "$repo"; then ok "cached: $repo"; else warn "not cached: $repo (restore.sh)"; fi
done

log "== result: $FAILS FAIL"
[ "$DRY_RUN" = 1 ] && exit 0
exit "$FAILS"
