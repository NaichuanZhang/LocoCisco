#!/usr/bin/env bash
# deploy/usb/restore.sh [USB_DIR] [--dry-run] [--verify] [--skip-images] [--native-decider]
#
# On the GB10, first thing after 00-preflight.sh. Everything goes to the user's home (no sudo):
#   docker load  images/*.tar(.zst)          -> local docker images (skips images already present)
#   tar -x       hf/*.tar                    -> $UNSPOKEN_HF_HOME/hub   (~/.cache/huggingface, shared with NemoClaw)
#   copy         speech/ wheels/             -> $UNSPOKEN_MODELS        (~/unspoken-models); Magpie's .nemo is
#                                               unpacked to speech/nvidia/magpie_tts_multilingual_357m/tokenizer
#   copy         sensors/*.onnx|*.model      -> ${UNSPOKEN_MODELS_DIR:-~/.cache/unspoken/models} (WeSpeaker CAM++, Light-ASD,
#                                               sha256-checked; where unspoken.sensors.assets looks first)
#   sudo apt     apt/*.deb                   -> libportaudio2 + alsa-utils, only the packages the box lacks (offline
#                                               .debs first, `apt-get install` from the network if that fails)
#   extract      tools/uv, tools/nemo-speech -> $UNSPOKEN_MODELS/tools ; tools/nemoclaw.sh + repo/nemoclaw-lkg.tar -> ~/nemoclaw-src
#   git clone    repo/unspoken.bundle        -> $UNSPOKEN_HOME (~/unspoken), branch hack/main, then an OFFLINE venv
#                                               from wheels/unspoken with the USB's uv
#   --native-decider  also builds $UNSPOKEN_STATE/decider-venv from wheels/torch-cu130 + wheels/decider
#                     (the no-docker fallback for deploy/gb10/decider.sh with DECIDER_RUNTIME=native)
# --verify runs `sha256sum -c MANIFEST.sha256` first (~10 min for 100 GB). Re-running is safe: finished parts are skipped.
# Needs on the box: docker (nvidia runtime), tar, zstd (only if the images were compressed), python3.12 (DGX OS has it),
# sudo for the apt step only (--skip-apt skips it).

source "$(dirname "${BASH_SOURCE[0]}")/gb10/lib.sh" 2>/dev/null || source "$(dirname "${BASH_SOURCE[0]}")/../gb10/lib.sh"
parse_common_flags "$@"
USB=""; VERIFY=0; SKIP_IMAGES=0; NATIVE=0; SKIP_APT=0
for a in "${ARGS[@]+"${ARGS[@]}"}"; do
  case "$a" in --verify) VERIFY=1 ;; --skip-images) SKIP_IMAGES=1 ;; --skip-apt) SKIP_APT=1 ;; --native-decider) NATIVE=1 ;; -*) die "unknown flag $a" ;; *) USB="$a" ;; esac
done
SENSORS_DIR="${UNSPOKEN_MODELS_DIR:-$HOME/.cache/unspoken/models}"
KIT="$(dirname "${BASH_SOURCE[0]}")/usbkit.py"
if [ -z "$USB" ]; then
  USB="${UNSPOKEN_USB:-}"
  if [ -z "$USB" ]; then
    for d in /media/*/*/unspoken-usb /media/*/unspoken-usb /mnt/*/unspoken-usb "$HOME/unspoken-usb" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; do
      [ -f "$d/MANIFEST.sha256" ] && { USB="$d"; break; }
    done
  fi
fi
[ -n "$USB" ] && [ -d "$USB" ] || die "USB dir not found; pass it: restore.sh /media/\$USER/<label>/unspoken-usb"
log "USB $USB -> HF $UNSPOKEN_HF_HOME, models $UNSPOKEN_MODELS, repo $UNSPOKEN_HOME"
export UV_OFFLINE=1 UV_PYTHON_PREFERENCE=only-system UV_NO_PROGRESS=1   # never fetch a managed python or wheels
T0=$(date +%s)
mkdirs; [ "$DRY_RUN" = 1 ] || mkdir -p "$UNSPOKEN_HF_HOME/hub" "$UNSPOKEN_MODELS/tools" "$UNSPOKEN_MODELS/speech" "$UNSPOKEN_MODELS/sensors" "$UNSPOKEN_MODELS/wheels"

# ---------------------------------------------------------------- 0. verify
if [ "$VERIFY" = 1 ]; then
  log "== verify sha256 (long)"
  (cd "$USB" && run sha256sum -c --quiet MANIFEST.sha256) && ok "sha256 verified" || die "sha256 mismatch: re-copy the bad files from the laptop"
fi

# ---------------------------------------------------------------- 1. docker images
log "== 1. docker images"
if [ "$SKIP_IMAGES" = 1 ]; then warn "--skip-images"; else
  need docker
  for f in "$USB"/images/*.tar "$USB"/images/*.tar.zst; do
    [ -f "$f" ] || continue
    ref=""; [ -f "${f%.tar*}.ref" ] && ref="$(cat "${f%.tar*}.ref")"
    if [ -n "$ref" ] && docker image inspect "$ref" >/dev/null 2>&1; then ok "already loaded: $ref"; continue; fi
    case "$f" in
      *.zst) need zstd; run_sh "zstd -dc $(printf '%q' "$f") | docker load" ;;
      *)     run docker load -i "$f" ;;
    esac
  done
  [ "$DRY_RUN" = 1 ] || docker images --format '     {{.Repository}}:{{.Tag}}  {{.Size}}' | grep -E 'nvcr.io' >&2 || true
fi

# ---------------------------------------------------------------- 2. HF cache
log "== 2. Hugging Face cache"
for f in "$USB"/hf/*.tar; do
  [ -f "$f" ] || continue
  name="$(basename "$f" .tar)"; repo="${name//--//}"
  if hf_cached "$repo"; then ok "already in cache: $repo"; continue; fi
  run tar -C "$UNSPOKEN_HF_HOME/hub" -xf "$f"
  [ "$DRY_RUN" = 1 ] || { hf_cached "$repo" && ok "restored $repo" || fail "no snapshot after extracting $f"; }
done
# offline loaders resolve refs/main; a pinned download from an older prepare.sh has none: point it at the only snapshot
for d in "$UNSPOKEN_HF_HOME"/hub/models--*; do
  [ -d "$d/snapshots" ] && [ ! -f "$d/refs/main" ] || continue
  snaps=("$d"/snapshots/*)
  if [ "${#snaps[@]}" -eq 1 ]; then run mkdir -p "$d/refs"; run_sh "printf %s $(basename "${snaps[0]}") > $(printf '%q' "$d/refs/main")"; fi
done

# ---------------------------------------------------------------- 3. speech + sensors + wheels
log "== 3. models and wheels -> $UNSPOKEN_MODELS"
copy_tree() { # SRC DST  (cp, not rsync: rsync >= 3.4 refuses paths through a symlinked home, e.g. /home -> /local/home)
  [ -d "$1" ] || return 0
  run mkdir -p "$2"; run cp -R "$1/." "$2/"
}
copy_tree "$USB/speech" "$UNSPOKEN_MODELS/speech"
copy_tree "$USB/wheels" "$UNSPOKEN_MODELS/wheels"

# sensors models (unspoken.sensors.assets: $UNSPOKEN_MODELS_DIR, then ~/.cache/unspoken/models, then vendored)
run mkdir -p "$SENSORS_DIR"
for f in "$USB"/sensors/*.onnx "$USB"/sensors/*.model; do
  [ -f "$f" ] || [ "$DRY_RUN" = 1 ] || continue
  name="$(basename "$f")"
  want="$(python3 "$KIT" plan --group sensors 2>/dev/null | awk -F'\t' -v d="sensors/$name" '$3 == "url" && $4 == d {print $7}' || true)"
  if [ -f "$SENSORS_DIR/$name" ] && [ -n "$want" ] && [ "$(sha256sum "$SENSORS_DIR/$name" | awk '{print $1}')" = "$want" ]; then
    ok "sensors model already in place: $SENSORS_DIR/$name"; continue
  fi
  run cp "$f" "$SENSORS_DIR/$name"
  if [ "$DRY_RUN" != 1 ] && [ -n "$want" ]; then
    if [ "$(sha256sum "$SENSORS_DIR/$name" | awk '{print $1}')" = "$want" ]; then ok "sensors model $name (sha256 ok)"
    else rm -f "$SENSORS_DIR/$name"; fail "sha256 mismatch for $name; fetch it online: python -m unspoken.sensors.fetch_models"; fi
  fi
done
NEMO="$UNSPOKEN_MODELS/speech/nvidia/magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.nemo"
TOK="$UNSPOKEN_MODELS/speech/nvidia/magpie_tts_multilingual_357m/tokenizer"
if [ -f "$NEMO" ] || [ "$DRY_RUN" = 1 ]; then
  if [ ! -f "$TOK/model_config.yaml" ]; then
    run mkdir -p "$TOK"; run tar -xf "$NEMO" -C "$TOK"   # a .nemo is a tar archive; the runtime wants the extracted dir
  else ok "Magpie tokenizer already extracted"; fi
fi

# ---------------------------------------------------------------- 3b. apt: libportaudio2 + alsa-utils (sensors audio)
log "== 3b. apt packages for the microphone"
if [ "$SKIP_APT" = 1 ]; then warn "--skip-apt"
elif [ ! -f "$USB/apt/PACKAGES" ] && [ "$DRY_RUN" != 1 ]; then warn "no apt/ on the USB"
elif ! have dpkg-query; then warn "not a Debian/Ubuntu system (no dpkg-query): skipping $(cat "$USB/apt/PACKAGES" 2>/dev/null || echo 'libportaudio2 alsa-utils')"
else
  WANT_PKGS="$(cat "$USB/apt/PACKAGES" 2>/dev/null || echo 'libportaudio2 alsa-utils')"
  installed() { dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q 'install ok installed'; }
  MISSING=()
  for p in $WANT_PKGS; do installed "$p" || MISSING+=("$p"); done
  if [ "${#MISSING[@]}" -eq 0 ]; then ok "already installed: $WANT_PKGS"
  else
    DEBS=()   # only .debs whose package is not installed: never downgrades what DGX OS already has
    for d in "$USB"/apt/*.deb; do
      [ -f "$d" ] || continue
      installed "$(dpkg-deb -f "$d" Package 2>/dev/null)" || DEBS+=("$d")
    done
    log "installing ${MISSING[*]} from ${#DEBS[@]} local .debs (sudo)"
    if ! run sudo apt-get install -y --no-install-recommends "${DEBS[@]}"; then
      warn "offline .deb install failed; trying the network"
      run sudo apt-get install -y --no-install-recommends "${MISSING[@]}" || fail "apt install failed: sensors fall back to arecord, or no mic"
    fi
  fi
fi

# ---------------------------------------------------------------- 4. tools
log "== 4. tools"
UVTGZ="$(ls "$USB"/tools/uv-*-aarch64-unknown-linux-gnu.tar.gz 2>/dev/null | head -1 || true)"
UV="$UNSPOKEN_MODELS/tools/uv/uv"
if [ -n "$UVTGZ" ] && [ ! -x "$UV" ]; then
  run mkdir -p "$UNSPOKEN_MODELS/tools/uv"
  run tar -xzf "$UVTGZ" -C "$UNSPOKEN_MODELS/tools/uv" --strip-components=1
fi
[ -x "$UV" ] || UV="$(command -v uv || true)"
NSTGZ="$(ls "$USB"/tools/nemo-speech-*-linux-aarch64-cuda13.tar.gz 2>/dev/null | head -1 || true)"
if [ -n "$NSTGZ" ] && [ ! -x "$UNSPOKEN_MODELS/tools/nemo-speech/bin/nemo-speech" ]; then
  run mkdir -p "$UNSPOKEN_MODELS/tools/nemo-speech"
  run tar -xzf "$NSTGZ" -C "$UNSPOKEN_MODELS/tools/nemo-speech" --strip-components=1
fi
if [ -f "$USB/repo/nemoclaw-lkg.tar" ] && [ ! -d "$HOME/nemoclaw-src" ]; then
  run mkdir -p "$HOME/nemoclaw-src"
  run tar -xf "$USB/repo/nemoclaw-lkg.tar" -C "$HOME/nemoclaw-src" --strip-components=1
  [ -f "$USB/tools/nemoclaw.sh" ] && run cp "$USB/tools/nemoclaw.sh" "$HOME/nemoclaw-src/nemoclaw-bootstrap.sh"
fi

# ---------------------------------------------------------------- 5. repo + venv (offline)
log "== 5. repo -> $UNSPOKEN_HOME"
if [ -f "$USB/repo/unspoken.bundle" ]; then
  if [ ! -d "$UNSPOKEN_HOME/.git" ]; then
    run git clone --branch hack/main "$USB/repo/unspoken.bundle" "$UNSPOKEN_HOME" || run git clone "$USB/repo/unspoken.bundle" "$UNSPOKEN_HOME"
  else ok "repo already cloned (git -C $UNSPOKEN_HOME fetch $USB/repo/unspoken.bundle to update)"; fi
  if [ -n "$UV" ] && [ ! -x "$UNSPOKEN_HOME/.venv/bin/python" ]; then
    run "$UV" venv --python 3.12 "$UNSPOKEN_HOME/.venv"
    REQS=(-r "$UNSPOKEN_HOME/requirements-dev.txt")
    for r in "$UNSPOKEN_HOME"/requirements/*.txt; do [ -f "$r" ] && REQS+=(-r "$r"); done
    run "$UV" pip install --python "$UNSPOKEN_HOME/.venv/bin/python" --offline --no-index --find-links "$UNSPOKEN_MODELS/wheels/unspoken" "${REQS[@]}"
  fi
  # Light-ASD: torch cu130 (GPU) + the sdist-built MFCC wheel, kept out of requirements/*.txt (see that file)
  if [ -n "$UV" ] && [ -f "$UNSPOKEN_HOME/requirements/gpu/transcribe-asd-torch.txt" ]; then
    run "$UV" pip install --python "$UNSPOKEN_HOME/.venv/bin/python" --offline --no-index \
      --find-links "$UNSPOKEN_MODELS/wheels/torch-cu130" --find-links "$UNSPOKEN_MODELS/wheels/sdist" \
      --find-links "$UNSPOKEN_MODELS/wheels/unspoken" -r "$UNSPOKEN_HOME/requirements/gpu/transcribe-asd-torch.txt"
  fi
else warn "no repo/unspoken.bundle on the USB"; fi

# ---------------------------------------------------------------- 6. optional native decider venv
if [ "$NATIVE" = 1 ]; then
  log "== 6. native decider venv (torch cu130 wheels)"
  [ -n "$UV" ] || die "uv not found"
  V="$UNSPOKEN_STATE/decider-venv"
  [ -x "$V/bin/python" ] || run "$UV" venv --python 3.12 "$V"
  run_sh "$(printf '%q' "$UV") pip install --python $(printf '%q' "$V/bin/python") --offline --no-index --no-deps $(printf '%q' "$UNSPOKEN_MODELS")/wheels/torch-cu130/*.whl $(printf '%q' "$UNSPOKEN_MODELS")/wheels/decider/*.whl"
  ok "DECIDER_RUNTIME=native DECIDER_BIN=$V/bin/strands-decider deploy/gb10/decider.sh 12b"
fi

ok "restore done in $(( $(date +%s) - T0 ))s. Next: deploy/gb10/00-preflight.sh ; NemoClaw onboarding ; deploy/gb10/up.sh ; python -m unspoken.sensors.check"
