#!/usr/bin/env bash
# deploy/usb/prepare.sh USB_DIR [--dry-run] [--offline] [--group=core,speech,sensors,fallback,extras] [--only=key,key] [--skip-optional] [--no-zstd]
#
# Run on your laptop (macOS Apple silicon or Linux) WITH internet, the night before. Fills USB_DIR with everything
# the GB10 needs so the venue Wi-Fi is only used for NemoClaw's own installer and Discord:
#   images/   docker images for linux/arm64 (docker pull --platform linux/arm64 + docker save, zstd when available)
#   hf/       Hugging Face cache directories as tar files (Qwen3.6 NVFP4, both deciders + their base models,
#             faster-whisper base.en + small.en for the CPU ASR, Kokoro)
#   speech/   GGUF files for NeMo-Speech.cpp (Parakeet, Nemotron diarization, Magpie TTS + codec)
#   sensors/  WeSpeaker CAM++ speaker ONNX (sha256-pinned; Silero VAD + YuNet are vendored in the repo)
#   apt/      Ubuntu noble arm64 .debs: libportaudio2 + alsa-utils and their dependencies (from ports.ubuntu.com)
#   wheels/   aarch64 cp312 wheelhouses: decider (no torch), unspoken (repo requirements), torch-cu130 (fallback)
#   tools/    uv, NeMo-Speech.cpp native build, the NemoClaw bootstrap installer
#   repo/     git bundle of this repo, shallow tar of NemoClaw @ lkg
#   bin/      restore.sh + the deploy/gb10 scripts, so the box can start before the repo is unpacked
#   MANIFEST.sha256, .done/<item> markers (resume-safe: re-running skips finished items), README.txt
# The list of items, pins and sizes live in usbkit.py (one source of truth). --dry-run prints every item with its
# size from the HF API / registry manifests / PyPI (or the committed sizes.lock.json with --offline) and the total.
# Needs: python3 (3.9+), docker, git, curl, tar, sha256sum or shasum. HF_TOKEN in the environment for the gated
# google/gemma-4-12B-it (accept Gemma's licence on huggingface.co first). The USB may be exFAT: nothing here needs
# symlinks on the drive (the HF cache goes on it as tar files; it is unpacked on the box).
# Staging (local disk, ~60 GB, reused by later runs): UNSPOKEN_STAGE (default ~/unspoken-usb-stage).

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
USB=""; DRY_RUN=0; OFFLINE=0; GROUP_LIST=""; ONLY=""; SKIP_OPTIONAL=0; ZSTD=auto
for a in "$@"; do
  case "$a" in
    --dry-run|-n) DRY_RUN=1 ;;
    --offline) OFFLINE=1 ;;
    --group=*) GROUP_LIST="${a#--group=}" ;;
    --only=*) ONLY=",${a#--only=}," ;;
    --skip-optional) SKIP_OPTIONAL=1 ;;
    --no-zstd) ZSTD=0 ;;
    -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --group) echo "use --group=core,speech" >&2; exit 2 ;;
    -*) echo "unknown flag $a" >&2; exit 2 ;;
    *) USB="$a" ;;
  esac
done
[ -n "$USB" ] || { echo "usage: $0 USB_DIR [--dry-run] [--offline] [--group=core,speech] [--skip-optional] [--no-zstd]" >&2; exit 2; }
STAGE="${UNSPOKEN_STAGE:-$HOME/unspoken-usb-stage}"
TOOLS="${UNSPOKEN_TOOLS:-$STAGE/tools-venv}"
PY="$TOOLS/bin/python"
GROUP_ARGS=()
if [ -n "$GROUP_LIST" ]; then
  OLDIFS="$IFS"; IFS=','; for g in $GROUP_LIST; do GROUP_ARGS+=(--group "$g"); done; IFS="$OLDIFS"
fi
if [ "$ZSTD" = auto ]; then if command -v zstd >/dev/null 2>&1; then ZSTD=1; else ZSTD=0; fi; fi

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf 'FAIL %s\n' "$*" >&2; exit 1; }
run()  { printf '+ %s\n' "$*" >&2; [ "$DRY_RUN" = 1 ] || "$@"; }
runsh(){ printf '+ %s\n' "$1" >&2; [ "$DRY_RUN" = 1 ] || bash -c "$1"; }
need() { command -v "$1" >/dev/null 2>&1 || { if [ "$DRY_RUN" = 1 ]; then log "WARN missing tool: $1 (needed for the real run)"; else die "missing tool: $1"; fi; }; }
sha()  { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

need python3; need docker; need git; need curl; need tar
command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1 || [ "$DRY_RUN" = 1 ] || die "need sha256sum or shasum"

# ---------------------------------------------------------------- 0. plan and sizes
log "plan from usbkit.py ${GROUP_LIST:+(groups: $GROUP_LIST)}"
PLAN="$(python3 "$HERE/usbkit.py" plan "${GROUP_ARGS[@]+"${GROUP_ARGS[@]}"}")"
SIZE_ARGS=(); [ "$OFFLINE" = 1 ] && SIZE_ARGS+=(--offline); [ "$ZSTD" = 1 ] && SIZE_ARGS+=(--zstd)
if [ "$DRY_RUN" = 1 ]; then
  log "sizes (${OFFLINE:+offline }$([ "$OFFLINE" = 1 ] && echo 'from sizes.lock.json' || echo 'HF API / registry / PyPI; falls back to sizes.lock.json per item'))"
  python3 "$HERE/usbkit.py" sizes "${SIZE_ARGS[@]+"${SIZE_ARGS[@]}"}" "${GROUP_ARGS[@]+"${GROUP_ARGS[@]}"}" ${PIP_PY:+--pip "$PIP_PY"} || true
  echo >&2
fi

# ---------------------------------------------------------------- 1. tools venv (hf CLI + pip)
tools_venv() {
  if [ -x "$PY" ] && [ -x "$TOOLS/bin/hf" ]; then return 0; fi
  log "tools venv at $TOOLS (pip + huggingface_hub)"
  if command -v uv >/dev/null 2>&1; then
    run uv venv --python 3.12 "$TOOLS" || run uv venv "$TOOLS"
    run uv pip install --python "$PY" "pip>=25" "huggingface_hub==2.2.0" "setuptools>=68" "setuptools-scm>=8" wheel
  else
    run python3 -m venv "$TOOLS"
    run "$PY" -m pip install --quiet --upgrade "pip>=25" "huggingface_hub==2.2.0" "setuptools>=68" "setuptools-scm>=8" wheel
  fi
}

# ---------------------------------------------------------------- 2. helpers
done_marker() { echo "$USB/.done/$1"; }
is_done() { [ -f "$(done_marker "$1")" ]; }
mark_done() { # mark_done KEY PATH...  (records sha256 of the produced files, relative to the USB root)
  local key="$1"; shift
  [ "$DRY_RUN" = 1 ] && return 0
  local f rel
  for f in "$@"; do   # a redone item replaces its old lines (no stale hashes)
    rel="${f#"$USB"/}"
    awk -v r="$rel" '{ p = substr($0, index($0, "  ") + 2) } p != r && index(p, r "/") != 1' "$USB/MANIFEST.sha256" > "$USB/MANIFEST.sha256.tmp"
    mv "$USB/MANIFEST.sha256.tmp" "$USB/MANIFEST.sha256"
  done
  for f in "$@"; do
    if [ -d "$f" ]; then
      (cd "$USB" && find "${f#"$USB"/}" -type f ! -name '*.part' -print0 | xargs -0 sha256sum 2>/dev/null || \
       cd "$USB" && find "${f#"$USB"/}" -type f ! -name '*.part' -exec shasum -a 256 {} +) >> "$USB/MANIFEST.sha256"
    else
      (cd "$USB" && sha "${f#"$USB"/}") >> "$USB/MANIFEST.sha256"
    fi
  done
  touch "$(done_marker "$key")"
}
PIP_FLAGS="$(python3 "$HERE/usbkit.py" pip-flags)"
export HF_HOME="$STAGE/hf-home" HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=0

fetch_image() { # KEY DEST REF
  local key="$1" dest="$USB/$2" ref="$3"
  run docker pull --platform linux/arm64 "$ref"
  if [ "$ZSTD" = 1 ]; then
    dest="$dest.zst"
    runsh "docker save $(printf '%q' "$ref") | zstd -T0 -3 -q -o $(printf '%q' "$dest.part") && mv $(printf '%q' "$dest.part") $(printf '%q' "$dest")"
  else
    runsh "docker save -o $(printf '%q' "$dest.part") $(printf '%q' "$ref") && mv $(printf '%q' "$dest.part") $(printf '%q' "$dest")"
  fi
  [ "$DRY_RUN" = 1 ] || printf '%s\n' "$ref" > "${dest%.tar*}.ref"
  mark_done "$key" "$dest" "${dest%.tar*}.ref"
}

fetch_hf() { # KEY DEST REPO REV
  local key="$1" dest="$USB/$2" repo="$3" rev="$4"
  local rev_args=(); [ -n "$rev" ] && rev_args=(--revision "$rev")
  run "$TOOLS/bin/hf" download "$repo" "${rev_args[@]+"${rev_args[@]}"}"
  local dir="models--${repo//\//--}" list="$STAGE/$key.files"
  # a --revision download writes no refs/main, and offline loaders (HF_HUB_OFFLINE=1, local_files_only) resolve
  # "main": point main at the pinned snapshot so faster-whisper / transformers find it with no network
  if [ -n "$rev" ] && [ "$DRY_RUN" != 1 ] && [ ! -f "$HF_HOME/hub/$dir/refs/main" ]; then
    mkdir -p "$HF_HOME/hub/$dir/refs"; printf '%s' "$rev" > "$HF_HOME/hub/$dir/refs/main"
  fi
  # huggingface_hub >= 2 (hf-xet) stores file content once per hub in hub/blobs/<xx>/<hash>; the repo's own
  # blobs/<sha256> entries are symlinks into it. Tar the repo dir AND the shared blobs it points to, keeping the
  # links, so the restored ~/.cache/huggingface/hub is the same layout (older hubs: no shared blobs, same result).
  runsh "cd $(printf '%q' "$HF_HOME/hub") && { echo $(printf '%q' "$dir"); find $(printf '%q' "$dir/blobs") -type l -exec readlink {} \\; | sed -n 's#^\\.\\./\\.\\./blobs/#blobs/#p'; } > $(printf '%q' "$list")"
  runsh "tar -C $(printf '%q' "$HF_HOME/hub") -cf $(printf '%q' "$dest.part") -T $(printf '%q' "$list") && mv $(printf '%q' "$dest.part") $(printf '%q' "$dest")"
  mark_done "$key" "$dest"
}

fetch_hf_files() { # KEY DEST REPO FILES(comma)
  local key="$1" dest="$USB/$2" repo="$3" files="$4"
  local list=(); OLDIFS="$IFS"; IFS=','; for f in $files; do list+=("$f"); done; IFS="$OLDIFS"
  run "$TOOLS/bin/hf" download "$repo" "${list[@]}" --local-dir "$dest"
  [ "$DRY_RUN" = 1 ] || rm -rf "$dest/.cache"
  mark_done "$key" "$dest"
}

# check_sha FILE SHA256: refuses (and deletes) a download whose hash is not the pinned one.
check_sha() {
  [ "$DRY_RUN" = 1 ] && { printf '+ (sha256 %s must be %s)\n' "$1" "$2" >&2; return 0; }
  local got; got="$(sha "$1" | awk '{print $1}')"
  if [ "$got" != "$2" ]; then rm -f "$1"; die "sha256 mismatch for $1: got $got, pinned $2"; fi
}

fetch_url() { # KEY DEST URL [SHA256]
  local key="$1" dest="$USB/$2" url="$3" want="${4:-}"
  run curl -fL --retry 5 --retry-delay 3 -C - -o "$dest.part" "$url"
  [ -n "$want" ] && check_sha "$dest.part" "$want"
  run mv "$dest.part" "$dest"
  mark_done "$key" "$dest"
}

fetch_apt() { # KEY DEST PACKAGES(comma): noble arm64 debs + deps, each sha256-checked against the Packages index
  local key="$1" dest="$USB/$2" pkgs="$3"
  if [ "$DRY_RUN" = 1 ] && [ "$OFFLINE" = 1 ]; then
    printf '+ python3 usbkit.py apt-resolve %s   (noble arm64 Packages.xz from ports.ubuntu.com; then curl + sha256 per .deb)\n' "${pkgs//,/ }" >&2
    mark_done "$key"; return 0
  fi
  local list; list="$(python3 "$HERE/usbkit.py" apt-resolve ${pkgs//,/ })" || die "apt resolve failed for $pkgs"
  run mkdir -p "$dest"
  local name ver size want url f
  while IFS=$'\t' read -r name ver size want url; do
    [ -n "$name" ] || continue
    f="$dest/$(basename "$url")"
    log "   $name $ver ($size bytes)"
    if [ -f "$f" ] && [ "$DRY_RUN" != 1 ] && [ "$(sha "$f" | awk '{print $1}')" = "$want" ]; then continue; fi
    run curl -fL --retry 5 --retry-delay 3 -o "$f.part" "$url"
    check_sha "$f.part" "$want"
    run mv "$f.part" "$f"
  done <<< "$list"
  [ "$DRY_RUN" = 1 ] || printf '%s\n' "${pkgs//,/ }" > "$dest/PACKAGES"
  mark_done "$key" "$dest"
}

fetch_git() { # KEY DEST URL REF MODE
  local key="$1" dest="$USB/$2" url="$3" ref="$4" mode="$5"
  if [ "$mode" = shallow ]; then
    local name; name="$(basename "$dest" .tar)"
    [ "$DRY_RUN" = 1 ] || rm -rf "$STAGE/git/$name"
    run git clone --depth 1 --branch "$ref" "$url" "$STAGE/git/$name"
    runsh "tar -C $(printf '%q' "$STAGE/git") -cf $(printf '%q' "$dest.part") $(printf '%q' "$name") && mv $(printf '%q' "$dest.part") $(printf '%q' "$dest")"
  else
    run git -C "$url" bundle create "$dest.part" --all
    run mv "$dest.part" "$dest"
  fi
  mark_done "$key" "$dest"
}

fetch_pip() { # KEY DEST MODE INDEX REQS(comma) EXTRA(|)
  local key="$1" dest="$USB/$2" mode="$3" index="$4" reqs="$5" extra="$6"
  local args=() r e
  OLDIFS="$IFS"; IFS=','; for r in $reqs; do [ -n "$r" ] && args+=(-r "$r"); done; IFS="$OLDIFS"
  local extras=(); OLDIFS="$IFS"; IFS='|'; for e in $extra; do [ -n "$e" ] && extras+=("$e"); done; IFS="$OLDIFS"
  local idx=(); [ -n "$index" ] && idx=(--index-url "$index" --extra-index-url https://pypi.org/simple)
  run mkdir -p "$dest"
  if [ "$mode" = nodeps ]; then
    # pinned wheels, no resolution; git extras are built into pure-Python wheels
    # shellcheck disable=SC2086
    [ ${#args[@]} -eq 0 ] || run "$PY" -m pip download --no-deps -d "$dest" $PIP_FLAGS "${idx[@]+"${idx[@]}"}" "${args[@]}"
    for e in "${extras[@]+"${extras[@]}"}"; do run "$PY" -m pip wheel --no-deps -w "$dest" "$e"; done
    if [ "$DRY_RUN" != 1 ]; then
      { for r in ${reqs//,/ }; do grep -v '^\s*#' "$r"; done; for e in "${extras[@]+"${extras[@]}"}"; do echo "${e%% @*}"; done; } > "$dest/requirements.txt"
    fi
  else
    # shellcheck disable=SC2086
    run "$PY" -m pip download -d "$dest" $PIP_FLAGS "${idx[@]+"${idx[@]}"}" "${args[@]+"${args[@]}"}" "${extras[@]+"${extras[@]}"}"
  fi
  mark_done "$key" "$dest"
}

# ---------------------------------------------------------------- 3. go
log "USB: $USB   stage: $STAGE   zstd: $ZSTD   dry-run: $DRY_RUN"
if [ "$DRY_RUN" != 1 ]; then
  mkdir -p "$USB/.done" "$USB/images" "$USB/hf" "$USB/speech" "$USB/sensors" "$USB/apt" "$USB/wheels" "$USB/tools" "$USB/repo" "$USB/bin" "$STAGE/git" "$HF_HOME"
  touch "$USB/MANIFEST.sha256"
  tools_venv
  [ -n "${HF_TOKEN:-}" ] || log "HF_TOKEN is not set: google/gemma-4-12B-it (gated) will fail; export HF_TOKEN=hf_... and re-run (resume-safe)"
fi

FAILED=(); SKIPPED=0; DONE=0
while IFS=$'\t' read -r key group kind dest optional s1 s2 s3 s4; do
  [ -n "$key" ] || continue
  [ "$s1" = - ] && s1=""; [ "$s2" = - ] && s2=""; [ "$s3" = - ] && s3=""; [ "$s4" = - ] && s4=""
  if [ -n "$ONLY" ] && [ "${ONLY#*,"$key",}" = "$ONLY" ]; then continue; fi
  if [ "$SKIP_OPTIONAL" = 1 ] && [ "$optional" = 1 ]; then log "skip optional $key"; SKIPPED=$((SKIPPED + 1)); continue; fi
  if is_done "$key"; then log "done already: $key"; DONE=$((DONE + 1)); continue; fi
  log "== $key ($group/$kind) -> $dest"
  [ "$DRY_RUN" = 1 ] || mkdir -p "$(dirname "$USB/$dest")"
  if ( case "$kind" in
        image)    fetch_image "$key" "$dest" "$s1" ;;
        hf)       fetch_hf "$key" "$dest" "$s1" "$s2" ;;
        hf-files) fetch_hf_files "$key" "$dest" "$s1" "$s2" ;;
        url)      fetch_url "$key" "$dest" "$s1" "$s2" ;;
        apt)      fetch_apt "$key" "$dest" "$s1" ;;
        git)      fetch_git "$key" "$dest" "$s1" "$s2" "$s3" ;;
        pip)      fetch_pip "$key" "$dest" "$s1" "$s2" "$s3" "$s4" ;;
        *) die "unknown kind $kind" ;;
      esac ); then DONE=$((DONE + 1)); else FAILED+=("$key"); log "FAILED $key (re-run to resume; optional=$optional)"; fi
done <<< "$PLAN"

# ---------------------------------------------------------------- 4. bin/, README, manifest tidy
if [ "$DRY_RUN" != 1 ]; then
  cp "$HERE/restore.sh" "$HERE/usbkit.py" "$USB/bin/"
  mkdir -p "$USB/bin/gb10" && cp "$REPO_ROOT"/deploy/gb10/*.sh "$USB/bin/gb10/"
  cp "$REPO_ROOT/docs/HACKDAY.md" "$USB/" 2>/dev/null || true
  sort -s -k2 "$USB/MANIFEST.sha256" -o "$USB/MANIFEST.sha256"
  cat > "$USB/README.txt" <<EOF
LocoCisco GB10 USB, prepared $(date -u +%Y-%m-%dT%H:%M:%SZ) from $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')
On the box:   bash <this drive>/bin/restore.sh            (then follow HACKDAY.md)
Verify:       cd <this drive> && sha256sum -c MANIFEST.sha256
Items done:   $(ls "$USB/.done" | tr '\n' ' ')
EOF
fi
log "finished: $DONE done, $SKIPPED skipped, ${#FAILED[@]} failed${FAILED[*]:+: ${FAILED[*]}}"
[ "$DRY_RUN" = 1 ] || du -sh "$USB" >&2
[ "${#FAILED[@]}" -eq 0 ]
