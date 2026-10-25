#!/usr/bin/env bash
# deploy/gb10/speech.sh asr|tts|all [--dry-run] [--stop] [--status]
#
# NVIDIA NeMo-Speech.cpp 0.2.0 (C++ runtime on ggml; arm64 + CUDA 13 build made for DGX Spark), loopback only.
#   asr -> 127.0.0.1:8770  POST /v1/audio/transcriptions  (OpenAI-compatible multipart; response_format=verbose_json
#                          adds words + timestamps, diarization=true adds speaker tags)
#                          Parakeet TDT 0.6B v3 Q8 (offline, 25 languages, 282x real time on GB10 per NVIDIA forum)
#                          + Nemotron 3 Diarization Q8 (<= 8 speakers). ASR_MODEL=nemotron35 swaps in the streaming
#                          multilingual Nemotron 3.5 ASR 0.6B Q8.
#   tts -> 127.0.0.1:8771  POST /v1/audio/speech  (OpenAI-compatible) MagpieTTS Multilingual 357M Q8 + NanoCodec.
#
# Runtime (SPEECH_RUNTIME): docker (default: nvcr.io/nvidia/nemo-speech.cpp:0.2.0, 186 MB, scratch image whose
#   entrypoint is `nemo-speech`) | native (release tarball nemo-speech-0.2.0-linux-aarch64-cuda13 from the USB,
#   NEMO_SPEECH_BIN=$UNSPOKEN_MODELS/tools/nemo-speech/bin/nemo-speech).
# Models live in $UNSPOKEN_MODELS/speech/<hf repo>/<file> (restore.sh puts them there; the GGUF is passed by
# path, so the runtime never touches the network). SPEECH_DIAR=0 drops diarization. SPEECH_THREADS (default 4).
# Health: curl -s 127.0.0.1:8770/ready ; curl -s 127.0.0.1:8771/ready

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_common_flags "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"

WHAT="${1:-}"; shift || true
ACTION=start
for a in "$@"; do case "$a" in --stop) ACTION=stop ;; --status) ACTION=status ;; *) die "unknown flag $a" ;; esac; done
case "$WHAT" in asr|tts|all) ;; *) usage; exit 2 ;; esac

RUNTIME="${SPEECH_RUNTIME:-docker}"
MODELS="$UNSPOKEN_MODELS/speech"
THREADS="${SPEECH_THREADS:-4}"
case "${ASR_MODEL:-parakeet}" in
  parakeet)   ASR_FILE="$ASR_GGUF" ;;
  nemotron35) ASR_FILE="$ASR_STREAMING_GGUF" ;;
  *) die "ASR_MODEL must be parakeet or nemotron35" ;;
esac

# model_path IN_CONTAINER_ROOT RELATIVE: path as the runtime sees it.
model_path() { echo "$1/$2"; }

# serve_args KIND ROOT: the `nemo-speech serve` arguments for asr|tts with models under ROOT.
serve_args() {
  local kind="$1" root="$2"
  case "$kind" in
    asr)
      printf '%s ' serve --host 127.0.0.1 --port "$PORT_ASR" --threads "$THREADS" --asr-model "$(model_path "$root" "$ASR_FILE")"
      [ "${SPEECH_DIAR:-1}" = 1 ] && printf '%s ' --diar-model "$(model_path "$root" "$DIAR_GGUF")"
      ;;
    tts)
      printf '%s ' serve --host 127.0.0.1 --port "$PORT_TTS" --threads "$THREADS" \
        --tts-model "$(model_path "$root" "$TTS_GGUF")" --codec-model "$(model_path "$root" "$TTS_CODEC_GGUF")" \
        --tokenizer-dir "$(model_path "$root" "$TTS_TOKENIZER_DIR")"
      ;;
  esac
  [ "${SPEECH_NO_UI:-0}" = 1 ] && printf '%s ' --no-ui
  [ -n "${SPEECH_EXTRA:-}" ] && printf '%s ' "${SPEECH_EXTRA}"
  return 0
}

check_models() {
  local kind="$1" f
  [ "$DRY_RUN" = 1 ] && return 0
  case "$kind" in
    asr) for f in "$ASR_FILE" "$DIAR_GGUF"; do [ -f "$MODELS/$f" ] || die "missing $MODELS/$f (run deploy/usb/restore.sh)"; done ;;
    tts) for f in "$TTS_GGUF" "$TTS_CODEC_GGUF" "$TTS_TOKENIZER_DIR/model_config.yaml"; do [ -e "$MODELS/$f" ] || die "missing $MODELS/$f (run deploy/usb/restore.sh)"; done ;;
  esac
}

port_of() { if [ "$1" = asr ]; then echo "$PORT_ASR"; else echo "$PORT_TTS"; fi; }

start_one() {
  local kind="$1" name="unspoken-$1" port; port="$(port_of "$kind")"
  local ready="http://127.0.0.1:$port/ready"
  log "speech $kind via $RUNTIME -> 127.0.0.1:$port"
  check_models "$kind"
  mkdirs
  if [ "$RUNTIME" = docker ]; then
    need docker
    if docker_running "$name"; then ok "$name already running"; return 0; fi
    docker_rm_if_exists "$name"
    # shellcheck disable=SC2046,SC2207
    local args=($(serve_args "$kind" /models))
    # shellcheck disable=SC2046
    run docker run -d --name "$name" --restart unless-stopped --network host $(gpu_flag) \
      -v "$MODELS:/models:ro" "$IMAGE_SPEECH" "${args[@]}"
    wait_http "$ready" "${SPEECH_WAIT:-180}" "speech $kind" || { docker logs --tail 40 "$name" >&2 || true; return 1; }
  else
    local bin="${NEMO_SPEECH_BIN:-$UNSPOKEN_MODELS/tools/nemo-speech/bin/nemo-speech}"
    [ -x "$bin" ] || [ "$DRY_RUN" = 1 ] || die "nemo-speech binary not found at $bin (set NEMO_SPEECH_BIN or run restore.sh)"
    local pidfile="$UNSPOKEN_STATE/$name.pid" logf="$UNSPOKEN_LOGS/$kind.log"
    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then ok "$name already running"; return 0; fi
    run_sh "nohup $(printf '%q' "$bin") $(serve_args "$kind" "$MODELS") > $(printf '%q' "$logf") 2>&1 </dev/null & echo \$! > $(printf '%q' "$pidfile")"
    wait_http "$ready" "${SPEECH_WAIT:-180}" "speech $kind" || { tail -40 "$logf" >&2 || true; return 1; }
  fi
  [ "$DRY_RUN" = 1 ] || { curl -fsS -m 3 "$ready" 2>/dev/null && echo >&2 || true; }
}

stop_one() {
  local kind="$1" name="unspoken-$1"
  if [ "$RUNTIME" = docker ]; then docker_rm_if_exists "$name"
  else local pidfile="$UNSPOKEN_STATE/$name.pid"
    if [ -f "$pidfile" ]; then run kill "$(cat "$pidfile")" || true; [ "$DRY_RUN" = 1 ] || rm -f "$pidfile"; fi
  fi
}

status_one() {
  local kind="$1" port; port="$(port_of "$kind")"
  if [ "$DRY_RUN" = 1 ]; then printf '+ curl -fsS http://127.0.0.1:%s/ready\n' "$port" >&2; return 0; fi
  if curl -fsS -m 3 "http://127.0.0.1:$port/ready" 2>/dev/null; then echo; ok "speech $kind ready on :$port"
  else fail "speech $kind not ready on :$port"; return 1; fi
}

KINDS=("$WHAT"); [ "$WHAT" = all ] && KINDS=(asr tts)
rc=0
for k in "${KINDS[@]}"; do
  case "$ACTION" in
    start)  start_one "$k" || rc=1 ;;
    stop)   stop_one "$k" ;;
    status) status_one "$k" || rc=1 ;;
  esac
done
exit $rc
