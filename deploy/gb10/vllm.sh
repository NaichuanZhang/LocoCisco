#!/usr/bin/env bash
# deploy/gb10/vllm.sh [--dry-run] [--stop] [--status]
#
# Standalone vLLM for nvidia/Qwen3.6-35B-A3B-NVFP4 on 127.0.0.1:8000. Use it ONLY when the NemoClaw-managed
# container `nemoclaw-vllm` is not running: NemoClaw wires OpenClaw to its own container on the same port, and
# LocoCisco calls whichever answers on :8000 (Chat Completions, same model id).
#
# Recipe: the model card's DGX Spark command (https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4), in the image
# NemoClaw ships (nvcr.io/nvidia/vllm:26.05.post1-py3). Tool calls need Chat Completions with
# --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 (NemoClaw docs).
#
# Memory budget (128 GB unified; ~120 GiB visible to CUDA; weights + KV + runtimes all share it):
#   vLLM at --gpu-memory-utilization 0.40  ~48 GiB cap   (23.5 GB NVFP4 weights + KV cache/activations)
#   Strands Decider 12B bf16               ~27 GB        (23 GB weights + activations)
#   NeMo-Speech.cpp ASR + diarization      ~3 GB
#   MagpieTTS (optional)                   ~1 GB
#   OS, Docker, NemoClaw/OpenShell sandbox, OpenClaw, LocoCisco host process  ~12 GB
#   headroom                               ~25 GB        (Decider 2B optional: +5 GB)
# NemoClaw's managed profile for >= 64 GB boxes uses the same 0.40 / 4 seqs / 8192 batched tokens, 262K ctx.
#
# Env: VLLM_GPU_UTIL (0.40)  VLLM_MAX_MODEL_LEN (65536)  VLLM_MAX_SEQS (4)  VLLM_MTP=1|0 (speculative MTP from the
#      card recipe; NemoClaw's managed profile leaves it off)  VLLM_LOAD_FORMAT (fastsafetensors)  VLLM_EXTRA
# Health: curl -s 127.0.0.1:8000/v1/models

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
parse_common_flags "$@"
set -- "${ARGS[@]+"${ARGS[@]}"}"
ACTION=start
for a in "$@"; do case "$a" in --stop) ACTION=stop ;; --status) ACTION=status ;; *) die "unknown flag $a" ;; esac; done

NAME=unspoken-vllm
MODELS_URL="http://127.0.0.1:$PORT_VLLM/v1/models"
GPU_UTIL="${VLLM_GPU_UTIL:-0.40}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
MAX_SEQS="${VLLM_MAX_SEQS:-4}"
LOAD_FORMAT="${VLLM_LOAD_FORMAT:-fastsafetensors}"

status() {
  if [ "$DRY_RUN" = 1 ]; then printf '+ curl -fsS %s\n' "$MODELS_URL" >&2; return 0; fi
  if curl -fsS -m 3 "$MODELS_URL" 2>/dev/null; then echo
    if docker_running nemoclaw-vllm; then ok "vLLM on :$PORT_VLLM is the NemoClaw-managed container (nemoclaw-vllm)"
    elif docker_running "$NAME"; then ok "vLLM on :$PORT_VLLM is $NAME"
    else ok "something answers on :$PORT_VLLM"; fi
    return 0
  fi
  fail "no vLLM on :$PORT_VLLM"; return 1
}

stop() { docker_rm_if_exists "$NAME"; }

start() {
  need docker
  if docker_running nemoclaw-vllm; then
    ok "nemoclaw-vllm is running; not starting $NAME (NemoClaw owns :$PORT_VLLM)"; status; return
  fi
  if docker_running "$NAME"; then ok "$NAME already running"; status; return; fi
  [ "$DRY_RUN" = 1 ] || ! port_open "$PORT_VLLM" || die "port $PORT_VLLM is taken by something else"
  docker_rm_if_exists "$NAME"
  mkdirs
  [ "$DRY_RUN" = 1 ] || hf_cached "$LLM_MODEL" || die "$LLM_MODEL not in HF cache $UNSPOKEN_HF_HOME (run deploy/usb/restore.sh)"
  local spec=()
  if [ "${VLLM_MTP:-1}" = 1 ]; then
    spec=(--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}')
  fi
  # shellcheck disable=SC2046,SC2086
  run docker run -d --name "$NAME" --restart always --network host $(gpu_flag) --ipc host \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
    -v "$UNSPOKEN_HF_HOME:/hf" -v "$UNSPOKEN_LOGS:/logs" \
    "$IMAGE_VLLM" \
    vllm serve "$LLM_MODEL" --host 127.0.0.1 --port "$PORT_VLLM" --served-model-name "$LLM_MODEL" \
      --tensor-parallel-size 1 --trust-remote-code --kv-cache-dtype fp8 \
      --attention-backend flashinfer --moe-backend marlin \
      --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_LEN" \
      --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens 8192 \
      --enable-chunked-prefill --async-scheduling --enable-prefix-caching \
      "${spec[@]+"${spec[@]}"}" \
      --load-format "$LOAD_FORMAT" --reasoning-parser qwen3 \
      --tool-call-parser qwen3_xml --enable-auto-tool-choice ${VLLM_EXTRA:-}
  # Model load from NVMe is 1-3 min; first start also compiles kernels (longer).
  wait_http "$MODELS_URL" "${VLLM_WAIT:-900}" "vLLM" || { docker logs --tail 60 "$NAME" >&2 || true; exit 1; }
  if [ "$DRY_RUN" != 1 ]; then
    log "smoke: one chat completion"
    curl -fsS -m 120 "http://127.0.0.1:$PORT_VLLM/v1/chat/completions" -H 'content-type: application/json' \
      -d "{\"model\":\"$LLM_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":8}" \
      | head -c 400 >&2; echo >&2
  fi
  status
}

case "$ACTION" in start) start ;; stop) stop ;; status) status ;; esac
