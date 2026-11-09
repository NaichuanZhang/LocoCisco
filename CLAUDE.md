# CLAUDE.md

Guidance for AI agents working in this repo. Keep changes consistent with
`README.md`, `DESCRIPTION.md`, and `BUILD_SPEC.md`; don't contradict them.

## What this is

LocoCisco is a local-first, privacy-driven conversation-intelligence prototype.
All inference in the demos runs locally (no cloud fallback in the meeting demo).
There are **two separate apps** that share config and core utilities:

- **`unspoken/`** — the current target: a **meeting-room assistant** for the Mac
  Studio (Apple M1 Max, 32 GB). Replays a local video through real speech/vision
  models, produces captions, proposes meeting items, and answers grounded
  questions. This is the main build. The package is named `unspoken` for
  deployment/compatibility reasons (not `lococisco`).
- **`homedemo/`** — an **earlier, standalone smart-home demo** with its own
  dev/Bedrock story (3D house, 59 scripted scenarios, cascade decider). Legacy
  and separate from the meeting target. It has an optional cloud path (Claude
  Haiku 4.5 on Amazon Bedrock); the meeting demo does not.

Both run bounded "Decider" choices first and escalate to a language model only
when uncertain. Deterministic code (not the model) owns safety and permissions.

## Tech stack

- Python **>=3.11** (dev venv uses 3.12). Package manager: **`uv`**.
- FastAPI + uvicorn (web/HTTP), pydantic v2, httpx.
- `strands-agents` (Decider agent + `DeciderToolRouter` intervention), boto3
  (Bedrock, homedemo only).
- Sensing models (meeting demo, `mac` profile, all CPU/in-process): Silero VAD
  (ONNX), Parakeet TDT via onnx-asr, WeSpeaker CAM++ (ONNX), YuNet (OpenCV),
  Light-ASD (PyTorch). The LLM (Qwen3-8B) is served by **LM Studio**; the
  Decider 2B by a local **`strands-decider`** service.
- No lint/format/type tooling is configured. No Docker in the demos (native only).

## Commands (verified)

```bash
# Dev environment + tests (no models/GPU/Mac needed):
scripts/dev-venv.sh            # uv venv .venv (py3.12) + install requirements-dev.txt
.venv/bin/python -m pytest     # or `pytest`; pyproject sets testpaths=tests, asyncio_mode=auto

# Meeting demo (Mac Studio target). Requires LM Studio @:1234 and strands-decider @:8765 already up:
export UNSPOKEN_PROFILE=mac
deploy/demo/up.sh              # orchestrates checks + deploy/demo/run.sh -> app on 127.0.0.1:8780
python -m unspoken.meeting.demo --host 127.0.0.1 --port 8780 --video <path>   # direct entry

# Decider service (native): deploy/mac/decider.sh 2b   ->  strands-decider serve on :8765
# Local llama.cpp LLM for the dev-local profile: scripts/dev-llm.sh start|stop|status  (:8010)

# Legacy smart-home demo:
python -m homedemo [--port 8780] [--mode cascade|2b|12b|fake] [--escalate bedrock|12b|llm] [--profile dev|dev-local|mac|fake]
python -m homedemo.play [ids...]     # run library scenarios through the full runtime

# Presentation (no models): python3 -m http.server 8890 --bind 127.0.0.1  -> /Presentation/
```

## Architecture

Meeting pipeline (see `README.md` diagram for the canonical version):

```
local audio+video -> speech specialists (VAD/ASR/diarization)
                  -> vision specialists (faces/active-speaker)
                  -> speaker fusion (who said what) -> captions (immediate)
                                                    -> Decider 2B (bounded choice)
   Decider >=0.60 -> policy directly
   0.40..<0.60    -> local Qwen (LM Studio) for reasoning/proposals -> policy
   <0.40          -> policy (normally ignores)
   policy (code: privacy/ownership/risk) -> meeting items / screen state / previews
```

- `unspoken/core/` — event `bus`, `decider` HTTP client, `events`, `router`.
- `unspoken/meeting/` — the meeting runtime, agents, cascade, policy, reflex,
  fusion, web server, and `demo.py` (the browser demo entrypoint).
- `unspoken/models/` — model registry/loader (`load_registry`), providers,
  netpolicy, pii. `unspoken/sensors/`, `unspoken/transcribe/`,
  `unspoken/devices/`, `unspoken/bridge/` — supporting workstreams.
- `homedemo/` — `bus`, `runtime`, `agents`, `decider`, `sim`, `scenarios`,
  `web`; entrypoints `__main__.py` and `play.py`.
- `config/models.toml` — central registry of **model roles** (decider,
  decider_big, llm, vlm, asr, vad, diarization, faces, asd, tts, embed, pii) by
  **profile** (`dev`, `dev-local`, `dev-sensors`, `dev-best`, `mac`, `fake`).
  Other config: `meeting.toml`, `room.toml`, `sensors.toml`, `bridge.toml`.
- `deploy/` — launch scripts per target: `demo/` + `mac/` (the Mac Studio
  meeting demo), plus `gb10/`, `usb/`, `nemoclaw/`.
- `requirements/<workstream>.txt` — pinned per-workstream runtime deps, merged
  by the integrator. `requirements-dev.txt` is enough for the meeting brain + tests.
- `tests/` mirrors the source layout (meeting, models, sensors, agents, …).

## Conventions & gotchas

- **Package name is `unspoken`, product name is LocoCisco.** Don't rename.
- **`gb10` profile was renamed `mac`** in `config/models.toml`. Treat `mac` as
  the Mac Studio meeting target; some legacy text may still say `gb10`.
- **Select the profile via `UNSPOKEN_PROFILE`** (or `--profile`). The meeting
  demo uses `mac`; use `fake` for deterministic, model-free runs.
- **Web UIs bind 127.0.0.1 only and have no auth by design** (loopback). View
  remotely via `ssh -L 8780:127.0.0.1:8780 <host>`.
- **Demos don't install or download anything.** `up.sh`/`run.sh` assume the venv
  exists, LM Studio is serving Qwen at `:1234`, and `strands-decider` is on PATH
  / running at `:8765`. They fail fast with instructions if not.
- **Runtime paths** come from env: `LOCOCISCO_RUNTIME` (default `../lococisco-runtime`),
  `UNSPOKEN_HF_HOME`/`HF_HOME`, `UNSPOKEN_STATE`, `UNSPOKEN_LOGS`. Weights,
  credentials, and recordings live outside Git (see `.gitignore`).
- **Privacy:** off-record masks image, mutes audio, discards protected text; it
  cannot erase prior input files/recordings. The decider never owns safety.
- Model/attribution/provenance details: `docs/MODELS.md`, `docs/SOURCE_PROVENANCE.md`,
  `docs/BROWSER_DEMO.md`, `docs/TRANSCRIBE.md`.
