# LocoCisco for meetings: build plan (Mac Studio, 2026-10-25)

Goal: everything except model deployment is built and tested before the day. On the day we only bring
models up on the Mac Studio, run one check command, and rehearse.

## 1. How it fits together

```
 USB speakerphone ─► sensors/audio ── VAD ─► ASR ─► diarization ─┐            (host process: unspoken)
 USB webcam ──────► sensors/video ── faces ─► ASD (who talks) ───┤
 Discord (/here, ✅ ❌, @LocoCisco) ◄─► OpenClaw ── MCP/HTTP ──────┤
                                                                 ▼
                         meeting.utterance (name + text + seat) ── reflex rules (off the record)
                                                                 │
                                   Strands Decider (typed questions) ── sure? ─► proposal (local-decider)
                                                                 │ unsure
                                   Strands agent + Qwen on LM Studio (tools) ─► proposal (local-llm)
                                                                 ▼
                                   policy (plain code: risk, guests, privacy) ─► verdict
                                                                 ▼
             actuator ─► devices (WiZ lamp, Kasa fan) · room screen (HDMI kiosk) · Notifier
                                                                 ▼
             Notifier ─► OpenClaw hooks (inside the NemoClaw/OpenShell sandbox) ─► Discord
```

Two processes matter:

| Process | Where | Job |
|---|---|---|
| `unspoken` | Mac Studio host, binds 127.0.0.1 | Real-time room brain: sensors, decider, policy, devices, room screen, MCP tools |
| OpenClaw agent | NemoClaw sandbox (OpenShell) | Discord face: posts notes/tasks/recaps, answers "@LocoCisco what did we decide?" by calling LocoCisco's tools |

Both use the same local Qwen served by LM Studio. Nothing calls a remote model. The only internet egress is the
sandbox's Discord access, which the OpenShell policy allows and nothing else.

Why the room brain is outside the sandbox: it needs the USB mic, camera and LAN devices, and it must
answer in under a second. OpenClaw is a chat agent loop; it is the right place for Discord, recaps and
questions, not for 5 Hz camera frames.

## 2. NemoClaw plug-in points

| Direction | Mechanism | Notes |
|---|---|---|
| LocoCisco -> Discord | OpenClaw gateway webhook (`/hooks/agent`, deliver to the Discord channel) on the NemoClaw-forwarded gateway port | Verify the exact route and token on the box; bridge workstream owns the client and a recorded fixture |
| Discord -> LocoCisco | OpenClaw agent calls LocoCisco tools: managed MCP server (`nemoclaw <sb> mcp add`) | NemoClaw requires HTTPS on a private (RFC1918) host IP with a bearer token, `--trusted-private-host`, and a CA it trusts. Loopback is rejected. We run a TLS proxy on the docker bridge IP, bound to that interface only |
| Fallback A | OpenClaw skill + custom policy preset to a plain HTTP host API | If managed MCP will not converge |
| Fallback B | `discord-rest` Notifier posts with the bot token from the host | Only if NemoClaw breaks on the day; the egress meter still counts every byte |
| Inference | Qwen served by LM Studio (OpenAI-compatible, `http://localhost:1234/v1`, Chat Completions + tools) | LocoCisco and the OpenClaw agent call the same local LM Studio server |
| Channel | `DISCORD_BOT_TOKEN`, `DISCORD_SERVER_ID`, `DISCORD_REQUIRE_MENTION=1` exported before onboarding | Message Content intent on in the Developer Portal |

Everything above is scripted in `deploy/nemoclaw/` and explained in `docs/NEMOCLAW.md` (bridge workstream).

## 3. Model switching

One file: `config/models.toml`. One variable: `UNSPOKEN_PROFILE` (`dev`, `mac`, `fake`).

```toml
[profiles.mac]
local_only = true                       # refuses any provider that is not on this box or its LAN

[profiles.mac.decider]
provider = "strands-decider"
url = "http://127.0.0.1:8765"
label = "Strands Decider 2B (CPU/Metal)"

[profiles.mac.llm]
provider = "openai"                     # any OpenAI-compatible server: LM Studio, llama.cpp, Ollama
url = "http://localhost:1234/v1"
model = "qwen3-8b"
label = "Qwen3-8B (4-bit) on LM Studio"
```

- Switch a model: edit one block, or override one key: `UNSPOKEN_LLM_MODEL=... UNSPOKEN_LLM_URL=...`.
- Check every role: `python -m unspoken.models check` prints role, provider, target, ok, probe latency.
- Code asks for roles (`reg.decider()`, `reg.strands_model("llm")`, `reg.asr()`), never for vendors.
- The home demo uses the same registry, so it can run fully local too.

Roles: decider, decider_big, llm, vlm, asr, vad, diarization, faces, asd, tts, embed, pii
(see `unspoken/contracts.py`).

## 4. Ports

| Port | Mac | Dev desktop |
|---|---|---|
| 1234 | LM Studio Qwen (OpenAI server, Metal) | — |
| 8010 | — | llama.cpp test server (small Qwen, CPU) for the OpenAI path |
| 8765 | Strands Decider 2B (`strands-decider serve`) | Decider 2B (CPU, already running) |
| 8766 | — (no second decider at 32 GB) | Decider 12B (CPU, already running) |
| 8780 / 8781 | Room screen + API (127.0.0.1) | 8781 |
| 8791 / 8792 | LocoCisco MCP (127.0.0.1) | 8792 |
| 8443 | TLS proxy for MCP on the docker bridge IP only | — |
| 18789 | OpenClaw gateway (NemoClaw forward) | — |

## 5. Repo layout and owners

| Path | Owner | What |
|---|---|---|
| `unspoken/core/` | integrator | events, bus, decider clients, DeciderToolRouter, test doubles |
| `unspoken/contracts.py`, `unspoken/meeting/contracts.py` | integrator | the contracts below everything |
| `unspoken/models/`, `config/models.toml`, `homedemo` wiring | models | registry, HTTP providers, `check` CLI |
| `unspoken/meeting/` (not `web/`) | meeting | agents, decider questions, policy, simulator, scenarios, runtime, CLI |
| `unspoken/meeting/web/` | screen | room screen page + API + dev fake |
| `unspoken/sensors/` | sensors | mic/camera capture, file replay, in-process VAD/ASR/diarization/faces/ASD providers |
| `unspoken/devices/`, `config/room.toml` | devices | WiZ, Kasa, fakes, discovery CLI |
| `unspoken/bridge/`, `deploy/nemoclaw/`, `docs/NEMOCLAW.md` | bridge | Notifiers, MCP server, OpenClaw skill, policy preset, TLS proxy |
| `deploy/demo/`, `docs/HACKDAY.md` | deploy | demo launcher and local media tooling, the day-of runbook |
| `homedemo/` | models (wiring only) | the original smart-home demo, kept working |

Each workstream lists its Python deps in `requirements/<workstream>.txt`; the integrator merges them.

## 6. Day-of runbook (summary; full version in `docs/HACKDAY.md`)

1. Mac preflight: free memory, disk, and LM Studio installed with Qwen3-8B (4-bit) downloaded.
2. macOS venv ready (`uv pip install -r requirements/...`) and model weights cached; no containers.
3. In LM Studio, load Qwen3-8B (4-bit) and start its local server (`http://localhost:1234`); optional
   NemoClaw onboarding with Discord env set for the bridge.
4. `strands-decider serve --model decider-2b --port 8765` (Parakeet and the other sensing models run in
   process on the CPU; no TTS).
5. `UNSPOKEN_PROFILE=mac python -m unspoken.models check` -> all green.
6. `deploy/nemoclaw/connect.sh` (MCP + hooks + policy), if the Discord bridge is being demoed.
7. `UNSPOKEN_PROFILE=mac deploy/demo/up.sh` (or `python -m unspoken.meeting --profile mac --mode live`),
   open the room screen, rehearse the 3-minute demo.

## 7. Risks

- Rules on code written before the event are not published on the Luma page; confirm at kickoff.
- This code started in an Amazon hack-week project; using it at an outside event needs Amazon's
  open-source approval.
- Discord voice receive broke for many bots after Discord's 2026 encryption change; remote voice is a stretch goal.
- WiZ bulbs accept commands with no password; keep them on our own travel router.
- The MCP TLS proxy is the one network-facing piece; it binds to the docker bridge IP only and requires a bearer token.

## 8. Status (2026-10-24 evening, branch hack/main)

Built and merged: all seven workstreams (reports in `docs/reports/`). 840 tests pass, 8 live-only skips.

| Checked on the dev desktop | Result |
|---|---|
| Simulator, keyword decider (`play --profile fake`) | 21/21 scenarios |
| 3-minute demo, real Decider 2B (CPU), no LLM | 17/17 (meeting workstream run) |
| 3-minute demo, real 2B + local Qwen3-4B escalation (`dev-local`) | 16/17: the 4B turned the fan the wrong way for "it's freezing" |
| Room screen with the real runtime | renders captions, tasks, guest notice, recap, egress meter |
| Real mic pipeline on the fixture WAV | 12/12 lines, 3/3 voices |
| MCP server through the TLS proxy, 401 without the token | ok (loopback) |
| `unspoken.models check --profile dev-local` | 11/11 roles |

Must be checked on the Mac Studio (in this order): the Decider 2B via `strands-decider serve`, Qwen3-8B tool
calls on LM Studio (thinking disabled), Parakeet via onnx-asr, mic + camera seen by `unspoken.sensors.check`,
NemoClaw sandbox trusts our CA (`mcp status` = match) if the Discord bridge is used, `verify.sh --post
--agent`, the WiZ lamp and Kasa plug, then `play demo_three_minutes --profile mac` and the live demo. On
32 GB the 2B is the only decider (no 12B); re-check `act_threshold` (0.60) against the 2B on the Mac if the
routing bands need tuning.

Integrator fixes after the merges: nothing said off the record or while paused is kept in the bus history
(`Bus.retain` / `Bus.forget`); the room never escalates to a hosted LLM (`local_llm_enabled`); `--profile fake`
for the meeting = models.toml profile `sim`; comfort is `warmer` / `cooler` for the LLM; the runbook makes the
CA before the NemoClaw installer.
