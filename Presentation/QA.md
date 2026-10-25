# LocoCisco — Submission Answers

> Draft answers to the hackathon submission form. Technical answers are grounded
> in `BUILD_SPEC.md`. Items marked **[TODO: team]** need a human value (links,
> ratings, personal feedback) before submitting.

**How we map to the judging criteria (25% each):**

- **Technical execution** — a working local agent loop: perception → attributed
  utterance → decision cascade → deterministic policy → action, with MCP tools.
- **Usefulness & business value** — front-line business and government teams get
  an auditable record and actions from everyday conversation; clear owner, clear
  workflow (see Q2, Q5).
- **Local-first design** — all reasoning and tool selection run on the GB10; no
  cloud LLM calls in the runtime path (see Q5, Q10–Q12).
- **Demo quality** — a browser presenter view shows live local inference on a
  prerecorded meeting; <2-minute walkthrough (see Q4).

---

## 1. Project name

**LocoCisco**

## 2. One-line pitch

A wake-word-free, local-first conversation agent for front-line business and
government teams: it understands live multi-speaker meetings, attributes and
translates who said what, and turns commitments and decisions into actions —
all running on the Dell Pro Max (GB10) with no cloud LLM calls.

## 3. Repository URL

`https://github.com/NaichuanZhang/LocoCisco`

- **[TODO: team]** Confirm the repo is public (or shared with judges) and that the
  work is on a branch judges can see. Current work is on the `presentation`
  branch; merge to `main` or point judges at the branch.

## 4. Demo video URL

**[TODO: team]** Public link to the <2-minute walkthrough. The rendered file is
`Presentation/Demo.mp4`; upload it somewhere publicly viewable (YouTube unlisted,
Drive public, etc.) and paste the link here. Script: `Presentation/DEMO_VIDEO.fountain`.

## 5. What you built, and what runs locally

**What it is.** A meeting-room / service-desk agent that listens without a wake
word, understands multi-speaker conversation, attributes speakers across
languages, and converts commitments and decisions into actions — shown in a
browser presenter view.

**Agent workflow (end to end):**

1. **Capture** — audio + video. The local models still process them live.
2. **Perception** — Silero VAD → Parakeet ASR → WeSpeaker CAM++ speaker
   attribution, with YuNet face tracking and Light-ASD active-speaker detection →
   an attributed utterance (text, speaker, seat, timestamp).
3. **Decision cascade** — a bounded local decider (Strands Decider 12B) handles
   confident calls; it escalates to the local Qwen LLM (via vLLM) only when
   uncertain.
4. **Deterministic policy gate** — runs before any external post or device
   action: act / ask owner / deny / ignore.
5. **Act & record** — creates tasks, drives LAN devices (WiZ lamp, Kasa fan), and
   posts policy-permitted text to a sandboxed Discord agent. A browser UI shows
   captions, tasks, the decision panel, privacy state, and the egress meter.

**MCP tool surface:** `room_status`, `checkin`, `transcript`, `list_items`,
`recap`, `search_docs`, `create_task`, `set_light`, `pause_notes`,
`resume_notes`, `undo_item`, `decide_request`.

**What runs locally (local-first confirmation):** **all inference — speech
recognition, speaker/face/active-speaker perception, the decision model, and the
escalation LLM — runs on the GB10.** There are **no cloud LLM calls in the
runtime path**, no automatic model downloads, and no cloud fallback for meeting
content. Core inference, browser playback, and recording work with no internet.
The only external egress is optional, text-only, policy-gated Discord
notifications — not inference.

## 6. Walk us through getting started with the DGX Spark (Dell Pro Max + GB10)

**[TODO: team — confirm against what you actually did; feedback goes to NVIDIA.]**

- **Hardware/OS:** NVIDIA GB10 with CUDA + NVIDIA container support; ARM64
  Ubuntu 24.04-compatible; 128 GB unified memory; Python 3.12.
- **Services (host loopback):** main LLM via vLLM on `:8000`, decider on `:8765`,
  ASR on `:8770`, optional TTS on `:8771`, browser demo/app on `:8780`, room MCP
  on `:8791` (bearer), MCP TLS proxy on `:8443` (sandbox bridge).
- **Offline-first setup:** resolve every model to an already-downloaded cache or
  a staged offline archive; reuse the existing authenticated vLLM endpoint;
  configure Hugging Face / Transformers for offline / local-files-only; warm
  models from local storage before showing "ready."
- **NVIDIA feedback (candidates):** the staged ARM64 GPU wheel set lacked some
  required CUDA packages; validating a complete ARM64 dependency closure up front
  would have saved time. **[TODO: team — add specifics.]**

## 7. Pitch deck

**[TODO: team]** Public link to the deck. File: `Presentation/LocoCisco_Presentation.pptx`
(11 slides). Export to PDF or Google Slides and paste a link judges can open
without screen share.

## 8. Which resources helped most, and what do you wish you'd had access to?

**[TODO: team — edit to reflect reality.]** Helped most: the NVIDIA model catalog
/ NGC container images, vLLM for serving the main model, and the local offline
model kit. Wished for: a pre-validated ARM64 CUDA wheel set and clearer GB10
unified-memory profiling under concurrent load.

## 9. What would have got you there faster?

**[TODO: team.]** A complete, checksummed ARM64 offline dependency bundle; a
ready authenticated vLLM endpoint; and a pre-cleared, consented sample meeting
video so we didn't have to source a fixture.

## 10. What model(s) did you move forward with, and for what?

| Model (size) | Role |
|---|---|
| **Qwen3.6 35B-A3B (NVFP4) via vLLM** | Main language model — escalation reasoning, recap, document Q&A |
| **Strands Decider 12B** over **Gemma 4 12B** | Bounded decisions (act / ask / deny / ignore) |
| **Parakeet TDT 0.6B v3 (Q8, NeMo)** | Speech recognition |
| **WeSpeaker CAM++** | Voice / speaker attribution |
| **Silero VAD** | Voice-activity detection |
| **YuNet** + **Light-ASD (AVA)** | Face tracking + active-speaker detection |
| **MagpieTTS + NanoCodec** (optional) | Speech synthesis |

**[TODO: team]** Adjust if you swapped any role (e.g. a Nemotron/Llama variant).

## 11. What drove your model choices?

Local-first operability on the GB10 inside the unified-memory budget (planning
estimate ~48 GiB main LLM + ~27 GB decider + ~3 GB speech), quantization
(NVFP4 / Q8) to fit, and a **role-based cascade** — a small decider for the many
bounded calls, the large LLM only on escalation — to keep latency and memory in
check. We reuse existing local endpoints and reject hosted inference, per the
local-first requirement.

## 12. How did you get set up with your models?

Resolved each role against the existing local inventory / offline kit (no venue
Wi-Fi downloads); reused the already-running authenticated vLLM endpoint;
restored decider, ASR, and perception assets from local archives into their
runtime caches; set offline / local-files-only loader flags; verified revisions
and checksums; and warmed the required models from local storage before the
"ready to record" indicator. **[TODO: team — confirm exact steps.]**

## 13. Did you use NVIDIA NemoClaw and/or OpenShell today?

**Yes.** The Discord-facing agent (OpenClaw) runs **inside NemoClaw / OpenShell**
as a sandbox. It holds the external identity and calls the room's MCP tools over
a bearer-authenticated TLS proxy; it has **no direct access** to local media or
models and must query the room rather than retaining the meeting record itself.

## 14. If you used NemoClaw/OpenShell, how likely are you to recommend it (1–10)?

**[TODO: team]** _(e.g. 8)_

## 15. Please share why or provide additional feedback.

**[TODO: team.]** Draft: the sandbox + MCP-over-TLS bridge made it clean to keep
the external agent isolated from the sensitive room process, which fit our
local-first/privacy model well. Note any friction (CA trust, proxy/port config,
gateway naming).

## 16. If you used NemoClaw/OpenShell, how secure did you feel building with it (1–10)?

**[TODO: team]** _(e.g. 8)_ — rationale: the sandbox isolation plus the
bearer-auth, TLS-validated boundary between the agent and the room gave us
confidence that the external integration couldn't reach local inference or media.

## 17. What influenced your rating? (optional)

**[TODO: team]** Concerns, positive experiences, or other feedback.

---

## Still needed before submitting

- **[TODO]** Demo video public link (Q4), pitch deck public link (Q7), repo public/accessible (Q3).
- **[TODO]** Team ratings and free-text feedback (Q8, Q9, Q14–Q17).
- **[TODO]** Confirm the exact model set and setup steps against what actually ran (Q6, Q10, Q12).
- **[TODO]** Project name, one-line pitch, and deck/contact slide names if different from the drafts above.
