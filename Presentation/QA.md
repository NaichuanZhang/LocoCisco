# LocoCisco — Submission Answers

> Draft answers to the hackathon submission form, grounded in the application
> source (`unspoken/`, `config/models.toml` gb10 profile, `deploy/gb10/`,
> `docs/BROWSER_DEMO.md`). Items marked **[TODO: team]** need a human value
> (links, ratings, personal feedback) before submitting.

**How we map to the judging criteria (25% each):**

- **Technical execution** — a working local agent loop: sensors → VAD/ASR →
  speaker + face + active-speaker fusion → attributed utterance → Decider→LLM
  cascade → deterministic policy → room actions, exposed over MCP tools.
- **Usefulness & business value** — turns everyday meetings/service conversations
  into attributed captions, tasks, decisions and actions, with an auditable
  record; clear owner and workflow (Q2, Q5).
- **Local-first design** — the `gb10` profile is `local_only = true`: it refuses
  any non-local endpoint and the Bedrock provider outright. All reasoning, tool
  selection, speech, and vision run on the GB10; no cloud LLM calls and no cloud
  fallback in the runtime path (Q5, Q10–Q13).
- **Demo quality** — a browser recording canvas plays the official clip through
  real local inference, showing captions, the decision route, probabilities and
  policy outcomes (Q4).

---

## 1. Project name

**LocoCisco**

## 2. One-line pitch

A local-first AI voice agent for meeting rooms and service desks: it hears who
said what without a wake word, decides what to do with an on-device
Decider→LLM cascade, and turns talk into tasks, decisions and room actions —
entirely on the Dell Pro Max (GB10), with no cloud LLM calls.

## 3. Repository URL

`https://github.com/NaichuanZhang/LocoCisco` — application source is under
`unspoken/` (the `homedemo/` tree is the earlier smart-home prototype).

- **[TODO: team]** Confirm the repo is public/accessible to judges. Code is on
  `main` and `presentation` (same commit).

## 4. Demo video URL

**[TODO: team]** Public link to the <2-minute walkthrough.

- The app's demo is the browser recording canvas at `http://127.0.0.1:8780/`,
  playing the official 30-second input clip `Presentation/Demo.mp4` through the
  real local models. Record that canvas running end to end and upload the
  screen-capture to a public link (the script outline is `Presentation/DEMO_VIDEO.fountain`).
- Note: `Demo.mp4` is the **input footage**, not the walkthrough; the submission
  wants a recording of the agent working.

## 5. What you built, and what runs locally

**What it is.** LocoCisco (`unspoken/`) is a wake-word-free meeting/room agent.
For the demo, the official video mocks the mic and camera; the local models still
process it live. Any names/translations baked into the source video are source
content, separate from the dashboard's own local inference.

**Agent loop (end to end):**

1. **Sensors** — audio + video frames (from the clip for the demo; USB devices in
   production). Seat zones map faces to chairs; a "screen" zone marks the display
   so a talking face there is treated as media, not a person.
2. **Perception** — **Silero VAD** segments speech → **Parakeet TDT** ASR
   produces captions → **WeSpeaker CAM++** assigns session-only voice labels, with
   **YuNet** faces and **Light-ASD** active-speaker detection → an attributed
   utterance (text, voice label, seat, optional checked-in name; unknown = Guest).
3. **Decision cascade** — the **Strands Decider 12B** gives a fast first opinion;
   at confidence ≥ 0.60 the room acts, below 0.40 it ignores, and in between it
   escalates to the local **Qwen3.6 35B-A3B** LLM (tool-calling) on vLLM.
4. **Deterministic policy** — plain code, not the model, enforces reflexes,
   permissions, media-origin rules, and the "off the record" wipe (60 s) before
   any output.
5. **Actions** — meeting tasks, the room screen (`:8780`), local device previews
   (WiZ lamp / Kasa fan), and policy-permitted Discord posts via a sandboxed
   OpenClaw bot. The Discord agent reaches the room only through an MCP server
   (`room_status`, `checkin`, `create_task`, `set_light`, `recap`, …).

**What runs locally (local-first confirmation).** The `gb10` model profile is
`local_only = true` — it refuses any endpoint not on the box or its LAN and
refuses the Bedrock provider outright. **All inference runs on the GB10:** speech
(Parakeet), VAD (Silero), speaker/face/active-speaker (WeSpeaker, YuNet,
Light-ASD), the Decider 12B, and the escalation LLM (Qwen3.6 on local vLLM). There
are **no cloud LLM calls** and **no cloud fallback**: the browser demo "fails when
preparation is missing instead of downloading," and inference failures stop replay
visibly rather than falling back. The only external egress is optional,
policy-gated, text-only Discord notifications — not inference.

## 6. Walk us through getting started with the DGX Spark (Dell Pro Max + GB10)

**[TODO: team — confirm against what you actually did; feedback goes to NVIDIA.]**

- **Profile:** `UNSPOKEN_PROFILE=gb10`; verify with
  `python -m unspoken.models check --profile gb10`.
- **Bring the stack up:** `deploy/gb10/up.sh` starts servers in memory order with
  health checks — vLLM Qwen3.6 on `:8000` (reuses the NemoClaw-managed
  `nemoclaw-vllm` if already running), Strands Decider 12B on `:8765`, ASR on
  `:8770`, optional TTS on `:8771`, optional 2B on `:8766`. Typical cold start:
  vLLM 2–4 min (longer on first kernel compile), decider 1–3 min, speech < 30 s.
  **Everything binds `127.0.0.1`.**
- **Browser demo:** set `UNSPOKEN_LOCAL_API_KEY_FILE` to the credential **file
  path** (never the secret itself), run `deploy/demo/up.sh`, open
  `http://127.0.0.1:8780/`. It reuses the existing Qwen service and loaded GPU
  image and starts/reuses the Parakeet, Decider and browser containers. Seven
  required roles must pass local probes before a take; TTS, a second decider, and
  external integrations are not prerequisites.
- **No startup downloads or installs** — models come from the local offline kit
  (`docs/DOWNLOAD.md`, `docs/BROWSER_DEMO.md`).
- **NVIDIA feedback (candidate):** the ARM64 GPU dependency closure was the main
  friction; a pre-validated CUDA wheel set would have saved time. **[TODO: team.]**

## 7. Pitch deck

**[TODO: team]** Public link to the deck. The **canonical deck** is the animated
HTML presentation at `Presentation/index.html` (five slides; talk track in
`PRESENTER_NOTES.md`) — host it so judges can open it by link (e.g. GitHub Pages
on the public repo, or export to PDF). The `LocoCisco_Presentation.pptx` is a
legacy/optional leave-behind, not the submission deck.

## 8. Which resources helped most, and what do you wish you'd had access to?

**[TODO: team — edit to reflect reality.]** Helped most: the NemoClaw-managed vLLM
(so the Qwen service was already serving), NeMo-Speech.cpp for Parakeet/MagpieTTS,
and the Strands Decider servers. Wished for: a pre-validated ARM64 CUDA wheel set
and clearer GB10 unified-memory profiling under concurrent load.

## 9. What would have got you there faster?

**[TODO: team.]** A complete, checksummed ARM64 offline dependency bundle; the
authenticated vLLM endpoint ready on arrival; and the consented demo clip staged
earlier so we didn't block on a fixture.

## 10. What model(s) did you move forward with, and for what?

| Model (size) | Role | Where |
|---|---|---|
| **Strands Decider 12B** | First-opinion bounded decisions (act / escalate / ignore) | GPU, `:8765` |
| **Qwen3.6 35B-A3B (NVFP4)** on vLLM | Escalation LLM (tool-calling) **and** VLM (reads the whiteboard) | GPU, `:8000` |
| **Parakeet TDT 0.6B v3** (NeMo-Speech.cpp) | Speech recognition | GPU, `:8770` |
| **WeSpeaker CAM++** (ONNX) | Speaker diarization — session-only voice labels | CPU |
| **Silero VAD v5** (ONNX) | Voice-activity segmentation (400 ms / 8 s) | CPU |
| **YuNet 2023mar** (OpenCV) | Face detection — no recognition | CPU |
| **Light-ASD** (AVA weights, MIT) | Active-speaker detection, 1 s window, 5 Hz | GPU |
| **MagpieTTS Multilingual 357M** (optional) | Speech synthesis | GPU, `:8771` |
| Hashed bag-of-words (placeholder) · Regex PII | Retrieval embeddings · sensitive-text spans | CPU |

Optional fallback: a **Strands Decider 2B** on `:8766`, off by default.

## 11. What drove your model choices?

A **role-based cascade** so the big model runs only when needed: the Decider 12B
(fast on the GPU) handles most calls, and Qwen3.6 is invoked only on uncertain
cases. **Local-first** is enforced by the `gb10` profile's `local_only = true`
(non-local endpoints and Bedrock are refused). One Qwen3.6 server covers both
tool-calling chat and vision (whiteboard), which saves memory. The perception
models were chosen for a sensible CPU/GPU split that fits the unified-memory
budget and runs fully offline. Thresholds (act ≥ 0.60) are from our own eval
(`eval/MEETING_RESULTS.md`).

## 12. How did you get set up with your models?

Each **role** in `config/models.toml` resolves to a local server; we selected the
`gb10` profile and verified with `python -m unspoken.models check --profile gb10`.
`deploy/gb10/up.sh` brings the servers up in memory order with health checks,
**reusing** the NemoClaw-managed vLLM instead of launching a duplicate. Models are
restored from the local offline kit — **no startup downloads**; a missing asset
fails with an actionable error rather than fetching or faking. **[TODO: team —
confirm exact steps.]**

## 13. Did you use NVIDIA NemoClaw and/or OpenShell today?

**Yes — in two places:**

1. **vLLM is NemoClaw-managed** (`nemoclaw-vllm`); our launcher reuses it rather
   than starting its own copy.
2. The **Discord-facing agent runs as a sandboxed OpenClaw bot**: the room posts
   verbatim via `nemoclaw exec … openclaw message send`, and the sandbox reaches
   the room only through an MCP server over a TLS proxy bound to the
   `openshell-docker` bridge (bearer-authenticated, loopback MCP on `:8791`).

## 14. If you used NemoClaw/OpenShell, how likely are you to recommend it (1–10)?

**[TODO: team]** _(e.g. 8)_

## 15. Please share why or provide additional feedback.

**[TODO: team.]** Draft: the sandbox + MCP-over-TLS bridge made it clean to keep
the external Discord agent isolated from the sensitive room process, and reusing
the NemoClaw-managed vLLM avoided a second heavy load. Note any friction (CA
trust, forwarded gateway ports, sandbox naming).

## 16. If you used NemoClaw/OpenShell, how secure did you feel building with it (1–10)?

**[TODO: team]** _(e.g. 8)_ — rationale: sandbox isolation plus the bearer-auth,
TLS-validated boundary between the agent and the room, with `local_only` refusing
any off-box endpoint, gave us confidence the integration couldn't reach local
inference or media.

## 17. What influenced your rating? (optional)

**[TODO: team]** Concerns, positive experiences, or other feedback.

---

## Honest status (for your own reference, not a form field)

Per `README.md` / `BUILD_SPEC.md`: the recording layout, repeated official-video
review captures, and privacy masking/muting are verified; **final native capture
quality and bilingual transcription/intent accuracy still need evaluation.** Keep
claims at that level in the deck and video.

## Still needed before submitting

- **[TODO]** Demo video public link (Q4), pitch deck public link (Q7), repo public/accessible (Q3).
- **[TODO]** Team ratings and free-text feedback (Q8, Q9, Q14–Q17).
- **[TODO]** Confirm exact setup steps and NVIDIA feedback (Q6, Q12).
- **[TODO]** Project name + one-line pitch wording.
