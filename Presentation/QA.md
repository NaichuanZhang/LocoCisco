# LocoCisco — Submission Answers

> Draft answers to the hackathon submission form, grounded in the application
> source (`unspoken/`, `config/models.toml` `mac` profile, `deploy/demo/`,
> `docs/BROWSER_DEMO.md`). Items marked **[TODO: team]** need a human value
> (links, ratings, personal feedback) before submitting.

**How we map to the judging criteria (25% each):**

- **Technical execution** — a working local agent loop: sensors → VAD/ASR →
  speaker + face + active-speaker fusion → attributed utterance → Decider→LLM
  cascade → deterministic policy → room actions, exposed over MCP tools.
- **Usefulness & business value** — turns everyday meetings/service conversations
  into attributed captions, tasks, decisions and actions, with an auditable
  record; clear owner and workflow (Q2, Q5).
- **Local-first design** — the `mac` profile is `local_only = true`: it refuses
  any non-local endpoint and the Bedrock provider outright. All reasoning, tool
  selection, speech, and vision run on the Mac Studio; no cloud LLM calls and no
  cloud fallback in the runtime path (Q5, Q10–Q13).
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
entirely on the Mac Studio (Apple M1 Max), with no cloud LLM calls.

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
3. **Decision cascade** — the **Strands Decider 2B** gives a fast first opinion;
   at confidence ≥ 0.60 the room acts, below 0.40 it ignores, and in between it
   escalates to the local **Qwen3-8B (4-bit)** LLM (tool-calling) on LM Studio.
4. **Deterministic policy** — plain code, not the model, enforces reflexes,
   permissions, media-origin rules, and the "off the record" wipe (60 s) before
   any output.
5. **Actions** — meeting tasks, the room screen (`:8780`), local device previews
   (WiZ lamp / Kasa fan), and policy-permitted Discord posts via a sandboxed
   OpenClaw bot. The Discord agent reaches the room only through an MCP server
   (`room_status`, `checkin`, `create_task`, `set_light`, `recap`, …).

**What runs locally (local-first confirmation).** The `mac` model profile is
`local_only = true` — it refuses any endpoint not on the box or its LAN and
refuses the Bedrock provider outright. **All inference runs on the Mac Studio:** speech
(Parakeet), VAD (Silero), speaker/face/active-speaker (WeSpeaker, YuNet,
Light-ASD), the Decider 2B, and the escalation LLM (Qwen3-8B on local LM Studio). There
are **no cloud LLM calls** and **no cloud fallback**: the browser demo "fails when
preparation is missing instead of downloading," and inference failures stop replay
visibly rather than falling back. The only external egress is optional,
policy-gated, text-only Discord notifications — not inference.

## 6. Walk us through getting started with the Mac Studio (Apple M1 Max)

**[TODO: team — confirm against what you actually did.]**

- **Profile:** `UNSPOKEN_PROFILE=mac`; verify with
  `python -m unspoken.models check --profile mac`.
- **Bring the stack up:** in LM Studio, load **Qwen3-8B (4-bit)** and start its
  local OpenAI server on `http://localhost:1234/v1`; start the decider with
  `strands-decider serve` (Strands Decider 2B on `:8765`). The sensing models
  (Silero VAD, Parakeet via onnx-asr, WeSpeaker CAM++, YuNet, Light-ASD) run in
  process on the CPU — there is no separate speech server. LM Studio loads the
  model once (reused if already loaded), then the decider starts; in-process
  sensing is ready quickly. TTS and the whiteboard VLM are disabled on the Mac
  target, and there is no second decider at 32 GB. **Everything binds loopback.**
- **Browser demo:** set `UNSPOKEN_LOCAL_API_KEY_FILE` to the credential **file
  path** (never the secret itself), run `deploy/demo/up.sh`, open
  `http://127.0.0.1:8780/`. It reuses the already-loaded LM Studio model and
  starts/reuses the decider and browser processes; the sensing models run in
  process on the CPU. Seven required roles must pass local probes before a take;
  TTS, a second decider, and external integrations are not prerequisites.
- **No startup downloads or installs** — models come from the local offline kit
  (`docs/DOWNLOAD.md`, `docs/BROWSER_DEMO.md`).
- **Setup feedback (candidate):** the macOS Python/venv dependency setup and
  getting LM Studio serving the model were the main friction. **[TODO: team.]**

## 7. Pitch deck

**[TODO: team]** Public link to the deck. The **canonical deck** is the animated
HTML presentation at `Presentation/index.html` (five slides; talk track in
`PRESENTER_NOTES.md`) — host it so judges can open it by link (e.g. GitHub Pages
on the public repo, or export to PDF). The `LocoCisco_Presentation.pptx` is a
legacy/optional leave-behind, not the submission deck.

## 8. Which resources helped most, and what do you wish you'd had access to?

**[TODO: team — edit to reflect reality.]** Helped most: LM Studio (so the Qwen
model was quick to serve over its OpenAI endpoint), onnx-asr for Parakeet, and the
Strands Decider service. Wished for: clearer 32 GB unified-memory profiling under
concurrent load on Apple Silicon.

## 9. What would have got you there faster?

**[TODO: team.]** A complete, checksummed macOS (Apple Silicon) offline dependency
bundle; LM Studio with the Qwen model ready on arrival; and the consented demo clip
staged earlier so we didn't block on a fixture.

## 10. What model(s) did you move forward with, and for what?

| Model (size) | Role | Where |
|---|---|---|
| **Strands Decider 2B** | First-opinion bounded decisions (act / escalate / ignore) | Local service (CPU/Metal), `:8765` |
| **Qwen3-8B (4-bit)** on LM Studio | Escalation LLM (tool-calling) | LM Studio (Metal), `:1234` |
| **Parakeet TDT 0.6B v3** (onnx-asr) | Speech recognition | CPU, in process |
| **WeSpeaker CAM++** (ONNX) | Speaker diarization — session-only voice labels | CPU |
| **Silero VAD v5** (ONNX) | Voice-activity segmentation (400 ms / 8 s) | CPU |
| **YuNet 2023mar** (OpenCV) | Face detection — no recognition | CPU |
| **Light-ASD** (AVA weights, MIT) | Active-speaker detection, 1 s window, 5 Hz | CPU |
| TTS (MagpieTTS) · whiteboard VLM | Disabled on the Mac target | — |
| Hashed bag-of-words (placeholder) · Regex PII | Retrieval embeddings · sensitive-text spans | CPU |

No second decider at 32 GB: the 2B is the only Decider on the Mac target.

## 11. What drove your model choices?

A **role-based cascade** so the LLM runs only when needed: the Decider 2B
(fast, on CPU/Metal) handles most calls, and Qwen is invoked only on uncertain
cases. **Local-first** is enforced by the `mac` profile's `local_only = true`
(non-local endpoints and Bedrock are refused). A single small Qwen on LM Studio
handles tool-calling chat; the whiteboard VLM is disabled on the Mac target. The
perception models were chosen for a sensible CPU/Metal split that fits the 32 GB
unified-memory budget and runs fully offline. Thresholds (act ≥ 0.60) are from our
own eval (`eval/MEETING_RESULTS.md`).

## 12. How did you get set up with your models?

Each **role** in `config/models.toml` resolves to a local endpoint; we selected the
`mac` profile and verified with `python -m unspoken.models check --profile mac`.
LM Studio serves the small Qwen on `:1234` and `strands-decider serve` hosts the
2B on `:8765`; `deploy/demo/up.sh` then brings the browser demo up with health
checks, **reusing** the already-loaded LM Studio model instead of launching a
duplicate. Models are restored from the local offline kit — **no startup
downloads**; a missing asset fails with an actionable error rather than fetching or
faking. **[TODO: team — confirm exact steps.]**

## 13. Did you use NVIDIA NemoClaw and/or OpenShell today?

**[TODO: team — revisit for the Mac Studio build.]** The language model now runs
on **LM Studio** (not a NemoClaw-managed vLLM), so the earlier LLM-serving use of
NemoClaw no longer applies. If the **Discord-facing agent** is still run as a
sandboxed OpenClaw bot, it reaches the room only through an MCP server over a
bearer-authenticated, TLS-protected boundary (loopback MCP on `:8791`); confirm
whether that sandbox path is used on this target.

## 14. If you used NemoClaw/OpenShell, how likely are you to recommend it (1–10)?

**[TODO: team]** _(e.g. 8)_

## 15. Please share why or provide additional feedback.

**[TODO: team.]** Draft: the sandbox + MCP-over-TLS bridge made it clean to keep
the external Discord agent isolated from the sensitive room process. On this
target the LLM runs on LM Studio rather than a NemoClaw-managed vLLM, so revisit
any NemoClaw/OpenShell-specific feedback. Note any friction (CA trust, forwarded
gateway ports, sandbox naming).

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
- **[TODO]** Confirm exact setup steps and setup feedback (Q6, Q12).
- **[TODO]** Project name + one-line pitch wording.
