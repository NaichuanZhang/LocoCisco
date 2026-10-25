# LocoCisco — Build Specification

- **Version:** 0.2
- **Date:** 2026-10-25
- **Status:** specification-first; GB10 integration and browser-demo acceptance testing pending
- **Target:** Dell Pro Max with NVIDIA GB10, ARM64 Linux, 128 GB unified memory

## 1. Product goal

Build a meeting-room assistant that understands ordinary conversation without
requiring a wake word. Speech recognition, speaker attribution, decision-making,
and document lookup run locally. Participants see what the assistant heard,
what it decided, and what information it sent outside the room.

The primary demonstration runs entirely in a browser, using a prerecorded
meeting video instead of a USB microphone and camera. Its audio track mocks
the microphone stream and its video frames mock the camera stream. Only the
capture hardware is mocked: the local perception and decision models must
process the recording during playback.

The demo must be easy to rehearse, reset, and record as a polished video.
Browser presentation and recording are first-class acceptance requirements,
not a final step left to improvised screen capture.

Venue Wi-Fi is not a model-distribution dependency. The build must use model
weights already downloaded on the Dell box or included in its local offline
kit, and reuse compatible model services already running there. No new model
download or cloud inference is permitted as an automatic startup or fallback
step. The core browser demo and recording must work without internet access.

The first public deliverable is this specification. A local prototype is the
starting point for implementation work; publishing this document does not
publish that source or its history.

## 2. Version-one scope

### Required

- Streaming captions from the audio track of a prerecorded meeting video;
  no USB microphone or camera is required for the main demo.
- Session-only voice labels, video-based seat attribution, and explicit participant
  check-in; unknown speakers remain unnamed rather than being guessed.
- Extraction of commitments, owners, due dates, and meeting decisions.
- A local decision cascade: bounded questions to a small decision model,
  followed by a larger local language model only when needed.
- Offline-first startup and inference using already-downloaded model assets,
  cached container images, and locally staged dependencies.
- Deterministic policy checks before any external post or device action.
- A browser-first demo showing the source video alongside captions, participants,
  tasks, privacy state, model health, decision explanations, and text egress.
- Spoken, browser, and Discord controls for pausing and resuming notes.
- A recording-ready presenter mode, synchronized playback controls, clean
  restart, and a tested local recording/export workflow.
- A WiZ lamp and Kasa-controlled fan on a dedicated local network.
- A sandboxed Discord agent that posts permitted text and answers questions
  through the room's tools.
- Video-backed replay as the default demo input, plus a clearly labeled
  simulator mode for deterministic tests.

### Optional after the core demo works

- Live USB microphone and webcam adapters; they must not block the video-backed demo.
- Local text-to-speech.
- Whiteboard or scene interpretation through the local vision-language model.
- A second decision model for fallback or comparison.
- A dedicated embedding model for better local document retrieval.

Optional models may be enabled only when their full artifacts are already
available locally and they fit the resource budget. A new model download
requires a separate, explicit preparation decision; it is not part of this build.

### Out of scope for the first build

- Cloud-model fallback for meeting content.
- Automatic model downloads, container pulls, or runtime dependency downloads
  during application startup, rehearsal, or recording.
- Persistent face recognition, cross-meeting voice identity, or biometric enrollment.
- Discord voice capture, multi-room orchestration, and long-term meeting archives.
- Arbitrary internet browsing or unrestricted tool execution.
- The earlier smart-home simulator as the primary product experience.
- Claims of production compliance, guaranteed transcription accuracy, or
  benchmark performance not measured on the target hardware.

## 3. Target environment

| Resource | Requirement |
|---|---|
| Compute | NVIDIA GB10 with working CUDA and NVIDIA container support |
| Operating system | ARM64 Ubuntu 24.04-compatible environment |
| Python | Python 3.12 for the staged dependency set |
| Audio | Decode the meeting video's audio locally into the microphone-stream contract |
| Video | Decode the same file's frames locally; target approximately 25 source frames/s for Light-ASD |
| Demo fixture | A locally available, consented or appropriately licensed meeting video with an audio track and at least two speakers |
| Display | Browser as the primary demo surface; 16:9 presenter layout, with HDMI optional |
| Recording | Local browser-tab/window capture with verified audio, or a tested OS-recorder fallback |
| Devices | WiZ lamp, Kasa plug/fan, and a dedicated room network |
| Storage | Existing downloaded model assets/offline kit, the pre-staged meeting video, and space for demo recordings |
| Network | No internet required for core inference, browser playback, or recording; LAN device access stays local, and Discord is a separate internet-dependent integration |

Deployment must adapt to an already-provisioned machine. Preserve existing
model services, authentication, and unrelated workloads. Stopping or replacing
an existing service requires explicit operator approval.

The primary path must work with no USB capture devices attached and without
requesting browser camera or microphone permission. Permission to record the
demo tab/window is separate and is requested only after an explicit user action.

## 4. Architecture

```text
Local meeting video -> synchronized replay clock
                         |                     |
                  decoded audio          decoded frames
                         |                     |
                  mock microphone         mock camera
                         |                     |
                 VAD -> ASR -> voices     faces -> active speaker
                         |                     |
                         +---------------------+----> attributed utterance
                                                               |
Browser/Discord check-in and commands -> room tools ------------+
                                                               v
                                                       privacy/reflex rules
                                                               |
                                                       Strands Decider 12B
                                                          /          \
                                                       confident    uncertain
                                                          |             |
                                                          |        local Qwen agent
                                                          +-------------+
                                                               |
                                                        proposed action
                                                               |
                                                     deterministic policy
                                                               |
                                                            actuator
                                                  /            |            \
                                           browser demo    LAN devices    notifier
                                                                              |
                                                                     sandboxed OpenClaw
                                                                              |
                                                                           Discord

Browser demo: synchronized video + captions + tasks + decision panel
             -> presenter mode -> explicit local recording and download
```

### Runtime boundaries

| Component | Responsibility |
|---|---|
| Host room process | Video decoding and paced audio/frame input, event bus, meeting state, decision cascade, policy, devices, browser API, and MCP tools |
| Browser demo | Synchronized source playback, captions and results, operator controls, presenter layout, and recording/export |
| Local model services | Decision model and OpenAI-compatible chat/speech endpoints |
| OpenClaw in NemoClaw/OpenShell | Discord-facing identity, permitted notifications, and calls to room tools |
| Private TLS proxy | Authenticated connection from the sandbox to the host's loopback MCP service |

The room process stays outside the sandbox because it owns local media decoding,
model access, and local-network appliances. Future USB adapters can use the
same input contracts. The Discord agent does not own the meeting record
and must query the room instead of inventing or independently retaining it.

## 5. Model and perception stack

| Role | Initial choice | Placement |
|---|---|---|
| Voice activity detection | Silero VAD | CPU, in process |
| Speech recognition | Parakeet TDT 0.6B v3, Q8 via NeMo-Speech.cpp | Local GPU speech service |
| Voice attribution | WeSpeaker CAM++ with causal window labeling | CPU, in process |
| Face tracking | YuNet | CPU, in process |
| Active speaker detection | Light-ASD with AVA weights | GPU, in process |
| Bounded decisions | Strands Decider 12B over Gemma 4 12B | Local GPU service |
| Language-model escalation | Qwen3.6 35B-A3B NVFP4 via vLLM | Existing local GPU service |
| Vision-language tasks | Same Qwen service, subject to capability verification | Optional |
| Speech synthesis | MagpieTTS and NanoCodec | Optional local service |
| Document retrieval | Local index with hashed bag-of-words baseline | In process; embedding upgrade optional |
| Sensitive-text screening | Regex-based PII checks plus explicit policy | In process; not a complete privacy guarantee |

Model selection belongs in one role-based configuration, not scattered through
application code. The GB10 profile must reject hosted inference providers.
Disabled optional roles must not fail the core readiness check.

### Already-downloaded models and offline operation

- Treat the model choices above as roles to resolve against the existing local
  inventory, not a download list. Check downloaded caches, staged archives,
  adapters, tokenizers, speech assets, and loaded container images before startup.
- Reuse the existing local Qwen/vLLM endpoint for the main language model,
  including its authentication. Do not launch a duplicate copy or replace it
  with a newly fetched model.
- Restore required decider, speech, and perception assets from the local kit
  when they are not yet in their runtime cache. Copying or unpacking these
  existing files is allowed; fetching replacement weights over Wi-Fi is not.
- Resolve exact local model revisions/paths and verify required files and
  checksums. Configure loaders for offline/local-files-only operation, including
  the Hugging Face and Transformers offline settings where applicable.
- Prevent implicit network downloads from model libraries, container launchers,
  tokenizers, package installers, and first-run frontend assets.
- A missing or incompatible required asset must produce a clear error naming
  the asset and the local preparation needed. Do not hang on a download or
  silently substitute a fake or cloud model.
- If a fallback is necessary, propose only already-downloaded, compatible
  alternatives and validate their role/API before switching with operator approval.
- Pre-stage the meeting-video fixture, browser assets, and any required codec
  or recording tools. Do not stream the demo source from an external site.
- Warm required models from local storage before the recording-ready indicator
  is shown. Reuse them across retakes instead of repeatedly loading the weights.

Discord still requires internet. A slow or unavailable connection must not
stall captions, local decisions, source playback, or recording. Mark outbound
messages as pending, failed, or delivered truthfully, with bounded retries and
no duplicate posts. Privacy changes and session reset must cancel pending
actions that are no longer permitted.

The initial confidence thresholds are 0.60 for acting and 0.40 for considering
escalation or approval. These are starting values from the prototype, not
calibrated guarantees for the 12B model. Evaluate them before accepting the demo.

### Resource budget

Start with one main language-model service, one decision model, ASR, and the
required in-process perception models. Keep TTS and the second decider disabled
until the core path passes.

The reference sizing estimates approximately 48 GiB for the main vLLM budget,
27 GB for the 12B decider, and 3 GB for speech. These are planning estimates.
Measure peak unified-memory use with the OS, Light-ASD, sandbox, and any existing
workloads included, including browser playback and active recording; do not
treat the estimates as guaranteed free capacity.

## 6. Interfaces and data flow

### Default service boundaries

| Service | Port | Access |
|---|---|---|
| Main vLLM API | 8000 | Host loopback; preserve authenticated sandbox access where already configured |
| Decider 12B | 8765 | Host loopback |
| Optional second decider | 8766 | Host loopback |
| ASR | 8770 | Host loopback |
| Optional TTS | 8771 | Host loopback |
| Browser demo, local media, and API | 8780 | Host loopback |
| Room MCP service | 8791 | Host loopback, bearer protected |
| MCP TLS proxy | 8443 | Private sandbox bridge interface only, bearer protected |

Treat the OpenClaw gateway address as deployment configuration rather than
assuming the port or sandbox name from a fresh installation.

### Video-backed input contract

- Select a local meeting video through a browser file picker or a preconfigured
  fixture selector. Do not upload it to an external service.
- Decode its audio to the configured ASR sample rate, initially mono 16 kHz PCM,
  and emit paced chunks through the same contract a microphone would use.
- Decode frames from the same media source and emit them through the camera
  contract. Select a fixture near 25 FPS; document frame-rate conversion rather
  than claiming repeated frames are new observations.
- Use one authoritative media clock for audio chunks, frames, browser playback,
  utterances, and result timestamps. Pace the default demo at 1x rather than
  processing the whole file before showing a prerecorded result.
- Expose load, play, pause, stop, restart, elapsed time, and duration in the
  browser. Pausing must stop ingestion and source playback together; clearly
  associate any remaining in-flight results with their original timestamps.
- Restart must clear meeting state, caption/task history, voice/seat bindings,
  queued actions, and stale in-flight results from the previous run.
- Restart does not undo previously confirmed Discord posts or physical device
  changes. Preserve truthful integration status and require explicit re-arming
  of external effects before another take; retries must not duplicate an action.
- Seeking is optional; if offered, rebuild/reset derived state and invalidate
  stale jobs. Do not allow unsynchronized video scrubbing.
- Identify the input visibly as **Prerecorded meeting · live local inference**.
  A synthetic test run must instead be explicitly labeled as simulated.
- Replay replaces capture, not the speaker-source policy: meeting participants
  in the fixture represent people in the room, while a television or other
  media source within that meeting remains subject to the media-origin rules.
- Ground-truth transcripts and expected tasks may be used for evaluation, but
  must not replace model outputs in the main demo.
- Choose a short, demo-safe fixture with speaker turns and the required story
  beats. Track its provenance and usage rights without adding the media to
  this specification-only repository.

### Event contracts

The event bus carries a stable sequence:

1. Audio segments and tracked speaker observations.
2. Attributed utterances with timestamps, text, session voice label, optional
   participant name, and seat.
3. Proposals with action, arguments, confidence, route, model, and latency.
4. Policy verdicts: act, ask the owner, deny, or ignore.
5. Resulting meeting items, device outcomes, privacy changes, and screen state.

All attribution must distinguish evidence from uncertainty. A visible face is
not proof of a person's name; overlapping and off-camera speech must remain
representable without forcing an incorrect identity.

### MCP tool surface

The Discord agent uses room-scoped tools such as `room_status`, `checkin`,
`transcript`, `list_items`, `recap`, `search_docs`, `create_task`, `set_light`,
`pause_notes`, `resume_notes`, `undo_item`, and `decide_request`.

Read `room_status` before answering room questions. Mutating tools create
proposals subject to the same policy as spoken requests. The agent may report
success only after the runtime confirms it.

## 7. Privacy and security requirements

- The prerecorded meeting file stays on the local machine. Decoded audio/frame
  buffers are transient; the existing source file is an intentional input asset,
  not a newly captured recording. Any future live-capture mode follows the same
  local-processing boundary.
- Only policy-permitted text may be sent to Discord. Demo recordings are saved
  locally and must never be automatically uploaded or posted.
- The screen must distinguish note-taking state (notes on, paused, off record)
  from demo capture state (not recording, recording, recording stopped).
- A recognized off-record request must take priority over normal model decisions.
  Do not retain subsequent meeting content in captions, transcript history,
  traces, document context, or queued notifications while that state applies.
- Off-record handling must remove the configured recent local transcript window,
  initially 60 seconds. Do not promise that already-delivered external messages
  have been erased; any external retraction must be separately verified.
- Off-record handling does not delete the prerecorded input file or erase
  footage already captured by a screen recorder. Make that boundary explicit.
  Use only demo-safe, consented or licensed content in the recorded presentation.
- Mask the source preview and mute its audible playback during off-record
  segments so the browser recording does not expose that content. Local,
  transient processing may continue to recognize a resume command, without
  retaining off-record utterances or generating normal actions.
- Resume behavior follows policy, including owner approval after a sensitive pause.
- Guest presence restricts document sharing. Recap publication requires approval.
- Ignore device-control instructions attributed to television or other media.
- Keep identity bindings session-scoped and clear meeting state at session end.
- Require bearer authentication for MCP, validate TLS certificates, and restrict
  the proxy to the sandbox network. Do not weaken authentication to pass checks.
- Supply API credentials and device-account secrets through protected runtime
  configuration; never commit values or print them in diagnostics.
- The egress meter reports application notification bytes. It is not a substitute
  for network-policy enforcement or a whole-machine traffic audit.
- Synthetic fixtures and test doubles must be visibly distinguishable from
  actual hardware and model results.
- Presenter mode must hide credentials, private file paths, setup forms, and
  debugging payloads that could expose sensitive configuration. Starting any
  recording requires an explicit operator action and a visible indicator.

## 8. Browser-demo and recording requirements

The browser is the main demo, not a secondary monitoring screen. The presenter
must be able to run the full story and record it without switching to a terminal
or a separate desktop application during the take.

### Main demo view

- Display the **LocoCisco** name consistently.
- Place the source meeting video beside synchronized captions, participants,
  tasks, decisions, and the current meeting context.
- Make captions, privacy state, and the prerecorded-input label legible at
  the intended recording resolution.
- Provide playback controls, a timeline, clean restart, check-in, room questions,
  and pause/resume-note controls directly in the browser.
- Show actual lamp/fan state and a feed of integration outcomes in the same view.
  Mirror confirmed Discord posts/replies when available; do not label a local
  preview or a browser-origin tool call as a real Discord exchange.
- Provide a decision-explanation panel with the selected route, model,
  confidence, policy verdict, and measured latency.
- Show model health and device failures rather than silently presenting fake success.
- Show a clear ready state after loading the fixture and warming required models.
  Missing audio, unsupported media, or unready models must prevent a misleading
  “ready to record” state.
- Keep application assets local; do not depend on a CDN during the demo.
- Distinguish **core demo ready** from **Discord connected**. The browser must
  be usable and recordable while internet access is unavailable, with external
  integration status clearly shown rather than reported as a successful post.

### Recording-ready presentation

- Provide a stable 16:9 presenter layout, targeting 1920×1080 output with readable
  captions and cards. Avoid scrolling or pop-ups during the planned demo.
- Keep video, captions, task outcomes, privacy state, and the decision explanation
  visible without switching tabs. Hide debug/setup panels, not input-mode or
  failure indicators.
- Support keyboard shortcuts for playback, restart, presenter mode, and the
  decision panel; avoid triggering shortcuts while typing.
- Offer an optional countdown and a repeatable start state so retakes do not
  require manual cleanup or page-by-page setup.

### Capture and export

- Provide a **Record demo** action that performs a capability/preflight check,
  obtains any required browser capture permission, then coordinates the
  countdown, capture start, and meeting-video playback.
- On supported browsers, capture the actual demo tab/window, including the
  meeting audio, and offer **Stop and download** for a playable local recording.
  Prefer WebM where supported; do not claim MP4 export without verifying the codec.
- Browser and OS audio-capture support varies. If integrated capture cannot
  include the required audio or resolution, show a clear fallback to a tested
  OS screen recorder, with a short setup checklist and the same presenter layout
  and countdown. Do not silently produce a muted recording.
- External-recorder setup may happen before a take; no terminal operations or
  application switching should be required while the demo is being recorded.
- Recording must show the application's actual live local inference, not a
  separately rendered animation of expected results.
- Capture denial, cancellation, source-video end, and export errors must leave
  the UI in a recoverable state. End-of-source handling must allow late results
  to finish before the take is finalized.
- Acceptance includes opening the saved file and confirming playable video,
  audible meeting sound, readable UI, synchronized results, and no sensitive
  setup material. A working browser page alone is not sufficient.

Existing internal package and deployment identifiers may remain `unspoken`
during the branding transition. They are implementation details, not the
user-facing product name.

## 9. Deployment prerequisites and known integration work

Resolve these before attempting a full startup:

1. **Offline dependency completeness.** Validate the complete ARM64 dependency
   closure, not just file checksums. The staged GPU wheel set currently lacks
   required CUDA packages; resolve this through a compatible already-loaded
   runtime or a completed local dependency bundle before installing the GPU
   environment. Do not work around it with an automatic network install.
2. **Authenticated inference.** The existing main model endpoint requires
   authentication. Both application clients and readiness probes must support
   it without hardcoding credentials.
3. **Replay cadence and synchronization.** The selected Light-ASD path expects
   approximately 25 FPS; reconcile the inherited sensor cadence with decoded
   video frames and a shared audio/video/browser media clock.
4. **Optional-role checks.** Disabling TTS or another optional service must not
   make core readiness fail.
5. **Meeting-video fixture.** Select a consented or appropriately licensed video,
   verify its audio track and decode support, and calibrate seat zones from its
   frames. USB microphone/webcam availability is not a readiness gate.
6. **Sandbox integration.** Verify installed NemoClaw/OpenClaw compatibility,
   CA trust, authenticated MCP access, and actual message behavior.
7. **Model capability checks.** Verify chat tool calls, reasoning options,
   speech response formats, and optional vision support against the real services.
8. **Browser recording.** Validate the target browser's capture/audio/export
   support and the OS-recorder fallback before the final rehearsal.
9. **Local asset inventory.** Resolve every required model to an existing
   download or staged archive and every required image to a local image/archive.
   Do not run a model downloader or container pull against venue Wi-Fi.

Use checksummed, versioned model and dependency manifests. Restore only the
needed assets first. Do not reinstall working drivers or automatically replace
an existing model service. Complete an internet-disconnected startup/replay/
recording check before declaring the core demo ready.

## 10. Implementation milestones

| Milestone | Work | Exit condition |
|---|---|---|
| M0 — Specification | Publish this specification before application source | Public repository contains only reviewed specification documents |
| M1 — Compatibility and offline readiness | Inventory existing downloads, resolve local GPU dependencies, authenticated inference, optional-role checks, and configuration drift | Every required asset resolves locally; authenticated completion and tool-call probes work without new downloads |
| M2 — Local foundation | Restore the application environment, define the replay adapters, and run deterministic scenarios | Unit tests pass; audio and frames from a local video follow the sensor contracts without USB hardware |
| M3 — Core inference | Reuse the existing main LLM and locally staged decider/ASR assets, then validate memory and model responses | Required roles pass real requests with internet unavailable; video inputs produce actual model outputs without downloads or scripted captions |
| M4 — Browser replay | Build the primary browser view, synchronized playback, voice/seat binding, check-in, and clean restart | Two-speaker video produces streaming attributed captions; repeated runs start clean with no USB input devices |
| M5 — Policy and integrations | Validate privacy, approvals, real devices, offline-safe notifier, TLS proxy, and sandbox tools | Local tools/device outcomes are confirmed; a separate online check proves Discord posting, while lack of internet does not block the core demo |
| M6 — Recording workflow | Complete the 16:9 presenter mode, countdown, capture controls, local export, and recorder fallback | A full browser-demo recording is saved and played back with audible meeting audio and readable results |
| M7 — Rehearsal and recovery | Repeat the video-backed browser demo, including an offline take, measure latency during recording, and exercise failures | Two consecutive clean takes with no downloads, stale state, missing audio, unapproved egress, privacy failure, or hidden fallback |

Do not start later integration work by assuming earlier readiness checks passed.
Failed or disabled components must remain visible to the operator.

## 11. Verification and acceptance

### Functional gates

| Area | Acceptance test |
|---|---|
| Input mocking | Disconnect or omit USB capture devices; load the meeting video and verify both audio and frame streams reach the real model pipeline |
| Perception | Play a two-speaker meeting fixture at 1x; inspect streaming captions, names or explicit unknowns, seats, and timestamps |
| Synchronization | Pause, resume, stop, and restart from the browser; source playback and processing share a clock, and no old-run outputs leak into a new run |
| Speech quality | Confirm the GB10 Q8 ASR path does not silently omit fixture utterances; evaluate overlap separately |
| Decision cascade | Exercise confident, uncertain, irrelevant, negated, and media-origin utterances |
| Task creation | A commitment creates the intended task, with an evidence-backed owner and due date or explicit unknowns |
| Policy | Guest, approval, low-confidence, sensitive-topic, and unauthorized-action cases behave as specified |
| Privacy | Pause/off-record tests inspect captions, local state, traces, pending actions, outgoing posts, and masked/muted source preview; distinguish notes from screen-recording state |
| MCP | Anonymous and invalid-token requests are denied; valid sandbox requests reach the actual meeting runtime |
| Discord | When internet is available, an approved task is posted and a mention-based question retrieves decisions through MCP; offline status/retries do not block or fake these outcomes |
| Devices | Intended lamp/fan actions are confirmed; unreachable devices are reported without pretending success |
| Recovery | Loss of an optional service does not fail core readiness; required-service failure is explicit |
| Branding | UI, assistant identity, prompts, and narration consistently use LocoCisco |
| Locality | Required inference stays local; network policy and application egress reporting agree with the intended boundaries |
| Offline startup | With internet unavailable, load the existing local models and fixture, run inference, and record/export the browser demo; no model, image, package, codec, or frontend download is attempted |
| Missing local assets | A missing required file yields an actionable local-preparation error; there is no automatic download, cloud fallback, or fake success |
| Browser demo | Complete the story from one browser presenter view without a terminal, webcam permission, microphone permission, or tab switching during the take |
| Recording | Save and play back a complete 16:9 take with audible meeting audio, readable captions/cards, and synchronized actual model results |
| Capture recovery | Denied/cancelled capture and unsupported audio capture lead to clear recovery or the tested recorder fallback, not a false success |
| Retake | Restart and record a second take without duplicate tasks, prior identities, stale captions, or repeated external actions from the first run |

### Performance targets

These are initial acceptance targets, **not measured GB10 results**:

- Measure the main path at 1x meeting-video playback with recording active;
  fast-forwarded evaluation is not a substitute for that measurement.
- Final caption latency: p95 within 2 seconds of the detected speech-segment end.
- Local task card: within 3 seconds of the completed commitment in the
  controlled demo, excluding a deliberate human-approval wait. Measure Discord
  delivery latency separately when online; slow Wi-Fi must not delay the local card.
- Privacy indicator: within 500 ms after recognition of a privacy command.
- Sustained rehearsal: no out-of-memory errors, accumulating processing backlog,
  unreported service failures, or progressive source-video/result drift.

Record model versions, runtime configuration, concurrent workloads, and timing
boundaries with every measurement, including browser and recording configuration.
If a target is missed, report the measured result and the remaining issue rather
than substituting simulated output.

## 12. Demonstration flow

### Before the take

Load a pre-staged meeting-video fixture containing the planned story beats,
resolve and warm the already-downloaded models, check the source audio, and
approve the intended external test posts and device actions. Arrange the browser
presenter layout and select the
recording destination. Complete any capture-permission or external-recorder setup
before the countdown. Verify that the core demo does not depend on Wi-Fi;
Discord availability is a separate status, not a reason to fetch models or block
the recording.

### Recorded browser flow

1. Start capture and the countdown, then play the meeting video at 1x. The
   prerecorded-input label remains visible beside the LocoCisco brand.
2. Check in the video's participants through the browser; as they speak in the
   recording, actual local inference produces attributed captions.
3. A request to send a deck by Friday becomes a task card. Any confirmed Discord
   post is mirrored in the browser's integration feed; without internet, show
   the local task and the truthful pending/failed integration state instead.
4. A comfort request and a slides request produce visible fan/lamp outcomes.
5. An off-record phrase in the video or the browser privacy control stops notes;
   the local wipe and masked/muted source preview are visible.
6. Resume according to policy, then ask “what did we decide?” through the browser
   room-question control. This uses the room's tools, not a scripted answer.
   Discord mention handling remains separately verified; do not mislabel this
   browser-origin query as a Discord round trip.
7. Open the decision panel to show the model route, policy verdict, measured
   latency, and text egress without leaving the browser.
8. Let outstanding results settle, stop recording, and download the take locally.

### After the take

Play back the exported file to check sound, readability, synchronization, and
privacy. Reset the session and repeat once to prove the flow is easy to record
again. Rehearsal starts with explicitly labeled synthetic tests, but the final
recorded demo must use the meeting video and real local model inference.

## 13. Publication boundary

This initial public repository contains only the README and build specification.
It excludes application source and previous Git history, model weights,
container archives, meeting-video fixtures, demo recordings, datasets,
transcripts, credentials, host-specific paths, and private network or account
configuration. Adding a sample clip or a finished demo video is a separate
publication decision after consent, licensing, and privacy review.

Before publishing implementation code or redistributing third-party artifacts,
review source provenance, release authorization, attribution, and applicable
licenses separately. A public specification does not grant redistribution
rights to the prototype, its dependencies, model weights, or evaluation data.
