# LocoCisco — Build Specification

- **Version:** 0.1
- **Date:** 2026-10-25
- **Status:** specification-first; live GB10 integration and acceptance testing pending
- **Target:** Dell Pro Max with NVIDIA GB10, ARM64 Linux, 128 GB unified memory

## 1. Product goal

Build a meeting-room assistant that understands ordinary conversation without
requiring a wake word. Speech recognition, speaker attribution, decision-making,
and document lookup run locally. Participants see what the assistant heard,
what it decided, and what information it sent outside the room.

The first public deliverable is this specification. A local prototype is the
starting point for implementation work; publishing this document does not
publish that source or its history.

## 2. Version-one scope

### Required

- Live captions from a USB speakerphone.
- Session-only voice labels, camera seat attribution, and explicit participant
  check-in; unknown speakers remain unnamed rather than being guessed.
- Extraction of commitments, owners, due dates, and meeting decisions.
- A local decision cascade: bounded questions to a small decision model,
  followed by a larger local language model only when needed.
- Deterministic policy checks before any external post or device action.
- A room screen showing captions, participants, tasks, privacy state, model
  health, decision explanations, and application-level text egress.
- Spoken and Discord controls for pausing and resuming notes.
- A WiZ lamp and Kasa-controlled fan on a dedicated local network.
- A sandboxed Discord agent that posts permitted text and answers questions
  through the room's tools.
- Simulator and replay modes for verification without live hardware.

### Optional after the core demo works

- Local text-to-speech.
- Whiteboard or scene interpretation through the local vision-language model.
- A second decision model for fallback or comparison.
- A dedicated embedding model for better local document retrieval.

### Out of scope for the first build

- Cloud-model fallback for meeting content.
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
| Audio | USB speakerphone with a selectable capture device |
| Video | USB webcam; approximately 25 real frames/s into Light-ASD |
| Display | Local HDMI screen with a browser in presenter mode |
| Devices | WiZ lamp, Kasa plug/fan, and a dedicated room network |
| Storage | Capacity for the selected model caches, images, and environments |
| Network | Local device access; restricted Discord access for the chat integration |

Deployment must adapt to an already-provisioned machine. Preserve existing
model services, authentication, and unrelated workloads. Stopping or replacing
an existing service requires explicit operator approval.

## 4. Architecture

```text
USB microphone -> VAD -> ASR -> voice labels ---------+
                                                     |
USB camera -> face tracking -> active speaker -------+-> attributed utterance
                                                     |          |
Discord check-in and commands -> room MCP tools ------+          v
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
                                             room screen   LAN devices    notifier
                                                                              |
                                                                     sandboxed OpenClaw
                                                                              |
                                                                           Discord
```

### Runtime boundaries

| Component | Responsibility |
|---|---|
| Host room process | Sensor access, event bus, meeting state, decision cascade, policy, devices, screen, and MCP tools |
| Local model services | Decision model and OpenAI-compatible chat/speech endpoints |
| OpenClaw in NemoClaw/OpenShell | Discord-facing identity, permitted notifications, and calls to room tools |
| Private TLS proxy | Authenticated connection from the sandbox to the host's loopback MCP service |

The room process stays outside the sandbox because it needs USB devices and
local-network appliances. The Discord agent does not own the meeting record
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
workloads included; do not treat the estimates as guaranteed free capacity.

## 6. Interfaces and data flow

### Default service boundaries

| Service | Port | Access |
|---|---|---|
| Main vLLM API | 8000 | Host loopback; preserve authenticated sandbox access where already configured |
| Decider 12B | 8765 | Host loopback |
| Optional second decider | 8766 | Host loopback |
| ASR | 8770 | Host loopback |
| Optional TTS | 8771 | Host loopback |
| Room screen and API | 8780 | Host loopback |
| Room MCP service | 8791 | Host loopback, bearer protected |
| MCP TLS proxy | 8443 | Private sandbox bridge interface only, bearer protected |

Treat the OpenClaw gateway address as deployment configuration rather than
assuming the port or sandbox name from a fresh installation.

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

- Raw microphone audio and camera video stay on the local machine. In normal
  live operation, processing buffers are transient rather than recordings.
- Only policy-permitted text may be sent to Discord. Recording or exporting
  diagnostic media requires a separate explicit operator action.
- The screen must make recording state visible: notes on, paused, or off record.
- A recognized off-record request must take priority over normal model decisions.
  Do not retain subsequent meeting content in captions, transcript history,
  traces, document context, or queued notifications while that state applies.
- Off-record handling must remove the configured recent local transcript window,
  initially 60 seconds. Do not promise that already-delivered external messages
  have been erased; any external retraction must be separately verified.
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

## 8. Room-screen requirements

The screen is a local browser interface, not a cloud dashboard.

- Display the **LocoCisco** name consistently.
- Make captions and privacy state legible on the room display.
- Show tasks, decisions, participants, and the current meeting context.
- Provide a decision-explanation panel with the selected route, model,
  confidence, policy verdict, and measured latency.
- Show model health and device failures rather than silently presenting fake success.
- Support a presenter layout and keyboard-accessible controls.
- Keep application assets local; do not depend on a CDN during the demo.

Existing internal package and deployment identifiers may remain `unspoken`
during the branding transition. They are implementation details, not the
user-facing product name.

## 9. Deployment prerequisites and known integration work

Resolve these before attempting a full startup:

1. **Offline dependency completeness.** Validate the complete ARM64 dependency
   closure, not just file checksums. The staged GPU wheel set currently lacks
   required CUDA packages; resolve them before installing the GPU environment.
2. **Authenticated inference.** The existing main model endpoint requires
   authentication. Both application clients and readiness probes must support
   it without hardcoding credentials.
3. **Camera cadence.** The selected Light-ASD path expects approximately 25 FPS;
   the inherited meeting-sensor configuration must be reconciled with that requirement.
4. **Optional-role checks.** Disabling TTS or another optional service must not
   make core readiness fail.
5. **Physical capture.** Attach and enumerate the intended microphone and webcam,
   then configure devices and calibrate seat zones.
6. **Sandbox integration.** Verify installed NemoClaw/OpenClaw compatibility,
   CA trust, authenticated MCP access, and actual message behavior.
7. **Model capability checks.** Verify chat tool calls, reasoning options,
   speech response formats, and optional vision support against the real services.

Use checksummed, versioned model and dependency manifests. Restore only the
needed assets first. Do not reinstall working drivers or automatically replace
an existing model service.

## 10. Implementation milestones

| Milestone | Work | Exit condition |
|---|---|---|
| M0 — Specification | Publish this specification before application source | Public repository contains only reviewed specification documents |
| M1 — Compatibility | Resolve GPU dependencies, authenticated inference, optional-role checks, and configuration drift | Dependency resolution succeeds; authenticated completion and tool-call probes work |
| M2 — Local foundation | Restore the selected application environment and run deterministic scenarios with fake I/O | Relevant unit tests and simulator expectations pass without touching physical devices or Discord |
| M3 — Core inference | Reuse the main LLM, start the decider and ASR, validate memory and model responses | Every required role passes real requests; scripted meeting passes with real models and fake I/O |
| M4 — Live perception | Configure microphone, webcam, frame cadence, voice/seat binding, and check-in | Two-person live captions work; unknown and off-camera speakers are handled explicitly |
| M5 — Policy and integrations | Validate privacy, approvals, real devices, notifier, TLS proxy, and sandbox tools | Approved task reaches Discord; room query uses live state; intended device actions are confirmed |
| M6 — Rehearsal and recovery | Run the demo, measure latency, inspect resource use, and exercise failures | Two consecutive successful rehearsals with no unapproved egress, privacy failure, or hidden fallback |

Do not start later integration work by assuming earlier readiness checks passed.
Failed or disabled components must remain visible to the operator.

## 11. Verification and acceptance

### Functional gates

| Area | Acceptance test |
|---|---|
| Perception | Transcribe a recorded fixture, then alternate between two live speakers; inspect names, seats, and timestamps |
| Speech quality | Confirm the GB10 Q8 ASR path does not silently omit fixture utterances; evaluate overlap separately |
| Decision cascade | Exercise confident, uncertain, irrelevant, negated, and media-origin utterances |
| Task creation | A commitment creates the intended task, with an evidence-backed owner and due date or explicit unknowns |
| Policy | Guest, approval, low-confidence, sensitive-topic, and unauthorized-action cases behave as specified |
| Privacy | Pause/off-record tests inspect screen, local state, traces, pending actions, and outgoing posts |
| MCP | Anonymous and invalid-token requests are denied; valid sandbox requests reach the actual meeting runtime |
| Discord | One approved task is posted; a mention-based question retrieves current decisions through MCP |
| Devices | Intended lamp/fan actions are confirmed; unreachable devices are reported without pretending success |
| Recovery | Loss of an optional service does not fail core readiness; required-service failure is explicit |
| Branding | UI, assistant identity, prompts, and narration consistently use LocoCisco |
| Locality | Required inference stays local; network policy and application egress reporting agree with the intended boundaries |

### Performance targets

These are initial acceptance targets, **not measured GB10 results**:

- Final caption latency: p95 within 2 seconds of the detected speech-segment end.
- Task notification: within 3 seconds of the completed commitment in the
  controlled demo, excluding a deliberate human-approval wait.
- Privacy indicator: within 500 ms after recognition of a privacy command.
- Sustained rehearsal: no out-of-memory errors, accumulating processing backlog,
  or unreported service failures.

Record model versions, runtime configuration, concurrent workloads, and timing
boundaries with every measurement. If a target is missed, report the measured
result and the remaining issue rather than substituting simulated output.

## 12. Demonstration flow

1. Two participants check in and speak naturally; named captions appear.
2. One asks another to send a deck by Friday; the commitment becomes a task.
3. A comfort request changes the fan, and a slides request changes the lamp scene.
4. Someone says “off the record”; notes stop and the local wipe is visible.
5. Notes resume according to policy.
6. In Discord, `@LocoCisco what did we decide?` returns permitted meeting state.
7. The presenter opens the decision panel and explains the local model route,
   policy decision, measured latency, and text-only notification egress.

Rehearse first with simulated I/O, then with real models, and finally with the
full room setup. External test posts and physical device actions require the
operator's explicit approval.

## 13. Publication boundary

This initial public repository contains only the README and build specification.
It excludes application source and previous Git history, model weights,
container archives, datasets, transcripts, credentials, host-specific paths,
and private network or account configuration.

Before publishing implementation code or redistributing third-party artifacts,
review source provenance, release authorization, attribution, and applicable
licenses separately. A public specification does not grant redistribution
rights to the prototype, its dependencies, model weights, or evaluation data.
