# LocoCisco — Build Specification

- **Version:** 0.4
- **Date:** 2026-10-25
- **Status:** recording frontend, official-video replay and privacy export verified; final native capture and semantic-quality evaluation remain open
- **Target:** Dell Pro Max with NVIDIA GB10, ARM64 Linux, 128 GB unified memory
- **Official demo input:** [`Presentation/Demo.mp4`](Presentation/Demo.mp4)

This is the technical reference for building the demo and preparing slides.
**Current build** statements describe the local implementation. **Acceptance
requirements** and **future integrations** describe work that must be verified
before it is presented as delivered. A designed story, a healthy endpoint, and
a successful model inference are different kinds of evidence.

## 1. Product goal

LocoCisco is a meeting assistant that listens to ordinary conversation without
a wake word. It combines speech, anonymous voice labels, visible face tracks,
and explicit check-in evidence; proposes useful actions; and applies code-based
policy before producing meeting items or room outputs.

The primary demonstration is a browser presentation intended for recording.
A local video supplies microphone-equivalent audio and camera-equivalent frames.
The local models process those inputs during playback. The surrounding
dashboard shows what was heard, what each model contributed, which decision
route was taken, and what policy allowed or declined.

The recording canvas prioritizes information over controls. Video, captions,
model interactions, probabilities, measured timings, and outcomes stay visible.
Playback, resume, upload, check-in, and recording setup controls are hidden from
the default canvas and available through keyboard shortcuts or operator setup.

The core recording path uses already-staged models, dependencies, and media.
It does not automatically download models, pull container images, install
packages, or fall back to cloud inference. Network-dependent integrations are
separate from the core browser demo.

## 2. Current scope and follow-up scope

### Current recording build

- A single shared meeting session, using the official local video by default.
- Real local VAD, ASR, anonymous speaker labeling, face tracking, active-speaker
  scores, Decider inference, and optional Qwen escalation for each utterance.
- Final captions after each VAD segment; the meeting demo does not use the
  separate live-transcriber program's partial or revisable captions.
- Session-only name/voice/seat association when check-in and speaker evidence
  support it. Unknown or ambiguous speakers remain unnamed.
- Proposals for tasks, decisions, questions, room requests, and sensitive-topic
  pauses, followed by deterministic policy. Correct extraction is an acceptance
  criterion, not a guaranteed result for every utterance or source video.
- A recording-focused, light-theme browser view based on the dashboard and
  explainer designs in [`design-reference/dashboard/`](design-reference/dashboard/).
- Real model observation counters, recent activity, sensor latency, decision
  probabilities, local LLM routing, and policy outcomes.
- Playback synchronization, clean restart, off-record barriers, and local
  browser WebM recording or the installed Firefox MP4 capture fallback.
- **Local preview adapters for room devices and notifications.** The demo does
  not send real WiZ/Kasa commands or Discord messages.
- An operator-only question action using the retained on-record transcript.
  The demo intentionally starts with an empty document directory and agenda.

Seven roles are required for readiness: `decider`, `llm`, `asr`, `vad`,
`diarization`, `faces`, and `asd`. TTS, VLM, a second Decider, real devices, and
Discord are not core readiness prerequisites.

### Separate integrations or future work

- Real WiZ lamp, Kasa plug/fan, and sandboxed OpenClaw/Discord delivery.
- Live USB microphone and camera operation. Adapters exist separately, but the
  main recording workflow needs neither device nor browser capture permission.
- Whiteboard/scene interpretation through a VLM; no `video.scene` model worker
  is wired into the current replay sensor pipeline.
- Local synthesized replies and live speech translation. Neither is enabled
  in the current browser demo.
- A dedicated neural embedding service and populated company document index.
- More precise turn splitting, overlapping-speaker handling, fixture-specific
  seating calibration, and further intent-quality evaluation.

### Out of scope

Cloud fallback for meeting content; automatic startup downloads; persistent
biometric enrollment or cross-meeting identity; multi-room/user isolation;
Discord voice capture; unrestricted tools or browsing; long-term meeting
archives; and claims of production compliance or unmeasured accuracy.

The earlier `homedemo` smart-home simulator remains a separate application.
Its optional Bedrock path is not part of this meeting demo.

## 3. Target environment and official footage

| Resource | Current requirement |
|---|---|
| Compute | NVIDIA GB10 with CUDA and NVIDIA container support |
| Operating system | ARM64 Ubuntu 24.04-compatible environment |
| Python | Python 3.12 for the staged application dependencies |
| Display | Browser viewport targeting 1920×1080; no scrolling during a normal take |
| Input | A local video with audio; no USB microphone or webcam required |
| Recording | Actual browser picture and source-video audio, exported locally |
| Storage | Existing model/offline kit, official input copy, and recording space |
| Network | Core inference/playback/recording local; any real Discord integration is separate |

The populated layout was also checked at 1440×900. At 1280×720 it allows
vertical scrolling to keep the information readable; use 1920×1080 for recording.

### Official video

The presentation branch introduced the official demo input, replacing the AMI
clips as the default. The source asset is now part of the presentation materials:

| Property | Value |
|---|---|
| Repository path | `Presentation/Demo.mp4` |
| Source revision | `8f6ad1028295ec456c6ebc946df84b7e1f4be408` |
| Duration | 30.08 seconds |
| Picture | 1280×720 H.264, 24 FPS |
| Sound | Stereo AAC, 32 kHz |
| SHA-256 | `a999d8f0ebd55e5121d889457639f6d14bf020496675837c450bb0efd8aa3e43` |
| Staged runtime copy | `lococisco-runtime/media/lococisco-official-demo.mp4` |

The file is copied byte-for-byte and needs no transcoding. Local decoding
resamples its audio to the sensor contract. The replay adapter is configured
for up to 25 FPS; a 24 FPS source does not become 25 unique observations per
second. A source frame and a model scoring window are also different units.

[`Presentation/DEMO_VIDEO.fountain`](Presentation/DEMO_VIDEO.fountain) describes
a bilingual negotiation with name boxes, translated captions/speech, and a
quiet aside. **Any names, translated speech, subtitles, boxes, or graphics
already inside the video are input footage, not new dashboard inference.**
Do not use them as proof of live translation, verified identity, or automatic
whisper privacy. The current ASR client is configured with `language = "en"`;
the completed review run produced five final speech segments, but transcription
and intent accuracy on its bilingual and voiced-over audio still need labelled
evaluation. There is no enabled translation/TTS stage.

The older AMI recordings remain regression/evaluation fixtures. Their saved
results must not be relabeled as results for the official video.

## 4. Architecture

```text
Official local video -> ReplayClock
                            |
              +-------------+--------------+
              |                            |
         mono PCM audio                 video frames
              |                            |
         Silero VAD                      YuNet
              |                            |
       speech segment                 face tracks
          /       \                        |
     Parakeet    CAM++            audio + Light-ASD
       text      voice label         speaking scores
          \       /                        |
           audio.segment        video.speakers
                    \             /
                  in-process async event bus
                            |
                       speaker fusion
                            |
                     meeting.utterance
                            |
           privacy/reflex rules + bounded Decider query
                            |
                 Strands Decider 12B
                 /          |            \
          confident     uncertain      very unsure
             |              |               |
             |        local Qwen agent       |
             +--------------+---------------+
                            |
                    action/no-action proposal
                            |
                   deterministic Python policy
                            |
                         actuator
                /           |            \
          meeting items   screen state   local output previews
                \           |            /
                 FastAPI / WebSocket -> recording canvas

Derived sensor telemetry -> recent model observations / face-score overlay
Operator setup and shortcuts -> replay, privacy, questions, recording
```

### Runtime boundaries

| Component | Responsibility |
|---|---|
| Application process | Replay decoding, sensor workers, event bus, meeting state, fusion, cascade, policy, preview adapters, FastAPI and WebSocket stream |
| Browser | Synchronized presentation of the same video; captions, model graph, decisions, notes, privacy, and recording/export |
| Local model services | Decider, Qwen/vLLM, and Parakeet/NeMo-Speech endpoints |
| Optional integration deployment | MCP, private TLS proxy, OpenClaw/NemoClaw, real Discord and LAN devices |

The application runs in the prepared GPU container, outside the OpenClaw
sandbox. Its small “agents” are Python coroutines sharing one in-memory state,
not independent model processes or a distributed agent cluster. Every browser
client controls the same room, replay clock, media, and privacy state.

The composition root is `unspoken/meeting/runtime.py`. The browser entry point
`python -m unspoken.meeting.demo` specializes it with `DemoRuntime`,
`ReplayController`, and `ReplayBus`. The latter adds replay generations and
privacy epochs so cancelled or late work cannot publish into a newer take or
privacy state. The generic meeting runtime and separate transcription programs
do not automatically inherit every replay-specific behavior.

## 5. Model stack and decision behavior

| Role | Current model/provider | Placement |
|---|---|---|
| Speech detection | Silero VAD v5 | CPU, in process |
| Speech recognition | Parakeet TDT 0.6B v3, Q8, NeMo-Speech.cpp | Local GPU service |
| Anonymous voice labels | WeSpeaker CAM++ with online window labeling | CPU, in process |
| Face tracks | YuNet 2023mar | CPU, in process; no face recognition |
| Active speaker scores | Light-ASD, AVA weights | GPU, in process |
| Bounded decisions | Strands Decider **12B**, Gemma 4 base/adapter | Local GPU service; first opinion |
| Language-model escalation | Qwen3.6 35B-A3B NVFP4, vLLM | Existing local GPU service |
| VLM, TTS, second Decider | Disabled by the demo entry point | Optional future roles |
| Document embeddings | Hashed bag-of-words baseline | Not a deployed neural embedding service |

Role/provider selection is centralized in `config/models.toml` and the model
registry. Heavy providers are constructed lazily and reused. The GB10 profile
rejects hosted inference providers. Its locality checks accept configured
loopback/private-network endpoints; they are not a whole-machine firewall.
The current demo's inference endpoints are loopback.

### Perception details that affect the demo

- Silero closes a segment after 400 ms of silence, with an 8-second maximum.
  One segment worker runs ASR and voice labeling concurrently, then publishes
  the combined final segment. Segment-end latency differs from first-word delay.
- CAM++ uses 1.5-second windows with 0.75-second hops and up to four anonymous
  voice labels. The adapter chooses one majority voice for the whole VAD
  segment; it does not split mixed-speaker text into separate turns.
- YuNet tracks faces. Light-ASD scores one-second audiovisual windows at an
  approximately 0.2-second cadence. A returned observation is not necessarily
  a new neural-model forward pass.
- Replay currently uses two broad image zones, `left` and `right`. Multiple
  people on one side of the official footage are not distinct calibrated seats.
- Names require supported spoken or manual check-in association. A source-video
  name label, roster membership, or visible face is not identity evidence.

### Cascade and policy

One typed Decider request asks about actionable intent, kind, candidate owner,
room intent when relevant, and sensitivity. Code maps answers into actions,
grounds owners in the utterance/known speaker, and parses due-date wording.
The current question variant is `v3`.

| Decider action confidence | Normal route |
|---|---|
| At least 0.60 | Publish the Decider proposal directly for policy |
| At least 0.40 and below 0.60 | Ask the local Qwen agent when escalation is enabled |
| Below 0.40 | Publish the low-confidence result; policy normally ignores it without Qwen |

These are routing thresholds, not calibrated accuracy guarantees. A confident
`none` is still no action. Qwen tool bodies record requested actions; policy and
the actuator execute them. LLM tool proposals currently use an **uncalibrated
adapter score** (normally 0.9), which must not be shown as a measured model
probability. A local provider failure can use the existing local recovery path;
there is no cloud fallback or replacement with scripted meeting output.

Policy distinguishes act, ask owner, deny, and ignore. It applies privacy,
media-origin, guest/document, risk, and confidence rules. Task ownership needs
evidence; negated or hypothetical commitments abstain, and tentative commitments
are reviewed or rejected. A selected model action can therefore be denied.
The policy verdict is not proof that an external delivery succeeded.

The available regex PII role is not a claim that all content is automatically
redacted. The recording build relies on explicit privacy barriers and preview-only
outputs; broader outbound screening needs its own verified integration.

## 6. Interfaces and data flow

### Service boundaries

| Service | Port | Current recording build |
|---|---|---|
| Qwen/vLLM | 8000 | Authenticated local chat/tool calls; reuse existing service |
| Decider 12B | 8765 | Local bounded-question API |
| Parakeet ASR | 8770 | Local audio transcription API |
| Browser, local media, API | 8780 | Loopback FastAPI application |
| Second Decider / TTS | 8766 / 8771 | Disabled; not readiness prerequisites |
| MCP / private TLS proxy | 8791 / 8443 | Separate integration; not started by demo CLI |

The browser application intentionally has no per-user authentication. It is
restricted to loopback and the demo adds trusted-host and same-origin checks.
Do not expose this single-session interface as an internet or multi-user service.
MCP, when deployed separately, requires bearer authentication and the private
TLS proxy boundary.

### Replay and browser transport

PyAV decodes the official source incrementally. Audio becomes mono 16 kHz PCM;
video observations and audio chunks wait on one pausable backend `ReplayClock`.
The HTML video plays the same locally served file independently. Replay state
is polled every 250 ms; the browser corrects presentation drift over 0.4 seconds.
The backend clock is authoritative, not the HTML video's decode clock.

HTTP handles snapshot/model reads and replay, chat, question, upload, and reset
commands. `/ws` sends meeting events and snapshots, throttled to at most five
snapshots per second. Each browser has a bounded queue. Privacy is checked again
when a queued event is delivered.

An alternate local upload is available only in operator setup. The limit is
300 MiB and ten minutes, with audio/video decode validation. Uploads remain on
the box. Restart clears the take and invalidates in-flight work; pause preserves
the take. Arbitrary scrubbing is not part of the recording interface.

### Event and observation contracts

| Event | Meaning |
|---|---|
| `audio.segment` | Final text, anonymous voice label, times and ASR/diarization latency |
| `video.speakers` | Anonymous face tracks, normalized boxes, zones and speaking scores |
| `meeting.utterance` | Fused text, speaker evidence, source and meeting context |
| `meeting.proposal` | Action, arguments, model route, scores, request/answer trace and latency |
| `meeting.verdict` | Policy result and rule tied to the proposal ID |
| `meeting.item` / `meeting.command` | Retained item or requested output |
| `screen.state` | Captions, cards, privacy and meeting display state |
| `sensor.telemetry` | Sanitized browser projection of completed sensor observations |

The new telemetry projection contains latest ASR/diarization timings, anonymous
voice evidence, face boxes/scores, and observed segment/frame counts. It carries
no raw transcript, PCM, images, or embeddings. Video updates are capped at 5 Hz;
completed audio segments can update immediately. A snapshot also includes the
latest permitted telemetry. Privacy/reset clears these observations and counters.

**Health is not activity.** A ready model has passed a probe. A recent-update
highlight means an observed output arrived. The frontend may show Qwen pending
when an actual shadow proposal requests escalation. Observation counts must not
be labeled model-call counts or GPU-forward counts without separate instrumentation.
Decision explanations show recorded inputs, structured scores and policy rules;
they do not expose or invent hidden model reasoning.

### Questions and optional room tools

The browser's operator-only “Ask locally” action gives Qwen the retained
on-record transcript (bounded to its configured context), with no tools and no
external knowledge retrieval. It must say when the record lacks an answer.
The replay's document directory is intentionally empty, preventing unrelated
example company documents from being used as video evidence.

The separate MCP integration offers room status, check-in, transcript/items,
recap, search, proposals and approvals. It must be verified independently before
slides describe actual Discord exchanges or physical device delivery.

## 7. Privacy and state boundaries

- Notes state (`on`, `paused`, `off_record`) and screen-recording state are
  separate and both must remain truthful.
- Spoken off-record phrases and operator privacy actions take priority over
  normal planning. The default wipe window is the previous 60 seconds.
- The demo's privacy barrier removes protected transcript spans, clears derived
  cards/items, identities, approvals and previews, retires old agent state, and
  invalidates pending model/question results. Stream and browser epoch checks
  prevent old queued messages or delayed HTTP responses from repopulating it.
- The demo refuses trace-file writing. Generic-runtime trace behavior is not a
  promise that previously written files can be erased.
- The source preview is masked and audible/recorded source audio is muted while
  private. Transient local processing may continue to recognize spoken resume;
  private speech is not passed into ordinary planning or retained as notes.
- A whisper or quiet aside is not automatically an off-record command. The
  official video's scripted quiet moment does not validate automatic privacy.
- Resume is explicit. Model-proposed resume after a sensitive pause follows the
  policy approval route; direct operator controls are available in setup.
- State is session-local. Stop/restart releases sensor sessions and clears
  associations. This is not persistent face recognition or biometric identity.
- Privacy cannot erase the original video, earlier frames of a recording,
  exported files, bytes already delivered to a browser, or any previously
  delivered external message. Recording the demo is an intentional local capture.
- Device/notification previews are explicitly labeled. The application egress
  meter is not a host-wide traffic monitor or proof of network isolation.
- Credentials are supplied through protected runtime configuration. Setup paths,
  credentials, request headers, and raw debug payloads stay off the recording canvas.

These are the browser demo's implemented boundaries and verification
requirements. They do not establish regulatory compliance or a security audit.

## 8. Recording presentation and export

### Default canvas

Use the references' Webex/Momentum-inspired light surfaces, local Figtree and
JetBrains Mono fonts, and LocoCisco branding. The current layout presents:

1. **In the room:** actual source playback and real anonymous face/speaking
   observations, clearly identified as local prerecorded input.
2. **The conversation:** supported speaker associations and on-record captions.
3. **Inside the local intelligence:** audio and vision specialists feeding
   fusion, then Decider, conditional Qwen, and the code policy gate.
4. **Why it decided:** source utterance, actual route, scores, threshold,
   measured latency, policy verdict, and the reason for acting or staying quiet.
5. **What the room keeps:** actual tasks/decisions and local output previews.

The canvas shows the two latest retained items and three recent activity rows.
Long utterance quotes are limited to two lines in the decision card; the
underlying transcript and meeting state retain their existing limits.

The default screen has no visible Play, Pause, Resume, Restart, upload, check-in,
or question forms. A read-only progress/time display, privacy state, recording
indicator, readiness/error status and provenance remain visible. Model activity
and outcomes are primary content, not a hidden debug panel.

Normal visits start the replay automatically once all required models and browser
media are ready. A browser that denies audible autoplay receives muted playback;
the sound state stays visible, and a click or Space enables sound. Existing
paused/private sessions are preserved. `?autoplay=0` disables automatic startup
for setup and recording automation; the Firefox recorder selects it automatically.

Reference `story.js`/`explainer-story.js` data are design samples. Do not copy
sample participants, decisions, timings, model activity, voice clusters, or
confirmed integrations into the real recording view. If the models produce no
task, show that result instead of manufacturing a better story.

### Operator shortcuts

| Key | Action |
|---|---|
| Space | Enable sound after muted autoplay; otherwise start or pause/continue |
| R | Start browser capture; press again to stop |
| Shift+R | Restart a take when no recording is active |
| O | Open/close operator setup outside the default canvas |
| Escape | Close open setup/panels |

Detailed X-ray and legacy help remain available for inspection, but normal
recording should not depend on opening them. Shortcuts must not trigger while
typing in a form. Browser autoplay restrictions may require user activation for
sound; the page does not claim audible playback while its fallback is muted.

### Capture and export

- Native capture starts through `R` or the hidden setup action. It checks browser
  support, requests the selected tab/window, resets the take, counts down and
  starts playback. It does not request webcam or microphone access.
- Picture comes from the selected browser tab/window. Audio comes from the
  source video through Web Audio and follows the privacy mute. Export is local
  WebM using a supported MediaRecorder codec. The recording buffer is bounded.
- The installed Firefox fallback captures actual rendered screenshots through
  WebDriver BiDi and source audio in the browser, then muxes H.264/AAC MP4 with
  PyAV. Its requested 8 FPS is not guaranteed throughput; the companion JSON
  reports achieved capture performance. Native recording is preferable for motion.
- The fallback can be launched before the take with
  `python -m unspoken.meeting.record_demo --output <new-local-file.mp4>`.
  `--privacy-demo` is a separate operator-driven privacy demonstration, not an
  event inferred from the official video's whisper.
- Capture denial, missing codecs/audio, or export failure must be visible.
  Do not report a recording as saved until it has been produced. Existing output
  files must not be overwritten.
- Allow final inference to settle before ending the recording. Verify the saved
  picture, audio, timing, text legibility, privacy and lack of setup controls.
  Screenshot validation alone is not recording acceptance.

Internal package/CLI names may remain `unspoken`; the product name is LocoCisco.

The application snapshot now ships in this repository alongside the reference
designs and presentation. [Source provenance](docs/SOURCE_PROVENANCE.md) identifies
the imported baseline and bundled dependencies; the inherited USB Git history,
runtime recordings, credentials and large downloaded models are excluded.

After the autoplay update, 428 meeting, sensor, browser and branding checks pass
(three optional sensor checks skipped). Fresh-profile Firefox verification confirms
automatic muted playback without user activation and sound activation on the first
Space without pausing. The recorded timing table below remains evidence from the
earlier complete review take, not a new performance benchmark for this update.

## 9. Deployment and remaining integration checks

The prepared startup is `deploy/demo/up.sh` → `deploy/demo/run.sh` →
`python -m unspoken.meeting.demo`. It reuses existing Qwen/vLLM and staged
Torch/CUDA, starts or reuses local Decider and Parakeet, then runs the application.
Startup uses the cached GPU image with `--pull=never` and offline model settings.
Missing prerequisites fail with a preparation error instead of a download.

`deploy/demo/run.sh` defaults to the staged official video.
`LOCOCISCO_VIDEO` can explicitly select another local source and
`LOCOCISCO_ATTRIBUTION` supplies its attribution. An existing application
container retains its original input bind mount; changing a default file path
alone does not switch that running container's input. Apply an intentional
application reload/recreation and verify the loaded file. Model services need
no media-specific replacement.

Before accepting a recording:

- Verify all seven required local roles, authenticated model requests, official
  media identity, actual audio decode, and browser playback.
- Review the official video's bilingual, translated-voiceover and overlapping
  speech behavior. Check that source graphics are not presented as inference.
- Check two-zone seating limitations and anonymous identity handling. Do not
  assign the screenplay's four names to weakly supported tracks.
- Validate the redesigned view at 1920×1080 and a smaller viewport, including
  long utterances, absent tasks, provider failure and privacy transitions.
- Exercise reset during in-flight inference and privacy changes during pending
  questions, telemetry and queued WebSocket messages.
- Verify offline dependencies and frontend fonts/assets. Existing loaded image
  CUDA/PyTorch is reused; incomplete standalone GPU wheels are not a reason to
  install packages during the take.
- Treat real WiZ/Kasa, OpenClaw/MCP/TLS and Discord as separate deployment tests.
  The current demo's fake adapters cannot prove their success.

The audio segment queue is unbounded and Light-ASD work can occupy the event
loop. Monitor backlog and drift during longer runs; async architecture by itself
does not establish a real-time latency bound. Unified-memory sizing estimates
are planning inputs, not measured peak memory with all workloads and recording.

## 10. Implementation status and milestones

| Milestone | Current status | Acceptance evidence still required |
|---|---|---|
| Public specification | This document, design references and presentation assets are available | Slides agree with current implementation and distinguish targets |
| Local model/service foundation | All seven roles reported ready during the official-video review capture | Readiness recheck before the final take |
| Video replay | Repeated official 30.08-second takes completed without reloading the models | Recheck before the final presentation take |
| Sensor/decision observability | Actual observations and decision traces displayed; cache/epoch regression checks pass | Additional long-session and semantic-quality evaluation |
| Policy/privacy | Regression checks pass; recorded off-record interval masks video, clears content and mutes audio | Wider live-input and integration evaluation |
| Recording redesign | Populated layouts checked at 1920×1080 and 1440×900; corrected-color H.264/AAC review export verified | Smooth native browser capture reviewed for final presentation use |
| Real external integrations | Adapters and integration code exist separately | Physical device confirmation and actual authenticated Discord exchange |
| Semantic quality | Honest unknown/abstention paths and grounding checks implemented | Labelled official-fixture evaluation; no guaranteed task/decision claim |

“Implemented” means present in the local source checkout. It does not mean the
final visual recording or every acceptance gate has passed. Record verification
results with the exact source revision and media hash used for the take.

## 11. Verification and evidence for slides

### Required checks

| Area | Acceptance test |
|---|---|
| Source | Loaded input is the official video/hash; no synthetic transcript or expected-item injection |
| Perception | Audio and frames reach real models; final captions, boxes and scores reflect observations |
| Attribution | Unknown, mixed and off-camera speech do not inherit invented names |
| Cascade | Confident, uncertain, none, negated and media-origin cases follow the documented routes |
| Model display | Ready vs recently updated vs pending are distinct; displayed counters have correct units |
| Policy | Proposal, verdict and output are distinguishable; no-action/denial remain visible |
| Privacy | Wipe/reset invalidates current content, queued telemetry and delayed inference/questions |
| Replay | Pause preserves the take; restart removes old state and asynchronous output |
| Presentation | No visible transport/setup forms in the normal recording view; no clipped text or required scrolling |
| Export | Saved recording plays with audible source sound, readable model trace and privacy masking/muting |
| Retake | A second take starts clean and records correctly without reloading model weights |
| Offline operation | Core inference, assets, playback and export work without automatic network downloads |
| External integrations | Evaluated separately; local previews are never labeled confirmed physical/Discord delivery |

### Official-video review capture

The local artifact
`lococisco-runtime/recordings/lococisco-official-redesign-final-review.mp4` and
adjacent `.json` record a completed repeat take of the official 30.08-second
source with an operator-driven privacy interval. The source and staged runtime
copy match the SHA-256 in section 3. Browser checks at 1920×1080 and 1440×900
reported no visible controls, page overflow or console errors. A separate
labelled UI fixture checked a long Qwen route, its score disclosure and retained
items; those fixture outputs are not official-video inference evidence.

| Measurement | Recorded result |
|---|---|
| Required model readiness | All seven roles ready |
| Export | 1920×1080 H.264, limited-range BT.709 matrix; stereo AAC; full-file decode passed |
| Capture | 111 screenshots over 35.23 seconds; requested 4 FPS, achieved 3.15 FPS |
| Published speech segments | 5; 0 empty segments |
| Vision observations | 689 published; 32 input frames dropped |
| Sensor errors | 0 reported |
| ASR request latency | 44.8 ms average; 145.2 ms p95 |
| Diarization latency | 71.9 ms average; 91.2 ms p95 |
| Speech end → segment publication | 339.6 ms p95 |
| Privacy export | Video masked and sensitive UI cleared; decoded audio is silent at 18.5–20.3 seconds and nonzero before/after |
| Frontend/branding regression suite | 125 tests passed, including telemetry barriers, stale responses, presenter state and H.264 color preservation |
| Deployment dry-run suite | 36 passed, 2 skipped with Docker-daemon access isolated and the staged offline `uv` on PATH |

The MP4 container duration is 35.23 seconds; capture wall time and encoded
duration differ slightly. The export SHA-256 is
`31152e6eade1333a54d559e94618990e56fc98a3e6ae369257844efd242660d2`.

This verifies official-source replay, sensor output, hidden-control browser
capture, privacy masking/muting, and a decodable local audiovisual export.
**It is a review capture at 3.15 FPS, not accepted presentation-quality motion.** Five segments are a small
sample; their timings are not a general benchmark or measurements of completed
decisions. Readiness does not establish that every role was invoked on every
segment, and this manifest does not establish successful committed task or
decision extraction. The take used `privacy_demo: true`; its privacy transition
was triggered by the recorder, not inferred from the video's quiet aside.

The final review uses the corrected layout and color encoding. The earlier
review exports remain local historical artifacts. Dashboard captions visibly
mistranscribe the opening Mandarin under the current English ASR configuration;
the clip's bilingual semantic quality is still unvalidated. Review a native
browser capture for final motion, sound and presentation quality.

The adjacent `.source.json` records hashes of the application, demo launcher,
assets and model configuration, including uncommitted working-tree changes.
Its base commit is `2fc471bc2b9373957210131d4f441f6d3370c530`; the source digest is
`45a6c203402c6de134f95272e1d7ad2c447eb717b17aca7b99043a37b13a9cb1`.
Generated recordings and source manifests remain outside this public repository.

### Earlier backend evidence and its limits

The local source report `docs/reports/BACKEND_VALIDATION.md` contains completed
real-inference AMI runs. It reports all seven roles operating and no unexpected
external connections in those instrumented runs. It also reports **no task or
decision items from the two real AMI clips**, anonymous identities, and an
unsupported fake-lamp action when “Brightness.” was spoken in a product-design
conversation. Controlled text tests separately exercised task, grounding,
negation, escalation and media-policy paths.

Those results establish useful backend execution/regression evidence. They do
not establish semantic accuracy, verified identity, physical-device/Discord
integration, browser recording quality, or performance on `Presentation/Demo.mp4`.
Do not transfer the AMI measurements to the official video or substitute its
screenplay for measured output. The official review capture above supplies
separate execution/export evidence with its own limits.

### Performance targets

These are targets, not current benchmark claims:

- Final caption p95 within 2 seconds of the detected speech-segment end.
- A local task card within 3 seconds of a supported completed commitment,
  excluding an explicit approval wait; no task is expected for every sentence.
- Privacy indication within 500 ms after recognition of the privacy request.
- No accumulating backlog, out-of-memory failures, hidden provider failures, or
  progressive source/result drift during the accepted take and retake.

Measure at 1x playback with recording active. Report timing boundaries clearly:
ASR request, speech end → segment publication, utterance → verdict, browser
rendering and export throughput are different measurements. Include model
configuration, source/media revision and concurrent workloads.

### Slide claim guide

| Supported description of this build | Description requiring additional evidence |
|---|---|
| Local specialist models feed a bounded Decider and code policy | Every sentence is understood correctly |
| Real inference processes an official prerecorded input | The source video's labels/translations were generated by this runtime |
| Anonymous speaker/face evidence with conservative check-in | Persistent recognition or verified identification of everyone |
| Qwen is used when the routing logic calls for it | A fixed percentage of LLM calls avoided, or calibrated LLM confidence |
| Preview-only room and notification outcomes | Real Discord delivery or confirmed physical device control |
| Explicit off-record handling with local state barriers | Automatic whisper privacy or erasure of original/saved recordings |
| Local inference and application egress counters | A host-wide network audit or regulatory compliance certification |

## 12. Demonstration flow

### Before the take

Load the official video, warm/check the seven local roles, verify sound, and
arrange the 1920×1080 recording canvas. Complete capture permission and any
operator-only setup before the countdown. Keep source labels/translations
visually distinguishable from dashboard observations. Do not preload the
screenplay's names, tasks or translations as model results.

### Recorded flow

1. Start recording with `R` (or the prepared Firefox launcher). Native capture
   includes a countdown; the official 30.08-second input then plays at 1x.
2. Show actual audio/vision observations appearing in the local specialist
   pipeline and final captions appearing in the conversation panel.
3. Explain the fusion boundary: voices and visible tracks are evidence; unknown
   speakers remain unknown when check-in support is missing.
4. Follow the actual Decider result, any real Qwen escalation, and the policy
   verdict. Show no action or abstention when that is the output. Do not force
   a task, translation, fan change or escalation just to match a storyboard.
5. Show any actual retained meeting items and clearly marked local previews.
   The official source is not guaranteed to trigger every product capability.
6. Let inference finish and stop/save the recording. There is no requirement to
   open playback forms or switch views during this take.

A separate privacy take may deliberately invoke off-record and resume through
operator controls or the recorder's `--privacy-demo` option. Show the mask,
mute, cleared content and discarded late results. Identify this as an operator
privacy demonstration; the footage's quiet aside does not itself implement it.
Likewise, an operator transcript question or controlled task example is separate
from the official video's naturally observed outputs.

### After the take

Play the exported file and inspect picture, source sound, model/decision
legibility, timing and privacy. Restart and repeat once. Use the actual accepted
recording for slides; keep diagnostic or synthetic fixtures visibly distinct.

## 13. Repository and publication boundaries

This public repository contains the build specification, README,
`design-reference/` browser designs/assets, and `Presentation/` materials,
including the official source video, screenplay, slide deck and outline.
The reference dashboards have scripted sample data. Presentation narrative and
source-video effects are not evidence that a corresponding feature is enabled
or validated in the local application.

Application source remains in the separate `lococisco/` checkout; models,
dependency caches, generated recordings, runtime media copies and validation
artifacts live outside this public source tree. Credentials and deployment-specific
account/network configuration are not presentation assets.

Publishing these design/presentation materials does not publish the application
history or grant new redistribution rights to model weights, third-party data,
or dependencies. Preserve the bundled font/asset licenses and source attribution.
For slides and demos, use this specification's current-build/evidence distinctions
rather than treating every storyboard or use-case statement as an implemented claim.
