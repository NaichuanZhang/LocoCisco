# LocoCisco

A local-first meeting assistant for the Dell Pro Max with NVIDIA GB10.

LocoCisco listens without a wake word, attributes captions to meeting participants,
captures tasks and decisions, and helps the room respond through its screen,
local devices, and a controlled Discord integration.

The main demo runs in a browser. A prerecorded meeting video supplies both the
mock microphone audio and mock camera frames, so no USB microphone or webcam is
required. The local models still process those inputs during playback; captions
and decisions are not scripted substitutes.

The build must reuse models already downloaded to the Dell box or staged in its
local offline kit, together with existing local model services. No automatic
model downloads or cloud inference are allowed. The core browser demo and local
recording must work without internet; Discord connectivity is reported separately.

A recording-ready presenter mode keeps the source video, captions, tasks,
privacy controls, and decision explanations in one view, with repeatable
playback and a local recording/export workflow.

## Build specification

Read [BUILD_SPEC.md](BUILD_SPEC.md) for the architecture, model stack, privacy
requirements, implementation milestones, acceptance criteria, and demo flow.
Browser presentation and demo recording are required deliverables, not optional
polish after the backend is finished.

**Specification-first publication:** this repository currently contains design
documentation only. Application source, model weights, datasets, credentials,
and deployment-specific configuration are not included. The specification
describes the intended build, not a claim that live deployment is complete.
