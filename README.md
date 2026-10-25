# LocoCisco

A local-first meeting assistant for the Dell Pro Max with NVIDIA GB10.

LocoCisco listens without a wake word, combines speech and anonymous speaker
evidence, and proposes tasks, decisions, and room actions. A bounded Decider
routes uncertain cases to a local language model; deterministic policy checks
the proposals before the room screen and local output previews change.

The main demo is a browser view designed for recording. The official input,
[`Presentation/Demo.mp4`](Presentation/Demo.mp4), supplies audio and camera frames;
no USB microphone or webcam is required. Local models process that input during
playback. Names, translations, and graphics already inside the source video are
source content, separate from the dashboard's actual inference results.

The build must reuse models already downloaded to the Dell box or staged in its
local offline kit, together with existing local model services. No automatic
model downloads or cloud inference are allowed. The core browser demo and local
recording must work without internet; Discord connectivity is reported separately.

The recording canvas shows the source video, captions, model activity and
interactions, decision scores, policy outcomes, and retained meeting items.
Playback starts automatically once the local pipeline is ready; browsers may
start muted until a click or Space enables sound. Setup controls are hidden by
default; keyboard shortcuts operate repeatable takes and local recording/export. Device and notification outputs
are local previews; real device and Discord integrations require separate checks.

## Build specification

Read [BUILD_SPEC.md](BUILD_SPEC.md) for the current architecture, model stack,
recording workflow, privacy boundaries, acceptance criteria, and slide claim
guide. It distinguishes implemented behavior, completed validation, and work
that still needs evidence. Repeated official-video review captures, the recording
layout and privacy masking/muting have been verified. Final native capture quality
and bilingual transcription/intent accuracy still need evaluation.

This repository contains the application source, build specification,
[design references](design-reference/dashboard/), and
[presentation materials](Presentation/), including the official source video.
Reference dashboards use scripted example data. The executable demo uses real
local inference. Large model weights, credentials and generated recordings remain
outside Git; small licensed Silero/YuNet assets and test fixtures are included.

## Run the prepared demo

On the prepared Dell box, follow [docs/BROWSER_DEMO.md](docs/BROWSER_DEMO.md) to
configure the credential-file path and run `deploy/demo/up.sh`. Open
`http://127.0.0.1:8780/`; use `?autoplay=0` for intentionally idle setup.
The guide documents the existing offline model kit and local services required
by the launcher. It does not download models or install packages at startup.

The implementation is under `unspoken/`; `homedemo/` is the earlier smart-home
prototype. See [source provenance](docs/SOURCE_PROVENANCE.md) for the imported
snapshot and retained dependency notices.
