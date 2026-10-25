# LocoCisco — five-slide talk track

Suggested pace: approximately 4–5 minutes, including your demo recording.
Right arrow reveals information; use 1–5 to jump between slides. The figures and
model details reflect the current local meeting build, inspected October 25, 2026.

## 1. A smarter room. A private one. (~35 seconds)

“We’re building a local-first, privacy-driven intelligent office. Our first use case
is the meeting: understand the words, add visual speaker context, and turn supported
commitments into useful follow-through. All of that intelligence runs on one Dell
edge box. The conversation does not need to go to a cloud model.”

First build: introduce local processing, privacy control and office awareness.
Second build: explain captions, proposed tasks and decisions, and grounded answers.
No wake word is needed. This is an architecture illustration, not live telemetry.

The current recording build uses a local video's audio and frames instead of a USB
microphone/camera. Device and notification actions are local previews. Task extraction
has controlled-test evidence, but genuine-video task/decision success is not established
by the recorded backend reports. Frame product behavior as designed capability rather
than guaranteed semantic accuracy.

## 2. Seven models. One understanding. (~90 seconds)

“Instead of asking a large language model to do every job, we use specialists.”

Advance through four phases: hear, see, decide, generate/check. Click individual
models when the audience wants more detail. Captions publish from perception before
the action-decision path finishes.

| Model | How it works | Why it is here |
|---|---|---|
| Silero VAD v5 | ONNX CPU model detects speech in short windows; ends segments after 400 ms silence, with an 8 s cap. | Cheap continuous listening gates expensive processing. |
| Parakeet TDT 0.6B v3, Q8 | GPU transducer predicts tokens and durations, producing words and timestamps through NeMo-Speech.cpp. | Compact speech specialist; project CPU comparison found similar word error to Whisper large-v3-turbo at lower compute cost. That comparison is not a GB10 throughput benchmark. |
| WeSpeaker CAM++ | CPU speaker embeddings from 1.5 s windows, 0.75 s hops, compared with session centroids. | Better results in the project's tested diarization configurations. Anonymous voices, no persistent enrollment. The current adapter assigns the majority voice to a whole VAD segment. |
| YuNet 2023mar | CPU face detection followed by overlap tracking. | Small detector supplies anonymous face tracks and crops; avoids unnecessary GPU use. This is not face recognition. |
| Light-ASD, AVA | Small learned audio/visual encoders and temporal processing score active speakers in 1 s windows. | Combines sound and face motion; performed better than motion-only ASD in project tests. Approximately 0.2 s scoring cadence is not one guaranteed neural forward every 0.2 s. |
| Strands Decider 12B | Gemma 4 base/adapter with classification readout; typed questions yield scored options. | Bounded, inspectable decisions about intent, action type, owner and sensitivity. This is the current first opinion, replacing the older 2B-first story. |
| Qwen3.6 35B-A3B NVFP4 | Local vLLM mixture-of-experts model, ~3B active parameters; quantized weights. | Larger capacity with sparse compute and smaller weight storage. Generates tool proposals on escalation and responses grounded in retained meeting content when asked. |

The normal Decider routing thresholds are **≥0.60 → policy**, **0.40–0.60 → local
Qwen when escalation is enabled**, and **<0.40 → normally no action**. A confident
`none` still means no action. These are routing thresholds, not calibrated accuracy.
Qwen's adapter score is not a measured probability.

Code fuses speaker evidence and independently checks proposals. Policy can allow,
ask, deny or ignore. Faces and anonymous voices do not prove identity; names need
supported check-in. Do not say the system recognizes everyone.

Scope: VLM, TTS and the second Decider are disabled in the browser demo. Document
retrieval uses a hashed bag-of-words baseline, not another neural model. Regex PII,
tracking, fusion and policy are code, not additional neural models. Whisper,
Sortformer, pyannote and motion-only ASD were evaluated alternatives.

## 3. One box. The whole stack. (~45 seconds)

“The Dell Pro Max with NVIDIA GB10 gives us 128 GB unified memory. Three CPU
specialists handle continuous lightweight perception. The GPU handles Parakeet,
Light-ASD, Decider and Qwen. We reuse loaded local model services. Quantization and
sparse experts make the compute strategy practical.”

Advance once to the GPU/quantization story; advance again to the measured figures.

The `lococisco-official-redesign-final-review` take of the official 30.08-second source video reported **five speech segments**,
**44.8 ms average ASR request** (145.2 ms p95), **71.9 ms average voice labeling**
(91.2 ms p95), and **339.6 ms p95 speech end to segment publication**. All seven
roles were ready; zero sensor errors were reported. These are **sensor timings**,
not completed decisions, browser latency, peak-memory measurements or a general
capacity benchmark. Readiness does not mean all models were invoked on each segment.

The recorded export SHA-256 is `31152e6eade1333a54d559e94618990e56fc98a3e6ae369257844efd242660d2`.
The capture also dropped 32 input frames and achieved 3.15 recording FPS. Do not
call it an accepted smooth final recording or claim literal spare headroom. Separate
180-second AMI backend runs show sustained local execution, but report Decider p95
of 2.15–2.79 seconds, anonymous identities, no committed task/decision items and an
unsupported fake-lamp action. Avoid combining measurements across fixtures.

## 4. Rethink the room. Own the intelligence. (~45 seconds)

“Our commercial direction is an edge appliance that can replace traditional Cisco
room hardware over time. We intend to sell Dell hardware, LocoCisco software, and
deployment and support. The value is private local intelligence, software updates
on general-purpose compute, and a path to lower total cost.”

Advance to reveal model upgrades; advance again to the cost thesis and first
customer. A reasonable first pilot is a privacy-conscious team modernizing one room.
Reuse compatible microphones, cameras and displays where practical. The core local
inference path has no per-call cloud inference bill.

There is no validated price, Cisco TCO comparison, replacement qualification,
multi-room capacity, production SLA or compliance certification. “Lower-cost” is
intended positioning, not a measured saving. Do not imply all Cisco deployments
require cloud transcription or lack privacy controls. Product privacy includes
explicit off-record barriers; it is not automatic redaction of everything, and it
does not erase original source media or already-saved/exported recordings.

## 5. Less note-taking. More meeting. (video + ~15 seconds)

Choose an accepted final recording from this device. It plays in the browser via
a local object URL and is not uploaded. Playback requires a click. Replacing a
video releases the prior object URL; the choice does not survive a reload.

“Watch the speech and vision evidence, the decision route and policy verdict,
and the explicit privacy controls. Your room. Your intelligence.”

The included `Demo.mp4` option is explicitly **source footage**. Its names,
translations and visual graphics are baked into the input and must not be
described as runtime inference. The deck intentionally starts with an empty video
placeholder and does not auto-select an unaccepted review recording.

## Evidence references

- [Current build specification](../BUILD_SPEC.md): “Model stack and decision
  behavior,” “Official-video review capture,” “Privacy,” and “Slide claim guide.”
- [Current browser-demo guide](../../lococisco/docs/BROWSER_DEMO.md).
- [Model configuration](../../lococisco/config/models.toml): `profiles.gb10`.
- [Model selection experiments](../../lococisco/docs/TRANSCRIBE.md).
- [Backend validation and semantic limits](../../lococisco/docs/reports/BACKEND_VALIDATION.md).
- [Hardware deployment context](../../lococisco/docs/HACKDAY.md).

The source-code checkout is separate from this public repository. Relative links
to that checkout work in the supplied workspace; they may not resolve in a
standalone copy. `BUILD_SPEC.md` remains the public evidence summary. Earlier broad
outline and legacy cloud/2B smart-home content are not the active deck's scope.
