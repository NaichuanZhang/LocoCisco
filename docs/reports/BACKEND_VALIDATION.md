# Backend-only validation: real local inference

## Scope and reproduction

`deploy/demo/validate_backend.py` runs in a **separate process**, with a temporary
`DemoRuntime`, `ReplayController`, empty document directory and empty agenda.
It does not instantiate a browser app, record anything, contact port 8780,
start/restart a service, download assets, or modify model/core settings.
Notifications and devices are the demo's local fakes; they are not deliveries.

Using the existing container/image, installed packages, weights and media:

```sh
# Standard-library checks only; no model/server access or report file.
python3 deploy/demo/validate_backend.py --self-test

# Default: both complete 180-second fixtures at 1x, 60-second reset/replay,
# then separate controlled-text cases. Creates a new JSON outside Git.
docker exec lococisco-demo python /app/deploy/demo/validate_backend.py

# A single complete design fixture, with explicitly enabled, bounded public
# action examples (at most six excerpts of at most 160 characters).
docker exec lococisco-demo python /app/deploy/demo/validate_backend.py \
  --media /runtime/media/ami-design-decisions.mp4 --include-public-examples
```

Default inputs:

* `/runtime/media/ami-meeting-highlights.mp4`: AMI IS1009a, edited source ranges
  50–128 and 650–752 seconds, total 180 seconds.
* `/runtime/media/ami-design-decisions.mp4`: AMI IS1009d, source range
  390–570 seconds, total 180 seconds.

Both are existing local, audio-normalized fixtures; the validator records their
SHA-256 digests and adjacent provenance manifests. It never creates or substitutes
synthetic video, speech, face detections, diarization, or ASD output.

Defaults use profile `gb10`, the credential **by path**
`/run/secrets/vllm-api-key`, non-streaming LLM requests, and disabled
`tts`, `vlm`, `decider_big`. The credential value, environment, headers, raw model
requests/replies, full real transcript, PCM, frames and embeddings are not saved.
Default media reports contain aggregates and content hashes; public excerpts
require the explicit flag above. Examples are selected only from retained
on-record state and are excluded if that state was wiped.

Useful options:

* Repeat `--media PATH` to select existing files.
* `--runtime-dir PATH` changes the default fixture/report root.
* `--meeting-config PATH` loads an existing config; thresholds are recorded, not
  lowered or overridden by the validator.
* `--replay-seconds 60` bounds the reset/replay phase.
* `--media-limit-seconds N` is a **partial diagnostic**, explicitly reported as
  `bounded: true, complete: false`; it is not a complete-clip result.
* `--output PATH` must be a new file outside the repository.

Exit zero means the recorded role/error/invariant/control checks passed. It does
**not** mean every real-video utterance was classified correctly, that identities
were verified, or that the video yielded positive task/decision items. Ambiguous
real-media inputs have no asserted positive-action expectation. Missing coverage
(for example, no retained ASR decision candidate) is a failed coverage check,
not permission to invent input.

## What is actually measured

Wrappers call original methods and return the **same output objects**:

* Silero stream feed/flush **and actual ONNX probability calls**.
* Real ASR transcriptions and online diarization calls; individual CAM++ embedding
  evaluations are counted without retaining their arrays.
* YuNet session updates and detector evaluations, returned face boxes and track IDs.
* Light-ASD `session.update`, `_score` attempts, actual `LightASDScorer.score`
  window evaluations, and the loaded Torch model's `forward` separately.
  An update is not a model forward: short/incomplete tracks may not score, and
  a scoring attempt may evaluate several faces.
* Decider requests, Qwen model streams, proposal routes/scores, verdicts,
  local item/notification counts and actual-egress counters.

Media bus aggregation records origins, known/unknown identity counts and voice
confidence, not a transcript. Python warning/error logs and bus warnings are
categorized and counted without logging potentially prompt-bearing messages;
`bus.errors` is checked separately. HTTP 422 is explicitly categorized.

Timing is CPU wall time. Torch-forward timings are **dispatch wall time**, not
synchronized GPU duration; ASD window scoring includes the output's CPU transfer.
The later validator revision also measures bus-observed utterance-to-verdict
time. Sensor `segment_publish_ms_p95` measures speech-end to segment publication,
**not** model decision completion or browser rendering. Shared model-service load
can affect these measurements.

Before registry/provider initialization, a process-only guard intercepts Python
socket connect/create_connection/connect_ex, DNS lookup and UDP sendto/sendmsg.
Public, LAN, external-hostname and mapped-public-IPv6 self-tests must be denied.
Only loopback destinations are permitted. This is not a host firewall, packet
capture or sandbox for native subprocess code. Replay's FFmpeg/ffprobe subprocesses
receive local file paths, not network URLs.

## Completed full default run

Report, outside Git:

```text
/runtime/validation/backend-20261025T213118Z-1304.json
```

Host equivalent:
`/home/dell/dell_hackathon/lococisco-runtime/validation/backend-20261025T213118Z-1304.json`.

The report has **exit_code 0**, no failed checks, and source hashes for the exact
working tree used. Git HEAD was `2fc471bc2b9373957210131d4f441f6d3370c530`;
the working-tree grounding/privacy fixes were present. Agent identifiers are not
Git commit IDs. No recorded source file changed during this run.

All seven real roles were healthy: Decider 12B, Qwen3.6 35B-A3B NVFP4, Parakeet,
Silero VAD, CAM++ online diarization, YuNet, Light-ASD. Warmup took **1,699.0 ms**;
the Decider health request returned `P(true)=0.98`, and real Qwen requests returned
`ok`. Actual configuration: act **0.60**, ask **0.40**, owner **0.50**, questions
**v3**, escalation enabled. Qwen's 0.9 adapter score is explicitly
`confidence_source: adapter_constant, confidence_calibrated: false`.

### Genuine media — no injected utterances/perception

| Measurement | IS1009a highlights | IS1009d design decisions |
|---|---:|---:|
| Complete 1x source playback | 180 s | 180 s |
| Wall time including final inference drain | 182.042 s | 181.594 s |
| Audio consumed / last video time | 180.0 / 179.96 s | 180.0 / 179.96 s |
| VAD ONNX evaluations | 5,625 | 5,625 |
| Real ASR / diarization calls | 36 / 36 | 37 / 37 |
| CAM++ embedding evaluations | 198 | 178 |
| YuNet detector / ASD update calls | 4,355 / 4,355 | 4,460 / 4,460 |
| Source video frames dropped | 145 | 40 |
| Distinct session-only face tracks | 14 | 9 |
| ASD scoring attempts | 801 | 808 |
| **Actual ASD window scores / Torch forwards** | **1,533 / 1,533** | **1,241 / 1,241** |
| Maximum per-window speaking probability | 0.85597 | 0.89274 |
| Maximum returned track speaking score | 0.807 | 0.798 |
| Segment voice confidence range | 0.244–0.646 | 0.333–0.792 |
| ASR p95 | 118.0 ms | 93.2 ms |
| Speech-end → segment publication p95 | 378.7 ms | 321.2 ms |
| Decider request p95 | 2,154.3 ms | 2,792.0 ms |
| Known / unknown utterance identities | 0 / 36 | 0 / 37 |
| Utterance origins | 36 person | 37 person |

Real box example from highlights: track `f1`, normalized box
`[0.2506, 0.5302, 0.0635, 0.0977]`, left seat. Design example:
`f1`, `[0.6136, 0.5472, 0.0691, 0.1067]`, right seat.
Tracks are anonymous, session-local observations, not verified human identities.

**Actual output, not expected output:**

* Highlights: 36 Decider `none` proposals, 36 ignore verdicts, no items.
* Design: 37 utterances, 38 proposals including one shadow `log_decision`;
  one genuine local Qwen call. Final verdicts were **36 ignore-none and one
  act-set_light**. There was **no task or decision item**. A shadow proposal
  is not a logged decision; a lamp action is a fake-device preview, not evidence
  that the group requested a physical room change.
* Both phases: zero sensor/provider/bus errors and zero warning/error log entries.
  Zero FakeNotifier posts/DMs and zero actual text/audio/video egress.

### Reset and bounded replay

Reset checks passed for identities, transcript, history/provenance, items/cards,
previews/counters, pending approvals, policy/fusion context and sensor-session
release. A fresh **60-second bounded highlights replay** then made real calls:
9 ASR/diarization/Decider calls, 1,504 YuNet/ASD updates and **525 actual ASD
window-score/Torch-forward calls**, with zero errors/warnings and zero egress.

This replay is **not complete**: wall time 60.168 s, audio consumed 60.128 s,
last video timestamp 60.12 s. Cancellation flushed a tenth VAD segment that was
not transcribed/published before teardown; the report does not disguise that
as an empty ASR response or complete-source result.

### Controlled text — explicitly not video transcription

These were separate, reset-isolated cases with synthetic participant metadata.
No controlled utterance was published during a genuine-media phase.

| Case | Actual route/outcome | Settled wall time |
|---|---|---:|
| Known Alex self-commitment | Decider create_task 0.7614 → act; Alex, due 2026-10-30 | 1,680.9 ms |
| Unknown speaker, Alex in roster | Decider create_task 0.7762 → act; **owner null**, same due date | 1,697.5 ms |
| Empty roster | Decider create_task 0.7924 → act; **valid Decider route, no 422**, owner null | 1,236.4 ms |
| Negated commitment | Decider none 0.3859 → ignore; no item/post | 1,670.1 ms |
| Tentative commitment | Decider create_task 0.5750 shadow → Qwen create_task call → **safety abstention**, none/ignore | 3,269.7 ms |
| Irrelevant weather remark | Decider none 0.9494 → ignore | 1,674.7 ms |
| Explicit media-origin commitment | Decider create_task 0.7835 → **policy deny**, no item/post | 1,692.2 ms |
| Unedited retained ASR decision candidate | Decider none 0.5708 shadow → Qwen none → ignore | 1,773.7 ms |

The tentative test proves the common safety gate vetoed an actual Qwen task call;
it does not mean Qwen itself understood the tentativeness. The neutral negation
case did not naturally escalate, so this first report does **not** claim coverage
of Qwen's negation path.

The real ASR-derived controlled candidate was a **single unchanged retained
utterance**, 105 characters, from design-clip **58.144–66.144 s** (source AMI
IS1009d approximately 448.144–456.144 s), SHA-256:
`f3c7a3955e165f3329cfec953fa00de5c921f2f7be5a83794efe6c59f2fc2977`.
Its actual outcome was none; no joined, cleaned or scripted decision text was
substituted.

All controlled outputs were explicitly local previews. The three explicit
commitment cases each generated one FakeNotifier task record; actual-egress
counters remained zero. Reset cleared these before subsequent cases.

Seven intentional network-gate probes were rejected. There were **zero unexpected
external connection attempts**; recorded real connections used only
127.0.0.1 ports **8765, 8000 and 8770**. Counts are connections, not HTTP-request
counts; provider wrappers separately count calls.

## Final validator revision: repeat run and negation escalation

The final script additionally records scored face/box examples, bus-observed
utterance-to-verdict latency and bounded action provenance. It includes the exact
previously failing synthetic negation context as a separate regression case.
The real model decides whether that case escalates; no route is forced.

Executed:

```sh
docker exec lococisco-demo python /app/deploy/demo/validate_backend.py \
  --media /runtime/media/ami-design-decisions.mp4 --include-public-examples
```

Final-revision report:

```text
/runtime/validation/backend-20261025T214030Z-1877.json
```

It also exited **0**, with no failed checks, errors, warning logs or unexpected
network attempts. Seven intentional gate self-tests were rejected. All connections
were again loopback to 8000, 8765 or 8770. Warmup was **1,725.4 ms**; thresholds
remained **0.60 / 0.40 / 0.50**. No recorded source changed during either run.
Only the validator itself changed between the two captured source manifests.

For exact reproduction, validator SHA-256:

* First default run:
  `013a8f8d23e367b3e60ccb089ea87c1f985167e2fbae0b5feeaa11230e9f19a7`
* Final revision:
  `f38ca44931485cebbf85f1900bfb7fce77c2c69d2d12db8afba76dbd5ee9db41`

The repeat design run completed all **180 seconds at 1x**, with **181.443 s**
wall time, 37 actual ASR/diarization/Decider requests, 4,376 YuNet/ASD updates,
178 embedding evaluations, **1,228 actual ASD window scores and 1,228 Torch
forwards**. There were 124 source-frame drops. It again produced 36 final
none/ignore verdicts and one local fake-lamp action, **no task/decision items**.
Bus-observed utterance-to-verdict p95 was **2,568.9 ms**; speech-end-to-segment
publication p95 was **335.7 ms**. These are different portions of the pipeline.

A real scored-track example was `f1`, normalized box
`[0.6143, 0.5421, 0.0693, 0.1096]`, right seat, speaking **0.003**. This low
score is still an actual returned model-derived score, not a synthetic positive.

The additional **60-second design replay was bounded, not complete**:
60.170 s wall time, 12 completed ASR/diarization/Decider requests, 1,492
YuNet/ASD updates and **530 actual ASD window/Torch evaluations**. Its thirteenth
VAD segment was not transcribed before deliberate teardown. All reset checks
passed again and all actual-egress counters remained zero.

### Confirmed semantic limit: isolated “Brightness.” activates fake lamp

The optional retained-public examples locate the actual action:

* Design-clip **145.312–146.784 s**, source approximately **535.312–536.784 s**.
* Unchanged ASR utterance: **“Brightness.”**
* Decider: `set_light`, `scene: bright`, probability mass **0.8215**.
* Policy: `act`; only the local FakeLight changed.

This is an **unsupported automatic room action / apparent false positive** in a
remote-control design discussion. A single product-feature noun does not establish
a request to change the room. The validator exposes the outcome but does not
change the core or suppress the report. Successful execution and zero exceptions
must not be presented as correct room-intent grounding.

The retained decision utterance at **58.144–66.144 s** produced a
`log_decision` **shadow** at **0.4815** in the actual media context, followed by
Qwen none and an ignore verdict. Again, this is **not** a logged decision.

### Naturally escalated negation now exercised

The separate synthetic `negation_context_regression` case uses the previously
failing meeting context, explicitly recorded in JSON:
`CONTROLLED integration test — NOT video transcription. Here: Alex.`
Its synthetic input remains **“I won't send the design notes by Friday”**.

Actual result: Decider none **0.5132** shadow → Qwen none, adapter score **0.9**
explicitly uncalibrated → policy ignore; **no item, post or DM**, **2,019.2 ms**
settled wall time. `negation_qwen_path_exercised` is **true** in this report.
The neutral negation case also remained none/ignore.

All nine controlled cases passed. Known, unknown-with-roster and empty-roster
commitments again produced grounded local task previews through the real Decider
route (owners Alex/null/null, due 2026-10-30); media-origin commitment was denied.
Tentative speech again naturally escalated: Qwen returned none this time,
whereas the first run required a common safety veto of its create_task call.
The real-ASR-derived controlled candidate remained none/ignore. These observed
differences are not a deterministic-output or calibrated-accuracy guarantee.

## Limits and interpretation

* This is actual backend integration coverage, not an ASR/DER/ASD accuracy
  benchmark. No manually labelled reference transcript, speaker truth or decision
  truth was supplied. The conservative all-unknown identity result must not be
  replaced by invented participant names.
* No genuine-video task/decision success has been established by this report.
  Media-origin policy rejection is tested with explicitly synthetic metadata in
  the separate controlled phase; both videos' observed origins were `person`.
* Every device action is local fake I/O. No Discord message was delivered, and
  clearing local preview records cannot recall already-delivered external messages.
* Readiness alone is insufficient: the report separately proves real sensor
  forward passes and captures warning/error logs as well as `bus.errors`.
* No frontend, browser rendering, recording quality, physical-device action,
  external Discord transport, or end-to-end browser latency is validated.
