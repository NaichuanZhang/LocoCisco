# tx-live: the per-person transcriber in real time

Branch `ws/tx-live`. Everything measured here ran on the shared x86_64 dev box (32 cores, no GPU) while five
other workstreams were running: the 1-minute load average was 38-100 during the 1x runs (see the load column);
every latency number carries that load. Thread budget per process: `OMP_NUM_THREADS=6`, faster-whisper
`cpu_threads=4`, Silero VAD 1, WeSpeaker 1, YuNet/OpenCV 1.

## 1. What I built

| Path | What |
|---|---|
| `unspoken/transcribe/live/engine.py` | `LiveTranscriber`: audio -> Silero VAD (segments cut at 400 ms pauses, capped at 8 s) -> partial ASR on the open segment every 1 s of new audio (latest wins, last 5 s window, throttled when the ASR runs above 0.5 RTF) and final ASR + CAM++ voice label on close; video -> YuNet faces + MotionASD at <= 10 Hz when the detector is free (stale frames skipped, never queued); per-segment ASD evidence per seat; revisable voice->seat binding; captions `{t_emit, start, end, speaker, text, final, ...}` |
| `unspoken/transcribe/live/binding.py` | `SeatBinder`: duration-weighted ASD evidence per (voice, seat); a voice binds to the seat holding >= 60 % of its evidence with >= 40 % of the seat owner's per-second score; several diarizer voices may merge into one on-camera person **only if their centroids are similar** (cosine >= 0.30); re-evaluated after every final, changes re-emit earlier captions as revisions; per-segment strong ASD (>= 0.5, margin 0.25) overrides the binding |
| `unspoken/transcribe/live/captions.py` | `Caption`, `CaptionLog` (captions.jsonl), `build_transcript` -> `unspoken.transcribe.contracts.Transcript` (latest final per segment, words with speaker, `speakers[id].seat`) |
| `unspoken/transcribe/live/metrics.py` | latency.json: final latency p50/p95 (speech end -> first final), partial staleness, first-partial delay, revision delay, RTF, CPU RTF, model busy RTF, frames taken/detected/stale/dropped |
| `unspoken/transcribe/live/web.py` | live viewer, **127.0.0.1 only**, stdlib `ThreadingHTTPServer`: `/` page, `/stream.mjpg` (frames with every face boxed, the talking face in its person's colour, clock + last latency in the HUD), `/events` SSE captions coloured by person (partials dimmed, revisions replace), `/state` JSON |
| `unspoken/transcribe/live/render.py` | demo renderer: burns the logged captions into the video at their real `t_emit` (never re-timed), boxes from faces.jsonl, muxes the original audio with ffmpeg; 10-frame contact sheet |
| `unspoken/transcribe/live/bench.py` | the results table below from run dirs + baseline scores |
| `unspoken/transcribe/live/__main__.py` | the CLI (file replay at `--speed`, `--fast`, or `--camera/--mic` live; `--profile gb10` for the real registry) |
| `tests/transcribe/live/` | 19 tests: binder rules, caption log/transcript round trip, metrics, engine end to end on the sensors fixture WAV (real Silero VAD, truth-driven fake ASR/diarizer, fake faces/ASD; strong-ASD seats, weak-ASD binding + revision, audio only, empty text), viewer HTTP endpoints, renderer timeline + mp4/PNG, CLI with an injected fake registry. No downloads, no servers; whole suite: `860 passed, 10 skipped` |
| `requirements/transcribe-live.txt` | no new Python dependency (re-pins what it uses from sensors.txt); system ffmpeg for the demo mux |
| `testdata/ami-runs/live/live/<case>/` | 1x runs: transcript.json, captions.jsonl, faces.jsonl, latency.json, scores.json, uptime before/after |
| `testdata/ami-runs/live/demo/duo-AC-dev-live.mp4`, `...-sheet.png`, `viewer-frame.jpg` | the demo (704x478, 25 fps, AAC audio) and a frame grabbed from the live viewer during a run |

## 2. How to run

```bash
cd unspoken-wt/tx-live
UV_CACHE_DIR=/local/home/zhangnai/kirocrew-sessions/2026-10-15-strands-decider-research/hosting/uv-cache scripts/dev-venv.sh
uv pip install --python .venv/bin/python -r requirements/sensors.txt -r requirements/transcribe-eval.txt -r requirements/transcribe-live.txt
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 T=../../testdata

# real time, like a camera + mic, with the viewer (8790 was taken by another workstream on this box: use any free loopback port)
.venv/bin/python -m unspoken.transcribe.live --video $T/ami/IS1009a-duo-AC-dev/video.mp4 --audio $T/ami/IS1009a-duo-AC-dev/audio.wav \
    --config $T/ami-seats/sensors.toml --out $T/ami-runs/live/live/IS1009a-duo-AC-dev/transcript.json --web 127.0.0.1:8798 --print
#   --audio is optional (the mp4's AAC track is decoded otherwise); --speed 2.0 plays faster; --fast = no pacing (accuracy only)
#   --camera 0 --mic Jabra --duration 600 uses the real devices; --profile gb10 uses config/models.toml (speech server ASR)
.venv/bin/python -m eval.transcribe.score --ref $T/ami/IS1009a-duo-AC-dev --hyp $T/ami-runs/live/live/IS1009a-duo-AC-dev/transcript.json
.venv/bin/python -m unspoken.transcribe.live.render --video $T/ami/IS1009a-duo-AC-dev/video.mp4 \
    --run $T/ami-runs/live/live/IS1009a-duo-AC-dev --out $T/ami-runs/live/demo/duo-AC-dev-live.mp4 --sheet $T/ami-runs/live/demo/duo-AC-dev-live-sheet.png
.venv/bin/python -m unspoken.transcribe.live.bench --runs $T/ami-runs/live/live --baseline $T/ami-baseline
.venv/bin/python -m pytest -q
```

## 3. Results

Baseline = today's live pipeline replayed offline (`testdata/ami-baseline`). "live 1x" = this engine at real time,
one process per case, four cases in parallel on the loaded box (dev: load 38 -> 100 during the three parallel
dev runs; full: 44 -> 30). Latency = `t_emit - segment end`, where the segment end is the VAD's (last speech
window + 200 ms pad; the VAD needs 400 ms of silence to close, so ~0.4 s of every number is the VAD wait).
RTF = wall / audio for the whole run (1.000 = kept up with real time); CPU RTF = process CPU seconds / audio
seconds (cores busy on average). Threads: ASR 4 + VAD 1 + speaker 1 + faces 1 under `OMP_NUM_THREADS=6`.

| case | split | method | WER | cpWER | DER | sa_acc | seat_acc | hyp spk | final lat p50 / p95 s | partial p50 s | partials / revisions | RTF | CPU RTF | load 1-min before->after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| duo-AC-dev | dev | baseline | 0.091 | 0.235 | 0.096 | 0.913 | 0.913 | 7 | - | - | - | - | - | - |
| duo-AC-dev | dev | **live 1x** (first run) | **0.072** | **0.125** | **0.074** | 0.909 | 0.909 | 8 | **1.25 / 2.43** | 0.78 | 83 / 0 | 1.000 | 1.85 | 49.5 -> 48.5 |
| duo-AC-dev | dev | **live 1x** (final code, demo source) | **0.072** | **0.125** | **0.074** | 0.909 | 0.909 | 8 | **1.10 / 2.41** | 0.52 | 25 / 0 | 1.000 | 0.94 | 42.6 -> 40.6 |
| duo-DB-dev | dev | baseline | 0.165 | 0.266 | 0.162 | 0.918 | 0.918 | 8 | - | - | - | - | - | - |
| duo-DB-dev | dev | live 1x | 0.165 | **0.253** | **0.088** | 0.877 | 0.877 | 8 | 2.15 / 41.9 | 1.00 | 19 / 2 | 1.224 | 1.01 | 37.8 -> 101.3 |
| L-room-dev | dev | baseline | 0.207 | 0.455 | 0.151 | 0.791 | 0.675 | 7 | - | - | - | - | - | - |
| L-room-dev | dev | live 1x | 0.224 | 0.571 | 0.314 | 0.674 | **0.778** | 6 | 9.84 / 43.4 | 1.40 | 36 / 1 | 1.241 | 2.21 | 37.8 -> 99.6 |
| R-room-dev | dev | baseline | 0.207 | 0.455 | 0.153 | 0.791 | 0.068 | 7 | - | - | - | - | - | - |
| R-room-dev | dev | live 1x | 0.224 | 0.484 | 0.206 | 0.751 | **0.315** | 8 | 4.15 / 18.2 | 1.09 | 41 / 4 | 1.094 | 1.96 | 37.8 -> 99.7 |
| duo-AC | full | baseline | 0.229 | 0.321 | 0.165 | 0.857 | 0.857 | 8 | - | - | - | - | - | - |
| duo-AC | full | live 1x | 0.242 | **0.310** | **0.144** | 0.825 | 0.825 | 9 | **1.40** / 19.4 | 0.71 | 80 / 0 | 1.000 | 1.24 | 44.1 -> 29.7 |
| duo-DB | full | baseline | 0.208 | 0.247 | 0.176 | 0.911 | 0.033 | 8 | - | - | - | - | - | - |
| duo-DB | full | live 1x (merge, no similarity gate) | 0.230 | 0.661 | 0.460 | 0.620 | 0.246 | 9 | **1.45 / 5.43** | 1.15 | 13 / 34 | 1.000 | 0.66 | 44.1 -> 29.7 |
| duo-DB | full | live, fast, similarity gate (final code) | 0.220 | 0.304 | 0.193 | 0.829 | 0.028 | 10 | n/a | n/a | - / 23 | 0.216 | - | - |
| L-room | full | baseline | 0.338 | 0.497 | 0.272 | 0.726 | 0.808 | 8 | - | - | - | - | - | - |
| L-room | full | live 1x | 0.347 | 0.763 | 0.433 | 0.552 | 0.793 | 6 | **1.67 / 6.86** | 0.73 | 115 / 22 | 1.000 | 1.37 | 44.1 -> 30.1 |
| L-room | full | live, fast, similarity gate (final code) | 0.349 | 0.761 | 0.433 | 0.552 | 0.793 | 6 | n/a | n/a | - / 22 | 0.221 | - | - |
| R-room | full | baseline | 0.335 | 0.488 | 0.272 | 0.726 | 0.517 | 8 | - | - | - | - | - | - |
| R-room | full | live 1x | 0.347 | 0.763 | 0.434 | 0.550 | 0.267 | 6 | **1.61 / 8.91** | 0.75 | 144 / 20 | 1.000 | 1.46 | 44.1 -> 29.7 |
| R-room | full | live, fast, similarity gate (final code) | 0.347 | 0.763 | 0.434 | 0.550 | 0.268 | 6 | n/a | n/a | - / 21 | 0.221 | - | - |

Third variant, `--merge-sim 2` (one voice per seat, the baseline's rule; committed code, `--fast`, full cases):

| case | split | method | WER | cpWER | DER | sa_acc | seat_acc | hyp spk | revisions |
|---|---|---|---|---|---|---|---|---|---|
| duo-AC | full | live, fast, no merge | 0.244 | 0.315 | 0.144 | 0.825 | 0.825 | 9 | 0 |
| duo-DB | full | live, fast, no merge | 0.255 | 0.339 | 0.193 | 0.829 | 0.028 | 10 | 23 |
| L-room | full | live, fast, no merge | 0.348 | **0.489** | **0.277** | 0.685 | 0.728 | 10 | 7 |
| R-room | full | live, fast, no merge | 0.348 | 0.572 | 0.390 | 0.601 | 0.052 | 9 | 2 |

(WER differs by up to 0.035 between variants on the same audio: faster-whisper int8 greedy decoding is not
bit-stable across runs on short segments, e.g. one 1.4 s segment decoded as "kiss them." in one run and "I'm
going to be kissing." in another; everything else about the segmentation is identical.)

Which code produced which row: the duo-AC-dev 1x row and the four "live 1x" full rows ran the committed engine
minus the centroid-similarity merge gate (added after the full 1x runs; it changes speaker merging only, not
timing). The "similarity gate" rows are the committed code in `--fast` mode (accuracy only, latencies
meaningless, so n/a). duo-AC-dev was re-run at 1x with the final code and is the source of the demo mp4 (second duo-AC-dev row:
same scores, p50 1.10 s / p95 2.41 s, load 43 -> 41; 25 partials, 63 throttled, because the ASR-speed
throttle engaged while the ASR ran above 0.5 RTF). The first dev run is kept in
`testdata/ami-runs/live/live-first-run-duo-AC-dev/`, the gated and no-merge full runs in
`testdata/ami-runs/live/live-gate-fast/` and `live-nomerge-fast/`.

More latency detail (1x, full cases): first partial 1.95-2.37 s after speech start (p50); partial staleness
p50 0.71-1.15 s; model times per final ASR 1.57-1.92 s (segments average ~4-6 s), speaker label 93-127 ms,
faces 18-19 ms, ASD 1.4 ms; model busy 0.33-0.60 of real time; RSS ~940 MB; warm-up 1.3-1.6 s. Partials
throttled by the ASR-speed rule: 157-345 per full case (the ASR ran at 0.3-0.5 s per audio second under this
load, right at the throttle). Frames: 12.5 fps taken, every second one detected (10 Hz ceiling), 0 stale, the
source itself skipped the other 12.5 fps (`frames_dropped` counts decode-skipped source frames). Revision
delay p50 58-155 s (a binding flips after minutes of evidence).

### Reading the numbers

- **Latency**: p50 1.25-1.67 s from VAD segment end to the final caption (so ~0.85-1.3 s after the VAD closed),
  partials ~0.7-1.1 s behind the audio, the first partial ~2 s after someone starts talking, at RTF 1.000 on
  all four full cases: the engine kept up with real time on a box with load 30-45. The p95 is the host:
  the duo-AC full p95 of 19 s is one backlog (another workstream's burst; the three parallel dev runs at load
  100 show the same thing at p95 42-43 s and RTF 1.22-1.24). The ASR-speed throttle on partials is what
  keeps finals flowing in those stretches; finals are never dropped.
- **duo-AC**: better than the baseline on dev (cpWER 0.125 vs 0.235, DER 0.074 vs 0.096, WER 0.072 vs 0.091)
  and on cpWER/DER on the full case (0.310 vs 0.321, 0.144 vs 0.165), slightly worse WER (0.242 vs 0.229:
  8 s pieces give base.en less context than 20 s pieces) and sa/seat_acc (0.825 vs 0.857: voice-A, the
  first voice, never bound; its evidence share stayed under 60 %).
- **duo-DB**: the ASD evidence is swapped for this pair exactly as in the baseline (seat_acc 0.03 both),
  so D's voice binds to the right seat. Without the similarity gate the right seat then swallowed B's voice
  too (two people -> one hyp speaker: cpWER 0.661, DER 0.460). The gate stops that (cpWER 0.304, DER 0.193,
  sa_acc 0.829) and the wrong seat remains; the fix is a real ASD, see risks.
- **Room cases**: worse than the baseline on cpWER/DER/sa_acc (0.76 vs 0.50, 0.43 vs 0.27, 0.55 vs 0.73) and
  better/equal on seat_acc (L 0.793 vs 0.808, R 0.268 vs 0.517 full; dev 0.778 vs 0.675, 0.315 vs 0.068).
  Two causes, both measured: (1) far-field 8 s segments fragment into more CAM++ voices than 20 s ones, and
  (2) MotionASD on 30 px faces (60 px after the 2x upscale) gives evidence that binds off-camera voices to
  seats (L-room: five voices -> left, R-room: four -> right), which the one-to-one mapping in cpWER/sa_acc
  punishes hard. The per-segment ASD scores on these clips are 0.00-0.03 for everybody: there is no
  usable "who is talking" signal in them for a heuristic; the binder is scale-free on purpose and still
  cannot make something out of noise.
- **Winner**: for the hack-day duo setup (close-ups, one headset mix) the live engine wins on dev and ties or
  beats the baseline offline fusion on the full case while producing captions ~1.3 s after speech ends,
  partials while people talk, and speaker labels that fix themselves. For the room cases the baseline's
  offline voice->seat vote is still better; the live engine should run with `--merge-sim 2` (one voice per
  seat) there until a real ASD lands.

## 4. What runs on the GB10

No code change: the engine asks the registry for the roles `vad`, `asr`, `diarization`, `faces`, `asd`.
`python -m unspoken.transcribe.live --profile gb10 ...` builds `unspoken.models.load_registry("gb10")` (verified
here: it resolves to "Parakeet TDT 0.6B v3 (GPU, NeMo-Speech.cpp)" for asr). The `openai-audio` provider returns
the same `Transcript` (text, words, latency_ms) the engine consumes, so partials and finals go to the speech
server on the GPU; VAD, CAM++, YuNet, MotionASD stay in-process on the CPU (all have aarch64 wheels, see
requirements/sensors.txt). Only ASR is heavy enough to matter: on the GB10 a partial/final is a ~50-150 ms
HTTP call instead of 0.9-1.9 s of CPU, so the throttle never engages and the latency floor becomes the VAD
wait (400 ms) + pad (200 ms) + the call.

Blocks for `config/models.toml` (the gb10 profile already has these roles; the only edits are the VAD cut
settings this engine was measured with):

```toml
[profiles.gb10.asr]                          # NeMo-Speech.cpp speech server: POST /v1/audio/transcriptions
provider = "openai-audio"
url = "http://127.0.0.1:8770/v1"
model = "parakeet-tdt-0.6b-v3"
label = "Parakeet TDT 0.6B v3 (GPU, NeMo-Speech.cpp)"
options = { response_format = "verbose_json", language = "en", diarization = false, timeout_s = 20, health = "/ready" }

[profiles.gb10.vad]                          # in-process Silero VAD v5 (ONNX, CPU)
provider = "python:unspoken.sensors.vad:SileroVAD"
label = "Silero VAD v5 (ONNX, CPU)"
options = { threshold = 0.5, min_silence_ms = 400, max_segment_s = 8 }   # live: cut at pauses, 8 s cap (was 500 / 20)

[profiles.gb10.diarization]                  # WeSpeaker CAM++ embeddings, session-only labels (CPU)
provider = "python:unspoken.sensors.diar:EmbeddingDiarizer"
label = "WeSpeaker CAM++ (ONNX, CPU)"
options = { threshold = 0.55, max_voices = 8 }

[profiles.gb10.faces]
provider = "python:unspoken.sensors.faces:YuNetFaces"
label = "YuNet 2023mar (OpenCV)"

[profiles.gb10.asd]                          # swap for tx-asd's model when it lands; the engine only needs ASDSession
provider = "python:unspoken.sensors.asd:MotionASD"
label = "MotionASD (heuristic)"
```

Run on the day: `python -m unspoken.transcribe.live --profile gb10 --camera 0 --mic Jabra --config config/sensors.toml
--out /tmp/live/transcript.json --web 127.0.0.1:8790` (check the port against docs/PLAN.md s4; the viewer refuses
anything but loopback). The CLI's `--asr-threads` is ignored by the server ASR. 20 Arm cores: the in-process
roles use 3 threads; `OMP_NUM_THREADS=6` is fine.

## 5. Not verified, contract requests, deps

Not verified: anything on the GB10 (no aarch64 host here); the `--camera/--mic` path (no devices on the dev
box; it is the sensors `CameraSource`/`MicSource` unchanged); latency under a quiet host (every 1x number
above ran at load 30-100 on 32 cores); the gb10 profile end to end (the registry resolves, the speech server
call was not made); `--speed` other than 1.0 at scale (unit-tested only).

Contract change requests (none required; suggestions):
- `unspoken/transcribe/contracts.py`: `Utterance` has no field for "when did this line reach the user";
  captions.jsonl carries `t_emit`/`latency` next to transcript.json instead. If the scorer ever reports
  latency, an optional `Utterance.t_emit: float | None = None` would let one file carry both.
- `eval/transcribe/score.py`: DER on VAD-cut hyp utterances counts every in-utterance pause as miss (the
  reference segments span pauses); a `--hyp-pad` or scoring on word spans only would compare segmenters fairly.
- `unspoken/sensors/vad.py`: the open segment is read from the private `_Stream._seg` (buf/start_w/window_s) for
  partials; a public `VADStream.open_segment() -> (start, end, pcm) | None` on the contract would make that
  official (the engine degrades to no partials when a VAD lacks it, tested).
- `unspoken/sensors/diar.py`: `DiarSession.centroids`/`d.embed` are used read-only for partial voice peeks
  and the merge gate; `peek(pcm) -> VoiceLabel` and `similarity(a, b)` on `DiarSession` would be the clean form.
- `unspoken/sensors/pipeline.py` needs nothing from me; `fusion.py` could take the `SeatBinder` (many voices
  per seat + revisions) once an ASD worth trusting exists.

Deps added: none (requirements/transcribe-live.txt re-pins sensors.txt entries it uses; all have aarch64
cp312 wheels per that file; ctranslate2's aarch64 wheel is CPU-only, which is why the GB10 uses the speech
server). System: ffmpeg for the demo mux (mpeg4 fallback used here: this ffmpeg build has no libx264).

## 6. Risks

- ASD quality decides the seat: MotionASD on AMI gives ~0.01-0.03 per-segment probabilities; on duo-DB it
  is swapped, on room views it is noise. The binder's merging makes a wrong ASD costlier (two people -> one
  seat) than the baseline's single-voice vote; the similarity gate limits it, `--merge-sim 2` removes it.
  Needs tx-asd's model; the engine only needs an `ASDSession`.
- Host load: at RTF > 1 the finals queue grows without bound (never dropped); the partial throttle buys
  ~40 % but a sustained overload needs a smaller ASR or the GPU path.
- 8 s segments cost ~1 WER point of context versus 20 s and fragment far-field voices more; `--max-segment-s`
  and `--min-silence-ms` are the knobs, measured only at 8 / 400.
- Revisions arrive late (p50 1-2.5 min): a caption shown with "voice B" may change to "left" minutes later;
  the viewer handles it, downstream consumers (notes, recap) must take the latest final per `seg`.
- Viewer: loopback only, no auth (local tool); the default port 8790 collided with another service on this box.
