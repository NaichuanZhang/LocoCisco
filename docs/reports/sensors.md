# SENSORS workstream report (ws/sensors)

Worktree `unspoken-wt/sensors`, branch `ws/sensors`, commit `84a6694` (+ this report). STOP condition met: the
fixture WAV replays into 12 correct `audio.segment` events (text + session-stable voice labels), the whole suite
passes (424 passed, 2 pre-existing live skips).

## 1. What was built

| Path | One line |
|---|---|
| `unspoken/sensors/__init__.py` | `async start_sensors(bus, registry, config, clock) -> SensorHandle` (fixed interface), `load_sensors_config`, `seat_zones` |
| `unspoken/sensors/pipeline.py` | audio -> VAD -> [ASR \|\| diarizer] -> `MT.SEGMENT` (asr_ms, diar_ms, PCM dropped); video -> faces -> ASD -> `MT.SPEAKERS` at 5 Hz; newest-frame-only video, audio never dropped; `SensorHandle.stop()` wipes voices/faces; `stats` for the X-ray |
| `unspoken/sensors/audio.py` | `MicSource` (sounddevice/PortAudio, device by name substring, software resample when the device lacks 16 kHz; `arecord` subprocess fallback), `WavFileSource` (real time or fast, paced), PCM helpers; nothing writes audio |
| `unspoken/sensors/vad.py` | `SileroVAD(spec)` (v5 ONNX, onnxruntime CPU, min speech/silence/pad/max-segment settings), `EnergyVAD(spec)` fallback; shared segmenter |
| `unspoken/sensors/asr_local.py` | `FasterWhisperASR(spec)` (model/device/compute_type/language options; decode in a worker thread; word timestamps; conf from avg logprob) |
| `unspoken/sensors/diar.py` | `EmbeddingDiarizer(spec)`: WeSpeaker VoxCeleb CAM++ ONNX + kaldi-native-fbank (80-bin, CMN) + online cosine clustering -> `voice-A`, `voice-B`, ...; `reset()` forgets every voice |
| `unspoken/sensors/video.py` | `CameraSource` (OpenCV V4L2 in a thread, size-1 mailbox, fps pacing, downscale), `Mp4FileSource` (real time / paced by audio position); frames never written |
| `unspoken/sensors/faces.py` | `YuNetFaces(spec)` via `cv2.FaceDetectorYN` + `IoUTracker` (stable ids, EMA boxes, lost-track timeout) + seat zones from `config/sensors.toml` (sticky seats, `screen` zone) |
| `unspoken/sensors/asd.py` | `MotionASD(spec)`: mouth-region motion minus face motion, gated by voice energy, single-face floor, screen faces excluded; Light-ASD documented as next step |
| `unspoken/sensors/replay.py` | `python -m unspoken.sensors.replay --audio x.wav [--video y.mp4] [--profile P] [--fast] [--local] [--set role.key=v]` -> JSON lines |
| `unspoken/sensors/check.py` | `python -m unspoken.sensors.check`: devices seen + every in-process role loads (day-of preflight) |
| `unspoken/sensors/local_registry.py` | `SensorsRegistry`: Registry-shaped wrapper that resolves `python:module:Factory` providers; used by replay/check until `unspoken.models` lands; its docstring is the `config/models.toml` block |
| `unspoken/sensors/assets.py`, `fetch_models.py` | model lookup order (`$UNSPOKEN_MODELS_DIR` -> `~/.cache/unspoken/models` -> vendored) and sha256-pinned downloader |
| `unspoken/sensors/fakes.py` | `FakeASR`, `FakeDiarizer`, `FakeFaceTracker`, `FakeASD`, `FakeVideoSource`, `FakeRegistry` for tests and other workstreams |
| `unspoken/sensors/models/` | vendored `silero_vad.onnx` (2.3 MB, MIT), `face_detection_yunet_2023mar.onnx` (232 KB, MIT), licences, `SOURCES.md` with URLs + sha256 |
| `config/sensors.toml` | audio/video devices, chunk/fps/size, `speakers.hz`, 4 seat zones + `screen` |
| `requirements/sensors.txt` | exact pins + aarch64 notes |
| `tests/sensors/` | 41 tests: config, audio sources, VAD (both), ASR WER, diarization purity + clustering, faces + tracker, MotionASD, pipeline end to end with fakes, replay CLI, mp4 source, local registry |
| `tests/sensors/fixtures/` | `meeting3.wav` (36 s, 1.16 MB, 3 Piper voices, 12 lines) + `meeting3.json` ground truth + `make_meeting_wav.py`; `face_256.jpg` (public-domain NASA portrait) + `README.md` provenance |
| `docs/SENSORS.md` | architecture, run commands, events, models/licences, seat zones, privacy, aarch64 table, Light-ASD steps |

## 2. How to run

```bash
cd unspoken-wt/sensors
UV_CACHE_DIR=.../uv-cache scripts/dev-venv.sh
uv pip install --python .venv/bin/python -r requirements/sensors.txt
.venv/bin/python -m unspoken.sensors.fetch_models                 # 29 MB WeSpeaker ONNX -> ~/.cache/unspoken/models
.venv/bin/python -m unspoken.sensors.check                        # devices + all five roles load
.venv/bin/python -m unspoken.sensors.replay --audio tests/sensors/fixtures/meeting3.wav --fast --local
.venv/bin/python -m unspoken.sensors.replay --audio x.wav --video y.mp4 --profile gb10   # real time, models registry
.venv/bin/python -m pytest -q                                      # whole suite
```

In code: `handle = await start_sensors(bus, registry, load_sensors_config(), clock)`; `await handle.stop()`.
For replay from the runtime set `config["audio"]["path"]`, `config["video"]["path"]`, `config["replay"] = {"realtime": False}`.

## 3. Verification (this x86 CPU box, no GPU, no mic, no camera)

- `pytest -q` (whole suite): **424 passed, 2 skipped** in 25 s (383 existing + 41 sensors; the 2 skips are the pre-existing live tests). `pyflakes` clean on `unspoken/sensors` and `tests/sensors`.
- VAD on the fixture: Silero and Energy both return exactly 12 segments, each covering >= 80 % of its line, none spanning two lines; Silero 136 ms for 36 s (266x real time), chunk-size independent.
- ASR (faster-whisper base.en, CPU int8, HF cache, offline): **WER 3.3 %** over the 12 lines (bound in test 15 %), 420-630 ms per 2-3 s segment, conf 0.6-0.8.
- Diarization (CAM++): same-voice cosine min 0.70 / mean 0.78, other-voice max 0.37 / mean 0.15; online clustering at threshold 0.55 -> **3 voices, purity 1.00**, A/B/C assigned in first-heard order; 67 ms per segment (ResNet34-LM also tested: 0.63/0.27, 89 ms; CAM++ chosen).
- Faces: YuNet finds the fixture face, keeps the track id across frames, assigns seats, two pasted faces -> two seats, blank frame -> no track; 8-16 ms per 640x360 frame.
- MotionASD unit tests: moving mouth + voice -> speaking > 0.5 for the talker and < 0.5 for the still face; no voice or no motion -> < 0.2; single-face floor and screen exclusion.
- Full replay, fixture WAV only (`--fast --local`, real models): 12 `audio.segment` events, voices A,B,C,A,B,C,A,B,C,A,B,A (matches Priya/Sam/Lee), texts correct, `asr_ms` avg 470, `diar_ms` avg 75, 36 s of audio in **6.2 s wall**.
- Full replay with a synthetic room clip (face in seat s1 for 18 s then s3, mouth flicker every other second; clip lives in scratch, not committed): 180 `video.speakers` events at 5 Hz (gaps 0.18-0.31 s), seats s1 then s3 exactly, 358 source frames dropped on purpose (15 fps -> 5 Hz), faces 8 ms/frame; mean speaking probability 0.79 when mouth moves and voice is present, 0.13 mouth-only, 0.25 neither (the audio gate works; 0.68 voice-only is the single-face floor by design). Real-time mode at 6x speed: same 12 segments / 180 speakers, 7.5 s wall.
- Fixture build: Piper TTS 1.8.0 on CPU, 3 voices, 4.7 s to generate.

## 4. Not verified here, and how to verify on the GB10

- **Live microphone and camera** (this box has neither PortAudio nor `arecord`, nor a camera). On the GB10:
  `apt install libportaudio2 alsa-utils` (or rely on the arecord fallback), `python -m unspoken.sensors.check`
  must list the speakerphone under audio inputs and `/dev/videoN` under video devices; set `[audio] device`
  and `[video] device` in `config/sensors.toml`; run
  `python -m unspoken.meeting --profile gb10 --mode live` (or the replay CLI with no `--audio`, once the
  runtime wires it) and watch `audio.segment` lines arrive within ~1 s after a sentence ends and
  `video.speakers` at 5 Hz with the right seats. If the speakerphone does not offer 16 kHz, MicSource opens
  its native rate and resamples; `arecord -D plughw:CARD=...` also resamples.
- **aarch64 wheels actually install**: `uv pip install -r requirements/sensors.txt` on the box; every pinned
  native package lists a cp312 manylinux aarch64 wheel on PyPI (see section 5), but nothing was installed on
  Arm here. Then `python -m unspoken.sensors.check` -> five `ok` rows.
- **CPU budget on the 20 Arm cores**: expect Silero + CAM++ + YuNet + MotionASD well under one core;
  faster-whisper base.en int8 ~1-2 cores while someone speaks (`threads` option). Measure with the replay's
  `done:` summary (`asr_ms_avg`, `faces_ms_avg`) and `top` during a live run. If ASR lags, point the `asr` role
  at the GPU speech server (NeMo-Speech.cpp / whisper.cpp arm64 CUDA 13) in `config/models.toml`; no code change.
- **Seat zones for the real room**: record a 20 s clip of the chairs, replay with `--video`, adjust `[seats]`.
- **Light-ASD** not built (see docs/SENSORS.md "next step"); MotionASD is what ships.
- `--profile` through `unspoken.models.load_registry` is untested (package not in this worktree); the replay
  falls back to the local registry with a message.

## 5. Contract changes, deps, config blocks for other owners

Contract change requests: **none required.** Two notes for the integrator:
- `FaceTrack.seat` docstring says "seat id from config/room.toml"; the brief puts seat zones in
  `config/sensors.toml`, which is what this code reads. Suggest editing the docstring to `config/sensors.toml`.
- `SegmentData.voice` is `null` when the diarization role is disabled and `"voice-?"` for audio shorter than
  0.6 s (conf 0); consumers should treat both as "no label".

Deps added (`requirements/sensors.txt`, all with cp312 manylinux aarch64 wheels on PyPI as of 2026-10-24):
`numpy==2.5.4`, `onnxruntime==1.31.0` (CPU; no prebuilt aarch64 CUDA build, CPU is enough here),
`opencv-python-headless==4.14.0.94`, `faster-whisper==1.2.1` (pure) + `ctranslate2==4.8.3` (aarch64 wheel is
CPU-only; GPU ASR = speech server), `tokenizers==0.23.3`, `huggingface-hub==2.3.0` (pure), `av==19.0.2`,
`kaldi-native-fbank==1.22.4`, `sounddevice==0.5.6` (pure; needs system `libportaudio2`, else `arecord`).
Lacking aarch64 wheels: none of the above. Not added: torch (Light-ASD only), silero-vad pip package (needs
torch; pinned ONNX vendored instead), piper-tts (fixture generation only).

Config blocks the **models** owner must add to `config/models.toml` (dev profile; gb10 the same unless the
`asr` role points at the GPU speech server):

```toml
[profiles.dev.vad]
provider = "python:unspoken.sensors.vad:SileroVAD"
label = "Silero VAD v5 (ONNX, CPU)"
[profiles.dev.vad.options]
threshold = 0.5
min_silence_ms = 500

[profiles.dev.asr]
provider = "python:unspoken.sensors.asr_local:FasterWhisperASR"
model = "Systran/faster-whisper-base.en"
label = "faster-whisper base.en (CPU int8)"
[profiles.dev.asr.options]
device = "cpu"
compute_type = "int8"
language = "en"
threads = 4

[profiles.dev.diarization]
provider = "python:unspoken.sensors.diar:EmbeddingDiarizer"
label = "WeSpeaker CAM++ (ONNX, CPU)"
[profiles.dev.diarization.options]
threshold = 0.55
max_voices = 8

[profiles.dev.faces]
provider = "python:unspoken.sensors.faces:YuNetFaces"
label = "YuNet 2023mar (OpenCV)"

[profiles.dev.asd]
provider = "python:unspoken.sensors.asd:MotionASD"
label = "MotionASD (heuristic)"
```

The registry must call `Factory(spec)` and may pass `options={...}` from the TOML in `spec.options`; every
factory also accepts plain kwargs. `python -m unspoken.models check` can use each provider's `healthy()`
(sync for vad/diarizer/faces/asd, async for asr).

For the **meeting** owner: call `start_sensors(bus, registry, load_sensors_config(), clock)` in live/replay
mode and `await handle.stop()` on meeting end (this is what wipes voices and faces); `handle.stats.summary()`
and `handle.models` are ready for the X-ray / room screen. For the **deploy** owner: pre-seed
`~/.cache/unspoken/models/wespeaker_en_voxceleb_CAM++.onnx` (sha256 in `assets.py`) and the faster-whisper
HF cache on the USB; install `libportaudio2` or `alsa-utils`.

## 6. Open risks

- MotionASD is a heuristic: it tells apart who talks when faces are reasonably still and lit; two people
  moving at once or a far camera will confuse it. Light-ASD is the fix (steps in docs/SENSORS.md).
- Diarization threshold 0.55 is tuned on TTS voices and VoxCeleb-trained embeddings; two similar real voices
  in a noisy room may merge (raise to 0.6-0.65 if labels merge, lower if one person splits). Short
  interjections (< 0.6 s) get `voice-?`.
- faster-whisper base.en is English-only and hears "sync" as "sink"; for Chinese lines switch `model` to a
  multilingual size (small/medium) or the GPU speech server.
- Live capture untested on this box (no devices); the sounddevice/arecord paths are exercised only for their
  error branches in tests.
- opencv-python-headless pinned to 4.14 while 5.0.0 is out; untested.
