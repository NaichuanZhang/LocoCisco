# Sensors: speakerphone + webcam -> `audio.segment` and `video.speakers`

Owner: sensors workstream. Code: `unspoken/sensors/`. Config: `config/sensors.toml`. Tests: `tests/sensors/`.

```
 USB speakerphone ─► audio.MicSource ──32 ms PCM16──► vad.SileroVAD ──SpeechSegment──► [asr || diarizer] ─► MT.SEGMENT
      (or WavFileSource)                 │                                                 PCM dropped here
                                         └─ last 2 s ring ─────────────────────────────┐
 USB webcam ───────► video.CameraSource ──newest frame──► faces.YuNetFaces ─► asd.MotionASD ─► MT.SPEAKERS (5 Hz)
      (or Mp4FileSource)                   (older frames dropped)        + seat zones, "screen" zone
```

Everything runs in-process on the CPU of the box; the ASR can also be a server (the registry's
`openai-audio` provider) without any change here. Both paths stamp times on the one meeting clock passed to
`start_sensors`, so a replayed meeting and a live one produce identical events.

## Run

```bash
cd <worktree> && scripts/dev-venv.sh && uv pip install --python .venv/bin/python -r requirements/sensors.txt
.venv/bin/python -m unspoken.sensors.fetch_models            # 29 MB speaker model -> ~/.cache/unspoken/models
.venv/bin/python -m unspoken.sensors.replay --audio tests/sensors/fixtures/meeting3.wav --fast          # JSON lines
.venv/bin/python -m unspoken.sensors.replay --audio x.wav --video y.mp4 --profile gb10                   # real time
.venv/bin/python -m pytest -q tests/sensors                   # 41 tests, ~10 s, no servers, no hardware
```

`replay` uses `unspoken.models.load_registry(profile)` when the models package is present, else the
in-process providers directly (`--local`). `--set asr.model=Systran/faster-whisper-small.en` overrides a role.

From the meeting runtime:

```python
from unspoken.sensors import start_sensors
handle = await start_sensors(bus, registry, load_sensors_config(), clock)   # publishes MT.SEGMENT / MT.SPEAKERS
...
await handle.stop()        # cancels capture, forgets every voice and face, drops buffers
```

`config` is the dict from `config/sensors.toml` (`unspoken.sensors.load_sensors_config()`); set
`config["audio"]["path"]` / `config["video"]["path"]` for replay (`config["replay"] = {"realtime": False}` for a
fast run) or pass source objects in `config["audio"]["source"]` / `config["video"]["source"]`.

## Events

`MT.SEGMENT` (`audio.segment`, source `sensors.audio`, `ts` = segment end):
`{segment_id, start, end, text, lang, asr_conf, voice, voice_conf, asr_ms, diar_ms}`; `voice` is
`voice-A`, `voice-B`, ... (session-stable, never a name; `null` when diarization is disabled, `voice-?` for
audio under 0.6 s). Segments with an empty transcript are not published (counted in `SensorHandle.stats`).

`MT.SPEAKERS` (`video.speakers`, source `sensors.video`, `ts` = frame time, ~5 Hz):
`{t, faces: [{track_id, seat, bbox: [x, y, w, h], speaking}], face_count, screen_talking}`; `face_count`
excludes faces in the `screen` zone; `screen_talking` is true when a face in that zone has
`speaking >= speakers.speaking_threshold`.

## Models (all local, all permissive)

| Role | Factory (`config/models.toml` `provider = "python:..."`) | Model | Licence | Where | CPU cost here (x86, 1 core) |
|---|---|---|---|---|---|
| vad | `unspoken.sensors.vad:SileroVAD` (fallback `vad:EnergyVAD`, no model) | Silero VAD v5.1.2 ONNX, 2.3 MB | MIT | vendored `unspoken/sensors/models/` | 0.3 ms per 32 ms window (266x real time) |
| asr | `unspoken.sensors.asr_local:FasterWhisperASR` | faster-whisper `Systran/faster-whisper-base.en`, int8 | MIT | HF cache | 0.45 s per 2-3 s segment, WER 3 % on the fixture |
| diarization | `unspoken.sensors.diar:EmbeddingDiarizer` | WeSpeaker VoxCeleb CAM++ ONNX (sherpa-onnx export), 29 MB, 512-d | Apache-2.0 | `fetch_models` -> `~/.cache/unspoken/models` | 67 ms per segment; 3 TTS voices: same-voice cosine >= 0.70, other-voice <= 0.37 |
| faces | `unspoken.sensors.faces:YuNetFaces` | YuNet 2023mar (opencv_zoo), 232 KB | MIT | vendored | 8-16 ms per 640x360 frame |
| asd | `unspoken.sensors.asd:MotionASD` | none (mouth motion x voice energy) | - | - | < 1 ms per face |

Options each factory reads from `RoleSpec.options` are in the module docstrings (`threshold`, `min_silence_ms`,
`model`, `device`, `compute_type`, `language`, `max_voices`, `score_threshold`, ...). Sources and sha256 of
every model file: `unspoken/sensors/models/SOURCES.md`.

Diarization labels are an online cosine clustering over the running centroids of the voices heard so far
(`threshold` 0.55, `max_voices` 8). They are stable for the session and wiped by `reset()` / `stop()`.

## Seat zones

`[seats]` in `config/sensors.toml`: `id = [x, y, w, h]` in 0..1 of the downscaled frame. A face gets the seat
holding most of its box (>= 30 %), else the one holding its centre; a face that drifts out of every zone
keeps its last seat. The zone named `screen` is the room display (remote participants, videos): a talking
face there sets `screen_talking` and is not counted as a person. Rehearsal: run the replay on a clip of the
real room and move the rectangles until each chair is its own zone.

## Privacy, by construction

- No audio or video is written to disk anywhere in this package (the only WAV/JPG are the small committed
  test fixtures, made with TTS and a public-domain photo). Replay sources read files; nothing records.
- PCM lives only in the VAD's open segment and in a 2 s ring for the ASD; a segment's PCM is released as soon
  as its `MT.SEGMENT` is published.
- Voice embeddings and face tracks are per-session objects; `SensorHandle.stop()` resets them. No face
  recognition, no emotion or attention scoring: track ids are counters, voices are letters.
- The in-process providers never open a network connection except `fetch_models` (explicit, sha256-pinned)
  and faster-whisper's first model download (pre-seeded on the GB10 by the deploy USB).

## aarch64 (GB10) status

From PyPI wheel listings on 2026-10-24 (versions in `requirements/sensors.txt`):

| Package | aarch64 wheel (cp312) | GB10 note |
|---|---|---|
| onnxruntime 1.31.0 | yes (CPU) | CPU is plenty for Silero (2.3 MB) and CAM++ (29 MB); the GPU build has no prebuilt aarch64 CUDA 13 wheel (community source build only). Keep both on the 20 Arm cores. |
| opencv-python-headless 4.14.0.94 | yes | V4L2 capture, YuNet, mp4 decode: CPU. |
| faster-whisper 1.2.1 / ctranslate2 4.8.3 | pure / yes | The aarch64 ctranslate2 wheel is CPU-only: base.en int8 at ~0.5 s per segment per 4 cores is fine for a meeting. For large-v3 on the GPU use the speech server (`asr` role -> `openai-audio` provider, NeMo-Speech.cpp or whisper.cpp arm64 CUDA 13 builds, see research/r2/asr.md) or build ctranslate2 with CUDA. Nothing in this package changes. |
| tokenizers 0.23.3, av 19.0.2, kaldi-native-fbank 1.22.4, hf-xet 1.7.0, cffi 2.1.1, numpy 2.5.4 | yes | - |
| sounddevice 0.5.6 | pure python | needs `libportaudio2` (apt); otherwise the pipeline uses `arecord` from `alsa-utils` automatically (`plughw:CARD=...` resamples to 16 kHz for us). |
| torch (not used) | yes (CPU and cu130 aarch64 wheels from the PyTorch index) | only needed for Light-ASD below. |
| piper-tts 1.8.0 (fixture generation only) | yes | not a runtime dependency. |

What needs CUDA on the GB10: nothing in this package. Everything here runs on the Arm cores
(VAD + diarization + faces + ASD well under one core; faster-whisper base.en ~1-2 cores while someone talks).
Big ASR models belong on the GPU behind the speech server, selected in `config/models.toml`.

Verify on the box: `UNSPOKEN_PROFILE=gb10 python -m unspoken.models check` (vad, asr, diarization, faces, asd
all `ok`), `arecord -l` / `v4l2-ctl --list-devices` for the USB devices, then
`python -m unspoken.sensors.replay --audio tests/sensors/fixtures/meeting3.wav --fast` and a live run with
the speakerphone (`[audio] device = "Jabra"`) watching `audio.segment` lines appear within ~1 s of a sentence
ending.

## Light-ASD (next step)

Light-ASD (https://github.com/Junhua-Liao/Light-ASD, MIT) replaces MotionASD with a learned audio-visual
model (per-face speaking probability from 112x112 grey face crops at 25 fps + 13-dim MFCC at 100 Hz):
1. vendor `model/Model.py` + `model/Classifier.py` + `weight/finetuning_TalkSet.model` (~3 MB) with its LICENSE
   under `unspoken/sensors/lightasd/`; deps: `torch` (CPU wheel), `python_speech_features` (pure python);
2. `asd.LightASD(spec)`: keep a per-track deque of the last 1 s of face crops (from the `faces` bboxes) and
   the last 1 s of PCM from the pipeline ring, run the model every update (~15 ms per face on CPU), EMA the
   logits into `FaceTrack.speaking`;
3. `config/models.toml`: `[profiles.gb10.asd] provider = "python:unspoken.sensors.asd:LightASD"`.
The pipeline, events and tests do not change; `tests/sensors/test_models.py::test_motion_asd_*` become the
template for its unit tests (synthetic moving-mouth frames + a voice-like tone).

## Fixtures

`tests/sensors/fixtures/meeting3.wav` (36 s, 1.16 MB, 16 kHz mono PCM16): 12 lines by 3 Piper TTS voices
(Priya = en_US-lessac, Sam = en_US-ryan, Lee = en_GB-alba) with 0.9 s gaps; ground truth in `meeting3.json`.
Regenerate with `tests/sensors/fixtures/make_meeting_wav.py` (needs `piper-tts==1.8.0` and the three voices).
`face_256.jpg`: NASA astronaut portrait (public domain, via scikit-image's data set), used for YuNet tests.
