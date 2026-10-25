# LocoCisco browser demo

The main demo uses the official `Presentation/Demo.mp4` checked into the public repository's
`presentation` branch. Its audio and video frames pass through the real local speech, speaker,
vision and decision models at playback time. The 30.08-second clip is staged on this Dell box;
no USB microphone or camera is needed.

## Start the prepared stack

From this checkout:

```bash
export UNSPOKEN_LOCAL_API_KEY_FILE="$HOME/.nemoclaw/dual-station-vllm-api-key"
deploy/demo/up.sh
```

The variable contains a **file path**, not the credential. Do not paste the credential into shell
arguments, chat, source, or Git. The browser container mounts that file read-only.

Open **http://127.0.0.1:8780/** on the Dell. The launcher reuses the existing Qwen service and loaded
GPU image; it leaves the pre-existing Qwen and Nano containers alone. It starts/reuses the prepared
Parakeet speech service, Decider container and LocoCisco browser container. It fails when preparation
is missing instead of downloading or installing anything at startup.

All seven required roles must pass their local probes before a take can start. TTS, a second Decider,
and external integrations are not prerequisites. Press **O** to open recording setup and use
**Check models** after restoring a failed local service. Background checks run every fifteen seconds.
Inference failures stop the replay visibly; there is no cloud or scripted-output fallback.

## A take

1. The official presentation video loads by default. Press **O** for setup if another local MP4/WebM
   is needed. Uploads stay on this box, have a 300 MiB limit and must be at most ten minutes.
2. Playback starts automatically when the local models and browser video are ready. If the browser
   blocks audible autoplay, playback starts muted; click the page or press **Space** to enable sound.
   Captions come from local ASR; names require a spoken or manual check-in. Unidentified voices
   remain **Guest**. The room view does not guarantee every participant is visible.
3. The recording canvas shows model contributions, the decision route, probabilities and the policy
   outcome. Playback and operator controls are hidden from this view.
4. Recording setup contains **Off the record**, **Resume notes**, check-in and **Ask locally**.
   Close setup with **O** before recording; spoken privacy phrases also work.
5. **Shift+R** clears the take and returns to the beginning. **Space** pauses or continues the current take.

Opening the page preserves a manually paused or private session. To prepare an idle page explicitly,
use **http://127.0.0.1:8780/?autoplay=0**. The Firefox recorder selects this option automatically so
audio capture is attached before playback starts. Native **R** capture keeps its own reset/countdown.

The source footage is the official presentation asset. Any labels, translations or other graphics
already present inside that video are part of the input footage; the surrounding dashboard's
captions and model traces are computed locally. There are no injected transcript, task or decision
results; a model may abstain or make an error.

Devices and Discord are explicitly labeled **local rehearsal previews**. This demo sends no real
device commands or external notifications. The footer measures application notifications, not all
machine traffic. Actual Discord/device deployment remains a separate integration step.

## Record and export

Use a 1920×1080 browser viewport. Press **R** and select this browser tab/window. The app resets
the take, counts down, and records the actual page with audio directly from the source video.
It never requests microphone or webcam access. Select the demo tab/window rather than an entire monitor.
Press **R** again to stop and save a local WebM; recording also ends after the source finishes.
Review both picture and sound before using a take.
Cancellation, denial and unsupported codecs are shown as errors; they are not reported as a saved take.

An offline fallback uses an isolated Firefox profile to capture the real rendered UI and the video's
audio. It does not need display-capture permission or additional packages/browsers. Run it before the
take; playback, capture and export are automatic, with no terminal interaction during the take:

```bash
.venv/bin/python -m unspoken.meeting.record_demo \
  --output ../lococisco-runtime/recordings/LocoCisco-demo.mp4
```

For a short sound check, add `--seconds 18`. Add `--privacy-demo` to briefly switch off-record and
resume during the take. The fallback captures screenshots at a requested 8 FPS; actual throughput is
reported in the companion JSON. Native browser recording is preferable for smoother video.
The MP4 contains H.264 video and AAC audio. Existing destination files are never overwritten.

The 2026-10-25 final review is saved at
`../lococisco-runtime/recordings/lococisco-official-redesign-final-review.mp4`,
with metrics in the adjacent JSON and application hashes in `.source.json`.
It completed the official 30.08-second input with all seven roles ready, five
published speech segments and no sensor errors. Its 35.23-second capture achieved
3.15 FPS; use native browser capture for a smoother final take. Full-file decoding,
source audio before/after privacy, silence within the off-record interval, and
the corrected light-surface colors were verified. Privacy in this take was
triggered by `--privacy-demo`, not by interpreting the quiet aside in the footage.

The populated layout was checked at 1920×1080 and 1440×900 with no visible controls,
page overflow or console errors. At 1280×720 the page scrolls vertically; the
recording target remains 1920×1080. The browser/branding suite passes 125 tests;
the isolated deployment dry-run suite passes 36 with two skipped checks.
Current English ASR mistranscribes the opening Mandarin. The source video's
translations and name labels do not establish runtime translation or identity.

Off-record hides the source image, mutes captured audio, wipes the recent transcript window, and
discards protected text and delayed inference output. It cannot erase original input files,
earlier frames in a recording, previously exported recordings, or messages already delivered outside the application. This workflow
intentionally records the demo; ordinary microphone/camera operation is a separate mode.

## Local preparation and provenance

Runtime files live in `../lococisco-runtime`, outside Git. The prepared Python environment uses the
offline kit's ARM64 wheels. Torch/CUDA are reused from the already-loaded NVIDIA image; the missing
standalone GPU-wheel dependencies are not downloaded.

Model artifacts used here are Parakeet TDT 0.6B v3, Silero VAD, WeSpeaker CAM++, YuNet, Light-ASD,
the Strands Decider 12B adapter plus Gemma 4 base, and the existing Qwen3.6 35B-A3B NVFP4 service.
The Decider's isolated user-site environment uses the bundled wheels; the host and existing model
containers are not upgraded. `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` are enabled.

The official input is staged byte-for-byte from Git:

- Repository: https://github.com/NaichuanZhang/LocoCisco
- Branch/path: `presentation:Presentation/Demo.mp4`
- Source commit: `8f6ad1028295ec456c6ebc946df84b7e1f4be408`
- Runtime path: `../lococisco-runtime/media/lococisco-official-demo.mp4`
- SHA-256: `a999d8f0ebd55e5121d889457639f6d14bf020496675837c450bb0efd8aa3e43`
- Format: 1280×720 H.264 at 24 FPS, stereo AAC at 32 kHz, duration 30.08 seconds.

No transcoding is required. Its adjacent `.source.json` records provenance. To stage the checked-out
asset on another prepared machine without overwriting an existing file:

```bash
cp -n ../lococisco-public/Presentation/Demo.mp4 ../lococisco-runtime/media/lococisco-official-demo.mp4
```

`LOCOCISCO_VIDEO` can explicitly select another local source, with `LOCOCISCO_ATTRIBUTION` for its
attribution. An already-created browser container retains its original input bind mount; recreating
that application container applies a changed default. The model services need no media-specific change.

The older AMI fixtures remain available for reproducing saved backend validation; they are no longer
the default demo input. The highlights fixture is **AMI Meeting Corpus, IS1009a**, licensed **CC BY 4.0**:

- Source: https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/IS1009a/
- License: https://groups.inf.ed.ac.uk/ami/corpus/license.shtml
- Inputs: `video/IS1009a.L.avi` and `audio/IS1009a.Array1-01.wav`
- Edits: excerpts 50–128 s and 650–752 s, room-view/microphone mux, H.264/AAC transcode and audio-level normalization.

To rebuild from those already-staged files:

```bash
.venv/bin/python deploy/demo/prepare_ami.py \
  --raw-dir ../lococisco-runtime/media \
  --output ../lococisco-runtime/media/ami-meeting-highlights.mp4 \
  --ranges 50:128 650:752
```

The AMI builder refuses to overwrite a file and saves attribution beside the result. Runtime copies,
recordings, caches and credentials remain outside the implementation checkout. The public
`presentation` branch contains the official video and presentation materials. The public `main`
also contains a reviewed application source snapshot, without the original USB bundle's Git history.
