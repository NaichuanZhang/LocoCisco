# LocoCisco dashboard — design reference

Two static, recordable 1920×1080 pages for demo videos. They use the Webex
daytime theme and play the same scripted 2-minute story on one clock. There is
no backend: every name, caption, decision and latency is **sample data**, and a
small "Sample data" chip on screen says so. Set `showSampleChip` in
`explainer.js` to hide it on the How it works page.

Open either file directly in a browser; everything loads from this folder, with
no CDN.

| Page | What it shows |
|---|---|
| `index.html` | The presenter dashboard: meeting video, captions, people, notes, decisions, Discord feed, room devices, off-record mode and model health. |
| `explainer.html` | How it works: **Vision** (face boxes, landmarks, speaking scores), **Transcription** (partial → final text, voice map, who spoke when), **Decisions** (rules → Decider 12B → Qwen 35B → policy, replayed slowly) and **Pipeline** (local-only flow, latency, memory budget). |

## Keys

| Key | Dashboard | How it works |
|---|---|---|
| Space | Play / pause | Play / pause |
| Shift+R | Restart | Restart |
| R | Record demo (countdown) | 3-2-1 countdown, then play |
| 1–9 | Jump to a story beat | Same beats |
| D / N / O / Q / S | Decision panel, notes, off record, ask, setup | — |
| H | Open How it works at this moment | — |
| V / T / D / P | — | Switch view |
| A | — | Tour: the view follows the story |
| ? | Shortcuts | Shortcuts |

`explainer.html?view=decisions&t=39` opens a given view at a given moment.

## Files

- `tokens.css` — all colours, fonts, sizes and motion values, shared by both pages.
  Colour values come from Cisco's open-source Momentum design tokens (light theme).
- `story.js` — the sample story: people, captions, decisions, events and beats.
- `explainer-story.js` — extra sample signals for the explainer: ASR partials and
  corrections, voice clusters, the Decider's structured outputs, targets and the tour.
- `app.*` — the dashboard; `explainer.*` — the How it works page.
- `assets/faces.json` — real YuNet (`face_detection_yunet_2023mar`) detections on
  the meeting frame, made by `tools/detect-faces.py`. Re-run it with OpenCV ≥ 4.8:
  `python tools/detect-faces.py <yunet.onnx> assets/meeting-frame.jpg assets/faces.json`.
  The model file isn't included.

## Credits

- Meeting frame: StockSnap photo G5HASN4EI9 (CC0), cropped to 16:9.
- Fonts: Figtree and JetBrains Mono, SIL Open Font License 1.1. The licences are in `fonts/`.
