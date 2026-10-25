// LocoCisco — How it works page (explainer.html): extra SAMPLE signals layered on story.js.
// story.js stays the single source for people, captions, decisions and events. This file only
// adds what the explainer draws on top: real YuNet detections, seat zones, ASR partials and
// corrections, voice-map clusters, the Decider's structured outputs, latency targets and the tour.
// Everything except the YuNet detections is sample data for the reference design.
(function (root) {
  root.EXPLAIN = {
    // YuNet (face_detection_yunet_2023mar) run once on assets/meeting-frame.jpg by tools/detect-faces.py.
    // Fractions of the frame; landmarks are right eye, left eye, nose tip, right and left mouth corner.
    faces: {
      dana: { track: 1, score: 0.695, landmarks: [[0.1007, 0.5487], [0.1007, 0.5466], [0.1093, 0.5706], [0.1025, 0.5995], [0.1004, 0.5967]], note: 'Profile view · lower score' },
      maya: { track: 2, score: 0.905, landmarks: [[0.2516, 0.4767], [0.2607, 0.474], [0.2648, 0.5034], [0.2501, 0.5313], [0.2589, 0.5302]] },
      sam: { track: 3, score: 0.927, landmarks: [[0.4486, 0.2411], [0.4645, 0.2427], [0.4677, 0.2645], [0.4481, 0.2931], [0.4609, 0.2942]] },
      v4: { track: 4, score: 0.900, landmarks: [[0.8147, 0.3679], [0.8261, 0.3709], [0.8077, 0.4049], [0.8158, 0.441], [0.8258, 0.4433]] },
    },
    // Seat zones [x, y, w, h]. Seat E is the person facing away: a zone with no face.
    seats: {
      A: [0.010, 0.40, 0.165, 0.42],
      B: [0.185, 0.36, 0.135, 0.38],
      C: [0.370, 0.10, 0.150, 0.40],
      D: [0.760, 0.24, 0.108, 0.36],
      E: [0.870, 0.352, 0.130, 0.241],
    },

    // Speech recognition: partial hypotheses every chunk while someone speaks, final text
    // (with casing and punctuation) asrLag seconds after the speech segment ends.
    asr: {
      chunk: 0.64,
      lag: 0.5,
      ms: 480,
      // Word index → what the partials heard first. 'next' is fixed by the next partial,
      // 'final' only when the full segment is decoded.
      corrections: {
        c1: { 7: { heard: 'lunch', fix: 'next' } },
        c2: { 10: { heard: 'side', fix: 'final' } },
        c4: { 3: { heard: 'boot', fix: 'final' } },
        c5: { 4: { heard: 'to', fix: 'next' }, 15: { heard: 'seem', fix: 'final' } },
        c7: { 2: { heard: 'worm', fix: 'next' } },
        c8: { 2: { heard: 'him', fix: 'next' } },
        c9: { 7: { heard: 'local', fix: 'final' } },
        c11: { 9: { heard: 'recorder', fix: 'final' } },
        c13: { 4: { heard: 'donna', fix: 'final' } },
      },
    },

    // Voice map: each ~1.2 s window of speech becomes a voice embedding (CAM++), drawn here as
    // a 2D projection. Centres are fractions of the map; voices are numbered by first speech.
    voices: {
      window: 1.2,
      spread: 0.05,
      centers: { maya: [0.28, 0.26], dana: [0.74, 0.26], sam: [0.70, 0.78], v4: [0.24, 0.78] },
    },

    // What the Decider returns, shown token by token (≤ 38 characters a line).
    decider: {
      d1: ['{ "tool": "create_task",', '  "title": "Pricing slide to Dana",', '  "owner": "sam",', '  "due": "tomorrow morning",', '  "confidence": 0.79 }'],
      d2: ['{ "tool": "answer_question",', '  "question": "booth confirmed?",', '  "addressed_to": "unclear",', '  "confidence": 0.46 }'],
      d3: ['{ "tool": "record_decision",', '  "text": "Demo booth Thu 2 pm",', '  "confirmed_by": "maya",', '  "confidence": 0.83 }'],
      d4: ['{ "tool": "create_task",', '  "title": "Final deck to partners",', '  "owner": "dana",', '  "due": "2026-10-30",', '  "confidence": 0.91 }'],
      d5: ['{ "tool": "set_fan",', '  "state": "on",', '  "reason": "room feels warm",', '  "confidence": 0.74 }'],
      d6: ['{ "tool": "set_light",', '  "scene": "slides",', '  "level": 30,', '  "confidence": 0.88 }'],
      d7: ['{ "tool": "none",', '  "reason": "remark, no request",', '  "confidence": 0.18 }'],
      d10: ['{ "tools": ["list_items",', '            "recap"],', '  "scope": "this meeting" }'],
    },
    replay: 3.2, // s: each decision is replayed in slow motion after it lands

    // Initial acceptance targets from BUILD_SPEC §11 — targets, not measured results.
    targets: { caption: 2.0, card: 3.0, privacy: 0.5 },
    // Planning estimates from BUILD_SPEC §5 (resource budget) — not measured.
    memory: {
      total: 128,
      parts: [
        { label: 'Qwen3.6 35B · vLLM', gb: 48, unit: 'GiB', tone: 'blue' },
        { label: 'Decider 12B', gb: 27, unit: 'GB', tone: 'purple' },
        { label: 'Speech', gb: 3, unit: 'GB', tone: 'green' },
      ],
    },

    // Tour mode (A): which view tells the story best at each moment.
    tour: [[0, 'vision'], [14, 'transcription'], [35, 'decisions'], [64, 'transcription'], [98, 'decisions'], [108, 'pipeline']],
  };
})(typeof window !== 'undefined' ? window : globalThis);
