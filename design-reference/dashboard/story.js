// LocoCisco presenter dashboard — sample demo story.
// Everything here is SAMPLE DATA for the reference design: names, captions, timings and
// latencies are placeholders that follow BUILD_SPEC.md §12 (demo flow). The real app fills
// the same shapes from live local inference.
// Times are media seconds on the fixture's clock (0 … duration).
(function (root) {
  const people = {
    // face: [x, y, w, h] as fractions of the source frame (assets/meeting-frame.jpg) — YuNet's own
    // detections from tools/detect-faces.py (assets/faces.json). Seat E faces away: no face, just a zone.
    dana: { name: 'Dana Ortiz', seat: 'A', tone: 'green', face: [0.0632, 0.4738, 0.0515, 0.1613] },
    maya: { name: 'Maya Chen', seat: 'B', tone: 'blue', role: 'Host', face: [0.2195, 0.4147, 0.049, 0.1532] },
    sam: { name: 'Sam Patel', seat: 'C', tone: 'purple', face: [0.4117, 0.1827, 0.0575, 0.1426] },
    v4: { name: null, label: 'Voice 4', seat: 'D', tone: 'neutral', face: [0.8039, 0.3063, 0.0562, 0.1863] },
    e: { name: null, label: 'Seat E', seat: 'E', tone: 'neutral', face: null, zone: [0.870, 0.352, 0.130, 0.241] },
  };

  // Captions: speaker, start, spoken duration, text. Final text lands ~0.5 s after speech ends.
  const captions = [
    { id: 'c1', who: 'maya', t: 3.0, dur: 4.2, text: 'Thanks for joining. Two things today: the launch deck and the demo booth.' },
    { id: 'c2', who: 'dana', t: 9.0, dur: 3.6, text: "The deck is nearly done. I'm still missing the pricing slide." },
    { id: 'c3', who: 'sam', t: 15.0, dur: 2.4, text: "I'll get pricing to you tomorrow morning." },
    { id: 'c4', who: 'v4', t: 23.0, dur: 2.4, text: 'Is the demo booth confirmed for Thursday?' },
    { id: 'c5', who: 'maya', t: 29.0, dur: 5.4, text: 'It is, Thursday at two. Dana, can you send the final deck to the partner team by Friday?' },
    { id: 'c6', who: 'dana', t: 36.5, dur: 2.0, text: "Yes, I'll send it by Friday." },
    { id: 'c7', who: 'sam', t: 45.0, dur: 1.7, text: "It's getting warm in here." },
    { id: 'c8', who: 'maya', t: 51.0, dur: 2.6, text: 'Could we dim the lights for the slides?' },
    { id: 'c9', who: 'dana', t: 58.0, dur: 2.6, text: "Slide four still has last year's partner logos." },
    { id: 'c10', who: 'sam', t: 64.0, dur: 2.3, text: "I'll pull the numbers up on screen." },
    { id: 'c11', who: 'maya', t: 70.0, dur: 3.0, text: "Before we get into pricing, let's go off the record." },
    // Spoken while off record: recognised transiently as the resume command, never retained.
    { id: 'c12', who: 'maya', t: 94.0, dur: 1.9, text: "Okay, we're back on the record.", transient: true },
    { id: 'c13', who: 'sam', t: 103.0, dur: 2.6, text: 'Great. Pricing goes to Dana in the morning.' },
  ];

  // Decision records: what the cascade saw, which route it took and what policy allowed.
  const decisions = [
    { id: 'd1', t: 18.2, from: 'c3', title: 'Task: pricing slide to Dana', action: 'create_task',
      steps: [
        { k: 'rules', label: 'Privacy & reflex rules', result: 'No match', ms: 2 },
        { k: 'decider', label: 'Strands Decider 12B', model: 'Gemma 4 12B', result: 'create_task', conf: 0.79, ms: 388 },
        { k: 'escalate', label: 'Qwen3.6 35B escalation', result: 'Skipped · confident', skipped: true },
        { k: 'policy', label: 'Policy', result: 'Act', note: 'Owner speaking; due date stated' , ms: 1 },
        { k: 'act', label: 'Actuator', result: 'Task card saved locally · Discord post waits for Sam' },
      ], verdict: 'act', total: 391, egress: null },
    { id: 'd2', t: 27.8, from: 'c4', title: 'Question for the room', action: 'none',
      steps: [
        { k: 'rules', label: 'Privacy & reflex rules', result: 'No match', ms: 2 },
        { k: 'decider', label: 'Strands Decider 12B', model: 'Gemma 4 12B', result: 'answer_question?', conf: 0.46, ms: 402 },
        { k: 'escalate', label: 'Qwen3.6 35B escalation', model: 'Qwen3.6 35B-A3B', result: 'Addressed to people, not the room', ms: 1480 },
        { k: 'policy', label: 'Policy', result: 'Ignore', note: 'Unnamed speaker; no tool needed', ms: 1 },
        { k: 'act', label: 'Actuator', result: 'Nothing sent' },
      ], verdict: 'ignore', total: 1885, egress: null },
    { id: 'd3', t: 35.7, from: 'c5', title: 'Decision: demo booth Thursday 2 pm', action: 'record_decision',
      steps: [
        { k: 'rules', label: 'Privacy & reflex rules', result: 'No match', ms: 2 },
        { k: 'decider', label: 'Strands Decider 12B', model: 'Gemma 4 12B', result: 'record_decision', conf: 0.83, ms: 365 },
        { k: 'escalate', label: 'Qwen3.6 35B escalation', result: 'Skipped · confident', skipped: true },
        { k: 'policy', label: 'Policy', result: 'Act', note: 'Host confirmed it', ms: 1 },
        { k: 'act', label: 'Actuator', result: 'Decision added to meeting notes' },
      ], verdict: 'act', total: 368, egress: null },
    { id: 'd4', t: 39.4, from: 'c6', title: 'Task: final deck to partner team', action: 'create_task', highlight: true,
      steps: [
        { k: 'rules', label: 'Privacy & reflex rules', result: 'No match', ms: 3 },
        { k: 'decider', label: 'Strands Decider 12B', model: 'Gemma 4 12B', result: 'create_task', conf: 0.91, ms: 412 },
        { k: 'escalate', label: 'Qwen3.6 35B escalation', result: 'Skipped · confident', skipped: true },
        { k: 'policy', label: 'Policy', result: 'Act', note: 'Owner committed; no guest restriction; not sensitive', ms: 2 },
        { k: 'act', label: 'Actuator', result: 'Task card created · posted to #launch-team' },
      ], verdict: 'act', total: 417,
      egress: { dest: 'Discord · #launch-team', text: 'Task: Send the final deck to the partner team. Owner: Dana Ortiz. Due: Fri 30 Oct.', bytes: 87 } },
    { id: 'd5', t: 47.6, from: 'c7', title: 'Comfort: fan on', action: 'set_fan',
      steps: [
        { k: 'rules', label: 'Privacy & reflex rules', result: 'No match', ms: 2 },
        { k: 'decider', label: 'Strands Decider 12B', model: 'Gemma 4 12B', result: 'set_fan(on)', conf: 0.74, ms: 351 },
        { k: 'escalate', label: 'Qwen3.6 35B escalation', result: 'Skipped · confident', skipped: true },
        { k: 'policy', label: 'Policy', result: 'Act', note: 'Person in the room, not media audio', ms: 1 },
        { k: 'act', label: 'Actuator', result: 'Kasa plug on · confirmed in 0.3 s' },
      ], verdict: 'act', total: 354, egress: null },
    { id: 'd6', t: 54.5, from: 'c8', title: 'Lighting: slides scene', action: 'set_light',
      steps: [
        { k: 'rules', label: 'Privacy & reflex rules', result: 'No match', ms: 2 },
        { k: 'decider', label: 'Strands Decider 12B', model: 'Gemma 4 12B', result: 'set_light(slides, 30%)', conf: 0.88, ms: 344 },
        { k: 'escalate', label: 'Qwen3.6 35B escalation', result: 'Skipped · confident', skipped: true },
        { k: 'policy', label: 'Policy', result: 'Act', note: 'Host request', ms: 1 },
        { k: 'act', label: 'Actuator', result: 'WiZ lamp to 30% · confirmed in 0.2 s' },
      ], verdict: 'act', total: 347, egress: null },
    { id: 'd7', t: 61.4, from: 'c9', title: 'No action', action: 'none',
      steps: [
        { k: 'rules', label: 'Privacy & reflex rules', result: 'No match', ms: 2 },
        { k: 'decider', label: 'Strands Decider 12B', model: 'Gemma 4 12B', result: 'none', conf: 0.18, ms: 330 },
        { k: 'escalate', label: 'Qwen3.6 35B escalation', result: 'Skipped · below 0.40', skipped: true },
        { k: 'policy', label: 'Policy', result: 'Ignore', ms: 1 },
        { k: 'act', label: 'Actuator', result: 'Nothing sent' },
      ], verdict: 'ignore', total: 333, egress: null },
    { id: 'd8', t: 73.6, from: 'c11', title: 'Off record', action: 'off_record',
      steps: [
        { k: 'rules', label: 'Privacy & reflex rules', result: 'Off-record phrase · takes priority', ms: 11 },
        { k: 'decider', label: 'Strands Decider 12B', result: 'Not consulted', skipped: true },
        { k: 'escalate', label: 'Qwen3.6 35B escalation', result: 'Not consulted', skipped: true },
        { k: 'policy', label: 'Policy', result: 'Act', ms: 1 },
        { k: 'act', label: 'Actuator', result: 'Notes off · last 60 s removed · source masked and muted' },
      ], verdict: 'act', total: 12, egress: null },
    { id: 'd9', t: 96.4, from: null, quote: 'Resume command from Maya Chen (host)', title: 'Back on the record', action: 'resume_notes',
      steps: [
        { k: 'rules', label: 'Privacy & reflex rules', result: 'Resume phrase', ms: 9 },
        { k: 'decider', label: 'Strands Decider 12B', result: 'Not consulted', skipped: true },
        { k: 'escalate', label: 'Qwen3.6 35B escalation', result: 'Not consulted', skipped: true },
        { k: 'policy', label: 'Policy', result: 'Act', note: 'Host approved resume after a sensitive pause', ms: 1 },
        { k: 'act', label: 'Actuator', result: 'Notes on · source unmasked' },
      ], verdict: 'act', total: 10, egress: null },
    { id: 'd10', t: 101.6, from: null, quote: 'Browser question: "What did we decide?"', title: 'Room question', action: 'recap',
      steps: [
        { k: 'rules', label: 'Privacy & reflex rules', result: 'No match', ms: 1 },
        { k: 'decider', label: 'Room tools', result: 'list_items · recap', ms: 120 },
        { k: 'escalate', label: 'Qwen3.6 35B summary', model: 'Qwen3.6 35B-A3B', result: 'Answer from meeting notes', ms: 860 },
        { k: 'policy', label: 'Policy', result: 'Act', note: 'Shown in the browser only', ms: 1 },
        { k: 'act', label: 'Actuator', result: 'Answer shown · nothing sent' },
      ], verdict: 'act', total: 982, egress: null },
  ];

  // Everything else that changes on the timeline.
  const events = [
    { t: 1.0, type: 'checkin', who: 'maya', via: 'browser' },
    { t: 1.6, type: 'checkin', who: 'dana', via: 'browser' },
    { t: 2.2, type: 'checkin', who: 'sam', via: 'browser' },
    { t: 18.2, type: 'task', id: 'T2', title: 'Send the pricing slide to Dana', short: 'Sam sends pricing to Dana tomorrow morning.', owner: 'sam', due: 'Tomorrow morning', dueNote: 'Tue 27 Oct · time not stated', quote: "I'll get pricing to you tomorrow morning.", qt: 15.0, post: 'approval' },
    { t: 18.4, type: 'feed', id: 'f-t2', kind: 'discord', title: 'Task post for Sam’s task', status: 'approval', detail: 'Waiting for Sam to approve' },
    { t: 35.7, type: 'decision-item', id: 'D1', text: 'Demo booth confirmed for Thursday 29 Oct, 2 pm', short: 'Demo booth: Thursday 29 Oct, 2 pm.', who: 'maya', qt: 29.0 },
    { t: 39.4, type: 'task', id: 'T1', title: 'Send the final deck to the partner team', short: 'Dana sends the final deck by Fri 30 Oct.', owner: 'dana', due: 'Fri 30 Oct', quote: "Yes, I'll send it by Friday.", qt: 36.5, post: 'pending' },
    { t: 39.5, type: 'feed', id: 'f-t1', kind: 'discord', title: 'Task posted to #launch-team', status: 'pending', detail: 'Sending…' },
    { t: 39.5, type: 'egress', bytes: 87, dest: 'Discord' },
    { t: 40.9, type: 'feed-update', id: 'f-t1', status: 'delivered', detail: 'Delivered in 1.4 s' },
    { t: 40.9, type: 'task-update', id: 'T1', post: 'delivered' },
    { t: 47.6, type: 'feed', id: 'f-fan', kind: 'device', title: 'Fan on (Kasa plug)', status: 'pending', detail: 'Switching…' },
    { t: 47.9, type: 'device', name: 'fan', state: 'On', detail: 'Confirmed 0.3 s' },
    { t: 47.9, type: 'feed-update', id: 'f-fan', status: 'confirmed', detail: 'Confirmed 0.3 s' },
    { t: 54.5, type: 'feed', id: 'f-lamp', kind: 'device', title: 'Lamp to slides scene (WiZ)', status: 'pending', detail: 'Switching…' },
    { t: 54.7, type: 'device', name: 'lamp', state: 'Slides · 30%', detail: 'Confirmed 0.2 s' },
    { t: 54.7, type: 'feed-update', id: 'f-lamp', status: 'confirmed', detail: 'Confirmed 0.2 s' },
    { t: 73.6, type: 'notes', state: 'off', by: 'maya', wipe: 60 },
    { t: 96.4, type: 'notes', state: 'on', by: 'maya' },
    { t: 98.6, type: 'ask', text: 'What did we decide?' },          // typing animation starts
    { t: 100.6, type: 'ask-submit' },
    { t: 101.6, type: 'answer' },
    { t: 108.0, type: 'drawer', open: true, select: 'd4' },
    { t: 116.0, type: 'drawer', open: false },
  ];

  const beats = [
    { key: '1', t: 0, label: 'Ready' },
    { key: '2', t: 2.6, label: 'Check-in' },
    { key: '3', t: 12.5, label: 'Live captions' },
    { key: '4', t: 41.5, label: 'Task created' },
    { key: '5', t: 56.0, label: 'Room devices' },
    { key: '6', t: 80.0, label: 'Off record' },
    { key: '7', t: 102.5, label: 'Resume + question' },
    { key: '8', t: 109.0, label: 'Decision panel' },
    { key: '9', t: 120, label: 'Take saved' },
  ];

  root.STORY = {
    duration: 120,
    meeting: { title: 'Partner launch sync', room: 'Room 3B', device: 'Dell Pro Max · GB10' },
    fixture: { name: 'partner-launch-sync.mp4', length: '02:00', video: '1920×1080 · 25 fps', audio: 'AAC · 48 kHz · 2 ch' },
    people, captions, decisions, events, beats,
    wipeWindow: 60,
    devices: { lamp: { label: 'Lamp', vendor: 'WiZ', state: 'Normal · 80%' }, fan: { label: 'Fan', vendor: 'Kasa plug', state: 'Off' } },
    answerSource: 'Room tools · list_items + recap · 0.98 s',
    models: [
      { role: 'Voice activity', model: 'Silero VAD', where: 'CPU', status: 'ok', value: '' },
      { role: 'Speech', model: 'Parakeet TDT 0.6B · Q8', where: 'GPU · :8770', status: 'ok', value: '480 ms' },
      { role: 'Voices', model: 'WeSpeaker CAM++', where: 'CPU', status: 'ok', value: '' },
      { role: 'Faces', model: 'YuNet', where: 'CPU', status: 'ok', value: '' },
      { role: 'Active speaker', model: 'Light-ASD', where: 'GPU', status: 'ok', value: '25 fps' },
      { role: 'Decider', model: 'Strands Decider 12B', where: 'GPU · :8765', status: 'ok', value: '390 ms' },
      { role: 'Escalation', model: 'Qwen3.6 35B-A3B', where: 'vLLM · :8000', status: 'ok', value: 'idle' },
      { role: 'Speech out', model: 'MagpieTTS', where: 'optional', status: 'off', value: 'off' },
    ],
  };
})(typeof window !== 'undefined' ? window : globalThis);
