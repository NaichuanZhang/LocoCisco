/* Recording dashboard. All changing values come from the room stream; no demo story or
   synthetic inference timings. The screen/replay clients own networking and feed this view. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const esc = value => String(value == null ? '' : value).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const number = value => typeof value === 'number' && Number.isFinite(value) ? value : null;
  const clamp = value => Math.max(0, Math.min(1, number(value) || 0));
  const pct = value => number(value) == null ? '—' : `${Math.round(clamp(value) * 100)}%`;
  const ms = value => number(value) == null ? '—' : value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
  const time = value => `${String(Math.floor(Math.max(0, value || 0) / 60)).padStart(2, '0')}:${String(Math.floor(Math.max(0, value || 0) % 60)).padStart(2, '0')}`;
  const actionName = value => ({ none: 'No action', create_task: 'Capture task', log_decision: 'Log decision',
    answer_question: 'Answer question', set_light: 'Adjust light', set_fan: 'Adjust fan', pause_notes: 'Pause notes',
    resume_notes: 'Resume notes', share_docs: 'Share documents', post_recap: 'Meeting recap', park_topic: 'Park topic', show: 'Show card' }[value] || String(value || 'Waiting').replace(/_/g, ' '));
  const verdictName = value => ({ act: 'Allowed', ignore: 'No action', deny: 'Blocked', ask_owner: 'Approval needed' }[value] || 'Policy pending');
  const ROLES = {
    vad: { name: 'Silero VAD', task: 'Find speech', hardware: 'CPU', icon: 'wave' },
    asr: { name: 'Parakeet', task: 'Speech → text', hardware: 'GPU', icon: 'text' },
    diarization: { name: 'CAM++', task: 'Distinguish voices', hardware: 'CPU', icon: 'voice' },
    faces: { name: 'YuNet', task: 'Track faces', hardware: 'CPU', icon: 'face' },
    asd: { name: 'Light-ASD', task: 'Find who is speaking', hardware: 'GPU', icon: 'focus' },
    decider: { name: 'Decider 12B', task: 'Classify intent', hardware: 'GPU', icon: 'route' },
    llm: { name: 'Qwen 35B', task: 'Reason when uncertain', hardware: 'GPU', icon: 'spark' },
  };
  const ICONS = {
    wave: '<path d="M3 10v4m4-8v12m5-15v18m5-15v12m4-8v4"/>',
    text: '<path d="M5 4h14M12 4v16m-4 0h8"/>',
    voice: '<rect x="9" y="3" width="6" height="12" rx="3"/><path d="M5 11v1a7 7 0 0014 0v-1M12 19v3"/>',
    face: '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/><circle cx="12" cy="10" r="3"/><path d="M7 18c1-4 9-4 10 0"/>',
    focus: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 1v3m0 16v3M1 12h3m16 0h3"/>',
    route: '<path d="M4 12h6m0 0 5-6h5m-10 6 5 6h5"/><circle cx="3" cy="12" r="1"/><path d="m18 3 3 3-3 3m0 6 3 3-3 3"/>',
    spark: '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3z"/>',
    shield: '<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3z"/><path d="m8 12 3 3 5-6"/>',
  };
  const icon = name => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ICONS.spark}</svg>`;
  let privacyState = 'on', streamEpoch = 0, generation = null, replayEpoch = -1;
  let latestSnapshot = {}, replayState = {}, telemetry = null, health = [], settings = {};
  let records = [], current = null, feed = [], lastAudio = null, lastVision = null;
  let observed = { utterances: 0, decider: 0, llm: 0, verdicts: 0 };
  const byUtterance = new Map(), byProposal = new Map(), recent = new Map(), seen = new Set();
  let dirty = true;

  function put(id, html) {
    const element = $(id);
    if (element && element.innerHTML !== html) element.innerHTML = html;
  }

  function clearData() {
    records = []; current = null; feed = []; telemetry = null; lastAudio = null; lastVision = null;
    observed = { utterances: 0, decider: 0, llm: 0, verdicts: 0 };
    byUtterance.clear(); byProposal.clear(); recent.clear(); seen.clear();
    latestSnapshot = { models: health, privacy: { state: privacyState } };
    dirty = true;
  }

  function privacy(value) {
    const next = typeof value === 'string' ? value : (value || {}).state;
    if (!['on', 'paused', 'off_record'].includes(next)) return;
    if (next !== privacyState) {
      privacyState = next;
      clearData();
    }
    dirty = true;
    if (next !== 'on') render();
  }

  function reset() {
    streamEpoch = 0;
    generation = null;
    replayEpoch = -1;
    clearData();
    render();
  }

  function recordFor(uid, event) {
    if (uid && byUtterance.has(uid)) return byUtterance.get(uid);
    const record = { id: uid || event.id, ts: event.ts || 0, utterance: null, proposals: [], verdict: null };
    records.push(record);
    if (uid) byUtterance.set(uid, record);
    while (records.length > 16) {
      const old = records.shift();
      byUtterance.delete(old.id);
      for (const proposal of old.proposals) byProposal.delete(proposal.id);
    }
    return record;
  }

  function pulse(role, detail) {
    recent.set(role, { at: Date.now(), detail });
    dirty = true;
  }

  function acceptTelemetry(data) {
    if (privacyState !== 'on' || !data) return;
    telemetry = data;
    const audio = data.audio;
    if (audio && audio.end !== lastAudio) {
      lastAudio = audio.end;
      pulse('vad', 'Speech segment detected');
      pulse('asr', `Transcribed · ${ms(audio.asr_ms)}`);
      pulse('diarization', `Voice labelled · ${ms(audio.diar_ms)}`);
    }
    const vision = data.vision;
    if (vision && vision.t !== lastVision) {
      lastVision = vision.t;
      pulse('faces', `${vision.face_count || 0} face tracks`);
      pulse('asd', 'Speaking scores updated');
    }
    dirty = true;
  }

  function replay(data) {
    if (!data) return false;
    if (generation != null && number(data.generation) != null && data.generation < generation) return false;
    if (generation === data.generation && number(data.privacy_epoch) != null && data.privacy_epoch < replayEpoch) return false;
    const changed = generation != null && data.generation !== generation;
    const privateChanged = replayEpoch >= 0 && data.privacy_epoch !== replayEpoch;
    generation = data.generation == null ? generation : data.generation;
    replayEpoch = data.privacy_epoch == null ? replayEpoch : data.privacy_epoch;
    if (changed || privateChanged) clearData();
    replayState = data;
    privacy(data.privacy || privacyState);
    dirty = true;
    return true;
  }

  function snapshot(data) {
    if (!data) return;
    if (data.demo && !replay(data.demo)) return;
    privacy((data.privacy || {}).state || privacyState);
    if (Array.isArray(data.models)) health = data.models;
    settings = data.settings || settings;
    latestSnapshot = privacyState === 'on' ? data : { models: health, privacy: { state: privacyState } };
    if (privacyState === 'on') acceptTelemetry(data.telemetry);
    dirty = true;
    render();
  }

  function addFeed(title, detail, ts, kind) {
    feed.unshift({ title, detail, ts, kind });
    feed = feed.slice(0, 5);
  }

  function event(ev) {
    if (!ev || !ev.topic) return;
    const epoch = Number(ev.stream_epoch || 0);
    if (epoch < streamEpoch) return;
    if (epoch > streamEpoch) { streamEpoch = epoch; clearData(); }
    if (ev.stream_privacy) privacy(ev.stream_privacy);
    const data = ev.data || {};
    if (ev.topic === 'ui.reset') { clearData(); privacy(data.privacy || ev.stream_privacy || 'on'); render(); return; }
    if (ev.topic === 'meeting.privacy') { privacy(data.state); render(); return; }
    if (ev.topic === 'ui.snapshot') { if (data.settings) settings = data.settings; snapshot(data.snapshot); return; }
    if (ev.topic === 'ui.settings') { settings = data.settings || settings; dirty = true; return; }
    if (privacyState !== 'on') return;
    if (ev.id && seen.has(ev.id)) return;
    if (ev.id) { seen.add(ev.id); if (seen.size > 1500) seen.delete(seen.values().next().value); }
    if (ev.topic === 'sensor.telemetry') acceptTelemetry(data);
    else if (ev.topic === 'meeting.utterance') {
      current = recordFor(data.utterance_id, ev);
      current.utterance = data;
      observed.utterances += 1;
      pulse('fusion', data.speaker && data.speaker.name ? 'Voice and seat associated' : 'Anonymous speaker retained');
    } else if (ev.topic === 'meeting.proposal') {
      const record = recordFor(data.utterance_id, ev);
      record.proposals.push({ id: ev.id, ...data });
      byProposal.set(ev.id, record);
      current = record;
      if (data.route === 'local-llm') {
        if (!record.countedLLM) { observed.llm += 1; record.countedLLM = true; }
        pulse('llm', `Reasoned · ${ms(data.latency_ms)}`);
      } else if (data.route === 'local-decider') {
        if (!record.countedDecider) { observed.decider += 1; record.countedDecider = true; }
        pulse('decider', `Classified · ${ms(data.latency_ms)}`);
      }
    } else if (ev.topic === 'meeting.verdict') {
      const record = byProposal.get(data.proposal_id);
      if (record) { record.verdict = data; if (!current || record.ts >= current.ts) current = record; }
      observed.verdicts += 1;
      pulse('policy', verdictName(data.verdict));
      addFeed(verdictName(data.verdict), actionName(data.action), ev.ts, data.verdict);
    } else if (ev.topic === 'meeting.item') {
      const items = latestSnapshot.items || [];
      latestSnapshot.items = [...items.filter(item => item.item_id !== data.item_id), data];
    } else if (ev.topic === 'meeting.command' && data.target && !['screen', 'notifier'].includes(data.target)) {
      addFeed('Room preview', `${data.target} · ${String(data.command || '').replace(/_/g, ' ')}`, ev.ts, 'preview');
    }
    dirty = true;
    render();
  }

  function node(role, detailOverride) {
    const spec = ROLES[role];
    const model = health.find(item => item.role === role);
    const signal = recent.get(role);
    const justUpdated = signal && Date.now() - signal.at < 1500;
    const last = current && current.proposals[current.proposals.length - 1];
    const awaitingLLM = role === 'llm' && last && last.mode === 'shadow' && !current.verdict;
    let status = model ? model.ok ? 'Ready' : 'Unavailable' : 'Checking';
    let classes = 'pr-node';
    let detail = detailOverride || (signal ? signal.detail : spec.task);
    if (model && !model.ok) classes += ' is-down';
    if (privacyState !== 'on') { status = 'Protected'; classes += ' is-private'; detail = 'Private observations hidden'; }
    else if (awaitingLLM) { status = 'Reasoning'; classes += ' is-working'; detail = 'Decider requested a second opinion'; }
    else if (justUpdated) { status = 'Updated'; classes += ' is-recent'; }
    return `<div class="${classes}" data-role="${esc(role)}"><div class="pr-node-top"><span class="pr-node-icon">${icon(spec.icon)}</span><span class="pr-node-hardware">${esc(spec.hardware)}</span><span class="pr-node-state">${esc(status)}</span></div><b class="pr-node-name">${esc(spec.name)}</b><span class="pr-node-detail">${esc(detail)}</span></div>`;
  }

  function renderPipeline() {
    const last = current && current.proposals[current.proposals.length - 1];
    const fused = recent.get('fusion');
    const fusionText = privacyState !== 'on' ? 'Privacy barrier · protected content discarded' :
      fused ? fused.detail : 'Voice + face + check-in → attributed utterance';
    const policy = current && current.verdict;
    const policyRecent = recent.get('policy');
    const policyClass = privacyState !== 'on' ? ' is-private' : policyRecent && Date.now() - policyRecent.at < 1500 ? ' is-recent' : '';
    const qwenDetail = last && last.route === 'local-decider' && last.mode === 'act' ? 'Skipped · Decider handled this' : undefined;
    put('model-pipeline', `<div class="pr-input-lanes"><div class="pr-lane"><span class="pr-lane-label">AUDIO</span>${node('vad')}<span class="pr-flow-link" aria-hidden="true">→</span>${node('asr')}<span class="pr-flow-link" aria-hidden="true">+</span>${node('diarization')}</div><div class="pr-lane"><span class="pr-lane-label">VISION</span>${node('faces')}<span class="pr-flow-link" aria-hidden="true">→</span>${node('asd')}</div></div><div class="pr-fusion${fused && Date.now() - fused.at < 1500 ? ' is-recent' : ''}"><span class="pr-merge-icon" aria-hidden="true">↳</span><b>Speaker fusion</b><span>${esc(fusionText)}</span><span aria-hidden="true">↓</span></div><div class="pr-reasoning">${node('decider')}<span class="pr-flow-link pr-conditional" aria-hidden="true">→<small>if unsure</small></span>${node('llm', qwenDetail)}<span class="pr-flow-link" aria-hidden="true">→</span><div class="pr-node pr-policy${policyClass}" data-role="policy"><div class="pr-node-top"><span class="pr-node-icon">${icon('shield')}</span><span class="pr-node-hardware">CODE</span><span class="pr-node-state">${privacyState !== 'on' ? 'Protected' : policy ? esc(verdictName(policy.verdict)) : 'Ready'}</span></div><b class="pr-node-name">Policy gate</b><span class="pr-node-detail">${privacyState !== 'on' ? 'Output blocked by privacy' : policy ? esc(policy.rule) : 'Privacy · confidence · approvals'}</span></div></div><p class="pr-flow-note">Confident Decider results go directly to policy. The language model is called only when needed.</p>`);
  }

  function renderDecision() {
    if (privacyState !== 'on') {
      put('decision-detail', `<div class="pr-empty pr-private-empty">${icon('shield')}<b>${privacyState === 'paused' ? 'Notes paused' : 'Off the record'}</b><span>Protected text and pending model results are discarded.</span></div>`);
      return;
    }
    if (!current) {
      put('decision-detail', `<div class="pr-empty"><span class="pr-empty-symbol">${icon('route')}</span><b>Every decision has a trace.</b><span>The next utterance will show the Decider’s probabilities, model route and policy outcome here.</span></div>`);
      return;
    }
    const proposals = current.proposals;
    const first = proposals.find(p => p.route === 'local-decider');
    const last = proposals[proposals.length - 1];
    const llm = proposals.find(p => p.route === 'local-llm');
    const verdict = current.verdict;
    const quote = current.utterance && current.utterance.text;
    const speaker = current.utterance && current.utterance.speaker;
    const route = llm ? 'Decider → Qwen → policy' : first ? first.mode === 'shadow' && !verdict ? 'Decider → Qwen' : 'Decider → policy' : last ? 'Rule → policy' : 'Utterance received';
    const elapsed = (first ? number(first.latency_ms) || 0 : 0) + (llm ? number(llm.latency_ms) || 0 : 0);
    const thresholds = { act: number(last && last.threshold) ?? number(settings.act_threshold) ?? .60,
      ask: number(settings.ask_threshold) ?? .40 };
    let bars = '';
    if (first && first.probabilities) {
      const probabilities = Object.entries(first.probabilities).filter(([, value]) => number(value) != null).sort((a, b) => b[1] - a[1]).slice(0, 3);
      bars = `<div class="pr-probabilities">${probabilities.map(([label, value]) => `<div class="pr-probability"><span>${esc(actionName(label))}</span><div class="pr-prob-track"><i style="width:${clamp(value) * 100}%"></i><em style="left:${clamp(thresholds.act) * 100}%" title="Act threshold"></em></div><b>${pct(value)}</b></div>`).join('')}<p>Decider probability mass · act ≥ ${pct(thresholds.act)} · consider Qwen ≥ ${pct(thresholds.ask)}</p></div>`;
    }
    const qwenText = llm ? `${ms(llm.latency_ms)} · ${llm.tool_calls && llm.tool_calls.length || 0} tool calls` : first && first.mode === 'shadow' && !verdict ? 'Reasoning…' : first ? 'Not called' : 'Waiting';
    const outcome = verdict ? verdict.rule : last && last.mode === 'shadow' ? 'The local language model is reviewing this utterance.' : last ? 'Waiting for the policy verdict.' : 'Fusion delivered the utterance to the meeting agents.';
    put('decision-detail', `<div class="pr-decision-top"><span class="pr-route ${llm ? 'pr-route-llm' : ''}">${esc(route)}</span><span class="pr-decision-time">${elapsed ? ms(elapsed) + ' model time' : time(current.ts)}</span></div>${quote ? `<blockquote class="pr-quote"><p>“${esc(quote)}”</p><cite>${esc(speaker && speaker.name && !speaker.guest ? speaker.name : 'Unconfirmed speaker')}</cite></blockquote>` : ''}<div class="pr-decision-steps"><div><span>01 · DECIDER</span><b>${first ? esc(actionName(first.action)) : 'Waiting'}</b><small>${first ? ms(first.latency_ms) : 'Typed intent questions'}</small></div><div class="${llm ? 'is-used' : ''}"><span>02 · QWEN</span><b>${llm ? esc(actionName(llm.action)) : first && first.mode === 'shadow' && !verdict ? 'Second opinion' : 'Skipped'}</b><small>${esc(qwenText)}</small></div><div><span>03 · POLICY</span><b>${esc(verdictName(verdict && verdict.verdict))}</b><small>${verdict ? esc(verdict.risk || 'code') + ' risk' : 'Deterministic rules'}</small></div></div>${bars}<p class="pr-outcome${verdict && verdict.verdict === 'act' ? ' is-allowed' : ''}">${icon('shield')}<span>${esc(outcome)}</span></p>${llm && llm.confidence_source === 'adapter_constant' ? '<p class="pr-score-note">Qwen’s adapter score is not a calibrated model probability.</p>' : ''}`);
  }

  function renderPeople() {
    if (privacyState !== 'on') { put('capture-people', '<p class="pr-quiet">Identity associations cleared.</p>'); return; }
    const people = latestSnapshot.participants || [];
    const unique = people.filter((person, index) => person.voice || person.name ||
      !people.some(other => other.voice && other.seat === person.seat)).slice(0, 4);
    if (!unique.length) { put('capture-people', '<p class="pr-quiet">Voice and face tracks appear as the meeting is processed.</p>'); return; }
    put('capture-people', unique.map(person => {
      const named = person.name && !person.guest;
      const label = named ? person.name : person.voice ? String(person.voice).replace(/^voice[-_]/i, 'Voice ') : 'Unconfirmed';
      const initials = named ? person.name.split(/\s+/).slice(0, 2).map(part => part[0]).join('') : '·';
      return `<div class="pr-person${named ? ' is-named' : ''}"><span class="pr-avatar">${esc(initials)}</span><span><b>${esc(label)}</b><small>${esc([person.seat ? `${person.seat} seat` : 'Seat unbound', named ? 'Checked in' : 'Identity unconfirmed'].join(' · '))}</small></span></div>`;
    }).join(''));
  }

  function renderNotes() {
    if (privacyState !== 'on') { put('capture-notes', '<div class="pr-empty"><b>Notes protected</b><span>Recent notes and derived outputs have been cleared.</span></div>'); return; }
    const items = (latestSnapshot.items || []).filter(item => item.status !== 'undone').slice(-2).reverse();
    if (!items.length) {
      put('capture-notes', '<div class="pr-empty"><span class="pr-empty-mark">✓</span><b>Listening for commitments.</b><span>Tasks and decisions appear only when the models identify them and policy allows them.</span></div>');
      return;
    }
    put('capture-notes', items.map(item => `<article class="pr-note pr-note-${esc(item.kind)}"><span class="pr-note-kind">${esc(item.kind === 'task' ? 'ACTION ITEM' : String(item.kind || 'note').replace(/_/g, ' ').toUpperCase())}</span><b>${esc(item.text)}</b><p>${esc([item.owner || (item.kind === 'task' ? 'Owner unassigned' : ''), item.due, item.status === 'pending_approval' ? 'Awaiting approval' : 'Captured locally'].filter(Boolean).join(' · '))}</p></article>`).join(''));
  }

  function renderFeed() {
    if (privacyState !== 'on') { put('capture-feed', '<p class="pr-quiet">Protected output cleared.</p>'); return; }
    put('capture-feed', feed.length ? feed.slice(0, 3).map(item => `<div class="pr-feed-row pr-feed-${esc(item.kind)}"><span class="pr-feed-dot" aria-hidden="true"></span><div><b>${esc(item.title)}</b><span>${esc(item.detail)}</span></div><time>${time(item.ts)}</time></div>`).join('') : '<p class="pr-quiet">Policy verdicts and local output previews will appear here.</p>');
  }

  function renderVision() {
    const vision = telemetry && telemetry.vision;
    if (privacyState !== 'on' || !vision) { put('capture-vision', ''); return; }
    put('capture-vision', (vision.faces || []).filter(face => Array.isArray(face.bbox) && face.bbox.length === 4 && face.bbox.every(value => number(value) != null)).slice(0, 8).map(face => {
      const [x, y, width, height] = face.bbox.map(clamp);
      const speaking = clamp(face.speaking);
      return `<div class="pr-face${speaking >= .5 ? ' is-speaking' : ''}" style="left:${x * 100}%;top:${y * 100}%;width:${Math.min(width, 1 - x) * 100}%;height:${Math.min(height, 1 - y) * 100}%"><span>${esc(face.track_id)} · ${pct(speaking)}</span></div>`;
    }).join(''));
  }

  function renderStats() {
    const counts = privacyState === 'on' && telemetry ? telemetry.counts || {} : {};
    const audio = privacyState === 'on' && telemetry ? telemetry.audio : null;
    const stats = [
      [number(counts.segments) ?? 0, 'Speech segments'],
      [number(counts.frames) ?? 0, 'Vision observations'],
      [observed.decider, 'Decider results'],
      [observed.llm, 'Qwen routes'],
    ];
    put('capture-stats', stats.map(([value, label]) => `<div class="pr-stat"><b>${typeof value === 'number' ? value.toLocaleString('en-US') : esc(value)}</b><span>${esc(label)}</span></div>`).join(''));
    const ready = Object.keys(ROLES).filter(role => health.some(model => model.role === role && model.ok)).length;
    put('model-summary', `<span class="pr-summary-status${ready === 7 ? ' is-ready' : ''}"><i aria-hidden="true"></i>${ready} / 7 models ready</span><span class="pr-summary-meta">${privacyState !== 'on' ? 'Privacy barrier active' : audio ? `Latest ASR ${ms(audio.asr_ms)} · voice ${ms(audio.diar_ms)}` : 'Local inference · CPU + GPU'}</span>`);
  }

  function render() {
    if (!dirty && ![...recent.values()].some(value => Date.now() - value.at < 2000)) return;
    renderPipeline(); renderDecision(); renderPeople(); renderNotes(); renderFeed(); renderVision(); renderStats();
    dirty = false;
  }

  window.LocoPresenter = { event, snapshot, replay, reset, privacy };
  document.addEventListener('lococisco:privacy', ev => privacy(ev.detail));
  // Expire the brief, event-driven result highlight; never simulate model work while idle.
  setInterval(render, 250);
  render();
})();
