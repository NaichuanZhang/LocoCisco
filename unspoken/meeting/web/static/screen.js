/* LocoCisco room screen. Vanilla JS, talks only to its own server (same origin): GET /api/snapshot,
   GET /api/models, GET /api/scenarios, POST /api/*, WS /ws. */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const PRIVACY = {
    on: { text: 'NOTES ON', sub: 'captions and notes are being written', icon: 'rec' },
    paused: { text: 'PAUSED', sub: 'notes and decision display paused', icon: 'pause' },
    off_record: { text: 'OFF THE RECORD', sub: 'local captions and decision display paused', icon: 'off' },
  };
  const ICONS = {
    rec: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2.2"/><circle class="pulse" cx="12" cy="12" r="5.5" fill="currentColor"/></svg>',
    pause: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2.2"/><rect x="8" y="7" width="3" height="10" fill="currentColor"/><rect x="13" y="7" width="3" height="10" fill="currentColor"/></svg>',
    off: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2.2"/><line x1="5.5" y1="5.5" x2="18.5" y2="18.5" stroke="currentColor" stroke-width="2.6"/><rect x="8" y="8" width="8" height="8" fill="currentColor" opacity=".9"/></svg>',
  };
  const CARD_KIND = {
    task: ['✅', 'New task'], decision: ['📌', 'Decision'], answer: ['📄', 'Answer from local docs'],
    catch_up: ['↻', 'Catch-up'], approval: ['⏳', 'Waiting for approval'], posted: ['✉', 'Posted to Discord'],
    tasks: ['☑', 'Tasks'], agenda: ['☰', 'Agenda'], timer: ['⏱', 'Timer'],
  };

  const state = { snapshot: null, settings: null, models: [], connected: false, maxLeft: null };
  const seenCaptions = new Set();
  const seenCards = new Set();
  let streamEpoch = 0, revision = 0, contentRevision = 0;

  function privateMode() {
    return state.snapshot && state.snapshot.privacy && state.snapshot.privacy.state !== 'on';
  }

  function clearSensitive() {
    contentRevision += 1;
    seenCaptions.clear(); seenCards.clear(); XRay.reset();
    window.LocoPresenter?.reset();
    if (state.snapshot) {
      state.snapshot.items = [];
      if (state.snapshot.screen) state.snapshot.screen = {
        ...state.snapshot.screen, captions: [], cards: [], banner: null,
      };
    }
    for (const id of ['captions', 'cards', 'banner', 'st-log', 'st-reply']) {
      const el = $(id); if (el) el.replaceChildren();
    }
    if ($('st-text')) $('st-text').value = '';
  }

  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'text') el.textContent = v;
      else if (k === 'html') el.innerHTML = v;
      else el.setAttribute(k, v === true ? '' : v);
    }
    for (const c of children.flat()) if (c != null) el.append(c.nodeType ? c : document.createTextNode(String(c)));
    return el;
  }
  const fmtBytes = (n) => (n == null ? '–' : n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(2)} MB`);
  const reduced = () => window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------------------------------------------------------------- render
  function renderPrivacy() {
    const p = (state.snapshot && state.snapshot.privacy) || { state: 'on' };
    const key = PRIVACY[p.state] ? p.state : 'on';
    if (key !== 'on') clearSensitive();
    const el = $('privacy');
    const changed = !el.classList.contains(`state-${key}`);
    el.className = `privacy state-${key}`;
    el.querySelector('.p-icon').innerHTML = ICONS[PRIVACY[key].icon];
    el.querySelector('.p-text').textContent = PRIVACY[key].text;
    el.querySelector('.p-sub').textContent = p.reason && key !== 'on' ? `${PRIVACY[key].sub} · ${p.reason}` : PRIVACY[key].sub;
    const wiped = $('wiped');
    if (key === 'off_record' && p.wiped_count > 0) {
      const secs = (state.settings && state.settings.wipe_seconds) || 60;
      const span = secs % 60 === 0 ? `${secs / 60} min` : `${secs} s`;
      $('wiped-text').textContent = `Last ${span} removed from local transcript · ${p.wiped_count} line${p.wiped_count === 1 ? '' : 's'}. Already-delivered messages are not erased.`;
      wiped.hidden = false;
    } else if (key === 'off_record') {
      $('wiped-text').textContent = 'Local display cleared. Already-delivered messages are not erased.';
      wiped.hidden = false;
    } else {
      wiped.hidden = true;
    }
    if (changed && key !== 'on') el.classList.add('flash');
    document.documentElement.dataset.notesState = key;
    document.dispatchEvent(new CustomEvent('lococisco:privacy', { detail: key }));
  }

  function renderCaptions() {
    const scr = (state.snapshot && state.snapshot.screen) || {};
    const list = scr.captions || [];
    const n = (state.settings && state.settings.caption_lines) || 6;
    const ol = $('captions');
    ol.replaceChildren();
    if (!list.length) {
      const p = (state.snapshot && state.snapshot.privacy) || {};
      ol.append(h('li', { class: 'cap empty', text: p.state === 'off_record' ? 'Off the record. Captions paused.' : 'Waiting for someone to speak…' }));
      return;
    }
    for (const c of list.slice(-n)) {
      const key = c.utterance_id || `${c.t}|${c.text}`;
      const isNew = !seenCaptions.has(key);
      seenCaptions.add(key);
      const guest = !c.name || c.name === 'Guest';
      ol.append(h('li', { class: `cap${isNew && !reduced() ? ' new' : ''}` },
        h('b', { class: `who${guest ? ' guest' : ''}`, text: guest ? 'Guest' : c.name }),
        h('span', { class: 'text', text: c.text })));
    }
    fitFromTop(ol);
    if (seenCaptions.size > 500) seenCaptions.clear();
  }

  // the newest lines win: drop whole lines from the top until nothing is clipped (no half-visible text).
  // Measured with rects, not scrollHeight: an entry animation's transform would inflate scrollHeight.
  function fitFromTop(el) {
    let guard = 40;
    while (el.children.length > 1 && guard-- > 0) {
      const box = el.getBoundingClientRect();
      if (el.firstElementChild.getBoundingClientRect().top >= box.top - 1) break;
      el.firstElementChild.remove();
    }
  }
  function fitFromBottom(el, clip) {
    let guard = 40;
    while (el.children.length > 1 && guard-- > 0) {
      const box = (clip || el).getBoundingClientRect();
      if (el.lastElementChild.getBoundingClientRect().bottom <= box.bottom + 1) break;
      el.lastElementChild.remove();
    }
  }

  function renderAgenda() {
    const snap = state.snapshot || {};
    const ag = (snap.screen && snap.screen.agenda) || {};
    const st = snap.status || {};
    $('ag-title').textContent = st.state && st.state !== 'ended' ? (st.title || 'Meeting') : (st.title ? `${st.title} · ended` : 'No meeting yet');
    $('ag-topic').textContent = ag.topic || '';
    const left = ag.minutes_left;
    const total = ag.minutes_total || state.maxLeft || null;
    if (left != null) {
      if (st.state === 'started' && (state.maxLeft == null || left > state.maxLeft)) state.maxLeft = left;
      const seconds = Math.max(0, Math.ceil(Number(left) * 60));
      $('ag-left').textContent = seconds <= 0 ? 'time is up'
        : `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')} left`;
    } else {
      $('ag-left').textContent = st.ends_at ? `ends ${st.ends_at}` : '';
    }
    const urgent = left != null && left <= 2 && st.state === 'started';
    $('ag-left').classList.toggle('urgent', urgent);
    const bar = $('ag-bar');
    bar.classList.toggle('urgent', urgent);
    const t = total || state.maxLeft;
    const pctLeft = left != null && t ? Math.max(0, Math.min(100, (left / t) * 100)) : 0;
    $('ag-fill').style.width = `${pctLeft}%`;
    bar.setAttribute('aria-valuenow', String(Math.round(pctLeft)));
    const topics = ag.topics || [];
    const wrap = $('ag-topics');
    wrap.replaceChildren();
    const idx = topics.indexOf(ag.topic);
    topics.forEach((tp, i) => wrap.append(h('span', { class: i === idx ? 'now' : i < idx ? 'done' : '', text: tp })));
  }

  function renderBanner() {
    const b = state.snapshot && state.snapshot.screen ? state.snapshot.screen.banner : null;
    const el = $('banner');
    el.hidden = !b;
    el.textContent = String(b || '').replace(/^0 min left\./, 'Under one minute left.');
  }

  function cardNode(c) {
    const [icon, title] = c.card === 'posted' && document.body.classList.contains('replay-demo')
      ? ['✉', 'Delivery preview · not sent']
      : CARD_KIND[c.card] || ['▪', (c.title || c.card || 'card').replace(/_/g, ' ')];
    const el = h('div', { class: `card kind-${c.card || 'other'}` }, h('h3', {}, h('span', { class: 'c-icon', 'aria-hidden': 'true', text: icon }), c.title || title));
    if (c.card === 'task') {
      el.append(h('p', { class: 'c-text', text: c.text || '' }));
      el.append(h('p', { class: 'c-meta' },
        c.owner ? h('span', {}, 'owner ', h('b', { text: c.owner })) : h('span', { class: 'c-noowner', text: 'no owner yet' }),
        c.due ? h('span', {}, 'due ', h('b', { text: c.due })) : null));
      el.append(h('p', { class: 'c-undo', text: c.undo || 'undo in Discord ❌' }));
    } else if (c.card === 'answer') {
      if (c.question) el.append(h('p', { class: 'c-q', text: c.question }));
      el.append(h('p', { class: 'c-text', text: c.text || '' }));
      if (c.doc) el.append(h('p', { class: 'c-doc' }, 'from ', h('b', { text: c.doc }), ' · local docs, nothing left the box'));
    } else if (c.card === 'catch_up') {
      el.append(h('p', { class: 'c-text', text: `For ${c.for || 'the late joiner'}${c.topic ? ` · ${c.topic}` : ''}` }));
      el.append(h('ul', { class: 'c-lines' }, ...(c.lines || []).map((l) => h('li', { text: l }))));
      if (c.text) el.append(h('p', { class: 'c-meta', text: c.text }));
    } else if (c.card === 'tasks') {
      el.append(h('ul', { class: 'c-lines' }, ...(c.items || []).map((i) => h('li', { text: `${i.text} — ${i.owner || 'no owner'}${i.due ? ', ' + i.due : ''}` }))));
    } else {
      el.append(h('p', { class: 'c-text', text: c.text || '' }));
      if (c.channel) el.append(h('p', { class: 'c-meta', text: c.channel }));
    }
    return el;
  }

  function renderCards() {
    const cards = (state.snapshot && state.snapshot.screen && state.snapshot.screen.cards) || [];
    const wrap = $('cards');
    wrap.replaceChildren();
    if (!cards.length) { wrap.append(h('p', { class: 'muted', id: 'cards-empty', text: 'Tasks, decisions and answers appear here.' })); return; }
    const max = document.body.classList.contains('present') ? 3 : 4;
    for (const c of cards.slice(0, max)) {
      const key = JSON.stringify([c.card, c.item_id, c.request_id, c.text, c.for]);
      const node = cardNode(c);
      if (!seenCards.has(key) && !reduced()) node.classList.add('new');
      seenCards.add(key);
      wrap.append(node);
    }
    fitFromBottom(wrap, wrap.parentElement);
    if (seenCards.size > 300) seenCards.clear();
  }

  function renderPeople() {
    const people = (state.snapshot && state.snapshot.participants) || [];
    const el = $('people');
    el.replaceChildren();
    if (!people.length) return;
    el.append(h('span', { class: 'muted', style: 'border:none;padding:0', text: 'here:' }));
    if (document.body.classList.contains('replay-demo')) {
      for (const p of people.filter(p => p.name && !p.guest)) el.append(h('span', { text: p.name }));
      if (people.some(p => p.guest || !p.name)) {
        el.append(h('span', { class: 'guest', text: 'Unconfirmed identities',
          title: 'Unmatched voice and face tracks are not a verified count of people.' }));
      }
    } else {
      for (const p of people) el.append(h('span', { class: p.guest || !p.name ? 'guest' : '', text: p.name || 'Guest' }));
    }
  }

  function renderEgress() {
    const e = (state.snapshot && state.snapshot.egress) || {};
    const a = $('eg-audio'), v = $('eg-video'), t = $('eg-text');
    a.textContent = fmtBytes(e.audio_bytes_out || 0); a.classList.toggle('zero', !(e.audio_bytes_out > 0));
    v.textContent = fmtBytes(e.video_bytes_out || 0); v.classList.toggle('zero', !(e.video_bytes_out > 0));
    t.textContent = fmtBytes(e.total_bytes || 0);
    const dest = (e.dest || 'discord').toLowerCase();
    $('eg-dest').textContent = document.body.classList.contains('replay-demo')
      ? 'External text sent'
      : `Text to ${dest === 'openclaw' || dest === 'discord' ? 'Discord' : dest}`;
  }

  function renderModels() {
    const ul = $('models');
    ul.replaceChildren();
    const rows = state.models.length ? state.models : ((state.snapshot && state.snapshot.models) || []).map((m) => ({ ...m, local: m.local != null ? m.local : true }));
    const enabled = rows.filter((m) => m.provider && m.provider !== 'none');
    const off = rows.length - enabled.length;
    for (const m of enabled) {
      const badge = !m.ok ? ['down', 'DOWN'] : m.local ? ['', 'LOCAL'] : ['remote', 'REMOTE'];
      ul.append(h('li', { title: `${m.role}: ${m.label || m.provider} @ ${m.target || ''}${m.latency_ms != null ? ` · ${Math.round(m.latency_ms)} ms` : ''}` },
        h('span', { class: 'm-role', text: m.role }), h('span', { class: 'm-label', text: m.label || m.provider }),
        h('span', { class: `m-badge ${badge[0]}`, text: badge[1] })));
    }
    if (off > 0) ul.append(h('li', { class: 'off' }, h('span', { class: 'm-role', text: `${off} role${off === 1 ? '' : 's'} off` })));
  }

  function renderAll() {
    renderPrivacy(); renderCaptions(); renderAgenda(); renderBanner(); renderCards(); renderPeople(); renderEgress(); renderModels();
    window.LocoPresenter?.snapshot(state.snapshot);
  }

  // ---------------------------------------------------------------- stream
  function setConn(v, text) {
    state.connected = v;
    const el = $('conn');
    el.textContent = text;
    el.className = `b-conn ${v ? 'live' : 'down'}`;
  }

  function applyEvent(ev) {
    const epoch = ev.stream_epoch || 0;
    if (epoch < streamEpoch) return;
    revision += 1;
    if (epoch > streamEpoch) {
      streamEpoch = epoch; clearSensitive();
      if (!state.snapshot) state.snapshot = {};
      if (ev.stream_privacy) state.snapshot.privacy = { state: ev.stream_privacy };
      renderPrivacy(); renderCaptions(); renderCards(); renderBanner();
    }
    const d = ev.data || {};
    switch (ev.topic) {
      case 'ui.reset':
        clearSensitive(); state.maxLeft = null;
        if (!state.snapshot) state.snapshot = {};
        state.snapshot.privacy = { state: ev.stream_privacy || d.privacy || 'on' };
        renderPrivacy(); renderCaptions(); renderCards(); renderBanner(); break;
      case 'ui.snapshot':
        state.snapshot = d.snapshot; if (d.settings) state.settings = d.settings;
        if (state.settings) XRay.setThreshold(state.settings.act_threshold);
        renderAll(); break;
      case 'ui.settings':
        state.settings = d.settings; XRay.setThreshold(state.settings.act_threshold); renderCaptions(); break;
      case 'ui.ping': break;
      case 'screen.state':
        if (!state.snapshot) state.snapshot = {};
        state.snapshot.screen = d;
        if (d.privacy) state.snapshot.privacy = { ...(state.snapshot.privacy || {}), state: d.privacy };
        renderPrivacy(); renderCaptions(); renderAgenda(); renderBanner(); renderCards(); break;
      case 'meeting.privacy':
        if (!state.snapshot) state.snapshot = {};
        state.snapshot.privacy = d; renderPrivacy(); renderCaptions(); renderCards(); renderBanner(); break;
      case 'egress.meter':
        if (!state.snapshot) state.snapshot = {};
        state.snapshot.egress = d; renderEgress(); break;
      case 'meeting.status':
        if (!state.snapshot) state.snapshot = {};
        state.snapshot.status = d; if (d.state === 'started') state.maxLeft = null; renderAgenda(); break;
      case 'meeting.utterance': case 'meeting.proposal': case 'meeting.verdict':
        if (!privateMode()) XRay.handle(ev); break;
      case 'scenario.status': stageStatus(d); break;
      case 'trace.log': stageLog(ev); break;
      default: break;
    }
    window.LocoPresenter?.event(ev);
  }

  let ws = null, retry = 500;
  function connect() {
    const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
    try { ws = new WebSocket(url); } catch (e) { setConn(false, '○ offline'); setTimeout(connect, retry); return; }
    ws.onopen = () => {
      // Epochs belong to one server stream; a restarted server begins at zero.
      revision += 1; streamEpoch = 0; clearSensitive(); XRay.resetStream();
      setConn(true, '● live'); retry = 500; loadModels();
    };
    ws.onmessage = (m) => { try { applyEvent(JSON.parse(m.data)); } catch (e) { /* ignore a bad frame */ } };
    ws.onclose = () => { setConn(false, '○ reconnecting'); setTimeout(connect, retry); retry = Math.min(8000, retry * 2); };
    ws.onerror = () => { try { ws.close(); } catch (e) { /* noop */ } };
  }

  async function loadModels() {
    try {
      const r = await fetch('api/models', { cache: 'no-store' });
      if (r.ok) { const j = await r.json(); state.models = j.models || []; renderModels(); }
    } catch (e) { /* the footer keeps the snapshot's list */ }
  }

  async function loadSnapshot() {
    const started = revision;
    try {
      const r = await fetch('api/snapshot', { cache: 'no-store' });
      if (r.ok) {
        const j = await r.json();
        if (started !== revision) return; // an older HTTP response must not undo a stream privacy/reset barrier
        const epoch = Number(r.headers.get('X-LocoCisco-Stream-Epoch') || 0);
        if (epoch < streamEpoch) return;
        if (epoch > streamEpoch) { streamEpoch = epoch; clearSensitive(); }
        state.snapshot = j.snapshot; state.settings = j.settings; XRay.setThreshold(j.settings.act_threshold); renderAll();
      }
    } catch (e) { /* the stream will bring one */ }
  }

  // ---------------------------------------------------------------- stage controls (rehearsal)
  const api = (path, body) => fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
  function stageStatus(d) {
    const el = $('st-state');
    el.textContent = d.state === 'step' ? `step ${d.step_index + 1}` : d.state === 'finished' ? `${d.passed} passed, ${d.failed} failed` : d.state;
    el.className = `chip ${d.state === 'started' || d.state === 'step' || d.state === 'expect_pass' ? 'running' : d.state === 'expect_fail' || d.state === 'error' ? 'failed' : 'idle'}`;
    if (d.state === 'expect_fail') stageLog({ ts: 0, data: { agent: 'scenario', level: 'warn', text: `FAILED: ${d.text}` } });
  }
  function stageLog(ev) {
    if (privateMode()) return;
    const ol = $('st-log');
    const d = ev.data || {};
    const li = h('li', { class: d.level === 'warn' ? 'warn' : '' }, h('span', { class: 't', text: `${(ev.ts || 0).toFixed(1)}s ${d.agent || ''}` }), d.text || '');
    ol.prepend(li);
    while (ol.children.length > 60) ol.lastChild.remove();
  }
  async function loadScenarios() {
    try {
      const r = await fetch('api/scenarios', { cache: 'no-store' });
      if (!r.ok) return;
      const sel = $('st-scenario');
      sel.replaceChildren();
      for (const s of await r.json()) sel.append(h('option', { value: s.id, text: `${s.title} — ${s.blurb}` }));
    } catch (e) { /* stage stays empty */ }
  }
  function wireStage() {
    $('st-run').addEventListener('click', () => api(`api/scenario/${encodeURIComponent($('st-scenario').value)}`));
    $('st-stop').addEventListener('click', () => api('api/stop'));
    $('st-reset').addEventListener('click', async () => {
      revision += 1; clearSensitive(); state.maxLeft = null;
      await api('api/reset'); loadSnapshot();
    });
    $('st-close').addEventListener('click', () => toggle('stage'));
    $('st-send').addEventListener('click', async () => {
      const started = contentRevision;
      const cmd = $('st-cmd').value, user = $('st-user').value || 'rehearsal', text = $('st-text').value.trim();
      const body = { command: cmd, user, args: {}, text: '' };
      if (cmd === 'here') body.args = { name: text || user };
      else if (cmd === 'ask') body.text = text || 'What did we decide?';
      try {
        const r = await api('api/chat', body);
        const j = await r.json();
        if (started === contentRevision && !privateMode()) $('st-reply').textContent = r.ok ? (j.text || 'ok') : `error ${r.status}`;
      } catch (e) { if (started === contentRevision && !privateMode()) $('st-reply').textContent = 'failed'; }
    });
    $('st-text').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('st-send').click(); });
  }

  // ---------------------------------------------------------------- panels, keys, theme
  function toggle(id, force) {
    const el = $(id);
    const show = force != null ? force : el.hidden;
    el.hidden = !show;
    if (show) { const b = el.querySelector('button, select, input'); if (b && id !== 'stage') b.focus(); }
  }
  function closeAll() { toggle('how', false); toggle('help', false); toggle('stage', false); XRay.close(); }
  function setPresent(v) { document.body.classList.toggle('present', v); renderCaptions(); renderCards(); }
  function setTheme(t) {
    if (t === 'dark' || t === 'light') document.documentElement.setAttribute('data-theme', t);
    else document.documentElement.removeAttribute('data-theme');
  }
  function onKey(e) {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA')) {
      if (e.key === 'Escape') { e.target.blur(); closeAll(); }
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    switch (e.key) {
      case 'x': case 'X': XRay.toggle(); break;
      case 'p': case 'P': setPresent(!document.body.classList.contains('present')); break;
      case 'c': case 'C': toggle('how'); break;
      case 's': case 'S': toggle('stage'); break;
      case '?': toggle('help'); break;
      case 't': case 'T': {
        const cur = document.documentElement.getAttribute('data-theme');
        const dark = cur ? cur === 'dark' : (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
        setTheme(dark ? 'light' : 'dark'); break;
      }
      case 'Escape': closeAll(); break;
      default: return;
    }
    e.preventDefault();
  }

  function clock() {
    const d = new Date();
    $('clock').textContent = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
  }

  document.addEventListener('DOMContentLoaded', () => {
    const q = new URLSearchParams(location.search);
    if (q.get('present') === '1') setPresent(true);
    if (q.get('theme')) setTheme(q.get('theme'));
    if (q.get('xray') === '1') XRay.open();
    document.addEventListener('keydown', onKey);
    document.addEventListener('lococisco:models', loadModels);
    document.addEventListener('lococisco:reset', () => {
      revision += 1; clearSensitive(); state.maxLeft = null; loadSnapshot();
    });
    $('how-close').addEventListener('click', () => toggle('how', false));
    $('help-close').addEventListener('click', () => toggle('help', false));
    for (const id of ['how', 'help']) $(id).addEventListener('click', (e) => { if (e.target.id === id) toggle(id, false); });
    wireStage();
    clock(); setInterval(clock, 10000);
    window.addEventListener('resize', () => { renderCaptions(); renderCards(); });
    loadSnapshot(); loadScenarios(); connect();
    setInterval(loadModels, 30000);
  });
})();
