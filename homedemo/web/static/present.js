/* Presenter mode for recording the demo video. Plain browser JS, loaded after app.js and xray.js.
   - ?present=1 (or key p, or the Presenter button): intent banner on top, 3D house left (~62%) with the
     scoreboard under it, Decider X-ray right (~38%), debug panes hidden, large type, captions over the
     house, decision cards dwell longer (main.js listens to feed 'present').
   - key x: hide/show the X-ray card (the house takes the full width in presenter mode).
   - key c: the "reuse it in your agent" code card (DeciderToolRouter wiring + the typed question JSON).
   Keys are ignored while typing in a field. */
(() => {
  'use strict';
  const PROJECT_NAME = 'LocoCisco';  // the product name, one constant
  const PROJECT_NAME_ZH = '';        // no localized product name configured
  const INTENT = "No wake word. A 2B Strands Decider on the hub picks the tool, so what you say stays home. Claude on Amazon Bedrock only when it's unsure.";
  const feed = window.homeFeed;
  const $ = (sel, root = document) => root.querySelector(sel);
  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'text') el.textContent = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? '' : v);
    }
    for (const c of children.flat()) if (c != null) el.append(c.nodeType ? c : document.createTextNode(String(c)));
    return el;
  }

  const state = { present: false, xray: true, code: false };

  // ------------------------------------------------------------------ presenter mode
  function setPresent(on, pushUrl = true) {
    state.present = !!on;
    document.body.classList.toggle('present', state.present);
    const banner = $('#present-banner');
    if (banner) banner.hidden = !state.present;
    $('#present-toggle')?.setAttribute('aria-pressed', String(state.present));
    const cap = $('#present-caption');
    if (cap) cap.hidden = !state.present || !cap.textContent;
    if (pushUrl) {
      const url = new URL(location.href);
      if (state.present) url.searchParams.set('present', '1'); else url.searchParams.delete('present');
      try { history.replaceState(null, '', url); } catch { /* file: or sandboxed */ }
    }
    feed?.emit('present', state.present);
  }
  function setXray(on) {
    state.xray = !!on;
    const card = $('#xray');
    if (card) card.hidden = !state.xray;
    document.body.classList.toggle('xray-off', !state.xray);
    feed?.emit('present', state.present);   // the house resizes
  }
  function setCode(on) {
    state.code = !!on;
    const card = $('#code-card');
    if (card) card.hidden = !state.code;
  }

  const typing = (e) => /^(input|select|textarea)$/i.test(e.target?.tagName || '') || e.target?.isContentEditable;
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || typing(e)) return;
    const k = e.key.toLowerCase();
    if (k === 'p') { setPresent(!state.present); e.preventDefault(); }
    else if (k === 'x') { setXray(!state.xray); e.preventDefault(); }
    else if (k === 'c') { setCode(!state.code); e.preventDefault(); }
    else if (k === 'escape' && state.code) { setCode(false); e.preventDefault(); }
  });
  $('#present-toggle')?.addEventListener('click', () => setPresent(!state.present));
  $('#cc-close')?.addEventListener('click', () => setCode(false));
  $('#code-card')?.addEventListener('click', (e) => { if (e.target === e.currentTarget) setCode(false); });

  // ------------------------------------------------------------------ banner: brand, clock, controls, captions
  for (const el of document.querySelectorAll('[data-project-name]')) el.textContent = PROJECT_NAME;
  for (const el of document.querySelectorAll('[data-project-name-zh]')) {
    el.textContent = PROJECT_NAME_ZH;
    el.hidden = !PROJECT_NAME_ZH;
  }
  for (const el of document.querySelectorAll('[data-intent]')) el.textContent = INTENT;
  function setClock(time, weekday) {
    for (const el of document.querySelectorAll('[data-clock-time]')) if (time) el.textContent = time;
    for (const el of document.querySelectorAll('[data-clock-day]')) if (weekday) el.textContent = weekday;
  }
  feed?.on('snapshot', (snap) => { if (snap) setClock(snap.time, snap.weekday); });
  feed?.on('event', (ev) => { if (ev?.topic === 'sim.clock') setClock(ev.data?.time, ev.data?.weekday); });

  const sel = $('#pb-scenario');
  feed?.on('state', (st) => {
    if (!sel || !st?.scenarios) return;
    const sorted = [...st.scenarios].sort((a, b) => (b.tags || []).includes('hero') - (a.tags || []).includes('hero'));
    sel.replaceChildren(...sorted.map((sc) => h('option', { value: sc.id, text: sc.title })));
  });
  // the banner controls drive the (hidden) scenario card, so both stay in sync
  sel?.addEventListener('change', () => { document.querySelector(`#scenario-list .scenario[data-id="${sel.value}"]`)?.click(); });
  $('#pb-run')?.addEventListener('click', () => { if (sel?.value) document.querySelector(`#scenario-list .scenario[data-id="${sel.value}"]`)?.click(); $('#run')?.click(); });
  $('#pb-stop')?.addEventListener('click', () => $('#stop')?.click());
  $('#pb-reset')?.addEventListener('click', () => $('#reset')?.click());
  const mirror = (fromSel, apply) => {
    const from = $(fromSel);
    if (!from) return;
    apply(from);
    new MutationObserver(() => apply(from)).observe(from, { childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['class'] });
  };
  mirror('#scn-state', (from) => { const to = $('#pb-state'); if (to) { to.textContent = from.textContent; to.className = from.className; } });
  // captions over the house: the scenario's narration notes (and the final tally for a moment)
  let captionTimer = null;
  function caption(text, forMs = 0) {
    const cap = $('#present-caption');
    if (!cap) return;
    clearTimeout(captionTimer);
    cap.textContent = text || '';
    cap.hidden = !state.present || !text;
    if (text && forMs) captionTimer = setTimeout(() => caption(''), forMs);
  }
  feed?.on('event', (ev) => {
    if (ev?.topic !== 'scenario.status') return;
    const d = ev.data || {};
    if (d.state === 'step' && d.text) caption(d.text);
    else if (d.state === 'started') caption('');
    else if (d.state === 'finished') caption(`${d.text || 'finished'}`, 6000);
    else if (d.state === 'stopped' || d.state === 'error') caption(d.state === 'error' ? `Scenario error: ${d.text}` : '', 4000);
  });
  feed?.on('reset', () => caption(''));

  // ------------------------------------------------------------------ code card content
  const WIRING = [
    'from strands import Agent',
    'from strands.models import BedrockModel',
    'from homedemo.patterns import DeciderToolRouter',
    '',
    '# decider: anything with `async ask(state, questions)`',
    'router = DeciderToolRouter(decider, threshold=0.8)',
    'model = BedrockModel(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0")',
    'agent = Agent(model=model, interventions=[router],',
    '              tools=[set_lights, set_thermostat, play_music])',
    'agent("Kitchen lights off.")  # 2B sure: tool runs, Claude never called',
    'print(router.last)  # RouteDecision(route="local", tool="set_lights", ...)',
  ].join('\n');
  const QUESTION = {
    tool: {
      type: 'choice',
      instructions: "A user said this to an assistant that has the tools listed as options. Which ONE tool should the assistant call to handle it? Pick 'No tool' if the message needs none of them or is not a request.",
      criteria: {
        set_lights: 'Switch or dim the lights in a room.',
        set_thermostat: 'Nudge the thermostat up or down by 1.5 C.',
        play_music: 'Play music or the news on the speaker in a room.',
        'No tool': 'The message needs none of these tools, or is not a request.',
      },
    },
    'set_lights.room': {
      type: 'choice',
      instructions: "If the assistant calls set_lights, what should its 'room' argument be?",
      criteria: { living: 'room = living', kitchen: 'room = kitchen', bedroom: 'room = bedroom' },
    },
  };
  const NOTE = 'before_invocation: the router sends the agent\'s tool names (first docstring line each) plus "No tool" to the Strands Decider 2B as ONE typed request, with one more choice question per enum argument. Sure → Deny: the model is never called (0 tokens, nothing leaves the device). Unsure → Proceed: Claude on Bedrock decides with the same tools. homedemo/patterns/decider_tool_router.py (~200 lines, no other homedemo imports).';

  /* Tiny Python highlighter: comments, strings, keywords, calls, numbers. Text only, no markup parsing. */
  function highlightPython(src) {
    const out = document.createDocumentFragment();
    const rx = /(#[^\n]*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b(from|import|print|async|def|return|await|True|False|None)\b|\b([A-Za-z_]\w*)(?=\()|\b(\d+(?:\.\d+)?)\b/g;
    let last = 0, m;
    while ((m = rx.exec(src))) {
      if (m.index > last) out.append(document.createTextNode(src.slice(last, m.index)));
      const cls = m[1] ? 'py-c' : m[2] ? 'py-s' : m[3] ? 'py-k' : m[4] ? 'py-f' : 'py-n';
      out.append(h('span', { class: cls, text: m[0] }));
      last = m.index + m[0].length;
    }
    if (last < src.length) out.append(document.createTextNode(src.slice(last)));
    return out;
  }
  function buildCodeCard() {
    const py = $('#cc-python'), js = $('#cc-json');
    if (py) py.replaceChildren(highlightPython(WIRING));
    if (js) {
      if (window.homeXray?.renderJson) js.replaceChildren(window.homeXray.renderJson(QUESTION));
      else js.textContent = JSON.stringify(QUESTION, null, 2);
    }
    for (const el of document.querySelectorAll('[data-cc-note]')) el.textContent = NOTE;
  }
  buildCodeCard();

  // ------------------------------------------------------------------ start
  const params = new URLSearchParams(location.search);
  setPresent(params.get('present') === '1' || params.get('present') === 'true', false);
  window.homePresent = { setPresent, setXray, setCode, state, PROJECT_NAME };
})();
