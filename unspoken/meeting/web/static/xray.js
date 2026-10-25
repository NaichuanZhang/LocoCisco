/* Decider X-ray (key x): the typed request the room sent, the answer against the act threshold, where the
   decision was made (LOCAL DECIDER / LOCAL LLM / RULE), latency, and the local LLM's tokens and tool calls
   when it was asked. Vanilla JS, no network. Exposes window.XRay. */
(function () {
  'use strict';
  const MAX_HISTORY = 8;
  const ROUTE_LABEL = { 'local-decider': 'LOCAL DECIDER', 'local-llm': 'LOCAL LLM', 'rule': 'RULE', 'user': 'USER' };
  const VERDICT_LABEL = { act: 'ACT', deny: 'DENIED', ask_owner: 'ASK OWNER', ignore: 'IGNORED' };

  const items = [];            // newest first: {id, utterance, proposals:[], verdict, startWall, endWall}
  const byUtterance = new Map();
  const byProposal = new Map();
  let current = null;
  let threshold = 0.8;
  let open = false;
  let timer = null;
  let privateMode = false;
  let epoch = 0;
  let contentAfter = 0;

  const $ = (id) => document.getElementById(id);
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
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const pct = (p) => `${Math.round((p || 0) * 100)}%`;
  const ms = (x) => (x == null ? '–' : x >= 1000 ? `${(x / 1000).toFixed(2)} s` : `${Math.round(x)} ms`);

  // ---------------------------------------------------------------- model of decisions
  function itemForUtterance(uid, ev) {
    if (uid && byUtterance.has(uid)) return byUtterance.get(uid);
    const it = { id: uid || ev.id, utterance: null, proposals: [], verdict: null, startWall: ev.wall * 1000, endWall: null, ts: ev.ts };
    items.unshift(it);
    if (uid) byUtterance.set(uid, it);
    while (items.length > MAX_HISTORY) {
      const gone = items.pop();
      if (gone.utterance) byUtterance.delete(gone.utterance.utterance_id);
      for (const p of gone.proposals) byProposal.delete(p.id);
    }
    return it;
  }

  function onUtterance(ev) {
    const d = ev.data;
    if (d.source === 'media') return;            // the X-ray shows decisions; media voices are ignored by rule
    const it = itemForUtterance(d.utterance_id, ev);
    it.utterance = d;
    it.startWall = ev.wall * 1000;
    it.endWall = null;
    current = it;
    render();
  }

  function onProposal(ev) {
    const d = ev.data;
    const it = itemForUtterance(d.utterance_id, ev);
    if (!it.utterance && !d.utterance_id && !it.proposals.length) {
      it.synthetic = d;                                        // camera scene, meeting end: no sentence behind it
      it.startWall = ev.wall * 1000 - (d.latency_ms || 0);     // so the stopwatch shows the model latency
    }
    it.proposals.push({ id: ev.id, data: d, wall: ev.wall * 1000 });
    byProposal.set(ev.id, it);
    if (d.mode === 'act' || d.route === 'rule' || d.route === 'user') it.endWall = ev.wall * 1000;
    current = it;
    render();
  }

  function onVerdict(ev) {
    const it = byProposal.get(ev.data.proposal_id);
    if (!it) return;
    it.verdict = ev.data;
    it.endWall = it.endWall || ev.wall * 1000;
    if (current === it) render();
    renderHistory();
  }

  function handle(ev) {
    const nextEpoch = ev.stream_epoch || 0;
    if (nextEpoch < epoch) return;
    if (nextEpoch > epoch) { epoch = nextEpoch; reset(); }
    if (ev.topic === 'meeting.privacy') { setPrivacy(ev.data.state); return; }
    if (ev.topic === 'ui.reset') { reset(); setPrivacy(ev.data.privacy || 'on'); return; }
    if (privateMode || (ev.wall && ev.wall * 1000 <= contentAfter)) return;
    if (ev.topic === 'meeting.utterance') onUtterance(ev);
    else if (ev.topic === 'meeting.proposal') onProposal(ev);
    else if (ev.topic === 'meeting.verdict') onVerdict(ev);
  }

  // ---------------------------------------------------------------- rendering helpers
  function qBadge(type) {
    return h('span', { class: `q-badge ${type}`, text: type === 'noul' ? 'yes / no' : type });
  }

  // Pretty JSON with syntax colours; question objects get a type badge after their key.
  function renderJson(value, indent, questions) {
    const pad = '  '.repeat(indent);
    if (value === null) return `<span class="j-bool">null</span>`;
    if (typeof value === 'boolean') return `<span class="j-bool">${value}</span>`;
    if (typeof value === 'number') return `<span class="j-num">${value}</span>`;
    if (typeof value === 'string') return `<span class="j-str">"${esc(value)}"</span>`;
    if (Array.isArray(value)) {
      if (!value.length) return '<span class="j-p">[]</span>';
      return `<span class="j-p">[</span>\n${value.map((v) => `${pad}  ${renderJson(v, indent + 1)}`).join(',\n')}\n${pad}<span class="j-p">]</span>`;
    }
    const keys = Object.keys(value);
    if (!keys.length) return '<span class="j-p">{}</span>';
    const rows = keys.map((k) => {
      const v = value[k];
      const badge = questions && v && typeof v === 'object' && v.type ? qBadge(v.type).outerHTML : '';
      return `${pad}  <span class="j-key">"${esc(k)}"</span><span class="j-p">:</span> ${renderJson(v, indent + 1, k === 'questions')}${badge}`;
    });
    return `<span class="j-p">{</span>\n${rows.join(',\n')}\n${pad}<span class="j-p">}</span>`;
  }

  function routeBadge(route) {
    return h('span', { class: `xr-badge r-${route || 'rule'}`, text: ROUTE_LABEL[route] || String(route || '').toUpperCase() });
  }

  function bars(d) {
    const probs = d.probabilities || {};
    const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 7);
    if (!entries.length) return null;
    const pickLabel = (() => {
      const ans = d.answers && Object.values(d.answers).find((a) => a && a.type === 'choice');
      if (ans && ans.choice in probs) return ans.choice;
      let best = entries[0][0];
      for (const [k, v] of entries) if (v > probs[best]) best = k;
      return best;
    })();
    const thr = d.threshold != null ? d.threshold : threshold;
    const wrap = h('div', { class: 'bars' });
    for (const [label, p] of entries) {
      const track = h('div', { class: 'b-track' },
        h('div', { class: 'b-fill', style: `width:${Math.max(1, p * 100)}%` }),
        h('div', { class: 'b-thr', style: `left:${thr * 100}%`, title: `act threshold ${pct(thr)}` }));
      wrap.append(h('div', { class: `bar${label === pickLabel ? ' pick' : ''}` },
        h('span', { class: 'b-label', text: label }), track, h('span', { class: 'b-pct', text: pct(p) })));
    }
    const conf = d.confidence != null ? d.confidence : (probs[pickLabel] || 0);
    const sure = conf >= thr;
    const note = h('p', { class: 'thr-note' },
      'act threshold ', h('b', { text: pct(thr) }), ' · ',
      sure ? `${pct(conf)}: sure, acted locally` : `${pct(conf)}: unsure → asked the local LLM`);
    return h('div', {}, wrap, note);
  }

  function sec(title, ...body) {
    return h('div', { class: 'xr-sec' }, h('h4', { text: title }), ...body);
  }

  function render() {
    const body = $('xr-body');
    const route = $('xr-route');
    if (!body || !route) return;
    const it = current;
    if (!it) return;
    body.replaceChildren();
    const last = it.proposals[it.proposals.length - 1];
    const d = last ? last.data : null;
    route.className = 'xr-badge ' + (d ? `r-${d.route || 'rule'}` : 'thinking');
    route.textContent = d ? (ROUTE_LABEL[d.route] || d.route) : 'deciding…';

    // what was heard (or seen)
    if (it.utterance) {
      const u = it.utterance;
      const who = (u.speaker && u.speaker.name) || 'Guest';
      body.append(sec('Heard',
        h('p', { class: 'xr-heard' }, h('b', { text: who }), ' ', `“${u.text}”`),
        u.context ? h('p', { class: 'xr-ctx', text: u.context }) : null));
    } else if (d) {
      body.append(sec('Seen', h('p', { class: 'xr-heard', text: d.reason || d.action })));
    }

    for (const p of it.proposals) {
      const pd = p.data;
      const model = h('p', { class: 'xr-model' }, routeBadge(pd.route), h('b', { text: pd.model || '' }),
        pd.kind ? h('span', { text: `kind: ${pd.kind}` }) : null,
        h('span', { text: `latency ${ms(pd.latency_ms)}` }),
        pd.mode === 'shadow' ? h('span', { class: 'xr-badge thinking', text: 'shadow: escalated' }) : null);
      const parts = [model];
      if (pd.request && pd.request.state) {
        const pre = h('pre', { class: 'code json', html: renderJson(pd.request, 0, false) });
        if (it.proposals.length > 1) {          // escalated: fold the request so the LLM section stays in view
          parts.push(h('details', { class: 'xr-fold' }, h('summary', { text: pd.route === 'local-llm' ? 'State the LLM was given (same typed state)' : 'Typed request sent to the decider' }), pre));
        } else {
          parts.push(h('h4', { text: 'Typed request sent to the decider' }));
          parts.push(pre);
        }
      }
      const b = pd.route === 'local-decider' ? bars(pd) : null;   // rules and the LLM have no option probabilities
      if (b) { parts.push(h('h4', { text: 'Answer: option probabilities' })); parts.push(b); }
      if (pd.route === 'local-llm') {
        const tok = pd.tokens || { input: 0, output: 0 };
        const llm = h('div', { class: 'llm' },
          h('p', { class: 'tok' }, `tokens in ${tok.input} · out ${tok.output} · never left the box`),
          ...(pd.tool_calls || []).map((t) => h('p', { class: 'tool' }, `→ ${t.name}(${JSON.stringify(t.input)})`)),
          pd.reply ? h('p', { class: 'reply', text: `“${pd.reply}”` }) : null);
        parts.push(llm);
      }
      parts.push(h('p', { class: 'muted small', text: pd.reason || '' }));
      body.append(sec(pd.route === 'local-llm' ? 'Local LLM (asked because the decider was unsure)'
        : pd.route === 'rule' ? 'Rule (plain code, no model)' : 'Decider', ...parts));
    }
    if (it.verdict) {
      const v = it.verdict;
      body.append(sec('Policy',
        h('p', { class: 'xr-verdict' }, h('span', { class: `xr-badge v-${v.verdict}`, text: VERDICT_LABEL[v.verdict] || v.verdict }),
          h('b', { text: `${v.action}${v.args && Object.keys(v.args).length ? ' ' + JSON.stringify(v.args) : ''}` }),
          h('span', { class: 'rule', text: `risk ${v.risk} · ${v.rule}` }))));
    }
    tick();
    renderHistory();
  }

  function renderHistory() {
    const ol = $('xr-history');
    if (!ol) return;
    ol.replaceChildren();
    for (const it of items) {
      const last = it.proposals[it.proposals.length - 1];
      if (!last) continue;
      const d = last.data;
      const li = h('li', { class: it === current ? 'current' : '', tabindex: 0, role: 'button' },
        h('span', { class: 'h-t', text: fmtTs(it.ts) }),
        routeBadge(d.route),
        h('span', { class: 'h-act', text: `${d.action}${it.verdict ? ' · ' + (VERDICT_LABEL[it.verdict.verdict] || it.verdict.verdict).toLowerCase() : ''}` }),
        h('span', { class: 'h-conf', text: d.route === 'rule' ? '' : pct(d.confidence) }));
      li.addEventListener('click', () => { current = it; render(); });
      li.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); current = it; render(); } });
      ol.append(li);
    }
  }

  function fmtTs(ts) {
    const s = Math.max(0, Math.round(ts || 0));
    return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  }

  // stopwatch: runs from the utterance until the acting proposal arrives
  function tick() {
    const el = $('xr-clock');
    if (!el || !current) return;
    const end = current.endWall || Date.now();
    el.textContent = ms(end - current.startWall);
    if (!current.endWall && open) {
      clearTimeout(timer);
      timer = setTimeout(tick, 100);
    }
  }

  // ---------------------------------------------------------------- open / close
  function setOpen(v) {
    open = v;
    const panel = $('xray');
    if (!panel) return;
    panel.hidden = !v;
    document.body.classList.toggle('xray-open', v);
    if (v) { render(); tick(); }
    window.dispatchEvent(new Event('resize'));
  }

  function reset() {
    items.length = 0;
    byUtterance.clear();
    byProposal.clear();
    current = null;
    clearTimeout(timer);
    timer = null;
    // The panel may be hidden; its DOM still retains the complete typed request and transcript.
    const body = $('xr-body'), route = $('xr-route'), clock = $('xr-clock');
    if (body) body.replaceChildren(h('p', { class: 'muted', text: privateMode
      ? 'Private mode. Decision content is not kept here.' : 'Reset. Waiting for the next decision.' }));
    if (route) { route.className = 'xr-badge'; route.textContent = ''; }
    if (clock) clock.textContent = '–';
    renderHistory();
  }

  function setPrivacy(value) {
    const next = value !== 'on';
    if (next || next !== privateMode) {
      privateMode = next;
      reset();
    }
  }

  document.addEventListener('lococisco:privacy', (ev) => setPrivacy(ev.detail));
  document.addEventListener('lococisco:reset', () => { contentAfter = Date.now(); reset(); });

  window.XRay = {
    handle, render,
    toggle: () => setOpen(!open),
    open: () => setOpen(true),
    close: () => setOpen(false),
    isOpen: () => open,
    setThreshold: (t) => { if (typeof t === 'number') { threshold = t; if (open) render(); } },
    reset,
    resetStream: () => { epoch = 0; contentAfter = 0; reset(); },
    _items: items,
  };
  document.addEventListener('DOMContentLoaded', () => {
    const close = $('xr-close');
    if (close) close.addEventListener('click', () => setOpen(false));
  });
})();
