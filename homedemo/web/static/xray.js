/* Decider X-ray ("How the house decided") and the scoreboard. Plain browser JS, loaded after app.js;
   it listens to the same feed (window.homeFeed) and uses the same labels (window.homeLabels).

   One X-ray item per thing the house heard or saw (voice.utterance / perception.scene root event). The
   item collects the proposals that answer it: the Decider 2B's typed request and raw answers, the shadow
   answer when the 2B was unsure, Claude's (or the 12B's) final answer with tool calls and tokens, the
   keyword baseline, the policy verdict and the outcome. Each model call is a "leg" with a stopwatch that
   runs from the moment the root event arrived until its proposal lands (then it snaps to latency_ms).
   The scoreboard is computed from the finished items since page load or reset. */
(() => {
  'use strict';
  const feed = window.homeFeed, L = window.homeLabels;
  if (!feed || !L) return;

  // ------------------------------------------------------------------ helpers
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
  const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };
  const words = (a) => String(a ?? '').replace(/_/g, ' ');
  const trunc = (str, n) => (str = String(str ?? ''), str.length > n ? str.slice(0, n - 1) + '…' : str);
  /* Whole percent, but one decimal near a threshold so 0.7955 never reads "80%" beside an 80% line. */
  const pct = (p, t) => {
    const v = (Number(p) || 0) * 100, r = Math.round(v);
    return t != null && r === Math.round(t * 100) && v !== t * 100 ? `${Math.floor(v * 10) / 10}%` : `${r}%`;
  };
  const secs = (ms) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(ms >= 10000 ? 1 : 2)} s`);
  const num = (n) => Math.round(Number(n) || 0).toLocaleString('en-US');
  const tokensOf = (d) => (d && d.tokens ? (Number(d.tokens.input) || 0) + (Number(d.tokens.output) || 0) : 0);
  const median = (xs) => { if (!xs.length) return null; const s = [...xs].sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let paused = false;
  feed.on('motion', (m) => { paused = !!(m && m.paused); });
  const motionOk = () => !paused && !reduced.matches;

  // scenes that can lead to a model decision (union of the real scene decider's and the fake's triggers)
  const SCENE_TRIGGERS = new Set(['activity_changed', 'idle', 'hazard', 'person_entered']);
  const SHOW_PENDING = new Set(['hazard', 'activity_changed']);   // shown before any proposal arrives
  const PENDING_TTL_MS = 15000;    // a root event nobody decided on is dropped after this
  const HISTORY = 6;
  const QTYPE = { choice: 'choice', noul: 'yes / no', score: 'score' };

  // ------------------------------------------------------------------ state
  const byId = new Map();          // event id -> event (cause chains)
  const items = [];                // X-ray items, oldest first
  const byRoot = new Map();        // root event id -> item
  let settings = null;
  let current = null;              // the item shown large
  let fullText = false;            // criteria descriptions: one line or full text
  let ticker = null;

  function remember(ev) {
    byId.set(ev.id, ev);
    if (byId.size > 3000) { const it = byId.keys(); for (let i = 0; i < 500; i++) byId.delete(it.next().value); }
  }
  function rootOf(ev) {
    let cur = ev, hops = 0;
    while (cur && hops++ < 24) {
      if (cur.topic === 'voice.utterance' || cur.topic === 'perception.scene') return cur;
      cur = cur.cause ? byId.get(cur.cause) : null;
    }
    return null;
  }
  const expectedFirst = () => (settings && settings.decider_mode === '12b' ? 'Strands Decider 12B' : 'Strands Decider 2B');
  const expectedRoute = () => (settings && settings.decider_mode === '12b' ? '12b' : 'local');
  const escalationRoute = () => (settings && settings.escalate_to === '12b' ? '12b' : 'bedrock');
  const escalationModel = () => (escalationRoute() === '12b' ? 'Strands Decider 12B' : 'Claude on Bedrock');

  function newItem(root, props) {
    const item = {
      id: root.id, root, startedAt: performance.now(), wall: Date.now(), legs: [], first: null, final: null, finals: [],
      keyword: null, reflex: null, verdict: null, outcome: null, done: false, visible: true, ...props,
    };
    item.legs.push({ label: expectedFirst(), route: expectedRoute(), startedAt: performance.now(), latency: null, prop: null });
    items.push(item);
    byRoot.set(root.id, item);
    while (items.length > 40) { const old = items.shift(); byRoot.delete(old.id); if (current === old) current = null; }
    return item;
  }

  // ------------------------------------------------------------------ events
  function onUtterance(ev) {
    const u = ev.data || {};
    const who = u.source === 'tv' ? 'TV voice' : u.source === 'outside' ? 'voice from outside the door'
      : `${u.speaker?.name || 'unknown voice'}${u.speaker?.role && u.speaker?.name ? ' (' + u.speaker.role + ')' : ''}`;
    const item = newItem(ev, { kind: 'speech', text: u.text || '', who, src: u.source || 'person', room: u.room, context: u.context || '' });
    show(item);
  }
  function onScene(ev) {
    const s = ev.data || {};
    if (!SCENE_TRIGGERS.has(s.trigger) || !s.text) return;
    const item = newItem(ev, { kind: 'scene', text: s.text, who: 'camera and sensors', src: 'camera', room: s.room, trigger: s.trigger, visible: SHOW_PENDING.has(s.trigger) });
    if (item.visible) show(item);
  }
  function itemFor(ev) {
    const root = rootOf(ev);
    let item = root && byRoot.get(root.id);
    if (item) return item;
    const d = ev.data || {};
    // an old producer, or a cause this page never saw: build the item from the proposal itself
    const fake = { id: root ? root.id : ev.id, topic: d.kind === 'scene' ? 'perception.scene' : 'voice.utterance', data: {} };
    const said = /Heard: "(.*)"\s*$/s.exec(d.state_text || d.request?.state || '');
    item = newItem(fake, {
      kind: d.kind || 'speech', text: d.kind === 'scene' ? (d.state_text || d.request?.state || '') : (said ? said[1] : d.state_text || ''),
      who: d.kind === 'scene' ? 'camera and sensors' : (d.speaker?.name || 'unknown voice'), src: d.kind === 'scene' ? 'camera' : 'person', room: d.room,
    });
    return item;
  }
  function onProposal(ev) {
    const d = ev.data || {};
    const item = itemFor(ev);
    if (d.by === 'keyword') item.keyword = d;
    else if (d.by === 'reflex') item.reflex = d;
    else if (L.isModel(d)) {
      if (item.final && d.mode !== 'shadow' && d.escalated_from && item.final.escalated_from === d.escalated_from) {
        item.finals.push(d); item.final = d;              // Claude made a second tool call
      } else {
        let leg = item.legs.find((l) => l.latency == null);
        if (!leg) { leg = { startedAt: performance.now(), latency: null, prop: null }; item.legs.push(leg); }
        leg.label = L.model(d); leg.route = L.route(d); leg.latency = Number(d.latency_ms) || 0; leg.prop = d;
        if (d.mode === 'shadow') {
          item.first = d;
          item.legs.push({ label: escalationModel(), route: escalationRoute(), startedAt: performance.now(), latency: null, prop: null });
        } else {
          if (!item.first) item.first = d;
          item.final = d; item.finals = [d]; item.done = true;
        }
      }
    }
    item.visible = true;
    show(item);
    scoreboard();
  }
  function onVerdict(ev) {
    const d = ev.data || {};
    const prop = byId.get(d.proposal_id) || (ev.cause && byId.get(ev.cause));
    const root = prop ? rootOf(prop) : rootOf(ev);
    const item = root && byRoot.get(root.id);
    if (!item) return;
    item.verdict = d;
    if (d.verdict === 'deny') item.outcome = 'nothing happened (denied)';
    else if (d.verdict === 'ignore') item.outcome = 'nothing happened';
    show(item, false);
    scoreboard();
  }
  function describeState(st) {
    if (!st) return '';
    if ('on' in st) return st.on ? `on${st.playing ? ' (' + st.playing + ')' : st.brightness ? ' ' + st.brightness + '%' : ''}` : 'off';
    if ('locked' in st) return st.locked ? 'locked' : 'unlocked';
    if ('setpoint_c' in st) return `set to ${Number(st.setpoint_c).toFixed(1)} °C`;
    if ('burner_on' in st) return st.burner_on ? 'burner on' : 'burner off';
    return JSON.stringify(st);
  }
  function onOutcome(ev) {
    const root = rootOf(ev);
    const item = root && byRoot.get(root.id);
    if (!item) return;
    const d = ev.data || {};
    if (ev.topic === 'action.applied') item.outcome = `${d.device_id}: ${describeState(d.state)}`;
    else if (ev.topic === 'action.command' && d.command === 'notify') item.outcome = `alert sent to ${d.args?.to || 'parents'}`;
    else if (ev.topic === 'notify.ask') item.outcome = `asked ${d.to}: “${trunc(d.text, 60)}”`;
    else if (ev.topic === 'notify.reply') item.outcome = `${d.by === 'web' ? 'parent' : d.by} answered ${d.answer}${d.answer === 'Yes' ? '' : ' → nothing happened'}`;
    else return;
    show(item, false);
  }

  function handle(ev) {
    if (!ev || !ev.topic) return;
    remember(ev);
    switch (ev.topic) {
      case 'voice.utterance': return onUtterance(ev);
      case 'perception.scene': return onScene(ev);
      case 'intent.proposal': return onProposal(ev);
      case 'policy.verdict': return onVerdict(ev);
      case 'action.applied': case 'action.command': case 'notify.ask': case 'notify.reply': return onOutcome(ev);
      default: return undefined;
    }
  }

  /* Same judgement as the compare table: would the keyword matcher have done the wrong thing? */
  function judgeKeyword(item) {
    if (!item.keyword || !item.verdict) return null;
    const kw = item.keyword.action, v = item.verdict.verdict;
    if (v === 'allow') return { wrong: kw !== item.verdict.action, why: kw === item.verdict.action ? 'same result' : kw === 'none' ? 'heard no command: nothing would have happened' : `would have done “${words(kw)}” instead` };
    if (v === 'ask_parent' || v === 'ask_speaker') return { wrong: kw !== 'none', why: kw !== 'none' ? `would have done “${words(kw)}” without asking` : 'same result' };
    return { wrong: kw !== 'none', why: kw !== 'none' ? `would have obeyed: “${words(kw)}”` : 'same result' };
  }

  // ------------------------------------------------------------------ selection + clocks
  function show(item, takeover = true) {
    if (takeover && (!current || item === current || item.startedAt >= current.startedAt)) current = item;
    if (item === current) render();
    else renderHistory();
    clocks();
  }
  function prune() {
    const now = performance.now();
    let changed = false;
    for (const item of [...items]) {
      if (item.first || item.reflex || item.final || now - item.startedAt < PENDING_TTL_MS) continue;
      items.splice(items.indexOf(item), 1); byRoot.delete(item.id); changed = true;
      if (current === item) current = [...items].reverse().find((x) => x.visible) || null;
    }
    if (changed) render();
  }
  /* Keeps the stopwatches running while any leg of the current item is in flight. */
  function clocks() {
    const pending = current && current.legs.some((l) => l.latency == null);
    if (pending && !ticker) ticker = setInterval(tick, motionOk() ? 50 : 250);
    if (!pending && ticker) { clearInterval(ticker); ticker = null; }
    tick();
  }
  function tick() {
    prune();
    if (!current) return;
    for (const el of document.querySelectorAll('#xray [data-leg]')) {
      const leg = current.legs[Number(el.dataset.leg)];
      if (leg && leg.latency == null) el.textContent = `${((performance.now() - leg.startedAt) / 1000).toFixed(2)} s…`;
    }
    $('#xr-clock').textContent = clockText(current);
  }
  function clockText(item) {
    return item.legs.map((l) => `${L.short(l.prop || { by: l.route === '12b' ? 'decider-12b' : l.route === 'bedrock' ? 'bedrock' : 'decider-2b' })} ${l.latency == null ? ((performance.now() - l.startedAt) / 1000).toFixed(2) + ' s…' : secs(l.latency)}`).join(' + ');
  }

  // ------------------------------------------------------------------ JSON view with syntax colours
  function strNode(s, cls = 'j-str') {
    return h('span', { class: cls, text: JSON.stringify(String(s)) });
  }
  function qBadge(q) {
    const t = q && q.type;
    const n = q && q.criteria ? Object.keys(q.criteria).length : 0;
    return h('span', { class: `qtype ${t || 'other'}`, text: `${QTYPE[t] || t || '?'}${t === 'choice' && n ? ` · ${n} options` : ''}` });
  }
  /* Pretty JSON as DOM lines. `answers` colours the criteria of each question with the decider's
     probabilities and marks the chosen option. */
  const presenting = () => document.body.classList.contains('present');
  /* The answer to a yes/no (noul) or choice question, as a short chip. */
  function answerChip(a) {
    if (!a) return null;
    if (a.type === 'noul' && a.noul != null) return h('span', { class: 'prob', text: `yes ${pct(a.noul)}` });
    if (a.choice != null) return h('span', { class: 'prob', text: `${a.choice}${a.confidence != null ? ' ' + pct(a.confidence) : ''}` });
    return null;
  }
  function renderJson(value, answers = null, primary = null, threshold = null) {
    const compact = presenting() && !fullText;   // presenter mode: secondary questions fold to one line
    const root = h('div', { class: `json ${fullText ? 'full' : ''}` });
    const line = (depth, ...parts) => root.append(h('div', { class: 'j-line', style: `--d:${depth}` }, ...parts));
    const walk = (v, depth, key, path, last) => {
      const comma = last ? '' : ',';
      const keyNode = key == null ? [] : [h('span', { class: 'j-key', text: JSON.stringify(key) }), h('span', { class: 'j-p', text: ': ' })];
      if (v && typeof v === 'object' && !Array.isArray(v)) {
        const keys = Object.keys(v);
        const isQuestion = path.length === 2 && path[0] === 'questions';
        const badge = isQuestion ? [' ', qBadge(v)] : [];
        if (!keys.length) { line(depth, ...keyNode, h('span', { class: 'j-p', text: '{}' + comma })); return; }
        if (isQuestion && compact && primary && key !== primary) {
          // one line: the key, its type badge, the instructions and the decider's answer
          root.append(h('div', { class: 'j-line folded', style: `--d:${depth}`, title: String(v.instructions || '') }, ...keyNode,
            h('span', { class: 'j-p', text: '{ ' }), ...badge, ' ',
            h('span', { class: 'j-str fold', text: JSON.stringify(String(v.instructions || '')) }),
            h('span', { class: 'j-p', text: ' ⋯ }' + comma }), answers ? answerChip(answers[key]) : null));
          return;
        }
        line(depth, ...keyNode, h('span', { class: 'j-p', text: '{' }), ...badge);
        const inCriteria = path.length === 3 && path[0] === 'questions' && path[2] === 'criteria';
        const ans = inCriteria && answers ? answers[path[1]] : null;
        keys.forEach((k, i) => {
          if (isQuestion && compact && k === 'type') return;           // the badge already says it
          if (isQuestion && k === 'instructions' && typeof v[k] === 'string') {
            line(depth + 1, h('span', { class: 'j-key', text: JSON.stringify(k) }), h('span', { class: 'j-p', text: ': ' }),
              h('span', { class: `j-str ${compact ? 'fold wide' : ''}`, text: JSON.stringify(v[k]), title: v[k] }), h('span', { class: 'j-p', text: i === keys.length - 1 ? '' : ',' }));
            return;
          }
          if (inCriteria) {
            const p = ans && ans.probabilities ? ans.probabilities[k] : null;
            const picked = ans ? ans.choice === k : false;
            const row = h('div', { class: `j-line opt ${picked ? 'pick' : ''} ${p != null ? 'has-bar' : ''}`, style: `--d:${depth + 1}`, title: String(v[k]) },
              h('span', { class: 'j-key opt-key', text: JSON.stringify(k) }), h('span', { class: 'j-p', text: ': ' }),
              h('span', { class: 'j-str opt-desc', text: JSON.stringify(String(v[k])) + (i === keys.length - 1 ? '' : ',') }));
            if (p != null) {
              // the option list doubles as the probability chart: a mini bar with the act threshold
              const fill = h('i', { class: 'fill', style: 'width:0%' });
              const bar = h('span', { class: 'mini' }, fill, threshold != null ? h('i', { class: 'th', style: `left:${threshold * 100}%` }) : null);
              row.append(bar, h('span', { class: 'prob', text: pct(p, threshold) }));
              const w = `${Math.max(1, Math.round(p * 1000) / 10)}%`;
              if (motionOk()) requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = w; })); else fill.style.width = w;
            }
            root.append(row);
          } else walk(v[k], depth + 1, k, [...path, k], i === keys.length - 1);
        });
        line(depth, h('span', { class: 'j-p', text: '}' + comma }));
      } else if (Array.isArray(v)) {
        if (!v.length) { line(depth, ...keyNode, h('span', { class: 'j-p', text: '[]' + comma })); return; }
        line(depth, ...keyNode, h('span', { class: 'j-p', text: '[' }));
        v.forEach((x, i) => walk(x, depth + 1, null, [...path, i], i === v.length - 1));
        line(depth, h('span', { class: 'j-p', text: ']' + comma }));
      } else {
        const cls = typeof v === 'string' ? 'j-str' : typeof v === 'number' ? 'j-num' : 'j-bool';
        const node = typeof v === 'string' ? strNode(v) : h('span', { class: cls, text: JSON.stringify(v) });
        line(depth, ...keyNode, node, h('span', { class: 'j-p', text: comma }));
      }
    };
    walk(value, 0, null, [], true);
    return root;
  }
  window.homeXray = { renderJson: (v) => renderJson(v) };

  // ------------------------------------------------------------------ render: the current item
  function primaryKey(d) {
    const a = d && d.answers ? d.answers : {};
    return ['intent', 'action', 'tool'].find((k) => a[k]) || Object.keys(a).find((k) => a[k] && a[k].type === 'choice') || null;
  }
  /* The question whose options are the actions (the choice question), when no answers are available. */
  function primaryQuestion(questions) {
    const q = questions || {};
    return ['intent', 'action', 'tool'].find((k) => q[k]) || Object.keys(q).find((k) => q[k] && q[k].type === 'choice') || null;
  }
  function render() {
    const body = clear($('#xr-body'));
    const badge = $('#xr-route');
    const item = current;
    if (!item) {
      badge.textContent = 'waiting'; badge.className = 'xr-badge waiting'; $('#xr-clock').textContent = '–';
      body.append(h('p', { class: 'muted', text: 'The newest decision appears here: what was heard or seen, the exact typed request sent to the Decider 2B, its answer against the act threshold, and where the decision was made.' }));
      renderHistory();
      return;
    }
    const final = item.final, first = item.first || final;
    const route = final ? L.route(final) : item.reflex && !first ? 'rule' : null;
    badge.className = `xr-badge ${route ? 'route-' + route : 'pending'}`;
    const lastLeg = item.legs[item.legs.length - 1];
    badge.textContent = route ? L.routeWord(final || item.reflex)
      : item.legs.length > 1 ? `asking ${lastLeg.route === '12b' ? 'the 12B' : 'Claude'}…` : `${item.kind === 'speech' ? 'listening' : 'looking'}…`;
    $('#xr-clock').textContent = clockText(item);

    // what was heard or seen
    const heard = h('div', { class: `xr-heard kind-${item.kind} src-${item.src}` },
      h('div', { class: 'xr-who' },
        h('span', { class: `chip src-${item.src}`, text: item.kind === 'speech' ? `voice: ${item.src}` : 'camera' }),
        h('b', { text: item.who }), item.room ? ` · ${roomName(item.room)}` : '', item.trigger ? ` · ${words(item.trigger)}` : ''),
      item.kind === 'speech' ? h('q', { class: 'xr-text', text: item.text }) : h('p', { class: 'xr-text scene', text: item.text }));
    body.append(heard);

    // the legs: who was asked, where, how long
    const legs = h('ol', { class: 'xr-legs' });
    item.legs.forEach((leg, i) => {
      const p = leg.prop;
      const done = leg.latency != null;
      const li = h('li', { class: `leg route-${leg.route} ${done ? 'done' : 'pending'}` },
        h('span', { class: 'leg-n', text: String(i + 1) }),
        h('span', { class: 'leg-model', text: leg.label }),
        h('span', { class: 'leg-where', text: whereWord(leg.route) }),
        h('span', { class: 'leg-time', 'data-leg': i, text: done ? secs(leg.latency) : '0.00 s…' }));
      if (p && L.route(p) === 'bedrock') {
        const calls = (p.tool_calls || []).map((c) => `${c.name}(${Object.entries(c.input || {}).map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`).join(', ')})`);
        li.append(h('span', { class: 'leg-result sure' }, 'tool call ', h('b', { text: calls.join(', ') || words(p.action) }),
          p.tokens ? ` · ${num(tokensOf(p))} tokens` : ''));
      } else if (p) {
        const unsure = p.mode === 'shadow';
        const short = presenting();
        li.append(h('span', { class: `leg-result ${unsure ? 'unsure' : 'sure'}` },
          h('b', { text: words(p.action) }), ` ${pct(p.confidence, p.threshold)}`,
          p.threshold != null && !short ? ` ${p.confidence >= p.threshold ? '≥' : '<'} ${pct(p.threshold)}` : '',
          unsure ? (short ? ' → unsure' : ' → unsure, escalated') : ' → sure'));
      }
      legs.append(li);
    });
    if (item.reflex && !first) legs.append(h('li', { class: 'leg route-rule done' }, h('span', { class: 'leg-n', text: '•' }), h('span', { class: 'leg-model', text: L.model(item.reflex) }), h('span', { class: 'leg-where', text: 'on the hub' }), h('span', { class: 'leg-time', text: secs(item.reflex.latency_ms || 0) }), h('span', { class: 'leg-result sure' }, h('b', { text: words(item.reflex.action) }), ` · ${item.reflex.reason || ''}`)));
    body.append(legs);

    // the typed request and the answer
    if (first && first.request) {
      const req = first.request;
      const sec = h('section', { class: 'xr-request' },
        h('div', { class: 'xr-sec-head' },
          h('h3', {}, 'Typed request → ', h('b', { text: L.model(first) })),
          h('span', { class: 'hint', text: 'one call: a state and bounded questions; the answer is a choice with a calibrated probability, never free text' }),
          h('button', { class: 'btn tiny', type: 'button', text: fullText ? 'short text' : 'full text', onclick: () => { fullText = !fullText; render(); } })));
      sec.append(renderJson({ state: req.state, questions: req.questions || {} }, first.answers || null, primaryKey(first) || primaryQuestion(req.questions),
        first.threshold ?? (settings && settings.act_threshold) ?? null));
      body.append(sec);
    } else if (first) {
      body.append(h('section', { class: 'xr-request' }, h('h3', { text: 'State the decider read' }), h('pre', { class: 'xr-state', text: first.state_text || '(no state text)' }), first.question ? h('p', { class: 'muted small', text: first.question }) : null));
    }
    // the answer and where the decision was made (one column next to the request)
    const side = h('div', { class: 'xr-side' });
    if (first) side.append(renderAnswer(item, first));
    side.append(renderRoute(item));
    body.append(side);
    renderHistory();
  }
  function roomName(id) {
    const snap = feed.last('snapshot');
    return (snap && (snap.rooms || []).find((r) => r.id === id)?.name) || id;
  }
  const whereWord = (route) => ({ local: 'on the hub', '12b': 'on the hub', bedrock: 'Amazon Bedrock', rule: 'on the hub' })[route] || '';

  function renderAnswer(item, d) {
    const key = primaryKey(d);
    const ans = key ? d.answers[key] : null;
    const probs = ans && ans.probabilities && Object.keys(ans.probabilities).length ? Object.entries(ans.probabilities) : Object.entries(d.probabilities || {});
    const picked = ans && ans.choice != null ? ans.choice : d.action;
    const threshold = d.threshold ?? (settings && settings.act_threshold) ?? 0.8;
    const sec = h('section', { class: 'xr-answer' });
    sec.append(h('div', { class: 'xr-sec-head' }, h('h3', {}, 'Answer · ', h('b', { text: L.model(d) })),
      h('span', { class: 'hint', text: `${key ? `“${key}” ` : ''}probabilities; act at ${pct(threshold)}` })));
    const bars = h('div', { class: 'xr-bars' });
    const sorted = probs.sort((a, b) => b[1] - a[1]).slice(0, 6);
    for (const [label, p] of sorted) {
      const isPick = label === picked;
      const fill = h('i', { class: 'fill', style: 'width:0%' });
      const track = h('div', { class: 'track' }, fill, h('i', { class: 'th act', style: `left:${threshold * 100}%`, title: `act at ${pct(threshold)}` }));
      bars.append(h('div', { class: `row ${isPick ? 'pick' : ''}` }, h('span', { class: 'lbl', text: words(label) }), track, h('span', { class: 'val', text: pct(p, threshold) })));
      const w = `${Math.max(0.5, Math.round(p * 1000) / 10)}%`;
      if (motionOk()) requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = w; })); else fill.style.width = w;
    }
    sec.append(bars);
    const gate = d.answers ? Object.entries(d.answers).find(([k, a]) => k !== key && a && a.type === 'noul') : null;
    const sure = d.confidence >= threshold;
    sec.append(h('div', { class: `xr-conf ${sure ? 'sure' : 'unsure'}` },
      h('b', { class: 'big', text: pct(d.confidence, threshold) }),
      h('span', { text: ` confidence ${sure ? '≥' : '<'} ${pct(threshold)} threshold → ` }),
      h('b', { text: sure ? (L.route(d) === 'bedrock' ? 'Claude decided' : `${L.short(d)} is sure: act on the hub`) : (d.mode === 'shadow' ? `unsure: ask ${item.legs[1] ? item.legs[1].label : L.escalation(settings)}` : 'unsure') }),
      gate ? h('span', { class: 'gate', text: ` · gate “${gate[0]}”: ${gate[1].noul != null ? `yes ${pct(gate[1].noul)}` : gate[1].choice}` }) : null));
    return sec;
  }

  function renderRoute(item) {
    const final = item.final;
    const sec = h('section', { class: 'xr-route' });
    if (final) {
      const route = L.route(final), tok = tokensOf(final);
      const t = final.tokens;
      const line = h('div', { class: `xr-where route-${route}` }, h('span', { class: `xr-badge route-${route}`, text: L.routeWord(final) }));
      if (route === 'bedrock') {
        line.append(h('b', { text: L.model(final) }), t ? ` · ${num(tok)} tokens (${num(t.input)} in / ${num(t.output)} out)` : '', ` · ${secs(final.latency_ms || 0)}`);
      } else if (route === '12b') {
        line.append(h('b', { text: `${L.model(final)} on the hub` }), ' · 0 tokens · nothing left the house', ` · ${secs(final.latency_ms || 0)}`);
      } else {
        line.append(h('b', { text: `${L.model(final)} on the hub` }), ' · 0 tokens · nothing left the house', ` · ${secs(final.latency_ms || 0)}`);
      }
      sec.append(line);
      if (route === 'bedrock') {
        // what crossed the house boundary: Claude gets text that the hub wrote, never the recording
        sec.append(h('div', { class: 'xr-privacy' }, h('span', { class: 'xr-badge route-local', text: 'PRIVATE' }),
          item.kind === 'scene' ? 'Claude got the scene as text · the video stayed home' : 'Claude got this sentence as text · the audio stayed home'));
      }
      const calls = (final.tool_calls && final.tool_calls.length) ? final.tool_calls : item.finals.flatMap((f) => f.tool_calls || []);
      if (calls.length) {
        sec.append(h('div', { class: 'xr-tools' }, 'Strands tool calls: ', calls.map((c) => h('code', { text: `${c.name}(${Object.entries(c.input || {}).map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`).join(', ')})` }))));
      } else if (route === 'bedrock') {
        sec.append(h('div', { class: 'xr-tools muted', text: 'no tool call: nothing to do' }));
      }
      if (final.reply) sec.append(h('div', { class: 'xr-reply' }, h('span', { class: 'spk', text: 'hub speaker' }), h('q', { text: final.reply })));
      if (final.reason && route !== 'bedrock') sec.append(h('div', { class: 'xr-reason muted', text: final.reason }));
    } else if (item.reflex) {
      sec.append(h('div', { class: 'xr-where route-rule' }, h('span', { class: 'xr-badge route-rule', text: 'RULE' }), h('b', { text: `${L.model(item.reflex)} on the hub` }), ` · 0 tokens · ${secs(item.reflex.latency_ms || 0)}`, ` · ${words(item.reflex.action)}: ${item.reflex.reason || ''}`));
    } else {
      sec.append(h('div', { class: 'xr-where pending muted', text: item.legs.length > 1 ? `the 2B was unsure; ${item.legs[item.legs.length - 1].label} is deciding…` : 'deciding…' }));
    }
    const kw = item.keyword ? judgeKeyword(item) : null;
    if (item.keyword) {
      sec.append(h('div', { class: `xr-kw ${kw ? (kw.wrong ? 'wrong' : 'ok') : 'pending'}` },
        h('span', { class: 'lbl', text: 'keyword baseline' }), h('code', { text: words(item.keyword.action) }),
        h('span', { text: kw ? ` ${kw.wrong ? '✗ wrong: ' : '✓ '}${kw.why}` : ' · judging…' })));
    }
    if (item.verdict) {
      const v = item.verdict;
      const verb = v.verdict === 'ask_parent' ? 'ask a parent' : v.verdict === 'ask_speaker' ? 'ask the speaker' : v.verdict;
      sec.append(h('div', { class: 'xr-policy' }, h('span', { class: `badge v-${v.verdict}`, text: verb }), h('span', { class: 'rule', text: ` policy: ${words(v.rule || '')}` }), item.outcome ? h('span', { class: 'outcome', text: ` → ${item.outcome}` }) : null));
    } else if (item.outcome) sec.append(h('div', { class: 'xr-policy' }, h('span', { class: 'outcome', text: item.outcome })));
    return sec;
  }

  // ------------------------------------------------------------------ history
  function renderHistory() {
    const box = clear($('#xr-history'));
    const rows = [...items].reverse().filter((it) => it !== current && (it.final || it.reflex)).slice(0, HISTORY);
    box.hidden = !rows.length;
    if (!rows.length) return;
    box.append(h('h3', { text: 'Earlier decisions' }));
    for (const it of rows) {
      const d = it.final || it.reflex;
      const route = L.route(d);
      const kw = judgeKeyword(it);
      const tok = tokensOf(d);
      box.append(h('button', { class: `xr-row route-${route}`, type: 'button', title: 'show this decision', onclick: () => { current = it; render(); clocks(); } },
        h('span', { class: `xr-badge mini route-${route}`, text: L.routeWord(d) }),
        h('q', { class: 'said', text: trunc(it.text, 70) }),
        h('span', { class: 'res' }, h('b', { text: words(d.action) }), ` · ${route === 'bedrock' ? L.short(d) : pct(d.confidence)} · ${secs(it.legs.reduce((s, l) => s + (l.latency || 0), 0) || d.latency_ms || 0)}${route === 'bedrock' ? ` · ${num(tok)} tokens` : ' · 0 tokens'}`),
        it.keyword ? h('span', { class: `kw ${kw ? (kw.wrong ? 'wrong' : 'ok') : ''}`, text: `keyword ${kw ? (kw.wrong ? '✗' : '✓') : '…'}` }) : null));
    }
  }

  // ------------------------------------------------------------------ scoreboard
  const sbEl = (k) => document.querySelector(`[data-sb="${k}"]`);
  function animateNumber(el, to, fmt) {
    if (!el) return;
    const from = Number(el.dataset.v || 0);
    el.dataset.v = String(to);
    if (!motionOk() || from === to || !Number.isFinite(from)) { el.textContent = fmt(to); return; }
    const t0 = performance.now(), dur = 650;
    const step = (now) => {
      const t = Math.min(1, (now - t0) / dur), k = 1 - Math.pow(1 - t, 3);
      el.textContent = fmt(from + (to - from) * k);
      if (t < 1) requestAnimationFrame(step); else el.textContent = fmt(to);
    };
    requestAnimationFrame(step);
    el.classList.remove('bump'); void el.offsetWidth; el.classList.add('bump');
  }
  function scoreboard() {
    const done = items.filter((it) => it.final);
    const local = done.filter((it) => L.route(it.final) === 'local');
    const esc = done.filter((it) => L.route(it.final) !== 'local');
    const legTotal = (it) => it.legs.reduce((s, l) => s + (l.latency || 0), 0) || (it.final.latency_ms || 0);
    const mLocal = median(local.map(legTotal)), mEsc = median(esc.map(legTotal));
    const tokens = esc.reduce((s, it) => s + Math.max(tokensOf(it.final), ...it.finals.map(tokensOf)), 0);
    const kwWrong = items.filter((it) => judgeKeyword(it)?.wrong).length;
    const kwJudged = items.filter((it) => judgeKeyword(it)).length;
    const escRoute = esc.some((it) => L.route(it.final) === 'bedrock') ? 'bedrock' : esc.some((it) => L.route(it.final) === '12b') ? '12b' : escalationRoute();
    animateNumber(sbEl('decisions'), done.length, (v) => num(v));
    const scenes = done.filter((it) => it.kind === 'scene').length;
    sbEl('decisions-sub').textContent = done.length ? `${done.length - scenes} speech · ${scenes} scene${scenes === 1 ? '' : 's'}` : 'speech and camera scenes';
    animateNumber(sbEl('local'), local.length, (v) => num(v));
    sbEl('local-sub').textContent = `${mLocal != null ? 'median ' + secs(mLocal) + ' · ' : ''}0 tokens · nothing left the house`;
    animateNumber(sbEl('escalated'), esc.length, (v) => num(v));
    sbEl('escalated-label').textContent = escRoute === '12b' ? 'sent to the Decider 12B' : 'sent to Claude on Bedrock';
    sbEl('escalated-sub').textContent = escRoute === '12b' ? `${mEsc != null ? 'median ' + secs(mEsc) + ' (2B + 12B) · ' : ''}0 tokens` : `${mEsc != null ? 'median ' + secs(mEsc) + ' (2B + Claude) · ' : ''}${num(tokens)} tokens`;
    const share = done.length ? local.length / done.length : null;
    if (share == null) { const el = sbEl('avoided'); el.textContent = '–'; el.dataset.v = '0'; } else animateNumber(sbEl('avoided'), share * 100, (v) => `${Math.round(v)}%`);
    sbEl('avoided-sub').textContent = done.length ? `${local.length} of ${done.length} on the hub` : 'no decisions yet';
    animateNumber(sbEl('kw'), kwWrong, (v) => num(v));
    sbEl('kw-sub').textContent = kwJudged ? `wrong on ${kwWrong} of ${kwJudged} sentences` : 'the old matcher, judged against the outcome';
  }

  function reset() {
    items.length = 0; byRoot.clear(); current = null;
    for (const el of document.querySelectorAll('[data-sb]')) if (el.tagName === 'B') { el.dataset.v = '0'; }
    render(); scoreboard(); clocks();
  }

  // ------------------------------------------------------------------ wiring
  feed.on('state', (st) => { settings = st?.settings || settings; });
  feed.on('settings', (s) => { settings = s; if (current) render(); });
  feed.on('present', () => { if (current) render(); });
  feed.on('event', handle);
  feed.on('reset', reset);
  $('#sb-reset')?.addEventListener('click', reset);
  render();
  scoreboard();
})();
