/* Smart-home decider demo UI. Plain browser JS, no build step, no network beyond this origin.
   Everything is driven by the SSE stream at api/stream; api/state seeds the page. */
(() => {
  'use strict';

  // ------------------------------------------------------------------ tiny DOM helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const planSvg = $('#plan');
  const pipeSvg = $('#pipeline');
  const NS = planSvg.namespaceURI; // the SVG namespace, read from the document instead of hard-coding a URL

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
  function s(tag, attrs, text) {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
    if (text != null) el.textContent = text;
    return el;
  }
  const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };
  const trunc = (str, n) => (str = String(str ?? ''), str.length > n ? str.slice(0, n - 1) + '…' : str);
  const fmt = (x, d = 2) => (typeof x === 'number' ? x.toFixed(d) : '–');
  const words = (a) => String(a ?? '').replace(/_/g, ' ');
  const esc = (v) => (typeof v === 'string' ? v : JSON.stringify(v));

  // ------------------------------------------------------------------ who decided: labels from the payload
  // Every label comes from the proposal itself (route, model, by); nothing is hard-coded to one model.
  // Old producers only send `by`, so each `by` has a default route and model name.
  const BY_DEFAULTS = {
    'decider-2b': { route: 'local', model: 'Strands Decider 2B' },
    'decider-12b': { route: '12b', model: 'Strands Decider 12B' },
    bedrock: { route: 'bedrock', model: 'Claude on Bedrock' },
    fake: { route: 'local', model: 'fake decider' },
    llm: { route: 'bedrock', model: 'LLM' },
    reflex: { route: 'rule', model: 'plain rule' },
    keyword: { route: 'rule', model: 'keyword matcher' },
    user: { route: 'rule', model: 'phone reply' },
  };
  const MODEL_BYS = new Set(['decider-2b', 'decider-12b', 'bedrock', 'fake', 'llm']);
  const labels = {
    route: (d) => (d && d.route) || (d && BY_DEFAULTS[d.by]?.route) || 'local',
    model: (d) => (d && d.model) || (d && BY_DEFAULTS[d.by]?.model) || (d && d.by) || 'decider',
    isModel: (d) => !!d && MODEL_BYS.has(d.by),
    // one word for chips: 2B / 12B / Claude / rule
    short(d) {
      const r = labels.route(d), m = labels.model(d);
      if (r === 'rule') return d?.by === 'keyword' ? 'keyword' : 'rule';
      if (/\b12B\b/i.test(m)) return '12B';
      if (/\b2B\b/i.test(m)) return '2B';
      if (/claude/i.test(m)) return 'Claude';
      return r === 'bedrock' ? 'Bedrock' : r === '12b' ? '12B' : m;
    },
    // where the decision was made, as a badge word
    routeWord: (d) => ({ local: 'LOCAL', bedrock: 'BEDROCK', '12b': '12B', rule: 'RULE' })[labels.route(d)] || labels.route(d).toUpperCase(),
    // who answers when the 2B is unsure, from the live settings
    escalation: (st) => ((st && st.escalate_to) === '12b' ? 'the Decider 12B' : 'Claude on Bedrock'),
  };
  window.homeLabels = labels;

  async function api(path, body) {
    const res = await fetch(path, body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = data.detail ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) : res.statusText;
      toast(`${path}: ${msg}`);
      throw new Error(msg);
    }
    return data;
  }
  function toast(msg) {
    const n = $('#narration');
    n.textContent = `Problem: ${msg}`;
    n.style.borderLeftColor = 'var(--bad)';
    setTimeout(() => { n.style.borderLeftColor = ''; }, 4000);
  }

  // ------------------------------------------------------------------ feed: one stream, two views
  // The 2D plan (below) and the 3D house (static/3d/main.js, an ES module) both subscribe here, so
  // they see exactly the same /api/state seed, snapshots and bus events. "Sticky" types replay their
  // latest value to late subscribers (the module script loads after this one).
  const feed = (() => {
    const subs = new Map();
    const last = new Map();
    const sticky = new Set(['state', 'snapshot', 'settings', 'motion', 'view', 'present']);
    return {
      on(type, fn) {
        if (!subs.has(type)) subs.set(type, new Set());
        subs.get(type).add(fn);
        if (sticky.has(type) && last.has(type)) fn(last.get(type));
        return () => subs.get(type).delete(fn);
      },
      emit(type, payload) {
        if (sticky.has(type)) last.set(type, payload);
        for (const fn of subs.get(type) || []) { try { fn(payload); } catch (err) { console.error(`feed ${type}`, err); } }
      },
      last: (type) => last.get(type),
    };
  })();
  window.homeFeed = feed;

  // ------------------------------------------------------------------ state
  const state = {
    snapshot: null, settings: null, agents: [], scenarios: [], home: null,
    byId: new Map(), count: 0, utteranceRows: new Map(), proposals: new Map(),
    selectedScenario: null, running: null, paused: false, lastSnapshotAt: 0,
    dec: { main: null, shadow: null }, view: '2d', scene3d: false,
  };
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const motionOk = () => !state.paused && !reduced.matches;

  const GROUPS = [
    ['sensors', (t) => t.startsWith('sensor.') || t === 'sim.clock'],
    ['perception', (t) => t.startsWith('perception.') || t.startsWith('voice.')],
    ['proposals', (t) => t.startsWith('intent.')],
    ['policy', (t) => t.startsWith('policy.')],
    ['actions', (t) => t.startsWith('action.')],
    ['phone', (t) => t.startsWith('notify.')],
    ['scenario', (t) => t.startsWith('scenario.') || t.startsWith('trace.')],
    ['ui', (t) => t.startsWith('ui.')],
  ];
  const groupOf = (topic) => (GROUPS.find(([, f]) => f(topic)) || ['other'])[0];
  const topicMatches = (pattern, topic) => pattern === '*' || pattern === topic
    || (pattern.endsWith('.*') && topic.startsWith(pattern.slice(0, -1)));

  // ------------------------------------------------------------------ floor plan
  const plan = { built: false, rooms: new Map(), devices: new Map(), avatars: new Map(), layers: {} };

  function buildPlan(snap) {
    clear(planSvg);
    plan.rooms.clear(); plan.devices.clear(); plan.avatars.clear();
    const layers = plan.layers = {};
    for (const name of ['rooms', 'devices', 'mics', 'rays', 'dets', 'avatars', 'bubbles']) {
      layers[name] = s('g', { id: `layer-${name}` });
      planSvg.append(layers[name]);
    }
    for (const r of snap.rooms) {
      const g = s('g', { class: `room ${r.indoor === false ? 'outdoor' : 'indoor'}`, 'data-room': r.id });
      g.append(s('rect', { class: 'room-floor', x: r.x + 2, y: r.y + 2, width: r.w - 4, height: r.h - 4, rx: 8 }));
      g.append(s('text', { class: 'room-name', x: r.x + 14, y: r.y + 28 }, r.name));
      const env = s('text', { class: 'room-env', x: r.x + 14, y: r.y + 46 }, '');
      g.append(env);
      layers.rooms.append(g);
      plan.rooms.set(r.id, { g, env, r });
    }
    for (const d of snap.devices) {
      const g = s('g', { class: `dev ${d.kind}`, 'data-id': d.device_id });
      const parts = {};
      if (d.kind === 'light') {
        parts.glow = s('circle', { class: 'glow', cx: d.x, cy: d.y, r: 95 });
        g.append(parts.glow, s('circle', { class: 'bulb', cx: d.x, cy: d.y, r: 9 }));
      } else if (d.kind === 'tv') {
        g.append(s('rect', { class: 'tv-frame', x: d.x - 16, y: d.y - 46, width: 32, height: 92, rx: 4 }));
        g.append(s('rect', { class: 'tv-screen', x: d.x - 11, y: d.y - 40, width: 22, height: 80, rx: 2 }));
        parts.text = s('text', { class: 'tv-text', x: d.x + 24, y: d.y + 4 }, 'TV off');
        g.append(parts.text);
      } else if (d.kind === 'stove') {
        g.append(s('rect', { class: 'stove-body', x: d.x - 22, y: d.y - 16, width: 44, height: 32, rx: 4 }));
        g.append(s('circle', { class: 'burner', cx: d.x - 9, cy: d.y, r: 7 }), s('circle', { class: 'burner', cx: d.x + 9, cy: d.y, r: 7 }));
        g.append(s('path', { class: 'flame', d: `M${d.x - 9} ${d.y - 2} q-7 -12 0 -22 q2 7 6 4 q6 8 0 18 q-4 2 -6 0z` }));
        parts.text = s('text', { class: 'lock-text', x: d.x, y: d.y + 30 }, 'stove');
        g.append(parts.text);
      } else if (d.kind === 'thermostat') {
        g.append(s('circle', { class: 'thermo-ring', cx: d.x, cy: d.y, r: 18 }));
        parts.text = s('text', { class: 'thermo-text', x: d.x, y: d.y + 4 }, '–');
        g.append(parts.text, s('text', { class: 'lock-text', x: d.x, y: d.y + 32 }, 'thermostat'));
      } else if (d.kind === 'lock') {
        const ly = d.y - 16; // sit just above the door line, label beside the padlock
        g.append(s('path', { class: 'lock-shackle', d: `M${d.x - 7} ${ly - 6} v-6 a7 7 0 0 1 14 0 v6` }));
        g.append(s('rect', { class: 'lock-body', x: d.x - 11, y: ly - 6, width: 22, height: 16, rx: 3 }));
        parts.text = s('text', { class: 'lock-text start', x: d.x + 16, y: ly + 6 }, 'locked');
        g.append(parts.text);
      }
      layers.devices.append(g);
      plan.devices.set(d.device_id, { g, parts, d });
    }
    for (const door of snap.doors || []) {
      const rect = s('rect', { class: 'door', x: door.x - 30, y: door.y - 4, width: 60, height: 8, rx: 2 });
      layers.devices.append(rect);
      plan.devices.set(`door:${door.door_id}`, { g: rect, parts: {}, d: door });
    }
    for (const m of micsOf(snap)) {
      layers.mics.append(s('circle', { class: 'mic', cx: m.x, cy: m.y, r: 7 }), s('circle', { class: 'mic-dot', cx: m.x, cy: m.y, r: 2.5 }));
    }
    plan.built = true;
  }

  function micsOf(snap) {
    return (snap && snap.mics) || (state.home && state.home.sensors && state.home.sensors.mics) || [];
  }

  function isNight(time) {
    const [hh, mm] = String(time || '12:00').split(':').map(Number);
    const t = hh * 60 + mm;
    return t >= 19 * 60 + 30 || t < 6 * 60 + 30;
  }

  function updatePlan(snap) {
    if (!snap) return;
    const sameShape = plan.built && snap.rooms.length === plan.rooms.size
      && snap.devices.every((d) => plan.devices.has(d.device_id));
    if (!sameShape) buildPlan(snap);
    state.snapshot = snap;
    planSvg.classList.toggle('night', isNight(snap.time));
    $('#clock-time').textContent = snap.time || '';
    $('#clock-day').textContent = snap.weekday || '';
    const lit = new Set();
    for (const d of snap.devices) {
      const ref = plan.devices.get(d.device_id);
      if (!ref) continue;
      const st = d.state || {};
      if (d.kind === 'light') {
        ref.g.classList.toggle('on', !!st.on);
        if (st.on) lit.add(d.room);
        ref.parts.glow.setAttribute('opacity', st.on ? Math.max(0.15, (st.brightness || 80) / 100 * 0.4) : 0);
      } else if (d.kind === 'tv') {
        ref.g.classList.toggle('on', !!st.on);
        ref.parts.text.textContent = st.on ? `TV: ${st.playing || 'on'}` : 'TV off';
      } else if (d.kind === 'stove') {
        ref.g.classList.toggle('on', !!st.burner_on);
        ref.parts.text.textContent = st.burner_on ? 'burner on' : 'stove';
      } else if (d.kind === 'thermostat') {
        ref.parts.text.textContent = `${fmt(st.setpoint_c, 1)}°`;
      } else if (d.kind === 'lock') {
        ref.g.classList.toggle('unlocked', st.locked === false);
        ref.parts.text.textContent = st.locked === false ? 'unlocked' : 'locked';
      }
    }
    for (const r of snap.rooms) {
      const ref = plan.rooms.get(r.id);
      if (!ref) continue;
      ref.g.classList.toggle('lit', lit.has(r.id));
      const env = r.env || {};
      ref.env.textContent = `${env.lux != null ? Math.round(env.lux) + ' lux' : ''}${env.celsius != null ? ' · ' + fmt(env.celsius, 1) + ' °C' : ''}`;
    }
    for (const door of snap.doors || []) {
      const ref = plan.devices.get(`door:${door.door_id}`);
      if (ref) ref.g.classList.toggle('open', !!door.open);
    }
    const seen = new Set();
    const spots = spreadAvatars([...(snap.people || []), ...(snap.pets || [])]);
    for (const p of snap.people || []) { placeAvatar(p, false, spots.get(p.id)); seen.add(p.id); }
    for (const p of snap.pets || []) { placeAvatar(p, true, spots.get(p.id)); seen.add(p.id); }
    for (const [id, a] of plan.avatars) if (!seen.has(id)) { a.g.remove(); plan.avatars.delete(id); }
    for (const g of plan.layers.dets.querySelectorAll('[data-person]')) {
      const pos = personPos(g.dataset.person);
      if (pos) positionDet(g, pos);
    }
    fillSayWho(snap.people || []);
  }

  /* Avatars that share a spot (or sit so close that their name/activity text would collide) are
     fanned out sideways and every second neighbour gets its labels above the circle instead of below. */
  function spreadAvatars(actors) {
    const placed = [];
    const out = new Map();
    for (const a of [...actors].sort((x, y) => String(x.id).localeCompare(String(y.id)))) {
      let x = a.x, y = a.y, tries = 0;
      const room = (state.snapshot?.rooms || []).find((r) => a.x >= r.x && a.x <= r.x + r.w && a.y >= r.y && a.y <= r.y + r.h);
      while (placed.some((p) => Math.abs(p.x - x) < 40 && Math.abs(p.y - y) < 40) && tries++ < 6) {
        x += 46;
        if (room && x > room.x + room.w - 24) { x = room.x + 24; y += 52; }
      }
      const neighbour = placed.filter((p) => Math.abs(p.x - x) < 150 && Math.abs(p.y - y) < 60).length;
      const spot = { x, y, above: neighbour % 2 === 1 };
      placed.push(spot);
      out.set(a.id, spot);
    }
    return out;
  }

  function positionDet(g, pos) {
    const rect = g.querySelector('rect'), text = g.querySelector('text');
    rect.setAttribute('x', pos.x - 26); rect.setAttribute('y', pos.y - 28);
    text.setAttribute('x', pos.x - 26); text.setAttribute('y', g.dataset.below ? pos.y + 66 : pos.y - 34);
  }

  function placeAvatar(p, isPet, spot) {
    const pos = spot || { x: p.x, y: p.y, above: false };
    let a = plan.avatars.get(p.id);
    if (!a) {
      const g = s('g', { class: `avatar ${isPet ? 'pet' : 'person role-' + (p.role || 'unknown')}`, 'data-id': p.id });
      g.style.transform = `translate(${pos.x}px, ${pos.y}px)`;
      g.append(s('circle', { r: 17 }));
      g.append(s('text', { class: 'initial', y: 5 }, isPet ? '🐈' : (p.name || p.id || '?')[0].toUpperCase()));
      const label = s('text', { class: 'label', y: 34 }, p.name || p.id);
      const act = s('text', { class: 'activity', y: 48 }, '');
      g.append(label, act);
      g.append(s('title', {}, ''));
      plan.layers.avatars.append(g);
      a = { g, act, label, title: g.lastChild };
      plan.avatars.set(p.id, a);
    }
    a.g.style.transform = `translate(${pos.x}px, ${pos.y}px)`;
    a.label.setAttribute('y', pos.above ? -26 : 34);
    a.act.setAttribute('y', pos.above ? -40 : 48);
    a.g.classList.toggle('away', p.home === false);
    a.act.textContent = p.home === false ? 'away' : trunc(p.activity || '', 22);
    a.title.textContent = `${p.name || p.id}${p.role ? ' · ' + p.role : ''}${p.room ? ' · ' + p.room : ''}${p.activity ? ' · ' + p.activity : ''}`;
  }

  function roomCenter(roomId) {
    const ref = plan.rooms.get(roomId);
    return ref ? { x: ref.r.x + ref.r.w / 2, y: ref.r.y + ref.r.h / 2 } : { x: 500, y: 300 };
  }
  function personPos(id) {
    const p = (state.snapshot?.people || []).find((x) => x.id === id);
    return p ? { x: p.x, y: p.y } : null;
  }
  function sourcePos(audio) {
    const src = audio.doa?.source || 'unknown';
    if (src === 'tv') {
      const tv = (state.snapshot?.devices || []).find((d) => d.kind === 'tv' && d.room === audio.room)
        || (state.snapshot?.devices || []).find((d) => d.kind === 'tv');
      if (tv) return { x: tv.x, y: tv.y };
    }
    if (src === 'outside') {
      const door = (state.snapshot?.doors || [])[0];
      if (door) return { x: door.x, y: door.y + 40 };
    }
    return personPos(audio.speaker?.id) || roomCenter(audio.room);
  }

  function fade(el, afterMs) {
    setTimeout(() => {
      el.classList.add('fading');
      setTimeout(() => el.remove(), motionOk() ? 650 : 0);
    }, afterMs);
  }

  function showSpeech(ev) {
    const a = ev.data;
    const src = a.doa?.source || 'unknown';
    const pos = sourcePos(a);
    const mic = micsOf(state.snapshot).find((m) => m.mic_id === a.mic_id) || micsOf(state.snapshot).find((m) => m.room === a.room);
    if (mic) {
      const ray = s('line', { class: `ray src-${src}`, x1: mic.x, y1: mic.y, x2: pos.x, y2: pos.y });
      const ring = s('circle', { class: `ray-source src-${src}`, cx: pos.x, cy: pos.y, r: 26 });
      plan.layers.rays.append(ray, ring);
      fade(ray, 3500); fade(ring, 3500);
    }
    const who = src === 'tv' ? 'TV' : src === 'outside' ? 'Outside the door' : (a.speaker?.name || 'Unknown voice');
    const text = trunc(a.text || '', 44);
    const w = Math.max(90, Math.min(330, 14 + text.length * 7.2));
    let x = pos.x - w / 2, y = pos.y - 86;
    x = Math.max(6, Math.min(1000 - w - 6, x));
    if (y < 6) y = pos.y + 44;
    const g = s('g', { class: `bubble src-${src}` });
    g.style.transform = `translate(${x}px, ${y}px)`;
    g.append(s('rect', { x: 0, y: 0, width: w, height: 46, rx: 10 }));
    g.append(s('text', { class: 'who', x: 10, y: 16 }, who));
    g.append(s('text', { x: 10, y: 36 }, text));
    plan.layers.bubbles.append(g);
    fade(g, 6500);
  }

  function showDetections(ev) {
    const cam = ev.data.camera_id || ev.data.room;
    for (const old of plan.layers.dets.querySelectorAll(`[data-cam="${cam}"]`)) old.remove();
    const dets = ev.data.detections || [];
    dets.forEach((d, i) => {
      const attrs = d.attrs || {};
      const id = attrs.identity?.id;
      const pos = personPos(id) || (() => { const c = roomCenter(ev.data.room); return { x: c.x - 60 + i * 60, y: c.y - 30 }; })();
      const hazard = attrs.near === 'stove' || /chair/.test(attrs.pose || '');
      const g = s('g', { class: `det ${attrs.age_group === 'child' ? 'child' : ''} ${hazard ? 'hazard' : ''}`, 'data-cam': cam, 'data-person': id || null, 'data-below': i % 2 ? '1' : null });
      g.append(s('rect', { x: pos.x - 26, y: pos.y - 28, width: 52, height: 82, rx: 4 }));
      const bits = [d.label, attrs.age_group, attrs.near ? 'near ' + attrs.near : null, attrs.pose].filter(Boolean);
      g.append(s('text', { x: pos.x - 26, y: i % 2 ? pos.y + 66 : pos.y - 34 }, trunc(bits.join(' · '), 30)));
      plan.layers.dets.append(g);
      fade(g, 5500);
    });
  }

  // ------------------------------------------------------------------ pipeline graph
  const pipe = { nodes: new Map(), edges: [], busyTimers: new Map(), dotsLayer: null };

  function buildPipeline(agents) {
    clear(pipeSvg);
    pipe.nodes.clear(); pipe.edges = [];
    const SENSOR = (t) => t.startsWith('sensor') || t === 'sim.clock';
    const producers = new Map(); // topic -> [agent names]
    for (const a of agents) for (const o of a.out) producers.set(o, [...(producers.get(o) || []), a.name]);
    const col = new Map([['sensors', 0]]);
    const depth = (name, seen = new Set()) => {
      if (col.has(name)) return col.get(name);
      if (seen.has(name)) return 1;
      seen.add(name);
      const a = agents.find((x) => x.name === name);
      let d = 1;
      for (const pat of a.in) {
        if (SENSOR(pat) || pat === 'notify.reply') continue;
        for (const [topic, names] of producers) if (topicMatches(pat, topic)) for (const n of names) if (n !== name) d = Math.max(d, depth(n, seen) + 1);
      }
      col.set(name, d);
      return d;
    };
    for (const a of agents) depth(a.name);
    const last = Math.max(...col.values()) + 1;
    col.set('devices', last); col.set('phone', last);
    const nodes = [
      { name: 'sensors', label: 'Sensors', sub: 'camera, mics, lux, presence', sink: true },
      ...agents.map((a) => ({ name: a.name, label: a.label, sub: a.blurb, busy: a.busy })),
      { name: 'devices', label: 'Devices', sub: 'lights, TV, lock, heat', sink: true },
      { name: 'phone', label: 'Phone', sub: 'asks and alerts', sink: true },
    ];
    const W = 1000, H = 420, nw = 140, nh = 46, ncols = last + 1;
    const colX = (c) => 12 + c * ((W - nw - 24) / Math.max(1, ncols - 1));
    const byCol = new Map();
    for (const n of nodes) { const c = col.get(n.name); byCol.set(c, [...(byCol.get(c) || []), n]); }
    for (const [c, list] of byCol) list.forEach((n, i) => {
      n.x = colX(c); n.y = (H / (list.length + 1)) * (i + 1) - nh / 2; n.col = c;
    });
    const edgeLayer = s('g', { id: 'edges' }); const nodeLayer = s('g', { id: 'nodes' }); pipe.dotsLayer = s('g', { id: 'dots' });
    pipeSvg.append(edgeLayer, nodeLayer, pipe.dotsLayer);
    const nodeOf = Object.fromEntries(nodes.map((n) => [n.name, n]));
    const addEdge = (from, to, topic) => {
      if (!nodeOf[from] || !nodeOf[to] || pipe.edges.some((e) => e.from === from && e.to === to && e.topic === topic)) return;
      const a = nodeOf[from], b = nodeOf[to];
      const back = b.col < a.col;
      const x1 = back ? a.x : a.x + nw, y1 = a.y + nh / 2, x2 = back ? b.x + nw : b.x, y2 = b.y + nh / 2;
      const dx = Math.max(40, Math.abs(x2 - x1) / 2) * (back ? -1 : 1);
      const path = s('path', { class: 'edge', d: `M${x1} ${y1} C${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}` });
      path.append(s('title', {}, `${topic}: ${from} → ${to}`));
      edgeLayer.append(path);
      pipe.edges.push({ from, to, topic, path });
    };
    for (const a of agents) {
      for (const pat of a.in) {
        if (SENSOR(pat)) addEdge('sensors', a.name, pat);
        else if (pat === 'notify.reply') addEdge('phone', a.name, pat);
        else for (const [topic, names] of producers) if (topicMatches(pat, topic)) for (const n of names) addEdge(n, a.name, topic);
      }
      for (const o of a.out) {
        if (o === 'action.command') addEdge(a.name, 'devices', o);
        if (o === 'notify.ask') addEdge(a.name, 'phone', o);
      }
    }
    for (const n of nodes) {
      const g = s('g', { class: `node ${n.sink ? 'sink' : ''} ${n.busy ? 'busy' : ''}`, 'data-node': n.name });
      g.append(s('rect', { x: n.x, y: n.y, width: nw, height: nh, rx: 10 }));
      g.append(s('text', { x: n.x + nw / 2, y: n.y + 20 }, n.label));
      g.append(s('text', { class: 'sub', x: n.x + nw / 2, y: n.y + 36 }, trunc(n.sub || '', 24)));
      g.append(s('title', {}, n.sub || n.label));
      nodeLayer.append(g);
      pipe.nodes.set(n.name, g);
    }
  }

  function sourceNode(ev) {
    const src = ev.source || '';
    if (src.startsWith('agent.')) return src.slice(6);
    if (src.startsWith('sim')) return ev.topic === 'action.applied' ? 'devices' : 'sensors';
    if (src === 'phone' || (src === 'web' && ev.topic === 'notify.reply')) return 'phone';
    return null;
  }

  function animatePipeline(ev) {
    if (!pipe.edges.length) return;
    const from = sourceNode(ev);
    const grp = groupOf(ev.topic);
    const hits = pipe.edges.filter((e) => topicMatches(e.topic, ev.topic) && (!from || e.from === from));
    for (const e of hits) {
      flashEdge(e, grp);
      setBusy(e.to, true);
    }
    if (from && pipe.nodes.has(from)) {
      setBusy(from, false);
      const g = pipe.nodes.get(from);
      g.classList.add('fired');
      setTimeout(() => g.classList.remove('fired'), 500);
    }
  }
  function setBusy(name, on) {
    const g = pipe.nodes.get(name);
    if (!g || name === 'sensors' || name === 'devices' || name === 'phone') return;
    clearTimeout(pipe.busyTimers.get(name));
    g.classList.toggle('busy', on);
    if (on) pipe.busyTimers.set(name, setTimeout(() => g.classList.remove('busy'), 6000));
  }
  function flashEdge(e, grp) {
    e.path.classList.add('lit');
    setTimeout(() => e.path.classList.remove('lit'), 700);
    if (!motionOk()) return;
    const dot = s('circle', { class: `dot grp-${grp}`, r: 5.5 });
    pipe.dotsLayer.append(dot);
    const len = e.path.getTotalLength();
    const t0 = performance.now(), dur = 650;
    const tick = (now) => {
      const t = Math.min(1, (now - t0) / dur);
      const p = e.path.getPointAtLength(len * t);
      dot.setAttribute('cx', p.x); dot.setAttribute('cy', p.y);
      if (t < 1 && motionOk()) requestAnimationFrame(tick); else dot.remove();
    };
    requestAnimationFrame(tick);
  }

  // ------------------------------------------------------------------ decider card
  function onProposal(ev) {
    const d = ev.data;
    state.proposals.set(ev.id, ev);
    if (!labels.isModel(d)) return;
    const twoB = d.escalated_from ? state.proposals.get(d.escalated_from) : null;
    const prevMain = state.dec.main;
    // an escalation keeps the first (shadow) answer beside the final one; otherwise the newest decision stands alone
    const shadow = twoB && labels.isModel(twoB.data) ? twoB
      : (d.escalated_from && prevMain && prevMain.data.mode === 'shadow' ? prevMain : null);
    state.dec = { main: ev, shadow };
    renderDecider();
  }

  function renderDecider() {
    const box = clear($('#decider'));
    const main = state.dec.main;
    const chip = $('#dec-model');
    if (!main) { chip.textContent = 'waiting for a question'; chip.className = 'chip model'; return; }
    const d = main.data, st = state.settings || {};
    const route = labels.route(d), short = labels.short(d);
    chip.textContent = `${labels.model(d)} · ${fmt(d.latency_ms / 1000, 1)} s`;
    chip.className = `chip model route-${route}`;
    box.append(h('p', { class: 'dec-q', text: d.question || (d.kind === 'scene' ? 'What should the house do?' : 'What does the speaker want?') }));
    box.append(h('pre', { class: 'dec-state', text: d.state_text || (d.request && d.request.state) || '(no state text)' }));
    const bars = h('div', { class: 'bars' });
    const probs = Object.entries(d.probabilities || {}).sort((a, b) => b[1] - a[1]);
    const act = d.threshold ?? st.act_threshold;
    for (const [action, p] of probs) {
      const row = h('div', { class: `bar-row ${action === d.action ? 'pick' : ''}` });
      row.append(h('span', { class: 'bar-label', text: words(action) }));
      const track = h('div', { class: 'bar-track' });
      track.append(h('div', { class: 'bar-fill', style: `width:${Math.round(p * 100)}%` }));
      if (st.ask_threshold != null) track.append(h('i', { class: 'th ask', style: `left:${st.ask_threshold * 100}%`, title: `ask at ${fmt(st.ask_threshold)}` }));
      if (act != null) track.append(h('i', { class: 'th act', style: `left:${act * 100}%`, title: `act at ${fmt(act)}` }));
      row.append(track, h('span', { class: 'bar-val', text: fmt(p) }));
      bars.append(row);
    }
    box.append(bars);
    box.append(h('div', { class: 'th-legend' },
      h('span', { class: 'act', text: `act at ${fmt(act)}` }),
      h('span', { class: 'ask', text: `ask at ${fmt(st.ask_threshold)}` })));
    const band = d.confidence >= (act ?? 0.8) ? 'act' : d.confidence >= (st.ask_threshold ?? 0.45) ? 'ask' : 'ignore';
    const tokens = d.tokens ? `${(d.tokens.input || 0) + (d.tokens.output || 0)} tokens` : (route === 'local' || route === 'rule' ? '0 tokens' : null);
    box.append(h('div', { class: 'dec-meta' },
      h('span', {}, 'answer ', h('b', { text: words(d.action) })),
      h('span', {}, 'confidence ', h('b', { text: fmt(d.confidence) }), ` → ${band}`),
      h('span', {}, 'latency ', h('b', { text: `${Math.round(d.latency_ms)} ms` })),
      h('span', {}, 'where ', h('b', { text: route === 'local' ? 'on the hub' : route === 'bedrock' ? 'Amazon Bedrock' : route })),
      h('span', {}, 'model ', h('b', { text: labels.model(d) })),
      tokens ? h('span', {}, h('b', { text: tokens })) : null,
      d.speaker?.name ? h('span', {}, 'speaker ', h('b', { text: `${d.speaker.name} (${d.speaker.role}, match ${fmt(d.speaker.match)})` })) : null,
      d.reason ? h('span', { class: 'muted', text: d.reason }) : null));
    if (d.tool_calls && d.tool_calls.length) {
      box.append(h('div', { class: 'dec-tools' }, 'tool calls: ', d.tool_calls.map((c) => h('code', { text: `${c.name}(${Object.entries(c.input || {}).map(([k, v]) => `${k}=${esc(v)}`).join(', ')})` }))));
    }
    if (d.reply) box.append(h('p', { class: 'dec-reply', text: `“${d.reply}”` }));
    const escalation = labels.escalation(st);
    if (st.decider_mode === 'cascade' || state.dec.shadow || d.escalated_from) {
      const sh = state.dec.shadow?.data;
      const unsure = d.confidence < (act ?? 0.8);
      const half = (title, pd, cls, placeholder) => h('div', { class: `half ${cls}` },
        h('h3', { text: title }),
        pd ? h('div', { class: 'big', text: words(pd.action) }) : h('div', { class: 'muted', text: placeholder }),
        pd ? h('div', { class: 'muted', text: `confidence ${fmt(pd.confidence)} · ${Math.round(pd.latency_ms)} ms${pd.tokens ? ' · ' + ((pd.tokens.input || 0) + (pd.tokens.output || 0)) + ' tokens' : ''}` }) : null);
      const wrap = h('div', { class: 'cascade' });
      if (d.escalated_from || sh) {
        wrap.append(half(`${sh ? labels.short(sh) : '2B'} shadow answer`, sh, 'shadow', 'not asked'), half(`${short} answer`, d, 'main'));
      } else if (d.mode === 'shadow') {
        wrap.append(half(`${short} answer`, d, 'main'), half(escalation, null, 'shadow', `asking ${escalation}…`));
      } else {
        wrap.append(half(`${short} answer`, d, 'main'), half(escalation, null, 'shadow', unsure && st.decider_mode === 'cascade' ? `asking ${escalation}…` : `not needed: ${short} was sure`));
      }
      box.append(wrap);
    }
  }

  // ------------------------------------------------------------------ keyword vs decider table
  function rootUtterance(ev) {
    let cur = ev, hops = 0;
    while (cur && hops++ < 12) {
      if (cur.topic === 'voice.utterance') return cur;
      cur = cur.cause ? state.byId.get(cur.cause) : null;
    }
    return null;
  }
  function rowFor(utt) {
    let row = state.utteranceRows.get(utt.id);
    if (row) return row;
    const tr = h('tr', { 'data-utt': utt.id });
    const cells = {};
    for (const k of ['heard', 'kw', 'dec', 'pol', 'out']) { cells[k] = h('td', { class: k }); tr.append(cells[k]); }
    const u = utt.data;
    const who = u.source === 'tv' ? 'TV voice' : u.source === 'outside' ? 'voice from outside' : `${u.speaker?.name || 'unknown voice'}${u.speaker?.role ? ' (' + u.speaker.role + ')' : ''}`;
    cells.heard.append(h('q', { text: u.text || '' }), h('span', { class: 'sub', text: `${who} · ${u.room || ''}` }));
    cells.kw.append(h('span', { class: 'pending', text: '…' }));
    cells.dec.append(h('span', { class: 'pending', text: 'thinking…' }));
    cells.pol.append(h('span', { class: 'pending', text: '…' }));
    cells.out.append(h('span', { class: 'pending', text: '…' }));
    row = { tr, cells, keyword: null, decider: null, verdict: null, outcome: null };
    state.utteranceRows.set(utt.id, row);
    $('#compare-body').prepend(tr);
    $('#compare-empty').hidden = true;
    return row;
  }
  function updateRow(ev) {
    const utt = rootUtterance(ev);
    if (!utt) return;
    const row = rowFor(utt);
    const d = ev.data;
    if (ev.topic === 'intent.proposal') {
      if (d.by === 'keyword') {
        row.keyword = d.action;
        clear(row.cells.kw).append(h('span', { class: 'action', text: words(d.action) }), h('span', { class: 'sub', text: d.reason || '' }));
      } else {
        row.decider = d;
        const prev = d.escalated_from ? state.proposals.get(d.escalated_from) : null;
        const after = d.escalated_from ? ` · after ${prev ? labels.short(prev.data) : 'the 2B'} was unsure` : '';
        const tokens = d.tokens ? ` · ${(d.tokens.input || 0) + (d.tokens.output || 0)} tokens` : '';
        clear(row.cells.dec).append(h('span', { class: 'action', text: words(d.action) }),
          h('span', { class: 'sub', text: `${labels.short(d)} · ${fmt(d.confidence)}${after}${tokens}` }));
      }
    } else if (ev.topic === 'policy.verdict') {
      row.verdict = d;
      clear(row.cells.pol).append(h('span', { class: `v-${d.verdict}`, text: words(d.verdict) }), h('span', { class: 'sub', text: d.rule || '' }));
      if (d.verdict === 'deny') setOutcome(row, 'nothing happened (denied)');
      if (d.verdict === 'ignore') setOutcome(row, 'nothing happened');
    } else if (ev.topic === 'action.applied') {
      setOutcome(row, `${d.device_id}: ${describeState(d.state)}`);
    } else if (ev.topic === 'action.command' && d.command === 'notify') {
      setOutcome(row, `alert sent to ${d.args?.to || 'parents'}`);
    } else if (ev.topic === 'notify.ask') {
      setOutcome(row, `asked ${d.to}: "${trunc(d.text, 60)}"`);
    } else if (ev.topic === 'notify.reply') {
      setOutcome(row, `${d.by === 'web' ? 'parent' : d.by} answered ${d.answer}` + (d.answer === 'Yes' ? '' : ' → nothing happened'));
    }
    judgeRow(row);
  }
  function setOutcome(row, text) {
    row.outcome = text;
    clear(row.cells.out).append(document.createTextNode(text));
  }
  function describeState(stt) {
    if (!stt) return '';
    if ('on' in stt) return stt.on ? `on${stt.playing ? ' (' + stt.playing + ')' : stt.brightness ? ' ' + stt.brightness + '%' : ''}` : 'off';
    if ('locked' in stt) return stt.locked ? 'locked' : 'unlocked';
    if ('setpoint_c' in stt) return `set to ${fmt(stt.setpoint_c, 1)} °C`;
    if ('burner_on' in stt) return stt.burner_on ? 'burner on' : 'burner off';
    return JSON.stringify(stt);
  }
  function judgeRow(row) {
    if (row.keyword == null || !row.verdict) return;
    const v = row.verdict.verdict;
    const kw = row.keyword;
    let wrong, why;
    if (v === 'allow') {
      wrong = kw !== row.verdict.action;
      why = !wrong ? 'same result' : kw === 'none' ? 'keyword heard no command: nothing would have happened' : `keyword would have done "${words(kw)}" instead`;
    } else if (v === 'ask_parent' || v === 'ask_speaker') {
      wrong = kw !== 'none';
      why = wrong ? `keyword would have done "${words(kw)}" without asking` : 'same result';
    } else {
      wrong = kw !== 'none';
      why = wrong ? `keyword would have obeyed: "${words(kw)}"` : 'same result';
    }
    row.tr.classList.toggle('kw-wrong', wrong);
    row.tr.classList.toggle('kw-ok', !wrong);
    const sub = row.cells.kw.querySelector('.sub');
    if (sub) sub.textContent = `${wrong ? '✗' : '✓'} ${why}`;
  }

  // ------------------------------------------------------------------ phone mock
  const phoneCards = new Map();
  function phoneAsk(ev) {
    const d = ev.data;
    const list = $('#phone-list');
    list.querySelector('.muted')?.remove();
    const card = h('div', { class: 'notif ask', 'data-rid': d.request_id },
      h('div', { class: 'from', text: `House → ${d.to}` }),
      h('div', { text: d.text }),
      h('div', { class: 'actions' }, (d.options || ['Yes', 'No']).map((opt) => h('button', {
        class: `btn tiny ${opt === 'Yes' ? 'primary' : ''}`, text: opt,
        onclick: () => api('api/phone/reply', { request_id: d.request_id, answer: opt }).catch(() => {}),
      }))));
    phoneCards.set(d.request_id, card);
    list.prepend(card);
    trimPhone(list);
  }
  function phoneReply(ev) {
    const card = phoneCards.get(ev.data.request_id);
    if (!card) return;
    card.classList.add('answered');
    card.append(h('div', { class: 'answer', text: `${ev.data.by === 'web' ? 'You' : ev.data.by} answered: ${ev.data.answer}` }));
  }
  function phoneAlert(ev) {
    const d = ev.data;
    const list = $('#phone-list');
    list.querySelector('.muted')?.remove();
    list.prepend(h('div', { class: 'notif alert' }, h('div', { class: 'from', text: `Alert → ${d.args?.to || 'parents'}` }), h('div', { text: d.args?.text || '' })));
    trimPhone(list);
  }
  function trimPhone(list) { while (list.children.length > 8) list.lastChild.remove(); }

  // ------------------------------------------------------------------ timeline
  function summarize(ev) {
    const d = ev.data || {};
    switch (ev.topic) {
      case 'sensor.audio': return `"${d.text}" — ${d.speaker?.name || d.doa?.source || '?'} (${d.room})`;
      case 'sensor.camera': return `${d.room}: ${(d.detections || []).length} detection(s)${d.activity ? ' · ' + d.activity : ''}`;
      case 'sensor.motion': return `${d.room}: motion ${d.motion ? 'yes' : 'no'}`;
      case 'sensor.lux': return `${d.room}: ${Math.round(d.lux)} lux`;
      case 'sensor.temp': return `${d.room}: ${fmt(d.celsius, 1)} °C`;
      case 'sensor.device': return `${d.device_id}: ${describeState(d.state)}`;
      case 'sensor.door': return `${d.door_id}: ${d.open ? 'open' : 'closed'}, ${d.locked ? 'locked' : 'unlocked'}`;
      case 'sensor.phone': return `${d.person_id} is ${d.home ? 'home' : 'away'}`;
      case 'sim.clock': return `${d.weekday} ${d.time}${d.school_night ? ' (school night)' : ''}`;
      case 'perception.scene': return `${d.room} · ${d.trigger}: ${d.text}`;
      case 'voice.utterance': return `${d.source === 'person' ? (d.speaker?.name || 'unknown') : d.source}: "${d.text}"`;
      case 'intent.proposal': return `${d.by}: ${words(d.action)} (${fmt(d.confidence)})${d.mode === 'shadow' ? ' · shadow' : ''}${d.escalated_from ? ' · escalated' : ''}`;
      case 'policy.verdict': return `${d.verdict}: ${words(d.action)} — ${d.rule}`;
      case 'action.command': return `${d.device_id} ${d.command} ${Object.keys(d.args || {}).length ? JSON.stringify(d.args) : ''}`;
      case 'action.applied': return `${d.device_id}: ${describeState(d.state)}`;
      case 'notify.ask': return `ask ${d.to}: ${d.text}`;
      case 'notify.reply': return `${d.by}: ${d.answer}`;
      case 'trace.log': return `${d.agent}: ${d.text}`;
      case 'scenario.status': return `${d.state}: ${d.text || ''}`;
      case 'ui.snapshot': return 'house snapshot';
      case 'ui.settings': return 'settings updated';
      default: return JSON.stringify(d);
    }
  }
  function addTimeline(ev) {
    const li = h('li', { class: `grp-${groupOf(ev.topic)}`, 'data-id': ev.id, tabindex: 0 },
      h('time', { text: `+${fmt(ev.ts, 1)}s` }),
      h('span', { class: 'topic', text: ev.topic }),
      h('span', { class: 'sum' }, h('span', { class: 'src', text: `${ev.source} ` }), summarize(ev)));
    const show = () => showDetail(ev, li);
    li.addEventListener('click', show);
    li.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); show(); } });
    const tl = $('#timeline');
    tl.prepend(li);
    while (tl.children.length > 400) tl.lastChild.remove();
    $('#tl-count').textContent = `${state.count} events`;
  }
  function showDetail(ev, li) {
    for (const x of $('#timeline').querySelectorAll('.selected')) x.classList.remove('selected');
    li.classList.add('selected');
    const chain = [];
    let cur = ev, hops = 0;
    while (cur && hops++ < 20) { chain.unshift(cur); cur = cur.cause ? state.byId.get(cur.cause) : null; }
    const box = clear($('#detail'));
    box.append(h('div', {}, h('b', { text: ev.topic }), ` from ${ev.source} at +${fmt(ev.ts, 1)}s`));
    box.append(h('div', { class: 'small muted', text: 'Cause chain (oldest first):' }));
    box.append(h('ol', { class: 'chain' }, chain.map((c) => h('li', { class: c.id === ev.id ? 'me' : '' },
      h('span', { class: 'topic', text: c.topic }), ` · ${trunc(summarize(c), 90)}`))));
    if (ev.cause && !state.byId.has(ev.cause)) box.append(h('div', { class: 'small muted', text: `(cause ${ev.cause} happened before this page connected)` }));
    box.append(h('pre', { text: JSON.stringify(ev, null, 2) }));
  }

  // ------------------------------------------------------------------ scenarios + narration
  function renderScenarios() {
    const list = clear($('#scenario-list'));
    const sorted = [...state.scenarios].sort((a, b) => (b.tags || []).includes('hero') - (a.tags || []).includes('hero'));
    if (!state.selectedScenario && sorted[0]) state.selectedScenario = sorted[0].id;
    for (const sc of sorted) {
      const btn = h('button', { class: 'scenario', role: 'option', 'aria-selected': String(sc.id === state.selectedScenario), 'data-id': sc.id,
        onclick: () => { state.selectedScenario = sc.id; renderScenarios(); } },
        h('span', { class: 't', text: sc.title }),
        h('span', { class: 'b', text: sc.blurb || '' }),
        h('span', { class: 'tags' }, (sc.tags || []).map((t) => h('span', { class: `tag ${t}`, text: t })), h('span', { class: 'tag', text: `${sc.steps} steps` })));
      list.append(btn);
    }
  }
  function onScenarioStatus(ev) {
    const d = ev.data;
    const chip = $('#scn-state');
    const checks = $('#checks');
    if (d.state === 'started') {
      state.running = d.scenario_id;
      clear(checks);
      chip.textContent = 'running'; chip.className = 'chip state running';
      $('#narration').textContent = d.text ? `Running: ${d.text}` : 'Running…';
    } else if (d.state === 'step') {
      if (d.text) $('#narration').textContent = d.text;
    } else if (d.state === 'expect_pass' || d.state === 'expect_fail') {
      checks.append(h('li', { class: d.state === 'expect_fail' ? 'fail' : '', text: d.text || '' }));
    } else if (d.state === 'finished' || d.state === 'stopped' || d.state === 'error') {
      state.running = null;
      const failed = (d.failed || 0) > 0 || d.state === 'error';
      chip.textContent = d.state === 'finished' ? `${d.passed || 0} passed, ${d.failed || 0} failed` : d.state;
      chip.className = `chip state ${d.state === 'finished' ? (failed ? 'failed' : 'finished') : d.state}`;
      if (d.state === 'error') $('#narration').textContent = `Scenario error: ${d.text}`;
    }
  }

  // ------------------------------------------------------------------ settings
  function applySettings(stg) {
    if (!stg) return;
    state.settings = stg;
    $('#set-mode').value = stg.decider_mode;
    const esc = $('#set-escalate');
    if (esc) { esc.value = stg.escalate_to || 'bedrock'; esc.disabled = stg.decider_mode !== 'cascade'; }
    $('#set-act').value = stg.act_threshold; $('#set-act-out').textContent = fmt(stg.act_threshold);
    $('#set-ask').value = stg.ask_threshold; $('#set-ask-out').textContent = fmt(stg.ask_threshold);
    $('#set-speed').value = stg.speed; $('#set-speed-out').textContent = `${stg.speed}×`;
    $('#set-shadow').checked = !!stg.keyword_shadow;
    if (state.dec.main) renderDecider();
    feed.emit('settings', stg);
  }
  let settingsTimer = null;
  function pushSettings(changes) {
    clearTimeout(settingsTimer);
    settingsTimer = setTimeout(() => api('api/settings', changes).then(applySettings).catch(() => {}), 200);
  }
  function wireSettings() {
    $('#set-mode').addEventListener('change', (e) => pushSettings({ decider_mode: e.target.value }));
    $('#set-escalate')?.addEventListener('change', (e) => pushSettings({ escalate_to: e.target.value }));
    $('#set-shadow').addEventListener('change', (e) => pushSettings({ keyword_shadow: e.target.checked }));
    $('#set-speed').addEventListener('input', (e) => { $('#set-speed-out').textContent = `${e.target.value}×`; pushSettings({ speed: Number(e.target.value) }); });
    $('#set-act').addEventListener('input', (e) => {
      const act = Number(e.target.value);
      $('#set-act-out').textContent = fmt(act);
      const ask = Math.min(Number($('#set-ask').value), act);
      $('#set-ask').value = ask; $('#set-ask-out').textContent = fmt(ask);
      pushSettings({ act_threshold: act, ask_threshold: ask });
    });
    $('#set-ask').addEventListener('input', (e) => {
      const ask = Math.min(Number(e.target.value), Number($('#set-act').value));
      e.target.value = ask; $('#set-ask-out').textContent = fmt(ask);
      pushSettings({ ask_threshold: ask });
    });
    $('#settings-toggle').addEventListener('click', (e) => {
      const panel = $('#settings');
      panel.hidden = !panel.hidden;
      e.currentTarget.setAttribute('aria-expanded', String(!panel.hidden));
    });
    $('#pause').addEventListener('click', (e) => {
      state.paused = !state.paused;
      document.body.classList.toggle('paused', state.paused);
      e.currentTarget.setAttribute('aria-pressed', String(state.paused));
      e.currentTarget.textContent = state.paused ? 'Resume animations' : 'Pause animations';
      feed.emit('motion', { paused: state.paused });
    });
    feed.emit('motion', { paused: false });
  }

  // ------------------------------------------------------------------ [3D | 2D] view toggle
  // 3D is the default once static/3d/main.js reports that WebGL2 works; until then (and whenever it
  // cannot) the 2D plan is shown, so a browser without WebGL2 or import maps still gets the house.
  function setView(v, remember = true) {
    if (v === '3d' && !state.scene3d) v = '2d';
    state.view = v;
    $('#house3d').hidden = v !== '3d';
    $('#house3d-bar').hidden = v !== '3d';
    $('#plan-wrap').hidden = v === '3d';
    $('#view-3d').setAttribute('aria-checked', String(v === '3d'));
    $('#view-2d').setAttribute('aria-checked', String(v === '2d'));
    if (remember) { try { localStorage.setItem('houseView', v); } catch { /* private mode */ } }
    feed.emit('view', v);
  }
  function wireViewToggle() {
    $('#view-3d').addEventListener('click', () => setView('3d'));
    $('#view-2d').addEventListener('click', () => setView('2d'));
    feed.on('scene3d', (info) => {
      state.scene3d = !!(info && info.available);
      const btn = $('#view-3d');
      btn.disabled = !state.scene3d;
      const note = $('#house3d-note');
      if (!state.scene3d) {
        btn.title = `3D view unavailable: ${info?.reason || 'WebGL2 missing'}`;
        note.textContent = `3D view unavailable (${info?.reason || 'WebGL2 missing'}); showing the 2D plan.`;
        note.hidden = false;
        setView('2d', false);
        return;
      }
      note.hidden = true;
      let pref = '3d';
      try { pref = localStorage.getItem('houseView') || '3d'; } catch { /* private mode */ }
      setView(pref === '2d' ? '2d' : '3d', false);
    });
    setView('2d', false);
  }

  // ------------------------------------------------------------------ controls
  function fillSayWho(people) {
    const sel = $('#say-who');
    const current = sel.value;
    const home = people.filter((p) => p.home !== false);
    if (home.length === sel.options.length && home.every((p, i) => sel.options[i].value === p.id)) return;
    clear(sel);
    for (const p of home) sel.append(h('option', { value: p.id, text: `${p.name} (${p.role})` }));
    if ([...sel.options].some((o) => o.value === current)) sel.value = current;
  }
  function wireControls() {
    $('#run').addEventListener('click', () => { if (state.selectedScenario) api(`api/scenario/${encodeURIComponent(state.selectedScenario)}/run`, {}).catch(() => {}); });
    $('#stop').addEventListener('click', () => api('api/scenario/stop', {}).catch(() => {}));
    $('#reset').addEventListener('click', async () => {
      try { await api('api/reset', {}); } catch { return; }
      state.dec = { main: null, shadow: null }; renderDecider();
      state.utteranceRows.clear(); clear($('#compare-body')); $('#compare-empty').hidden = false;
      phoneCards.clear(); clear($('#phone-list')).append(h('p', { class: 'muted', text: 'No notifications yet.' }));
      clear($('#checks')); $('#narration').textContent = 'House reset. Pick a scenario and press Run, or say something.';
      const chip = $('#scn-state'); chip.textContent = 'idle'; chip.className = 'chip state idle';
      for (const l of ['rays', 'dets', 'bubbles']) clear(plan.layers[l] || h('div'));
      feed.emit('reset', {});
    });
    const say = (kind) => {
      const text = $('#say-text').value.trim();
      if (!text) { $('#say-text').focus(); return; }
      const step = kind === 'say' ? { do: 'say', who: $('#say-who').value, text }
        : kind === 'tv' ? { do: 'tv_say', text, device: (state.snapshot?.devices || []).find((d) => d.kind === 'tv')?.device_id || 'tv_living' }
        : { do: 'outside_say', text, door: (state.snapshot?.doors || [])[0]?.door_id || 'front' };
      api('api/step', step).then(() => { $('#say-text').value = ''; }).catch(() => {});
    };
    $('#say-person').addEventListener('click', () => say('say'));
    $('#say-tv').addEventListener('click', () => say('tv'));
    $('#say-outside').addEventListener('click', () => say('outside'));
    $('#say-text').addEventListener('keydown', (e) => { if (e.key === 'Enter') say('say'); });
    $('#tl-filter').addEventListener('change', (e) => { $('#timeline').dataset.filter = e.target.value; });
  }

  // ------------------------------------------------------------------ event dispatch
  function handle(ev) {
    if (!ev || !ev.topic) return;
    if (ev.topic === 'ui.snapshot') { feed.emit('snapshot', ev.data.snapshot); return; } // not in the timeline: too chatty
    state.byId.set(ev.id, ev); state.count += 1;
    switch (ev.topic) {
      case 'ui.settings': applySettings(ev.data.settings); break;
      case 'sim.clock': $('#clock-time').textContent = ev.data.time; $('#clock-day').textContent = ev.data.weekday; break;
      case 'intent.proposal': onProposal(ev); updateRow(ev); break;
      case 'policy.verdict': case 'action.applied': updateRow(ev); break;
      case 'action.command': if (ev.data.command === 'notify') phoneAlert(ev); updateRow(ev); break;
      case 'notify.ask': phoneAsk(ev); updateRow(ev); break;
      case 'notify.reply': phoneReply(ev); updateRow(ev); break;
      case 'scenario.status': onScenarioStatus(ev); break;
      default: break;
    }
    if (state.byId.size > 3000) { const k = state.byId.keys(); for (let i = 0; i < 500; i++) state.byId.delete(k.next().value); }
    feed.emit('event', ev);
    animatePipeline(ev);
    addTimeline(ev);
  }

  // the 2D plan listens to the same feed as the 3D house
  feed.on('snapshot', updatePlan);
  feed.on('event', (ev) => {
    if (ev.topic === 'sensor.audio') showSpeech(ev);
    else if (ev.topic === 'sensor.camera') showDetections(ev);
  });

  async function loadState() {
    const st = await api('api/state');
    state.home = st.home; state.agents = st.agents; state.scenarios = st.scenarios;
    if (st.home?.name) { $('#home-name').textContent = st.home.name; document.title = `${st.home.name} · LocoCisco`; }
    feed.emit('state', st);
    applySettings(st.settings);
    buildPipeline(st.agents);
    renderScenarios();
    const rules = clear($('#house-rules'));
    for (const r of st.house_rules || []) rules.append(h('li', { text: r.text }));
    feed.emit('snapshot', st.snapshot);
  }

  function connect() {
    const es = new EventSource('api/stream');
    const conn = $('#conn');
    let dropped = false;
    es.onopen = () => {
      conn.textContent = '● live'; conn.className = 'conn on';
      if (dropped) { dropped = false; loadState().catch(() => {}); } // resync after a reconnect
    };
    es.onerror = () => { dropped = true; conn.textContent = '● reconnecting'; conn.className = 'conn off'; };
    es.onmessage = (m) => { try { handle(JSON.parse(m.data)); } catch (err) { console.error(err); } };
  }

  wireSettings();
  wireControls();
  wireViewToggle();
  loadState().catch(() => {});
  connect();
})();
