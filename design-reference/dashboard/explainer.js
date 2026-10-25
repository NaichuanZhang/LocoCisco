// LocoCisco — How it works page (explainer.html). Reference design for recording demo videos.
// Plays the dashboard's sample story (story.js + explainer-story.js) on one clock and shows what
// the local models see, hear and decide. Every view is a function of the clock position, so
// pausing, restarting, jumping to a beat or opening ?view=decisions&t=39 always shows the same state.
(function () {
  'use strict';
  const S = window.STORY, X = window.EXPLAIN;
  const $ = (sel) => document.querySelector(sel);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const mmss = (s) => { s = Math.max(0, Math.floor(s)); return String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0'); };
  const mmssd = (s) => mmss(s) + '.' + Math.floor((Math.max(0, s) % 1) * 10);
  const smooth = (a, b, x) => { const u = clamp((x - a) / (b - a), 0, 1); return u * u * (3 - 2 * u); };
  // Seeded noise: the same moment always draws the same jitter, scores and waveform.
  const hash = (i, s) => { let h = Math.imul((i | 0) ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul((s | 0) + 0x632be5ab, 0xc2b2ae35); h ^= h >>> 15; h = Math.imul(h, 0x27d4eb2f); h ^= h >>> 13; return (h >>> 0) / 4294967296; };
  const noise = (x, s) => { const i = Math.floor(x), f = x - i, u = f * f * (3 - 2 * f); return hash(i, s) * (1 - u) + hash(i + 1, s) * u; };
  const seedOf = (str) => { let h = 7; for (const ch of String(str)) h = (Math.imul(h, 31) + ch.charCodeAt(0)) | 0; return h; };
  const gauss = (s, i) => { const u = Math.max(1e-6, hash(i * 2 + 1, s)), v = hash(i * 2 + 2, s); return clamp(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v), -2.2, 2.2); };

  // ---- settings ----
  const CFG = {
    renderEvery: 50,       // ms between DOM refreshes (canvases draw every frame)
    countdown: 3,          // s
    speeds: [1, 2, 4],
    frameRate: 25,         // fixture frame rate shown in the Vision view
    waveWindow: 12,        // s of audio on screen
    sparkWindow: 10,       // s of speaking probability per face
    showSampleChip: true,  // the build spec: scripted output must never pass as live inference
  };
  const VIEWS = [
    { id: 'vision', key: 'v', label: 'Vision', icon: 'face', title: 'Who is in the room, and who is speaking',
      desc: 'YuNet finds each face, Light-ASD scores who is talking, and seats plus check-in turn faces into names.' },
    { id: 'transcription', key: 't', label: 'Transcription', icon: 'wave', title: 'Speech to text, a moment behind the room',
      desc: 'Silero VAD finds speech, Parakeet streams partial text and fixes it, and CAM++ tells the voices apart.' },
    { id: 'decisions', key: 'd', label: 'Decisions', icon: 'route', title: 'What to do with what was said',
      desc: 'Rules run first. The 12B Decider proposes an action; Qwen 35B is asked only when the Decider is unsure.' },
    { id: 'pipeline', key: 'p', label: 'Pipeline', icon: 'chip', title: 'Everything runs on one machine in the room',
      desc: 'One media clock drives every model on the GB10. Only an approved Discord post ever leaves the room.' },
  ];

  // ---- one outline icon family (24 px grid, 1.8 stroke) — same paths as the dashboard ----
  const P = {
    play: 'M8 5.8v12.4a.8.8 0 0 0 1.2.7l10-6.2a.8.8 0 0 0 0-1.4l-10-6.2A.8.8 0 0 0 8 5.8z',
    pause: 'M9 5v14M15 5v14',
    restart: 'M3.5 12a8.5 8.5 0 1 0 2.5-6M3.5 4v4.5H8',
    film: 'M3 6.5A1.5 1.5 0 0 1 4.5 5h15A1.5 1.5 0 0 1 21 6.5v11a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5zM7.5 5v14M16.5 5v14M3 9.5h4.5M3 14.5h4.5M16.5 9.5H21M16.5 14.5H21',
    check: 'M5 12.5l4.5 4.5L19 7.5',
    checkCircle: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM8 12.3l2.7 2.7L16 9.7',
    lock: 'M7 11V8a5 5 0 0 1 10 0v3M6 11h12a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1v-8a1 1 0 0 1 1-1z',
    lamp: 'M9 18h6M10 21h4M12 3a6 6 0 0 0-3.6 10.8c.7.5 1.1 1.3 1.1 2.1v.1h5v-.1c0-.8.4-1.6 1.1-2.1A6 6 0 0 0 12 3z',
    fan: 'M10.4 12a1.6 1.6 0 1 0 3.2 0 1.6 1.6 0 1 0-3.2 0M12 10.4c-.6-3.3.2-6.4 2.6-6.4 2.6 0 2.7 3.6-1.2 6.9M13.4 12.8c3.1 1.1 5.4 3.4 4.2 5.5-1.3 2.2-4.4.4-5.4-4.6M10.6 12.8c-2.5 2.2-5.6 3-6.8.9-1.3-2.2 1.8-4 6.7-2.6',
    chat: 'M5 5h14a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-7l-4.5 3.5V17H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z',
    send: 'M5 12h13M12 5l7 7-7 7',
    eyeOff: 'M3 3l18 18M10.6 6.1A9.6 9.6 0 0 1 12 6c5 0 8.5 4.5 9.5 6-.5.8-1.5 2.1-2.9 3.3M6.6 7.7C4.6 9 3.2 10.9 2.5 12c1 1.5 4.5 6 9.5 6 1.6 0 3-.4 4.3-1.1M9.9 9.9a3 3 0 0 0 4.2 4.2',
    face: 'M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2M9 10h.01M15 10h.01M9.5 15a3.5 3.5 0 0 0 5 0',
    wave: 'M4 10v4M8 7v10M12 4v16M16 8v8M20 11v2',
    chip: 'M7 7h10v10H7zM9.5 3v4M14.5 3v4M9.5 17v4M14.5 17v4M3 9.5h4M3 14.5h4M17 9.5h4M17 14.5h4',
    shield: 'M12 3l7 3v5.5c0 4.3-3 8-7 9.5-4-1.5-7-5.2-7-9.5V6z',
    clock: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 7.5V12l3 2',
    keyboard: 'M3 7a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2zM7 9.5h.5M11 9.5h.5M15 9.5h.5M7 13h.5M16.5 13h.5M8 16h8',
    route: 'M5 6h6a3 3 0 0 1 0 6H9a3 3 0 0 0 0 6h10M16 15l3 3-3 3',
    note: 'M6 4h9l4 4v12H6zM15 4v4h4M9 12h7M9 16h5',
    flag: 'M5 21V4M5 4h11l-2 4 2 4H5',
    arrow: 'M9 6l6 6-6 6',
    pin: 'M12 21s-6-5.6-6-10.5a6 6 0 0 1 12 0C18 15.4 12 21 12 21zM12 8.5a2 2 0 1 0 0 4 2 2 0 0 0 0-4z',
    user: 'M12 4a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM4.5 20a7.5 7.5 0 0 1 15 0',
    layout: 'M4 4h7v7H4zM13 4h7v4h-7zM13 10h7v10h-7zM4 13h7v7H4z',
  };
  const ic = (name, cls = '') => `<svg class="ic ${cls}" viewBox="0 0 24 24" aria-hidden="true"><path d="${P[name]}"/></svg>`;
  const icFill = (name, cls = '') => `<svg class="ic ic--fill ${cls}" viewBox="0 0 24 24" aria-hidden="true"><path d="${P[name]}"/></svg>`;

  // ---- live state (URL: ?view=decisions&t=39&tour=1&play=1) ----
  const q = new URLSearchParams(location.search);
  const st = {
    t: clamp(parseFloat(q.get('t')) || 0, 0, S.duration), playing: q.get('play') === '1', speed: 1,
    view: VIEWS.some((v) => v.id === q.get('view')) ? q.get('view') : 'vision', tour: q.get('tour') === '1',
    layers: { boxes: true, landmarks: true, seats: true, scores: true },
    capture: 'idle', countEnd: 0,
  };

  // ---- facts derived once from the story ----
  const PEOPLE = S.people, ORDER = ['dana', 'maya', 'sam', 'v4'];  // face tracks, left to right
  const capById = Object.fromEntries(S.captions.map((c) => [c.id, c]));
  const notesEv = S.events.filter((e) => e.type === 'notes').sort((a, b) => a.t - b.t);
  const notesAt = (x) => { let s = 'on'; for (const n of notesEv) if (n.t <= x) s = n.state; return s; };
  const offAt = (x) => notesAt(x) === 'off';
  const OFF = [];
  { let a = null; for (const n of notesEv) { if (n.state === 'off' && a === null) a = n.t; else if (n.state !== 'off' && a !== null) { OFF.push([a, n.t]); a = null; } } if (a !== null) OFF.push([a, S.duration]); }
  const WIPES = notesEv.filter((e) => e.state === 'off').map((e) => ({ t: e.t, from: e.t - (e.wipe || S.wipeWindow) }));
  const wipeOf = (c) => WIPES.find((w) => c.t >= w.from && c.t < w.t) || null;
  const removedAt = (c, t) => { const w = wipeOf(c); return !!w && t >= w.t; };
  const recorded = (c) => !c.transient && notesAt(c.t) === 'on';
  const checkinAt = {};
  for (const e of S.events) if (e.type === 'checkin') checkinAt[e.who] = e.t;
  const named = (id, t) => !!PEOPLE[id].name && checkinAt[id] !== undefined && checkinAt[id] <= t;
  const voiceNo = {}, firstHeard = {};
  { let n = 0; for (const c of [...S.captions].sort((a, b) => a.t - b.t)) if (recorded(c) && !(c.who in voiceNo)) { voiceNo[c.who] = ++n; firstHeard[c.who] = c.t; } }
  const VOICE_ORDER = Object.keys(voiceNo).sort((a, b) => voiceNo[a] - voiceNo[b]);
  const heardBy = (id, t) => firstHeard[id] !== undefined && firstHeard[id] <= t;
  const unnamedLabel = (id) => PEOPLE[id].label || 'Voice ' + voiceNo[id];
  const nameOf = (id, t) => (named(id, t) ? PEOPLE[id].name : unnamedLabel(id) + ' · unnamed');
  const shortName = (id, t) => (named(id, t) ? PEOPLE[id].name.split(' ')[0] : unnamedLabel(id));
  const initials = (id) => PEOPLE[id].name.split(' ').map((w) => w[0]).join('');
  const avatar = (id, t) => (named(id, t) ? `<span class="avatar avatar--${PEOPLE[id].tone}">${initials(id)}</span>` : '<span class="avatar avatar--neutral">?</span>');
  const finalAt = (c) => c.t + c.dur + X.asr.lag;
  const attributedAt = (c) => c.t + Math.min(c.dur, X.voices.window);
  const simOf = (c) => 0.8 + 0.12 * hash(99, seedOf(c.id));
  const capAt = (t) => { let cur = null; for (const c of S.captions) if (c.t <= t) cur = c; return cur; };
  const speakingAt = (t) => S.captions.find((c) => t >= c.t && t < c.t + c.dur) || null;
  const lastFinal = (t) => { let f = null; for (const c of S.captions) if (recorded(c) && finalAt(c) <= t && !removedAt(c, t)) f = c; return f; };
  const decAt = (t) => { let d = null; for (const x of S.decisions) if (x.t <= t) d = x; return d; };
  const stepOf = (d, k) => d.steps.find((s) => s.k === k);

  // Light-ASD speaking probability for a face at media time x (null while off record).
  function asd(id, x) {
    if (x < 0 || offAt(x)) return null;
    const s = seedOf(id);
    let v = 0.05 + 0.08 * noise(x * 0.9, s) + 0.16 * Math.pow(noise(x * 0.37, s + 9), 6);
    for (const c of S.captions) {
      if (c.who !== id || x < c.t - 0.4 || x > c.t + c.dur + 0.6) continue;
      const e = smooth(c.t - 0.3, c.t + 0.2, x) * (1 - smooth(c.t + c.dur - 0.1, c.t + c.dur + 0.4, x));
      const lvl = 0.8 + 0.15 * noise(x * 2.6, s + 5) - 0.12 * Math.pow(Math.max(0, Math.sin(x * 6.3 + s)), 8);
      v += (lvl - v) * e;
    }
    return clamp(v, 0, 0.99);
  }
  const peakAsd = (c) => asd(c.who, c.t + c.dur * 0.5) || 0;
  const detScore = (id, f) => clamp(X.faces[id].score + 0.03 * (noise(f * 0.21, seedOf(id) + 3) - 0.5), 0, 0.999);
  const jit = (id, f, k) => (noise(f * 0.33, seedOf(id) + k * 17) - 0.5) * 1.6;  // px of detector noise

  // Speech recognition: partial hypotheses every chunk, then the final line with casing and punctuation.
  const norm = (w) => w.toLowerCase().replace(/[.,!?:;]/g, '');
  function asrAt(c, t) {
    const W = c.text.split(' '), N = W.length, fixes = X.asr.corrections[c.id] || {};
    const end = c.t + c.dur, fin = finalAt(c);
    const count = (x) => clamp(Math.floor(((x - c.t) / c.dur) * N - 0.6), 0, N);
    const revs = [];
    let prev = null;
    for (let k = 1; c.t + k * X.asr.chunk < end && c.t + k * X.asr.chunk <= t; k++) {
      const at = c.t + k * X.asr.chunk, n = count(at);
      if (!n) continue;
      const prevN = prev ? prev.words.length : 0;
      const words = [];
      for (let i = 0; i < n; i++) {
        const f = fixes[i];
        words.push(f && (f.fix === 'final' || i >= prevN) ? f.heard : norm(W[i]));
      }
      const changed = words.map((w, i) => !!prev && i < prevN && w !== prev.words[i]);
      if (prev && n === prevN && !changed.some(Boolean)) continue;
      revs.push((prev = { at, k, words, changed, final: false }));
    }
    if (t >= fin) {
      const lw = prev ? prev.words : [];
      revs.push({ at: fin, words: W, changed: W.map((w, i) => i < lw.length && norm(w) !== lw[i]), final: true });
    }
    return { revs, final: t >= fin, cur: revs[revs.length - 1] || null, fin, end };
  }

  // Voice map: one CAM++ embedding per ~1.2 s of speech, drawn as a 2D projection.
  const POINTS = [];
  for (const c of S.captions) {
    if (!recorded(c)) continue;
    const n = Math.max(1, Math.round(c.dur / X.voices.window)), sd = seedOf(c.id), [cx, cy] = X.voices.centers[c.who];
    for (let j = 0; j < n; j++) {
      POINTS.push({ c, who: c.who, at: c.t + Math.min(c.dur, (j + 1) * X.voices.window), x: cx + gauss(sd, j * 2) * X.voices.spread,
        y: cy + gauss(sd, j * 2 + 1) * X.voices.spread, sim: 0.8 + 0.13 * hash(50 + j, sd), wipe: wipeOf(c) });
    }
  }

  // Audio level for the waveform (seeded; follows each caption's words).
  function amp(x) {
    let a = 0.025 + 0.03 * noise(x * 23, 3);
    for (const c of S.captions) {
      if (x < c.t - 0.1 || x > c.t + c.dur + 0.2) continue;
      const e = smooth(c.t - 0.06, c.t + 0.06, x) * (1 - smooth(c.t + c.dur - 0.08, c.t + c.dur + 0.1, x));
      const N = c.text.split(' ').length, u = ((x - c.t) / c.dur) * N, wp = u - Math.floor(u);
      const word = 0.3 + 0.7 * Math.sin(Math.PI * clamp(wp * 1.18, 0, 1));
      const syl = 0.55 + 0.45 * Math.abs(Math.sin(x * 13.5 + seedOf(c.id)));
      const gain = { maya: 0.9, dana: 0.8, sam: 1, v4: 0.72 }[c.who] || 0.8;
      a = Math.max(a, e * word * syl * gain * (0.7 + 0.3 * noise(x * 9, seedOf(c.id))));
    }
    return clamp(a, 0, 1);
  }

  // Decision routes and their slow-motion replay (each step's share of the replay).
  const SEQ = {
    act: [['utter', 1], ['u-r', 0.4], ['rules', 1], ['r-d', 0.4], ['decider', 3], ['d-p', 1.2], ['policy', 1], ['p-a', 0.4], ['actuator', 1.2]],
    escalate: [['utter', 1], ['u-r', 0.4], ['rules', 1], ['r-d', 0.4], ['decider', 2.4], ['d-q', 0.6], ['qwen', 2.2], ['q-p', 0.6], ['policy', 1], ['p-a', 0.4], ['actuator', 1.2]],
    ignore: [['utter', 1], ['u-r', 0.4], ['rules', 1], ['r-d', 0.4], ['decider', 3], ['d-i', 0.6], ['ignore', 1]],
    rule: [['utter', 1], ['u-r', 0.4], ['rules', 1.6], ['r-p', 1.6], ['policy', 1], ['p-a', 0.4], ['actuator', 1.2]],
  };
  function routeOf(d) {
    const dec = stepOf(d, 'decider');
    if (d.action === 'recap') return 'escalate';
    if (dec.skipped) return 'rule';
    return dec.conf >= 0.6 ? 'act' : dec.conf >= 0.4 ? 'escalate' : 'ignore';
  }
  function replayOf(d, t) {
    const seq = SEQ[routeOf(d)], total = seq.reduce((a, s) => a + s[1], 0);
    const p = ((t - d.t) / (X.replay * 0.9)) * total;   // the last 10% holds the result
    const out = {};
    let acc = 0;
    for (const [k, w] of seq) { out[k] = clamp((p - acc) / w, 0, 1); acc += w; }
    return out;
  }
  const stateIn = (R, k) => (!R ? 'wait' : !(k in R) ? 'skip' : R[k] <= 0 ? 'wait' : R[k] < 1 ? 'active' : 'done');

  // Everything else that changes on the timeline (notes, devices, Discord).
  function derive(t) {
    const D = { tasks: [], items: [], devices: JSON.parse(JSON.stringify(S.devices)), feed: {}, egress: 0, last: {} };
    for (const e of S.events) {
      if (e.t > t) continue;
      switch (e.type) {
        case 'task': D.tasks.push(e); D.last.notes = e.t; break;
        case 'decision-item': D.items.push(e); D.last.notes = e.t; break;
        case 'device': Object.assign(D.devices[e.name], { state: e.state, tu: e.t }); D.last[e.name] = e.t; break;
        case 'feed': if (e.kind === 'discord') { D.feed[e.id] = { ...e }; D.last.discord = e.t; } break;
        case 'feed-update': if (D.feed[e.id]) { Object.assign(D.feed[e.id], { status: e.status, detail: e.detail, tu: e.t }); D.last.discord = e.t; } break;
        case 'egress': D.egress += e.bytes; break;
      }
    }
    return D;
  }

  // ---- colours come from tokens.css so the canvases follow the theme ----
  let C = {};
  function readColours() {
    const cs = getComputedStyle(document.documentElement), v = (n) => cs.getPropertyValue(n).trim();
    C = { t1: v('--t1'), t2: v('--t2'), t3: v('--t3'), line: v('--line'), line2: v('--line-2'), line3: v('--line-3'), s1: v('--s1'), s2: v('--s2'), s3: v('--s3'),
      blue: v('--accent-bg'), green: v('--ok-bg'), purple: v('--purple'), neutral: v('--t3'), warn: v('--warn-bg'), warnDeep: v('--warn-deep'), okDeep: v('--ok-deep'),
      bad: v('--bad-bg'), font: v('--font'), mono: v('--mono') };
  }
  const toneCol = (id) => C[PEOPLE[id].tone] || C.neutral;
  function alpha(c, a) {
    if (c[0] === '#') { const n = parseInt(c.slice(1), 16); return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`; }
    const m = c.match(/rgba?\(([^)]+)\)/);
    if (!m) return c;
    const p = m[1].split(',').map((x) => x.trim());
    return `rgba(${p[0]},${p[1]},${p[2]},${(p[3] !== undefined ? parseFloat(p[3]) : 1) * a})`;
  }

  // ---- DOM helpers: only touch a node when its markup changes ----
  const cache = new Map();
  function put(node, html, key) {
    key = key || node.id;
    if (cache.get(key) === html) return false;
    cache.set(key, html); node.innerHTML = html; return true;
  }
  const head = (h2, meta) => `<h2>${h2}</h2>${meta ? `<span class="meta">${meta}</span>` : ''}`;

  // ---- canvases (sized to the stage scale so lines stay crisp) ----
  let fitScale = 1;
  function ctx2d(cv) {
    const w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return null;
    const R = Math.min(3, Math.max(1, (window.devicePixelRatio || 1) * fitScale));
    if (cv._w !== w || cv._h !== h || cv._r !== R) { cv.width = Math.round(w * R); cv.height = Math.round(h * R); cv._w = w; cv._h = h; cv._r = R; }
    const g = cv.getContext('2d');
    g.setTransform(R, 0, 0, R, 0, 0); g.clearRect(0, 0, w, h);
    return { g, w, h };
  }
  function rr(g, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
  }
  let hatchCache = null;
  function hatch(g) {
    if (!hatchCache) {
      const c = document.createElement('canvas'); c.width = c.height = 8;
      const h = c.getContext('2d'); h.strokeStyle = C.line3; h.lineWidth = 1.2;
      h.beginPath(); h.moveTo(-2, 10); h.lineTo(10, -2); h.moveTo(6, 10); h.lineTo(10, 6); h.moveTo(-2, 2); h.lineTo(2, -2); h.stroke();
      hatchCache = c;
    }
    return g.createPattern(hatchCache, 'repeat');
  }

  // =============================== top bar + transport ===============================
  function buildChrome() {
    $('#tabs').innerHTML = VIEWS.map((v) => `<button class="tab" type="button" role="tab" id="tab-${v.id}" data-view="${v.id}" aria-controls="view-${v.id}">${ic(v.icon)}${v.label}<span class="kbd">${v.key.toUpperCase()}</span></button>`).join('');
    $('#exSub').textContent = S.meeting.title + ' · ' + S.meeting.room;
    $('#dashLink').innerHTML = ic('layout') + 'Dashboard';
    $('#restartBtn').innerHTML = ic('restart');
    $('#keysBtn').innerHTML = ic('keyboard');
    $('#inputChip').innerHTML = ic('film') + 'Prerecorded meeting · live local inference';
    $('#sampleChip').hidden = !CFG.showSampleChip;
    $('#tlMarks').innerHTML = S.beats.map((b) => `<span class="tl-mark" data-beat="${b.key}" style="left:${(b.t / S.duration) * 100}%" title="${b.key} · ${esc(b.label)}"></span>`).join('');
  }
  function renderChrome() {
    for (const v of VIEWS) $('#tab-' + v.id).setAttribute('aria-selected', String(st.view === v.id));
    const tb = $('#tourBtn');
    tb.classList.toggle('is-on', st.tour); tb.setAttribute('aria-pressed', String(st.tour));
    put(tb, ic('flag') + (st.tour ? 'Tour on' : 'Tour') + '<span class="kbd">A</span>', 'tour');
    const pb = $('#playBtn');
    put(pb, st.playing ? ic('pause') : icFill('play'), 'play');
    pb.setAttribute('aria-label', st.playing ? 'Pause (Space)' : 'Play (Space)');
    put($('#clock'), `<b>${mmss(st.t)}</b> / ${mmss(S.duration)}`);
    $('#tlFill').style.width = (st.t / S.duration) * 100 + '%';
    document.querySelectorAll('.tl-mark').forEach((m) => m.classList.toggle('is-past', st.t >= S.beats.find((b) => b.key === m.dataset.beat).t));
    let beat = S.beats[0];
    for (const b of S.beats) if (st.t >= b.t) beat = b;
    put($('#beatLabel'), `<b>${beat.key}</b> · ${esc(beat.label)}`);
    put($('#speedBtn'), st.speed + '×');
  }
  function headHtml(v, stats) {
    return `<div class="view-title"><h1>${v.title}</h1><p>${v.desc}</p></div><div class="stats">${stats.map(([b, s]) => `<span class="stat"><b>${b}</b><span>${s}</span></span>`).join('')}</div>`;
  }

  // =============================== shared: attribution ===============================
  function attrHtml(title, t, note) {
    const c = capAt(t), off = offAt(t);
    const speaking = c && t < c.t + c.dur;
    let meta = '', body;
    if (off) body = `<div class="attr-empty"><b>Off record</b><span>Nothing is attributed or kept until notes are back on.</span></div>`;
    else if (!c) body = `<div class="attr-empty"><b>Waiting for someone to speak</b><span>Voice, face and seat evidence for each line appears here.</span></div>`;
    else if (c.transient) body = `<div class="attr-empty"><b>Resume phrase</b><span>Heard once to switch notes back on, then dropped. It was never attributed or kept.</span></div>`;
    else {
      const p = PEOPLE[c.who], nm = named(c.who, c.t), sim = simOf(c), face = peakAsd(c);
      const ready = t >= attributedAt(c);
      meta = mmss(c.t) + (speaking ? ' · speaking now' : ' · last speaker');
      const row = (yes, icon, b, d, v, bar) => `<div class="ev${yes ? ' is-yes' : ''}">${ic(icon)}<b>${b}</b><span class="d">${d}</span>${bar !== null ? `<span class="mbar"><i style="width:${bar * 100}%"></i></span>` : '<span></span>'}<span class="v">${v}</span></div>`;
      body = `<div class="attr-result">${ready ? avatar(c.who, c.t) : '<span class="avatar avatar--neutral">?</span>'}<span class="who">${ready ? esc(nameOf(c.who, c.t)) + ' · Seat ' + p.seat : 'Matching the voice…'}</span>` +
        (ready ? (nm ? '<span class="chip chip--ok">' + ic('check') + 'Voice and face agree</span>' : '<span class="chip">Name unknown</span>') : '') + '</div>' +
        row(ready, 'wave', ready ? 'Voice ' + voiceNo[c.who] : 'Voice …', 'CAM++ match', ready ? sim.toFixed(2) : '…', ready ? sim : 0) +
        (p.face ? row(face > 0.5, 'face', 'Track #' + X.faces[c.who].track, 'Light-ASD speaking', face.toFixed(2), face) : row(false, 'face', 'No face', 'voice only', '—', null)) +
        row(true, 'pin', 'Seat ' + p.seat, p.face ? 'face inside the seat zone' : 'voice from this seat', ic('check'), null) +
        row(nm, nm ? 'checkCircle' : 'user', nm ? esc(p.name) : 'No check-in', nm ? 'browser check-in · ' + mmss(checkinAt[c.who]) : 'stays unnamed', nm ? ic('check') : '—', null);
    }
    return `<div class="card-h">${head(title, meta)}</div><div class="card-b">${body}${note ? `<span class="attr-note">${note}</span>` : ''}</div>`;
  }

  // =============================== 1 · Vision ===============================
  function buildVision() {
    const pct = (b) => `left:${b[0] * 100}%;top:${b[1] * 100}%;width:${b[2] * 100}%;height:${b[3] * 100}%`;
    $('#seatLayer').innerHTML = Object.entries(X.seats).map(([k, z]) => `<div class="seat" style="${pct(z)}"><span class="seat-tag">Seat ${k}</span></div>`).join('');
    $('#detLayer').innerHTML = ORDER.map((id) => `<div class="det" id="det-${id}" style="${pct(PEOPLE[id].face)}"><div class="det-box"></div><div class="det-label"></div><div class="asd"></div></div>`).join('');
    $('#lmLayer').innerHTML = ORDER.map((id) => X.faces[id].landmarks.map((p, i) => `<i class="lm" id="lm-${id}-${i}" style="left:${p[0] * 100}%;top:${p[1] * 100}%"></i>`).join('')).join('');
    $('#fChipTL').innerHTML = ic('face') + 'YuNet · face boxes and 5 landmarks';
    $('#fChipTR').innerHTML = ic('wave') + 'Light-ASD · who is speaking';
    const rows = ORDER.concat('e').map((id) => `<div class="track"><span class="trk-av" id="trkav-${id}"></span><span class="track-name" id="trkn-${id}"></span><span class="track-meta" id="trkm-${id}"></span>` +
      `<div class="track-spark">${id === 'e' ? '<span class="novoice">No face in view · speech from this seat relies on the voice alone</span>' : `<canvas class="spark-c" id="spark-${id}" aria-hidden="true"></canvas><span class="v" id="trkv-${id}"></span>`}</div></div>`).join('');
    $('#tracksCard').innerHTML = `<div class="card-h">${head('Face tracks', 'speaking probability · last ' + CFG.sparkWindow + ' s')}</div><div class="card-b"><div class="tracks">${rows}</div></div>`;
  }
  function renderVision() {
    const t = st.t, f = Math.floor(t * CFG.frameRate), off = offAt(t), L = st.layers;
    put($('#head-vision'), headHtml(VIEWS[0], [['YuNet', 'CPU · every frame'], ['Light-ASD', 'GPU · 25 fps'], ['Tracked', '4 faces · 5 seats']]));
    $('#seatLayer').classList.toggle('is-hidden', !L.seats || off);
    $('#lmLayer').classList.toggle('is-hidden', !L.landmarks || off);
    const det = $('#detLayer');
    det.classList.toggle('is-hidden', off || (!L.boxes && !L.scores));
    det.classList.toggle('layers-hide-boxes', !L.boxes);
    det.classList.toggle('layers-hide-scores', !L.scores);
    let speaker = null;
    for (const id of ORDER) {
      const el = $('#det-' + id), v = asd(id, t) || 0, active = !off && v > 0.5;
      el.classList.toggle('is-active', active && L.scores);
      el.style.transform = `translate(${jit(id, f, 0).toFixed(2)}px, ${jit(id, f, 1).toFixed(2)}px)`;
      put(el.children[1], `<b>#${X.faces[id].track}</b>face ${detScore(id, f).toFixed(2)}`, 'dl-' + id);
      put(el.children[2], `<span class="mbar"><i style="width:${(v * 100).toFixed(1)}%"></i></span><span>speaking</span><span class="v">${v.toFixed(2)}</span>`, 'da-' + id);
      X.faces[id].landmarks.forEach((p, i) => { $('#lm-' + id + '-' + i).style.transform = `translate(${jit(id, f + i * 7, 2).toFixed(2)}px, ${jit(id, f + i * 7, 3).toFixed(2)}px)`; });
      if (active && (!speaker || v > speaker.v)) speaker = { id, v };
    }
    put($('#fFrame'), `frame ${String(f).padStart(4, '0')} · ${mmssd(t)} · ${CFG.frameRate} fps`);
    const sc = $('#fSpeaking');
    sc.classList.toggle('is-empty', !speaker);
    if (speaker) put(sc, `<span class="dotk" style="background:${toneCol(speaker.id)}"></span>Speaking now · ${esc(nameOf(speaker.id, t))} · <span class="mono">${speaker.v.toFixed(2)}</span>`, 'spk');
    const mask = $('#fMask');
    mask.hidden = !off;
    if (off) put(mask, ic('eyeOff', 'ic--xl') + '<h3>Off record</h3><p>The video is masked and no faces are tracked. Nothing is kept until someone says “back on the record”.</p>', 'fmask');
    // layer switches
    const tg = (k, label) => `<button class="toggle" type="button" data-layer="${k}" aria-pressed="${L[k]}"><span class="sw">${ic('check')}</span>${label}</button>`;
    put($('#layersCard'), `<span class="lbl">Layers</span>${tg('boxes', 'Face boxes')}${tg('landmarks', 'Landmarks')}${tg('seats', 'Seat zones')}${tg('scores', 'Speaking scores')}` +
      '<span class="note">Detection runs on every frame and a tracker keeps the IDs. A face is evidence of a seat, never proof of a name.</span>');
    // tracks
    for (const id of ORDER.concat('e')) {
      const p = PEOPLE[id], nm = named(id, t), heard = heardBy(id, t);
      put($('#trkav-' + id), nm ? avatar(id, t) : '<span class="avatar avatar--neutral">?</span>', 'av-' + id);
      const name = nm ? esc(p.name) : id === 'e' ? 'Seat E' : heard ? esc(unnamedLabel(id)) : 'Unnamed face';
      const prov = nm ? 'checked in · ' + mmss(checkinAt[id]) : id === 'e' ? 'facing away' : heard ? 'heard, no check-in' : 'no check-in';
      put($('#trkn-' + id), `<b class="${nm ? '' : 'is-unnamed'}">${name}</b><span>${prov}</span>`, 'tn-' + id);
      put($('#trkm-' + id), id === 'e' ? '<span class="seat-chip">Seat E</span><span class="mono">no face</span>'
        : `<span class="seat-chip">Seat ${p.seat}</span><span class="mono">#${X.faces[id].track} · face ${off ? '—' : detScore(id, f).toFixed(2)}</span>`, 'tm-' + id);
      if (id !== 'e') { const v = asd(id, t); put($('#trkv-' + id), v === null ? '—' : v.toFixed(2), 'tv-' + id); }
    }
    put($('#attrVision'), attrHtml('From face to name', t));
  }
  function drawSparks() {
    const t = st.t;
    for (const id of ORDER) {
      const k = ctx2d($('#spark-' + id));
      if (!k) continue;
      const { g, w, h } = k, col = toneCol(id), n = 100, t0 = t - CFG.sparkWindow;
      const xOf = (x) => ((x - t0) / CFG.sparkWindow) * w, yOf = (v) => h - 3 - v * (h - 6);
      g.fillStyle = C.s2; rr(g, 0, 0, w, h, 6); g.fill();
      for (const [a, b] of OFF) if (b > t0 && a < t) { g.fillStyle = hatch(g); g.fillRect(xOf(Math.max(a, t0)), 0, xOf(Math.min(b, t)) - xOf(Math.max(a, t0)), h); }
      g.setLineDash([4, 4]); g.strokeStyle = C.line2; g.lineWidth = 1; g.beginPath(); g.moveTo(0, yOf(0.5)); g.lineTo(w, yOf(0.5)); g.stroke(); g.setLineDash([]);
      // area + line, broken where there is no output
      let run = [];
      const flush = () => {
        if (run.length > 1) {
          g.beginPath(); g.moveTo(run[0][0], h); for (const [x, y] of run) g.lineTo(x, y); g.lineTo(run[run.length - 1][0], h); g.closePath(); g.fillStyle = alpha(col, 0.14); g.fill();
          g.beginPath(); run.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.strokeStyle = col; g.lineWidth = 2; g.lineJoin = 'round'; g.stroke();
        }
        run = [];
      };
      for (let i = 0; i <= n; i++) {
        const x = t0 + (i / n) * CFG.sparkWindow;
        const v = x < 0 ? null : asd(id, x);
        if (v === null) flush(); else run.push([xOf(x), yOf(v)]);
      }
      flush();
    }
  }

  // =============================== 2 · Transcription ===============================
  function renderTranscription() {
    const t = st.t, off = offAt(t);
    const heard = VOICE_ORDER.filter((id) => heardBy(id, t)).length;
    put($('#head-transcription'), headHtml(VIEWS[1], [['VAD', 'Silero · CPU'], ['ASR', 'Parakeet TDT · ' + X.asr.ms + ' ms'], ['Voices', heard + ' heard']]));
    put($('#waveHead'), head('Audio', `16 kHz mono · shaded where Silero VAD hears speech · last ${CFG.waveWindow} s`));
    put($('#vmapHead'), head('Voice map', 'WeSpeaker CAM++ · 2D view'));
    put($('#lanesHead'), head('Who spoke when', 'whole meeting · hatched: off record or removed'));

    // transcript: the line being heard, and how its text was refined
    const c = capAt(t);
    let now, revs = '';
    if (off) {
      const sp = speakingAt(t);
      now = `<div class="now-off">${ic('lock')}<div><b>${sp && sp.transient ? 'Resume phrase heard' : 'Off record'}</b><span>${sp && sp.transient ? '“Back on the record” switches notes on. The phrase itself is not kept.' : 'Audio is checked only for “back on the record”. Nothing is transcribed or kept.'}</span></div></div>`;
    } else if (!c) {
      now = `<div class="now-h"><span class="who faint">Listening</span></div><span class="now-text"><span class="p">Waiting for speech</span><span class="caret"></span></span>`;
    } else if (c.transient) {
      now = `<div class="now-off is-ok">${ic('checkCircle')}<div><b>Back on the record</b><span>The resume phrase switched notes on and was dropped. Transcription continues with the next line.</span></div></div>`;
    } else {
      const A = asrAt(c, t), ready = t >= attributedAt(c), speaking = t < A.end;
      const who = ready ? `<span class="who tone-${named(c.who, c.t) ? PEOPLE[c.who].tone : 'neutral'}">${esc(nameOf(c.who, c.t))}</span><span class="seat-chip">Seat ${PEOPLE[c.who].seat}</span>` : '<span class="who faint">Voice …</span>';
      const chip = A.final ? `<span class="chip chip--ok">${ic('check')}Final</span>` : A.cur ? `<span class="chip chip--accent">Partial · chunk ${A.cur.k}</span>` : '<span class="chip">Listening</span>';
      let text;
      if (!A.cur) text = '<span class="p">…</span><span class="caret"></span>';
      else {
        const fresh = A.final ? t - A.fin < 1.5 : true;
        text = A.cur.words.map((w, i) => (A.cur.changed[i] && fresh ? `<mark>${esc(w)}</mark>` : A.final ? esc(w) : `<span class="p">${esc(w)}</span>`)).join(' ') + (speaking ? '<span class="caret"></span>' : '');
      }
      now = `<div class="now-h">${who}${chip}<span class="ts">${mmss(c.t)}</span></div><span class="now-text">${text}</span>`;
      const rows = A.revs.slice(-4).map((r) => {
        let words = r.words.map((w, i) => (r.changed[i] ? `<mark>${esc(w)}</mark>` : esc(w)));
        let len = 0, from = words.length;
        while (from > 0 && len + r.words[from - 1].length + 1 <= 58) { len += r.words[from - 1].length + 1; from--; }
        const txt = (from > 0 ? '… ' : '') + words.slice(from).join(' ');
        return `<div class="rev${r.final ? ' is-final' : ''}"><span class="at">+${(r.at - c.t).toFixed(1)} s</span><span class="kind">${r.final ? 'Final' : 'Partial ' + r.k}</span><span class="txt">${txt}</span></div>`;
      }).join('');
      revs = rows;
    }
    put($('#transcriptCard'), `<div class="card-h">${head('Live transcript', 'Parakeet TDT 0.6B · local GPU')}</div><div class="card-b"><div class="now">${now}</div>` +
      `<div class="revs-h">How the line was refined<span>partials every ${X.asr.chunk} s · final ${X.asr.lag} s after speech</span></div><div class="revs">${revs || '<div class="rev"><span class="at"></span><span class="kind faint">—</span><span class="txt faint">Partial guesses appear while someone speaks</span></div>'}</div></div>`);

    // voice map readout
    const vis = POINTS.filter((p) => p.at <= t && !(p.wipe && t >= p.wipe.t));
    const last = vis[vis.length - 1];
    put($('#vmapRead'), off ? '<b>Off record · no voices are recorded</b><span>Segments from the last 60 s were removed when notes went off.</span>'
      : last ? `<b>Latest segment → Voice ${voiceNo[last.who]} · ${esc(named(last.who, t) ? PEOPLE[last.who].name : 'unnamed')} · cosine ${last.sim.toFixed(2)}</b><span>Names come from check-in, never from the voice itself.</span>`
        : '<b>No voices yet</b><span>Each 1.2 s of speech becomes a point. Close points are the same voice.</span>');
    put($('#attrTrans'), attrHtml('Who said it', t, 'A face is evidence of a seat, never proof of a name. Overlapping or off-camera speech stays unnamed rather than guessed.'));
  }
  function drawWave() {
    const k = ctx2d($('#waveCanvas'));
    if (!k) return;
    const { g, w, h } = k, t = st.t, win = CFG.waveWindow, t0 = t - win, xOf = (x) => ((x - t0) / win) * w;
    const mid = 72, half = 42;
    g.font = `600 16px ${C.font}`; g.textBaseline = 'middle';
    // VAD speech segments, coloured once the voice is matched
    for (const c of S.captions) {
      const a = c.t - 0.05, b = Math.min(t, c.t + c.dur + 0.15);
      if (b < t0 || a > t) continue;
      const x0 = xOf(Math.max(a, t0)), x1 = xOf(b);
      const off = offAt(c.t), gone = removedAt(c, t), ready = t >= attributedAt(c) && !off && !gone;
      g.fillStyle = off || gone ? hatch(g) : ready ? alpha(toneCol(c.who), 0.12) : alpha(C.t1, 0.05);
      rr(g, x0, 0, Math.max(2, x1 - x0), h, 8); g.fill();
      const label = off ? (c.transient ? 'Resume phrase · not kept' : 'Off record') : gone ? 'Removed by off record' : ready ? nameOf(c.who, c.t) : 'Speech · matching voice…';
      if (x1 - x0 > 90) { g.save(); rr(g, x0, 0, x1 - x0, h, 8); g.clip(); g.fillStyle = ready ? toneCol(c.who) : C.t2; g.fillText(label, Math.max(x0, 0) + 10, 14); g.restore(); }
    }
    for (const [a, b] of OFF) {
      if (b < t0 || a > t) continue;
      const x0 = xOf(Math.max(a, t0)), x1 = xOf(Math.min(b, t));
      g.fillStyle = alpha(C.bad, 0.05); g.fillRect(x0, 0, x1 - x0, h);
      if (x1 - x0 > 420) { g.fillStyle = C.t2; g.fillText('Off record · listening only for “back on the record”', x0 + 10, h - 12); }
    }
    // level bars on a fixed time grid, so they scroll without shimmering
    const dt = win / (w / 4);
    for (let i = Math.ceil(t0 / dt); i * dt <= t; i++) {
      const x = i * dt; if (x < 0) continue;
      const a = amp(x), px = xOf(x), hh = Math.max(1.5, a * half);
      const sp = S.captions.find((c) => x >= c.t && x < c.t + c.dur);
      let col = C.line3;
      if (sp) col = offAt(x) || removedAt(sp, t) ? C.t3 : t >= attributedAt(sp) ? toneCol(sp.who) : C.t2;
      g.fillStyle = col; g.fillRect(px - 1.5, mid - hh, 3, hh * 2);
    }
    g.fillStyle = C.t1; g.fillRect(w - 2, 0, 2, h);
  }
  function drawVoiceMap() {
    const k = ctx2d($('#vmapCanvas'));
    if (!k) return;
    const { g, w, h } = k, t = st.t, pad = 28;
    const px = (p) => [pad + p[0] * (w - pad * 2), pad + 10 + p[1] * (h - pad * 2 - 10)];
    g.fillStyle = C.line; for (let x = 16; x < w; x += 24) for (let y = 16; y < h; y += 24) g.fillRect(x, y, 1.5, 1.5);
    const pts = POINTS.filter((p) => p.at <= t).map((p) => ({ ...p, fade: p.wipe && t >= p.wipe.t ? 1 - clamp((t - p.wipe.t) / 0.8, 0, 1) : 1 })).filter((p) => p.fade > 0);
    for (const id of VOICE_ORDER) {
      if (!heardBy(id, t)) continue;
      const mine = pts.filter((p) => p.who === id);
      const cx = mine.length ? mine.reduce((a, p) => a + p.x, 0) / mine.length : X.voices.centers[id][0];
      const cy = mine.length ? mine.reduce((a, p) => a + p.y, 0) / mine.length : X.voices.centers[id][1];
      const [x, y] = px([cx, cy]), col = toneCol(id);
      g.beginPath(); g.arc(x, y, 40, 0, Math.PI * 2); g.fillStyle = alpha(col, mine.length ? 0.1 : 0.04); g.fill();
      g.setLineDash([4, 4]); g.strokeStyle = alpha(col, 0.45); g.lineWidth = 1.5; g.stroke(); g.setLineDash([]);
      const a = 'Voice ' + voiceNo[id], b = ' · ' + (named(id, t) ? PEOPLE[id].name : 'unnamed');
      g.textBaseline = 'alphabetic';
      g.font = `700 16px ${C.font}`; const wa = g.measureText(a).width;
      g.font = `400 16px ${C.font}`; const wb = g.measureText(b).width;
      const lx = clamp(x - (wa + wb) / 2, 6, w - wa - wb - 6), ly = cy < 0.5 ? y - 50 : y + 60;
      g.font = `700 16px ${C.font}`; g.fillStyle = col; g.fillText(a, lx, ly);
      g.font = `400 16px ${C.font}`; g.fillStyle = C.t2; g.fillText(b, lx + wa, ly);
      for (const p of mine) {
        const [qx, qy] = px([p.x, p.y]), age = t - p.at;
        if (age < 1.2) { g.beginPath(); g.moveTo(qx, qy); g.lineTo(x, y); g.strokeStyle = alpha(col, 0.5 * (1 - age / 1.2)); g.lineWidth = 1.5; g.stroke(); }
        if (age < 0.6) { g.beginPath(); g.arc(qx, qy, 5 + age * 22, 0, Math.PI * 2); g.strokeStyle = alpha(col, 1 - age / 0.6); g.lineWidth = 2; g.stroke(); }
        g.beginPath(); g.arc(qx, qy, 5, 0, Math.PI * 2); g.fillStyle = alpha(col, 0.9 * p.fade); g.fill();
      }
    }
    g.textAlign = 'left';
    if (offAt(t)) { g.fillStyle = alpha(C.s1, 0.7); g.fillRect(0, 0, w, h); g.font = `600 18px ${C.font}`; g.fillStyle = C.t2; g.textAlign = 'center'; g.fillText('Off record · no voices recorded', w / 2, h / 2); g.textAlign = 'left'; }
  }
  function drawLanes() {
    const k = ctx2d($('#lanesCanvas'));
    if (!k) return;
    const { g, w, h } = k, t = st.t, x0 = 230, x1 = w - 8, xOf = (s) => x0 + (s / S.duration) * (x1 - x0);
    g.textBaseline = 'middle';
    VOICE_ORDER.forEach((id, i) => {
      const y = 2 + i * 24, col = toneCol(id), heard = heardBy(id, t);
      g.font = `700 16px ${C.font}`; g.fillStyle = heard ? col : C.t3;
      g.fillText(heard ? 'Voice ' + voiceNo[id] : 'Not heard yet', 0, y + 10);
      if (heard) { const wv = g.measureText('Voice ' + voiceNo[id] + ' ').width; g.font = `400 16px ${C.font}`; g.fillStyle = C.t2; g.fillText('· ' + (named(id, t) ? PEOPLE[id].name : 'unnamed'), wv, y + 10); }
      g.fillStyle = C.s2; rr(g, x0, y, x1 - x0, 20, 5); g.fill();
      for (const c of S.captions) {
        if (c.who !== id || c.t > t || !recorded(c)) continue;
        const a = xOf(c.t), b = xOf(Math.min(t, c.t + c.dur));
        g.fillStyle = removedAt(c, t) ? hatch(g) : col; rr(g, a, y + 2, Math.max(3, b - a), 16, 4); g.fill();
      }
    });
    for (const [a, b] of OFF) {
      if (t <= a) continue;
      const xa = xOf(a), xb = xOf(Math.min(t, b));
      g.fillStyle = hatch(g); g.fillRect(xa, 0, xb - xa, 98);
      g.fillStyle = alpha(C.bad, 0.06); g.fillRect(xa, 0, xb - xa, 98);
      if (xb - xa > 170) { g.font = `600 16px ${C.font}`; g.fillStyle = C.t1; g.fillText('Off record · not kept', xa + 10, 50); }
    }
    g.font = `400 16px ${C.mono}`; g.fillStyle = C.t3; g.textAlign = 'center';
    for (let m = 0; m <= S.duration; m += 30) g.fillText(mmss(m), clamp(xOf(m), x0 + 24, x1 - 24), h - 6);
    g.textAlign = 'left';
    g.fillStyle = C.t1; g.fillRect(xOf(t) - 1, 0, 2, 98);
  }

  // =============================== 3 · Decisions ===============================
  const CAS = {
    nodes: { utter: [0, 70, 300, 200], rules: [340, 70, 200, 200], decider: [580, 70, 420, 200], ignore: [700, 296, 180, 44], qwen: [1072, 206, 240, 120], policy: [1384, 70, 200, 200], actuator: [1624, 70, 208, 200] },
    links: { 'u-r': 'M300 170H340', 'r-d': 'M540 170H580', 'd-p': 'M1000 170H1384', 'd-q': 'M1000 230C1036 230 1036 266 1072 266', 'q-p': 'M1312 266C1348 266 1348 230 1384 230',
      'd-i': 'M790 270V296', 'r-p': 'M440 70V40Q440 24 456 24H1468Q1484 24 1484 40V70', 'p-a': 'M1584 170H1624' },
    chips: { 'd-p': [1112, 154, 160, '≥ 0.60 · act'], 'r-p': [812, 8, 300, 'Rule match · skips the models'] },
  };
  const grad = (id) => `<defs><linearGradient id="${id}" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="1832" y2="0"><stop offset="0" stop-color="#3492eb"/><stop offset=".5" stop-color="#00a3b5"/><stop offset="1" stop-color="#3cc29a"/></linearGradient></defs>`;
  function buildDecisions() {
    const paths = Object.entries(CAS.links).map(([k, d]) => `<path id="lk-${k}" d="${d}"/>`).join('');
    const nodes = Object.entries(CAS.nodes).map(([k, [x, y, w, h]]) => `<div class="node${k === 'ignore' ? ' node--pill' : ''}" id="n-${k}" style="left:${x}px;top:${y}px;width:${w}px;height:${h}px"></div>`).join('');
    const chips = Object.entries(CAS.chips).map(([k, [x, y, w, label]]) => `<span class="lchip" id="lc-${k}" style="left:${x}px;top:${y}px;width:${w}px">${label}</span>`).join('');
    $('#cascade').innerHTML = `<svg class="links" width="1832" height="340" viewBox="0 0 1832 340" aria-hidden="true">${grad('litGradC')}${paths}<g id="casPkts"></g></svg>${nodes}${chips}`;
  }
  function jsonHtml(lines, chars) {
    let left = chars, out = [];
    for (const raw of lines) {
      if (left <= 0) break;
      const s = raw.slice(0, left); left -= raw.length + 1;
      out.push(s.replace(/("[^"]*"?)(\s*:)?|(\b\d+(?:\.\d+)?\b)|([^"\d]+)/g, (m, str, colon, num, rest) =>
        str ? `<span class="${colon ? 'k' : 's'}">${esc(str)}</span>${colon || ''}` : num ? `<span class="n">${num}</span>` : esc(rest)));
    }
    return out.join('\n');
  }
  function renderDecisions() {
    const t = st.t, d = decAt(t), R = d ? replayOf(d, t) : null, route = d ? routeOf(d) : null;
    const n = S.decisions.filter((x) => x.t <= t).length;
    put($('#head-decisions'), headHtml(VIEWS[2], [['Decider 12B', '~390 ms'], ['Thresholds', '0.40 / 0.60'], ['Decisions', String(n)]]));
    put($('#cascadeHead'), head('Decision cascade', d ? `${esc(d.title)} · replayed slowly · real time ${d.total} ms` : 'every final caption is checked here'));
    const S_ = (k) => stateIn(R, k);
    for (const k of Object.keys(CAS.nodes)) $('#n-' + k).className = 'node' + (k === 'ignore' ? ' node--pill' : '') + ' is-' + S_(k);
    for (const k of Object.keys(CAS.links)) { const s = S_(k); $('#lk-' + k).setAttribute('class', s === 'active' || s === 'done' ? 'is-lit' : s === 'skip' ? 'is-off' : ''); }
    for (const k of Object.keys(CAS.chips)) $('#lc-' + k).classList.toggle('is-lit', S_(k) === 'done' || S_(k) === 'active');
    const step = (k) => (d ? stepOf(d, k) : null), ms = (k, s) => (S_(s || k) === 'done' && step(k) && step(k).ms !== undefined ? `<span class="ms">${step(k).ms} ms</span>` : '');
    // input
    const c = d && d.from ? capById[d.from] : null;
    let utter;
    if (!d) utter = '<div class="node-h"><h3>Final caption</h3></div><span class="sub">from transcription</span><span class="res">Waiting for the first request.</span>';
    else if (c) utter = `<div class="node-h"><h3>Final caption</h3><span class="ms">${mmss(c.t)}</span></div><div class="utter-who">${avatar(c.who, c.t)}<b>${esc(nameOf(c.who, c.t))}</b><span class="seat-chip">Seat ${PEOPLE[c.who].seat}</span></div><span class="utter-q">“${esc(c.text)}”</span>`;
    else if (d.action === 'resume_notes') utter = `<div class="node-h"><h3>Spoken command</h3><span class="ms">${mmss(d.t)}</span></div><div class="utter-who">${avatar('maya', d.t)}<b>Maya Chen · host</b></div><span class="utter-q">“Okay, we’re back on the record.” <span class="faint">Heard once, not kept.</span></span>`;
    else utter = `<div class="node-h"><h3>Typed in the browser</h3><span class="ms">${mmss(d.t)}</span></div><div class="utter-who"><span class="avatar avatar--neutral">${ic('keyboard')}</span><b>Ask the room</b></div><span class="utter-q">“What did we decide?”</span>`;
    put($('#n-utter'), utter, 'n-utter');
    // rules
    const rs = S_('rules'), rules = step('rules');
    put($('#n-rules'), `<div class="node-h"><h3>Rules</h3>${ms('rules')}</div><span class="sub">privacy &amp; reflex phrases</span><span class="res">${rs === 'done' ? esc(rules.result) : rs === 'active' ? 'Checking phrases…' : 'Checked first, before any model'}</span>`, 'n-rules');
    // decider
    const ds = S_('decider'), lines = d && X.decider[d.id];
    let json;
    if (ds === 'skip') json = '<span class="res">Not consulted · the rules handled it</span>';
    else if (!lines || ds === 'wait') json = `<span class="json faint">{ "tool": …,\n  "confidence": … }</span>`;
    else {
      const total = lines.join('\n').length;
      json = `<span class="json">${jsonHtml(lines, ds === 'done' ? total : Math.floor(R.decider * total))}${ds === 'active' ? '<span class="cur"></span>' : ''}</span>`;
    }
    const tools = d && d.action === 'recap';
    put($('#n-decider'), `<div class="node-h"><h3>${tools ? 'Room tools' : 'Strands Decider 12B'}</h3>${ms('decider')}</div><span class="sub">${tools ? 'list_items · recap' : 'Gemma 4 12B · structured output'}</span>${json}`, 'n-decider');
    put($('#n-ignore'), '&lt; 0.40 · ignore', 'n-ignore');
    // escalation
    const qs = S_('qwen'), qw = step('escalate');
    put($('#n-qwen'), `<div class="node-h"><h3>Qwen3.6 35B</h3>${qs === 'done' && qw && qw.ms ? `<span class="ms">${qw.ms} ms</span>` : ''}</div><span class="sub">${tools ? 'writes the answer' : 'only for 0.40–0.60'}</span>` +
      `<span class="res res--2">${qs === 'done' ? esc(qw.result) : qs === 'active' ? 'Taking a second look…' : qs === 'skip' ? 'Not needed' : 'A second look when unsure'}</span>`, 'n-qwen');
    // policy
    const ps = S_('policy'), pol = step('policy');
    const vcls = { act: 'act', ignore: 'ignore', ask: 'ask', deny: 'deny' }[d ? d.verdict : 'act'];
    put($('#n-policy'), `<div class="node-h"><h3>Policy</h3>${ms('policy')}</div><span class="sub">fixed rules, no model</span>` +
      (ps === 'done' ? `<span class="verdict verdict--${vcls}">${esc(pol.result)}</span>${pol.note ? `<span class="res res--2">${esc(pol.note)}</span>` : ''}`
        : `<span class="res">${ps === 'active' ? 'Checking owner, guests, sensitivity…' : ps === 'skip' ? 'Nothing to check' : 'Act, ask, deny or ignore'}</span>`), 'n-policy');
    // actuator
    const as = S_('actuator');
    put($('#n-actuator'), `<div class="node-h"><h3>Actuator</h3></div><span class="sub">devices, notes, notifier</span><span class="res res--4">${as === 'done' ? esc(step('act').result) : as === 'active' ? 'Carrying it out…' : as === 'skip' ? 'Nothing to do' : 'Does only what policy allows'}</span>`, 'n-actuator');

    // confidence gauge
    const dec = step('decider'), scored = d && route !== 'rule' && !tools && dec.conf !== undefined, shown = scored && (S_('decider') === 'done');
    let sent;
    if (!d) sent = 'Waiting for the first request.';
    else if (route === 'rule') sent = 'Not scored. A privacy phrase is matched by the rules before any model runs.';
    else if (tools) sent = 'Not scored. A typed question goes to the room tools, and Qwen 35B writes the answer from the meeting notes.';
    else if (!shown) sent = 'The Decider is reading the line…';
    else if (route === 'act') sent = `<b>${dec.conf.toFixed(2)}</b> is above 0.60, so the proposed action goes straight to policy.`;
    else if (route === 'escalate') sent = `<b>${dec.conf.toFixed(2)}</b> is between 0.40 and 0.60, so Qwen 35B takes a second look.`;
    else sent = `<b>${dec.conf.toFixed(2)}</b> is below 0.40. No action, nothing sent.`;
    put($('#confCard'), `<div class="card-h">${head('Confidence', d ? esc(d.title) : '')}</div><div class="card-b">` +
      `<div class="gauge-wrap"><div class="gauge"><span class="z1"></span><span class="z2"></span><span class="z3"></span></div>` +
      `<span class="gauge-marker" style="left:${shown ? dec.conf * 100 : 0}%;opacity:${shown ? 1 : 0}"><span class="gauge-val">${shown ? dec.conf.toFixed(2) : ''}</span></span></div>` +
      `<div class="gauge-axis"><span>Ignore</span><span>Escalate</span><span>Act</span></div><div class="gauge-ticks"><span>0</span><span>0.40</span><span>0.60</span><span>1</span></div>` +
      `<span class="gauge-sent">${sent}</span><span class="foot">0.40 and 0.60 are the prototype’s starting thresholds, not values calibrated for the 12B model.</span></div>`);

    // decision stream
    put($('#streamHead'), head('Every decision so far', `<span class="dotk" style="background:${C.green}"></span>act <span class="dotk" style="background:${C.warn}"></span>escalated <span class="dotk" style="background:${C.neutral}"></span>ignored`));
    put($('#streamBody'), streamSvg(t));

    // latency for the current decision
    let lat;
    if (!d) lat = '<span class="wf-empty">Latency for each decision appears here.</span>';
    else {
      const parts = [['rules', 'Rules', C.t3], ['decider', tools ? 'Room tools' : 'Decider 12B', C.blue], ['escalate', 'Qwen 35B', C.warn], ['policy', 'Policy', C.green]];
      const bar = parts.map(([k, , col]) => { const s = step(k); return s && !s.skipped && s.ms ? `<i style="width:${(s.ms / d.total) * 100}%;background:${col}"></i>` : ''; }).join('');
      const rows = parts.map(([k, label]) => { const s = step(k); const skip = !s || s.skipped || !s.ms; return `<div class="lat-row${skip ? ' is-skip' : ''}"><b>${label}</b><span>${skip ? 'skipped' : s.ms + ' ms'}</span></div>`; }).join('');
      let sum = '';
      if (d.action === 'off_record') sum = `<div class="lat-sum"><b>Phrase → off record</b><span><span class="ok">${d.total} ms</span> · target ≤ ${X.targets.privacy * 1000} ms</span></div>`;
      else if (c) {
        const dt = d.t - (c.t + c.dur);
        sum = `<div class="lat-sum"><b>Speech end → result</b><span><span class="ok">${dt.toFixed(1)} s</span>${d.action === 'create_task' ? ' · target ≤ ' + X.targets.card + ' s' : ''}</span></div>`;
      } else sum = `<div class="lat-sum"><b>Total</b><span>${d.total} ms</span></div>`;
      lat = `<div class="lat-bar">${bar}</div><div class="lat-rows">${rows}</div>${sum}`;
    }
    put($('#latCard'), `<div class="card-h">${head('Latency', d ? 'real time · ' + d.total + ' ms' : '')}</div><div class="card-b">${lat}</div>`);
  }
  function streamSvg(t) {
    const W = 720, H = 268, x0 = 64, x1 = 704, yT = 44, yB = 222;
    const xOf = (s) => x0 + (s / S.duration) * (x1 - x0), yOf = (c) => yB - c * (yB - yT), yRule = 16;
    let s = `<svg class="stream-svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Decider confidence for each decision across the meeting">` +
      `<defs><pattern id="hatchS" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="2" height="8" fill="${C.line2}"/></pattern></defs>`;
    s += `<rect x="${x0}" y="${yOf(1)}" width="${x1 - x0}" height="${yOf(0.6) - yOf(1)}" fill="${C.okDeep}"/>`;
    s += `<rect x="${x0}" y="${yOf(0.6)}" width="${x1 - x0}" height="${yOf(0.4) - yOf(0.6)}" fill="${C.warnDeep}"/>`;
    s += `<rect x="${x0}" y="${yOf(0.4)}" width="${x1 - x0}" height="${yB - yOf(0.4)}" fill="${C.s2}"/>`;
    s += `<line x1="${x0}" x2="${x1}" y1="${yRule}" y2="${yRule}" stroke="${C.line2}" stroke-dasharray="3 5"/><text x="${x0 - 10}" y="${yRule + 5}" text-anchor="end">rules</text>`;
    for (const [v, l] of [[1, '1.0'], [0.6, '0.60'], [0.4, '0.40'], [0, '0']]) s += `<text class="mono" x="${x0 - 10}" y="${yOf(v) + 5}" text-anchor="end">${l}</text>`;
    for (const [a, b] of OFF) if (t > a) {
      const xa = xOf(a), xb = xOf(Math.min(t, b));
      s += `<rect x="${xa}" y="4" width="${xb - xa}" height="${yB - 4}" fill="url(#hatchS)"/>`;
      if (xb - xa > 84) s += `<text class="lbl" x="${(xa + xb) / 2}" y="${yOf(0.2)}" text-anchor="middle">off record</text>`;
    }
    for (let m = 0; m <= S.duration; m += 30) s += `<text class="mono" x="${xOf(m)}" y="${H - 16}" text-anchor="middle">${mmss(m)}</text>`;
    s += `<line x1="${xOf(t)}" x2="${xOf(t)}" y1="4" y2="${yB}" stroke="${C.t1}" stroke-width="2"/>`;
    const shown = S.decisions.filter((d) => d.t <= t);
    shown.forEach((d, i) => {
      const r = routeOf(d), dec = stepOf(d, 'decider'), x = xOf(d.t), last = i === shown.length - 1;
      const rule = r === 'rule' || d.action === 'recap', y = rule ? yRule : yOf(dec.conf);
      const fill = rule ? C.t2 : r === 'act' ? C.green : r === 'escalate' ? C.warn : C.neutral;
      if (last) s += `<circle cx="${x}" cy="${y}" r="13" fill="none" stroke="${fill}" stroke-width="2" opacity=".5"/>`;
      s += rule ? `<rect x="${x - 6}" y="${y - 6}" width="12" height="12" transform="rotate(45 ${x} ${y})" fill="${fill}" stroke="${C.s1}" stroke-width="2"/>`
        : `<circle cx="${x}" cy="${y}" r="7" fill="${fill}" stroke="${C.s1}" stroke-width="2"/>`;
      if (last) {
        const right = x < 470, lx = right ? x + 20 : x - 20, ly = rule ? y + 5 : y - 14 < 30 ? y + 26 : y - 14;
        s += `<text class="lbl" x="${lx}" y="${ly}" text-anchor="${right ? 'start' : 'end'}" stroke="${C.s1}" stroke-width="5" paint-order="stroke">${esc(d.title)}</text>`;
      }
    });
    return s + '</svg>';
  }

  // =============================== 4 · Pipeline ===============================
  const FLOW = {
    boxes: { in: [16, 280], per: [341, 400], und: [786, 320], dec: [1151, 330], act: [1526, 290] },
    links: { 'in-hear': 'M296 206C318 206 318 178 341 178', 'in-see': 'M296 206C318 206 318 302 341 302', 'hear-u': 'M741 178C764 178 764 206 786 206',
      'see-u': 'M741 302C764 302 764 206 786 206', 'u-d': 'M1106 206H1151', 'd-a': 'M1481 206H1526', 'a-x': 'M1671 372V412' },
  };
  function buildPipeline() {
    const paths = Object.entries(FLOW.links).map(([k, d]) => `<path id="fl-${k}" d="${d}"/>`).join('');
    const boxes = Object.entries(FLOW.boxes).map(([k, [x, w]]) => `<div class="stagebox" id="sb-${k}" style="left:${x}px;width:${w}px"></div>`).join('');
    const dot = (cls, label) => `<span><svg width="14" height="14" aria-hidden="true"><circle class="pkt pkt--${cls}" cx="7" cy="7" r="5" style="stroke:#fff;stroke-width:2"/></svg>${label}</span>`;
    $('#flow').innerHTML = `<div class="boundary"></div><span class="boundary-tag">${ic('shield')}Stays in the room · ${esc(S.meeting.device)} · no cloud inference</span>` +
      `<svg class="links" width="1832" height="472" viewBox="0 0 1832 472" aria-hidden="true">${grad('litGradF')}${paths}<g id="flowPkts"></g></svg>${boxes}` +
      `<span class="flow-chip" id="egressChip" style="left:1290px;top:418px"></span><div class="discord" id="discordNode"></div>` +
      `<div class="flow-legend">${dot('audio', 'audio chunks')}${dot('frame', 'video frames')}${dot('utter', 'final lines')}${dot('dec', 'decisions')}${dot('out', 'Discord post')}</div>`;
    $('#sb-in').innerHTML = `<h3>Meeting video</h3><span class="sub">${esc(S.fixture.name)}</span><div class="thumb-wrap"><img src="assets/meeting-frame.jpg" width="960" height="540" alt="" /><div class="mask" id="inMask" hidden>${ic('eyeOff')}Masked</div></div>` +
      `<div class="io-row io-clock">${ic('clock')}<b>Media clock</b><span id="inClock"></span></div><div class="io-row">${ic('wave')}<b>Audio</b><span id="inAudio"></span></div><div class="io-row">${ic('film')}<b>Frames</b><span>${CFG.frameRate} fps</span></div>`;
    const mini = (id, b, sub) => `<div class="mini" id="m-${id}"><b>${b}</b><span>${sub}</span><span class="live" id="ml-${id}"></span></div>`;
    const arr = `<span class="arr">${ic('arrow')}</span>`;
    $('#sb-per').innerHTML = `<h3>Perceive</h3><span class="sub">hear and see, in parallel</span>` +
      `<span class="lane-l">Hear</span><div class="lane">${mini('vad', 'VAD', 'Silero')}${arr}${mini('asr', 'ASR', 'Parakeet')}${arr}${mini('voi', 'Voices', 'CAM++')}</div>` +
      `<span class="lane-l">See</span><div class="lane">${mini('fac', 'Faces', 'YuNet')}${arr}${mini('spk', 'Speaker', 'Light-ASD')}</div>`;
  }
  function renderPipeline() {
    const t = st.t, off = offAt(t), D = derive(t), sp = speakingAt(t), lf = lastFinal(t), recentFinal = lf && t - finalAt(lf) < 1.0;
    const models = S.models.filter((m) => m.status === 'ok').length;
    put($('#head-pipeline'), headHtml(VIEWS[3], [['Inference', 'local only'], ['Left the room', D.egress + ' B'], ['Models', models + ' warm']]));
    put($('#flowHead'), head('From meeting video to action', 'one media clock · every box runs on this machine'));
    // input
    $('#inMask').hidden = !off;
    put($('#inClock'), mmssd(t));
    put($('#inAudio'), off ? 'muted' : '16 kHz mono');
    // perceive
    const voicePt = POINTS.some((p) => p.at <= t && t - p.at < 0.8);
    let spk = null;
    for (const id of ORDER) { const v = asd(id, t); if (v !== null && v > 0.5 && (!spk || v > spk.v)) spk = { id, v }; }
    const mini = (id, on, live) => { $('#m-' + id).className = 'mini' + (on ? ' is-on' : ' is-idle'); put($('#ml-' + id), live, 'ml-' + id); };
    mini('vad', !!sp, off ? (sp ? 'resume?' : 'listening') : sp ? 'speech' : 'silence');
    mini('asr', (!!sp && !off) || !!recentFinal, off ? 'paused' : sp ? 'partial' : recentFinal ? 'final' : X.asr.ms + ' ms');
    mini('voi', voicePt && !off, off ? 'paused' : VOICE_ORDER.filter((id) => heardBy(id, t)).length + ' voices');
    mini('fac', !off, off ? 'masked' : '4 faces');
    mini('spk', !!spk, spk ? shortName(spk.id, t) + ' ' + spk.v.toFixed(2) : off ? 'masked' : 'nobody');
    // understand
    let und = '<h3>Attributed utterance</h3><span class="sub">voice + face + seat → who said what</span>';
    if (off) und += '<span class="u-q faint">Off record. Nothing is attributed or kept.</span>';
    else if (!lf) und += '<span class="u-q faint">The first final line will appear here.</span>';
    else {
      const p = PEOPLE[lf.who];
      und += `<div class="u-who">${avatar(lf.who, lf.t)}<b>${esc(nameOf(lf.who, lf.t))}</b><span class="seat-chip">Seat ${p.seat}</span></div><span class="u-q">“${esc(lf.text)}”</span>` +
        `<div class="u-ev"><span class="chip">${ic('wave')}Voice ${voiceNo[lf.who]} · ${simOf(lf).toFixed(2)}</span>${p.face ? `<span class="chip">${ic('face')}Face · ${peakAsd(lf).toFixed(2)}</span>` : ''}<span class="chip mono">${mmss(lf.t)}</span></div>`;
    }
    put($('#sb-und'), und, 'sb-und');
    // decide
    const d = decAt(t), R = d ? replayOf(d, t) : null, rs = (k) => stateIn(R, k);
    const row = (k, label, val) => { const s = rs(k); return `<div class="step-row${s === 'active' ? ' is-on' : s === 'skip' ? ' is-skip' : ''}"><b>${label}</b><span>${s === 'skip' ? 'skipped' : s === 'done' || s === 'active' ? val : '—'}</span></div>`; };
    let decH = '<h3>Decide</h3><span class="sub">rules first, then models</span>';
    if (!d) decH += '<span class="dec-title faint">Waiting for a request</span>' + ['Rules', 'Decider 12B', 'Qwen 35B', 'Policy'].map((l) => `<div class="step-row"><b>${l}</b><span>—</span></div>`).join('');
    else {
      const ds = stepOf(d, 'decider'), es = stepOf(d, 'escalate'), po = stepOf(d, 'policy');
      decH += `<span class="dec-title">${esc(d.title)}</span>` + row('rules', 'Rules', stepOf(d, 'rules').ms + ' ms') +
        row('decider', d.action === 'recap' ? 'Room tools' : 'Decider 12B', ds.conf !== undefined ? ds.conf.toFixed(2) + ' · ' + ds.ms + ' ms' : (ds.ms || 0) + ' ms') +
        row('qwen', 'Qwen 35B', (es.ms || 0) + ' ms') + row('policy', 'Policy', esc(po.result));
    }
    put($('#sb-dec'), decH, 'sb-dec');
    // act
    const recent = (x) => x !== undefined && t - x < 2;
    const fs = Object.values(D.feed);
    const disc = fs.length ? fs.map((f) => f.status).includes('delivered') ? fs.filter((f) => f.status === 'delivered').length + ' delivered' : fs[fs.length - 1].status === 'approval' ? 'needs approval' : 'sending…' : 'nothing yet';
    const arow = (on, label, val) => `<div class="step-row${on ? ' is-on' : ''}"><b>${label}</b><span>${val}</span></div>`;
    put($('#sb-act'), '<h3>Act</h3><span class="sub">only what policy allows</span><span class="dec-title faint">Results</span>' +
      arow(recent(D.last.notes), 'Meeting notes', `${D.tasks.length} tasks · ${D.items.length} decided`) + arow(recent(D.last.lamp), 'Lamp · WiZ', esc(D.devices.lamp.state)) +
      arow(recent(D.last.fan), 'Fan · Kasa', esc(D.devices.fan.state)) + arow(recent(D.last.discord), 'Notifier', disc), 'sb-act');
    const last = fs[fs.length - 1];
    put($('#discordNode'), `${ic('chat')}<div><b>Discord · #launch-team</b><span>${last ? esc(last.detail) : 'Outside the room · nothing sent'}</span></div>`);
    const ec = $('#egressChip');
    ec.classList.toggle('is-out', D.egress > 0);
    put(ec, ic('send') + 'Left the room · ' + D.egress + ' B');
    // link states
    const egressLive = S.events.some((e) => e.type === 'egress' && t >= e.t && t < e.t + 1.4);
    const lit = { 'in-hear': !!sp, 'in-see': !off, 'hear-u': !!recentFinal, 'see-u': !!recentFinal, 'u-d': !!d && t - d.t < 1.2, 'd-a': !!d && d.verdict === 'act' && t - d.t < 1.4, 'a-x': egressLive };
    for (const k of Object.keys(FLOW.links)) $('#fl-' + k).setAttribute('class', lit[k] ? 'is-lit' : '');

    // latency of the latest commitment
    let cm = null;
    for (const x of S.decisions) if (x.t <= t && x.from && x.verdict === 'act' && x.action !== 'off_record') cm = x;
    const AX = 3.5, pc = (v) => (clamp(v, 0, AX) / AX) * 100;
    let wf;
    if (!cm) wf = '<span class="wf-empty">Waiting for the first commitment.</span>';
    else {
      const c = capById[cm.from], end = c.t + c.dur, onScreen = cm.t - end;
      const wrow = (label, a, b, col, target, tlabel, v) => `<div class="wf-row"><b>${label}</b><div class="wf-track">${b !== null ? `<span class="wf-bar" style="left:${pc(a)}%;width:${Math.max(0.6, pc(b) - pc(a))}%;background:${col}"></span>` : ''}` +
        `${target ? `<span class="wf-target" style="left:${pc(target)}%" data-label="${tlabel}"></span>` : ''}</div><span class="v">${v}</span></div>`;
      let out = wrow('Nothing sent outside', 0, null, '', 0, '', '—');
      const task = S.events.find((e) => e.type === 'task' && Math.abs(e.t - cm.t) < 0.05);
      const feed = task && D.feed['f-' + task.id.toLowerCase()];
      const dev = S.events.find((e) => e.type === 'device' && e.t >= cm.t && e.t - cm.t < 2);
      if (feed && feed.status === 'delivered') out = wrow('Discord delivered (measured apart)', onScreen, feed.tu - end, C.neutral, 0, '', (feed.tu - end).toFixed(1) + ' s');
      else if (feed) out = wrow('Discord post', 0, null, '', 0, '', feed.status === 'approval' ? 'awaits approval' : 'sending');
      else if (dev && dev.t <= t) out = wrow('Device confirmed', onScreen, dev.t - end, C.neutral, 0, '', (dev.t - end).toFixed(1) + ' s');
      wf = '<div class="wf">' + wrow('Speech end → final caption', 0, X.asr.lag, C.blue, X.targets.caption, 'p95 ≤ 2 s', X.asr.lag.toFixed(1) + ' s') +
        wrow('Rules + Decider 12B', X.asr.lag, onScreen, C.purple, 0, '', cm.total + ' ms') +
        wrow(cm.action === 'create_task' ? 'Task card on screen' : cm.action === 'record_decision' ? 'Decision in the notes' : 'Result on screen', onScreen - 0.03, onScreen, C.green, cm.action === 'create_task' ? X.targets.card : 0, '≤ 3 s', onScreen.toFixed(1) + ' s') +
        out + `<div class="wf-axis"><span></span><div class="wf-ticks">${[0, 1, 2, 3].map((v) => `<span style="left:${pc(v)}%">${v} s</span>`).join('')}</div><span></span></div></div>`;
      const pr = S.decisions.find((x) => x.action === 'off_record' && x.t <= t);
      if (pr) wf += `<div class="lat-sum wf-priv"><b>Off record indicator</b><span><span class="ok">${pr.total} ms</span> after the phrase · target ≤ ${X.targets.privacy * 1000} ms</span></div>`;
    }
    put($('#waterCard'), `<div class="card-h">${head('Latency of the latest commitment', cm ? esc(cm.title) + ' · sample timings, targets from the build spec' : 'sample timings · targets from the build spec')}</div><div class="card-b">${wf}</div>`);
    // memory budget (planning estimates)
    const M = X.memory;
    const segs = M.parts.map((p) => `<i style="width:${(p.gb / M.total) * 100}%;background:${C[p.tone]}" title="${esc(p.label)}"></i>`).join('') + '<i class="unk" style="width:12%"></i>';
    const leg = M.parts.map((p) => `<span><i class="dotk" style="background:${C[p.tone]}"></i><b>${esc(p.label)}</b><em>≈ ${p.gb} ${p.unit}</em></span>`).join('') +
      `<span><i class="dotk" style="background:${C.s3}"></i><b>OS, perception, browser</b><em>measure</em></span>`;
    put($('#memCard'), `<div class="card-h">${head('Memory budget', `GB10 · ${M.total} GB unified · planning estimates`)}</div><div class="card-b"><div class="mem-bar">${segs}</div><div class="mem-legend">${leg}</div>` +
      '<span class="foot">Estimates from the build spec, not measurements. Measure peak use with recording active before a take.</span></div>');
  }

  // ---- packets: dots that travel along the lit links (decorative, drawn every frame) ----
  const lenCache = {};
  function pointOn(id, u) {
    const p = document.getElementById(id);
    if (!lenCache[id]) lenCache[id] = p.getTotalLength();
    const pt = p.getPointAtLength(u * lenCache[id]);
    return [pt.x.toFixed(1), pt.y.toFixed(1)];
  }
  function drawCascadePackets() {
    const t = st.t, d = decAt(t), R = d ? replayOf(d, t) : null;
    let html = '';
    if (R) for (const k of Object.keys(CAS.links)) if (R[k] > 0 && R[k] < 1) { const [x, y] = pointOn('lk-' + k, R[k]); html += `<circle class="pkt" cx="${x}" cy="${y}" r="7"/>`; }
    put($('#casPkts'), html, 'casPkts');
  }
  function drawFlowPackets() {
    const t = st.t, pk = [];
    const stream = (link, period, travel, cls, cond) => {
      for (let k = Math.floor((t - travel) / period); k <= Math.floor(t / period); k++) {
        const e = k * period, u = (t - e) / travel;
        if (e >= 0 && u >= 0 && u <= 1 && cond(e)) pk.push([link, u, cls]);
      }
    };
    stream('in-hear', 0.25, 0.5, 'audio', (e) => !!speakingAt(e));
    stream('in-see', 0.4, 0.5, 'frame', (e) => !offAt(e));
    for (const c of S.captions) { if (!recorded(c)) continue; const u = (t - finalAt(c)) / 0.6; if (u >= 0 && u <= 1) { pk.push(['hear-u', u, 'utter']); pk.push(['see-u', u, 'frame']); } }
    for (const d of S.decisions) {
      let u = (t - (d.t - 0.5)) / 0.5; if (u >= 0 && u <= 1) pk.push(['u-d', u, 'dec']);
      u = (t - d.t) / 0.6; if (u >= 0 && u <= 1 && d.verdict === 'act') pk.push(['d-a', u, 'dec']);
    }
    for (const e of S.events) if (e.type === 'egress') { const u = (t - e.t) / 1.4; if (u >= 0 && u <= 1) pk.push(['a-x', u, 'out']); }
    put($('#flowPkts'), pk.map(([k, u, cls]) => { const [x, y] = pointOn('fl-' + k, u); return `<circle class="pkt pkt--${cls}" cx="${x}" cy="${y}" r="6"/>`; }).join(''), 'flowPkts');
  }

  // =============================== views, overlay, shortcuts ===============================
  function setView(v, manual) {
    if (manual && st.tour) st.tour = false;
    st.view = v;
    for (const x of VIEWS) $('#view-' + x.id).hidden = x.id !== v;
  }
  const tourView = (t) => { let v = X.tour[0][1]; for (const [at, id] of X.tour) if (t >= at) v = id; return v; };
  function renderOverlay(now) {
    const ov = $('#overlay');
    if (st.capture === 'countdown') {
      ov.hidden = false;
      put(ov, `<div class="countdown">${Math.max(1, Math.ceil((st.countEnd - now) / 1000))}</div><span class="countdown-sub">Playback starts at ${mmss(st.t)} · record with your screen recorder</span>`, 'overlay');
    } else { ov.hidden = true; cache.delete('overlay'); }
  }
  function renderShortcuts() {
    const rows = [['Space', 'Play or pause'], ['⇧R', 'Restart'], ['R', '3-2-1 countdown, then play'], ['V T D P', 'Vision, Transcription, Decisions, Pipeline'],
      ['A', 'Tour: the view follows the story'], ['1–9', 'Jump to a story beat (same as the dashboard)'], ['?', 'This list'], ['Esc', 'Close']];
    $('#shortcuts').innerHTML = `<div class="shortcuts-card"><h2>Keyboard shortcuts</h2><dl>${rows.map(([k, v]) => `<dt><span class="kbd">${k}</span></dt><dd>${v}</dd>`).join('')}</dl></div>`;
  }
  function render(now) {
    renderChrome();
    if (st.view === 'vision') renderVision();
    else if (st.view === 'transcription') renderTranscription();
    else if (st.view === 'decisions') renderDecisions();
    else renderPipeline();
    renderOverlay(now);
  }
  function drawFrame() {
    if (st.view === 'vision') drawSparks();
    else if (st.view === 'transcription') { drawWave(); drawVoiceMap(); drawLanes(); }
    else if (st.view === 'decisions') drawCascadePackets();
    else drawFlowPackets();
  }

  // ---- controls ----
  function restart() { st.t = 0; st.playing = false; }
  function togglePlay() { if (st.capture === 'countdown') return; if (!st.playing && st.t >= S.duration) st.t = 0; st.playing = !st.playing; }
  function countdown(now) {
    if (st.capture === 'countdown') { st.capture = 'idle'; return; }
    if (st.t >= S.duration) st.t = 0;
    st.playing = false; st.capture = 'countdown'; st.countEnd = now + CFG.countdown * 1000;
  }
  function jump(key) { const b = S.beats.find((x) => x.key === key); if (b) { st.t = clamp(b.t, 0, S.duration); st.playing = false; } }

  document.addEventListener('click', (e) => {
    const now = performance.now();
    const tab = e.target.closest('[data-view]'); if (tab) return setView(tab.dataset.view, true);
    const lay = e.target.closest('[data-layer]'); if (lay) { st.layers[lay.dataset.layer] = !st.layers[lay.dataset.layer]; return; }
    if (e.target.closest('#tourBtn')) { st.tour = !st.tour; if (st.tour) setView(tourView(st.t)); return; }
    if (e.target.closest('#playBtn')) return togglePlay();
    if (e.target.closest('#restartBtn')) return restart();
    if (e.target.closest('#speedBtn')) { const i = CFG.speeds.indexOf(st.speed); st.speed = CFG.speeds[(i + 1) % CFG.speeds.length]; return; }
    if (e.target.closest('#keysBtn')) { renderShortcuts(); $('#shortcuts').hidden = !$('#shortcuts').hidden; return; }
    const tl = e.target.closest('#timeline');
    if (tl) { const r = tl.getBoundingClientRect(); st.t = clamp((e.clientX - r.left) / r.width, 0, 1) * S.duration; return; }
    if (e.target.closest('#shortcuts')) $('#shortcuts').hidden = true;
    if (e.target.closest('#overlay') && st.capture === 'countdown') countdown(now);
  });
  document.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key, now = performance.now();
    if (k === ' ') { e.preventDefault(); togglePlay(); }
    else if (k === 'R' && e.shiftKey) restart();
    else if (k === 'r') countdown(now);
    else if (k === 'a') { st.tour = !st.tour; if (st.tour) setView(tourView(st.t)); }
    else if (k === '?') { renderShortcuts(); $('#shortcuts').hidden = !$('#shortcuts').hidden; }
    else if (k === 'Escape') $('#shortcuts').hidden = true;
    else if (/^[1-9]$/.test(k)) jump(k);
    else { const v = VIEWS.find((x) => x.key === k); if (v) setView(v.id, true); }
  });

  // ---- fit the 1920×1080 stage to the window ----
  function fit() { fitScale = Math.min(innerWidth / 1920, innerHeight / 1080); $('#stage').style.setProperty('--fit', fitScale); }
  addEventListener('resize', fit);

  // ---- clock ----
  let prev = performance.now(), lastDraw = 0;
  function frame(now) {
    const dt = Math.min(0.1, (now - prev) / 1000); prev = now;
    if (st.capture === 'countdown' && now >= st.countEnd) { st.capture = 'idle'; st.playing = true; }
    if (st.playing) { st.t = Math.min(S.duration, st.t + dt * st.speed); if (st.t >= S.duration) st.playing = false; }
    if (st.tour) { const v = tourView(st.t); if (v !== st.view) setView(v); }
    if (now - lastDraw >= CFG.renderEvery) { lastDraw = now; render(now); }
    drawFrame();
    requestAnimationFrame(frame);
  }

  readColours();
  buildChrome(); buildVision(); buildDecisions(); buildPipeline();
  fit();
  setView(st.tour ? tourView(st.t) : st.view);
  (document.fonts ? document.fonts.ready : Promise.resolve()).then(() => { render(performance.now()); requestAnimationFrame(frame); });
})();
