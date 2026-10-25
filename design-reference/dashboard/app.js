// LocoCisco presenter dashboard — reference design.
// Plays the sample story in story.js on one media clock. Every panel is derived from the
// clock position (plus anything the presenter does in the browser), so the story can be
// paused, restarted or jumped to any beat and always shows a consistent state.
(function () {
  'use strict';
  const S = window.STORY;
  const $ = (sel) => document.querySelector(sel);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const mmss = (s) => { s = Math.max(0, Math.floor(s)); return String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0'); };

  // ---- presenter settings ----
  const CFG = {
    renderEvery: 50,        // ms between UI refreshes
    preflight: 900,         // ms the recording preflight toast shows before the countdown
    countdown: 3,           // s
    settle: 1.6,            // s to let late results land after the source ends
    asrLag: 0.5,            // s from end of speech to final caption
    liveWords: 16,          // words kept in the on-video caption
    speeds: [1, 2, 4],
  };

  // ---- one outline icon family (24 px grid, 1.8 stroke) ----
  const P = {
    play: 'M8 5.8v12.4a.8.8 0 0 0 1.2.7l10-6.2a.8.8 0 0 0 0-1.4l-10-6.2A.8.8 0 0 0 8 5.8z',
    pause: 'M9 5v14M15 5v14',
    restart: 'M3.5 12a8.5 8.5 0 1 0 2.5-6M3.5 4v4.5H8',
    film: 'M3 6.5A1.5 1.5 0 0 1 4.5 5h15A1.5 1.5 0 0 1 21 6.5v11a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5zM7.5 5v14M16.5 5v14M3 9.5h4.5M3 14.5h4.5M16.5 9.5H21M16.5 14.5H21',
    check: 'M5 12.5l4.5 4.5L19 7.5',
    checkCircle: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM8 12.3l2.7 2.7L16 9.7',
    alert: 'M12 4l9 16H3zM12 10v4M12 17h.01',
    lock: 'M7 11V8a5 5 0 0 1 10 0v3M6 11h12a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1v-8a1 1 0 0 1 1-1z',
    unlock: 'M7 11V8a5 5 0 0 1 9.6-2M6 11h12a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1v-8a1 1 0 0 1 1-1z',
    lamp: 'M9 18h6M10 21h4M12 3a6 6 0 0 0-3.6 10.8c.7.5 1.1 1.3 1.1 2.1v.1h5v-.1c0-.8.4-1.6 1.1-2.1A6 6 0 0 0 12 3z',
    fan: 'M10.4 12a1.6 1.6 0 1 0 3.2 0 1.6 1.6 0 1 0-3.2 0M12 10.4c-.6-3.3.2-6.4 2.6-6.4 2.6 0 2.7 3.6-1.2 6.9M13.4 12.8c3.1 1.1 5.4 3.4 4.2 5.5-1.3 2.2-4.4.4-5.4-4.6M10.6 12.8c-2.5 2.2-5.6 3-6.8.9-1.3-2.2 1.8-4 6.7-2.6',
    chat: 'M5 5h14a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-7l-4.5 3.5V17H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z',
    send: 'M5 12h13M12 5l7 7-7 7',
    download: 'M12 4v11M7 10l5 5 5-5M5 20h14',
    x: 'M6 6l12 12M18 6L6 18',
    eyeOff: 'M3 3l18 18M10.6 6.1A9.6 9.6 0 0 1 12 6c5 0 8.5 4.5 9.5 6-.5.8-1.5 2.1-2.9 3.3M6.6 7.7C4.6 9 3.2 10.9 2.5 12c1 1.5 4.5 6 9.5 6 1.6 0 3-.4 4.3-1.1M9.9 9.9a3 3 0 0 0 4.2 4.2',
    volOff: 'M4 9.5h3.5L12 6v12l-4.5-3.5H4zM16 9.5l5 5M21 9.5l-5 5',
    vol: 'M4 9.5h3.5L12 6v12l-4.5-3.5H4zM16 9a4.5 4.5 0 0 1 0 6M18.5 6.5a8 8 0 0 1 0 11',
    face: 'M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2M9 10h.01M15 10h.01M9.5 15a3.5 3.5 0 0 0 5 0',
    wave: 'M4 10v4M8 7v10M12 4v16M16 8v8M20 11v2',
    chip: 'M7 7h10v10H7zM9.5 3v4M14.5 3v4M9.5 17v4M14.5 17v4M3 9.5h4M3 14.5h4M17 9.5h4M17 14.5h4',
    shield: 'M12 3l7 3v5.5c0 4.3-3 8-7 9.5-4-1.5-7-5.2-7-9.5V6z',
    clock: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 7.5V12l3 2',
    keyboard: 'M3 7a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2zM7 9.5h.5M11 9.5h.5M15 9.5h.5M7 13h.5M16.5 13h.5M8 16h8',
    route: 'M5 6h6a3 3 0 0 1 0 6H9a3 3 0 0 0 0 6h10M16 15l3 3-3 3',
    wifiOff: 'M3 3l18 18M8.5 16.5a5 5 0 0 1 7 0M5 13a10 10 0 0 1 4.2-2.6M19 13a10 10 0 0 0-2.4-1.8M2 9.5a14 14 0 0 1 5-2.9M22 9.5a14 14 0 0 0-9.6-3.4M12 20h.01',
    link: 'M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1',
    pauseCircle: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM10 9v6M14 9v6',
    note: 'M6 4h9l4 4v12H6zM15 4v4h4M9 12h7M9 16h5',
    flag: 'M5 21V4M5 4h11l-2 4 2 4H5',
    arrow: 'M9 6l6 6-6 6',
  };
  const ic = (name, cls = '') => `<svg class="ic ${cls}" viewBox="0 0 24 24" aria-hidden="true"><path d="${P[name]}"/></svg>`;
  const icFill = (name, cls = '') => `<svg class="ic ic--fill ${cls}" viewBox="0 0 24 24" aria-hidden="true"><path d="${P[name]}"/></svg>`;

  // ---- live state ----
  const st = {
    t: 0, playing: false, speed: 1,
    capture: 'idle',          // idle | preflight | countdown | recording | settling | saved
    recWall: 0, countEnd: 0, settleEnd: 0, takeLen: 0, take: 3,
    manual: [],               // things the presenter did in the browser, on the same clock
    drawer: null, selected: null,
    view: 'presenter', offline: false, countdownOn: true,
    armed: { discord: true, lamp: true, fan: true },
  };

  const PEOPLE = S.people;
  const nameOf = (id) => PEOPLE[id].name || PEOPLE[id].label + ' · unnamed';
  const initials = (id) => (PEOPLE[id].name ? PEOPLE[id].name.split(' ').map((w) => w[0]).join('') : '?');

  // ---- derive everything shown at time t ----
  function derive(t) {
    const evs = S.events.concat(st.manual).filter((e) => e.t <= t).sort((a, b) => a.t - b.t);
    const D = {
      checked: new Set(), tasks: [], items: [], feed: [], devices: JSON.parse(JSON.stringify(S.devices)),
      egress: { bytes: 0, msgs: 0 }, notes: 'on', notesAt: 0, notesBy: null, ask: null, drawer: null, select: null, wipes: [], sys: [],
    };
    const notesLog = [{ t: -1, state: 'on' }];
    const taskById = {}, feedById = {};
    for (const e of evs) {
      switch (e.type) {
        case 'checkin': D.checked.add(e.who); break;
        case 'task': { const k = { ...e }; D.tasks.push(k); taskById[e.id] = k; break; }
        case 'task-update': if (taskById[e.id]) Object.assign(taskById[e.id], e, { t: taskById[e.id].t }); break;
        case 'decision-item': D.items.push(e); break;
        case 'feed': { const f = { ...e }; D.feed.push(f); feedById[e.id] = f; break; }
        case 'feed-update': if (feedById[e.id]) Object.assign(feedById[e.id], { status: e.status, detail: e.detail, tu: e.t }); break;
        case 'device': Object.assign(D.devices[e.name], { state: e.state, detail: e.detail, tu: e.t }); break;
        case 'egress': D.egress.bytes += e.bytes; D.egress.msgs += 1; break;
        case 'notes':
          if (e.state === D.notes) break;
          D.notes = e.state; D.notesAt = e.t; D.notesBy = e.by;
          notesLog.push({ t: e.t, state: e.state, wipe: e.wipe, by: e.by });
          if (e.state === 'off') D.wipes.push({ t: e.t, from: e.t - (e.wipe || S.wipeWindow), window: e.wipe || S.wipeWindow });
          else if (e.state === 'on' && notesLog.length > 1) D.sys.push({ t: e.t, kind: 'resume', by: e.by });
          if (e.state === 'paused') D.sys.push({ t: e.t, kind: 'paused', by: e.by });
          break;
        case 'ask': D.ask = { t: e.t, text: e.text, typed: !!e.instant, submitted: false, answered: false, manual: !!e.instant }; break;
        case 'ask-submit': if (D.ask) D.ask.submitted = true; break;
        case 'answer': if (D.ask) { D.ask.answered = true; D.ask.at = e.t; } break;
        case 'drawer': D.drawer = e.open; if (e.select) D.select = e.select; break;
      }
    }
    // Offline: Discord deliveries stay queued (retried later, never duplicated); the local demo is unaffected.
    if (st.offline) {
      for (const f of D.feed) if (f.kind === 'discord' && (f.status === 'delivered' || f.status === 'pending')) { f.status = 'queued'; f.detail = 'Offline · will retry, no duplicate posts'; }
      for (const k of D.tasks) if (k.post === 'delivered' || k.post === 'pending') k.post = 'queued';
    }
    if (D.ask && !D.ask.manual) {
      const sub = S.events.find((e) => e.type === 'ask-submit');
      D.ask.typedChars = D.ask.submitted ? D.ask.text.length : Math.floor(clamp((t - D.ask.t) / (sub.t - D.ask.t), 0, 1) * D.ask.text.length);
    } else if (D.ask) D.ask.typedChars = D.ask.text.length;

    const notesAt = (x) => { let s = 'on'; for (const n of notesLog) if (n.t <= x) s = n.state; return s; };
    // Captions: only those spoken while notes were on, minus anything inside an off-record wipe window.
    D.caps = [];
    D.active = null;
    for (const c of S.captions) {
      if (c.t > t) continue;
      const speaking = t < c.t + c.dur;
      if (speaking && notesAt(t) !== 'off') D.active = c;
      if (c.transient || notesAt(c.t) !== 'on') continue;
      const wiped = D.wipes.find((w) => c.t >= w.from && c.t < w.t);
      if (wiped) { wiped.count = (wiped.count || 0) + 1; continue; }
      const words = c.text.split(' ');
      const p = clamp((t - c.t) / c.dur, 0, 1);
      const shown = Math.max(1, Math.ceil(p * words.length));
      const final = t >= c.t + c.dur + CFG.asrLag;
      D.caps.push({ ...c, words, shown, final });
    }
    if (D.active && D.active.transient) D.active = null;
    // Speakers appear in the roster once they have been heard (named or not).
    D.heard = new Set(S.captions.filter((c) => c.t <= t && !c.transient && notesAt(c.t) !== 'off').map((c) => c.who));
    D.decisions = S.decisions.filter((d) => d.t <= t);
    return D;
  }

  // ---- rendering helpers: only touch the DOM when a panel's markup changes ----
  const cache = new Map();
  function put(node, html, key = node.id) {
    if (cache.get(key) === html) return false;
    cache.set(key, html); node.innerHTML = html; return true;
  }
  const toneOf = (id) => PEOPLE[id].tone;
  const speakingBars = '<span class="speaking" aria-hidden="true"><i></i><i></i><i></i></span>';

  // ---- top bar ----
  function renderTop(D, now) {
    $('#meetingTitle').textContent = S.meeting.title;
    $('#meetingSub').textContent = S.meeting.room + ' · local ' + S.meeting.device.split(' · ').pop();
    put($('#inputChip'), ic('film') + 'Prerecorded meeting · live local inference');
    const np = $('#notesPill');
    const notes = { on: ['pill--on', ic('note'), 'Notes on'], paused: ['pill--paused', ic('pauseCircle'), 'Notes paused'], off: ['pill--off', ic('lock'), 'Off record'] }[D.notes];
    np.className = 'pill ' + notes[0];
    put(np, notes[1] + notes[2]);
    const cp = $('#capturePill');
    let cap;
    switch (st.capture) {
      case 'preflight': cap = ['pill--idle', '<span class="dot"></span>Checking capture…']; break;
      case 'countdown': cap = ['pill--rec', '<span class="dot"></span>Starting in ' + Math.max(1, Math.ceil((st.countEnd - now) / 1000))]; break;
      case 'recording': cap = ['pill--rec', '<span class="dot"></span>Recording <span class="mono">' + mmss((now - st.recWall) / 1000) + '</span>']; break;
      case 'settling': cap = ['pill--rec', '<span class="dot"></span>Finishing results…']; break;
      case 'saved': cap = ['pill--saved', ic('checkCircle') + 'Take saved']; break;
      default: cap = ['pill--idle', '<span class="dot" style="background:transparent;border:1.5px solid currentColor"></span>Not recording'];
    }
    cp.className = 'pill ' + cap[0];
    put(cp, cap[1]);
    put($('#coreState'), ic('checkCircle') + 'Core demo ready');
    $('#coreState').className = 'state state--ok';
    const ds = $('#discordState');
    ds.className = 'state ' + (st.offline ? 'state--warn' : 'state--ok');
    put(ds, st.offline ? ic('wifiOff') + 'Discord offline' : ic('chat') + 'Discord connected');
    const rb = $('#recordBtn');
    const recording = st.capture === 'recording' || st.capture === 'settling';
    rb.classList.toggle('is-stop', recording);
    put(rb, st.capture === 'countdown' || st.capture === 'preflight' ? '<span class="rec-dot"></span>Cancel'
      : recording ? '<span class="rec-dot"></span>Stop and download' : '<span class="rec-dot"></span>Record demo');
    rb.setAttribute('aria-label', recording ? 'Stop recording and download (R)' : 'Record demo (R)');
  }

  // ---- video stage ----
  function renderVideo(D, now) {
    const off = D.notes === 'off';
    let html = '';
    for (const id of Object.keys(PEOPLE)) {
      const p = PEOPLE[id];
      const box = p.face || p.zone;
      const active = D.active && D.active.who === id && !off;
      const label = p.name && D.checked.has(id) ? p.name : p.face ? (p.label || 'Seat ' + p.seat) + ' · unnamed' : 'Face not visible';
      html += `<div class="face${p.face ? '' : ' is-zone'}${active ? ' is-active' : ''}" style="left:${box[0] * 100}%;top:${box[1] * 100}%;width:${box[2] * 100}%;height:${box[3] * 100}%">` +
        `<span class="face-tag"><span class="seat">${p.seat}</span>${esc(label)}${active ? '<span class="tone-' + p.tone + '">' + speakingBars + '</span>' : ''}</span></div>`;
    }
    put($('#faces'), off ? '' : html, 'faces');
    put($('#sourceChip'), ic('film') + esc(S.fixture.name) + ' · ' + S.fixture.video);
    put($('#trackChip'), ic('face') + 'Light-ASD · 4 faces · 5 seats');
    // On-video caption: the line being spoken now, or the last final line for a moment after.
    const lc = $('#liveCaption');
    let cur = D.active && !off ? D.caps.find((c) => c.id === D.active.id) : null;
    if (!cur) { const last = D.caps[D.caps.length - 1]; if (last && st.t - (last.t + last.dur) < 2.2) cur = last; }
    if (cur) {
      const words = cur.words.slice(0, cur.shown);
      const lead = words.length > CFG.liveWords ? '… ' + words.slice(-CFG.liveWords).join(' ') : words.join(' ');
      put(lc, `<span class="who tone-${toneOf(cur.who)}">${esc(nameOf(cur.who))}</span><span class="txt">${esc(lead)}</span>`, 'live');
      lc.classList.remove('is-empty');
    } else lc.classList.add('is-empty');
    // Off record: the source is masked and muted for the recording.
    const mask = $('#videoMask');
    mask.hidden = !off;
    if (off) put(mask, ic('eyeOff', 'ic--xl') + '<h3>Off record</h3><p>Source masked and muted. Nothing is kept until someone says “back on the record”.</p>', 'mask');
    renderOverlay(D, now);
  }

  function renderOverlay(D, now) {
    const ov = $('#overlay');
    let html = '';
    if (st.capture === 'countdown') {
      const n = Math.max(1, Math.ceil((st.countEnd - now) / 1000));
      html = `<div><div class="countdown">${n}</div><span class="countdown-sub">Recording starts with playback</span></div>`;
    } else if (st.capture === 'saved') {
      html = `<div class="overlay-card"><h3>${ic('checkCircle', 'ic--lg tone-green')}Take ${st.take} saved locally</h3>` +
        `<dl class="kv"><dt>File</dt><dd class="mono">lococisco-take-${String(st.take).padStart(2, '0')}.webm</dd><dt>Length</dt><dd>${mmss(st.takeLen)} · 1920×1080</dd><dt>Audio</dt><dd>Meeting audio included</dd></dl>` +
        `<span class="note">Saved on this machine. Nothing is uploaded or posted.</span>` +
        `<div class="row"><button class="btn btn--primary" data-act="download" type="button">${ic('download')}Download</button><button class="btn" data-act="retake" type="button">${ic('restart')}Restart for another take</button></div></div>`;
    } else if (st.t === 0 && !st.playing && st.capture === 'idle') {
      html = `<div class="overlay-card"><h3>${ic('checkCircle', 'ic--lg tone-green')}Ready to record</h3>` +
        `<div class="checks"><div>${ic('check')}Fixture loaded</div><div>${ic('check')}Audio track found</div><div>${ic('check')}Required models warm</div><div>${ic('check')}Runs without internet</div></div>` +
        `<div class="row"><button class="btn btn-record" data-act="record" type="button"><span class="rec-dot"></span>Record demo <span class="kbd" style="color:#fff;border-color:rgba(255,255,255,.5)">R</span></button>` +
        `<button class="btn" data-act="play" type="button">${icFill('play')}Play without recording <span class="kbd">Space</span></button></div></div>`;
    } else if (st.t >= S.duration && st.capture === 'idle') {
      html = `<div class="overlay-card"><h3>${ic('flag', 'ic--lg')}End of meeting video</h3><span class="note">All results have landed. Restart for a clean run, or record a take.</span>` +
        `<div class="row"><button class="btn btn--primary" data-act="retake" type="button">${ic('restart')}Restart <span class="kbd">⇧R</span></button><button class="btn btn-record" data-act="record" type="button"><span class="rec-dot"></span>Record demo</button></div></div>`;
    }
    ov.hidden = !html;
    put(ov, html, 'overlay');
  }

  // ---- transport ----
  function renderTransport(D) {
    const pb = $('#playBtn');
    pb.className = 'icon-btn icon-btn--primary';
    put(pb, st.playing ? ic('pause') : icFill('play'));
    pb.setAttribute('aria-label', st.playing ? 'Pause (Space)' : 'Play (Space)');
    put($('#restartBtn'), ic('restart'));
    put($('#clock'), `<b>${mmss(st.t)}</b> / ${mmss(S.duration)}`);
    $('#tlFill').style.width = (st.t / S.duration) * 100 + '%';
    put($('#tlMarks'), S.beats.map((b) => `<span class="tl-mark${st.t >= b.t ? ' is-past' : ''}" style="left:${(b.t / S.duration) * 100}%" title="${b.key} · ${esc(b.label)}"></span>`).join(''), 'marks');
    let beat = S.beats[0];
    for (const b of S.beats) if (st.t >= b.t) beat = b;
    put($('#beatLabel'), `<b>${beat.key}</b> · ${esc(beat.label)}`);
    const as = $('#audioState');
    const off = D.notes === 'off';
    as.classList.toggle('is-muted', off);
    if (off) put(as, ic('volOff') + 'Muted', 'audio');
    else {
      const talking = D.active && st.playing;
      const bars = [0, 1, 2, 3, 4].map((i) => `<i style="height:${talking ? 3 + Math.round(13 * Math.abs(Math.sin(st.t * 9 + i * 1.7))) : 3}px"></i>`).join('');
      put(as, ic('vol') + `<span class="level" aria-hidden="true">${bars}</span>`, 'audio');
    }
    put($('#speedBtn'), st.speed + '×');
  }

  // ---- room row ----
  function renderDevices(D) {
    const L = D.devices.lamp, F = D.devices.fan;
    const lampOn = L.state !== S.devices.lamp.state, fanOn = F.state === 'On';
    const row = (key, d, on, icn) => `<div class="device"><span class="device-ic${on ? ' is-' + key + '-on' : ''}">${ic(icn, 'ic--lg')}</span>` +
      `<span class="device-main"><b>${d.label}</b><span>${d.vendor}</span></span>` +
      `<span class="device-state"><b>${esc(d.state)}</b>${d.detail ? '<span>' + ic('check') + esc(d.detail) + '</span>' : '<span class="faint">Reported by device</span>'}</span></div>`;
    put($('#devicesCard'), `<div class="card-h"><h2>Room devices</h2><span class="meta">${ic('link')}Local LAN</span></div>` +
      `<div class="card-b">${row('lamp', L, lampOn, 'lamp')}${row('fan', F, fanOn, 'fan')}</div>`);
  }

  function renderPrivacy(D) {
    const b = (state, label) => `<button type="button" data-notes="${state}" class="${D.notes === state ? 'is-on' : ''}" aria-pressed="${D.notes === state}">${label}</button>`;
    const wipe = D.wipes[D.wipes.length - 1];
    put($('#privacyCard'), `<div class="card-h"><h2>Notes &amp; privacy</h2><span class="meta"><span class="kbd">N</span><span class="kbd">O</span></span></div>` +
      `<div class="card-b"><div class="seg" role="group" aria-label="Note-taking">${b('on', 'Notes on')}${b('paused', 'Pause')}${b('off', 'Off record')}</div>` +
      `<div class="priv-lines">` +
      `<div class="priv-line">${ic('lock')}Off record wipes<b>last ${S.wipeWindow} s</b></div>` +
      `<div class="priv-line">${ic('send')}Left the room<b>${D.egress.bytes} B · ${D.egress.msgs} msg</b></div>` +
      `<div class="priv-line">${ic('eyeOff')}${wipe ? 'Last wipe' : 'Source preview'}<b>${wipe ? mmss(wipe.t) + ' · ' + (wipe.count || 0) + ' lines' : 'Unmasked'}</b></div>` +
      `</div></div>`);
  }

  function renderAsk(D) {
    const a = D.ask;
    const asking = a && a.submitted && !a.answered;
    let body = '';
    if (a && a.answered) {
      const lines = D.items.map((i) => i.short).concat(D.tasks.map((k) => k.short));
      body = `<ul class="answer">${(lines.length ? lines : ['Nothing has been decided yet.']).map((l) => `<li>${esc(l)}</li>`).join('')}</ul>`;
    } else if (asking) body = `<span class="thinking">${ic('route')}Checking the room’s notes…</span>`;
    else body = `<span class="ask-hint">Answers come from this meeting’s notes through the room tools.</span>`;
    const val = a ? a.text.slice(0, a.typedChars) : '';
    const html = `<div class="card-h"><h2>Ask the room</h2><span class="meta">${a && a.answered ? esc(S.answerSource.split(' · ').slice(-1)[0]) + ' · browser' : 'Browser · not Discord'}</span></div>` +
      `<div class="card-b"><form class="ask-form" id="askForm" autocomplete="off"><input class="ask-input" id="askInput" aria-label="Ask about this meeting (Q)" placeholder="Ask about this meeting…" value="${esc(val)}" ${a && !a.answered && !a.manual ? 'readonly' : ''}/>` +
      `<button class="ask-send" type="submit" aria-label="Ask">${ic('send')}</button></form>${body}</div>`;
    const card = $('#askCard');
    const focused = document.activeElement && document.activeElement.id === 'askInput';
    if (focused && !(a && !a.manual && !a.answered)) {
      // Don't rebuild the form under the presenter's cursor; only refresh the answer area.
      const key = 'askbody';
      const tail = card.querySelector('.card-b');
      if (tail && cache.get(key) !== body) { cache.set(key, body); tail.querySelectorAll('.answer,.thinking,.ask-hint').forEach((n) => n.remove()); tail.insertAdjacentHTML('beforeend', body); }
      return;
    }
    if (put(card, html, 'ask')) cache.set('askbody', body);
  }

  // ---- participants ----
  function renderPeople(D) {
    const ids = Object.keys(PEOPLE);
    const named = ids.filter((id) => PEOPLE[id].name && D.checked.has(id)).length;
    const rows = ids.map((id) => {
      const p = PEOPLE[id];
      const checked = p.name && D.checked.has(id);
      const heard = D.heard.has(id);
      const speaking = D.active && D.active.who === id && D.notes !== 'off';
      const name = checked ? esc(p.name) : p.face ? esc(p.label || 'Seat ' + p.seat) + ' · unnamed' : 'Face not visible';
      const tone = checked ? p.tone : 'neutral';
      return `<div class="person"><span class="avatar avatar--${tone}">${checked ? initials(id) : '?'}</span>` +
        `<span class="person-name${checked ? '' : ' is-unnamed'}">${name}${speaking ? '<span class="tone-' + tone + '">' + speakingBars + '</span>' : ''}</span>` +
        `<span class="seat-chip">Seat ${p.seat}</span>` +
        `<span class="evidence"><span class="${heard ? 'yes' : ''}" title="Voice">${ic('wave')}</span><span class="${p.face ? 'yes' : ''}" title="Face">${ic('face')}</span></span></div>`;
    }).join('');
    put($('#peopleCard'), `<div class="card-h"><h2>In the room</h2><span class="meta">${named} checked in · ${ids.length - named} unnamed</span></div><div class="card-b"><div class="people">${rows}</div></div>`);
  }

  // ---- captions ----
  function renderCaptions(D) {
    const items = [];
    for (const c of D.caps) items.push({ t: c.t, html: capHtml(c) });
    for (const w of D.wipes) items.push({ t: w.t, html: `<div class="cap-wipe">${ic('lock')}<div><b>Off record at ${mmss(w.t)}</b><span>${w.count || 0} lines from the last ${w.window} s removed on this machine. Messages already delivered are not erased.</span></div></div>` });
    for (const s of D.sys) {
      if (s.kind === 'resume') items.push({ t: s.t, html: `<div class="cap-sys is-ok">${ic('unlock')}<div><b>Back on the record · ${mmss(s.t)}</b><span>${s.by === 'browser' ? 'Resumed from the browser' : 'Resumed by ' + esc(nameOf(s.by)) + ' (host)'}</span></div></div>` });
      if (s.kind === 'paused') items.push({ t: s.t, html: `<div class="cap-sys">${ic('pauseCircle')}<div><b>Notes paused · ${mmss(s.t)}</b><span>Captions aren’t kept while paused.</span></div></div>` });
    }
    if (D.notes === 'off') items.push({ t: 1e9, html: `<div class="cap-sys is-off">${ic('eyeOff')}<div><b>Not keeping notes</b><span>Listening only for “back on the record”.</span></div></div>` });
    items.sort((a, b) => a.t - b.t);
    const log = items.length ? '<div class="cap-spacer"></div>' + items.map((i) => i.html).join('') : '<span class="cap-empty">Captions appear here as people speak.</span>';
    const card = $('#captionsCard');
    if (!card.firstChild) card.innerHTML = `<div class="card-h"><h2>Live captions</h2><span class="meta">${ic('chip')}Parakeet · on this machine</span></div><div class="card-b"><div class="cap-log" id="capLog"></div></div>`;
    const logEl = $('#capLog');
    if (put(logEl, log, 'caplog')) logEl.scrollTop = logEl.scrollHeight;
  }
  function capHtml(c) {
    const p = PEOPLE[c.who];
    const named = !!p.name;
    const said = c.words.slice(0, c.shown).join(' ');
    return `<div class="cap"><div class="cap-h"><span class="who tone-${named ? p.tone : 'neutral'}">${esc(nameOf(c.who))}</span><span class="seat-chip">Seat ${p.seat}</span><span class="ts">${mmss(c.t)}</span></div>` +
      `<p class="cap-t${c.final ? '' : ' interim'}">${esc(said)}</p></div>`;
  }

  // ---- meeting notes ----
  function renderNotes(D) {
    const post = (k) => ({
      delivered: `<span class="post post--delivered">${ic('chat')}Posted to #launch-team</span>`,
      pending: `<span class="post post--pending">${ic('chat')}Posting to Discord…</span>`,
      queued: `<span class="post post--queued">${ic('wifiOff')}Queued · Discord offline</span>`,
      approval: `<span class="post post--approval">${ic('clock')}Discord post waits for ${esc(PEOPLE[k.owner].name.split(' ')[0])}</span>`,
    })[k.post] || '';
    const tasks = D.tasks.slice().sort((a, b) => b.t - a.t).map((k) => `<div class="task"><span class="task-title">${esc(k.title)}</span>` +
      `<div class="task-meta"><span class="owner"><span class="avatar avatar--${toneOf(k.owner)}">${initials(k.owner)}</span>${esc(PEOPLE[k.owner].name)}</span>` +
      `<span class="chip">${ic('clock')}${esc(k.due)}</span></div>` +
      `<span class="quote">“${esc(k.quote)}” <span class="ts">${mmss(k.qt)}</span></span>${post(k)}</div>`).join('');
    const items = D.items.map((i) => `<div class="decision-item">${ic('checkCircle')}<div><b>${esc(i.text)}</b><span>${esc(nameOf(i.who))} · ${mmss(i.qt)}</span></div></div>`).join('');
    put($('#notesCard'), `<div class="card-h"><h2>Meeting notes</h2><span class="meta">${D.tasks.length} task${D.tasks.length === 1 ? '' : 's'} · ${D.items.length} decision${D.items.length === 1 ? '' : 's'}</span></div>` +
      `<div class="card-b"><div class="sec-label">Tasks</div><div class="tasks">${tasks || '<span class="notes-empty">Commitments become tasks here, with an owner and a due date.</span>'}</div>` +
      `<div class="sec-label" style="margin-top:14px">Decisions</div>${items || '<span class="notes-empty">Nothing decided yet.</span>'}</div>`);
  }

  // ---- decisions ----
  const verdictLabel = { act: 'Act', ignore: 'Ignore', ask: 'Ask owner', deny: 'Deny' };
  function quoteOf(d) { const c = S.captions.find((x) => x.id === d.from); return c ? { who: c.who, text: c.text, t: c.t } : { who: null, text: d.quote, t: d.t }; }
  function confBar(conf) {
    return `<div class="conf" role="img" aria-label="Confidence ${conf.toFixed(2)}"><div class="conf-fill" style="width:${conf * 100}%"></div>` +
      `<span class="conf-mark" style="left:40%"></span><span class="conf-mark" style="left:60%"></span></div>`;
  }
  function renderDecision(D) {
    const d = D.decisions[D.decisions.length - 1];
    let body;
    if (!d) body = `<span class="notes-empty">Each decision shows its route, confidence, policy verdict and latency.</span>`;
    else {
      const q = quoteOf(d);
      const dec = d.steps.find((s) => s.k === 'decider');
      const chips = d.steps.filter((s) => s.k === 'rules' || s.k === 'decider' || s.k === 'escalate').map((s) => {
        const name = { rules: 'Rules', decider: s.label.includes('Room') ? 'Room tools' : 'Decider 12B', escalate: s.label.includes('summary') ? 'Qwen 35B' : 'Qwen 35B' }[s.k];
        const cls = s.skipped ? ' is-skip' : s.k === 'rules' ? (s.result.startsWith('No') ? '' : ' chip--warn') : ' chip--accent';
        return `<span class="chip${cls}">${name}${s.conf ? ' · ' + s.conf.toFixed(2) : ''}</span>`;
      }).join(`<span class="arrow">${ic('arrow')}</span>`);
      body = `<span class="utter">${q.who ? '“' + esc(q.text) + '”' : esc(q.text)}</span><div class="route">${chips}</div>` +
        (dec && dec.conf ? confBar(dec.conf) + `<div class="conf-legend"><span>consider 0.40</span><span>act 0.60</span></div>` : '') +
        `<div class="dec-foot"><span class="verdict verdict--${d.verdict}">${verdictLabel[d.verdict]}</span><span class="muted">${esc(d.title)}</span><span class="ms">${d.total} ms</span></div>`;
    }
    put($('#decisionCard'), `<div class="card-h"><h2>Last decision</h2><span class="meta">Details <span class="kbd">D</span></span></div><div class="card-b">${body}</div>`);
  }

  // ---- integrations ----
  function renderFeed(D) {
    const chip = { delivered: ['chip--ok', 'Delivered'], confirmed: ['chip--ok', 'Confirmed'], pending: ['chip--warn', 'Sending'], approval: ['chip--warn', 'Approval'], queued: ['chip--warn', 'Queued'], failed: ['chip--bad', 'Failed'] };
    const rows = D.feed.slice().sort((a, b) => (b.tu || b.t) - (a.tu || a.t)).slice(0, 3).map((f) => {
      const c = chip[f.status] || ['', f.status];
      return `<div class="feed-row"><span class="feed-ic">${ic(f.kind === 'discord' ? 'chat' : f.title.startsWith('Fan') ? 'fan' : 'lamp')}</span>` +
        `<span class="feed-main"><b>${esc(f.title)}</b><span>${esc(f.detail)}</span></span><span class="chip ${c[0]}">${c[1]}</span></div>`;
    }).join('');
    put($('#feedCard'), `<div class="card-h"><h2>Integrations</h2><span class="meta">${st.offline ? ic('wifiOff') + 'Discord offline' : ic('chat') + 'Discord'}</span></div>` +
      `<div class="card-b"><div class="feed">${rows || '<span class="notes-empty">Discord posts and device outcomes appear here once they are confirmed.</span>'}</div></div>`);
  }

  // ---- decision drawer ----
  function renderDrawer(D) {
    const open = (st.drawer !== null ? st.drawer : D.drawer) && D.decisions.length > 0;
    const dr = $('#drawer');
    dr.classList.toggle('is-open', !!open);
    dr.setAttribute('aria-hidden', open ? 'false' : 'true');
    if (!D.decisions.length) return;
    const pick = st.selected || D.select || (D.decisions.find((d) => d.highlight) || D.decisions[D.decisions.length - 1]).id;
    const d = D.decisions.find((x) => x.id === pick) || D.decisions[D.decisions.length - 1];
    const q = quoteOf(d);
    const who = q.who ? `<span class="who tone-${PEOPLE[q.who].name ? toneOf(q.who) : 'neutral'}">${esc(nameOf(q.who))} · Seat ${PEOPLE[q.who].seat} · <span class="mono">${mmss(q.t)}</span></span>` : `<span class="who">Presenter · <span class="mono">${mmss(q.t)}</span></span>`;
    const steps = d.steps.map((s, i) => {
      const hit = !s.skipped;
      return `<div class="step${hit ? ' is-hit' : ' is-skip'}"><span class="step-n">${i + 1}</span><div class="step-main"><b>${esc(s.label)}</b>` +
        (s.model ? `<span class="model">${esc(s.model)}</span>` : '') + `<span class="result">${esc(s.result)}${s.note ? ' — ' + esc(s.note) : ''}</span>` +
        (s.conf ? confBar(s.conf) + `<div class="conf-legend"><span>confidence ${s.conf.toFixed(2)}</span><span>consider ≥ 0.40 · act ≥ 0.60</span></div>` : '') +
        `</div><span class="step-ms">${s.ms !== undefined ? s.ms + ' ms' : '—'}</span></div>`;
    }).join('');
    const egress = d.egress
      ? `<span class="egress-text">${esc(d.egress.text)}</span><span class="muted">${esc(d.egress.dest)} · ${d.egress.bytes} B</span>`
      : `<span class="muted">Nothing left the room for this decision.</span>`;
    const recent = D.decisions.slice().reverse().map((x) => `<button type="button" data-pick="${x.id}" class="${x.id === d.id ? 'is-sel' : ''}"><span class="vdot vdot--${x.verdict}"></span><span class="r-title">${esc(x.title)}</span><span class="r-ts">${mmss(x.t)}</span></button>`).join('');
    put(dr, `<div class="drawer-h"><div><h2>Decision explanation</h2><span class="sub">${esc(d.title)} · ${d.total} ms end to end · measured on this run</span></div><div class="grow"></div>` +
      `<span class="verdict verdict--${d.verdict}">${verdictLabel[d.verdict]}</span><button class="icon-btn" type="button" data-act="close-drawer" aria-label="Close (D)">${ic('x')}</button></div>` +
      `<div class="drawer-b"><div class="drawer-main"><div class="utter-block"><div>${who}${q.who ? '<q>' + esc(q.text) + '</q>' : '<q style="quotes:none">' + esc(q.text) + '</q>'}</div></div><div class="steps">${steps}</div></div>` +
      `<div class="drawer-side"><div class="side-box"><h3>Text sent outside the room</h3>${egress}</div>` +
      `<div class="side-box"><h3>Recent decisions</h3><div class="recent">${recent}</div></div></div></div>`, 'drawer');
  }

  // ---- model health ----
  function renderHealth(D) {
    const chips = S.models.map((m) => `<span class="hchip" title="${esc(m.model)} · ${esc(m.where)}"><span class="hdot hdot--${m.status}"></span><b>${esc(m.role)}</b>${m.value ? '<span>' + esc(m.value) + '</span>' : ''}</span>`).join('');
    put($('#health'), chips + `<div class="health-right"><span class="hchip">${ic('shield')}<b>Local only</b><span>${D.egress.bytes} B out</span></span>` +
      `<span class="sample-chip">Sample data · reference design</span><button class="hchip" type="button" data-act="shortcuts" aria-label="Keyboard shortcuts">${ic('keyboard')}<b>?</b></button></div>`);
  }

  // ---- setup view (before the take; hidden in presenter mode) ----
  function renderSetup() {
    const sw = (key, on, label, sub) => `<div class="toggle-row"><span class="lbl"><b>${label}</b><span>${sub}</span></span><button class="switch" type="button" role="switch" aria-checked="${on}" aria-label="${label}" data-switch="${key}"></button></div>`;
    const models = [
      ['Voice activity', 'Silero VAD', 'Offline kit', 'CPU · in process'],
      ['Speech', 'Parakeet TDT 0.6B v3 · Q8', 'Offline kit', 'GPU · :8770'],
      ['Voices', 'WeSpeaker CAM++', 'Local cache', 'CPU · in process'],
      ['Faces', 'YuNet', 'Local cache', 'CPU · in process'],
      ['Active speaker', 'Light-ASD · AVA', 'Offline kit', 'GPU · in process'],
      ['Decider', 'Strands Decider 12B', 'Offline kit', 'GPU · :8765'],
      ['Escalation', 'Qwen3.6 35B-A3B NVFP4', 'Existing vLLM', ':8000 · auth'],
      ['Speech out', 'MagpieTTS', 'Optional', 'Off'],
      ['Retrieval', 'Bag-of-words index', 'In process', 'CPU'],
    ];
    const rows = models.map((m) => `<tr><td>${m[0]}</td><td class="model">${m[1]}</td><td class="src">${m[2]}</td><td class="src">${m[3]}</td>` +
      `<td>${m[2] === 'Optional' ? '<span class="chip">Disabled</span>' : '<span class="chip chip--ok">' + ic('check') + 'Warm</span>'}</td></tr>`).join('');
    const svc = [['Main vLLM API', '8000', 'Existing service · authenticated'], ['Decider 12B', '8765', 'Loopback'], ['Speech', '8770', 'Loopback'], ['Browser demo', '8780', 'Loopback'], ['Room MCP', '8791', 'Bearer token'], ['MCP TLS proxy', '8443', 'Sandbox bridge only']]
      .map((s) => `<div class="svc"><span class="port">:${s[1]}</span><b>${s[0]}</b><span class="muted">${s[2]}</span></div>`).join('');
    put($('#setup'), `<div class="setup-head"><div class="big">${ic('checkCircle')}<div><h2>Core demo ready</h2><span class="sub">Fixture loaded · required models verified and warm · audio track found · works with internet off</span></div></div>` +
      `<div class="grow"></div><span class="state ${st.offline ? 'state--warn' : 'state--ok'}">${st.offline ? ic('wifiOff') + 'Discord offline — posts will queue' : ic('chat') + 'Discord connected'}</span>` +
      `<button class="btn" type="button" data-act="presenter">Presenter view <span class="kbd">P</span></button><button class="btn btn-record" type="button" data-act="record"><span class="rec-dot"></span>Record demo</button></div>` +
      `<div class="setup-col"><div class="card"><div class="card-h"><h2>Meeting video</h2><span class="meta">Stays on this machine</span></div><div class="card-b"><img class="thumb" src="assets/meeting-frame.jpg" width="960" height="540" alt="First frame of the meeting video"/>` +
      `<dl class="kv"><dt>File</dt><dd><span class="mono">${esc(S.fixture.name)}</span> · ${S.fixture.length}</dd><dt>Video</dt><dd>${S.fixture.video}</dd><dt>Audio</dt><dd>${S.fixture.audio} → 16 kHz mono for speech</dd><dt>Seats</dt><dd>5 zones calibrated · 4 faces tracked</dd></dl></div></div>` +
      `<div class="card fill"><div class="card-h"><h2>Recording preflight</h2><span class="meta">Chrome</span></div><div class="card-b"><div class="checks">` +
      `<div class="check">${ic('checkCircle')}Tab capture with meeting audio<span>Supported</span></div><div class="check">${ic('checkCircle')}Output<span>1920×1080 · WebM (VP9 + Opus)</span></div>` +
      `<div class="check">${ic('checkCircle')}Saved to<span>Local Videos folder · never uploaded</span></div><div class="check is-off">${ic('film')}Fallback<span>OS screen recorder checklist</span></div></div>` +
      sw('countdown', st.countdownOn, 'Countdown before playback', '3 seconds, then capture and the video start together') + `</div></div></div>` +
      `<div class="setup-col"><div class="card fill"><div class="card-h"><h2>Local models</h2><span class="meta">Resolved from local files · no downloads</span></div><div class="card-b">` +
      `<table class="mtable"><thead><tr><th>Role</th><th>Model</th><th>Source</th><th>Runs on</th><th>Status</th></tr></thead><tbody>${rows}</tbody></table>` +
      `<span class="foot-note">Checksums match the offline kit manifest. Hosted inference providers are rejected by the GB10 profile; a missing file stops startup with a named error instead of downloading.</span></div></div></div>` +
      `<div class="setup-col"><div class="card"><div class="card-h"><h2>Services</h2><span class="meta">Host loopback</span></div><div class="card-b">${svc}</div></div>` +
      `<div class="card"><div class="card-h"><h2>Room and Discord</h2></div><div class="card-b"><div class="checks"><div class="check">${ic('checkCircle')}WiZ lamp<span>Reachable · room LAN</span></div><div class="check">${ic('checkCircle')}Kasa fan plug<span>Reachable · room LAN</span></div>` +
      `<div class="check ${st.offline ? 'is-warn' : ''}">${st.offline ? ic('wifiOff') : ic('checkCircle')}Discord bot<span>${st.offline ? 'Offline · separate from core' : 'Connected · separate from core'}</span></div></div>` +
      sw('offline', st.offline, 'Rehearse without internet', 'Discord posts queue; the local demo keeps running') + `</div></div>` +
      `<div class="card fill"><div class="card-h"><h2>External effects</h2><span class="meta">Arm before each take</span></div><div class="card-b">` +
      sw('discord', st.armed.discord, 'Discord test posts', 'Approved tasks post to #launch-team') + sw('lamp', st.armed.lamp, 'Lamp', 'Slides scene on request') + sw('fan', st.armed.fan, 'Fan', 'Comfort requests') +
      `<span class="foot-note">Restart doesn’t undo posts or device changes. Re-arm before another take.</span></div></div></div>`, 'setup');
  }

  function renderShortcuts() {
    const list = [['Space', 'Play / pause'], ['R', 'Record demo / stop and download'], ['⇧ R', 'Restart (clean state)'], ['D', 'Decision explanation'], ['N', 'Pause / resume notes'], ['O', 'Off record'], ['Q', 'Ask the room'], ['1 – 9', 'Jump to a story beat'], ['S / P', 'Setup view / presenter view'], ['H', 'How it works page, at this moment'], ['Esc', 'Close panels']];
    put($('#shortcuts'), `<div class="shortcuts-card"><h2>Keyboard shortcuts</h2><dl>${list.map(([k, v]) => `<dt><span class="kbd">${k}</span></dt><dd>${v}</dd>`).join('')}</dl></div>`, 'shortcuts');
  }

  // ---- toast ----
  let toastTimer = 0;
  function toast(html, ms = 2600) {
    const tEl = $('#toast'); tEl.innerHTML = html; tEl.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { tEl.hidden = true; }, ms);
  }

  // ---- render all ----
  function render(now) {
    const D = derive(st.t);
    renderTop(D, now);
    $('#presenter').hidden = st.view !== 'presenter';
    $('#setup').hidden = st.view !== 'setup';
    if (st.view === 'setup') renderSetup();
    else {
      renderVideo(D, now); renderTransport(D); renderDevices(D); renderPrivacy(D); renderAsk(D);
      renderPeople(D); renderCaptions(D); renderNotes(D); renderDecision(D); renderFeed(D); renderDrawer(D);
    }
    renderHealth(D);
  }

  // ---- actions ----
  function restart() { st.t = 0; st.playing = false; st.manual = []; st.drawer = null; st.selected = null; }
  function play() { if (st.t >= S.duration) restart(); st.playing = true; }
  function togglePlay() { if (st.capture === 'countdown' || st.capture === 'preflight') return; st.playing ? (st.playing = false) : play(); }
  function record(now) {
    if (st.capture === 'recording' || st.capture === 'settling') return stopTake(now);
    if (st.capture === 'countdown' || st.capture === 'preflight') { st.capture = 'idle'; return; }
    st.view = 'presenter';
    st.capture = 'preflight';
    toast(ic('checkCircle') + 'Preflight passed · tab capture with audio · 1920×1080 · WebM', 2200);
    setTimeout(() => {
      if (st.capture !== 'preflight') return;
      restart();
      if (st.countdownOn) { st.capture = 'countdown'; st.countEnd = performance.now() + CFG.countdown * 1000; }
      else startTake(performance.now());
    }, CFG.preflight);
  }
  function startTake(now) { st.capture = 'recording'; st.recWall = now; play(); }
  function stopTake(now) { st.takeLen = (now - st.recWall) / 1000; st.playing = false; st.capture = 'saved'; }
  function setNotes(state) {
    const D = derive(st.t);
    if (D.notes === state) return;
    st.manual.push({ t: st.t + 0.001, type: 'notes', state, by: 'browser', wipe: state === 'off' ? S.wipeWindow : undefined });
  }
  function jump(key, now) {
    const b = S.beats.find((x) => x.key === key);
    if (!b) return;
    st.manual = []; st.drawer = null; st.selected = null; st.playing = false; st.view = 'presenter';
    st.t = clamp(b.t, 0, S.duration);
    if (b.key === '1') st.capture = 'idle';
    else if (b.key === '9') { st.capture = 'saved'; st.takeLen = S.duration + 4; }
    else { st.capture = 'recording'; st.recWall = now - (b.t + CFG.countdown) * 1000; }
  }

  // ---- events ----
  document.addEventListener('click', (e) => {
    const now = performance.now();
    const a = e.target.closest('[data-act]');
    if (a) {
      const act = a.dataset.act;
      if (act === 'record') record(now);
      else if (act === 'play') play();
      else if (act === 'retake') { if (st.capture === 'saved') st.take += 1; restart(); st.capture = 'idle'; }
      else if (act === 'download') toast(ic('download') + 'Reference design: no file is written here. The app saves the WebM take locally.', 3600);
      else if (act === 'close-drawer') st.drawer = false;
      else if (act === 'shortcuts') { const s = $('#shortcuts'); renderShortcuts(); s.hidden = !s.hidden; }
      else if (act === 'presenter') st.view = 'presenter';
      return;
    }
    const n = e.target.closest('[data-notes]'); if (n) return setNotes(n.dataset.notes);
    const pick = e.target.closest('[data-pick]'); if (pick) { st.selected = pick.dataset.pick; st.drawer = true; return; }
    const sw = e.target.closest('[data-switch]');
    if (sw) {
      const k = sw.dataset.switch;
      if (k === 'offline') st.offline = !st.offline;
      else if (k === 'countdown') st.countdownOn = !st.countdownOn;
      else st.armed[k] = !st.armed[k];
      return;
    }
    if (e.target.closest('#decisionCard')) { st.drawer = !(st.drawer !== null ? st.drawer : derive(st.t).drawer); return; }
    if (e.target.closest('#recordBtn')) return record(now);
    if (e.target.closest('#playBtn')) return togglePlay();
    if (e.target.closest('#restartBtn')) { if (st.capture !== 'recording') { restart(); if (st.capture === 'saved') st.capture = 'idle'; } return; }
    if (e.target.closest('#speedBtn')) { const i = CFG.speeds.indexOf(st.speed); st.speed = CFG.speeds[(i + 1) % CFG.speeds.length]; return; }
    const tl = e.target.closest('#timeline');
    if (tl) { const r = tl.getBoundingClientRect(); const x = (e.clientX - r.left) / r.width; const target = clamp(x, 0, 1) * S.duration; let best = S.beats[0]; for (const b of S.beats) if (Math.abs(b.t - target) < Math.abs(best.t - target)) best = b; jump(best.key, now); return; }
    if (e.target.closest('#shortcuts')) $('#shortcuts').hidden = true;
  });
  document.addEventListener('submit', (e) => {
    if (e.target.id !== 'askForm') return;
    e.preventDefault();
    const input = $('#askInput');
    const text = (input.value || '').trim();
    if (!text) return;
    const t0 = st.t;
    st.manual.push({ t: t0, type: 'ask', text, instant: true }, { t: t0 + 0.001, type: 'ask-submit' }, { t: t0 + 1.0, type: 'answer' });
    input.blur();
    if (!st.playing) { st.t = Math.min(S.duration, t0 + 1.0); }
  });
  document.addEventListener('keydown', (e) => {
    const typing = e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA');
    if (typing) { if (e.key === 'Escape') e.target.blur(); return; }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const now = performance.now();
    const k = e.key;
    if (k === ' ') { e.preventDefault(); togglePlay(); }
    else if (k === 'R' && e.shiftKey) { if (st.capture !== 'recording') { restart(); if (st.capture === 'saved') st.capture = 'idle'; } }
    else if (k === 'r') record(now);
    else if (k === 'd') st.drawer = !(st.drawer !== null ? st.drawer : derive(st.t).drawer);
    else if (k === 'n') setNotes(derive(st.t).notes === 'on' ? 'paused' : 'on');
    else if (k === 'o') setNotes('off');
    else if (k === 'q') { e.preventDefault(); st.view = 'presenter'; requestAnimationFrame(() => $('#askInput') && $('#askInput').focus()); }
    else if (k === 's') st.view = st.view === 'setup' ? 'presenter' : 'setup';
    else if (k === 'p') st.view = 'presenter';
    else if (k === 'h') location.href = 'explainer.html?t=' + st.t.toFixed(1);
    else if (k === '?') { renderShortcuts(); $('#shortcuts').hidden = !$('#shortcuts').hidden; }
    else if (k === 'Escape') { st.drawer = false; $('#shortcuts').hidden = true; }
    else if (/^[1-9]$/.test(k)) jump(k, now);
    else if (k === 'Enter' && e.target && e.target.id === 'decisionCard') st.drawer = !(st.drawer !== null ? st.drawer : derive(st.t).drawer);
  });

  // ---- fit the 1920×1080 stage to the window ----
  function fit() { $('#stage').style.setProperty('--fit', Math.min(innerWidth / 1920, innerHeight / 1080)); }
  addEventListener('resize', fit);
  fit();

  // ---- clock ----
  let prev = performance.now(), lastDraw = 0;
  function frame(now) {
    const dt = Math.min(0.1, (now - prev) / 1000); prev = now;
    if (st.capture === 'countdown' && now >= st.countEnd) startTake(now);
    if (st.playing) {
      st.t = Math.min(S.duration, st.t + dt * st.speed);
      if (st.t >= S.duration) {
        st.playing = false;
        if (st.capture === 'recording') { st.capture = 'settling'; st.settleEnd = now + CFG.settle * 1000; }
      }
    }
    if (st.capture === 'settling' && now >= st.settleEnd) stopTake(now);
    if (now - lastDraw >= CFG.renderEvery) { lastDraw = now; render(now); }
    requestAnimationFrame(frame);
  }
  (document.fonts ? document.fonts.ready : Promise.resolve()).then(() => { render(performance.now()); requestAnimationFrame(frame); });
})();
