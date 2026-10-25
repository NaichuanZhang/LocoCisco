/* What the house hears, sees, thinks and does, drawn from the bus events:
   - sensor.audio: speech bubble at the source, ring pulse at the mic, a ray mic -> source coloured by
     doa.source; TV voices also emit sound rings from the TV.
   - sensor.camera: translucent cone from the camera dome to each detected person, with its label.
   - intent.proposal / policy.verdict / notify.*: a decision card floating above the room (bars with
     the act/ask thresholds, the 2B -> Claude on Bedrock (or 12B) cascade labelled from the payload's
     route/model, keyword chip judged against the outcome, verdict badge); Claude's reply as a bubble
     over the hub.
   - action.command / action.applied: glowing packets sensor -> hub -> device, then a flash at the device.
   Events are indexed by id so cause chains can be followed back to the utterance/scene and the sensor. */
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { roomCenter, toWorld, SOURCE_COLORS, el, words, trunc, makeGlowSprite, UNIT_RING, UNIT_SPHERE, smoothstep } from './world.js';

const ROOT_TOPICS = new Set(['voice.utterance', 'perception.scene']);
const LIFE = { bubble: 9, ray: 6, cone: 7, det: 7.5, detRoutine: 3.5, card: 25, cardAfterVerdict: 14, hubReply: 10 };
const CONE = new THREE.ConeGeometry(0.5, 1, 20, 1, true);
const CONE_WIRE = new THREE.ConeGeometry(0.5, 1, 6, 1, true);
const DOWN = new THREE.Vector3(0, -1, 0);
const fmtLatency = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);
/* Whole percent, but one decimal near a threshold so 0.7955 never reads "80%" beside an 80% line. */
const pct = (p, t) => {
  const v = (Number(p) || 0) * 100, r = Math.round(v);
  return t != null && r === Math.round(t * 100) && v !== t * 100 ? `${Math.floor(v * 10) / 10}%` : `${r}%`;
};
/* Who decided, read from the proposal (route / model / by). app.js shares its table through
   window.homeLabels; the fallback keeps this module usable on its own. */
const LABELS = window.homeLabels || {
  route: (d) => (d && d.route) || (d && d.by === 'bedrock' ? 'bedrock' : d && d.by === 'decider-12b' ? '12b' : 'local'),
  model: (d) => (d && d.model) || (d && d.by) || 'decider',
  short: (d) => (d && d.by === 'decider-12b' ? '12B' : d && d.by === 'decider-2b' ? '2B' : d && d.by === 'bedrock' ? 'Claude' : (d && d.by) || '?'),
  escalation: (st) => ((st && st.escalate_to) === '12b' ? 'the Decider 12B' : 'Claude on Bedrock'),
};

function describeState(st) {
  if (!st) return '';
  if ('on' in st) return st.on ? `on${st.playing ? ' (' + st.playing + ')' : st.brightness ? ' ' + st.brightness + '%' : ''}` : 'off';
  if ('locked' in st) return st.locked ? 'locked' : 'unlocked';
  if ('setpoint_c' in st) return `set to ${Number(st.setpoint_c).toFixed(1)} °C`;
  if ('burner_on' in st) return st.burner_on ? 'burner on' : 'burner off';
  if ('text' in st) return `sent to ${st.to || 'phone'}: "${st.text}"`;
  return JSON.stringify(st);
}

export class Overlays {
  constructor(scene, house, people, getSettings) {
    this.scene = scene;
    this.house = house;
    this.people = people;
    this.getSettings = getSettings;
    this.group = new THREE.Group();
    this.group.name = 'overlays';
    scene.add(this.group);
    this.index = new Map();
    this.effects = [];
    this.cards = new Map();      // root id -> card
    this.roomCards = new Map();  // room id -> root id
    this.dock = null;
    this.dwell = 1;              // presenter mode stretches card lifetimes (set from main.js)
    this.now = 0;
    this.count = 0;
  }

  // ------------------------------------------------------------------ event index helpers
  remember(ev) {
    this.index.set(ev.id, ev);
    if (this.index.size > 2500) { const it = this.index.keys(); for (let i = 0; i < 500; i++) this.index.delete(it.next().value); }
  }
  chainOf(ev) {
    const chain = [];
    let cur = ev, hops = 0;
    while (cur && hops++ < 24) { chain.unshift(cur); cur = cur.cause ? this.index.get(cur.cause) : null; }
    return chain;
  }
  rootOf(ev) {
    const chain = this.chainOf(ev);
    const root = chain.find((e) => ROOT_TOPICS.has(e.topic));
    if (root) return root.id;
    const prop = [...chain].reverse().find((e) => e.topic === 'intent.proposal');
    return prop ? prop.id : null;
  }

  // ------------------------------------------------------------------ dispatch
  onEvent(ev) {
    if (!ev || !ev.topic) return;
    this.remember(ev);
    this.count += 1;
    switch (ev.topic) {
      case 'sensor.audio': return this.onAudio(ev);
      case 'sensor.camera': return this.onCamera(ev);
      case 'intent.proposal': return this.onProposal(ev);
      case 'policy.verdict': return this.onVerdict(ev);
      case 'notify.ask': return this.onAsk(ev);
      case 'notify.reply': return this.onReply(ev);
      case 'action.command': return this.onCommand(ev);
      case 'action.applied': return this.onApplied(ev);
      default: return undefined;
    }
  }

  roomPoint(roomId, y = 1.2) {
    const room = this.house.room(roomId);
    return room ? roomCenter(room, y) : new THREE.Vector3(0, y, 0);
  }

  // ------------------------------------------------------------------ hearing
  onAudio(ev) {
    const a = ev.data || {};
    const src = a.doa?.source || 'unknown';
    const color = SOURCE_COLORS[src] ?? SOURCE_COLORS.unknown;
    const mic = this.house.mics.get(a.mic_id) || this.house.micInRoom(a.room);
    let srcPos, who;
    if (src === 'tv') {
      const tv = this.house.tv();
      srcPos = tv ? tv.screenPos() : this.roomPoint(a.room, 0.9);
      who = 'TV';
      if (tv) this.tvRings(tv, color);
    } else if (src === 'outside') {
      srcPos = this.house.doorFront(0.7)?.setY(1.3) || this.roomPoint(a.room, 1.3);
      who = 'Outside the door';
    } else {
      const id = a.speaker?.id;
      srcPos = (id && this.people.headOf(id)) || this.roomPoint(a.room, 1.5);
      who = a.speaker?.name || 'Unknown voice';
      if (id) this.people.gesture(id, 'emote-yes');
    }
    const bubble = this.label(`bubble3d src-${src}`, srcPos.clone().add(new THREE.Vector3(0, 0.4, 0)), LIFE.bubble,
      el('span', { class: 'who', text: who }), el('span', { class: 'said', text: trunc(a.text || '', 60) }));
    bubble.element.style.setProperty('--src', `#${color.toString(16).padStart(6, '0')}`);
    if (mic) {
      this.ringBurst(mic.pos, color, { count: 3, every: 0.35, from: 0.12, to: 0.7, duration: 1.3 });
      this.ray(mic.pos, srcPos, color, LIFE.ray);
    }
  }

  tvRings(tv, color) {
    const origin = tv.screenPos();
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), tv.facing.clone().normalize());
    this.ringBurst(origin, color, { count: 4, every: 0.4, from: 0.2, to: 1.4, duration: 1.6, quaternion: q, drift: tv.facing.clone().multiplyScalar(0.9) });
  }

  // ------------------------------------------------------------------ seeing
  onCamera(ev) {
    const d = ev.data || {};
    const dets = (d.detections || []).filter((x) => x && x.label !== 'cat' && x.label !== 'dog').slice(0, 3);
    if (!dets.length) return;
    const cam = this.house.cameras.get(d.camera_id) || this.house.cameraInRoom(d.room);
    const from = cam ? cam.pos : this.roomPoint(d.room, 1.05);
    dets.forEach((det, i) => {
      const attrs = det.attrs || {};
      const id = attrs.identity?.id;
      const fallback = this.roomPoint(d.room, 0).add(new THREE.Vector3(-0.6 + i * 0.6, 0, 0.3));
      const feetOf = () => (id && this.people.positionOf(id)) || fallback.clone();
      const headOf = () => ((id && this.people.headOf(id)) || fallback.clone().setY(1.3)).add(new THREE.Vector3(0, 0.95, 0));
      const hazard = attrs.near === 'stove' || /chair/.test(attrs.pose || '');
      const child = attrs.age_group === 'child';
      const color = hazard ? 0xff5a5a : child ? 0xf472b6 : 0x4ade80;
      this.cone(from, () => feetOf().setY(0.55), color, hazard ? 0.45 : 0.26, LIFE.cone);
      const bits = [det.label, attrs.age_group, attrs.near ? 'near ' + attrs.near : null, attrs.pose].filter(Boolean);
      this.label(`det3d ${child ? 'child' : ''} ${hazard ? 'hazard' : ''}`, headOf, hazard || child ? LIFE.det : LIFE.detRoutine,
        el('span', { text: trunc(bits.join(' · '), 40) }));
    });
  }

  // ------------------------------------------------------------------ thinking
  cardFor(root, roomId) {
    let card = this.cards.get(root);
    if (card) return card;
    const prev = this.roomCards.get(roomId);
    if (prev && prev !== root) this.removeCard(prev);
    if (this.dock) for (const other of [...this.cards.keys()]) this.removeCard(other);   // one docked card at a time
    const element = el('div', { class: 'card3d' });
    const room = this.house.room(roomId);
    let obj = null;
    if (this.dock) {
      // narrow canvas: cards stack in a fixed dock instead of floating (where they would clip)
      this.dock.prepend(element);
      this.dock.hidden = false;
    } else {
      obj = new CSS2DObject(element);
      obj.center.set(0.5, 1);   // the card hangs from its bottom edge: it grows upward from the anchor
      obj.position.copy(room ? toWorld(room.x + room.w / 2, room.y + room.h * 0.12, 1.45) : this.roomPoint(roomId, 1.45));
      this.group.add(obj);
    }
    card = { root, roomId, roomName: room?.name || roomId || '', obj, element, main: null, shadow: null, keyword: null, reflex: null, verdict: null, ask: null, reply: null, outcome: null, escalating: false, expireAt: this.now + LIFE.card * this.dwell };
    this.cards.set(root, card);
    this.roomCards.set(roomId, root);
    return card;
  }
  removeCard(root) {
    const card = this.cards.get(root);
    if (!card) return;
    this.cards.delete(root);
    if (this.roomCards.get(card.roomId) === root) this.roomCards.delete(card.roomId);
    card.element.classList.add('fading');
    setTimeout(() => {
      if (card.obj) this.group.remove(card.obj); else card.element.remove();
      if (this.dock && !this.dock.childElementCount) this.dock.hidden = true;
    }, 500);
  }
  /* Where cards go: a DOM dock (narrow canvases) or null for floating CSS2D cards. */
  setDock(dockEl) {
    this.dock = dockEl || null;
    if (this.dock) this.dock.hidden = !this.dock.childElementCount;
  }

  onProposal(ev) {
    const d = ev.data || {};
    const root = this.rootOf(ev) || ev.id;
    const card = this.cardFor(root, d.room);
    if (d.by === 'keyword') card.keyword = d;
    else if (d.by === 'reflex') card.reflex = d;
    else {
      // a shadow answer means the house is still deciding (the 2B was unsure and someone else is asked next)
      card.escalating = d.mode === 'shadow';
      if (d.escalated_from) card.shadow = this.index.get(d.escalated_from)?.data || (card.main?.mode === 'shadow' ? card.main : null);
      else if (!card.escalating) card.shadow = null;
      card.main = d;
      if (d.reply) this.hubSays(d);
    }
    card.expireAt = this.now + LIFE.card * this.dwell;
    this.renderCard(card);
  }

  /* Claude's spoken sentence comes out of the hub speaker: a bubble over the hub, which pulses. */
  hubSays(d) {
    const hub = this.house.hubPos();
    if (!hub) return;
    this.house.pulseHub();
    this.ringBurst(hub, 0x22d3ee, { count: 3, every: 0.3, from: 0.15, to: 0.9, duration: 1.4 });
    const bubble = this.label('bubble3d hub-bubble', hub.clone().add(new THREE.Vector3(0, 0.75, 0)), LIFE.hubReply * this.dwell,
      el('span', { class: 'who', text: `home hub · ${LABELS.short(d)}` }), el('span', { class: 'said', text: trunc(d.reply, 90) }));
    bubble.element.style.setProperty('--src', '#0e7490');
  }

  onVerdict(ev) {
    const d = ev.data || {};
    const prop = this.index.get(d.proposal_id) || (ev.cause && this.index.get(ev.cause));
    const root = (prop && this.rootOf(prop)) || this.rootOf(ev);
    if (!root) return;
    const card = this.cardFor(root, d.room || prop?.data?.room);
    card.verdict = d;
    card.escalating = false;
    card.expireAt = this.now + LIFE.cardAfterVerdict * this.dwell;
    this.renderCard(card);
  }

  onAsk(ev) {
    const root = this.rootOf(ev);
    const card = root && this.cards.get(root);
    if (!card) return;
    card.ask = ev.data; card.expireAt = this.now + 30 * this.dwell;
    this.renderCard(card);
  }
  onReply(ev) {
    const root = this.rootOf(ev);
    const card = root && this.cards.get(root);
    if (!card) return;
    card.reply = ev.data; card.expireAt = this.now + 10 * this.dwell;
    this.renderCard(card);
  }

  /* Same judgement as the 2D table: would the keyword matcher have done the wrong thing? */
  judgeKeyword(card) {
    if (!card.keyword || !card.verdict) return null;
    const kw = card.keyword.action, v = card.verdict.verdict;
    if (v === 'allow') return kw !== card.verdict.action;
    return kw !== 'none';
  }

  renderCard(card) {
    const st = this.getSettings() || {};
    const e = card.element;
    e.replaceChildren();
    const main = card.main;
    const kind = main?.kind || card.reflex?.kind || 'speech';
    e.className = `card3d kind-${kind} ${card.verdict ? 'v-' + card.verdict.verdict : ''}`;
    if (!card.obj && card.roomName) e.append(el('div', { class: 'room', text: card.roomName }));
    const head = el('div', { class: 'head' });
    if (main) {
      head.append(el('span', { class: 'q', text: kind === 'scene' ? 'What should the house do?' : 'What does the speaker want?' }),
        el('span', { class: `model route-${LABELS.route(main)}`, text: `${LABELS.short(main)} · ${fmtLatency(main.latency_ms || 0)}` }));
    } else if (card.reflex) {
      head.append(el('span', { class: 'q', text: 'Reflex rule' }), el('span', { class: 'model rule', text: `${Math.round(card.reflex.latency_ms || 0)} ms` }));
    } else {
      head.append(el('span', { class: 'q', text: 'Decider thinking…' }), el('span', { class: 'model pending', text: '…' }));
    }
    e.append(head);
    if (main) {
      const bars = el('div', { class: 'bars' });
      const barsOf = card.shadow && LABELS.route(main) === 'bedrock' ? card.shadow : main;   // Claude's {action: 1.0} says nothing; show the 2B's spread
      const probs = Object.entries(barsOf.probabilities || {}).sort((a, b) => b[1] - a[1]).slice(0, 3);
      const act = barsOf.threshold ?? st.act_threshold;
      for (const [action, p] of probs) {
        const track = el('div', { class: 'track' }, el('i', { class: 'fill', style: `width:${Math.round(p * 100)}%` }));
        if (st.ask_threshold != null) track.append(el('i', { class: 'th ask', style: `left:${st.ask_threshold * 100}%` }));
        if (act != null) track.append(el('i', { class: 'th act', style: `left:${act * 100}%` }));
        bars.append(el('div', { class: `row ${action === barsOf.action ? 'pick' : ''}` }, el('span', { class: 'lbl', text: words(action) }), track, el('span', { class: 'val', text: pct(p, act) })));
      }
      e.append(bars);
      if (card.escalating) {
        e.append(el('div', { class: 'cascade pending', text: `${LABELS.short(main)} ${pct(main.confidence, act)} → asking ${LABELS.escalation(st)}…` }));
      } else if (card.shadow) {
        e.append(el('div', { class: 'cascade', text: `${LABELS.short(card.shadow)} ${pct(card.shadow.confidence, card.shadow.threshold ?? act)} ${words(card.shadow.action)} → ${LABELS.model(main)}` }));
      } else if (LABELS.route(main) === 'local') {
        e.append(el('div', { class: 'cascade local', text: `${LABELS.model(main)} on the hub · 0 tokens · nothing left the house` }));
      }
      if ((main.tool_calls && main.tool_calls.length) || main.tokens) {
        const calls = (main.tool_calls || []).map((c) => `${c.name}(${Object.entries(c.input || {}).map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`).join(', ')})`);
        const bits = [calls.join(' · ') || words(main.action)];
        if (main.tokens) bits.push(`${((main.tokens.input || 0) + (main.tokens.output || 0)).toLocaleString()} tokens`, fmtLatency(main.latency_ms || 0));
        e.append(el('div', { class: 'tools', text: bits.join(' · ') }));
      }
    } else if (card.reflex) {
      e.append(el('div', { class: 'reflex' }, el('b', { text: words(card.reflex.action) }), ' ', el('span', { text: card.reflex.reason || '' })));
    }
    const foot = el('div', { class: 'foot' });
    if (card.keyword) {
      const wrong = this.judgeKeyword(card);
      foot.append(el('span', { class: `kw ${wrong == null ? '' : wrong ? 'wrong' : 'ok'}`, text: `keyword: ${words(card.keyword.action)}${wrong == null ? '' : wrong ? ' ✗' : ' ✓'}` }));
    }
    if (card.verdict) {
      const v = card.verdict;
      const verb = v.verdict === 'ask_parent' ? 'ask a parent' : v.verdict === 'ask_speaker' ? 'ask the speaker' : v.verdict;
      foot.append(el('span', { class: `badge v-${v.verdict}`, text: `${verb}: ${words(v.rule || '')}` }));
    }
    if (card.ask && !card.reply) foot.append(el('span', { class: 'ask', text: `📱 asking ${card.ask.to}: "${trunc(card.ask.text, 48)}"` }));
    if (card.reply) foot.append(el('span', { class: 'reply', text: `${card.reply.by === 'web' ? 'parent' : card.reply.by} answered ${card.reply.answer}` }));
    if (card.outcome) foot.append(el('span', { class: 'outcome', text: card.outcome }));
    if (foot.childElementCount) e.append(foot);
  }

  // ------------------------------------------------------------------ doing
  originFor(ev) {
    const chain = this.chainOf(ev);
    const audio = chain.find((e) => e.topic === 'sensor.audio');
    if (audio) { const mic = this.house.mics.get(audio.data?.mic_id) || this.house.micInRoom(audio.data?.room); if (mic) return mic.pos.clone(); }
    const cam = chain.find((e) => e.topic === 'sensor.camera');
    if (cam) { const c = this.house.cameras.get(cam.data?.camera_id) || this.house.cameraInRoom(cam.data?.room); if (c) return c.pos.clone(); }
    const scene = chain.find((e) => e.topic === 'perception.scene');
    const room = scene?.data?.room || chain.find((e) => e.data?.room)?.data?.room;
    if (room) {
      const c = this.house.cameraInRoom(room) || this.house.micInRoom(room);
      return c ? c.pos.clone() : this.roomPoint(room, 1.2);
    }
    return null;
  }

  onCommand(ev) {
    const d = ev.data || {};
    const origin = this.originFor(ev);
    const hub = this.house.hubPos();
    const notify = d.command === 'notify';
    let dest = this.house.devicePos(d.device_id);
    if (!dest && notify) {
      const parent = this.people.parentsAtHome()[0];
      dest = parent ? this.people.headOf(parent.id).add(new THREE.Vector3(0, 0.3, 0)) : hub?.clone().add(new THREE.Vector3(0, 1.4, 0));
    }
    if (!dest) return;
    const color = notify ? 0xf472b6 : 0x4ade80;
    const legs = [];
    if (origin && hub) legs.push([origin, hub]);
    else if (origin) legs.push([origin, dest]);
    if (hub) legs.push([hub, dest]);
    if (!legs.length) legs.push([dest.clone().add(new THREE.Vector3(0, 1.5, 0)), dest]);
    let delay = 0;
    for (const [a, b] of legs) {
      const toHub = b === hub;
      this.packet(a, b, color, 0.9, delay, () => { if (toHub) this.house.pulseHub(); else this.flash(b, color); });
      delay += 0.9;
    }
    const root = this.rootOf(ev);
    const card = root && this.cards.get(root);
    if (card && notify) { card.outcome = `alert sent to ${d.args?.to || 'parents'}`; this.renderCard(card); }
  }

  onApplied(ev) {
    const d = ev.data || {};
    const pos = this.house.devicePos(d.device_id);
    const st = d.state || {};
    const color = 'locked' in st ? (st.locked ? 0xef4444 : 0x22c55e) : 'on' in st ? (st.on ? 0xfbbf24 : 0x94a3b8) : 0x60a5fa;
    if (pos) this.flash(pos, color);
    const root = this.rootOf(ev);
    const card = root && this.cards.get(root);
    if (card) { card.outcome = `${d.device_id}: ${describeState(st)}`; card.expireAt = Math.max(card.expireAt, this.now + 8 * this.dwell); this.renderCard(card); }
  }

  // ------------------------------------------------------------------ effect primitives
  label(className, pos, lifetime, ...children) {
    const element = el('div', { class: className }, ...children);
    const obj = new CSS2DObject(element);
    const track = typeof pos === 'function' ? pos : null;
    obj.position.copy(track ? track() : pos);
    this.group.add(obj);
    const dieAt = this.now + lifetime;
    this.effects.push({
      update: (now) => {
        if (track) obj.position.copy(track());
        if (now > dieAt - 0.5) element.classList.add('fading');
        return now < dieAt;
      },
      dispose: () => this.group.remove(obj),
    });
    return obj;
  }

  ringBurst(pos, color, { count = 3, every = 0.35, from = 0.1, to = 0.7, duration = 1.2, quaternion = null, drift = null }) {
    for (let i = 0; i < count; i++) {
      const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false });
      const ring = new THREE.Mesh(UNIT_RING, mat);
      ring.position.copy(pos).add(new THREE.Vector3(0, 0.02, 0));
      if (quaternion) ring.quaternion.copy(quaternion); else ring.rotation.x = -Math.PI / 2;
      ring.scale.setScalar(from);
      ring.visible = false;
      this.group.add(ring);
      const start = this.now + i * every, end = start + duration;
      this.effects.push({
        update: (now) => {
          if (now < start) return true;
          const t = (now - start) / duration;
          ring.visible = true;
          ring.scale.setScalar(from + (to - from) * t);
          mat.opacity = 0.9 * (1 - t);
          if (drift) ring.position.copy(pos).addScaledVector(drift, t);
          return now < end;
        },
        dispose: () => { this.group.remove(ring); mat.dispose(); },
      });
    }
  }

  ray(from, to, color, lifetime) {
    const geom = new THREE.BufferGeometry().setFromPoints([from, to]);
    const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85, depthWrite: false });
    const line = new THREE.Line(geom, mat);
    this.group.add(line);
    const dots = [];
    for (let i = 0; i < 3; i++) {
      const dot = makeGlowSprite(color, 0.28);
      dot.material.opacity = 0.9;
      this.group.add(dot);
      dots.push(dot);
    }
    const dieAt = this.now + lifetime, t0 = this.now, period = 0.9;
    this.effects.push({
      update: (now) => {
        const life = now - t0;
        for (let i = 0; i < dots.length; i++) {
          const t = ((life / period) + i / dots.length) % 1;    // sound travels source -> mic
          dots[i].position.lerpVectors(to, from, t);
        }
        if (now > dieAt - 0.6) { const k = Math.max(0, (dieAt - now) / 0.6); mat.opacity = 0.85 * k; for (const d of dots) d.material.opacity = 0.9 * k; }
        return now < dieAt;
      },
      dispose: () => { this.group.remove(line); for (const d of dots) { this.group.remove(d); d.material.dispose(); } geom.dispose(); mat.dispose(); },
    });
  }

  cone(from, to, color, opacity, lifetime) {
    const track = typeof to === 'function' ? to : null;
    const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false });
    const wireMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: Math.min(1, opacity * 2), wireframe: true, depthWrite: false });
    const mesh = new THREE.Mesh(CONE, mat);
    const wire = new THREE.Mesh(CONE_WIRE, wireMat);
    const aim = (target) => {
      const dir = target.clone().sub(from);
      const len = Math.max(0.3, dir.length());
      mesh.scale.set(0.9, len, 0.9);
      mesh.quaternion.setFromUnitVectors(DOWN, dir.clone().normalize());
      mesh.position.copy(from).addScaledVector(dir, 0.5);
      wire.scale.copy(mesh.scale); wire.quaternion.copy(mesh.quaternion); wire.position.copy(mesh.position);
    };
    aim(track ? track() : to);
    this.group.add(mesh, wire);
    const dieAt = this.now + lifetime, t0 = this.now;
    this.effects.push({
      update: (now) => {
        if (track) aim(track());
        const pulse = 0.75 + 0.25 * Math.sin((now - t0) * 6);
        const fade = now > dieAt - 0.6 ? Math.max(0, (dieAt - now) / 0.6) : 1;
        mat.opacity = opacity * pulse * fade;
        wireMat.opacity = Math.min(1, opacity * 2) * fade;
        return now < dieAt;
      },
      dispose: () => { this.group.remove(mesh, wire); mat.dispose(); wireMat.dispose(); },
    });
  }

  /* A glowing sphere along an arc from a to b; `after` seconds delay; `onArrive` fires once at b. */
  packet(a, b, color, duration, after, onArrive) {
    const mid = a.clone().lerp(b, 0.5);
    mid.y = Math.max(a.y, b.y) + Math.min(1.6, 0.5 + a.distanceTo(b) * 0.3);
    const curve = new THREE.QuadraticBezierCurve3(a.clone(), mid, b.clone());
    const trailGeom = new THREE.BufferGeometry().setFromPoints(curve.getPoints(24));
    const trailMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0, depthWrite: false });
    const trail = new THREE.Line(trailGeom, trailMat);
    const core = new THREE.Mesh(UNIT_SPHERE, new THREE.MeshBasicMaterial({ color: 0xffffff }));
    core.scale.setScalar(0.09);
    const glow = makeGlowSprite(color, 0.55);
    glow.material.opacity = 0.95;
    core.visible = glow.visible = false;
    this.group.add(trail, core, glow);
    const start = this.now + after, end = start + duration, fadeEnd = end + 2.0;
    let arrived = false;
    this.effects.push({
      update: (now) => {
        if (now < start) return true;
        const t = Math.min(1, (now - start) / duration);
        const p = curve.getPoint(smoothstep(t));
        core.visible = glow.visible = t < 1;
        core.position.copy(p); glow.position.copy(p);
        trailMat.opacity = t < 1 ? 0.6 : 0.6 * Math.max(0, (fadeEnd - now) / 2.0);
        if (t >= 1 && !arrived) { arrived = true; onArrive?.(); }
        return now < fadeEnd;
      },
      dispose: () => { this.group.remove(trail, core, glow); trailGeom.dispose(); trailMat.dispose(); core.material.dispose(); glow.material.dispose(); },
    });
  }

  flash(pos, color) {
    this.ringBurst(pos.clone().setY(0.03), color, { count: 2, every: 0.2, from: 0.15, to: 1.3, duration: 1.0 });
    const glow = makeGlowSprite(color, 1.0);
    glow.position.copy(pos);
    glow.material.opacity = 0.9;
    this.group.add(glow);
    const dieAt = this.now + 0.9, t0 = this.now;
    this.effects.push({
      update: (now) => { glow.material.opacity = 0.9 * Math.max(0, 1 - (now - t0) / 0.9); return now < dieAt; },
      dispose: () => { this.group.remove(glow); glow.material.dispose(); },
    });
  }

  // ------------------------------------------------------------------ frame
  tick(dt, camera = null, width = 0, height = 0) {
    this.now += dt;
    this.effects = this.effects.filter((fx) => { const alive = fx.update(this.now); if (!alive) fx.dispose(); return alive; });
    for (const [root, card] of [...this.cards]) if (this.now > card.expireAt) this.removeCard(root);
    if (camera && width) this.layoutCards(camera, width, height);
  }

  /* Floating cards of neighbouring rooms can project on top of each other; nudge them apart
     horizontally (CSS2DRenderer owns `transform`, so the nudge goes through margin-left). A card that
     would grow past the top edge of the canvas is pushed down the same way (margin-top). */
  layoutCards(camera, width, height = 0) {
    const items = [];
    for (const card of this.cards.values()) {
      if (!card.obj) continue;
      const p = card.obj.position.clone().project(camera);
      const w = card.element.offsetWidth || 220;
      items.push({ card, x: (p.x + 1) / 2 * width, y: p.y, w });
      if (height) {
        const top = (1 - p.y) / 2 * height - (card.element.offsetHeight || 0);   // the card hangs from its anchor
        card.element.style.marginTop = top < 6 ? `${Math.round(6 - top)}px` : '0px';
      }
    }
    if (items.length < 2) { for (const it of items) it.card.element.style.marginLeft = '0px'; return; }
    items.sort((a, b) => a.x - b.x);
    const pos = items.map((it) => it.x);
    for (let pass = 0; pass < 3; pass++) {
      for (let i = 1; i < items.length; i++) {
        const a = items[i - 1], b = items[i];
        if (Math.abs(a.y - b.y) > 0.6) continue;           // far apart vertically (NDC): no overlap
        const gap = (a.w + b.w) / 2 + 10 - (pos[i] - pos[i - 1]);
        if (gap > 0) { pos[i - 1] -= gap / 2; pos[i] += gap / 2; }
      }
    }
    items.forEach((it, i) => {
      const x = Math.min(width - it.w / 2 - 4, Math.max(it.w / 2 + 4, pos[i]));
      it.card.element.style.marginLeft = `${Math.round(x - it.x)}px`;
    });
  }

  clearAll() {
    for (const fx of this.effects) fx.dispose();
    this.effects = [];
    for (const root of [...this.cards.keys()]) this.removeCard(root);
  }
}
