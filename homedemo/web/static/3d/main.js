/* Entry point of the 3D house. Subscribes to the page's feed (window.homeFeed, set up by app.js) for
   the same /api/state seed, ui.snapshot frames and bus events the 2D plan uses. Announces itself with
   feed.emit('scene3d', {available}) so app.js can switch the [3D | 2D] toggle; when WebGL2 is missing
   the 2D plan simply stays. Exposes window.__scene3d = {ready, fps(), info(), ...} for tests. */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { Daylight } from './daylight.js';
import { House } from './house.js';
import { Furniture } from './furniture.js';
import { People } from './people.js';
import { Overlays } from './overlays.js';
import { Director } from './director.js';
import { clamp } from './world.js';

const feed = window.homeFeed;
const container = document.getElementById('house3d');
const api = { available: false, ready: false, fps: () => 0, info: () => ({}), reason: '' };
window.__scene3d = api;

function webgl2Available() {
  try {
    const canvas = document.createElement('canvas');
    return !!(window.WebGL2RenderingContext && canvas.getContext('webgl2', { failIfMajorPerformanceCaveat: false }));
  } catch { return false; }
}

function unavailable(reason) {
  api.available = false;
  api.reason = reason;
  console.warn('3D house unavailable:', reason);
  feed?.emit('scene3d', { available: false, reason });
}

if (!feed || !container) {
  unavailable('page scaffold missing');
} else if (!webgl2Available()) {
  unavailable('WebGL2 is not available in this browser');
} else {
  try { start(); } catch (err) { unavailable(`3D failed to start: ${err && err.message || err}`); }
}

function start() {
  THREE.Cache.enabled = true;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const status = container.querySelector('.house3d-status');
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));   // full Retina for the recording
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.toneMappingExposure = 1.2;
  renderer.domElement.className = 'house3d-canvas';
  container.prepend(renderer.domElement);

  const labelRenderer = new CSS2DRenderer();
  labelRenderer.domElement.className = 'house3d-labels';
  container.insertBefore(labelRenderer.domElement, renderer.domElement.nextSibling);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(26, 16 / 10, 0.1, 120);
  camera.position.set(9, 10, 12);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;
  controls.maxPolarAngle = Math.PI / 2 - 0.08;
  controls.minDistance = 2;
  controls.maxDistance = 60;
  controls.target.set(0, 0.4, 0);
  controls.update();

  const house = new House(scene, null);
  const daylight = new Daylight(scene, house.lawn.material);
  const furniture = new Furniture(scene);
  const people = new People(scene, house);
  let settings = null;
  const overlays = new Overlays(scene, house, people, () => settings);
  const director = new Director(camera, controls);
  director.instant = reduced.matches;
  reduced.addEventListener?.('change', () => { director.instant = reduced.matches; });

  const state = { snapshot: null, home: null, seeded: false, visible: false, paused: false, building: false, built: false, snapshots: 0, events: 0, lastEvent: null };
  let last = performance.now(), frames = 0, fpsWindow = last, fps = 0;

  // ---------------------------------------------------------------- sizing
  const dock = container.querySelector('.card-dock');
  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    labelRenderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    const narrow = w < 560;
    container.classList.toggle('narrow', narrow);
    overlays.setDock(narrow ? dock : null);
    director.compact = narrow;
    director.refit();
  }
  new ResizeObserver(() => resize()).observe(container);

  // ---------------------------------------------------------------- HUD
  const bar = document.getElementById('house3d-bar') || container;
  const hud = {
    auto: bar.querySelector('[data-act="auto"]'),
    overview: bar.querySelector('[data-act="overview"]'),
    xray: bar.querySelector('[data-act="xray"]'),
    caption: bar.querySelector('.house3d-caption'),
  };
  function syncHud() {
    if (hud.auto) {
      const on = director.auto && !director.paused;
      hud.auto.setAttribute('aria-pressed', String(on));
      hud.auto.textContent = director.paused ? 'Auto camera: paused' : director.auto ? 'Auto camera' : 'Auto camera: off';
    }
    if (hud.xray) hud.xray.setAttribute('aria-pressed', String(house.xray));
    house.focusRoom = director.mode === 'room' ? director.focusRoom : null;
    container.classList.toggle('overview', director.mode !== 'room');
    if (hud.caption) {
      const room = director.focusRoom ? house.room(director.focusRoom)?.name : null;
      hud.caption.textContent = room ? `Looking at: ${room}` : 'Overview';
    }
  }
  director.onChange = syncHud;
  hud.auto?.addEventListener('click', () => { if (director.paused) director.resume(); else director.setAuto(!director.auto); });
  hud.overview?.addEventListener('click', () => director.resume());
  hud.xray?.addEventListener('click', () => { house.xray = !house.xray; syncHud(); });

  // ---------------------------------------------------------------- data
  async function build(snap) {
    state.building = true;
    if (status) status.textContent = 'Building the 3D house…';
    house.home = state.home;
    house.build(snap);
    director.setRooms(snap.rooms);
    director.overview(true);
    await Promise.all([furniture.build(snap), people.preload(snap)]);
    people.update(snap, true);
    await Promise.all(people.pending);
    house.update(snap);
    daylight.setTime(snap.time);
    daylight.tick(10);
    resize();
    try { await renderer.compileAsync(scene, camera); } catch (err) { console.warn('shader warm-up failed', err); }
    state.built = true;
    state.building = false;
    if (status) status.hidden = true;
    renderer.setAnimationLoop(frame);
    api.ready = true;
    feed.emit('scene3d', { available: true, ready: true });
    syncHud();
    if (state.snapshot && state.snapshot !== snap) onSnapshot(state.snapshot);   // catch up on what arrived meanwhile
  }

  function onSnapshot(snap) {
    if (!snap) return;
    state.snapshot = snap;
    state.snapshots += 1;
    daylight.setTime(snap.time);
    if (!state.built) {
      // the camera list comes from /api/state (home.sensors); build once both have arrived
      if (!state.building && state.seeded) build(snap).catch((err) => unavailable(`3D build failed: ${err && err.message || err}`));
      return;
    }
    if (house.build(snap)) {  // the room/device set changed: rebuild furniture too
      director.setRooms(snap.rooms);
      furniture.build(snap).catch(() => {});
    }
    house.update(snap);
    people.update(snap, state.paused || director.instant);
  }

  const FOCUS_TOPICS = new Set(['sensor.audio', 'intent.proposal', 'policy.verdict', 'notify.ask', 'action.applied', 'sensor.camera']);
  function onEvent(ev) {
    if (!ev || !ev.topic) return;
    state.events += 1;
    state.lastEvent = ev.topic;
    if (ev.topic === 'sim.clock' && ev.data?.time) daylight.setTime(ev.data.time);
    if (state.built) overlays.onEvent(ev);
    if (!FOCUS_TOPICS.has(ev.topic)) return;
    let room = ev.data?.room;
    if (ev.topic === 'sensor.camera' && !(ev.data?.detections || []).length) return;
    if (ev.topic === 'intent.proposal' && ev.data?.by === 'keyword') return;
    if (ev.topic === 'action.applied') room = state.snapshot?.devices.find((d) => d.device_id === ev.data?.device_id)?.room;
    if (room) { director.focus(room); syncHud(); }
  }

  feed.on('state', (st) => {
    state.home = st?.home || null;
    state.seeded = true;
    settings = st?.settings || settings;
    house.home = state.home;
    if (!state.built && !state.building && state.snapshot) onSnapshot(state.snapshot);
  });
  feed.on('settings', (s) => { settings = s; });
  feed.on('snapshot', onSnapshot);
  feed.on('event', onEvent);
  feed.on('motion', ({ paused }) => { state.paused = !!paused; director.motion = !paused; });
  feed.on('view', (v) => { state.visible = v === '3d'; if (state.visible) resize(); });
  feed.on('reset', () => { overlays.clearAll(); director.resume(); });
  feed.on('present', (on) => {   // on camera: cards dwell longer and the room shots leave headroom for them
    overlays.dwell = on ? 1.8 : 1;
    director.headroom = on ? 0.45 : 0;
    requestAnimationFrame(() => { resize(); director.refit(); });
  });

  // ---------------------------------------------------------------- frame loop
  function frame(tMs) {
    const real = clamp((tMs - last) / 1000, 0, 0.1);
    last = tMs;
    frames += 1;
    if (tMs - fpsWindow >= 1000) { fps = frames * 1000 / (tMs - fpsWindow); frames = 0; fpsWindow = tMs; }
    const dt = state.paused ? 0 : real;
    const motion = !state.paused && !director.instant;
    daylight.tick(dt);
    house.tick(dt, camera);
    people.tick(dt, motion);
    overlays.tick(dt, camera, container.clientWidth, container.clientHeight);
    director.tick(dt);
    if (!state.visible) return;
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
  }

  // ---------------------------------------------------------------- test/debug surface
  Object.assign(api, {
    available: true,
    fps: () => Math.round(fps * 10) / 10,
    info: () => ({
      render: { ...renderer.info.render }, memory: { ...renderer.info.memory }, programs: renderer.info.programs?.length ?? 0,
      labels: labelRenderer.domElement.childElementCount, effects: overlays.effects.length, cards: overlays.cards.size,
      snapshots: state.snapshots, events: state.events, lastEvent: state.lastEvent, visible: state.visible,
      people: [...people.people.keys()], pets: [...people.pets.keys()], walls: house.walls.length,
      focus: director.focusRoom, mode: director.mode, time: daylight.time, dayFactor: daylight.factor,
    }),
    focus: (room) => director.focus(room, 20),
    overview: () => director.resume(),
    setXray: (on) => { house.xray = !!on; syncHud(); },
    setAuto: (on) => director.setAuto(!!on),
    renderer, scene, camera, controls, house, people, overlays, director, daylight,
  });
}
