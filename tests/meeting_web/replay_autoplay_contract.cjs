const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));

async function browser(options = {}) {
  const nodes = new Map(), intervals = [], frames = [], actions = [], plays = [];
  let current = {generation: 1, privacy_epoch: 1, privacy: 'on', position: 0,
    state: 'ready', ready: true, unavailable_roles: [], input_mode: 'test',
    integrations: 'preview', attribution: '', media: {url: '/media', duration: 30},
    ...options.state};
  class Target {
    constructor() {
      this.listeners = {}; this.dataset = {}; this.style = {};
      this.classList = {add() {}, contains() {return false;}};
      this.hidden = false; this.value = ''; this.textContent = '';
      this.paused = true; this.readyState = options.readyState ?? 4;
      this.duration = 30; this.currentTime = 0; this.ended = false; this.muted = false;
    }
    addEventListener(type, fn) {(this.listeners[type] ||= []).push(fn);}
    dispatchEvent(event) {for (const fn of this.listeners[event.type] || []) fn(event);}
    prepend() {} append() {} load() {} removeAttribute() {} setAttribute() {}
    matches() {return false;}
    pause() {this.paused = true;}
    play() {
      plays.push(this.muted);
      if (options.rejectPlay) {
        const rejection = options.rejectPlay(this, plays.length);
        if (rejection) return Promise.reject(rejection);
      }
      this.paused = false;
      return Promise.resolve();
    }
  }
  const document = Object.assign(new Target(), {
    body: new Target(), documentElement: new Target(),
    getElementById: id => {if (!nodes.has(id)) nodes.set(id, new Target()); return nodes.get(id);},
    querySelector: () => new Target(),
    createElement: () => Object.assign(new Target(), {firstElementChild: new Target()}),
  });
  document.documentElement.dataset.notesState = current.privacy;
  const context = {document, URLSearchParams, performance: {now: () => 0}, console,
    location: {search: options.search || ''}, addEventListener() {},
    Event: class {constructor(type) {this.type = type;}},
    CustomEvent: class {constructor(type, options) {this.type = type; this.detail = options.detail;}},
    setTimeout, clearTimeout,
    setInterval: fn => intervals.push(fn), requestAnimationFrame: fn => frames.push(fn),
    fetch: async (path, init) => {
      if (init?.method === 'POST' && path === '/api/replay/control') {
        const {action} = JSON.parse(init.body); actions.push(action);
        if (action === 'restart') current = {...current, generation: current.generation + 1, state: 'ready', position: 0};
        if (action === 'play') current = {...current, state: 'playing',
          ...(current.state === 'ended' ? {generation: current.generation + 1, position: 0} : {})};
        if (action === 'pause') current = {...current, state: 'paused'};
      }
      return {ok: true, json: async () => structuredClone(current), text: async () => '<section/>'};
    },
  };
  context.window = context;
  vm.createContext(context);
  vm.runInContext(source, context);
  for (let i = 0; i < 30 && !intervals.length; i++) await tick();
  assert.equal(intervals.length, 1, 'Replay finished initialization');
  return {document, actions, plays, nodes, frames, context,
    node: id => document.getElementById(id),
    poll: intervals[0], state: () => current,
    update: change => {current = {...current, ...change};},
    signal: privacy => document.dispatchEvent(new context.CustomEvent('lococisco:privacy', {detail: privacy})),
    key: (key, code) => document.dispatchEvent({type: 'keydown', key, code, isTrusted: true, preventDefault() {}}),
  };
}

(async () => {
  const ready = await browser();
  assert.deepEqual(ready.actions, ['play']);
  assert.equal(ready.node('replay-video').paused, false);
  for (let i = 0; i < 3; i++) await ready.poll();
  assert.deepEqual(ready.actions, ['play'], 'Polling must not restart a take');
  await ready.node('replay-pause').onclick();
  await ready.poll();
  assert.equal(ready.state().state, 'paused');
  assert.deepEqual(ready.actions, ['play', 'pause'], 'Manual pause stays paused');

  const waiting = await browser({state: {ready: false}, readyState: 0});
  assert.deepEqual(waiting.actions, []);
  waiting.update({ready: true}); await waiting.poll();
  assert.deepEqual(waiting.actions, [], 'Media must be decoded before starting');
  waiting.node('replay-video').readyState = 4; await waiting.poll();
  assert.deepEqual(waiting.actions, ['play']);

  for (const state of [{state: 'paused'}, {privacy: 'paused'}, {privacy: 'off_record'}]) {
    const held = await browser({state}); await held.poll();
    assert.deepEqual(held.actions, [], JSON.stringify(state));
  }
  const ended = await browser({state: {state: 'ended', position: 30}});
  assert.deepEqual(ended.actions, ['play'], 'The server atomically replays a finished demo once');
  assert.equal(ended.state().generation, 2);
  const joining = await browser({state: {state: 'playing', position: 10}});
  joining.frames.shift()(); await tick();
  assert.deepEqual(joining.actions, [], 'Joining preserves the shared take');
  assert.equal(joining.node('replay-video').paused, false);

  let unlocked = false;
  const muted = await browser({rejectPlay: video => !video.muted && !unlocked
    ? Object.assign(new Error('Browser requires activation'), {name: 'NotAllowedError'}) : null});
  assert.deepEqual(muted.actions, ['play'], 'Autoplay blocking must not pause server processing');
  assert.deepEqual(muted.plays, [false, true]);
  assert.match(muted.node('capture-playback-state').textContent, /Playing muted/);
  await muted.poll();
  assert.equal(muted.node('replay-video').muted, true, 'Polling preserves the browser mute fallback');
  muted.signal('off_record');
  muted.key(' ', 'Space'); await tick();
  assert.equal(muted.node('replay-video').muted, true, 'Activation never unmutes private content');
  muted.update({privacy: 'on', privacy_epoch: 2, state: 'playing'});
  muted.signal('on'); await muted.poll();
  unlocked = true; muted.key(' ', 'Space'); await tick();
  assert.equal(muted.node('replay-video').muted, false);
  assert.equal(muted.state().state, 'playing', 'First Space unlocks audio without pausing');
  assert.doesNotMatch(muted.node('capture-playback-state').textContent, /muted/);

  const failure = await browser({rejectPlay: () => Object.assign(new Error('Unsupported media'), {name: 'NotSupportedError'})});
  assert.deepEqual(failure.actions, ['play', 'pause']);
  for (let i = 0; i < 3; i++) {await failure.poll(); failure.frames.shift()();}
  assert.equal(failure.plays.length, 1, 'A decode failure must not loop play attempts');

  const capture = await browser({search: '?autoplay=0'});
  for (let i = 0; i < 3; i++) await capture.poll();
  assert.deepEqual(capture.actions, [], 'Recorder setup stays idle');
  await capture.node('replay-play').onclick();
  assert.deepEqual(capture.actions, ['play'], 'Recorder can explicitly start after connecting audio');

  const external = await browser({state: {ready: false}});
  external.document.dispatchEvent(new external.context.CustomEvent('lococisco:capture', {detail: true}));
  external.update({ready: true}); await external.poll();
  assert.deepEqual(external.actions, [], 'External capture owns startup');
  external.document.dispatchEvent(new external.context.CustomEvent('lococisco:capture', {detail: false}));
  await external.poll();
  assert.deepEqual(external.actions, [], 'Capture completion does not start an unwanted take');

  console.log('Autoplay checks passed: readiness, one start, pause/private preservation, shared take, muted fallback, activation, failure, recorder opt-out.');
})().catch(error => {console.error(error); process.exitCode = 1;});
