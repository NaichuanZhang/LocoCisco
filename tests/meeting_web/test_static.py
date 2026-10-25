"""Static checks on the page: it works offline (no remote URL anywhere outside comments), every privacy
state has its text, and the accessibility hooks the brief asks for are present."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from unspoken.meeting.web import server as web

STATIC = Path(web.__file__).resolve().parent / "static"


def _files() -> list[Path]:
    return sorted(f for f in STATIC.rglob("*") if f.is_file())


def _strip_comments(path: Path, text: str) -> str:
    if path.suffix == ".html":
        text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    if path.suffix in (".css", ".js"):
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    if path.suffix == ".js":
        text = re.sub(r"^\s*//.*$", "", text, flags=re.M)
    return text


def test_expected_files_exist():
    assert {f.name for f in _files()} >= {"index.html", "screen.css", "screen.js", "xray.js"}


def test_no_external_requests():
    for f in (p for p in _files() if p.suffix in {'.html', '.css', '.js'}):
        text = _strip_comments(f, f.read_text(encoding="utf-8"))
        assert not re.search(r"https?://", text), f"{f.name} references a remote URL"
        assert "<script src=\"http" not in text and "@import" not in text and "url(http" not in text
        assert "<link" not in text or 'href="static/' in text or 'href="data:,"' in text
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for src in re.findall(r'<script src="([^"]+)"', html) + re.findall(r'<link rel="stylesheet" href="([^"]+)"', html):
        assert src.startswith("static/") and (STATIC / src[len("static/"):]).exists(), src
    js = (STATIC / "screen.js").read_text(encoding="utf-8")
    for url in re.findall(r"fetch\('([^']+)'", js):
        assert url.startswith("api/"), url
    assert "/ws`" in js  # the one stream, same origin


def test_privacy_text_for_every_state():
    js = (STATIC / "screen.js").read_text(encoding="utf-8")
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for text in ("NOTES ON", "PAUSED", "OFF THE RECORD"):
        assert text in js, text
    assert "NOTES ON" in html                                   # the state before the first event is still readable
    for key in ("on:", "paused:", "off_record:"):
        assert key in js
    for icon in ("rec:", "pause:", "off:"):                     # text + icon, never colour alone
        assert icon in js
    css = (STATIC / "screen.css").read_text(encoding="utf-8")
    for cls in (".privacy.state-paused", ".privacy.state-off_record"):
        assert cls in css
    assert "wiped" in js and "Last " in js                      # the "1 min wiped" note


def test_accessibility_hooks():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="privacy"' in html and 'aria-live="assertive"' in html and 'role="status"' in html
    assert 'id="captions"' in html and 'aria-live="polite"' in html
    assert '<html lang="en">' in html
    assert '<span class="b-name">LocoCisco</span>' in html
    assert "<kbd>x</kbd>" in html and "<kbd>p</kbd>" in html and "<kbd>c</kbd>" in html   # shortcuts listed
    css = (STATIC / "screen.css").read_text(encoding="utf-8")
    assert "prefers-reduced-motion" in css and "prefers-color-scheme: dark" in css and "prefers-contrast" in css
    assert ":focus-visible" in css
    js = (STATIC / "screen.js").read_text(encoding="utf-8")
    assert "get('present')" in js and "'x'" in js and "'p'" in js and "'c'" in js
    assert "nothing but text leaves the box" in html.lower()


def test_browser_privacy_clears_hidden_xray_and_stage_and_rejects_old_epochs():
    """Execute the actual JS with a tiny local DOM, without a browser download or network."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute browser cache regression checks")
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Target {
  constructor() { this.listeners = {}; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  dispatchEvent(ev) { for (const fn of this.listeners[ev.type] || []) fn(ev); }
}
class Element extends Target {
  constructor() {
    super(); this.nodeType = 1; this.children = []; this.className = ''; this.attrs = {};
    this._text = ''; this.hidden = false; this.value = ''; this.style = {}; this.selectors = {}; this.dataset = {};
    this.classList = {
      contains: c => this.className.split(' ').includes(c),
      add: c => { this.className += ' ' + c; },
      toggle: (c, on) => {
        this.className = this.className.split(' ').filter(v => v !== c).join(' ');
        if (on) this.className += ' ' + c;
      },
    };
  }
  set textContent(text) { this.children = []; this._text = String(text); }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  set innerHTML(text) { this.textContent = text; }
  get innerHTML() { return this.textContent; }
  get outerHTML() { return this.textContent; }
  append(...children) {
    for (const c of children) {
      const el = c && c.nodeType ? c : Object.assign(new Element(), {textContent: String(c)});
      el.parentElement = this; this.children.push(el);
    }
  }
  prepend(child) { child.parentElement = this; this.children.unshift(child); }
  replaceChildren(...children) { this.children = []; this._text = ''; this.append(...children); }
  get firstElementChild() { return this.children[0]; }
  get lastElementChild() { return this.children.at(-1); }
  get lastChild() { return this.lastElementChild; }
  remove() { this.parentElement.children = this.parentElement.children.filter(c => c !== this); }
  setAttribute(k, v) { this.attrs[k] = v; }
  getAttribute(k) { return this.attrs[k]; }
  removeAttribute(k) { delete this.attrs[k]; }
  querySelector(s) { return this.selectors[s] ||= new Element(); }
  getBoundingClientRect() { return {top: 0, bottom: 1000}; }
  focus() {}
}
const elements = new Map();
global.document = Object.assign(new Target(), {
  body: new Element(), documentElement: new Element(),
  getElementById: id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); },
  createElement: () => new Element(),
  createTextNode: text => Object.assign(new Element(), {textContent: text}),
});
global.window = global;
const windowEvents = new Target();
global.dispatchEvent = ev => windowEvents.dispatchEvent(ev);
global.addEventListener = (t, f) => windowEvents.addEventListener(t, f);
global.matchMedia = () => ({matches: false});
global.CustomEvent = class { constructor(type, options) { this.type = type; this.detail = options.detail; } };
global.location = {protocol: 'http:', host: 'test', search: ''};
global.fetch = () => new Promise(() => {}); // intentionally pending, never a network request
global.setTimeout = () => 1;
global.clearTimeout = () => {};
global.setInterval = () => 1;
let socket;
global.WebSocket = class { constructor() { socket = this; } };
vm.runInThisContext(fs.readFileSync(process.argv[1] + '/xray.js', 'utf8'));
vm.runInThisContext(fs.readFileSync(process.argv[1] + '/screen.js', 'utf8'));
document.dispatchEvent(new Event('DOMContentLoaded'));
const el = id => document.getElementById(id);
const frame = (topic, data, epoch = 0, privacy) => socket.onmessage({
  data: JSON.stringify({topic, data, id: String(Math.random()), ts: 60, wall: Date.now()/1000 + 1,
    stream_epoch: epoch, stream_privacy: privacy}),
});
const snapshot = text => ({
  privacy: {state: 'on'}, screen: {
    captions: [{text, t: 60}], cards: [{card: 'task', text}], banner: text, agenda: {},
  }, items: [{text}], status: {}, participants: [], egress: {}, models: [],
});
frame('ui.snapshot', {snapshot: snapshot('SECRET'), settings: {act_threshold: 0.8}});
frame('meeting.utterance', {utterance_id: 'u', text: 'SECRET', speaker: {name: 'Guest'}});
frame('meeting.proposal', {utterance_id: 'u', action: 'create_task', route: 'rule',
  request: {state: 'SECRET', questions: {}}, reason: 'SECRET'});
frame('trace.log', {agent: 'test', text: 'SECRET'});
assert.equal(XRay._items.length, 1);
assert.ok(el('xr-body').textContent.includes('SECRET'));
XRay.open(); XRay.close();
assert.equal(el('xray').hidden, true);
el('st-reply').textContent = 'SECRET'; el('st-text').value = 'SECRET';
frame('meeting.privacy', {state: 'off_record', wiped_count: 1}, 1, 'off_record');
assert.equal(XRay._items.length, 0);
for (const id of ['xr-body', 'xr-history', 'xr-route', 'captions', 'cards', 'banner', 'st-log', 'st-reply']) {
  assert.ok(!el(id).textContent.includes('SECRET'), id);
}
assert.equal(el('st-text').value, '');
assert.ok(el('wiped-text').textContent.includes('Already-delivered messages are not erased'));
frame('meeting.utterance', {utterance_id: 'private', text: 'SECRET'}, 1, 'off_record');
frame('trace.log', {text: 'SECRET'}, 1, 'off_record');
assert.equal(XRay._items.length, 0);
assert.equal(el('st-log').textContent, '');
// A resume control frame might be evicted from a tiny queue: any newer epoch carries privacy state.
frame('egress.meter', {total_bytes: 20}, 2, 'on');
frame('meeting.utterance', {utterance_id: 'new', text: 'public'}, 2, 'on');
assert.equal(XRay._items.length, 1);
frame('ui.snapshot', {snapshot: snapshot('SECRET')}, 0, 'on');
assert.ok(!el('captions').textContent.includes('SECRET'));
frame('ui.reset', {privacy: 'on'}, 3, 'on');
frame('meeting.utterance', {utterance_id: 'stale', text: 'SECRET'}, 2, 'on');
assert.equal(XRay._items.length, 0);
assert.ok(!el('xr-body').textContent.includes('SECRET'));
frame('meeting.utterance', {utterance_id: 'fresh', text: 'public'}, 3, 'on');
frame('trace.log', {text: 'public'}, 3, 'on');
document.dispatchEvent(new Event('lococisco:reset'));
assert.equal(XRay._items.length, 0);
assert.equal(el('xr-history').textContent, '');
assert.equal(el('st-log').textContent, '');
assert.equal(el('xr-route').textContent, '');
socket.onopen(); // a restarted server's first epoch is zero, not the previous stream's three
frame('ui.snapshot', {snapshot: snapshot('new session'), settings: {act_threshold: 0.8}});
frame('meeting.utterance', {utterance_id: 'new-session', text: 'new session'});
assert.equal(XRay._items.length, 1);
"""
    result = subprocess.run([node, "-e", script, str(STATIC)], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def test_replay_discards_responses_across_privacy_and_restart():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for the replay response-race check")
    script = r"""
const assert = require('node:assert/strict'), fs = require('node:fs'), vm = require('node:vm');
class Target {
  constructor(){this.listeners={};this.dataset={};this.classList={add(){},contains(){return false;}};
    this.hidden=false;this.value='';this.textContent='';this.paused=true;this.readyState=4;this.duration=180;this.currentTime=0;}
  addEventListener(t,f){(this.listeners[t] ||= []).push(f);}
  dispatchEvent(e){for(const f of this.listeners[e.type] || [])f(e);}
  prepend(){} append(){} load(){} pause(){this.paused=true;} play(){this.paused=false;return Promise.resolve();}
  removeAttribute(){} matches(){return false;}
}
const nodes=new Map(), intervals=[];
global.window=global;global.addEventListener=()=>{};
global.document=Object.assign(new Target(),{
  body:new Target(),documentElement:new Target(),
  getElementById:id=>{if(!nodes.has(id))nodes.set(id,new Target());return nodes.get(id);},
  querySelector:()=>new Target(),createElement:()=>Object.assign(new Target(),{firstElementChild:new Target()}),
});
global.CustomEvent=class {constructor(type,options){this.type=type;this.detail=options.detail;}};
global.setInterval=fn=>{intervals.push(fn);};global.requestAnimationFrame=()=>{};
let current={generation:1,privacy_epoch:1,privacy:'on',media:null,state:'ready',position:0,
  ready:true,unavailable_roles:[],input_mode:'test',integrations:'preview',attribution:''};
const response=data=>({ok:true,json:async()=>structuredClone(data),text:async()=>'<section/>'});
global.fetch=async path=>response(current);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const signal=value=>document.dispatchEvent(new CustomEvent('lococisco:privacy',{detail:value}));
(async()=>{
  vm.runInThisContext(fs.readFileSync(process.argv[1]+'/replay.js','utf8'));
  for(let i=0;i<20&&!intervals.length;i++)await tick();
  assert.equal(intervals.length,1);
  const node=id=>document.getElementById(id), poll=intervals[0];
  assert.equal(node('replay-video').muted,false);
  let release;
  fetch=()=>new Promise(resolve=>{release=resolve;});
  const pending=poll();
  signal('off_record');
  release(response(current));await pending;
  assert.equal(node('replay-video').muted,true,'a stale on-record poll must not unmute');
  assert.equal(node('replay-privacy').hidden,false);
  current={...current,privacy_epoch:2,privacy:'off_record'};
  fetch=async()=>response(current);await poll();
  signal('on');current={...current,privacy_epoch:3,privacy:'on'};await poll();
  node('replay-question').value='Question';
  fetch=()=>new Promise(resolve=>{release=resolve;});
  const answer=node('replay-ask').onclick();
  signal('off_record');signal('on');
  release(response({text:'SECRET ANSWER'}));await answer;
  assert.ok(!node('replay-answer').textContent.includes('SECRET'));
  // A fresh poll can advance the generation, but a delayed response cannot roll it back.
  current={...current,generation:2,privacy_epoch:6,state:'ready'};
  fetch=async()=>response(current);await poll();
  fetch=async()=>response({...current,generation:1,privacy_epoch:5,state:'playing'});
  await poll();assert.equal(node('replay-status').textContent,'Local models ready · ready');
  // A reset invalidates a pending answer even when privacy remains on.
  fetch=()=>new Promise(resolve=>{release=resolve;});
  const oldAnswer=node('replay-ask').onclick();
  document.dispatchEvent(new Event('lococisco:reset'));
  release(response({text:'OLD TAKE'}));await oldAnswer;
  assert.ok(!node('replay-answer').textContent.includes('OLD TAKE'));
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run([node, "-e", script, str(STATIC)], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
