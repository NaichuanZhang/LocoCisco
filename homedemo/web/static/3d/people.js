/* People and the cat. Positions come from snapshot.people[].x/y and snapshot.pets[].x/y; characters
   ease along the floor toward their snapshot spot (walk animation while moving, facing the way they
   go), then idle, sit or interact depending on the activity text. Labels are CSS2D elements. */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { PERSON_SCALE, CHILD_SCALE, CAT_HEIGHT, toWorld, el, trunc, approach, clamp } from './world.js';

export const PEOPLE_DIR = 'static/assets/people/';
export const PETS_DIR = 'static/assets/pets/';
export const PERSON_MODELS = {
  alex: 'character-male-a.glb', sam: 'character-female-b.glb', maya: 'character-female-f.glb',
  child: 'character-female-f.glb', guest: 'character-male-d.glb',
};
export const CAT_MODEL = 'animal-cat.glb';

const WALK_SPEED = 0.625 * PERSON_SCALE;   // m/s; keeps stride length per step as the people are rescaled
const SIT_RE = /couch|sofa|sit|seated|relax|watch|lounging|reading/i;
const INTERACT_RE = /cook|stove|reach|snack|looking for|lifting|search|hunt|putting on|opening|grab|fridge/i;
const SLEEP_RE = /asleep|sleep|nap|doz|snor|bed/i;
const STILL_RE = /sitting still|idle|standing still/i;
const ZZZ_AFTER_S = 25;

function modelFor(p) {
  return PERSON_MODELS[p.id] || (p.role === 'child' ? PERSON_MODELS.child : PERSON_MODELS.guest);
}

/* Bounds of a loaded model in its parent's space. Skinned meshes use their geometry bounds under the
   root transform (SkinnedMesh.boundingBox over-reports for these rigs); static parts use world bounds. */
function measure(root) {
  root.updateMatrixWorld(true);
  const box = new THREE.Box3();
  const tmp = new THREE.Box3();
  root.traverse((o) => {
    if (!o.isMesh) return;
    if (!o.geometry.boundingBox) o.geometry.computeBoundingBox();
    tmp.copy(o.geometry.boundingBox).applyMatrix4(o.isSkinnedMesh ? root.matrixWorld : o.matrixWorld);
    box.union(tmp);
  });
  return box;
}

class Actor {
  constructor(scene, id) {
    this.scene = scene;
    this.id = id;
    this.group = new THREE.Group();
    this.group.name = `actor:${id}`;
    scene.add(this.group);
    this.model = null;
    this.mixer = null;
    this.actions = new Map();
    this.current = null;
    this.oneShot = null;
    this.pos = new THREE.Vector3();
    this.target = new THREE.Vector3();
    this.placed = false;
    this.yaw = 0;
    this.height = 1.4;
    this.speed = WALK_SPEED;
  }

  attach(gltf, scale) {
    this.model = gltf.scene;
    this.model.scale.setScalar(scale);
    this.model.traverse((o) => { if (o.isMesh || o.isSkinnedMesh) { o.castShadow = true; o.frustumCulled = false; } });
    this.group.add(this.model);
    const box = measure(this.model);
    this.height = Math.max(0.3, box.max.y - box.min.y);
    this.model.position.y = -Math.min(0, box.min.y);
    this.mixer = new THREE.AnimationMixer(this.model);
    for (const clip of gltf.animations) this.actions.set(clip.name, this.mixer.clipAction(clip));
    this.mixer.addEventListener('finished', () => { this.oneShot = null; this.play(this.baseState || 'idle', true); });
  }

  has(name) { return this.actions.has(name); }

  play(name, force = false) {
    if (!this.mixer) return;
    if (!this.actions.has(name)) name = this.actions.has('idle') ? 'idle' : [...this.actions.keys()][0];
    if (!name || (this.current === name && !force)) return;
    const next = this.actions.get(name);
    next.reset().setLoop(THREE.LoopRepeat, Infinity);
    next.clampWhenFinished = false;
    next.enabled = true;
    next.setEffectiveWeight(1);
    if (this.current && this.actions.get(this.current) !== next) {
      next.crossFadeFrom(this.actions.get(this.current), 0.3, true);
    }
    next.play();
    this.current = name;
  }

  /* Plays a clip once (a gesture) and returns to the base state when it ends. */
  gesture(name) {
    if (!this.mixer || !this.actions.has(name) || this.moving) return;
    const act = this.actions.get(name);
    act.reset().setLoop(THREE.LoopOnce, 1);
    act.clampWhenFinished = false;
    if (this.current) act.crossFadeFrom(this.actions.get(this.current), 0.2, true);
    act.play();
    this.oneShot = name;
    this.current = name;
  }

  setTarget(v, teleport) {
    this.target.copy(v);
    if (!this.placed || teleport) { this.pos.copy(v); this.placed = true; }
  }

  move(dt, motion) {
    const d = this.target.clone().sub(this.pos);
    d.y = 0;
    const dist = d.length();
    this.moving = dist > 0.03;
    if (this.moving) {
      if (!motion || dt <= 0) { this.pos.copy(this.target); this.moving = false; }
      else {
        const step = Math.min(dist, this.speed * dt);
        this.pos.addScaledVector(d.normalize(), step);
        const wantYaw = Math.atan2(d.x, d.z);
        let diff = wantYaw - this.yaw;
        diff = Math.atan2(Math.sin(diff), Math.cos(diff));
        this.yaw += diff * clamp(dt * 9, 0, 1);
      }
    }
    this.group.position.copy(this.pos);
    this.group.rotation.y = this.yaw;
    if (this.mixer && dt > 0) this.mixer.update(dt);
  }

  dispose() {
    this.scene.remove(this.group);
    this.mixer?.stopAllAction();
  }
}

class Person extends Actor {
  constructor(scene, p) {
    super(scene, p.id);
    this.data = p;
    this.labelEl = el('div', { class: `person-label3d role-${p.role || 'unknown'}` },
      el('b', { text: p.name || p.id }), el('span', { class: 'act' }), el('span', { class: 'zzz', text: 'z z z' }));
    this.label = new CSS2DObject(this.labelEl);
    this.group.add(this.label);
    this.stillSince = 0;
    this.baseState = 'idle';
  }

  apply(p, teleport) {
    const changed = p.activity !== this.data.activity || p.x !== this.data.x || p.y !== this.data.y;
    this.data = p;
    this.setTarget(toWorld(p.x, p.y, 0), teleport);
    this.home = p.home !== false;
    if (changed) this.stillSince = 0;
    if (this.model) this.model.visible = this.home;
    this.labelEl.classList.toggle('away', !this.home);
    this.labelEl.querySelector('.act').textContent = this.home ? trunc(p.activity || '', 26) : 'away';
    this.label.position.y = (this.home ? this.height : 0.4) + 0.22;
    const act = String(p.activity || '');
    this.baseState = SLEEP_RE.test(act) || SIT_RE.test(act) ? 'sit' : INTERACT_RE.test(act) ? 'interact-right' : 'idle';
    this.sleepy = SLEEP_RE.test(act);
    this.still = STILL_RE.test(act);
  }

  tick(dt, motion) {
    this.move(dt, motion);
    if (!this.model) return;
    this.stillSince += dt;
    if (this.moving) this.play('walk');
    else if (!this.oneShot) this.play(this.baseState);
    const zzz = this.home && !this.moving && (this.sleepy || (this.still && this.stillSince > ZZZ_AFTER_S));
    this.labelEl.classList.toggle('asleep', zzz);
  }
}

class Cat extends Actor {
  constructor(scene, p) {
    super(scene, p.id);
    this.data = p;
    this.anchor = new THREE.Vector3();
    this.room = null;
    this.wanderIn = 2;
    this.speed = 1.2 * CAT_HEIGHT;
    this.labelEl = el('div', { class: 'person-label3d pet' }, el('b', { text: p.name || p.id }));
    this.label = new CSS2DObject(this.labelEl);
    this.group.add(this.label);
    this.baseState = 'static';
  }

  apply(p, room, teleport) {
    const moved = p.x !== this.data.x || p.y !== this.data.y;
    this.data = p;
    this.room = room;
    toWorld(p.x, p.y, 0, this.anchor);
    if (moved || !this.placed) this.setTarget(this.anchor, teleport);
    this.label.position.y = this.height + 0.15;
  }

  tick(dt, motion) {
    this.wanderIn -= dt;
    if (this.wanderIn <= 0 && !this.moving && motion) {
      this.wanderIn = 3 + Math.random() * 5;
      const t = this.anchor.clone().add(new THREE.Vector3((Math.random() - 0.5) * 1.4, 0, (Math.random() - 0.5) * 1.4));
      if (this.room) {   // stay inside the room, away from the walls
        const r = this.room;
        const minX = (r.x - 500) / 100 + 0.35, maxX = (r.x + r.w - 500) / 100 - 0.35;
        const minZ = (r.y - 300) / 100 + 0.35, maxZ = (r.y + r.h - 300) / 100 - 0.35;
        t.x = clamp(t.x, Math.min(minX, maxX), Math.max(minX, maxX));
        t.z = clamp(t.z, Math.min(minZ, maxZ), Math.max(minZ, maxZ));
      }
      this.target.copy(t);
    }
    this.move(dt, motion);
    if (!this.model) return;
    this.play(this.moving ? 'walk' : 'static');
  }
}

/* A boxy stand-in when the cat model is unavailable. */
function proceduralCat() {
  const g = new THREE.Group();
  const fur = new THREE.MeshStandardMaterial({ color: 0x6b5b95, roughness: 0.9 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.22, 0.24), fur); body.position.y = 0.2; g.add(body);
  const head = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.2, 0.2), fur); head.position.set(0, 0.36, 0.2); g.add(head);
  for (const sx of [-1, 1]) { const ear = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.1, 4), fur); ear.position.set(sx * 0.07, 0.5, 0.2); g.add(ear); }
  const tail = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, 0.3), fur); tail.position.set(0, 0.3, -0.25); tail.rotation.x = -0.6; g.add(tail);
  g.traverse((o) => { if (o.isMesh) o.castShadow = true; });
  return { scene: g, animations: [] };
}

export class People {
  constructor(scene, house) {
    this.scene = scene;
    this.house = house;
    this.loader = new GLTFLoader();
    this.people = new Map();
    this.pets = new Map();
    this.loading = new Map();
    this.pending = [];
  }

  async loadModel(url) {
    if (!this.loading.has(url)) this.loading.set(url, this.loader.loadAsync(url));
    return this.loading.get(url);
  }

  /* Loads every model the snapshot needs (used before the first frame so nobody pops in later). */
  async preload(snapshot) {
    const urls = new Set((snapshot.people || []).map((p) => PEOPLE_DIR + modelFor(p)));
    if ((snapshot.pets || []).some((p) => p.species === 'cat')) urls.add(PETS_DIR + CAT_MODEL);
    await Promise.all([...urls].map((u) => this.loadModel(u).catch((e) => console.warn('model failed', u, e))));
  }

  update(snapshot, teleport = false) {
    const seen = new Set();
    for (const p of snapshot.people || []) {
      seen.add(p.id);
      let person = this.people.get(p.id);
      if (!person) {
        person = new Person(this.scene, p);
        this.people.set(p.id, person);
        const scale = PERSON_SCALE * (p.role === 'child' ? CHILD_SCALE : 1);
        this.pending.push(this.loadModel(PEOPLE_DIR + modelFor(p)).then((gltf) => {
          if (!this.people.has(p.id)) return;
          person.attach(gltf, scale);
          person.apply(person.data, true);
        }).catch((e) => console.warn('person model failed', p.id, e)));
      }
      person.apply(p, teleport);
    }
    for (const [id, person] of this.people) if (!seen.has(id)) { person.dispose(); this.people.delete(id); }
    const seenPets = new Set();
    for (const p of snapshot.pets || []) {
      seenPets.add(p.id);
      let cat = this.pets.get(p.id);
      const room = snapshot.rooms.find((r) => r.id === p.room) || null;
      if (!cat) {
        cat = new Cat(this.scene, p);
        this.pets.set(p.id, cat);
        const load = p.species === 'cat' ? this.loadModel(PETS_DIR + CAT_MODEL) : Promise.reject(new Error('no model'));
        this.pending.push(load.catch(() => proceduralCat()).then((gltf) => {
          if (!this.pets.has(p.id)) return;
          const box = new THREE.Box3().setFromObject(gltf.scene);
          const h = Math.max(0.2, box.max.y - box.min.y);
          cat.attach(gltf, CAT_HEIGHT / h);
          cat.apply(cat.data, room, true);
        }));
      }
      cat.apply(p, room, teleport);
    }
    for (const [id, cat] of this.pets) if (!seenPets.has(id)) { cat.dispose(); this.pets.delete(id); }
  }

  gesture(id, clip = 'emote-yes') { this.people.get(id)?.gesture(clip); }

  positionOf(id) {
    const a = this.people.get(id) || this.pets.get(id);
    return a ? a.pos.clone() : null;
  }
  headOf(id) {
    const a = this.people.get(id) || this.pets.get(id);
    return a ? a.pos.clone().setY(a.pos.y + a.height * 1.02) : null;
  }
  parentsAtHome() {
    return [...this.people.values()].filter((p) => p.data.role === 'parent' && p.home);
  }

  tick(dt, motion) {
    for (const p of this.people.values()) p.tick(dt, motion);
    for (const c of this.pets.values()) c.tick(dt, motion);
  }
}
