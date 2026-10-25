/* The procedural house: floors, low walls with doorways, a porch, room labels and every device the
   snapshot describes. Geometry comes from the snapshot's rooms/devices/doors/mics and home.sensors;
   nothing the snapshot provides is hard-coded. Only doorway placement between rooms needs a tiny
   config because the snapshot has no wall-opening geometry. */
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import {
  SCALE, WALL_H, WALL_T, CEILING_H, toWorld, roomCenter, roomContains, floorColor, approach,
  el, makeGlowSprite, UNIT_BOX, UNIT_CYL, UNIT_SPHERE, UNIT_RING, UNIT_PLANE,
} from './world.js';

const DOOR_GAP = 100;                 // plan units (1 m) for a doorway
const DOORWAYS = {                    // fraction along a shared wall; null = solid wall
  'kitchen|bedroom_main': 0.72,
  'bedroom_main|bedroom_maya': null,
};
const LIGHT_INTENSITY = 16;           // candela-ish PointLight intensity at full brightness

const MAT = {
  wall: new THREE.MeshStandardMaterial({ color: 0xece7dd, roughness: 0.95, transparent: true }),
  wallOut: new THREE.MeshStandardMaterial({ color: 0xd9d2c4, roughness: 0.95, transparent: true }),
  wood: new THREE.MeshStandardMaterial({ color: 0x9a6b3f, roughness: 0.8 }),
  woodLight: new THREE.MeshStandardMaterial({ color: 0xd7b58a, roughness: 0.8 }),
  dark: new THREE.MeshStandardMaterial({ color: 0x2a2f3a, roughness: 0.6, metalness: 0.2 }),
  metal: new THREE.MeshStandardMaterial({ color: 0x9aa3b2, roughness: 0.4, metalness: 0.6 }),
  white: new THREE.MeshStandardMaterial({ color: 0xf3f4f6, roughness: 0.6 }),
  slate: new THREE.MeshStandardMaterial({ color: 0x4b5563, roughness: 0.7 }),
  cord: new THREE.MeshStandardMaterial({ color: 0x33363d, roughness: 0.9 }),
  shade: new THREE.MeshStandardMaterial({ color: 0xf5efe2, roughness: 0.9, side: THREE.DoubleSide }),
  glass: new THREE.MeshStandardMaterial({ color: 0x1f2937, roughness: 0.25, metalness: 0.3 }),
  teal: new THREE.MeshStandardMaterial({ color: 0x0e7490, emissive: 0x22d3ee, emissiveIntensity: 0.6, roughness: 0.5 }),
  mat: new THREE.MeshStandardMaterial({ color: 0x6b4a2b, roughness: 1 }),
};
const CONE = new THREE.ConeGeometry(0.5, 1, 24, 1, true);
const HALF_SPHERE = new THREE.SphereGeometry(0.5, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2);

function box(mat, sx, sy, sz, x, y, z) {
  const mesh = new THREE.Mesh(UNIT_BOX, mat);
  mesh.scale.set(sx, sy, sz);
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}
function cyl(mat, r, h, x, y, z) {
  const mesh = new THREE.Mesh(UNIT_CYL, mat);
  mesh.scale.set(r * 2, h, r * 2);
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  return mesh;
}
function label(className, pos, ...children) {
  const node = el('div', { class: className }, ...children);
  const obj = new CSS2DObject(node);
  obj.position.copy(pos);
  return obj;
}

/* Every axis-aligned wall segment of the indoor rooms, split at room corners and shared by the
   rooms on both sides. Returns {x1,y1,x2,y2,rooms:Set,horizontal}. */
export function wallSegments(rooms) {
  const indoor = rooms.filter((r) => r.indoor !== false);
  const X = [...new Set(indoor.flatMap((r) => [r.x, r.x + r.w]))].sort((a, b) => a - b);
  const Y = [...new Set(indoor.flatMap((r) => [r.y, r.y + r.h]))].sort((a, b) => a - b);
  const segs = new Map();
  const add = (x1, y1, x2, y2, id) => {
    const key = `${x1},${y1},${x2},${y2}`;
    let s = segs.get(key);
    if (!s) { s = { x1, y1, x2, y2, rooms: new Set(), horizontal: y1 === y2 }; segs.set(key, s); }
    s.rooms.add(id);
  };
  for (const r of indoor) {
    const xs = X.filter((x) => x >= r.x && x <= r.x + r.w);
    for (let i = 0; i < xs.length - 1; i++) { add(xs[i], r.y, xs[i + 1], r.y, r.id); add(xs[i], r.y + r.h, xs[i + 1], r.y + r.h, r.id); }
    const ys = Y.filter((y) => y >= r.y && y <= r.y + r.h);
    for (let i = 0; i < ys.length - 1; i++) { add(r.x, ys[i], r.x, ys[i + 1], r.id); add(r.x + r.w, ys[i], r.x + r.w, ys[i + 1], r.id); }
  }
  return [...segs.values()];
}

/* The opening in a segment as [from, to] along its axis, or null for a solid wall. */
export function doorwayFor(seg, doors) {
  const a = seg.horizontal ? seg.x1 : seg.y1, b = seg.horizontal ? seg.x2 : seg.y2;
  const len = b - a;
  for (const d of doors || []) {
    const on = seg.horizontal ? (d.y === seg.y1 && d.x >= a && d.x <= b) : (d.x === seg.x1 && d.y >= a && d.y <= b);
    if (on) { const c = seg.horizontal ? d.x : d.y; return [Math.max(a, c - DOOR_GAP / 2), Math.min(b, c + DOOR_GAP / 2)]; }
  }
  if (seg.rooms.size < 2) return null;
  const key = [...seg.rooms].sort().join('|');
  const frac = key in DOORWAYS ? DOORWAYS[key] : 0.5;
  if (frac == null || len < 60) return null;
  const gap = Math.min(DOOR_GAP * 0.9, len * 0.7);
  const c = a + len * frac;
  return [Math.max(a + 10, c - gap / 2), Math.min(b - 10, c + gap / 2)];
}

export class House {
  constructor(scene, home) {
    this.scene = scene;
    this.home = home || {};
    this.group = new THREE.Group();
    this.group.name = 'house';
    scene.add(this.group);
    this.rooms = new Map();
    this.walls = [];
    this.devices = new Map();
    this.mics = new Map();
    this.cameras = new Map();
    this.door = null;
    this.hub = null;
    this.xray = false;
    this.focusRoom = null;
    this.built = false;
    this.signature = '';
    this.lawn = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.MeshStandardMaterial({ color: 0x7faa66, roughness: 1 }));
    this.lawn.rotation.x = -Math.PI / 2;
    this.lawn.position.y = -0.09;
    this.lawn.receiveShadow = true;
    scene.add(this.lawn);
  }

  /* Builds once; rebuilds only when the set of rooms/devices changes. */
  build(snapshot) {
    const sig = JSON.stringify([snapshot.rooms.map((r) => [r.id, r.x, r.y, r.w, r.h]), snapshot.devices.map((d) => d.device_id), (snapshot.mics || []).map((x) => x.mic_id)]);
    if (this.built && sig === this.signature) return false;
    this.signature = sig;
    this.clear();
    this.snapshot = snapshot;
    snapshot.rooms.forEach((r, i) => this.buildRoom(r, i));
    this.buildWalls(snapshot.rooms, snapshot.doors || []);
    for (const door of snapshot.doors || []) this.buildDoor(door);
    for (const d of snapshot.devices) this.buildDevice(d);
    this.buildKitchenExtras(snapshot);
    this.buildHub(snapshot.rooms);
    for (const mic of this.micsOf(snapshot)) this.buildMic(mic);
    for (const cam of this.home?.sensors?.cameras || []) this.buildCamera(cam, snapshot.rooms);
    this.built = true;
    return true;
  }

  clear() {
    for (const child of [...this.group.children]) this.group.remove(child);
    this.rooms.clear(); this.devices.clear(); this.mics.clear(); this.cameras.clear();
    this.walls = []; this.door = null; this.hub = null;
  }

  micsOf(snapshot) { return snapshot.mics || this.home?.sensors?.mics || []; }

  roomOf(x, y) { return this.snapshot?.rooms.find((r) => roomContains(r, x, y)) || null; }
  room(id) { return this.rooms.get(id)?.room || null; }

  // ------------------------------------------------------------------ rooms and walls
  buildRoom(room, index) {
    const floor = new THREE.Mesh(UNIT_BOX, new THREE.MeshStandardMaterial({ color: floorColor(room, index), roughness: 0.9 }));
    floor.scale.set(room.w * SCALE, 0.08, room.h * SCALE);
    roomCenter(room, -0.04, floor.position);
    floor.receiveShadow = true;
    floor.userData.room = room.id;
    this.group.add(floor);
    const outdoor = room.indoor === false;
    const lx = room.x + room.w / 2, ly = outdoor ? room.y + room.h * 0.86 : room.y + room.h * 0.06;
    const env = el('span', { class: 'env' });
    const lbl = label('room-label3d' + (outdoor ? ' outdoor' : ''), toWorld(lx, ly, outdoor ? 0.45 : WALL_H + 0.12), el('b', { text: room.name || room.id }), env);
    this.group.add(lbl);
    if (outdoor) this.buildPorch(room);
    this.rooms.set(room.id, { room, floor, label: lbl, env, lit: 0 });
  }

  buildPorch(room) {
    // two posts and a thin awning along the front edge make the outdoor room read as a porch
    const y = room.y + room.h - 12;
    for (const x of [room.x + 25, room.x + room.w - 25]) {
      const post = cyl(MAT.woodLight, 0.06, 1.5, 0, 0.75, 0);
      post.position.copy(toWorld(x, y, 0.75));
      this.group.add(post);
    }
    const awning = box(MAT.wood, room.w * SCALE, 0.05, 0.42, 0, 0, 0);
    awning.position.copy(toWorld(room.x + room.w / 2, y - 8, 1.52));
    this.group.add(awning);
  }

  buildWalls(rooms, doors) {
    for (const seg of wallSegments(rooms)) {
      const gap = doorwayFor(seg, doors);
      const a = seg.horizontal ? seg.x1 : seg.y1, b = seg.horizontal ? seg.x2 : seg.y2;
      const pieces = gap ? [[a, gap[0], true, false], [gap[1], b, false, true]] : [[a, b, true, true]];
      const exterior = seg.rooms.size < 2;
      for (const [p0, p1, extStart, extEnd] of pieces) {
        if (p1 - p0 < 2) continue;
        const len = (p1 - p0) * SCALE + (extStart ? WALL_T / 2 : 0) + (extEnd ? WALL_T / 2 : 0);
        const mid = (p0 + p1) / 2 + ((extEnd ? WALL_T / 2 : 0) - (extStart ? WALL_T / 2 : 0)) / SCALE / 2;
        const mesh = new THREE.Mesh(UNIT_BOX, (exterior ? MAT.wallOut : MAT.wall).clone());
        if (seg.horizontal) { mesh.scale.set(len, WALL_H, WALL_T); mesh.position.copy(toWorld(mid, seg.y1, WALL_H / 2)); }
        else { mesh.scale.set(WALL_T, WALL_H, len); mesh.position.copy(toWorld(seg.x1, mid, WALL_H / 2)); }
        mesh.castShadow = true; mesh.receiveShadow = true;
        mesh.material.opacity = 1;
        this.group.add(mesh);
        this.walls.push({ mesh, seg, exterior, rooms: [...seg.rooms], normal: this.outwardNormal(seg, rooms), opacity: 1 });
      }
    }
  }

  /* For an exterior wall: the normal pointing away from its room. For interior walls: the normal
     pointing from the first room into the second (tick() flips it per focused room). */
  outwardNormal(seg, rooms) {
    const ids = [...seg.rooms];
    const r = rooms.find((x) => x.id === ids[0]);
    if (seg.horizontal) return new THREE.Vector3(0, 0, r.y === seg.y1 ? -1 : 1);
    return new THREE.Vector3(r.x === seg.x1 ? -1 : 1, 0, 0);
  }

  buildDoor(door) {
    const pos = toWorld(door.x, door.y, 0);
    const g = new THREE.Group();
    g.position.copy(pos);
    // frame posts just outside the gap
    for (const sx of [-1, 1]) g.add(box(MAT.wood, 0.08, WALL_H + 0.06, WALL_T + 0.06, sx * (DOOR_GAP * SCALE / 2 - 0.02), (WALL_H + 0.06) / 2, 0));
    const hinge = new THREE.Group();
    hinge.position.set(-DOOR_GAP * SCALE / 2 + 0.06, 0, 0);
    const slab = box(MAT.wood, 0.86, WALL_H - 0.04, 0.06, 0.43, (WALL_H - 0.04) / 2, 0);
    hinge.add(slab);
    const knob = cyl(MAT.metal, 0.03, 0.04, 0.78, 0.55, 0.05);
    knob.rotation.x = Math.PI / 2;
    hinge.add(knob);
    const lockMat = new THREE.MeshStandardMaterial({ color: 0xb91c1c, emissive: 0xb91c1c, emissiveIntensity: 0.9, roughness: 0.4 });
    const lock = cyl(lockMat, 0.055, 0.03, 0.62, 0.62, 0.045);
    lock.rotation.x = Math.PI / 2;
    hinge.add(lock);
    const halo = makeGlowSprite(0xff4d4d, 0.45);
    halo.position.set(0.62, 0.62, 0.12);
    hinge.add(halo);
    g.add(hinge);
    const mat = box(MAT.mat, 0.8, 0.02, 0.45, 0, 0.01, 0.4);
    mat.castShadow = false;
    g.add(mat);
    const dot = el('i', { class: 'dot' });
    const text = el('span', { text: 'locked' });
    const lbl = label('door-label3d locked', new THREE.Vector3(0, WALL_H + 0.3, 0.05), dot, el('b', { text: door.name || 'Door' }), text);
    g.add(lbl);
    this.group.add(g);
    this.door = { door, group: g, hinge, lockMat, halo, lbl, text, open: 0, openTarget: 0, locked: true, pos: pos.clone().setY(0.6) };
  }

  // ------------------------------------------------------------------ devices
  buildDevice(d) {
    const room = this.snapshot.rooms.find((r) => r.id === d.room) || this.roomOf(d.x, d.y);
    const builder = { light: this.buildLight, tv: this.buildTv, stove: this.buildStove, thermostat: this.buildThermostat, lock: this.buildLock }[d.kind];
    const view = builder ? builder.call(this, d, room) : { pos: toWorld(d.x, d.y, 0.5), update() {}, tick() {} };
    view.kind = d.kind; view.room = d.room; view.id = d.device_id;
    this.devices.set(d.device_id, view);
    view.update(d.state || {});
  }

  buildLight(d, room) {
    const outdoor = room?.indoor === false;
    const g = new THREE.Group();
    const bulbY = outdoor ? 1.75 : CEILING_H;
    g.position.copy(toWorld(d.x, d.y, 0));
    if (outdoor) {
      g.add(cyl(MAT.cord, 0.03, 1.7, 0, 0.85, 0));
      g.add(cyl(MAT.slate, 0.07, 0.06, 0, 1.68, 0));
    } else {
      g.add(cyl(MAT.cord, 0.012, 0.6, 0, bulbY + 0.42, 0));
      const shade = new THREE.Mesh(CONE, MAT.shade);
      shade.scale.set(0.46, 0.2, 0.46);
      shade.position.y = bulbY + 0.16;
      g.add(shade);
    }
    const bulbMat = new THREE.MeshStandardMaterial({ color: 0xfff1cf, emissive: 0xffd27a, emissiveIntensity: 0.1, roughness: 0.3 });
    const bulb = new THREE.Mesh(UNIT_SPHERE, bulbMat);
    bulb.scale.setScalar(0.2);
    bulb.position.y = bulbY;
    g.add(bulb);
    const halo = makeGlowSprite(0xffd27a, 2.2);
    halo.position.y = bulbY;
    g.add(halo);
    const point = new THREE.PointLight(0xffe0b0, 0, 8, 2);
    point.position.y = bulbY - 0.12;
    point.castShadow = false;
    g.add(point);
    this.group.add(g);
    const roomRef = room ? this.rooms.get(room.id) : null;
    const view = {
      pos: g.position.clone().setY(bulbY), target: 0, cur: 0, bulbMat, halo, point, roomRef,
      update(state) { view.target = state.on ? Math.max(0.3, (state.brightness ?? 80) / 100) : 0; },
      tick: (dt) => {
        view.cur = dt <= 0 ? view.target : approach(view.cur, view.target, 3.2, dt);
        point.intensity = view.cur * LIGHT_INTENSITY;
        bulbMat.emissiveIntensity = 0.1 + view.cur * 2.6;
        halo.material.opacity = view.cur * 0.8;
        if (roomRef) { roomRef.lit = view.cur; roomRef.floor.material.emissive.setRGB(0.25 * view.cur, 0.19 * view.cur, 0.08 * view.cur); }
      },
    };
    return view;
  }

  buildTv(d, room) {
    const g = new THREE.Group();
    g.position.copy(toWorld(d.x, d.y, 0));
    const c = room ? roomCenter(room) : new THREE.Vector3();
    g.rotation.y = Math.atan2(c.x - g.position.x, c.z - g.position.z);   // screen faces the room
    g.add(box(MAT.dark, 0.6, 0.06, 0.34, 0, 0.03, 0));
    g.add(box(MAT.dark, 0.08, 0.3, 0.06, 0, 0.2, 0));
    g.add(box(MAT.dark, 1.34, 0.8, 0.07, 0, 0.75, 0));
    const canvas = document.createElement('canvas');
    canvas.width = 512; canvas.height = 288;
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    const screenMat = new THREE.MeshBasicMaterial({ map: tex });
    const screen = new THREE.Mesh(UNIT_PLANE, screenMat);
    screen.scale.set(1.24, 0.7, 1);
    screen.position.set(0, 0.75, 0.04);
    g.add(screen);
    const glow = makeGlowSprite(0x8fd3ff, 1.7);
    glow.position.set(0, 0.75, 0.3);
    g.add(glow);
    this.group.add(g);
    const ctx = canvas.getContext('2d');
    const draw = (state) => {
      ctx.fillStyle = '#0b0e14';
      ctx.fillRect(0, 0, 512, 288);
      if (!state.on) {
        ctx.fillStyle = '#1f2937'; ctx.fillRect(0, 0, 512, 288);
        ctx.fillStyle = '#ef4444'; ctx.beginPath(); ctx.arc(480, 268, 6, 0, Math.PI * 2); ctx.fill();
      } else {
        const grad = ctx.createLinearGradient(0, 0, 512, 288);
        grad.addColorStop(0, '#1d4ed8'); grad.addColorStop(0.55, '#7c3aed'); grad.addColorStop(1, '#db2777');
        ctx.fillStyle = grad; ctx.fillRect(0, 0, 512, 288);
        ctx.fillStyle = 'rgba(255,255,255,0.12)';
        for (let i = 0; i < 6; i++) ctx.fillRect(40 + i * 75, 40 + (i % 2) * 30, 50, 120);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 44px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('▶  ' + (state.playing || 'on'), 256, 180, 470);
        ctx.font = '26px system-ui, sans-serif';
        ctx.fillStyle = 'rgba(255,255,255,0.8)';
        ctx.fillText(`volume ${state.volume ?? ''}`, 256, 240, 470);
      }
      tex.needsUpdate = true;
    };
    const facing = new THREE.Vector3(0, 0, 1).applyAxisAngle(new THREE.Vector3(0, 1, 0), g.rotation.y);
    const view = {
      pos: g.position.clone().setY(0.75), facing, on: 0, target: 0, lastKey: null,
      screenPos: () => g.localToWorld(new THREE.Vector3(0, 0.75, 0.06)),
      update(state) {
        const key = `${!!state.on}|${state.playing || ''}|${state.volume ?? ''}`;
        if (key !== view.lastKey) { view.lastKey = key; draw(state); }
        view.target = state.on ? 1 : 0;
      },
      tick(dt) {
        view.on = dt <= 0 ? view.target : approach(view.on, view.target, 4, dt);
        glow.material.opacity = view.on * 0.35;
        screenMat.color.setScalar(0.35 + 0.65 * view.on);
      },
    };
    return view;
  }

  buildStove(d) {
    const g = new THREE.Group();
    g.position.copy(toWorld(d.x, d.y, 0));
    g.add(box(MAT.white, 0.62, 0.86, 0.62, 0, 0.43, 0));
    g.add(box(MAT.dark, 0.64, 0.04, 0.64, 0, 0.88, 0));
    g.add(box(MAT.glass, 0.46, 0.3, 0.02, 0, 0.42, 0.31));      // oven window
    g.add(box(MAT.metal, 0.5, 0.03, 0.03, 0, 0.62, 0.33));      // handle
    const burnerMat = new THREE.MeshStandardMaterial({ color: 0x15181d, emissive: 0xff7a1a, emissiveIntensity: 0, roughness: 0.6 });
    const burners = [];
    for (const [bx, bz] of [[-0.16, -0.16], [0.16, -0.16], [-0.16, 0.16], [0.16, 0.16]]) {
      const b = cyl(burnerMat, 0.1, 0.025, bx, 0.91, bz);
      b.castShadow = false;
      g.add(b);
      burners.push(b);
    }
    for (let i = 0; i < 4; i++) {
      const knob = cyl(MAT.metal, 0.02, 0.02, -0.21 + i * 0.14, 0.8, 0.32);
      knob.rotation.x = Math.PI / 2;
      g.add(knob);
    }
    const halo = makeGlowSprite(0xff8a2a, 1.1);
    halo.position.set(0, 1.05, 0);
    g.add(halo);
    this.group.add(g);
    const view = {
      pos: g.position.clone().setY(0.95), target: 0, cur: 0, t: 0,
      update(state) { view.target = state.burner_on ? 1 : 0; },
      tick(dt) {
        view.t += dt;
        view.cur = dt <= 0 ? view.target : approach(view.cur, view.target, 4, dt);
        const flicker = 0.75 + 0.25 * Math.sin(view.t * 9) * Math.sin(view.t * 3.7);
        burnerMat.emissiveIntensity = view.cur * (1.8 + flicker);
        halo.material.opacity = view.cur * (0.45 + 0.2 * flicker);
      },
    };
    return view;
  }

  buildThermostat(d, room) {
    // a wall panel on the room's nearest north/south wall, facing into the room
    const north = !room || (d.y - room.y) <= (room.y + room.h - d.y);
    const y = room ? (north ? room.y + WALL_T / SCALE / 2 + 2 : room.y + room.h - WALL_T / SCALE / 2 - 2) : d.y;
    const g = new THREE.Group();
    g.position.copy(toWorld(d.x, y, 0));
    g.rotation.y = north ? 0 : Math.PI;
    g.add(box(MAT.white, 0.26, 0.26, 0.05, 0, 0.9, 0.025));
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 128;
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    const face = new THREE.Mesh(UNIT_PLANE, new THREE.MeshBasicMaterial({ map: tex }));
    face.scale.setScalar(0.2);
    face.position.set(0, 0.9, 0.052);
    g.add(face);
    const chip = label('chip3d thermo', new THREE.Vector3(0, 1.32, 0.1), el('b', { text: '–' }), ' °C');
    g.add(chip);
    this.group.add(g);
    const ctx = canvas.getContext('2d');
    const view = {
      pos: g.position.clone().setY(0.9), last: null,
      update(state) {
        const v = state.setpoint_c;
        if (v === view.last) return;
        view.last = v;
        ctx.fillStyle = '#111827'; ctx.fillRect(0, 0, 128, 128);
        ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 8; ctx.beginPath(); ctx.arc(64, 64, 50, 0.75 * Math.PI, 2.25 * Math.PI); ctx.stroke();
        ctx.fillStyle = '#f9fafb'; ctx.font = 'bold 40px system-ui, sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(typeof v === 'number' ? v.toFixed(1) + '°' : '–', 64, 80);
        tex.needsUpdate = true;
        chip.element.querySelector('b').textContent = typeof v === 'number' ? v.toFixed(1) : '–';
      },
      tick() {},
    };
    return view;
  }

  buildLock(d) {
    // the lock is drawn on the door; this view just forwards its state
    const house = this;
    return {
      pos: toWorld(d.x, d.y, 0.6),
      update(state) { if (house.door) house.setLocked(state.locked !== false); },
      tick() {},
    };
  }

  setLocked(locked) {
    const dv = this.door;
    if (!dv || dv.locked === locked) return;
    dv.locked = locked;
    const col = locked ? 0xb91c1c : 0x16a34a;
    dv.lockMat.color.setHex(col); dv.lockMat.emissive.setHex(col);
    dv.halo.material.color.setHex(locked ? 0xff4d4d : 0x4ade80);
    dv.text.textContent = locked ? 'locked' : 'unlocked';
    dv.lbl.element.classList.toggle('locked', locked);
    dv.lbl.element.classList.toggle('unlocked', !locked);
  }

  buildKitchenExtras(snapshot) {
    const stove = snapshot.devices.find((d) => d.kind === 'stove');
    const room = stove ? snapshot.rooms.find((r) => r.id === stove.room) : null;
    if (!stove || !room) return;
    // counter along the north wall up to the stove, fridge against the nearest side wall below it
    const left = room.x + 18 + 30, right = stove.x - 34;
    if (right - left > 40) {
      const cx = (left + right) / 2, w = (right - left) * SCALE;
      const base = box(MAT.slate, w, 0.84, 0.6, 0, 0.42, 0);
      base.position.copy(toWorld(cx, room.y + 42, 0.42));
      const top = box(MAT.woodLight, w + 0.04, 0.05, 0.64, 0, 0, 0);
      top.position.copy(toWorld(cx, room.y + 42, 0.865));
      const sink = box(MAT.metal, 0.5, 0.02, 0.4, 0, 0, 0);
      sink.position.copy(toWorld(cx, room.y + 42, 0.9));
      const tap = cyl(MAT.metal, 0.015, 0.25, 0, 0, 0);
      tap.position.copy(toWorld(cx, room.y + 24, 1.0));
      this.group.add(base, top, sink, tap);
    }
    const sideRight = stove.x > room.x + room.w / 2;
    const fx = sideRight ? room.x + room.w - 42 : room.x + 42;
    const fridge = box(MAT.white, 0.7, 1.55, 0.7, 0, 0, 0);
    fridge.position.copy(toWorld(fx, stove.y + 95, 0.775));
    const fridgeDoorLine = box(MAT.metal, 0.03, 0.9, 0.02, 0, 0, 0);
    fridgeDoorLine.position.copy(toWorld(fx + (sideRight ? -38 : 38), stove.y + 95 - 18, 0.8));
    this.group.add(fridge, fridgeDoorLine);
  }

  buildHub(rooms) {
    const hall = rooms.find((r) => r.id === 'hallway') || rooms.find((r) => r.indoor !== false) || rooms[0];
    if (!hall) return;
    const x = hall.x + hall.w * 0.66, y = hall.y + hall.h - 16;
    const console = box(MAT.wood, 0.52, 0.56, 0.3, 0, 0, 0);
    console.position.copy(toWorld(x, y, 0.28));
    const puck = cyl(MAT.white, 0.12, 0.05, 0, 0, 0);
    puck.position.copy(toWorld(x, y, 0.585));
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.85, side: THREE.DoubleSide });
    const ring = new THREE.Mesh(UNIT_RING, ringMat);
    ring.scale.setScalar(0.22);
    ring.rotation.x = -Math.PI / 2;
    ring.position.copy(toWorld(x, y, 0.615));
    const halo = makeGlowSprite(0x22d3ee, 0.9);
    halo.position.copy(toWorld(x, y, 0.66));
    halo.material.opacity = 0.2;
    const chip = label('chip3d hub', toWorld(x, y, 1.05), el('b', { text: 'home hub' }));
    this.group.add(console, puck, ring, halo, chip);
    this.hub = { pos: toWorld(x, y, 0.62), ring, ringMat, halo, pulse: 0 };
  }

  buildMic(mic) {
    const g = new THREE.Group();
    g.position.copy(toWorld(mic.x, mic.y, 0));
    g.add(cyl(MAT.slate, 0.05, 0.5, 0, 0.25, 0));
    g.add(cyl(MAT.white, 0.09, 0.05, 0, 0.525, 0));
    const ring = new THREE.Mesh(UNIT_RING, MAT.teal);
    ring.scale.setScalar(0.16);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.552;
    g.add(ring);
    this.group.add(g);
    this.mics.set(mic.mic_id, { mic, pos: g.position.clone().setY(0.56), room: mic.room });
  }

  buildCamera(cam, rooms) {
    const room = rooms.find((r) => r.id === cam.room);
    if (!room) return;
    const outdoor = room.indoor === false;
    const x = room.x + 20, y = room.y + 20;
    const g = new THREE.Group();
    g.position.copy(toWorld(x, y, 0));
    const h = outdoor ? 1.45 : WALL_H - 0.05;
    if (outdoor) g.add(cyl(MAT.cord, 0.025, h, 0, h / 2, 0));
    g.add(box(MAT.white, 0.16, 0.03, 0.16, 0, h, 0));
    const dome = new THREE.Mesh(HALF_SPHERE, MAT.glass);
    dome.scale.setScalar(0.2);
    dome.rotation.x = Math.PI;        // hangs down from the plate
    dome.position.y = h - 0.005;
    g.add(dome);
    const led = new THREE.Mesh(UNIT_SPHERE, new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0xef4444, emissiveIntensity: 1.2 }));
    led.scale.setScalar(0.03);
    led.position.set(0.06, h - 0.07, 0.06);
    g.add(led);
    this.group.add(g);
    this.cameras.set(cam.camera_id, { cam, pos: g.position.clone().setY(h - 0.08), room: cam.room });
  }

  // ------------------------------------------------------------------ lookups used by overlays
  devicePos(id) { return this.devices.get(id)?.pos || null; }
  micPos(id) { return this.mics.get(id)?.pos || null; }
  micInRoom(room) { return [...this.mics.values()].find((m) => m.room === room) || null; }
  cameraInRoom(room) { return [...this.cameras.values()].find((c) => c.room === room) || null; }
  hubPos() { return this.hub?.pos || null; }
  tv() { return [...this.devices.values()].find((v) => v.kind === 'tv') || null; }
  doorFront(offset = 0.55) { return this.door ? this.door.pos.clone().add(new THREE.Vector3(0, 0, offset)) : null; }
  pulseHub() { if (this.hub) this.hub.pulse = 1; }

  // ------------------------------------------------------------------ live updates
  update(snapshot) {
    this.snapshot = snapshot;
    for (const d of snapshot.devices) this.devices.get(d.device_id)?.update(d.state || {});
    for (const door of snapshot.doors || []) {
      if (this.door && this.door.door.door_id === door.door_id) {
        this.door.openTarget = door.open ? 1 : 0;
        if (typeof door.locked === 'boolean') this.setLocked(door.locked);
      }
    }
    for (const r of snapshot.rooms) {
      const ref = this.rooms.get(r.id);
      if (!ref) continue;
      const env = r.env || {};
      ref.env.textContent = `${env.lux != null ? Math.round(env.lux) + ' lux' : ''}${env.celsius != null ? ' · ' + Number(env.celsius).toFixed(1) + ' °C' : ''}`;
    }
  }

  tick(dt, camera) {
    for (const v of this.devices.values()) v.tick(dt);
    if (this.door) {
      const dv = this.door;
      dv.open = dt <= 0 ? dv.openTarget : approach(dv.open, dv.openTarget, 4, dt);
      dv.hinge.rotation.y = -1.5 * dv.open;
    }
    if (this.hub) {
      this.hub.pulse = Math.max(0, this.hub.pulse - dt * 1.6);
      this.hub.ringMat.opacity = 0.5 + 0.5 * this.hub.pulse;
      this.hub.ring.scale.setScalar(0.22 + 0.5 * this.hub.pulse);
      this.hub.halo.material.opacity = 0.15 + 0.6 * this.hub.pulse;
    }
    for (const ref of this.rooms.values()) ref.label.element.classList.toggle('lit', ref.lit > 0.4);
    this.fadeWalls(dt, camera);
  }

  /* Exterior walls facing the camera fade so the dollhouse stays open; interior walls fade only when
     they stand between the camera and the focused room. X-ray fades everything. */
  fadeWalls(dt, camera) {
    const toCam = new THREE.Vector3();
    for (const w of this.walls) {
      let target = 1;
      if (this.xray) target = 0.22;
      else {
        toCam.copy(camera.position).sub(w.mesh.position).normalize();
        if (w.exterior) target = w.normal.dot(toCam) > 0.12 ? 0.1 : 1;
        else if (this.focusRoom && w.rooms.includes(this.focusRoom)) {
          const sign = w.rooms[0] === this.focusRoom ? 1 : -1;   // normal pointing out of the focused room
          target = w.normal.dot(toCam) * sign > 0.2 ? 0.15 : 1;
        }
      }
      if (Math.abs(w.opacity - target) < 0.005) continue;
      w.opacity = dt <= 0 ? target : approach(w.opacity, target, 7, dt);
      w.mesh.material.opacity = w.opacity;
      w.mesh.material.depthWrite = w.opacity > 0.6;
      w.mesh.castShadow = w.opacity > 0.6;
    }
  }
}
