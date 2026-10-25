/* Shared constants and helpers for the 3D house. One coordinate system: home.json rooms, devices,
   mics and snapshot.people[].x/y live on a 1000x600 plan (origin top-left); the world is that plan in
   metres (100 units = 1 m), centred on the origin, with plan y mapped to world +z (toward the porch). */
import * as THREE from 'three';

export const PLAN = { w: 1000, h: 600 };
export const SCALE = 0.01;
export const WALL_H = 1.1;
export const WALL_T = 0.12;
export const CEILING_H = 2.05;        // where pendant bulbs hang (there is no ceiling mesh)
export const FURNITURE_SCALE = 0.6;   // KayKit units -> metres for this dollhouse
/* Kenney minis are 0.67-0.72 units tall. The kitchen table (table_medium, 1.0 unit, placed at 0.85) is
   0.51 m and a chair back 0.60 m, so 1.55 makes adults 1.04-1.12 m: about 2.1x the table, like a real
   adult at a real table (1.7 m / 0.75 m), instead of the 2.7x that 2.0 gave. */
export const PERSON_SCALE = 1.55;
export const CHILD_SCALE = 0.75;      // Maya is 0.78 m: shoulder-high to the adults
export const CAT_HEIGHT = 0.3;        // metres; ~28% of an adult, the same ratio the 1.4 m people had

export function toWorld(x, y, elevation = 0, out = new THREE.Vector3()) {
  return out.set((x - PLAN.w / 2) * SCALE, elevation, (y - PLAN.h / 2) * SCALE);
}
export const m = (units) => units * SCALE;

export function roomCenter(room, elevation = 0, out = new THREE.Vector3()) {
  return toWorld(room.x + room.w / 2, room.y + room.h / 2, elevation, out);
}
export function roomContains(room, x, y) {
  return x >= room.x && x <= room.x + room.w && y >= room.y && y <= room.y + room.h;
}

export const FLOOR_COLORS = {
  living: 0xcdb892, kitchen: 0xb8c6cc, hallway: 0xc9b89d, bedroom_main: 0xb9a9cf,
  bedroom_maya: 0xf0b9c9, porch: 0x8d6a4c,
};
const FALLBACK_FLOORS = [0xd4c3a3, 0xb7c7c0, 0xc8b7d6, 0xe3c1b4, 0xbfcbb0, 0xd1c2c0];
export function floorColor(room, index) {
  return FLOOR_COLORS[room.id] ?? FALLBACK_FLOORS[index % FALLBACK_FLOORS.length];
}

export const SOURCE_COLORS = { person: 0x2563eb, tv: 0x7c3aed, outside: 0xd97706, unknown: 0x6b7280 };
export const VERDICT_COLORS = { allow: 0x15803d, deny: 0xb91c1c, ask_parent: 0xb45309, ask_speaker: 0xb45309, ignore: 0x6b7280 };

export const lerp = THREE.MathUtils.lerp;
export const clamp = THREE.MathUtils.clamp;
export const smoothstep = (t) => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
/* Exponential approach: moves `cur` toward `target`; `rate` ~ 1/seconds to cover 63% of the gap. */
export const approach = (cur, target, rate, dt) => cur + (target - cur) * (1 - Math.exp(-rate * dt));

export function isNight(time) {
  const [hh, mm] = String(time || '12:00').split(':').map(Number);
  const t = hh * 60 + mm;
  return t >= 19 * 60 + 30 || t < 6 * 60 + 30;
}
export const words = (a) => String(a ?? '').replace(/_/g, ' ');
export const trunc = (str, n) => (str = String(str ?? ''), str.length > n ? str.slice(0, n - 1) + '…' : str);

/* Small DOM helper for CSS2D label content. */
export function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'style') node.style.cssText = v;
    else node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) if (c != null) node.append(c.nodeType ? c : document.createTextNode(String(c)));
  return node;
}

/* A soft radial glow texture shared by halos and packets. */
let glowTexture = null;
export function getGlowTexture() {
  if (glowTexture) return glowTexture;
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.35, 'rgba(255,255,255,0.45)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  glowTexture = new THREE.CanvasTexture(canvas);
  glowTexture.colorSpace = THREE.SRGBColorSpace;
  return glowTexture;
}

export function makeGlowSprite(color, scale = 1) {
  const mat = new THREE.SpriteMaterial({
    map: getGlowTexture(), color, transparent: true, opacity: 0, depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.setScalar(scale);
  return sprite;
}

/* Shared geometries (reused everywhere; scaled per mesh). */
export const UNIT_BOX = new THREE.BoxGeometry(1, 1, 1);
export const UNIT_CYL = new THREE.CylinderGeometry(0.5, 0.5, 1, 20);
export const UNIT_SPHERE = new THREE.SphereGeometry(0.5, 18, 14);
export const UNIT_RING = new THREE.RingGeometry(0.42, 0.5, 40);
export const UNIT_PLANE = new THREE.PlaneGeometry(1, 1);
