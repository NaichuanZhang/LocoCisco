/* Time of day -> sky colour, hemisphere fill and one sun/moon directional light (the only shadow
   caster). snapshot.time ("HH:MM") drives a day factor 0..1; the current values ease toward their
   targets so a scenario's set_time jump animates instead of popping. */
import * as THREE from 'three';
import { approach, lerp } from './world.js';

const KEY = {
  // night (day factor 0), dusk/dawn (0.5), day (1)
  sky: [0x0b1330, 0xe9895a, 0xcfe6ff],
  hemiSky: [0x6c7fc9, 0xffd3ad, 0xeaf4ff],
  horizon: [0x1e2440, 0x8a6a56, 0x9aa98a],
  lawn: [0x2a4630, 0x6e7d44, 0x8fba6f],
  hemi: [0.5, 1.45, 1.75],
  sun: [0.32, 2.0, 2.8],
  sunColor: [0xaebcff, 0xffb070, 0xfff3df],
};

function minutes(time) {
  const [hh, mm] = String(time || '12:00').split(':').map(Number);
  return ((hh || 0) * 60 + (mm || 0)) % 1440;
}

/* 0 at night, 1 in full day, with a dawn ramp 05:30-07:00 and a dusk ramp 17:30-19:30. */
export function dayFactor(time) {
  const t = minutes(time);
  const ramp = (a, b) => Math.min(1, Math.max(0, (t - a) / (b - a)));
  if (t < 5 * 60 + 30) return 0;
  if (t < 7 * 60) return ramp(5 * 60 + 30, 7 * 60);
  if (t < 17 * 60 + 30) return 1;
  if (t < 19 * 60 + 30) return 1 - ramp(17 * 60 + 30, 19 * 60 + 30);
  return 0;
}

const cA = new THREE.Color(), cB = new THREE.Color();
function keyColor(keys, f, out) {
  if (f < 0.5) { cA.setHex(keys[0]); cB.setHex(keys[1]); return out.copy(cA).lerp(cB, f * 2); }
  cA.setHex(keys[1]); cB.setHex(keys[2]);
  return out.copy(cA).lerp(cB, (f - 0.5) * 2);
}
function keyNum(keys, f) {
  return f < 0.5 ? lerp(keys[0], keys[1], f * 2) : lerp(keys[1], keys[2], (f - 0.5) * 2);
}

export class Daylight {
  constructor(scene, lawnMaterial) {
    this.scene = scene;
    this.lawnMaterial = lawnMaterial;
    this.target = 1;
    this.factor = 1;
    this.time = '12:00';
    scene.background = new THREE.Color(0xcfe6ff);
    this.hemi = new THREE.HemisphereLight(0xcfe6ff, 0x7a8f68, 1.1);
    scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(0xfff3df, 2.3);
    this.sun.position.set(6, 11, 5);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    const cam = this.sun.shadow.camera;
    cam.left = -7.5; cam.right = 7.5; cam.top = 6; cam.bottom = -6; cam.near = 2; cam.far = 40;
    this.sun.shadow.bias = -0.0006;
    this.sun.shadow.normalBias = 0.02;
    scene.add(this.sun, this.sun.target);
    this.apply(1);
  }

  setTime(time) {
    this.time = time;
    this.target = dayFactor(time);
  }

  get isNight() { return this.target < 0.25; }

  tick(dt) {
    if (Math.abs(this.factor - this.target) < 0.002) { this.factor = this.target; return; }
    this.factor = dt <= 0 ? this.target : approach(this.factor, this.target, 2.2, dt);
    this.apply(this.factor);
  }

  apply(f) {
    keyColor(KEY.sky, f, this.scene.background);
    keyColor(KEY.hemiSky, f, this.hemi.color);
    keyColor(KEY.horizon, f, this.hemi.groundColor);
    this.hemi.intensity = keyNum(KEY.hemi, f);
    this.sun.intensity = keyNum(KEY.sun, f);
    keyColor(KEY.sunColor, f, this.sun.color);
    // the sun climbs with the day; the moon sits low and to the other side
    const elev = lerp(0.35, 1.0, f);
    this.sun.position.set(6 * (f > 0.25 ? 1 : -0.6), 11 * elev + 2, 5);
    if (this.lawnMaterial) keyColor(KEY.lawn, f, this.lawnMaterial.color);
  }
}
