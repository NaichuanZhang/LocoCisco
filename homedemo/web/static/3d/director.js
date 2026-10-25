/* Camera director. "Auto camera" eases to the room of the latest important event and drifts back to
   the overview when things go quiet; a manual orbit pauses it until Overview/Auto is pressed again.
   With prefers-reduced-motion (or the page's "Pause animations") moves become cuts. */
import * as THREE from 'three';
import { toWorld, smoothstep } from './world.js';

const AZIMUTH = 0.62;      // radians east of south: the isometric-looking three-quarter view
const ELEVATION = 0.74;    // radians above the floor
const MOVE_S = 1.25;
const HOLD_S = 9;

export class Director {
  constructor(camera, controls) {
    this.camera = camera;
    this.controls = controls;
    this.auto = true;
    this.paused = false;        // set by a manual orbit
    this.instant = false;       // cuts instead of moves
    this.compact = false;       // narrow canvas: centre the house, no headroom for floating cards
    this.headroom = 0;          // extra metres above the room for floating cards (presenter mode)
    this.motion = true;
    this.mode = 'overview';
    this.focusRoom = null;
    this.holdUntil = 0;
    this.now = 0;
    this.move = null;
    this.overviewShot = { target: new THREE.Vector3(0, 0.4, 0), radius: 6 };
    this.rooms = new Map();
    this.onChange = null;
    controls.addEventListener('start', () => this.onManual());
  }

  setRooms(rooms) {
    this.rooms = new Map(rooms.map((r) => [r.id, r]));
    const xs = rooms.flatMap((r) => [r.x, r.x + r.w]), ys = rooms.flatMap((r) => [r.y, r.y + r.h]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    this.overviewShot = this.shotForRect(minX, minY, maxX - minX, maxY - minY, 0.35, 0.7, 1.7, 0.7);
  }

  shotFor(room) {
    return this.shotForRect(room.x, room.y, room.w, room.h, 0.5, 0.55, 1.6, 0.55, this.headroom);
  }

  /* A shot = target + the half extents (metres) of the plan rectangle as seen from the fixed
     three-quarter direction: width across the screen, height up the screen (floor depth foreshortened
     by the elevation, plus room for walls, people and labels). `backShift` moves the target toward the
     back of the plan so decision cards floating above the back wall get headroom; `headroom` adds more
     of it (presenter mode, room shots only). */
  shotForRect(x, y, w, h, targetY, padW, padH, backShift = 0, headroom = 0) {
    if (this.compact) { backShift = 0; padH = Math.min(padH, 0.9); }
    else if (headroom) { padH += headroom; backShift += headroom * 0.6; }   // room for the decision card above the room
    const target = toWorld(x + w / 2, y + h / 2 - backShift * 100, targetY);
    const wm = w / 100, hm = h / 100;
    const halfW = 0.5 * (wm * Math.cos(AZIMUTH) + hm * Math.sin(AZIMUTH)) + padW;
    const halfH = 0.5 * (wm * Math.sin(AZIMUTH) + hm * Math.cos(AZIMUTH)) * Math.sin(ELEVATION) + padH;
    return { target, halfW, halfH };
  }

  /* Camera position that fits the shot's extents from the fixed three-quarter direction. */
  positionFor(shot) {
    const vHalf = THREE.MathUtils.degToRad(this.camera.fov) / 2;
    const hHalf = Math.atan(Math.tan(vHalf) * Math.max(0.5, this.camera.aspect));
    const dist = Math.max(shot.halfW / Math.tan(hHalf), shot.halfH / Math.tan(vHalf), 4) * 1.04;
    const dir = new THREE.Vector3(Math.sin(AZIMUTH) * Math.cos(ELEVATION), Math.sin(ELEVATION), Math.cos(AZIMUTH) * Math.cos(ELEVATION));
    return shot.target.clone().addScaledVector(dir, dist);
  }

  goTo(shot, instant = false) {
    const to = { pos: this.positionFor(shot), target: shot.target.clone() };
    if (instant || this.instant || !this.motion) {
      this.camera.position.copy(to.pos);
      this.controls.target.copy(to.target);
      this.controls.update();
      this.move = null;
      return;
    }
    this.move = { from: { pos: this.camera.position.clone(), target: this.controls.target.clone() }, to, t: 0 };
  }

  overview(instant = false) {
    this.mode = 'overview';
    this.focusRoom = null;
    this.goTo(this.overviewShot, instant);
    this.onChange?.();
  }

  focus(roomId, hold = HOLD_S) {
    const room = this.rooms.get(roomId);
    if (!room) return;
    this.holdUntil = this.now + hold;
    if (!this.auto || this.paused) return;
    if (this.mode === 'room' && this.focusRoom === roomId) return;
    this.mode = 'room';
    this.focusRoom = roomId;
    this.goTo(this.shotFor(room));
    this.onChange?.();
  }

  onManual() {
    this.move = null;   // the user grabbed the camera mid-move: hand it over
    if (!this.paused) { this.paused = true; this.onChange?.(); }
  }

  resume() {
    this.paused = false;
    this.auto = true;
    this.overview();
  }

  setAuto(on) {
    this.auto = on;
    this.paused = false;
    if (on) this.overview();
    this.onChange?.();
  }

  /* Re-frame the current shot (after a resize changes the aspect ratio). */
  refit() {
    if (this.paused || this.move) return;
    const shot = this.mode === 'room' && this.rooms.get(this.focusRoom) ? this.shotFor(this.rooms.get(this.focusRoom)) : this.overviewShot;
    this.goTo(shot, true);
  }

  tick(dt) {
    this.now += dt;
    if (this.move) {
      this.move.t = dt <= 0 ? 1 : Math.min(1, this.move.t + dt / MOVE_S);
      const k = smoothstep(this.move.t);
      this.camera.position.lerpVectors(this.move.from.pos, this.move.to.pos, k);
      this.controls.target.lerpVectors(this.move.from.target, this.move.to.target, k);
      this.controls.update();
      if (this.move.t >= 1) this.move = null;
    }
    if (this.mode === 'room' && this.auto && !this.paused && this.now > this.holdUntil) this.overview();
  }
}
