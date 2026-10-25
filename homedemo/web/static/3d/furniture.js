/* KayKit Furniture Bits placed from the snapshot's room rectangles (fractions of each room) so the
   layout follows home.json. The couch faces whatever the TV device faces from; beds, tables, chairs,
   lamps and rugs are keyed by room id with a plain fallback for unknown rooms. */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { FURNITURE_SCALE, toWorld } from './world.js';

export const FURNITURE_DIR = 'static/assets/furniture/';
export const FURNITURE_MODELS = [
  'couch_pillows', 'armchair', 'table_low', 'table_medium', 'chair_A', 'bed_double_A', 'bed_single_A',
  'lamp_standing', 'lamp_table', 'rug_rectangle_A', 'rug_oval_A', 'cabinet_small', 'shelf_B_small_decorated',
  'cactus_small_A', 'pictureframe_medium', 'book_set',
];

/* {model, fx, fy, yaw (radians, 0 = facing +z/south), scale?, y?}. fx/fy are fractions of the room. */
const LAYOUT = {
  living: [
    { model: 'rug_rectangle_A', fx: 0.42, fy: 0.5, yaw: Math.PI / 2, scale: 1.1 },
    { model: 'couch_pillows', fx: 0.58, fy: 0.5, faceTv: true },
    { model: 'table_low', fx: 0.32, fy: 0.5, yaw: Math.PI / 2, scale: 0.9 },
    { model: 'armchair', fx: 0.5, fy: 0.17, yaw: Math.PI * 0.75 },
    { model: 'lamp_standing', fx: 0.93, fy: 0.1 },
    { model: 'shelf_B_small_decorated', fx: 0.75, fy: 0.07 },
    { model: 'pictureframe_medium', fx: 0.3, fy: 0.03, y: 0.95 },
  ],
  kitchen: [
    { model: 'table_medium', fx: 0.42, fy: 0.68, yaw: Math.PI / 2, scale: 0.85 },
    { model: 'chair_A', fx: 0.3, fy: 0.68, yaw: Math.PI / 2, scale: 0.8 },
    { model: 'chair_A', fx: 0.54, fy: 0.68, yaw: -Math.PI / 2, scale: 0.8 },
    { model: 'chair_A', fx: 0.42, fy: 0.52, yaw: Math.PI, scale: 0.8 },
    { model: 'chair_A', fx: 0.42, fy: 0.84, yaw: 0, scale: 0.8 },
  ],
  hallway: [
    { model: 'pictureframe_medium', fx: 0.3, fy: 0.04, y: 0.9 },
  ],
  bedroom_main: [
    { model: 'bed_double_A', fx: 0.5, fy: 0.36, yaw: 0 },
    { model: 'cabinet_small', fx: 0.86, fy: 0.12, scale: 0.55 },
    { model: 'lamp_table', fx: 0.86, fy: 0.12, y: 0.52, scale: 0.5 },
    { model: 'rug_oval_A', fx: 0.5, fy: 0.78, yaw: Math.PI / 2, scale: 0.9 },
  ],
  bedroom_maya: [
    { model: 'bed_single_A', fx: 0.72, fy: 0.3, yaw: 0 },
    { model: 'rug_oval_A', fx: 0.4, fy: 0.62, scale: 0.8 },
    { model: 'cabinet_small', fx: 0.16, fy: 0.12, scale: 0.55 },
    { model: 'book_set', fx: 0.16, fy: 0.12, y: 0.52, scale: 0.5 },
    { model: 'lamp_table', fx: 0.4, fy: 0.12, y: 0, scale: 0.5 },
  ],
  porch: [
    { model: 'chair_A', fx: 0.88, fy: 0.5, yaw: 0, scale: 0.85 },
    { model: 'cactus_small_A', fx: 0.06, fy: 0.55, scale: 0.9 },
  ],
};
const FALLBACK = [{ model: 'rug_oval_A', fx: 0.5, fy: 0.5, scale: 0.8 }];

export class Furniture {
  constructor(scene) {
    this.group = new THREE.Group();
    this.group.name = 'furniture';
    scene.add(this.group);
    this.loader = new GLTFLoader();
    this.cache = new Map();
    this.ready = false;
  }

  async load(name) {
    if (!this.cache.has(name)) {
      this.cache.set(name, this.loader.loadAsync(`${FURNITURE_DIR}${name}.gltf`).then((gltf) => {
        gltf.scene.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
        return gltf.scene;
      }));
    }
    return this.cache.get(name);
  }

  /* Places every item for the snapshot's rooms. Missing models are skipped (logged), never fatal. */
  async build(snapshot) {
    for (const child of [...this.group.children]) this.group.remove(child);
    const tv = snapshot.devices.find((d) => d.kind === 'tv');
    const jobs = [];
    for (const room of snapshot.rooms) {
      const items = LAYOUT[room.id] || (room.indoor === false ? [] : FALLBACK);
      for (const item of items) {
        const x = room.x + room.w * item.fx, y = room.y + room.h * item.fy;
        let yaw = item.yaw ?? 0;
        if (item.faceTv && tv && tv.room === room.id) yaw = Math.atan2(tv.x - x, tv.y - y);
        jobs.push(this.load(item.model).then((proto) => {
          const obj = proto.clone();
          obj.scale.setScalar(FURNITURE_SCALE * (item.scale ?? 1));
          obj.position.copy(toWorld(x, y, item.y ?? 0));
          obj.rotation.y = yaw;
          obj.userData.room = room.id;
          this.group.add(obj);
        }).catch((err) => console.warn(`furniture ${item.model} failed`, err)));
      }
    }
    await Promise.all(jobs);
    this.ready = true;
  }
}
