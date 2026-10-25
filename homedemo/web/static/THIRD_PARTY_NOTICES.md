# Third-party notices for `homedemo/web/static`

Everything in this directory is served locally by the demo; nothing is fetched from a CDN or any
other remote host at runtime. The files below were copied from their packages unchanged (apart from
selecting a subset).

## three.js 0.186.1 — MIT

Vendored under `vendor/three/` keeping the package layout:
`build/three.module.js`, `build/three.core.js`, `examples/jsm/controls/OrbitControls.js`,
`examples/jsm/loaders/GLTFLoader.js`, `examples/jsm/renderers/CSS2DRenderer.js`,
`examples/jsm/utils/BufferGeometryUtils.js`, `examples/jsm/utils/SkeletonUtils.js` (the two utils are
imported by GLTFLoader), plus the package's `LICENSE` and `package.json` (version pin).

The MIT License

Copyright © 2010-2026 three.js authors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

(The verbatim licence file is `vendor/three/LICENSE`.)

## Kenney — Mini Characters 1 (1.0) — CC0 1.0 Universal

`assets/people/character-*.glb` and `assets/people/Textures/colormap.png`, with the pack's
`assets/people/License.txt`. Created and distributed by Kenney (www.kenney.nl). CC0: no attribution
required; credited here with thanks.

## Kenney — Cube Pets (1.0) — CC0 1.0 Universal

`assets/pets/animal-cat.glb` and `assets/pets/Textures/colormap.png`, with the pack's
`assets/pets/License.txt`. Created and distributed by Kenney (www.kenney.nl); downloaded from the
OpenGameArt mirror of the pack. CC0: no attribution required; credited here with thanks.

## KayKit — Furniture Bits (1.0) — CC0 1.0 Universal

`assets/furniture/*.gltf`, `assets/furniture/*.bin` and `assets/furniture/furniturebits_texture.png`,
with the pack's `assets/furniture/LICENSE.txt`. Created and distributed by Kay Lousberg
(www.kaylousberg.com). CC0: no attribution required; credited here with thanks.

## Everything else

`index.html`, `style.css`, `app.js` and `3d/*.js|*.css` are part of this demo. The kitchen counter,
stove, fridge, TV, front door, thermostat, home hub, microphone pucks and camera domes are procedural
three.js geometry, not third-party assets.
