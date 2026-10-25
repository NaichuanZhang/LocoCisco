// LocoCisco intro — the two collab avatars, drawn once and shared by the film (../index.html)
// and the standalone SVGs (../../tools/export-avatars.mjs → avatars/*.svg).
// Both are original characters in brand colours, not brand artwork:
//   dell   a squat graphite mini-desktop standing in for the Dell Pro Max with GB10
//   nemo   NVIDIA NemoClaw: a chubby green lobster (OpenClaw's claw) in a clear sandbox dome (OpenShell)
// The Dell box wears the real Dell logo as a badge (assets/brand-logos.js must load first).
// Local coordinates: (0, 0) is the point between the feet; up is negative y. Parts the film
// animates have ids (arms, eyes, pupils, claws, antennae) and pivots listed in `rig`.
(function (root) {
  // A pincer: a disc with a wedge opening `open` (radians) facing direction `dir` (0 = right, y down).
  function clawPath(cx, cy, r, dir, open) {
    const a1 = dir + open / 2, a2 = dir - open / 2, f = (v) => v.toFixed(2);
    return `M${f(cx + r * Math.cos(a1))} ${f(cy + r * Math.sin(a1))}A${r} ${r} 0 1 1 ${f(cx + r * Math.cos(a2))} ${f(cy + r * Math.sin(a2))}` +
      `L${f(cx + r * 0.12 * Math.cos(dir))} ${f(cy + r * 0.12 * Math.sin(dir))}Z`;
  }

  // Dell box: vent slots set on a smile curve.
  const vents = [-24, -12, 0, 12, 24].map((x) => {
    const y = -68 - 0.012 * x * x;
    return `<rect x="${x - 3}" y="${(y - 6).toFixed(1)}" width="6" height="12" rx="3" fill="#5E6672"/>`;
  }).join('');
  const dellEye = (side, x) => `<g id="dl-eye-${side}"><ellipse cx="${x}" cy="-108" rx="15" ry="19" fill="#F3F7FB"/>` +
    `<g id="dl-pup-${side}"><circle cx="${x + 2}" cy="-105" r="8.5" fill="#101215"/><circle cx="${x - 2}" cy="-111" r="3.2" fill="#fff"/></g></g>`;
  const dellArm = (side, s) => `<g id="dl-arm-${side}"><path d="M${86 * s} -96L${108 * s} -64" stroke="#2C3036" stroke-width="20" stroke-linecap="round"/>` +
    `<circle cx="${108 * s}" cy="-64" r="12" fill="#3A3F47"/></g>`;
  // The real Dell mark, reversed out in light grey on the graphite case (from brand-logos.js).
  const B = root.BRANDS || {};
  const dellBadge = B.dell ? `<svg x="-15" y="-55" width="30" height="30" viewBox="${B.dell.viewBox}">${B.dell.markup.replace(/#007db8/gi, '#E6ECF2')}</svg>` : '';
  const dell = {
    defs: '<linearGradient id="dl-body" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#4A5059"/><stop offset=".45" stop-color="#31363D"/><stop offset="1" stop-color="#22262B"/></linearGradient>' +
      '<linearGradient id="dl-face" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#16191D"/><stop offset="1" stop-color="#0D0F12"/></linearGradient>' +
      '<radialGradient id="dl-led"><stop offset="0" stop-color="#8FD0FF"/><stop offset=".55" stop-color="#0076CE"/><stop offset="1" stop-color="#0076CE" stop-opacity="0"/></radialGradient>',
    shadow: '<ellipse cx="0" cy="4" rx="78" ry="11" fill="#000" opacity="0.14"/>',
    body: dellArm('l', -1) + dellArm('r', 1) +
      '<rect x="-54" y="-30" width="34" height="26" rx="10" fill="#1A1D21"/><rect x="20" y="-30" width="34" height="26" rx="10" fill="#1A1D21"/>' +
      '<rect x="-88" y="-178" width="176" height="156" rx="36" fill="url(#dl-body)"/>' +
      '<rect x="-74" y="-172" width="148" height="12" rx="6" fill="#fff" opacity="0.1"/>' +
      '<path d="M-70 -146Q-70 -166 -50 -166L-14 -166" stroke="#fff" stroke-opacity="0.32" stroke-width="6" stroke-linecap="round" fill="none"/>' +
      '<rect x="-68" y="-152" width="136" height="94" rx="24" fill="url(#dl-face)"/>' +
      dellEye('l', -30) + dellEye('r', 30) +
      '<ellipse cx="-50" cy="-80" rx="9" ry="5" fill="#FF7A93" opacity="0.45"/><ellipse cx="50" cy="-80" rx="9" ry="5" fill="#FF7A93" opacity="0.45"/>' +
      vents +
      '<circle cx="52" cy="-70" r="10" fill="url(#dl-led)"/><circle cx="52" cy="-70" r="3.6" fill="#CDEBFF"/>' +
      dellBadge +
      '<g transform="translate(-64 -188) rotate(-10)"><rect x="0" y="0" width="52" height="21" rx="6" fill="#76B900"/>' +
      '<text x="26" y="15.5" text-anchor="middle" font-family="Figtree, system-ui, sans-serif" font-weight="800" font-size="13" fill="#173300">GB10</text></g>',
    eyes: { l: [-30, -108], r: [30, -108] },
    rig: { shoulder: { l: [-86, -96], r: [86, -96] }, armLen: Math.hypot(22, 32), fistR: 12 },
    box: [-132, -204, 264, 224],
  };

  // NemoClaw: claws hang off short arms; eyes on stalks and curly antennae sit inside the dome.
  const nemoEye = (side, x) => `<g id="nc-eye-${side}"><circle cx="${x}" cy="-160" r="16" fill="#F3F7FB"/>` +
    `<g id="nc-pup-${side}"><circle cx="${x + 2 * Math.sign(x) * -1}" cy="-157" r="8" fill="#101215"/><circle cx="${x - 3}" cy="-165" r="3" fill="#fff"/></g></g>`;
  const nemoArm = (side, s) => `<g id="nc-arm-${side}"><path d="M${50 * s} -74Q${78 * s} -78 ${96 * s} -100" stroke="#6AA80A" stroke-width="14" stroke-linecap="round" fill="none"/>` +
    `<path id="nc-claw-${side}" d="${clawPath(100 * s, -104, 26, s > 0 ? -Math.PI / 4 : -3 * Math.PI / 4, 0.9)}" fill="url(#nc-claw)"/>` +
    `<circle cx="${92 * s}" cy="-116" r="5" fill="#fff" opacity="0.45"/></g>`;
  const antenna = (side, s) => `<path id="nc-ant-${side}" d="M${6 * s} -120C${8 * s} -150 ${30 * s} -160 ${28 * s} -184C${27 * s} -194 ${18 * s} -194 ${19 * s} -186" stroke="#6AA80A" stroke-width="4.5" stroke-linecap="round" fill="none"/>`;
  const nemo = {
    defs: '<radialGradient id="nc-body" cx=".38" cy=".3" r=".8"><stop offset="0" stop-color="#B5E553"/><stop offset=".45" stop-color="#84C41A"/><stop offset="1" stop-color="#5C9600"/></radialGradient>' +
      '<radialGradient id="nc-claw" cx=".35" cy=".3" r=".85"><stop offset="0" stop-color="#A6DD3F"/><stop offset=".55" stop-color="#76B900"/><stop offset="1" stop-color="#4E8500"/></radialGradient>' +
      '<linearGradient id="nc-glass" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#DCF4FF" stop-opacity=".55"/><stop offset=".6" stop-color="#C8E8FA" stop-opacity=".16"/><stop offset="1" stop-color="#B4DCF5" stop-opacity=".3"/></linearGradient>',
    shadow: '<ellipse cx="0" cy="4" rx="74" ry="11" fill="#000" opacity="0.14"/>',
    body: '<g stroke="#5C9600" stroke-width="7" stroke-linecap="round"><path d="M-40 -18L-52 -2"/><path d="M-22 -10L-28 2"/><path d="M22 -10L28 2"/><path d="M40 -18L52 -2"/></g>' +
      nemoArm('l', -1) + nemoArm('r', 1) +
      '<ellipse cx="0" cy="-66" rx="64" ry="60" fill="url(#nc-body)"/>' +
      '<ellipse cx="0" cy="-46" rx="36" ry="30" fill="#D4F28A" opacity="0.5"/>' +
      '<path d="M-30 -50Q0 -40 30 -50M-24 -34Q0 -26 24 -34" stroke="#7DB51C" stroke-width="3" stroke-linecap="round" fill="none" opacity="0.75"/>' +
      '<path d="M-11 -92Q0 -83 11 -92" stroke="#2F5200" stroke-width="3.5" stroke-linecap="round" fill="none"/>' +
      '<ellipse cx="-32" cy="-86" rx="9" ry="5" fill="#FF7A93" opacity="0.45"/><ellipse cx="32" cy="-86" rx="9" ry="5" fill="#FF7A93" opacity="0.45"/>' +
      '<path d="M-16 -118L-24 -148M16 -118L24 -148" stroke="#6AA80A" stroke-width="8" stroke-linecap="round"/>' +
      antenna('l', -1) + antenna('r', 1) + nemoEye('l', -25) + nemoEye('r', 25) +
      '<path d="M-58 -116A58 96 0 0 1 58 -116Z" fill="url(#nc-glass)" stroke="#9CC8E0" stroke-opacity="0.75" stroke-width="3"/>' +
      '<path d="M-40 -146A40 70 0 0 1 -16 -194" stroke="#fff" stroke-opacity="0.8" stroke-width="6" stroke-linecap="round" fill="none"/>' +
      '<rect x="-62" y="-124" width="124" height="14" rx="7" fill="#C7D0D9"/><rect x="-54" y="-121" width="60" height="3" rx="1.5" fill="#fff" opacity="0.7"/>',
    eyes: { l: [-25, -160], r: [25, -160] },
    rig: { shoulder: { l: [-50, -74], r: [50, -74] }, claw: { l: [-100, -104], r: [100, -104] }, clawR: 26,
      dir: { l: -3 * Math.PI / 4, r: -Math.PI / 4 }, open: 0.9, antBase: { l: [-6, -120], r: [6, -120] } },
    box: [-140, -224, 280, 244],
  };

  root.AVATARS = { scale: 0.9, clawPath, dell, nemo };
})(typeof window !== 'undefined' ? window : globalThis);
