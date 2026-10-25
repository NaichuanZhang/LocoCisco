# LocoCisco logo and avatar finale

Responsive, day-themed 6.8-second finale adapted from the complete updated
`design-reference/index.html` at mainline commit `6de9ded`. The original ribbon
writes LocoCisco, followed by the Dell box and NVIDIA NemoClaw avatars hopping in,
bumping fist to claw, and revealing the Dell × NVIDIA marks.

The animation plays once and holds the final frame. It has no audio, looping
player, external assets, or background rendering after completion. Reduced-motion
users see the final logo, both avatars, and both brand marks immediately.

Embed `index.html` in an iframe. On readiness the iframe posts
`{type: "loco-logo-ready"}` to its parent. Parent commands are
`{type: "loco-logo", action: "play" | "pause" | "finish"}`; only messages
whose source is `window.parent` are accepted. `play` restarts the animation.

For testing, `window.LogoAnimation` exposes `play()`, `pause()`, `finish()`,
`seek(seconds)` and `state()` (time, duration, playing, reducedMotion, avatars).
`window.logoReady` resolves after the font and renderer are ready.

Dependencies are local: `../gsap.min.js`, the three adjacent data scripts, and
`fonts/figtree-800.woff2` (SIL OFL license included). The two standalone avatar
SVGs in `avatars/` are also provided for use elsewhere in the deck. The logo,
avatars, and partner marks remain the supplied mainline artwork.
