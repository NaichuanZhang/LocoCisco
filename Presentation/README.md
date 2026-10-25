# LocoCisco — animated demo presentation

**[Open the live slides on GitHub Pages](https://naichuanzhang.github.io/LocoCisco/)**

Open **[index.html](index.html)** in a desktop browser. It is a five-slide, 16:9
presentation with local fonts, a vendored GSAP animation library, model explanations,
presenter notes and a local video player. No build step, CDN, backend or internet
connection is required. Keep this directory and its `assets/` folder together.

For a local HTTP preview, from this directory run:

```sh
python3 -m http.server 8890 --bind 127.0.0.1 --directory ..
```

Then open `http://127.0.0.1:8890/Presentation/`. Serving the public repository root also makes the evidence links available.

## Presenting

Each slide has deliberate animation builds. **Right arrow / Space** reveals the
next build, then advances to the next slide. **Left arrow** moves back.

| Control | Action |
|---|---|
| 1–5 or bottom navigation | Jump to a slide |
| Shift + Right / Left | Next / previous slide, skipping builds |
| A | Auto-advance one build every 8.5 seconds; pauses on the opening demo and final build |
| F | Fullscreen |
| N | Notes for the current slide |
| R | Replay the current slide from its first build |
| Home / End | First / last slide |
| ? | Keyboard help |
| Escape | Close a dialog |

On slide 3, click any of the **seven models** to reveal its mechanism, model-choice
rationale and runtime placement. The four buttons below the graph walk through the
pipeline. A small Decider threshold diagram describes the conditional Qwen route.

The deck uses a daylight palette, springy reveals and playful motion. It scales to
the viewport without cropping. Present in landscape, ideally 1920×1080. System
reduced-motion settings are respected. Browser printing shows all five slides with
their content revealed; enable background graphics for the colors.

## Demo video

Slide 1 initially shows a placeholder. **Choose demo video** opens a local file
picker; the browser plays a local object URL without uploading the recording.
The two-minute countdown follows playback and pauses with the video; it does not loop or stretch short clips. Use **Replace video** to select another file. MP4/H.264 is the most portable choice.
Playback is deliberately manual and pauses when you leave the slide. A chosen file
is not persisted across page reloads.

The optional **included source footage** button plays `Demo.mp4`. This file is the
existing input fixture, **not runtime output**. An on-screen label distinguishes
its baked-in names, translations and graphics from model inference. Use an accepted
screen recording for the final presentation. The earlier review capture is not
silently substituted.

## Content and evidence

The deck follows the current `BUILD_SPEC.md`, model configuration and implementation,
not the earlier broad `PRESENTATION_OUTLINE.md`. It covers:

1. **Opening demo**, with a local video slot and two-minute playback countdown.
2. Local-first, privacy-driven smart office product.
3. All seven enabled models and their decision/policy pipeline.
4. Dell GB10 compute placement and measured sensor timings.
5. **Use cases and market opportunity:** confidential conversations across businesses
   and government in the near term, then a private smart-home hub in the long term.

The final slide has three builds: near-term business and government use cases,
customer value and a pilot path, then the long-term smart-home vision. It ends on
the opportunity to bring private local intelligence wherever people talk.

[PRESENTER_NOTES.md](PRESENTER_NOTES.md) contains the talk track, exact source references
and evidence limits. The deck does not claim measured Cisco cost savings, certified
replacement compatibility, unlimited compute headroom or semantic accuracy. Speech
timings are attributed to a small recorded review sample. The model diagram is
explanatory; it is not live runtime telemetry. The final slide describes customer
opportunities and a product roadmap, rather than current production deployments or
certification for regulated settings.

## Files and credits

- `index.html`: semantic slide content.
- `presentation.css`: layout, theme, animation accents, reduced motion and print.
- `presentation.js`: build navigation, GSAP transitions, model details and video.
- `assets/gsap.min.js`: GSAP 3.14.2, copied from the existing design reference;
  upstream license notice retained in the file.
- `assets/*.woff2`: existing Figtree and JetBrains Mono fonts, SIL OFL licenses included.
- `assets/lococisco-*.svg`: existing project brand assets.
- `assets/logo/`: the earlier ribbon + Dell/NVIDIA avatar finale, preserved as a
  standalone optional asset. It is outside the current five-slide sequence.
- `assets/avatar-*.svg`: current mainline character artwork.
- `assets/memory-snapshot.json`: the earlier point-in-time GPU allocation evidence,
  preserved for reference; the current closing slide focuses on use cases.
- The Dell device is a stylized vector illustration, not a hardware photograph.

Existing source media, PowerPoint and outline files are preserved.

## GitHub Pages

The hosted deck is published from the [`lococisco` branch](https://github.com/NaichuanZhang/LocoCisco/tree/lococisco),
using `/ (root)` as the Pages source. That branch contains the static presentation
and `.nojekyll`; the editable source stays here on `main`.

The publishing branch's README records the source revision and release steps.
Changes on `main` need a new release to `lococisco` to update the hosted slides.
The hosted evidence and full-notes links open the matching source revision on
GitHub. Local video selection still stays in the browser without uploading.
