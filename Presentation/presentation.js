/* LocoCisco presentation. Local assets only; no network services or inference. */
(() => {
  'use strict';
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const slides = $$('.slide');
  const deck = $('#deck');
  const motionQuery = matchMedia('(prefers-reduced-motion: reduce)');
  const state = { slide: 0, build: 0, auto: false, model: 'silero' };
  let animation, autoTimer, toastTimer, videoURL;
  const video = $('#demo-video');
  const modelOrder = ['silero', 'parakeet', 'campp', 'yunet', 'lightasd', 'decider', 'qwen'];
  const models = {
    silero: {
      phase: 0, category: 'SPEECH DETECTION', name: 'Silero VAD v5',
      runtime: 'ONNX · CPU · continuous speech gate',
      how: 'Scores short audio windows for speech. A pause closes the segment sent to transcription.',
      why: 'A lightweight CPU gate keeps silence out of the heavier inference pipeline.',
      fact: '400 ms silence · 8 s segment cap', graphic: 'wave'
    },
    parakeet: {
      phase: 0, category: 'SPEECH RECOGNITION', name: 'Parakeet TDT 0.6B v3',
      runtime: 'Q8 · GPU · NeMo-Speech.cpp',
      how: 'A token-and-duration transducer converts audio into words and their timestamps.',
      why: 'Compact speech specialist. Project tests found similar word error to Whisper turbo with less CPU compute.',
      fact: '0.6B parameters · quantized speech service', graphic: 'words'
    },
    campp: {
      phase: 0, category: 'ANONYMOUS VOICE LABELS', name: 'WeSpeaker CAM++',
      runtime: 'ONNX · CPU · session-local voice clusters',
      how: 'Embeds short speech windows, then compares them with voice clusters to label the speaker.',
      why: 'Won the project’s tested diarization comparison. No persistent speaker enrollment required.',
      fact: '1.5 s windows · 0.75 s hops · up to 4 voices', graphic: 'voices'
    },
    yunet: {
      phase: 1, category: 'COMPUTER VISION', name: 'YuNet 2023mar',
      runtime: 'OpenCV · CPU · face detection + tracking',
      how: 'Detects face boxes. Overlap-based tracking links them into temporary face tracks.',
      why: 'A small CPU detector supplies face crops without spending the GPU budget on recognition.',
      fact: 'Anonymous tracks · not face recognition', graphic: 'face'
    },
    lightasd: {
      phase: 1, category: 'AUDIOVISUAL SPEAKER EVIDENCE', name: 'Light-ASD',
      runtime: 'AVA weights · PyTorch · GPU',
      how: 'Combines face motion and audio features over time to score which visible person is speaking.',
      why: 'Learned audiovisual evidence beat the project’s motion-only baseline. A face alone is not enough.',
      fact: '1 s windows · ~0.2 s scoring cadence', graphic: 'face'
    },
    decider: {
      phase: 2, category: 'BOUNDED DECISIONS', name: 'Strands Decider 12B',
      runtime: 'Gemma 4 base + adapter · local GPU service',
      how: 'Answers typed questions with scored options: actionable intent, action kind, owner and sensitivity.',
      why: 'Inspectable choices instead of free-form action text. Routes only uncertain cases to the local LLM.',
      fact: 'p ≥ .60 → policy · .40 ≤ p < .60 → Qwen\np < .40 → normally no action', graphic: 'choices'
    },
    qwen: {
      phase: 3, category: 'LOCAL GENERATION', name: 'Qwen3.6 35B-A3B',
      runtime: 'NVFP4 · vLLM · GPU · sparse experts',
      how: 'Activates a subset of experts to generate tool proposals and answers grounded in retained meeting text.',
      why: '35B model capacity with ~3B active parameters. Quantization reduces the weight footprint.',
      fact: 'On-demand reasoning · every action still meets policy', graphic: 'experts'
    }
  };
  const notes = [
    ['The vision',
      'LocoCisco is a local-first, privacy-driven smart office system. The first use case is the meeting room: speech and computer vision provide context, a bounded Decider identifies what matters, and a local language model generates proposals and grounded answers.',
      'Advance twice: reveal local/private/office-aware pillars and captions, then useful follow-through. Emphasize that the intelligence runs on a Dell edge box without a wake word or required cloud inference.',
      'These are product capabilities and an architecture illustration. Generated items are proposals; real device and notification integrations are local previews in the current demo.'],
    ['Seven specialists, one pipeline',
      'Advance through audio, vision, Decider, and generation plus policy. Click any model to explain its mechanism and the reason for its selection. Three CPU roles support four GPU roles.',
      'Silero finds speech. Parakeet gives words; CAM++ gives anonymous voices. YuNet tracks faces; Light-ASD combines audio and face motion. Code fuses the evidence. Decider 12B asks bounded questions; Qwen generates only when needed or explicitly asked.',
      'Decider ≥0.60 routes directly to policy, 0.40–0.60 may escalate, and <0.40 normally produces no action. These are routing thresholds, not calibrated accuracy. Captions do not wait for Decider. VLM, TTS and a second Decider are disabled.'],
    ['The Dell advantage',
      'This is the Dell Pro Max with NVIDIA GB10 and 128 GB unified memory. The architecture puts small continuous models on CPU and heavy inference on GPU. We reuse loaded services rather than reloading weights for every request.',
      'Advance to discuss quantization and Qwen’s sparse experts, then reveal actual measurements: 44.8 ms average ASR request and 339.6 ms p95 from speech end to segment publication.',
      'Measured in a 30.08-second official-source review with five speech segments. Seven roles were ready; zero sensor errors were reported. These are sensor timings, not completed decision latency. The review dropped frames and is not proof of spare capacity or production throughput.'],
    ['The commercial opportunity',
      'The commercial direction is an edge appliance that can replace traditional Cisco room hardware over time: Dell hardware, LocoCisco software, deployment and support. Start with privacy-conscious teams modernizing one meeting room.',
      'Advance to reveal upgradeable software, then the cost thesis: consolidate models on owned compute, reuse suitable room peripherals and avoid metered cloud inference for the core workflow.',
      'Lower total cost is an ambition, not a quoted saving. There is no Cisco pricing study, validated replacement qualification, production SLA or proven multi-room capacity in this project. Off-record does not erase original or already-exported recordings.'],
    ['The demo',
      'Choose a final demo recording from this device. The browser uses a local object URL; the file is not uploaded. Video playback starts only after you press play. Arrow keys are left to the player when it has focus.',
      'Point out speech plus visual speaker evidence, the Decider and any actual Qwen escalation, then the explicit privacy controls. Finish on “Your room. Your intelligence.”',
      'The optional included Demo.mp4 is source footage, not a recording of runtime output. Its names, translations and visual labels are baked into the input. The placeholder does not automatically load the earlier low-frame-rate review capture.']
  ];

  function resize() {
    const scale = Math.min(innerWidth / 1600, innerHeight / 900);
    deck.style.transform = `scale(${scale})`;
    $('#stage-wrap').style.width = `${1600 * scale}px`;
    $('#stage-wrap').style.height = `${900 * scale}px`;
  }
  function animate(targets, from, to) {
    if (!targets || (Array.isArray(targets) && !targets.length)) return;
    if (window.gsap && !motionQuery.matches) return gsap.fromTo(targets, from, to);
    const nodes = typeof targets === 'string' ? $$(targets) : (targets instanceof Element ? [targets] : targets);
    [...nodes].forEach(el => {
      el.style.opacity = String(to.opacity ?? 1);
      el.style.visibility = 'visible';
      el.style.transform = '';
    });
  }
  function setBuildVisibility(slide, build, withAnimation) {
    $$('.build', slide).forEach(el => {
      if (window.gsap) gsap.killTweensOf(el);
      const visible = Number(el.dataset.build) <= build;
      const changed = el.dataset.revealed !== String(visible);
      el.dataset.revealed = String(visible);
      el.style.visibility = visible ? 'visible' : 'hidden';
      el.style.opacity = visible ? '1' : '0';
      el.inert = !visible;
      el.setAttribute('aria-hidden', String(!visible));
      if (visible && changed && withAnimation) animate(el, {opacity: 0, y: 18}, {opacity: 1, y: 0, duration: .65, ease: 'power3.out'});
    });
  }
  function updateControls() {
    $$('[data-go]').forEach((button, index) => {
      button.classList.toggle('active', index === state.slide);
      if (index === state.slide) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
    const max = Number(slides[state.slide].dataset.builds);
    $('#build-count').textContent = state.slide === 4 ? 'PRESS PLAY' : `BUILD ${state.build + 1} / ${max + 1}`;
    $('#previous-button').disabled = state.slide === 0 && state.build === 0;
    $('#next-button').disabled = state.slide === slides.length - 1 && state.build === max;
    $('#deck-progress-fill').style.width = `${((state.slide + (state.build + 1) / (max + 1)) / slides.length) * 100}%`;
    $('#slide-announcer').textContent = `Slide ${state.slide + 1} of 5: ${slides[state.slide].dataset.title}. Build ${state.build + 1} of ${max + 1}.`;
    const [title, ...paragraphs] = notes[state.slide];
    $('#notes-title').textContent = title;
    $('#notes-content').replaceChildren(...paragraphs.map((text, index) => {
      const p = document.createElement('p'); p.textContent = text;
      if (index === paragraphs.length - 1) p.className = 'notes-caution';
      return p;
    }));
    const hash = `#${slides[state.slide].id}/${state.build}`;
    try { history.replaceState(null, '', hash); } catch (_) { /* Some file URL viewers disallow history. */ }
  }
  function modelGraphic(kind) {
    if (kind === 'wave') return `<div class="detail-wave">${Array.from({length: 49}, (_, i) => `<i style="height:${8 + Math.abs(Math.sin(i * .73) * Math.sin(i * .21)) * 36}px;animation-delay:-${(i * .07).toFixed(2)}s"></i>`).join('')}</div>`;
    if (kind === 'words') return '<span class="graphic-label">audio → Aa</span>';
    if (kind === 'voices') return '<svg viewBox="0 0 300 60"><circle cx="36" cy="27" r="20"/><circle cx="128" cy="27" r="20"/><circle cx="220" cy="27" r="20"/><text x="30" y="32" fill="currentColor" stroke="none" font-size="14">A</text><text x="122" y="32" fill="currentColor" stroke="none" font-size="14">B</text><text x="214" y="32" fill="currentColor" stroke="none" font-size="14">C</text><path d="M62 27h40m51 0h41" stroke-dasharray="3 4"/></svg>';
    if (kind === 'face') return '<svg viewBox="0 0 300 60"><path d="M15 19V7h17m44 0h17v12M15 39v12h17m44 0h17V39"/><ellipse cx="54" cy="28" rx="14" ry="19"/><path d="M49 34q5 5 10 0M110 28h40m20-6v12m7-19v26m7-34v42m7-29v18m7-16v14m7-23v31m7-21v13m7-26v39m7-29v19"/></svg>';
    if (kind === 'choices') return '<svg viewBox="0 0 300 60"><rect x="0" y="4" width="189" height="10" rx="4" fill="currentColor" opacity=".6" stroke="none"/><rect x="0" y="25" width="98" height="10" rx="4" fill="currentColor" opacity=".3" stroke="none"/><rect x="0" y="46" width="39" height="10" rx="4" fill="currentColor" opacity=".15" stroke="none"/><path d="m213 9 5 5 9-10"/></svg>';
    return '<span class="graphic-label">35B → ~3B active</span>';
  }
  function selectModel(key, withAnimation = true) {
    const model = models[key]; if (!model) return;
    state.model = key;
    $$('.model-node').forEach(node => {
      const selected = node.dataset.model === key;
      node.classList.toggle('selected', selected);
      node.setAttribute('aria-pressed', String(selected));
    });
    $('#detail-category').textContent = `${String(modelOrder.indexOf(key) + 1).padStart(2, '0')} / ${model.category}`;
    $('#detail-index').textContent = `${modelOrder.indexOf(key) + 1} / 7`;
    $('#detail-name').textContent = model.name;
    $('#detail-runtime').textContent = model.runtime;
    $('#detail-how').textContent = model.how;
    $('#detail-why').textContent = model.why;
    $('#detail-fact').textContent = model.fact;
    $('#detail-graphic').innerHTML = modelGraphic(model.graphic);
    if (withAnimation) animate($$('.model-detail h3, .detail-runtime, .detail-block, .detail-fact, .detail-graphic'), {opacity: 0, y: 7}, {opacity: 1, y: 0, duration: .35, stagger: .025, ease: 'power2.out', overwrite: true});
  }
  function setPhase(phase, withAnimation) {
    $('.model-map').dataset.activePhase = String(phase);
    $$('[data-phase-jump]').forEach(button => {
      button.classList.toggle('current', Number(button.dataset.phaseJump) === phase);
      button.setAttribute('aria-pressed', String(Number(button.dataset.phaseJump) === phase));
    });
    selectModel(['silero', 'lightasd', 'decider', 'qwen'][phase], withAnimation);
    if (withAnimation) {
      const lane = $(`[data-phase="${phase}"]`);
      animate(lane, {x: -5}, {x: 0, duration: .5, ease: 'power2.out'});
    }
  }
  function animateHardware(build, withAnimation) {
    const gpu = $('.gpu-lane');
    gpu.style.opacity = build === 0 ? '.42' : '1';
    if (build === 1 && withAnimation) animate($$('.gpu-lane .compute-chip'), {opacity: .3, x: 14}, {opacity: 1, x: 0, duration: .45, stagger: .09, ease: 'power2.out'});
    if (build === 2 && withAnimation && window.gsap && !motionQuery.matches) {
      $$('[data-counter]').forEach(el => {
        const counter = {value: 0};
        gsap.to(counter, {value: Number(el.dataset.counter), duration: 1.2, ease: 'power2.out', onUpdate: () => el.textContent = counter.value.toFixed(1)});
      });
    }
  }
  function render(withAnimation = true, entering = false) {
    const current = slides[state.slide];
    // Finish interrupted entrances so fast navigation never leaves a slide transparent.
    if (animation) { animation.progress(1); animation.kill(); animation = null; }
    slides.forEach((slide, index) => {slide.hidden = index !== state.slide; slide.classList.toggle('active', index === state.slide); slide.inert = index !== state.slide;});
    if (window.gsap) gsap.set($$('.enter', current), {opacity: 1, y: 0});
    else $$('.enter', current).forEach(el => {el.style.opacity = '1'; el.style.transform = '';});
    setBuildVisibility(current, state.build, withAnimation);
    if (entering && withAnimation) {
      animation = animate($$('.enter', current), {opacity: 0, y: 17}, {opacity: 1, y: 0, duration: .7, stagger: .045, ease: 'power3.out', overwrite: true});
    }
    if (state.slide === 1) setPhase(state.build, withAnimation);
    if (state.slide === 2) animateHardware(state.build, withAnimation);
    if (state.slide !== 4) video.pause();
    if (state.slide === 4 && state.auto) setAuto(false);
    updateControls();
  }
  function goTo(index, build = 0, withAnimation = true) {
    const next = Math.max(0, Math.min(slides.length - 1, index));
    const entering = next !== state.slide;
    state.slide = next;
    state.build = Math.max(0, Math.min(Number(slides[next].dataset.builds), build));
    render(withAnimation, entering);
    restartAutoTimer();
  }
  function next() {
    if (state.build < Number(slides[state.slide].dataset.builds)) goTo(state.slide, state.build + 1);
    else if (state.slide < slides.length - 1) goTo(state.slide + 1);
  }
  function previous() {
    if (state.build > 0) goTo(state.slide, state.build - 1);
    else if (state.slide > 0) goTo(state.slide - 1, Number(slides[state.slide - 1].dataset.builds));
  }
  function restartAutoTimer() {
    clearTimeout(autoTimer);
    if (state.auto && state.slide < 4 && !document.hidden && !$$('dialog[open]').length) autoTimer = setTimeout(next, 8500);
  }
  function setAuto(enabled) {
    state.auto = enabled && state.slide !== 4;
    const button = $('#autoplay-button');
    button.setAttribute('aria-pressed', String(state.auto));
    button.setAttribute('aria-label', state.auto ? 'Pause auto-advance' : 'Start auto-advance');
    button.textContent = state.auto ? 'Ⅱ' : '▷';
    restartAutoTimer();
  }
  function toast(message) {
    clearTimeout(toastTimer);
    $('#toast').textContent = message; $('#toast').hidden = false;
    toastTimer = setTimeout(() => $('#toast').hidden = true, 3200);
  }
  async function fullscreen() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (document.documentElement.requestFullscreen) await document.documentElement.requestFullscreen();
      else toast('Use your browser’s fullscreen command.');
    } catch (_) { toast('Use your browser’s fullscreen command.'); }
  }
  function toggleDialog(id) {
    const dialog = $(id);
    if (dialog.open) dialog.close();
    else {
      $$('dialog[open]').forEach(other => other.close());
      dialog.showModal(); clearTimeout(autoTimer);
    }
  }
  function loadVideo(source, label, isSource) {
    video.pause();
    $('#video-error').hidden = true;
    video.src = source; video.hidden = false;
    $('#video-placeholder').hidden = true;
    $('#replace-video').hidden = false;
    $('#video-caption').hidden = false;
    $('#video-caption').textContent = isSource ? 'SOURCE FOOTAGE · Baked-in names, translations and graphics are not runtime output.' : `LOCAL RECORDING · ${label}`;
    video.load(); setAuto(false);
  }
  function releaseVideoURL() {
    if (videoURL) URL.revokeObjectURL(videoURL);
    videoURL = undefined;
  }
  function readHash() {
    const [id, build] = location.hash.replace('#', '').split('/');
    const index = slides.findIndex(slide => slide.id === id);
    return {index: index < 0 ? 0 : index, build: /^\d+$/.test(build || '') ? Number(build) : 0};
  }

  $$('[data-go]').forEach(button => button.addEventListener('click', () => goTo(Number(button.dataset.go))));
  $$('[data-phase-jump]').forEach(button => button.addEventListener('click', () => goTo(1, Number(button.dataset.phaseJump))));
  $$('.model-node').forEach(button => button.addEventListener('click', () => {
    const key = button.dataset.model;
    goTo(1, models[key].phase);
    selectModel(key);
  }));
  $('.brand').addEventListener('click', event => {event.preventDefault(); goTo(0);});
  $('#next-button').addEventListener('click', next);
  $('#previous-button').addEventListener('click', previous);
  $('#autoplay-button').addEventListener('click', () => {setAuto(!state.auto); toast(state.auto ? 'Auto-advance: one build every 8.5 seconds' : 'Auto-advance paused');});
  $('#fullscreen-button').addEventListener('click', fullscreen);
  $('#help-button').addEventListener('click', () => toggleDialog('#help-dialog'));
  $('#notes-button').addEventListener('click', () => toggleDialog('#notes-dialog'));
  $$('[data-close-dialog]').forEach(button => button.addEventListener('click', () => button.closest('dialog').close()));
  $$('dialog').forEach(dialog => {
    dialog.addEventListener('close', restartAutoTimer);
    dialog.addEventListener('click', event => {if (event.target === dialog) {const rect = dialog.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();}});
  });
  $('#choose-video').addEventListener('click', () => $('#video-input').click());
  $('#replace-video').addEventListener('click', () => $('#video-input').click());
  $('#video-input').addEventListener('change', event => {
    const file = event.target.files[0]; if (!file) return;
    if (!(file.type.startsWith('video/') || /\.(mp4|webm|mov|m4v|ogv)$/i.test(file.name))) {
      $('#video-error').textContent = 'Choose a video file, such as MP4 or WebM.'; $('#video-error').hidden = false; event.target.value = ''; return;
    }
    releaseVideoURL(); videoURL = URL.createObjectURL(file);
    loadVideo(videoURL, file.name, false);
    event.target.value = '';
  });
  $('#load-source').addEventListener('click', () => {releaseVideoURL(); loadVideo('Demo.mp4', 'Demo.mp4', true);});
  video.addEventListener('error', () => {
    if (!video.getAttribute('src')) return;
    $('#video-error').textContent = 'This video could not be played. Choose an MP4 with H.264 video or a supported WebM recording.';
    $('#video-error').hidden = false;
    video.hidden = true; $('#video-placeholder').hidden = false; $('#video-caption').hidden = true;
  });
  video.addEventListener('play', () => setAuto(false));
  document.addEventListener('keydown', event => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.target instanceof Element && event.target.closest('input,textarea,select,[contenteditable="true"],video')) return;
    if ($$('dialog[open]').length) {if (event.key === 'Escape') return; if (event.key === '?' || event.key.toLowerCase() === 'n') {event.preventDefault(); $$('dialog[open]').forEach(dialog => dialog.close());} return;}
    if ((event.key === ' ' || event.key === 'Enter') && event.target instanceof Element && event.target.closest('button,a')) return;
    const key = event.key.toLowerCase();
    if (['arrowright', 'pagedown', ' '].includes(key)) {event.preventDefault(); event.shiftKey ? goTo(state.slide + 1) : next();}
    else if (['arrowleft', 'pageup'].includes(key)) {event.preventDefault(); event.shiftKey ? goTo(state.slide - 1) : previous();}
    else if (/^[1-5]$/.test(key)) {event.preventDefault(); goTo(Number(key) - 1);}
    else if (key === 'home') {event.preventDefault(); goTo(0);}
    else if (key === 'end') {event.preventDefault(); goTo(4);}
    else if (key === 'r') {event.preventDefault(); goTo(state.slide, 0);}
    else if (key === 'a') {event.preventDefault(); setAuto(!state.auto); toast(state.auto ? 'Auto-advance on' : 'Auto-advance paused');}
    else if (key === 'f') {event.preventDefault(); fullscreen();}
    else if (key === 'n') {event.preventDefault(); toggleDialog('#notes-dialog');}
    else if (key === '?') {event.preventDefault(); toggleDialog('#help-dialog');}
  });
  window.addEventListener('resize', resize);
  window.addEventListener('hashchange', () => {const route = readHash(); goTo(route.index, route.build);});
  document.addEventListener('visibilitychange', () => {deck.classList.toggle('motion-paused', document.hidden); restartAutoTimer();});
  document.addEventListener('fullscreenchange', () => {$('#fullscreen-button').setAttribute('aria-label', document.fullscreenElement ? 'Exit fullscreen' : 'Enter fullscreen'); resize();});
  window.addEventListener('pagehide', releaseVideoURL);
  window.addEventListener('beforeprint', () => {if (window.gsap) gsap.globalTimeline.getChildren().forEach(tween => tween.progress(1));});
  window.Deck = Object.freeze({goTo, next, previous, selectModel, get state() {return {...state};}});
  resize();
  const route = readHash(); state.slide = route.index; state.build = Math.min(route.build, Number(slides[route.index].dataset.builds));
  render(false);
  animation = animate($$('.enter', slides[state.slide]), {opacity: 0, y: 17}, {opacity: 1, y: 0, duration: .85, stagger: .07, ease: 'power3.out'});
})();
