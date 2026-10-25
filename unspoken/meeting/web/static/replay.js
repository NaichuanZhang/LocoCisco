(async function () {
  'use strict';
  let initial;
  try {
    const response = await fetch('/api/replay', { cache: 'no-store' });
    if (!response.ok) return;
    initial = await response.json();
  } catch (_) { return; }
  const template = await fetch('/static/replay.html');
  if (!template.ok) return;
  const holder = document.createElement('div');
  holder.innerHTML = await template.text();
  document.querySelector('main').prepend(holder.firstElementChild);
  document.body.classList.add('replay-demo');
  document.documentElement.dataset.theme = 'light';
  document.querySelector('.flow li:first-child span').textContent =
    'Audio and frames from a prerecorded meeting run through local speech, speaker and vision models during playback.';
  document.querySelector('.flow li:last-child span').textContent =
    'Room-device and Discord actions are local rehearsal previews in this demo. No external action is sent.';
  document.querySelector('.flow li:nth-child(2) span').textContent =
    'Answers bounded questions with probabilities for each option. The X-ray shows measured latency and the resulting policy decision.';
  document.querySelector('.flow li:nth-child(3) span').textContent =
    'Qwen runs on this box when a decision needs more context. Meeting questions use the on-record transcript.';
  document.querySelector('.how-foot').textContent =
    'The footer counts application notifications, not all machine traffic. Off-record hides and mutes the replay; it does not erase source videos or recordings already saved.';
  const $ = id => document.getElementById(id);
  $('capture-transcript').append(document.querySelector('.captions-wrap'));
  document.querySelector('.top').append($('conn'));
  document.querySelector('.cap-label').textContent = 'Live captions · unverified speakers stay Guest';
  const video = $('replay-video');
  let state = initial, generation = null, mediaURL = '', receivedAt = performance.now();
  let busy = false, polling = false, capturePhase = 'idle', captureEpoch = 0;
  let displayStream = null, recorder = null, chunks = [], recordingBytes = 0;
  let audioContext = null, audioSource = null, audioGain = null, audioDestination = null;
  let recordingURL = null, finishTimer = null, privateMode = false;
  let externalCapture = false;
  let autoplayBlocked = false, autoplayMuted = false, browserPlay = null;
  // Automation sets autoplay=0 so capture audio is ready before its explicit start.
  let autoStartPending = new URLSearchParams(window.location?.search || '').get('autoplay') !== '0';
  let privacyValue = null, privacyVersion = 0, knownPrivacyEpoch = -1;
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  const formatTime = value => {
    const s = Math.max(0, Math.floor(value || 0));
    return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  };
  function error(message) {
    $('replay-error').textContent = message || '';
    $('replay-error').hidden = !message;
  }
  async function request(path, body) {
    const issued = privacyVersion;
    const response = await fetch(path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status})`);
    if (issued !== privacyVersion) {
      if (path === '/api/replay/control' || path === '/api/replay/refresh') {
        const current = await readCurrent();
        if (current) return current;
      } else if (path === '/api/chat' && ['offrecord', 'resume'].includes(body?.command)) {
        return data;
      }
      throw new Error('The meeting or privacy state changed; this response was discarded.');
    }
    return data;
  }
  function privacy(value) {
    if (value !== privacyValue) privacyVersion += 1;
    privacyValue = value;
    state.privacy = value;
    privateMode = value !== 'on';
    $('replay-privacy').hidden = !privateMode;
    video.muted = privateMode || autoplayMuted;
    if (audioGain) audioGain.gain.value = privateMode ? 0 : 1;
    if (privateMode) {
      $('replay-answer').textContent = '';
      $('replay-answer').hidden = true;
    }
    window.LocoPresenter?.privacy(value);
    renderPlaybackState();
  }
  document.addEventListener('lococisco:privacy', event => privacy(event.detail));
  document.addEventListener('lococisco:reset', () => { privacyVersion += 1; });
  document.addEventListener('lococisco:capture', event => {
    externalCapture = event.detail === true;
    if (externalCapture) autoStartPending = false;
    render(state);
  });

  function render(next) {
    if (generation !== null && (next.generation < generation ||
      (next.generation === generation && (next.privacy_epoch ?? -1) < knownPrivacyEpoch))) return;
    state = next;
    receivedAt = performance.now();
    if (generation !== null && next.generation !== generation) {
      document.dispatchEvent(new Event('lococisco:reset'));
      $('replay-answer').textContent = '';
      $('replay-answer').hidden = true;
      video.currentTime = 0;
    }
    generation = next.generation;
    knownPrivacyEpoch = next.privacy_epoch ?? knownPrivacyEpoch;
    const currentURL = next.media?.url || '';
    if (currentURL !== mediaURL) {
      video.pause();
      mediaURL = currentURL;
      if (mediaURL) video.src = mediaURL;
      else video.removeAttribute('src');
      video.load();
    }
    $('replay-input-label').textContent = next.media
      ? `${next.input_mode} · ${formatTime(next.media.duration)}` : 'Choose a meeting video with audio';
    $('replay-status').textContent = next.ready
      ? `Local models ready · ${next.state}` : `Not ready: ${next.unavailable_roles.join(', ')}`;
    $('replay-integration').textContent = next.integrations;
    $('replay-attribution').textContent = next.attribution || '';
    privacy(next.privacy);
    window.LocoPresenter?.replay(next);
    const capturing = capturePhase !== 'idle' || externalCapture;
    $('replay-play').disabled = busy || capturing || !next.media || !next.ready ||
      (next.state === 'playing' && !autoplayBlocked);
    $('replay-pause').disabled = busy || !['playing', 'paused'].includes(next.state);
    $('replay-pause').textContent = next.state === 'paused' ? 'Resume' : 'Pause';
    $('replay-restart').disabled = busy || capturing || !next.media;
    $('replay-file').disabled = busy || capturing;
    $('replay-record').disabled = busy || capturing || !next.media || !next.ready;
    $('replay-stop-record').disabled = !capturing || capturePhase === 'stopping';
    $('replay-recording').hidden = capturePhase !== 'recording' && !externalCapture;
    $('replay-refresh').disabled = busy || capturing || ['playing', 'finishing'].includes(next.state);
    if (next.error) error(next.error);
    if (next.state === 'ended' && capturePhase === 'recording' && !finishTimer) {
      finishTimer = setTimeout(() => { finishTimer = null; stopRecording(); }, 1500);
    }
  }

  function renderPlaybackState() {
    $('capture-playback-state').textContent = state.state === 'playing' && autoplayMuted && !privateMode
      ? 'Playing muted · click or Space for sound'
      : ({ playing: 'Processing the meeting', paused: 'Take paused', ended: 'Take complete',
          finishing: 'Finishing inference', ready: 'Ready for a take',
          error: 'Pipeline interrupted' })[state.state] || 'Preparing the pipeline';
  }

  function playVideo() {
    if (browserPlay) return browserPlay;
    browserPlay = (async () => {
      try {
        try { await video.play(); }
        catch (e) {
          if (e.name !== 'NotAllowedError' || video.muted) throw e;
          // Audibly autoplay when the browser permits it; otherwise keep the actual
          // video moving silently until a real interaction can unlock sound.
          autoplayMuted = true;
          privacy(state.privacy);
          await video.play();
        }
        autoplayBlocked = false;
        renderPlaybackState();
      } catch (e) {
        autoplayBlocked = true;
        throw e;
      } finally { browserPlay = null; }
    })();
    return browserPlay;
  }

  function enableSound() {
    if (privateMode || (!autoplayMuted && !autoplayBlocked)) return false;
    autoplayMuted = false;
    autoplayBlocked = false;
    privacy(state.privacy);
    if (audioContext) audioContext.resume().catch(e => error(e.message));
    if (state.state === 'playing') playVideo().catch(e => error(e.message));
    return true;
  }

  async function maybeAutoplay() {
    if (!autoStartPending || busy || capturePhase !== 'idle' || externalCapture) return;
    // A visit can join a running take, but cannot resume a paused/private take.
    if (state.privacy !== 'on' || ['paused', 'playing', 'finishing', 'error'].includes(state.state)) {
      autoStartPending = false;
      return;
    }
    if (!state.ready || !state.media || video.readyState < 2 || !['ready', 'ended'].includes(state.state)) return;
    autoStartPending = false;
    // The server's locked play action resets an ended take atomically. Avoid a
    // separate reset which could interrupt another browser starting at the same time.
    await control('play');
  }

  async function control(action) {
    if (busy) return;
    autoStartPending = false;
    busy = true;
    if (action !== 'play') video.pause();
    error('');
    try {
      const next = await request('/api/replay/control', { action });
      render(next);
      if (action === 'play' && next.state === 'playing') {
        if (Number.isFinite(video.duration)) video.currentTime = next.position;
        autoplayBlocked = false;
        await playVideo();
      }
    } catch (e) {
      error(e.message);
      if (action === 'play') {
        try { render(await request('/api/replay/control', { action: 'pause' })); } catch (_) {}
      }
      throw e;
    } finally {
      busy = false;
      render(state);
    }
  }

  async function readCurrent() {
    const issued = privacyVersion;
    const response = await fetch('/api/replay', { cache: 'no-store' });
    if (!response.ok) throw new Error('Replay service unavailable.');
    const data = await response.json();
    return issued === privacyVersion ? data : null;
  }
  async function poll() {
    if (busy || polling) return;
    polling = true;
    try {
      const current = await readCurrent();
      if (current) { render(current); await maybeAutoplay(); }
    } catch (e) {
      video.pause();
      error(`${e.message} Playback paused; reconnect before continuing.`);
    } finally { polling = false; }
  }
  function syncVideo() {
    const position = Math.min(state.media?.duration || 0, state.position +
      (state.state === 'playing' ? (performance.now() - receivedAt) / 1000 : 0));
    $('replay-time').textContent = `${formatTime(position)} / ${formatTime(state.media?.duration)}`;
    const progress = $('capture-progress-fill');
    const percentage = state.media?.duration ? 100 * position / state.media.duration : 0;
    if (progress.style) progress.style.width = `${percentage}%`;
    progress.parentElement?.setAttribute('aria-valuenow', String(Math.round(percentage)));
    // Face boxes use source-image coordinates, including when the video is letterboxed.
    if (video.videoWidth && video.videoHeight && video.clientWidth && video.clientHeight) {
      const scale = Math.min(video.clientWidth / video.videoWidth, video.clientHeight / video.videoHeight);
      const width = video.videoWidth * scale, height = video.videoHeight * scale;
      Object.assign($('capture-vision').style, { width: `${width}px`, height: `${height}px`,
        left: `${(video.clientWidth - width) / 2}px`, top: `${(video.clientHeight - height) / 2}px` });
    }
    if (!busy && video.readyState >= 2 && mediaURL) {
      if (state.state === 'playing') {
        if (Math.abs(video.currentTime - position) > 0.4) video.currentTime = position;
        if (video.paused && !video.ended && !autoplayBlocked) playVideo().catch(e => {
          error(e.name === 'NotAllowedError' ? 'Click or press Space to enable this browser’s playback.' : e.message);
          render(state);
        });
      } else if (!video.paused) video.pause();
    }
    requestAnimationFrame(syncVideo);
  }
  video.addEventListener('error', () => {
    error('This browser cannot play the selected video. Use an MP4 with H.264 video and AAC audio.');
  });
  $('replay-play').onclick = () => control('play').catch(() => {});
  $('replay-pause').onclick = () => control(state.state === 'paused' ? 'play' : 'pause').catch(() => {});
  $('replay-restart').onclick = () => control('restart').catch(() => {});
  $('replay-refresh').onclick = async () => {
    busy = true;
    error('');
    render(state);
    $('replay-status').textContent = 'Checking local models…';
    try {
      render(await request('/api/replay/refresh'));
      document.dispatchEvent(new Event('lococisco:models'));
    } catch (e) { error(e.message); }
    finally { busy = false; render(state); }
  };
  $('replay-xray').onclick = () => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'x' }));
  $('replay-file').onchange = async () => {
    const file = $('replay-file').files[0];
    if (!file) return;
    if (file.size > 300 * 1024 * 1024) { error('Choose a video smaller than 300 MiB.'); return; }
    busy = true;
    video.pause();
    error('');
    render(state);
    $('replay-status').textContent = 'Loading video locally…';
    try {
      const form = new FormData();
      form.append('file', file);
      const response = await fetch('/api/replay/upload', { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Video could not be loaded.');
      render(data);
    } catch (e) { error(e.message); }
    finally { busy = false; $('replay-file').value = ''; render(state); }
  };

  async function chat(command, args = {}) {
    try {
      error('');
      await request('/api/chat', { command, user: 'Demo operator', args });
      await poll();
    } catch (e) { error(e.message); }
  }
  $('replay-offrecord').onclick = () => { privacy('off_record'); chat('offrecord'); };
  $('replay-resume').onclick = () => chat('resume');
  $('replay-checkin').onclick = () => {
    const name = $('replay-person').value.trim();
    if (name) chat('here', { name });
  };
  $('replay-ask').onclick = async () => {
    const question = $('replay-question').value.trim() || 'What did we decide?';
    $('replay-ask').disabled = true;
    $('replay-answer').hidden = false;
    $('replay-answer').textContent = 'Asking the local model…';
    try {
      const data = await request('/api/replay/question', { question });
      if (!privateMode) $('replay-answer').textContent = data.text;
    } catch (e) { error(e.message); $('replay-answer').textContent = ''; }
    finally { $('replay-ask').disabled = false; }
  };

  function releaseDisplay() {
    if (displayStream) {
      const old = displayStream;
      displayStream = null;
      for (const track of old.getTracks()) track.stop();
    }
  }
  function stopRecording() {
    if (externalCapture) {
      document.dispatchEvent(new Event('lococisco:stop-capture'));
      return;
    }
    if (capturePhase === 'idle' || capturePhase === 'stopping') return;
    captureEpoch += 1;
    $('replay-countdown').hidden = true;
    if (finishTimer) { clearTimeout(finishTimer); finishTimer = null; }
    capturePhase = 'stopping';
    if (recorder && recorder.state !== 'inactive') recorder.stop();
    else {
      releaseDisplay();
      capturePhase = 'idle';
    }
    render(state);
  }
  async function record() {
    if (capturePhase !== 'idle' || !state.ready || !state.media) return;
    autoStartPending = false;
    enableSound();
    $('replay-operator').hidden = true;
    if (!navigator.mediaDevices?.getDisplayMedia || !window.MediaRecorder || !window.AudioContext) {
      error('Browser recording is unavailable. Use the local Firefox recording launcher described in docs/BROWSER_DEMO.md; it records the actual UI and meeting audio without capture permission.');
      return;
    }
    const mime = ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm']
      .find(type => MediaRecorder.isTypeSupported(type));
    if (!mime) { error('WebM recording is unsupported. Use the OS screen recorder with system audio enabled.'); return; }
    const epoch = ++captureEpoch;
    capturePhase = 'preparing';
    error('');
    render(state);
    try {
      if (!audioContext) {
        audioContext = new AudioContext();
        audioSource = audioContext.createMediaElementSource(video);
        audioGain = audioContext.createGain();
        audioDestination = audioContext.createMediaStreamDestination();
        audioSource.connect(audioGain);
        audioGain.connect(audioContext.destination);
        audioGain.connect(audioDestination);
      }
      await audioContext.resume();
      privacy(state.privacy);
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: 30, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false,
      });
      if (epoch !== captureEpoch) { stream.getTracks().forEach(track => track.stop()); return; }
      displayStream = stream;
      const track = stream.getVideoTracks()[0];
      if (!track) throw new Error('No screen video was selected.');
      track.addEventListener('ended', stopRecording, { once: true });
      if (track.getSettings().displaySurface === 'monitor') {
        throw new Error('Select the demo tab or browser window rather than the entire monitor.');
      }
      const combined = new MediaStream([track, ...audioDestination.stream.getAudioTracks()]);
      if (!combined.getAudioTracks().length) throw new Error('Audio capture is unavailable; use the OS recorder with system audio.');
      recorder = new MediaRecorder(combined, { mimeType: mime, videoBitsPerSecond: 4_000_000 });
      chunks = [];
      recordingBytes = 0;
      recorder.ondataavailable = event => {
        if (event.data.size) { chunks.push(event.data); recordingBytes += event.data.size; }
        if (recordingBytes > 256 * 1024 * 1024) {
          error('Recording reached the 256 MiB limit and was stopped. Save this take before continuing.');
          stopRecording();
        }
      };
      recorder.onerror = () => { error('Recording failed. Try another take or the local recording launcher.'); stopRecording(); };
      recorder.onstop = () => {
        releaseDisplay();
        if (recordingURL) URL.revokeObjectURL(recordingURL);
        if (chunks.length) {
          recordingURL = URL.createObjectURL(new Blob(chunks, { type: mime }));
          $('replay-download').href = recordingURL;
          $('replay-download').download = `LocoCisco-demo-${new Date().toISOString().replace(/[:.]/g, '-')}.webm`;
          $('replay-download').hidden = false;
          $('replay-download').textContent = 'Download WebM · review video & sound';
          $('replay-operator').hidden = false;
        }
        chunks = [];
        recorder = null;
        capturePhase = 'idle';
        render(state);
      };
      await control('restart');
      for (let count = 3; count > 0; count -= 1) {
        if (epoch !== captureEpoch) return;
        $('replay-countdown').textContent = count;
        $('replay-countdown').hidden = false;
        await sleep(1000);
      }
      if (epoch !== captureEpoch) return;
      $('replay-countdown').hidden = true;
      recorder.start(1000);
      capturePhase = 'recording';
      render(state);
      await control('play');
    } catch (e) {
      error(e.name === 'NotAllowedError' ? 'Capture was cancelled or denied. Try again, or use the local recording launcher in docs/BROWSER_DEMO.md.' : e.message);
      stopRecording();
    } finally {
      if (capturePhase === 'preparing' || (capturePhase === 'stopping' && !recorder)) {
        releaseDisplay();
        capturePhase = 'idle';
      }
      $('replay-countdown').hidden = true;
      render(state);
    }
  }
  $('replay-record').onclick = record;
  $('replay-stop-record').onclick = stopRecording;
  $('replay-operator-close').onclick = () => { $('replay-operator').hidden = true; };
  window.addEventListener('beforeunload', () => { releaseDisplay(); if (recordingURL) URL.revokeObjectURL(recordingURL); });
  document.addEventListener('pointerdown', event => { if (event.isTrusted) enableSound(); });
  document.addEventListener('keydown', event => {
    if (event.target?.matches?.('input, textarea, select, [contenteditable=true]')) return;
    if (event.metaKey || event.ctrlKey || event.altKey || event.repeat) return;
    const soundEnabled = event.isTrusted && enableSound();
    if (event.key === 'Escape') $('replay-operator').hidden = true;
    if (event.key.toLowerCase() === 'o') {
      event.preventDefault();
      if (capturePhase !== 'idle' || externalCapture) return;
      $('replay-operator').hidden = !$('replay-operator').hidden;
    }
    if (event.key.toLowerCase() === 'r') {
      event.preventDefault();
      if (event.shiftKey && capturePhase === 'idle' && !externalCapture) control('restart').catch(() => {});
      else if (!event.shiftKey) {
        if (capturePhase !== 'idle' || externalCapture) stopRecording();
        else record();
      }
    }
    if (event.code === 'Space') {
      event.preventDefault();
      $('replay-operator').hidden = true;
      if (soundEnabled && state.state === 'playing') error('');
      else if (autoplayBlocked && state.state === 'playing') enableSound();
      else control(state.state === 'playing' ? 'pause' : 'play').catch(() => {});
    }
  });
  privacy(document.documentElement.dataset.notesState || 'off_record');
  await poll();
  document.dispatchEvent(new Event('lococisco:reset'));
  setInterval(poll, 250);
  requestAnimationFrame(syncVideo);
})();
