"""Offline recording fallback: render the real browser UI and capture its source audio.

Run before a take; the recorder opens an isolated Firefox profile, starts a fresh
replay through the browser controls, and saves a local H.264/AAC MP4.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import shutil
import socket
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

import av
import httpx
import websockets

from .web.server import require_loopback


class Browser:
    def __init__(self, websocket) -> None:
        self.websocket = websocket
        self.serial = 0
        self.context = ""

    async def call(self, method: str, **params):
        self.serial += 1
        await self.websocket.send(json.dumps({"id": self.serial, "method": method, "params": params}))
        while True:
            result = json.loads(await asyncio.wait_for(self.websocket.recv(), 30))
            if result.get("id") == self.serial:
                if result.get("type") == "error":
                    raise RuntimeError(f"Browser {method}: {result.get('message', result.get('error'))}")
                return result["result"]

    async def evaluate(self, expression: str, *, activate: bool = False):
        result = await self.call("script.evaluate", expression=expression,
                                 target={"context": self.context}, awaitPromise=True, userActivation=activate)
        if result["type"] != "success":
            raise RuntimeError("Browser evaluation failed: " + str(result.get("exceptionDetails", {}).get("text", "")))
        return result["result"].get("value")

    async def click(self, element_id: str) -> None:
        """Activate the existing handler without exposing backstage controls in the take."""
        await self.evaluate(
            f"(async()=>{{const button=document.getElementById({json.dumps(element_id)});"
            "if(!button||button.disabled||typeof button.onclick!=='function')"
            "throw new Error('Recording control is unavailable.');"
            "await button.onclick(new MouseEvent('click'));})()", activate=True)


AUDIO_START = """(async () => {
  const video = document.getElementById('replay-video');
  const context = new AudioContext();
  await context.resume();
  if (context.state !== 'running') throw new Error('Browser audio did not start.');
  const source = context.createMediaElementSource(video);
  const gain = context.createGain();
  gain.gain.value = document.documentElement.dataset.notesState === 'on' ? 1 : 0;
  const destination = context.createMediaStreamDestination();
  source.connect(gain); gain.connect(destination); gain.connect(context.destination);
  const protect = event => { gain.gain.value = event.detail === 'on' ? 1 : 0; };
  document.addEventListener('lococisco:privacy', protect);
  const type = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus']
    .find(value => MediaRecorder.isTypeSupported(value));
  if (!type) throw new Error('No local audio encoder is available.');
  const recorder = new MediaRecorder(destination.stream, {mimeType: type});
  const chunks = [];
  recorder.ondataavailable = event => { if(event.data.size) chunks.push(event.data); };
  const stopped = new Promise((resolve, reject) => {
    recorder.onerror = () => reject(new Error('Audio recorder failed.'));
    recorder.onstop = () => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(',')[1]);
      reader.onerror = reject;
      reader.readAsDataURL(new Blob(chunks, {type}));
    };
  });
  recorder.start(1000);
  window.__locoCapture = {recorder, stopped, context, stopRequested:false};
  document.addEventListener('lococisco:stop-capture', () => {
    window.__locoCapture.stopRequested = true;
  });
  return performance.now();
})()"""


def mux(video_path: Path, audio_path: Path, output: Path) -> None:
    with av.open(str(video_path)) as video, av.open(str(audio_path)) as audio, av.open(
        str(output), "w", options={"movflags": "+faststart"}
    ) as target:
        video_out = target.add_stream_from_template(video.streams.video[0])
        audio_out = target.add_stream("aac", rate=48000)
        audio_out.layout, audio_out.bit_rate = "stereo", 128000
        for packet in video.demux(video=0):
            if packet.dts is not None:
                packet.stream = video_out
                target.mux(packet)
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=48000)
        samples = 0
        for decoded in audio.decode(audio=0):
            for frame in resampler.resample(decoded):
                frame.pts, frame.time_base = samples, Fraction(1, 48000)
                samples += frame.samples
                for packet in audio_out.encode(frame):
                    target.mux(packet)
        for frame in resampler.resample(None):
            frame.pts, frame.time_base = samples, Fraction(1, 48000)
            samples += frame.samples
            for packet in audio_out.encode(frame):
                target.mux(packet)
        for packet in audio_out.encode():
            target.mux(packet)


def screenshot_frame(png: bytes, width: int, height: int) -> av.VideoFrame:
    """Convert full-range browser RGB to explicitly limited-range HD video."""
    with av.open(io.BytesIO(png)) as image:
        frame = next(image.decode(video=0))
    # An implicit conversion inherits PNG's full range. Untagged H.264 players
    # then expand it again, clipping the light canvas and dark text.
    return frame.reformat(width=width, height=height, format="yuv420p",
                          src_color_range="JPEG", dst_color_range="MPEG",
                          dst_colorspace="ITU709")


def recording_page_url(url: str) -> str:
    """Keep the dashboard idle until the recorder has connected its audio graph."""
    endpoint = urlsplit(url)
    query = [(key, value) for key, value in parse_qsl(endpoint.query, keep_blank_values=True)
             if key != "autoplay"]
    return endpoint._replace(query=urlencode([*query, ("autoplay", "0")]), fragment="").geturl()


async def record(url: str, output: Path, *, seconds: float | None, fps: int,
                 width: int, height: int, privacy_demo: bool = False) -> dict:
    endpoint = urlsplit(url)
    require_loopback(endpoint.hostname or "")
    if endpoint.scheme != "http" or endpoint.username or endpoint.password:
        raise ValueError("Use the local HTTP demo URL without credentials.")
    if output.exists():
        raise FileExistsError("The recording destination already exists.")
    firefox = shutil.which("firefox")
    if not firefox:
        raise RuntimeError("Firefox must already be installed. No browser will be downloaded.")
    output.parent.mkdir(parents=True, exist_ok=True)
    url = url.rstrip("/")
    with tempfile.TemporaryDirectory(prefix="lococisco-record-", dir=output.parent) as temp:
        root = Path(temp)
        profile = root / "profile"
        profile.mkdir()
        profile.joinpath("user.js").write_text(
            'user_pref("app.update.auto", false);\n'
            'user_pref("extensions.update.enabled", false);\n'
            'user_pref("datareporting.healthreport.uploadEnabled", false);\n'
            'user_pref("toolkit.telemetry.enabled", false);\n'
            'user_pref("browser.newtabpage.enabled", false);\n'
            'user_pref("network.proxy.type", 1);\n'
            'user_pref("network.proxy.http", "127.0.0.1");\n'
            'user_pref("network.proxy.http_port", 9);\n'
            'user_pref("network.proxy.ssl", "127.0.0.1");\n'
            'user_pref("network.proxy.ssl_port", 9);\n'
            'user_pref("network.proxy.no_proxies_on", "localhost,127.0.0.1,::1");\n'
            'user_pref("network.trr.mode", 5);\n'
            'user_pref("network.dns.disablePrefetch", true);\n'
            'user_pref("network.prefetch-next", false);\n'
            'user_pref("network.http.http3.enable", false);\n'
        )
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        with root.joinpath("firefox.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                firefox, "--headless", "--no-remote", "--profile", str(profile),
                "--remote-debugging-port", str(port), "about:blank",
                stdout=log, stderr=asyncio.subprocess.STDOUT)
            connection = None
            replay_generation = None
            try:
                for _ in range(150):
                    try:
                        connection = await websockets.connect(
                            f"ws://127.0.0.1:{port}/session", max_size=32 * 1024 * 1024)
                        break
                    except OSError:
                        if process.returncode is not None:
                            raise RuntimeError("The isolated Firefox recorder could not start.")
                        await asyncio.sleep(0.1)
                if connection is None:
                    raise RuntimeError("Timed out starting the local browser recorder.")
                async with httpx.AsyncClient(base_url=url, timeout=30) as client:
                    state = (await client.get("/api/replay")).raise_for_status().json()
                    if not state["ready"] or not state["media"]:
                        raise RuntimeError("Load a video and make all local models ready before recording.")
                    duration = min(float(seconds or state["media"]["duration"] + 5), state["media"]["duration"] + 5)
                    replay_generation = (await client.post(
                        "/api/replay/control", json={"action": "restart"})).raise_for_status().json()["generation"]
                    browser = Browser(connection)
                    await browser.call("session.new", capabilities={"alwaysMatch": {}})
                    browser.context = (await browser.call("browsingContext.create", type="tab"))["context"]
                    await browser.call("browsingContext.setViewport", context=browser.context,
                                       viewport={"width": width, "height": height}, devicePixelRatio=1)
                    await browser.call("browsingContext.navigate", context=browser.context,
                                       url=recording_page_url(url), wait="complete")
                    ready = await browser.evaluate("""(async()=>{
                      for(let n=0;n<100;n++){
                        const b=document.getElementById('replay-play'),v=document.getElementById('replay-video');
                        if(b&&!b.disabled&&v.readyState>=2)return true;
                        await new Promise(r=>setTimeout(r,100));
                      }return false;
                    })()""")
                    if not ready:
                        raise RuntimeError("The browser video did not become ready.")
                    audio_start = await browser.evaluate(AUDIO_START, activate=True)
                    await browser.click("replay-play")
                    await browser.evaluate("document.dispatchEvent(new CustomEvent('lococisco:capture',{detail:true}))")
                    video_path, audio_path = root / "screen.mp4", root / "source-audio.webm"
                    frames, last_pts, privacy_off, privacy_on = 0, -1, False, False
                    started = time.monotonic()
                    with av.open(str(video_path), "w") as video:
                        stream = video.add_stream("libx264", rate=fps)
                        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
                        stream.codec_context.color_range = 1  # MPEG / limited
                        stream.codec_context.colorspace = 1  # BT.709 matrix
                        stream.options = {"preset": "veryfast", "crf": "20"}
                        stream.codec_context.thread_count = 2
                        while time.monotonic() - started < duration:
                            before = await browser.evaluate("performance.now()")
                            shot = await browser.call("browsingContext.captureScreenshot", context=browser.context)
                            after = await browser.evaluate("performance.now()")
                            pts = max(last_pts + 1, round(((before + after) / 2 - audio_start) / 1000 * fps))
                            frame = screenshot_frame(base64.b64decode(shot["data"]), width, height)
                            frame.pts, frame.time_base = pts, Fraction(1, fps)
                            for packet in stream.encode(frame):
                                video.mux(packet)
                            frames, last_pts = frames + 1, pts
                            elapsed = time.monotonic() - started
                            if privacy_demo and elapsed >= duration / 2 and not privacy_off:
                                await browser.click("replay-offrecord")
                                privacy_off = True
                            if privacy_off and elapsed >= duration / 2 + 3 and not privacy_on:
                                await browser.click("replay-resume")
                                privacy_on = True
                            if frames % fps == 0:
                                state = (await client.get("/api/replay")).raise_for_status().json()
                                if state["generation"] != replay_generation:
                                    raise RuntimeError("The meeting was restarted during capture; this mixed take was discarded.")
                                if state["state"] == "error":
                                    raise RuntimeError(state["error"])
                                if await browser.evaluate("window.__locoCapture.stopRequested"):
                                    break
                                if elapsed > 2 and state["state"] == "paused":
                                    raise RuntimeError("Replay was paused before the take finished; no completed recording was saved.")
                            await asyncio.sleep(max(0, 1 / fps - (after - before) / 1000))
                        for packet in stream.encode():
                            video.mux(packet)
                    capture_seconds = ((await browser.evaluate("performance.now()")) - audio_start) / 1000
                    audio = await browser.evaluate("""(async()=>{
                      const c=window.__locoCapture;c.recorder.stop();
                      const bytes=await c.stopped;
                      await c.context.close();
                      document.dispatchEvent(new CustomEvent('lococisco:capture',{detail:false}));
                      return bytes;
                    })()""")
                    audio_path.write_bytes(base64.b64decode(audio))
                    state = (await client.post("/api/replay/control", json={"action": "pause"})).raise_for_status().json()
                    snapshot = (await client.get("/api/snapshot")).raise_for_status().json()["snapshot"]
                    complete = root / "complete.mp4"
                    await asyncio.to_thread(mux, video_path, audio_path, complete)
                    if output.exists():
                        raise FileExistsError("The recording destination appeared during the take.")
                    complete.replace(output)
                    result = {
                        "file": output.name, "format": "H.264/AAC MP4", "viewport": [width, height],
                        "target_fps": fps, "captured_frames": frames, "capture_seconds": round(capture_seconds, 2),
                        "actual_fps": round(frames / capture_seconds, 2),
                        "privacy_demo": privacy_demo, "attribution": state.get("attribution"),
                        "media_seconds": state["position"],
                        "models": [{k: row.get(k) for k in ("role", "label", "ok")} for row in snapshot["models"]],
                        "sensors": snapshot.get("sensors"),
                    }
                    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
                    return result
            finally:
                if replay_generation is not None:
                    try:
                        async with httpx.AsyncClient(base_url=url, timeout=3) as cleanup:
                            current = (await cleanup.get("/api/replay")).raise_for_status().json()
                            if current["generation"] == replay_generation:
                                await cleanup.post("/api/replay/control", json={"action": "pause"})
                    except httpx.HTTPError:
                        pass
                if connection:
                    await connection.close()
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), 10)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8780")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--privacy-demo", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.fps <= 30 or args.width % 2 or args.height % 2:
        parser.error("Use 1–30 FPS and even viewport dimensions.")
    if args.seconds is not None and not 0 < args.seconds <= 600:
        parser.error("Recording duration must be in (0, 600] seconds.")
    print(json.dumps(asyncio.run(record(args.url, args.output, seconds=args.seconds, fps=args.fps,
                                       width=args.width, height=args.height, privacy_demo=args.privacy_demo)), indent=2))


if __name__ == "__main__":
    main()
