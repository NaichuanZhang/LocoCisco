"""Live viewer on 127.0.0.1 only: the frames the engine sees as MJPEG with every face boxed (the talking one
in its person's colour), captions under the video coloured by person (partials dimmed), and the latency of
the latest final caption. Standard library only (ThreadingHTTPServer in a thread); the engine pushes frames
and captions from the asyncio loop with two cheap calls. Nothing is recorded here.

  GET /             the page
  GET /stream.mjpg  multipart/x-mixed-replace JPEG stream (~10 fps)
  GET /events       text/event-stream: {"type": "caption", ...} | {"type": "stats", ...}
  GET /state        JSON snapshot (tests, curl)
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .captions import Caption

log = logging.getLogger("unspoken.transcribe.live.web")

PALETTE = {"person-left": "#4fc3f7", "person-right": "#ffb74d", "person-s1": "#4fc3f7", "person-s2": "#ffb74d",
           "person-s3": "#aed581", "person-s4": "#f06292"}
EXTRA = ["#ba68c8", "#aed581", "#f06292", "#90a4ae", "#fff176", "#80cbc4"]


def speaker_color(key: str, cache: dict[str, str]) -> str:
    if key in PALETTE:
        return PALETTE[key]
    if key not in cache:
        cache[key] = EXTRA[len(cache) % len(EXTRA)]
    return cache[key]


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


class ViewerState:
    """Thread-safe latest frame + caption history + stats, shared by the engine (writer) and HTTP handlers."""

    def __init__(self, keep_captions: int = 400) -> None:
        self.cond = threading.Condition()
        self.frame: bytes | None = None
        self.frame_seq = 0
        self.events: deque[tuple[int, dict[str, Any]]] = deque(maxlen=keep_captions)
        self.event_seq = 0
        self.captions: dict[str, Caption] = {}
        self.colors: dict[str, str] = {}
        self.latency: list[float] = []
        self.media_t = 0.0
        self.closed = False

    def push_frame(self, jpeg: bytes, media_t: float) -> None:
        with self.cond:
            self.frame, self.frame_seq, self.media_t = jpeg, self.frame_seq + 1, media_t
            self.cond.notify_all()

    def push_caption(self, cap: Caption) -> None:
        d = cap.to_dict()
        d.pop("words", None)
        d["type"] = "caption"
        d["color"] = speaker_color(cap.speaker, self.colors)
        with self.cond:
            self.captions[cap.seg] = cap
            if cap.final and cap.revision == 0 and cap.latency is not None:
                self.latency.append(cap.latency)
            self.event_seq += 1
            self.events.append((self.event_seq, d))
            self.cond.notify_all()

    def push_stats(self, stats: dict[str, Any]) -> None:
        d = dict(stats)
        d["type"] = "stats"
        with self.cond:
            self.event_seq += 1
            self.events.append((self.event_seq, d))
            self.cond.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self.cond:
            lat = sorted(self.latency)
            caps = sorted(self.captions.values(), key=lambda c: c.start)[-12:]
            return {"media_t": round(self.media_t, 2), "frames": self.frame_seq, "events": self.event_seq,
                    "latency_p50": lat[len(lat) // 2] if lat else None,
                    "latency_last": self.latency[-1] if self.latency else None,
                    "captions": [{k: v for k, v in c.to_dict().items() if k != "words"} for c in caps]}

    def close(self) -> None:
        with self.cond:
            self.closed = True
            self.cond.notify_all()


def annotate(image: Any, tracks: list[Any], colors: dict[str, str], speaking_threshold: float = 0.5,
             media_t: float | None = None, latency: float | None = None) -> Any:
    """Draw the face boxes (talking face in its seat colour, others grey) and a clock line; returns a copy."""
    import cv2

    img = image.copy()
    h, w = img.shape[:2]
    for tr in tracks:
        x, y, bw, bh = tr.bbox
        p0, p1 = (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h))
        talking = float(tr.speaking) >= speaking_threshold
        key = f"person-{tr.seat}" if tr.seat else tr.track_id
        color = hex_to_bgr(speaker_color(key, colors)) if talking else (160, 160, 160)
        cv2.rectangle(img, p0, p1, color, 2 if talking else 1)
        label = f"{tr.seat or tr.track_id} {float(tr.speaking):.2f}"
        cv2.putText(img, label, (p0[0], max(12, p0[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    line = []
    if media_t is not None:
        line.append(f"t={media_t:6.1f}s")
    if latency is not None:
        line.append(f"last caption +{latency:.2f}s")
    if line:
        cv2.putText(img, "  ".join(line), (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>LocoCisco live captions</title>
<style>
body{background:#111;color:#eee;font-family:system-ui,sans-serif;margin:0;padding:12px}
#wrap{max-width:960px;margin:auto}
img{width:100%;background:#000;display:block;border-radius:6px}
#caps{min-height:160px;margin-top:8px}
.cap{padding:4px 8px;margin:3px 0;border-left:4px solid #666;background:#1b1b1b;border-radius:4px;font-size:18px}
.cap .who{font-weight:600;margin-right:8px}
.cap.partial{opacity:.55;font-style:italic}
.cap .meta{float:right;font-size:12px;color:#999}
#stats{font-size:13px;color:#aaa;margin-top:6px}
</style></head><body><div id="wrap">
<img id="v" src="/stream.mjpg" alt="live video with face boxes">
<div id="stats">waiting for captions…</div>
<div id="caps"></div>
</div><script>
const caps = {}; const order = []; let lat = [];
function render(){
  const box = document.getElementById('caps'); box.innerHTML = '';
  const recent = order.slice(-6);
  for (const id of recent){ const c = caps[id]; if(!c) continue;
    const d = document.createElement('div'); d.className = 'cap' + (c.final ? '' : ' partial');
    d.style.borderColor = c.color;
    const who = document.createElement('span'); who.className='who'; who.style.color = c.color;
    who.textContent = c.speaker.replace('person-','').replace('voice-','voice ');
    const meta = document.createElement('span'); meta.className='meta';
    meta.textContent = (c.final ? ('final' + (c.revision ? ' r'+c.revision : '') + (c.latency!=null ? ' +'+c.latency.toFixed(2)+'s' : '')) : 'partial…') + '  ' + c.start.toFixed(1)+'–'+c.end.toFixed(1)+'s';
    d.appendChild(who); d.appendChild(document.createTextNode(c.text)); d.appendChild(meta); box.appendChild(d);
  }
  const s = document.getElementById('stats');
  if (lat.length){ const ys=[...lat].sort((a,b)=>a-b); const p50=ys[Math.floor(ys.length/2)]; const p95=ys[Math.min(ys.length-1,Math.floor(ys.length*0.95))];
    s.textContent = `final captions: ${lat.length}   latency p50 ${p50.toFixed(2)} s   p95 ${p95.toFixed(2)} s   last ${lat[lat.length-1].toFixed(2)} s`; }
}
const es = new EventSource('/events');
es.onmessage = (e) => { const m = JSON.parse(e.data);
  if (m.type === 'caption'){ if(!(m.seg in caps)) order.push(m.seg); caps[m.seg] = m;
    if (m.final && m.revision === 0 && m.latency != null) lat.push(m.latency); render(); }
  else if (m.type === 'stats'){ document.getElementById('stats').textContent += '   ' + (m.text || ''); } };
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    state: ViewerState  # set on the server class

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        log.debug(fmt, *args)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/state":
            body = json.dumps(self.server.state.snapshot()).encode("utf-8")  # type: ignore[attr-defined]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/stream.mjpg":
            self._stream()
        elif path == "/events":
            self._events()
        else:
            self.send_error(404)

    def _stream(self) -> None:
        st: ViewerState = self.server.state  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        seen = -1
        try:
            while not st.closed:
                with st.cond:
                    if st.frame_seq == seen:
                        st.cond.wait(1.0)
                    if st.frame is None or st.frame_seq == seen:
                        continue
                    frame, seen = st.frame, st.frame_seq
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode()
                                 + b"\r\n\r\n" + frame + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _events(self) -> None:
        st: ViewerState = self.server.state  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        seen = 0
        try:
            while not st.closed:
                with st.cond:
                    pending = [(n, d) for n, d in st.events if n > seen]
                    if not pending:
                        st.cond.wait(1.0)
                        continue
                for n, d in pending:
                    self.wfile.write(b"data: " + json.dumps(d, ensure_ascii=False).encode("utf-8") + b"\n\n")
                    seen = n
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class LiveViewer:
    """Owns the server thread and the state; `on_frame` / `on_caption` are what the engine calls."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8790, fps: float = 10.0, speaking_threshold: float = 0.5) -> None:
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("the live viewer binds to loopback only")
        self.state = ViewerState()
        self.fps = fps
        self.speaking_threshold = speaking_threshold
        self._last_push = -1.0
        self.server = ThreadingHTTPServer((host if host != "localhost" else "127.0.0.1", port), _Handler)
        self.server.daemon_threads = True
        self.server.state = self.state  # type: ignore[attr-defined]
        self.host, self.port = self.server.server_address[0], self.server.server_address[1]
        self._thread = threading.Thread(target=self.server.serve_forever, name="live-viewer", daemon=True)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> "LiveViewer":
        self._thread.start()
        return self

    def on_frame(self, frame: Any, tracks: list[Any], media_t: float) -> None:
        now = time.monotonic()
        if now - self._last_push < 1.0 / self.fps:
            return
        self._last_push = now
        import cv2

        lat = self.state.latency[-1] if self.state.latency else None
        img = annotate(frame.image, tracks, self.state.colors, self.speaking_threshold, media_t, lat)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            self.state.push_frame(buf.tobytes(), media_t)

    def on_caption(self, cap: Caption) -> None:
        self.state.push_caption(cap)

    def stop(self) -> None:
        self.state.close()
        self.server.shutdown()
        self.server.server_close()


__all__ = ["LiveViewer", "ViewerState", "annotate", "speaker_color", "hex_to_bgr", "PALETTE"]
