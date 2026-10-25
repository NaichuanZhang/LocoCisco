"""Render a demo mp4 from a live run's logs: the source video with the captions burned in at the moment they
were really emitted (captions.jsonl t_emit; partials dimmed, finals solid, revisions replace the line) and the
face boxes from faces.jsonl (the talking face in its person's colour), plus a contact sheet PNG.

Nothing here re-runs a model or invents a time: a caption appears at frame time >= its t_emit and shows the
latest emission of its segment at that time, so the demo is exactly as late as the live run was.

  python -m unspoken.transcribe.live.render --video CASE/video.mp4 --run RUNDIR --out demo.mp4 [--sheet sheet.png]
"""
from __future__ import annotations

import argparse
import bisect
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .captions import Caption, read_captions
from .web import hex_to_bgr, speaker_color

CAPTION_ROWS = 3
ROW_H = 30
PAD = 10


def _timeline(captions: list[Caption]) -> tuple[list[float], list[Caption]]:
    caps = sorted(captions, key=lambda c: c.t_emit)
    return [c.t_emit for c in caps], caps


def state_at(t: float, times: list[float], caps: list[Caption], state: dict[str, Caption], cursor: int) -> int:
    """Apply every emission with t_emit <= t (in order) to `state`; returns the new cursor."""
    n = bisect.bisect_right(times, t)
    for c in caps[cursor:n]:
        state[c.seg] = c
    return n


def visible_rows(state: dict[str, Caption], t: float, rows: int = CAPTION_ROWS, linger_s: float = 8.0) -> list[Caption]:
    """The newest `rows` segments (by start) whose text is still relevant at t: anything not final yet, and
    finals that ended less than `linger_s` ago."""
    live = [c for c in state.values() if (not c.final) or t - c.end <= linger_s]
    live.sort(key=lambda c: (c.start, c.end))
    return live[-rows:]


def _wrap(text: str, max_chars: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def draw_frame(img: Any, t: float, rows: list[Caption], faces: dict[str, Any] | None, colors: dict[str, str],
               speaking_threshold: float, last_latency: float | None) -> Any:
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    canvas = np.zeros((h + PAD + CAPTION_ROWS * 2 * ROW_H, w, 3), dtype=np.uint8)
    canvas[:h] = img
    canvas[h:] = (18, 18, 18)
    if faces:
        for f in faces.get("faces", []):
            x, y, bw, bh = f["bbox"]
            p0, p1 = (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h))
            talking = float(f.get("speaking", 0.0)) >= speaking_threshold
            key = f"person-{f['seat']}" if f.get("seat") else str(f.get("track_id"))
            color = hex_to_bgr(speaker_color(key, colors)) if talking else (150, 150, 150)
            cv2.rectangle(canvas, p0, p1, color, 2 if talking else 1)
            cv2.putText(canvas, f"{f.get('seat') or f.get('track_id')} {float(f.get('speaking', 0)):.2f}",
                        (p0[0], max(12, p0[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    hud = f"t={t:6.1f}s"
    if last_latency is not None:
        hud += f"   last final caption +{last_latency:.2f}s after speech end"
    cv2.putText(canvas, hud, (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    max_chars = max(20, int(w / 9))
    y = h + PAD + 20
    for c in rows:
        color = hex_to_bgr(speaker_color(c.speaker, colors))
        who = c.speaker.replace("person-", "").replace("voice-", "voice ")
        tag = ("" if c.final else " ...") + (f" (r{c.revision})" if c.final and c.revision else "")
        lines = _wrap(f"{who}: {c.text}{tag}", max_chars)[:2]
        dim = 0.55 if not c.final else 1.0
        col = tuple(int(v * dim) for v in color)
        for ln in lines:
            cv2.putText(canvas, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
            y += ROW_H - 8
        y += 8
        if y > canvas.shape[0] - 6:
            break
    return canvas


def render(video: str | Path, run_dir: str | Path, out: str | Path, *, sheet: str | Path | None = None,
           sheet_frames: int = 10, speaking_threshold: float = 0.5, max_seconds: float | None = None,
           mux_audio: bool = True) -> dict[str, Any]:
    import cv2

    run_dir = Path(run_dir)
    captions = read_captions(run_dir / "captions.jsonl")
    faces_path = run_dir / "faces.jsonl"
    face_frames = [json.loads(l) for l in faces_path.read_text().splitlines() if l.strip()] if faces_path.exists() else []
    face_times = [f["t"] for f in face_frames]
    times, caps = _timeline(captions)
    first_final: dict[str, Caption] = {}
    for c in caps:
        if c.final and c.seg not in first_final:
            first_final[c.seg] = c
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="live-render-"))
    tmp_video = tmp_dir / "frames.mp4"
    writer = None
    state: dict[str, Caption] = {}
    cursor = 0
    colors: dict[str, str] = {}
    sheet_imgs: list[Any] = []
    sheet_at = {int(round(total * (i + 0.5) / sheet_frames)) for i in range(sheet_frames)} if total else set()
    n = 0
    last_latency: float | None = None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = n / fps
        if max_seconds is not None and t > max_seconds:
            break
        cursor = state_at(t, times, caps, state, cursor)
        fin = [c for c in first_final.values() if c.t_emit <= t and c.latency is not None]
        last_latency = max(fin, key=lambda c: c.t_emit).latency if fin else None
        faces = None
        if face_times:
            i = bisect.bisect_right(face_times, t) - 1
            if i >= 0 and t - face_times[i] <= 0.4:
                faces = face_frames[i]
        frame = draw_frame(img, t, visible_rows(state, t), faces, colors, speaking_threshold, last_latency)
        if writer is None:
            writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        if n in sheet_at:
            sheet_imgs.append(frame.copy())
        n += 1
    cap.release()
    if writer is not None:
        writer.release()
    ffmpeg = shutil.which("ffmpeg")
    muxed = False
    if ffmpeg and mux_audio and n:
        enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23"] if _has_encoder(ffmpeg, "libx264") else \
            ["-c:v", "mpeg4", "-q:v", "4"]
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(tmp_video), "-i", str(video), "-map", "0:v:0", "-map", "1:a:0?",
               *enc, "-c:a", "aac", "-shortest", str(out)]
        if subprocess.run(cmd, check=False).returncode == 0 and out.exists():
            muxed = True
    if not muxed and tmp_video.exists():
        shutil.copyfile(tmp_video, out)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    sheet_path = None
    if sheet and sheet_imgs:
        sheet_path = Path(sheet)
        cols = 5 if len(sheet_imgs) >= 5 else len(sheet_imgs)
        rows_n = -(-len(sheet_imgs) // cols)
        fh, fw = sheet_imgs[0].shape[:2]
        import numpy as np
        grid = np.zeros((rows_n * fh, cols * fw, 3), dtype=np.uint8)
        for i, im in enumerate(sheet_imgs):
            r, c = divmod(i, cols)
            grid[r * fh:(r + 1) * fh, c * fw:(c + 1) * fw] = im
        cv2.imwrite(str(sheet_path), grid)
    return {"frames": n, "fps": fps, "out": str(out), "muxed_audio": muxed, "sheet": str(sheet_path) if sheet_path else None,
            "captions": len(captions), "segments": len(first_final)}


def _has_encoder(ffmpeg: str, name: str) -> bool:
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20)
        return f" {name} " in r.stdout
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--run", required=True, help="run dir with captions.jsonl (+ faces.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet", default=None, help="contact sheet PNG path")
    ap.add_argument("--sheet-frames", type=int, default=10)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args(argv)
    r = render(a.video, a.run, a.out, sheet=a.sheet, sheet_frames=a.sheet_frames, max_seconds=a.max_seconds,
               mux_audio=not a.no_audio)
    print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
