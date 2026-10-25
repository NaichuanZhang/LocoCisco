"""The loopback viewer (page, MJPEG, SSE, state), the demo renderer on a synthetic clip, and the CLI with an
injected fake registry (file replay in fast mode, outputs written)."""
from __future__ import annotations

import json
import socket
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from unspoken.contracts import FaceTrack
from unspoken.sensors.fakes import FakeASD, FakeFaceTracker, FakeRegistry
from unspoken.sensors.vad import SileroVAD
from unspoken.sensors.video import VideoFrame
from unspoken.transcribe.contracts import load_transcript
from unspoken.transcribe.live.captions import Caption, CaptionLog
from unspoken.transcribe.live.render import render, state_at, visible_rows
from unspoken.transcribe.live.web import LiveViewer, annotate, hex_to_bgr, speaker_color

from .conftest import LEFT_FACE, MEETING_WAV, RIGHT_FACE, TruthASR, TruthDiarizer, talker_for, write_tiny_mp4


def _get(url: str, timeout: float = 5.0) -> tuple[int, bytes, str]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read(), r.headers.get("Content-Type", "")


def _read_some(host: str, port: int, path: str, nbytes: int = 400, timeout: float = 5.0) -> bytes:
    s = socket.create_connection((host, port), timeout=timeout)
    s.sendall(f"GET {path} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode())
    buf = b""
    try:
        while len(buf) < nbytes:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
    except socket.timeout:
        pass
    finally:
        s.close()
    return buf


def test_colors():
    assert speaker_color("person-left", {}) == "#4fc3f7" and hex_to_bgr("#4fc3f7") == (0xF7, 0xC3, 0x4F)
    cache: dict[str, str] = {}
    a, b = speaker_color("voice-B", cache), speaker_color("voice-C", cache)
    assert a != b and speaker_color("voice-B", cache) == a


def test_viewer_serves_page_stream_events_and_state():
    v = LiveViewer("127.0.0.1", 0, fps=1000.0).start()
    try:
        status, body, ctype = _get(v.url)
        assert status == 200 and b"<html" in body and "text/html" in ctype and b"/events" in body
        img = np.full((48, 96, 3), 60, np.uint8)
        v.on_frame(VideoFrame(img, 1.0), [FaceTrack("f1", (0.1, 0.1, 0.3, 0.5), "left", 0.9),
                                         FaceTrack("f2", (0.6, 0.1, 0.3, 0.5), "right", 0.1)], 1.0)
        v.on_caption(Caption("s1", 1.5, 0.2, 1.1, "person-left", "hello", True, 0, "voice-A", 0.8, "video", [], 0.4))
        v.on_caption(Caption("s2", 1.9, 1.3, 1.8, "voice-?", "so", False, 0))
        status, body, ctype = _get(v.url + "state")
        st = json.loads(body)
        assert st["frames"] == 1 and st["events"] == 2 and st["latency_last"] == 0.4 and st["latency_p50"] == 0.4
        assert [c["seg"] for c in st["captions"]] == ["s1", "s2"] and "words" not in st["captions"][0]
        raw = _read_some(v.host, v.port, "/stream.mjpg", 600)
        assert b"multipart/x-mixed-replace" in raw and b"--frame" in raw and b"\xff\xd8" in raw
        raw = _read_some(v.host, v.port, "/events", 300)
        assert b"text/event-stream" in raw
        ev = json.loads(raw.split(b"data: ", 1)[1].split(b"\n\n", 1)[0])
        assert ev["type"] == "caption" and ev["seg"] == "s1" and ev["color"] == "#4fc3f7" and "words" not in ev
        with pytest.raises(urllib.error.HTTPError):
            _get(v.url + "nope")
    finally:
        v.stop()
    with pytest.raises(ValueError):
        LiveViewer("0.0.0.0", 0)


def test_annotate_draws_on_a_copy():
    img = np.zeros((40, 80, 3), np.uint8)
    out = annotate(img, [FaceTrack("f1", (0.1, 0.1, 0.4, 0.6), "left", 0.9)], {}, 0.5, 3.0, 0.7)
    assert out.shape == img.shape and out.any() and not img.any()


def test_render_burns_captions_at_t_emit(tmp_path):
    import cv2

    video = write_tiny_mp4(tmp_path / "clip.mp4", seconds=2.0, fps=10.0)
    run = tmp_path / "run"
    log = CaptionLog(run / "captions.jsonl")
    log.append(Caption("a", 0.5, 0.1, 0.4, "voice-?", "hel", False, 0))
    log.append(Caption("a", 0.9, 0.1, 0.8, "person-left", "hello there", True, 0, "voice-A", 0.8, "video", [], 0.1))
    log.append(Caption("a", 1.5, 0.1, 0.8, "person-right", "hello there", True, 1, "voice-A", 0.8, "fused", [], None))
    log.close()
    (run / "faces.jsonl").write_text("\n".join(json.dumps({"t": t / 10, "faces": [
        {"track_id": "f1", "seat": "left", "bbox": [0.1, 0.1, 0.3, 0.6], "speaking": 0.9 if t < 10 else 0.0}]})
        for t in range(20)) + "\n")
    # the timeline logic: nothing before the first t_emit, partial then final then the revision
    caps = sorted(log.captions, key=lambda c: c.t_emit)
    times = [c.t_emit for c in caps]
    state: dict[str, Caption] = {}
    cur = state_at(0.4, times, caps, state, 0)
    assert state == {} and visible_rows(state, 0.4) == []
    cur = state_at(0.6, times, caps, state, cur)
    assert state["a"].final is False
    cur = state_at(1.0, times, caps, state, cur)
    assert state["a"].final and state["a"].revision == 0
    state_at(1.6, times, caps, state, cur)
    assert state["a"].speaker == "person-right" and visible_rows(state, 1.6)[0].revision == 1
    out = render(video, run, tmp_path / "demo.mp4", sheet=tmp_path / "sheet.png", sheet_frames=4, mux_audio=False)
    assert out["frames"] == 20 and out["segments"] == 1 and out["captions"] == 3
    cap = cv2.VideoCapture(str(tmp_path / "demo.mp4"))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[0] > 48 and frame.shape[1] == 96         # caption band appended below the video
    sheet = cv2.imread(str(tmp_path / "sheet.png"))
    assert sheet is not None and sheet.shape[1] == 4 * 96


async def test_cli_runs_a_file_in_fast_mode_and_writes_outputs(truth, meeting_pcm, tmp_path):
    from unspoken.transcribe.live.__main__ import build_parser, run

    video = write_tiny_mp4(tmp_path / "clip.mp4", seconds=36.5, fps=5.0, size=(96, 48))
    reg = FakeRegistry(vad=SileroVAD(), asr=TruthASR(meeting_pcm, truth), diarizer=TruthDiarizer(meeting_pcm, truth),
                       faces=FakeFaceTracker([(0.0, [LEFT_FACE, RIGHT_FACE])]), asd=FakeASD(talker=talker_for(truth)))
    out = tmp_path / "out" / "transcript.json"
    cfg = tmp_path / "sensors.toml"
    cfg.write_text('[video]\nwidth = 96\nheight = 48\n[seats]\nleft = [0.0, 0.0, 0.5, 1.0]\nright = [0.5, 0.0, 0.5, 1.0]\n')
    args = build_parser().parse_args(["--video", str(video), "--audio", str(MEETING_WAV), "--fast", "--out", str(out),
                                      "--session-id", "meeting3", "--view-fps", "5", "--config", str(cfg)])
    assert await run(args, registry=reg) == 0
    tr = load_transcript(out)
    assert tr.session_id == "meeting3" and len(tr.utterances) == 12 and abs(tr.duration - 36.2) < 0.2
    assert {u.speaker for u in tr.utterances} == {"person-left", "person-right", "voice-C"}
    lat = json.loads((out.parent / "latency.json").read_text())
    assert lat["finals"] == 12 and lat["final_latency_s"]["p50"] is not None and lat["realtime"] is False
    assert lat["threads"]["asr_threads"] == 4 and lat["frames"]["frames_detected"] > 0
    lines = (out.parent / "captions.jsonl").read_text().splitlines()
    assert len(lines) == lat["finals"] + lat["partials"] + lat["revisions"]
    assert (out.parent / "faces.jsonl").exists()


def test_cli_rejects_missing_source(tmp_path):
    import asyncio

    from unspoken.transcribe.live.__main__ import build_parser, run

    args = build_parser().parse_args(["--out", str(tmp_path / "t.json")])
    assert asyncio.run(run(args, registry=FakeRegistry())) == 2
