"""ASR stage tests. No model downloads, no servers: the model backends are faked; the Silero chunker runs on
the vendored ONNX (unspoken/sensors/models) over the synthetic 3-voice fixture. Real models only with
UNSPOKEN_LIVE=1 and the files already on disk."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval.transcribe.asr_timing import align, norm, timing_report
from unspoken.transcribe import asr, asr_chunks, asr_parakeet, asr_server, asr_whisper
from unspoken.transcribe.contracts import AsrResult, AsrSegment, Word, load_asr

FIXTURES = Path(__file__).resolve().parents[1] / "sensors" / "fixtures"
LIVE = os.environ.get("UNSPOKEN_LIVE") == "1"


@pytest.fixture(scope="module")
def meeting():
    from unspoken.sensors.audio import pcm16_to_float, read_wav_pcm16
    return pcm16_to_float(read_wav_pcm16(FIXTURES / "meeting3.wav")), json.loads((FIXTURES / "meeting3.json").read_text())


# ----------------------------------------------------------------------------------------------- chunker
def test_speech_chunks_find_the_lines_on_the_recording_clock(meeting):
    audio, truth = meeting
    chunks = asr_chunks.speech_chunks(audio)
    lines = truth["lines"]
    assert len(chunks) == len(lines), chunks                       # 0.9 s gaps > 500 ms min silence
    for (a, b), line in zip(chunks, lines):
        assert abs(a - line["start"]) < 0.35 and abs(b - line["end"]) < 0.45, (a, b, line)
    assert all(0 <= a < b <= len(audio) / 16000 for a, b in chunks)
    piece = asr_chunks.cut(audio, chunks[0])
    assert piece.dtype == np.float32 and abs(len(piece) / 16000 - (chunks[0][1] - chunks[0][0])) < 1e-3


def test_merge_chunks_respects_gap_and_window():
    chunks = [(0.0, 5.0), (5.5, 9.0), (12.0, 20.0), (20.5, 40.0), (40.7, 41.0)]
    assert asr_chunks.merge_chunks(chunks, max_s=25, gap_s=1.0) == [(0.0, 9.0), (12.0, 20.0), (20.5, 41.0)]
    assert asr_chunks.merge_chunks(chunks, max_s=30, gap_s=1.0) == [(0.0, 9.0), (12.0, 41.0)]
    assert asr_chunks.merge_chunks(chunks, max_s=30, gap_s=0.0) == chunks
    assert asr_chunks.merge_chunks([], max_s=30) == []


# ----------------------------------------------------------------------------------------------- parakeet
def test_words_from_tokens_groups_on_space_or_underscore_and_shifts_times():
    toks = [" Okay", ".", " E", "ver", "yb", "ody", " re", "ady", "?", " ", "▁Uh", "h"]
    times = [0.24, 0.72, 1.04, 1.12, 1.28, 1.44, 1.6, 1.76, 2.0, 2.4, 2.8, 2.96]
    lp = [-0.1] * 12
    words = asr_parakeet.words_from_tokens(toks, times, lp, offset=10.0)
    assert [w.text for w in words] == ["Okay.", "Everybody", "ready?", "Uhh"]
    assert words[0].start == pytest.approx(10.24) and words[0].end == pytest.approx(11.04)   # ends at next token
    assert words[1].start == pytest.approx(11.04) and words[2].end == pytest.approx(12.4)
    assert words[3].end == pytest.approx(12.96 + 2 * asr_parakeet.FRAME_S)                   # last token: +2 frames
    assert all(w.end > w.start for w in words)
    assert words[0].conf == pytest.approx(np.exp(-0.1), abs=1e-3)
    # a long gap does not stretch a word onto the next one
    w2 = asr_parakeet.words_from_tokens([" a", " b"], [0.0, 3.0], None)
    assert w2[0].end == pytest.approx(2 * asr_parakeet.FRAME_S) and w2[0].conf == 1.0
    assert asr_parakeet.words_from_tokens([], [], None) == []


def test_split_segments_cuts_at_pauses():
    ws = [Word(0.0, 0.3, "a", 0.9), Word(0.35, 0.6, "b,", 0.7), Word(1.5, 1.8, "c", 0.5), Word(1.9, 2.0, "d", 1.0)]
    segs = asr_parakeet.split_segments(ws, split_gap_s=0.6)
    assert [(s.start, s.end, s.text) for s in segs] == [(0.0, 0.6, "a b,"), (1.5, 2.0, "c d")]
    assert segs[0].conf == pytest.approx(0.8) and segs[1].words == ws[2:]
    assert asr_parakeet.split_segments([]) == [] and len(asr_parakeet.split_segments(ws, split_gap_s=5)) == 1


def test_parakeet_transcribe_uses_chunks_and_offsets(monkeypatch, meeting):
    audio, truth = meeting
    calls: list[int] = []

    class FakeModel:
        def recognize(self, waves, sample_rate=16000):
            calls.append(len(waves))
            return [SimpleNamespace(text="hi there", tokens=[" hi", " there"], timestamps=[0.08, 0.4],
                                    logprobs=[-0.05, -0.2]) for _ in waves]

    monkeypatch.setattr(asr_parakeet, "load", lambda **kw: FakeModel())
    res = asr_parakeet.transcribe(audio, threads=2, offset_s=-0.08, max_s=0.0, gap_s=0.0, batch_size=5)
    assert res.timing["chunks"] == len(truth["lines"]) and sum(calls) == len(truth["lines"]) and max(calls) <= 5
    assert "fp32" in res.model
    assert "int8" in asr_parakeet.transcribe(audio, threads=2, quantization="int8", max_s=0.0).model
    first = res.segments[0]
    assert first.words[0].start == pytest.approx(truth["lines"][0]["start"], abs=0.5)
    assert first.words[1].start - first.words[0].start == pytest.approx(0.32)        # token frames kept, offset shared
    assert first.words[0].text == "hi" and first.words[1].text == "there" and first.text == "hi there"
    assert 0 < first.conf <= 1 and first.start == first.words[0].start


def test_models_root_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSPOKEN_DEV_MODELS", str(tmp_path))
    assert asr_parakeet.models_root() == tmp_path / "asr"
    assert asr_parakeet.model_dir("int8").name == "parakeet-tdt-0.6b-v3-int8"
    assert asr_parakeet.model_dir(None).name == "parakeet-tdt-0.6b-v3"


# ----------------------------------------------------------------------------------------------- whisper
def _fw_segment(start, end, text, words, logprob=-0.2):
    return SimpleNamespace(start=start, end=end, text=text, avg_logprob=logprob,
                           words=[SimpleNamespace(start=a, end=b, word=w, probability=p) for a, b, w, p in words])


def test_whisper_segments_shift_and_skip_empty():
    segs = [_fw_segment(0.5, 2.0, " Hello world. ", [(0.5, 0.9, " Hello", 0.9), (1.0, 1.0, " world.", 0.7), (1.5, 1.6, " ", 1.0)]),
            _fw_segment(2.0, 3.0, "   ", [])]
    out = asr_whisper._segments(iter(segs), offset=10.0)
    assert len(out) == 1 and out[0].start == 10.5 and out[0].end == 12.0 and out[0].text == "Hello world."
    assert [w.text for w in out[0].words] == ["Hello", "world."]
    assert out[0].words[1].start == 11.0 and out[0].words[1].end == pytest.approx(11.02)   # zero-length word widened
    assert 0 < out[0].conf < 1


def test_whisper_chunked_calls_model_per_chunk(monkeypatch, meeting):
    audio, truth = meeting
    seen: list[float] = []

    class FakeWhisper:
        def transcribe(self, piece, **kw):
            seen.append(len(piece) / 16000)
            assert kw["word_timestamps"] and not kw["vad_filter"] and not kw["condition_on_previous_text"]
            return iter([_fw_segment(0.1, 0.5, "ok", [(0.1, 0.5, " ok", 0.8)])]), None

    monkeypatch.setattr(asr_whisper, "load", lambda size, **kw: FakeWhisper())
    res = asr_whisper.chunked(audio, "base.en", threads=2, beam_size=1)
    assert len(seen) == len(truth["lines"]) == res.timing["chunks"]
    assert res.segments[0].words[0].start == pytest.approx(truth["lines"][0]["start"] + 0.1, abs=0.5)
    assert "live pipeline" in res.model and "beam 1" in res.model


# ----------------------------------------------------------------------------------------------- speech server
def test_server_method_posts_windows_and_shifts_words(meeting):
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    posts: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers["Content-Length"])
            body = self.rfile.read(n)
            posts.append({"path": self.path, "wav": b"RIFF" in body, "verbose": b"verbose_json" in body,
                          "word_gran": b"timestamp_granularities[]" in body, "diar": b"diarization" in body})
            out = {"text": "hi there", "words": [{"word": "hi", "start": 0.1, "end": 0.3},
                                                  {"word": " there ", "start": 0.3, "end": 0.3, "probability": 0.5}]}
            data = _json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        audio, truth = meeting
        res = asr.transcribe(audio, "server", url=f"http://127.0.0.1:{srv.server_port}/v1", max_s=0.0, gap_s=0.0)
    finally:
        srv.shutdown()
    assert len(posts) == len(truth["lines"]) == len(res.segments) == res.timing["chunks"]
    assert all(p["path"] == "/v1/audio/transcriptions" and p["wav"] and p["verbose"] and p["word_gran"] and p["diar"]
               for p in posts)
    first = res.segments[0]
    assert first.text == "hi there" and [w.text for w in first.words] == ["hi", "there"]
    assert first.words[0].start == pytest.approx(first.start + 0.1) and first.words[1].end > first.words[1].start
    assert first.words[1].conf == 0.5 and "speech server" in res.model and res.timing["rtf"] > 0


def test_words_from_body_handles_nested_segments_and_junk():
    body = {"segments": [{"words": [{"word": "a", "start": 1.0, "end": 1.2}, "junk", {"word": "  "}]}]}
    ws = asr_server.words_from_body(body, offset=5.0)
    assert len(ws) == 1 and ws[0].start == 6.0 and ws[0].end == 6.2 and ws[0].text == "a"
    assert asr_server.words_from_body({"text": "nothing"}, 0.0) == []


# ----------------------------------------------------------------------------------------------- entry point
def _fake_method(audio, *, threads, **o):
    n = len(audio) / 16000
    return AsrResult("fake", [AsrSegment(0.0, n, "one two", [Word(0.0, 0.5, "one"), Word(0.5, 1.0, "two")])],
                     {"decode_s": 0.01})


def test_transcribe_timing_fields_and_unknown_method(monkeypatch):
    monkeypatch.setitem(asr.METHODS, "fake", _fake_method)
    res = asr.transcribe(np.zeros(32000, dtype=np.float32), "fake", threads=3)
    t = res.timing
    assert t["audio_s"] == 2.0 and t["threads"] == 3 and t["method"] == "fake" and t["rtf"] == pytest.approx(0.005)
    assert t["wall_s"] >= t["model_load_s"] >= 0 and "load1_start" in t and "x86" in t["host"] or "aarch64" in t["host"]
    with pytest.raises(KeyError):
        asr.transcribe(np.zeros(16000, dtype=np.float32), "nope")


def test_cli_writes_loadable_asr_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(asr.METHODS, "fake", _fake_method)
    out = tmp_path / "o" / "asr.json"
    assert asr.main(["--in", str(FIXTURES / "meeting3.wav"), "--out", str(out), "--method", "fake", "--threads", "2",
                     "--opt", "beam_size=1", "--opt", "offset_s=-0.08", "--opt", "quantization=none"]) == 0
    loaded = load_asr(out)
    assert loaded.model == "fake" and loaded.segments[0].words[1].text == "two" and loaded.timing["threads"] == 2
    assert "RTF" in capsys.readouterr().err
    assert asr.main(["--list"]) == 0 and "parakeet-tdt-0.6b-v3" in capsys.readouterr().out
    assert asr._parse_opts(["a=1", "b=0.5", "c=x", "d=true", "e=none"]) == {"a": 1, "b": 0.5, "c": "x", "d": True, "e": None}


def test_load_audio_wav_and_registry_names():
    a = asr.load_audio(FIXTURES / "meeting3.wav")
    assert a.dtype == np.float32 and abs(len(a) / 16000 - 36.197) < 0.01
    for size in asr.WHISPER_SIZES:
        assert f"fw-{size}" in asr.METHODS and f"vad-fw-{size}" in asr.METHODS and size in asr_whisper.MODEL_IDS


# ----------------------------------------------------------------------------------------------- timing eval
def test_timing_report_aligns_and_measures():
    ref = [{"text": "Okay", "start": 1.0, "end": 1.4}, {"text": "everybody", "start": 1.4, "end": 2.0},
           {"text": "ready", "start": 2.0, "end": 2.5}, {"text": "um", "start": 2.6, "end": 2.7},
           {"text": "twenty-five", "start": 3.0, "end": 3.6}]
    hyp = {"segments": [{"start": 1.0, "end": 4.0, "text": "okay, everybody ready? twenty-five",
                         "words": [{"start": 1.1, "end": 1.5, "text": "okay,"}, {"start": 1.5, "end": 2.1, "text": "everybody"},
                                   {"start": 2.4, "end": 2.6, "text": "ready?"}, {"start": 3.1, "end": 3.7, "text": "Twenty-five"}]}]}
    assert norm("Twenty-five!") == "twentyfive" and len(align(ref, hyp_words_of(hyp))) == 4
    rep = timing_report(ref, hyp)
    assert rep["ref_words"] == 5 and rep["hyp_words"] == 4 and rep["matched"] == 4 and rep["match_rate"] == 0.8
    assert rep["start"]["median_abs"] == 0.1 and rep["start"]["median_signed"] == 0.1 and rep["start"]["p90_abs"] > 0.1
    assert rep["start"]["within_0.25"] == 0.75 and rep["end"]["median_abs"] == 0.1
    empty = timing_report(ref, {"segments": []})
    assert empty["matched"] == 0 and np.isnan(empty["start"]["median_abs"])


def hyp_words_of(asr_json):
    from eval.transcribe.asr_timing import hyp_words
    return hyp_words(asr_json)


# ----------------------------------------------------------------------------------------------- live
@pytest.mark.skipif(not LIVE, reason="UNSPOKEN_LIVE=1 and the Parakeet ONNX files under dev-models/asr")
def test_live_parakeet_on_the_fixture(meeting):
    audio, truth = meeting
    if not asr_parakeet.model_dir(None).is_dir():
        pytest.skip("Parakeet fp32 ONNX not downloaded")
    res = asr.transcribe(audio, "parakeet-tdt-0.6b-v3", threads=4)
    text = " ".join(s.text for s in res.segments).lower()
    assert "priya" in text and "friday" in text and res.timing["rtf"] < 1.0
    words = [w for s in res.segments for w in s.words]
    assert len(words) > 60 and all(w.end > w.start for w in words)
