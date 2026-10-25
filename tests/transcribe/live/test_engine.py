"""LiveTranscriber end to end on the fixture WAV (real Silero VAD, fake ASR / diarizer / faces / ASD, a fake
video source), as fast as it goes: finals in order with the scripted text, partials for long lines, seats from
strong ASD, binding + revision from weak ASD, the Transcript contract, the latency report, session wipe."""
from __future__ import annotations

import asyncio
import json

import pytest

from unspoken.sensors.audio import WavFileSource
from unspoken.sensors.fakes import FakeASD, FakeASR, FakeFaceTracker, FakeVideoSource
from unspoken.sensors.vad import SileroVAD
from unspoken.transcribe.contracts import load_transcript, to_json
from unspoken.transcribe.live import LiveSettings, LiveTranscriber, SeatBinder
from unspoken.transcribe.live.captions import CaptionLog
from unspoken.transcribe.live.metrics import latency_report

from .conftest import LEFT_FACE, MEETING_WAV, RIGHT_FACE, SEATS, VOICE_OF, TruthASR, TruthDiarizer, WeakASD, talker_for


def _clock():
    import time
    base = time.monotonic()
    return lambda: time.monotonic() - base


async def _run(engine: LiveTranscriber, clock, with_video: bool = True, seconds: float = 36.3):
    audio = WavFileSource(MEETING_WAV, clock=clock, realtime=False, t0=0.0,
                          pace=(lambda: float("inf") if engine.video_done.is_set() else engine.video_media_time + 1.0)
                          if with_video else None)
    video = FakeVideoSource(clock=clock, seconds=seconds, fps=5.0, t0=0.0,
                            pace=lambda: float("inf") if engine.audio_done.is_set() else engine.audio_media_time) \
        if with_video else None
    await asyncio.wait_for(engine.run(audio, video), 60)
    return engine


@pytest.fixture
def faces():
    return FakeFaceTracker([(0.0, [LEFT_FACE, RIGHT_FACE])])


async def test_finals_in_order_with_strong_asd(truth, meeting_pcm, faces, tmp_path):
    clock = _clock()
    captions = []
    engine = LiveTranscriber(vad=SileroVAD(), asr=TruthASR(meeting_pcm, truth), diar=TruthDiarizer(meeting_pcm, truth), faces=faces,
                             asd=FakeASD(talker=talker_for(truth)), settings=LiveSettings(realtime=False, seats=SEATS),
                             clock=clock, log=CaptionLog(tmp_path / "captions.jsonl"), faces_log=tmp_path / "faces.jsonl",
                             on_caption=captions.append)
    await _run(engine, clock)
    finals = [c for c in engine.log.captions if c.final and c.revision == 0]
    assert len(finals) == 12 and engine.stats.finals == 12 and engine.stats.segments == 12
    assert captions == engine.log.captions
    for c, line in zip(finals, truth["lines"]):
        assert c.text == line["text"]
        assert abs(c.start - line["start"]) < 0.5 and abs(c.end - line["end"]) < 0.7
        assert c.voice == VOICE_OF[line["speaker"]]
        expect = {"Priya": "person-left", "Sam": "person-right", "Lee": "voice-C"}[line["speaker"]]
        assert c.speaker == expect, (c, line)
        assert c.source == ("audio" if line["speaker"] == "Lee" else "video")
        assert c.latency is not None and c.t_emit >= c.end - 1e-6
    assert [c.start for c in finals] == sorted(c.start for c in finals)
    # partials: the longer lines got at least one, each tied to its final's segment id and never after it
    partials = [c for c in engine.log.captions if not c.final]
    assert partials and engine.stats.partials == len(partials)
    seg_final_t = {c.seg: c.t_emit for c in finals}
    for p in partials:
        assert p.seg in seg_final_t and p.t_emit <= seg_final_t[p.seg] and p.start <= p.end
        assert p.voice is None                      # TruthDiarizer exposes no centroids: no peek
    # the ASD saw every frame, faces got seats from the zones, the face log has boxes
    assert engine.stats.frames_detected > 150 and engine.stats.frames_stale == 0
    lines = [json.loads(l) for l in (tmp_path / "faces.jsonl").read_text().splitlines()]
    assert lines and {f["seat"] for f in lines[10]["faces"]} == {"left", "right"}
    assert engine.tracks_by_seat() == {"left": "f1", "right": "f2"}
    # transcript contract
    tr = engine.transcript("meeting3", 36.2)
    assert len(tr.utterances) == 12 and tr.duration == 36.2 and tr.session_id == "meeting3"
    assert set(tr.speakers) == {"person-left", "person-right", "voice-C"}
    assert tr.speakers["person-left"].seat == "left" and tr.speakers["person-left"].voice == "voice-A"
    assert tr.speakers["person-left"].track_id == "f1" and not tr.speakers["voice-C"].on_camera
    assert tr.meta["models"]["asr"] == "TruthASR" and tr.meta["stats"]["finals"] == 12
    to_json(tr, tmp_path / "transcript.json")
    assert len(load_transcript(tmp_path / "transcript.json").utterances) == 12
    # latency report shape
    rep = latency_report(engine.log.captions, engine.stats.summary(), threads={"asr_threads": 1}, load=(0, 0, 0))
    assert rep["final_latency_s"]["n"] == 12 and rep["partials"] == len(partials) and rep["rtf"] is not None
    assert rep["frames"]["frames_detected"] == engine.stats.frames_detected
    # session-only state is gone
    assert engine.diar.sessions[0].resets == 1 and faces.sessions[0].resets == 1 and not engine._ring


async def test_weak_asd_binds_voices_and_revises_earlier_captions(truth, meeting_pcm, faces, tmp_path):
    clock = _clock()
    engine = LiveTranscriber(vad=SileroVAD(), asr=TruthASR(meeting_pcm, truth), diar=TruthDiarizer(meeting_pcm, truth), faces=faces,
                             asd=WeakASD(talker_for(truth), level=0.2), settings=LiveSettings(realtime=False, seats=SEATS),
                             clock=clock, binder=SeatBinder(min_dur=4.0))
    await _run(engine, clock)
    caps = engine.log.captions
    firsts = {c.seg: c for c in caps if c.final and c.revision == 0}
    by_start = sorted(firsts.values(), key=lambda c: c.start)
    # Priya's first line (3.2 s) came out before voice-A had 4 s of evidence: a bare voice ...
    assert by_start[0].speaker == "voice-A" and by_start[0].source == "audio"
    # ... and was revised to the left seat once her second line bound the voice
    revs = [c for c in caps if c.final and c.revision > 0]
    assert any(c.seg == by_start[0].seg and c.speaker == "person-left" and c.source == "fused" for c in revs)
    assert engine.stats.revisions == len(revs) >= 1
    assert engine.binder.bound == {"voice-A": "left", "voice-B": "right"}
    tr = engine.transcript("meeting3")
    speakers = {u.text: u.speaker for u in tr.utterances}
    for line in truth["lines"]:
        expect = {"Priya": "person-left", "Sam": "person-right", "Lee": "voice-C"}[line["speaker"]]
        assert speakers[line["text"]] == expect, line
    assert tr.meta["bindings"] == {"voice-A": "left", "voice-B": "right"}
    rev_t = [c.t_emit for c in revs]
    assert all(c.t_emit <= t for c in by_start[:1] for t in rev_t)


async def test_audio_only_and_no_partials(truth, meeting_pcm, tmp_path):
    clock = _clock()
    engine = LiveTranscriber(vad=SileroVAD(), asr=TruthASR(meeting_pcm, truth), settings=LiveSettings(realtime=False, partials=False),
                             clock=clock)
    await _run(engine, clock, with_video=False)
    assert engine.stats.finals == 12 and engine.stats.partials == 0 and engine.video_done.is_set()
    assert engine.models == {"vad": "SileroVAD", "asr": "TruthASR", "diarization": "disabled", "faces": "disabled",
                             "asd": "disabled"}
    assert all(c.speaker == "voice-?" and c.voice is None and c.source == "none" for c in engine.log.captions)
    assert set(engine.transcript("x").speakers) == {"voice-?"}


async def test_empty_asr_text_makes_no_caption(truth):
    clock = _clock()
    engine = LiveTranscriber(vad=SileroVAD(), asr=FakeASR(""), settings=LiveSettings(realtime=False), clock=clock)
    await _run(engine, clock, with_video=False)
    assert engine.stats.segments == 12 and engine.stats.segments_empty == 12 and engine.log.captions == []


def test_open_segment_reads_the_sensors_segmenter():
    vad = SileroVAD()
    stream = vad.stream(16000)
    assert LiveTranscriber.open_segment(stream, 0.0) is None

    class NoSeg:  # a VAD stream that exposes nothing
        pass
    assert LiveTranscriber.open_segment(NoSeg(), 0.0) is None
