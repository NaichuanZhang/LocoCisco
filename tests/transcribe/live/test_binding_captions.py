"""SeatBinder: scale-free binding, many voices to one seat, revision, off-camera voices stay off camera;
Caption log + Transcript builder; latency report numbers."""
from __future__ import annotations

import json

from unspoken.transcribe.contracts import load_transcript
from unspoken.transcribe.live.binding import SeatBinder, rank_seats
from unspoken.transcribe.live.captions import Caption, CaptionLog, build_transcript, latest_finals, read_captions
from unspoken.transcribe.live.metrics import dist, latency_report, pct


def test_rank_seats():
    assert rank_seats({}) == (None, 0.0, 0.0)
    assert rank_seats({"left": 0.2}) == ("left", 0.2, 0.0)
    assert rank_seats({"left": 0.2, "right": 0.5}) == ("right", 0.5, 0.2)


def test_binder_binds_on_relative_evidence_and_merges_voices():
    b = SeatBinder(min_dur=3.0)
    # MotionASD-sized numbers: the talker's seat gets ~0.02, the other seat ~0.004
    b.add("voice-A", 4.0, {"left": 0.02, "right": 0.004})
    assert b.recompute() == {"voice-A": (None, "left")}
    b.add("voice-E", 2.0, {"left": 0.003, "right": 0.015})
    assert b.recompute() == {}                      # 2 s of evidence: not yet
    b.add("voice-E", 2.0, {"left": 0.003, "right": 0.015})
    assert b.recompute() == {"voice-E": (None, "right")}
    # a second voice of the LEFT person (the online diarizer split them) joins the same seat
    b.add("voice-F", 3.5, {"left": 0.018, "right": 0.002})
    assert b.recompute() == {"voice-F": (None, "left")}
    assert b.seat_of("voice-A") == b.seat_of("voice-F") == "left" and b.seat_of("voice-E") == "right"
    assert b.seat_of("voice-Z") is None and b.seat_of(None) is None


def test_off_camera_voice_with_even_noise_does_not_bind():
    b = SeatBinder(min_dur=3.0)
    b.add("voice-A", 6.0, {"left": 0.02, "right": 0.004})
    b.recompute()
    b.add("voice-C", 6.0, {"left": 0.005, "right": 0.004})     # leaks a little onto both seats
    assert b.recompute() == {}
    assert b.seat_of("voice-C") is None
    b.add("voice-D", 6.0, {"left": 0.0, "right": 0.0})          # pure zeros never bind
    assert b.recompute() == {} and b.seat_of("voice-D") is None


def test_binder_revises_when_evidence_flips():
    b = SeatBinder(min_dur=2.0)
    b.add("voice-A", 2.0, {"left": 0.02, "right": 0.005})
    assert b.recompute() == {"voice-A": (None, "left")}
    for _ in range(6):
        b.add("voice-A", 2.0, {"left": 0.002, "right": 0.03})
    assert b.recompute() == {"voice-A": ("left", "right")}
    assert b.seat_of("voice-A") == "right"
    b.reset()
    assert b.bound == {} and b.dur == {}


def test_decide_priorities():
    b = SeatBinder(min_dur=1.0)
    d = b.decide("voice-A", 0.8, {"left": 0.7, "right": 0.1})           # strong per-segment ASD wins
    assert (d.speaker, d.source, d.from_binding) == ("person-left", "video", False)
    d = b.decide("voice-A", 0.8, {"left": 0.02, "right": 0.01})         # nothing known: bare voice
    assert (d.speaker, d.source, d.from_binding) == ("voice-A", "audio", True)
    d = b.decide(None, 0.0, {})
    assert (d.speaker, d.source) == ("voice-?", "none")
    b.add("voice-A", 3.0, {"left": 0.02, "right": 0.004})
    b.recompute()
    d = b.decide("voice-A", 0.8, {"left": 0.01, "right": 0.009})        # weak ASD: the binding decides
    assert (d.speaker, d.source, d.from_binding) == ("person-left", "fused", True)
    d = b.decide("voice-A", 0.8, {"left": 0.1, "right": 0.9})           # strong contradiction overrides
    assert (d.speaker, d.source) == ("person-right", "video")
    d = b.decide(None, 0.0, {"left": 0.45, "right": 0.1})               # moderate ASD, no voice
    assert (d.speaker, d.source) == ("person-left", "video")


def test_caption_log_and_transcript(tmp_path):
    log = CaptionLog(tmp_path / "captions.jsonl")
    log.append(Caption("s1", 2.0, 1.0, 1.9, "voice-?", "hello", False, 0))
    log.append(Caption("s1", 3.2, 1.0, 2.5, "voice-A", "hello there", True, 0, "voice-A", 0.7, "audio",
                       [{"start": 1.0, "end": 1.4, "text": "hello", "conf": 1.0},
                        {"start": 1.5, "end": 2.4, "text": "there", "conf": 1.0}], 0.7))
    log.append(Caption("s2", 6.1, 4.0, 5.5, "person-right", "yes", True, 0, "voice-B", 0.9, "video", [], 0.6))
    log.append(Caption("s1", 6.2, 1.0, 2.5, "person-left", "hello there", True, 1, "voice-A", 0.8, "fused",
                       log.captions[1].words, None))
    log.close()
    caps = read_captions(tmp_path / "captions.jsonl")
    assert [c.to_dict() for c in caps] == [c.to_dict() for c in log.captions]
    assert set(json.loads((tmp_path / "captions.jsonl").read_text().splitlines()[0])) >= {
        "t_emit", "start", "end", "speaker", "text", "final"}
    fin = latest_finals(caps)
    assert fin["s1"].revision == 1 and fin["s1"].speaker == "person-left" and fin["s2"].speaker == "person-right"
    tr = build_transcript(caps, "sess", 10.0, {"method": "test"}, {"left": "f1"})
    assert [u.speaker for u in tr.utterances] == ["person-left", "person-right"]
    assert tr.speakers["person-left"].seat == "left" and tr.speakers["person-left"].on_camera
    assert tr.speakers["person-left"].voice == "voice-A" and tr.speakers["person-left"].track_id == "f1"
    assert tr.speakers["person-right"].label == "Right person" and tr.speakers["person-right"].track_id is None
    assert tr.utterances[0].words[1].text == "there" and tr.utterances[0].words[1].speaker == "person-left"
    assert tr.utterances[0].conf == 0.8 and tr.utterances[0].source == "fused" and tr.meta["method"] == "test"
    # round-trips through the integrator's loader
    from unspoken.transcribe.contracts import to_json
    to_json(tr, tmp_path / "transcript.json")
    back = load_transcript(tmp_path / "transcript.json")
    assert back.duration == 10.0 and len(back.utterances) == 2 and back.speakers["person-right"].seat == "right"


def test_metrics():
    assert pct([], 0.5) is None and dist([])["n"] == 0
    assert pct([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0 and pct([1.0, 2.0, 3.0, 4.0, 5.0], 0.95) == 4.8
    caps = [Caption("a", 2.0, 0.5, 1.5, "voice-A", "x", False, 0),          # partial: staleness 0.5
            Caption("a", 3.0, 0.5, 2.0, "voice-A", "x y", True, 0, latency=1.0),
            Caption("a", 7.0, 0.5, 2.0, "person-left", "x y", True, 1),      # revision 4 s later
            Caption("b", 9.0, 5.0, 6.0, "person-left", "z", True, 0, latency=3.0)]
    stats = {"audio_s": 10.0, "wall_s": 10.0, "cpu_s": 15.0, "finals": 2, "partials": 1, "frames_detected": 10,
             "asr_ms_avg": 500.0, "partial_asr_ms_avg": 400.0, "diar_ms_avg": 100.0, "faces_ms_avg": 20.0, "asd_ms_avg": 1.0}
    r = latency_report(caps, stats, threads={"asr_threads": 4}, load=(1.0, 2.0, 3.0))
    assert r["final_latency_s"]["n"] == 2 and r["final_latency_s"]["p50"] == 2.0 and r["final_latency_s"]["max"] == 3.0
    assert r["partial_latency_s"] == {"n": 1, "p50": 0.5, "p95": 0.5, "max": 0.5, "mean": 0.5}
    assert r["first_partial_delay_s"]["p50"] == 1.5 and r["revision_delay_s"]["p50"] == 4.0
    assert r["rtf"] == 1.0 and r["cpu_rtf"] == 1.5 and r["cpu_cores_avg"] == 1.5
    assert r["busy_rtf"] == round((500 * 2 + 400 * 1 + 100 * 2 + 20 * 10 + 1 * 10) / 1000 / 10, 3)
    assert r["threads"] == {"asr_threads": 4} and r["load_avg"] == [1.0, 2.0, 3.0] and r["revisions"] == 1


def test_dissimilar_voice_does_not_join_an_occupied_seat():
    sims = {("voice-A", "voice-B"): 0.05, ("voice-A", "voice-F"): 0.62}
    b = SeatBinder(min_dur=3.0, similarity=lambda x, y: sims.get((x, y), sims.get((y, x))))
    b.add("voice-A", 6.0, {"left": 0.02, "right": 0.004})
    assert b.recompute() == {"voice-A": (None, "left")}
    b.add("voice-B", 6.0, {"left": 0.02, "right": 0.004})      # a different person, but the ASD votes left
    assert b.recompute() == {} and b.seat_of("voice-B") is None  # stays an off-camera voice, not merged
    b.add("voice-F", 6.0, {"left": 0.02, "right": 0.004})      # sounds like voice-A: the same person, merged
    assert b.recompute() == {"voice-F": (None, "left")}
    b.add("voice-G", 6.0, {"left": 0.02, "right": 0.004})      # unknown similarity: allowed
    assert b.recompute() == {"voice-G": (None, "left")}
