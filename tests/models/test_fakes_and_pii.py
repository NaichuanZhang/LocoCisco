"""The fakes are deterministic and implement the contracts; the regex PII detector catches the obvious
leaks and leaves ordinary meeting talk alone."""
from __future__ import annotations

import math
import struct

import pytest

from unspoken.contracts import (ASR, TTS, VAD, ActiveSpeaker, Diarizer, Embedder, FaceTracker, PIIDetector,
                                RoleSpec)
from unspoken.models.audio import mean_abs, pcm16_to_wav, silence, wav_duration_s, wav_to_pcm16
from unspoken.models.fakes import (FakeASD, FakeASR, FakeDiarizer, FakeEmbedder, FakeFaces, FakePII, FakeTTS,
                                   FakeVAD)
from unspoken.models.pii import RegexPII


def tone(seconds: float, amplitude: int = 8000, rate: int = 16000, freq: float = 300.0) -> bytes:
    n = int(seconds * rate)
    return b"".join(struct.pack("<h", int(amplitude * math.sin(2 * math.pi * freq * i / rate))) for i in range(n))


def test_wav_round_trip_and_helpers():
    pcm = tone(0.1)
    wav = pcm16_to_wav(pcm, 16000)
    assert wav[:4] == b"RIFF" and wav_to_pcm16(wav) == (pcm, 16000) and abs(wav_duration_s(wav) - 0.1) < 1e-6
    assert mean_abs(silence(0.5)) == 0 and len(silence(0.5)) == 16000 and mean_abs(pcm) > 1000


def test_fakes_implement_the_protocols():
    assert isinstance(FakeASR(), ASR) and isinstance(FakeTTS(), TTS) and isinstance(FakeVAD(), VAD)
    assert isinstance(FakeDiarizer(), Diarizer) and isinstance(FakeFaces(), FaceTracker)
    assert isinstance(FakeASD(), ActiveSpeaker) and isinstance(FakeEmbedder(), Embedder) and isinstance(FakePII(), PIIDetector)


async def test_fake_asr_scripts_and_silence():
    asr = FakeASR(RoleSpec("asr", "fake", options={"texts": ["hello", "send the deck"]}))
    assert (await asr.transcribe(tone(0.2), 16000)).text == "hello"
    assert (await asr.transcribe(tone(0.2), 16000, lang="zh")).text == "send the deck"
    assert (await asr.transcribe(tone(0.2), 16000)).text == "hello"      # cycles
    plain = FakeASR(RoleSpec("asr", "fake", options={"text": "fixed"}))
    assert (await plain.transcribe(silence(0.5), 16000)).text == "" and (await plain.transcribe(tone(0.1), 16000)).text == "fixed"
    assert await asr.healthy() and asr.calls[1][2] == "zh"


async def test_fake_tts_returns_wav_that_grows_with_the_text():
    tts = FakeTTS()
    short, long = await tts.synthesize("ok"), await tts.synthesize("send the deck by friday", voice="x")
    assert short[:4] == b"RIFF" and wav_duration_s(long) > wav_duration_s(short) and tts.calls[1] == ("send the deck by friday", "x")


def test_fake_vad_cuts_speech_segments_deterministically():
    vad = FakeVAD()
    audio = silence(0.3) + tone(0.5) + silence(0.5) + tone(0.2)
    stream = vad.stream(16000)
    segs = []
    for i in range(0, len(audio), 640):                            # 20 ms frames, any size is fine
        segs.extend(stream.feed(audio[i:i + 640]))
    segs.extend(stream.flush())
    assert len(segs) == 2
    assert abs(segs[0].start - 0.3) < 0.04 and abs(segs[0].end - 0.8) < 0.04 and segs[0].sample_rate == 16000
    assert abs(segs[1].start - 1.3) < 0.04 and mean_abs(segs[0].pcm16) > 1000
    again = vad.stream(16000)
    assert [(s.start, s.end) for s in again.feed(audio) + again.flush()] == [(s.start, s.end) for s in segs]


async def test_fake_diarizer_labels_are_session_stable_and_reset():
    sess = FakeDiarizer().session()
    a1 = await sess.label(tone(0.2, 8000), 16000)
    b1 = await sess.label(tone(0.2, 2000), 16000)
    a2 = await sess.label(tone(0.2, 8100), 16000)
    assert a1.label == "voice-A" and b1.label == "voice-B" and a2.label == "voice-A" and 0 < a1.conf <= 1
    sess.reset()
    assert (await sess.label(tone(0.2, 2000), 16000)).label == "voice-A"


def test_fake_faces_and_asd():
    faces = FakeFaces(RoleSpec("faces", "fake", options={"tracks": [{"track_id": "f1", "bbox": [0.1, 0.1, 0.2, 0.2], "seat": "A"},
                                                                     {"track_id": "f2"}]})).session()
    assert faces.update(None, 0.0) == []
    tracks = faces.update(object(), 1.0)
    assert [t.track_id for t in tracks] == ["f1", "f2"] and tracks[0].seat == "A" and tracks[0].bbox == (0.1, 0.1, 0.2, 0.2)
    asd = FakeASD().session()
    quiet = asd.update(object(), 1.0, tracks, silence(0.1))
    assert [t.speaking for t in quiet] == [0.0, 0.0]
    loud = asd.update(object(), 1.1, tracks, tone(0.1))
    assert [t.speaking for t in loud] == [1.0, 0.0]


async def test_fake_embedder_is_deterministic_normalised_and_semantic_ish():
    e = FakeEmbedder()
    v1, v2, v3 = await e.embed(["send the deck by friday", "send the deck by monday", "turn the lamp off"])
    assert len(v1) == 64 and abs(math.sqrt(sum(x * x for x in v1)) - 1) < 1e-9
    assert (await e.embed(["send the deck by friday"]))[0] == v1
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert dot(v1, v2) > dot(v1, v3)
    assert await e.embed([]) == []


async def test_fake_pii_marks_configured_words():
    p = FakePII(RoleSpec("pii", "fake", options={"words": {"SECRET": "secret", "Sam": "person"}}))
    spans = await p.find("Sam said SECRET twice: SECRET")
    assert [(s.start, s.end, s.label) for s in spans] == [(0, 3, "person"), (9, 15, "secret"), (23, 29, "secret")]
    assert await FakePII().find("nothing here") == []


# --------------------------------------------------------------------------- regex pii
@pytest.mark.parametrize("text,labels", [
    ("mail sam.lee@example.com tonight", ["email"]),
    ("call me at +1 415 555 0133 or (415) 555-0133", ["phone", "phone"]),
    ("the UK office is +44 20 7946 0958", ["phone"]),
    ("budget is $1,200 and about 3k EUR, plus 500 dollars", ["money", "money", "money"]),
    ("the key is sk-abcdefghijklmnopqrstuvwxyz0123 and AKIAABCDEFGHIJKLMNOP", ["secret", "secret"]),
    ("token: ghp_abcdefghijklmnopqrstuvwxyz0123456789", ["secret"]),
    ("password = hunter2222", ["secret"]),
    ("hash 0123456789abcdef0123456789abcdef", ["secret"]),
    ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.e30.abc", ["secret"]),
])
async def test_regex_pii_finds(text, labels):
    spans = await RegexPII().find(text)
    assert [s.label for s in spans] == labels, [(text[s.start:s.end], s.label) for s in spans]
    assert all(0 <= s.start < s.end <= len(text) and 0 < s.score <= 1 for s in spans)


@pytest.mark.parametrize("text", [
    "Sam, send the deck by Friday.",
    "we moved the launch from 2025 to 2026 in room 1204",
    "ticket UNS-4812 is blocked on the 3 reviews",
    "off the record for the next 10 minutes",
    "about 15 people, 2 rooms, 40% done",
])
async def test_regex_pii_leaves_meeting_talk_alone(text):
    assert await RegexPII().find(text) == []


async def test_regex_pii_spans_do_not_overlap_and_are_sorted():
    spans = await RegexPII().find("pay $500 to a@b.co, key sk-0123456789abcdefghijklmnop, +1 415 555 0133")
    assert [s.label for s in spans] == ["money", "email", "secret", "phone"]
    assert all(a.end <= b.start for a, b in zip(spans, spans[1:]))
    assert await RegexPII().healthy()
