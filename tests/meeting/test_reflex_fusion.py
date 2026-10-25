"""Reflex rules (off the record, check-ins), due dates, and fusion (who said it)."""
from __future__ import annotations

import datetime as dt

import pytest

from unspoken.meeting.contracts import MT
from unspoken.meeting.dates import humanize, parse_due
from unspoken.meeting.reflex import checkin_name, checkin_only, detect

from .conftest import JOIN_PRIYA, JOIN_SAM, START, scripted

MONDAY = dt.date(2026, 10, 26)


# ------------------------------------------------------------------ pure rules
@pytest.mark.parametrize("text,expected", [
    ("Off the record: the layoffs are in Q3.", "off_record"),
    ("let's go off the record for a sec", "off_record"),
    ("Can we go off-the-record?", "off_record"),
    ("这个不要记。", "off_record"),
    ("不要记录这段", "off_record"),
    ("Okay, back on the record.", "on"),
    ("可以记了，继续。", "on"),
    ("Sam, send the deck by Friday.", None),
    ("We have a record quarter.", None),
    ("", None),
])
def test_detect_off_record(text, expected):
    assert detect(text) == expected


@pytest.mark.parametrize("text,name", [
    ("Hi, I'm Priya.", "Priya"),
    ("hi I'm Sam", "Sam"),
    ("Hello everyone, this is Lin.", "Lin"),
    ("My name is Alex, joining from Berlin.", "Alex"),
    ("我是李雷", "李雷"),
    ("I'm not sure about that.", None),
    ("I'm going to send it tomorrow.", None),
    ("Sam, can you send the deck by Friday?", None),
    ("I'm freezing.", None),
])
def test_checkin_name(text, name):
    assert checkin_name(text) == name


@pytest.mark.parametrize("text,name", [
    # Actual ASR output from the public AMI replay, not asserted ground-truth spellings/identities.
    ("Yeah that's a good and um his name and what your function is so maybe must you yeah "
     "my name is Francena and I", "Francena"),
    ("And I'm a marketing person. I want to figure out how to sell them. Uhhuh. "
     "And your name is? My name is Eileen.", "Eileen"),
    ("Yeah, uh I'm Januvi. I'm um my role is industrial designer and my responsibilities are uh", "Januvi"),
    ("Very good and as you already know I'm Betsy, I'm the project manager for today.", "Betsy"),
    ("Hi, I'm Priya and I'll send the deck.", "Priya"),
    ("Hi, I'm Priya here.", "Priya"),
    ("I'm Freezing.", None),
    ('She said "My name is Priya."', None),
    ("If I'm Priya, what is my role?", None),
    ("My name is Priya. My name is Sam.", None),
])
def test_natural_checkin_candidates(text, name):
    assert checkin_name(text) == name


@pytest.mark.parametrize("text,expected", [
    ("Hi, I'm Priya.", True),
    ("Hello everyone, this is Lin.", True),
    ("Sam here.", True),
    ("我是李雷", True),
    ("I'm Sam. I'll send it.", False),
    ("My name is Alex, joining from Berlin.", False),
    ("Very good and as you already know I'm Betsy, I'm the project manager for today.", False),
])
def test_only_whole_checkins_skip_planning(text, expected):
    assert checkin_only(text) is expected


@pytest.mark.parametrize("text,expected", [
    ("Sam, send the deck by Friday.", "2026-10-30"),
    ("I'll do it tomorrow.", "2026-10-27"),
    ("EOD today please", "2026-10-26"),
    ("by end of week", "2026-10-30"),
    ("next week", "2026-11-02"),
    ("next Friday", "2026-11-06"),
    ("in 3 days", "2026-10-29"),
    ("on the 9th", "2026-11-09"),
    ("September 16", "2027-09-16"),
    ("due 11/2", "2026-11-02"),
    ("你周三前把报告发我", "2026-10-28"),
    ("下周一给大家", "2026-11-02"),
    ("明天前", "2026-10-27"),
    ("no date here", None),
])
def test_parse_due(text, expected):
    assert parse_due(text, MONDAY) == expected


def test_humanize():
    assert humanize("2026-10-30") == "Fri Oct 30"
    assert humanize(None) == "no due date"


# ------------------------------------------------------------------ reflex agent
async def test_off_record_wipes_transcript_and_lamp_goes_red(make_pipeline):
    p = await make_pipeline()
    await p.say("priya", "We should raise the Team price next year.")
    assert [u["text"] for u in p.rt.transcript()][-1] == "We should raise the Team price next year."   # check-ins are on record too
    assert any("Team price" in c["text"] for c in p.state.screen["captions"])
    await p.say("sam", "Let's go off the record for a second.")
    priv = p.events(MT.PRIVACY)[-1].data
    assert priv["state"] == "off_record" and priv["by"] == "reflex" and priv["wiped_count"] >= 1
    assert p.rt.transcript() == [] and p.state.screen["captions"] == []
    assert p.state.devices["lamp"]["state"]["scene"] == "off_record"
    await p.say("sam", "Their salaries are way above band.")
    last = p.utterances()[-1]["utterance_id"]
    assert p.rt.transcript() == [] and not p.items() and not p.proposals(utterance_id=last)   # not even sent to a model
    assert not any("salaries" in c["text"] for c in p.state.screen["captions"])
    await p.say("priya", "Back on the record.")
    assert p.state.privacy["state"] == "on" and p.state.devices["lamp"]["state"]["scene"] == "talk"
    await p.say("priya", "The Team plan stays at seventy-nine.")
    assert [u["text"] for u in p.rt.transcript()] == ["The Team plan stays at seventy-nine."]
    assert not p.bus.errors


async def test_chat_offrecord_and_resume(make_pipeline):
    p = await make_pipeline()
    await p.say("priya", "The Team plan stays at seventy-nine.")
    await p.steps({"do": "chat", "command": "offrecord", "user": "Priya"})
    assert p.state.privacy["state"] == "off_record" and p.state.privacy["by"] == "user" and p.rt.transcript() == []
    await p.steps({"do": "chat", "command": "resume", "user": "Priya"})
    assert p.state.privacy["state"] == "on"


async def test_wipe_window_keeps_older_lines(make_pipeline):
    p = await make_pipeline(wipe_seconds=5.0)
    await p.say("priya", "Old line, well before the window.")
    p.rt.world.clock += 30
    await p.say("sam", "Recent line.")
    await p.say("sam", "This is off the record.")
    assert [u["text"] for u in p.rt.transcript()][-1] == "Old line, well before the window."
    assert not any("Recent" in u["text"] for u in p.rt.transcript())


async def test_slides_scene_dims_the_lamp_by_rule(make_pipeline):
    p = await make_pipeline()
    await p.steps({"do": "scene", "mode": "slides"})
    props = p.proposals(by="reflex")
    assert props and props[-1]["action"] == "set_light" and props[-1]["route"] == "rule" and props[-1]["args"] == {"scene": "slides"}
    assert p.state.devices["lamp"]["state"]["scene"] == "slides"
    assert p.verdicts(action="set_light")[-1]["verdict"] == "act"


# ------------------------------------------------------------------ fusion
async def test_checkin_binds_name_voice_and_seat(make_pipeline):
    p = await make_pipeline()
    names = {s["name"]: s for s in p.state.participants()}
    assert names["Priya"]["voice"] == "voice-A" and names["Priya"]["seat"] == "s1" and names["Priya"]["guest"] is False
    assert names["Sam"]["voice"] == "voice-B" and names["Sam"]["seat"] == "s2"
    await p.say("sam", "It's freezing in here.")
    utt = p.utterances()[-1]
    assert utt["speaker"]["name"] == "Sam" and utt["source"] == "person" and "Here: Priya (owner), Sam" in utt["context"]


async def test_media_source_when_only_the_screen_talks(make_pipeline):
    p = await make_pipeline()
    await p.steps({"do": "media_say", "text": "Sam, send the deck by Friday."})
    utt = p.utterances()[-1]
    assert utt["source"] == "media" and utt["speaker"]["name"] is None
    assert p.verdicts(action="create_task")[-1]["verdict"] == "deny"
    assert "screen" in p.verdicts(action="create_task")[-1]["rule"]
    assert not p.items()
    assert p.state.screen["captions"][-1]["name"] == "Screen"


async def test_guest_is_a_face_without_checkin_and_leaves(make_pipeline):
    p = await make_pipeline()
    await p.steps({"do": "enter", "who": "visitor", "seat": "s4"})
    assert p.state.guests_present() == 1
    assert "1 guest without a check-in" in p.state.context_line()
    await p.steps({"do": "leave", "who": "visitor"})
    assert p.state.guests_present() == 0


async def test_chat_here_binds_the_next_unknown_voice(make_pipeline):
    p = await make_pipeline(start=False)
    await p.steps(START, JOIN_PRIYA, {"do": "chat", "command": "here", "user": "Lin", "args": {"name": "Lin"}},
                  {"do": "enter", "who": "lin", "seat": "s3", "voice": "voice-C"})
    await p.say("lin", "I can take the finance review.")
    utt = p.utterances()[-1]
    assert utt["speaker"]["name"] == "Lin" and utt["speaker"]["voice"] == "voice-C" and utt["speaker"]["seat"] == "s3"
    assert sorted(p.state.names()) == ["Lin", "Priya"]


async def test_meeting_end_wipes_people(make_pipeline):
    p = await make_pipeline()
    assert len(p.state.participants()) == 2
    await p.steps({"do": "end"})
    assert p.state.participants() == [] and p.state.status["state"] == "ended"
    assert "forgotten" in (p.state.screen.get("banner") or "")


async def test_unknown_voice_with_no_face_is_unknown_source(make_pipeline):
    p = await make_pipeline(start=False)
    await p.steps(START)
    await p.publish(MT.SEGMENT, {"segment_id": "x", "start": 1.0, "end": 2.0, "text": "Can we dim the lights?", "lang": "en",
                                 "voice": "voice-Z", "voice_conf": 0.5}, source="sim.audio", ts=2.0)
    utt = p.utterances()[-1]
    assert utt["source"] == "unknown" and utt["speaker"]["guest"] is True and utt["speaker"]["name"] is None


@pytest.mark.parametrize("confidence", [0.0, 0.244, 0.402, 0.5, 0.69])
async def test_uncertain_intro_does_not_bind_name_or_remove_guest(make_pipeline, confidence):
    p = await make_pipeline(start=False)
    await p.steps(START)
    await p.publish(MT.SEGMENT, {
        "start": 1.0, "end": 2.0, "text": "My name is Alex, and I'm the project manager today.",
        "voice": "voice-Z", "voice_conf": confidence,
    }, ts=2.0)
    sp = p.utterances()[-1]["speaker"]
    assert sp["name"] is None and sp["guest"] is True
    assert sp["conf"] == confidence
    assert "Alex" not in p.state.names()
    assert any("identity not bound" in line for line in p.logs())


async def test_clear_voice_binds_natural_intro(make_pipeline):
    p = await make_pipeline(start=False)
    await p.steps(START)
    await p.publish(MT.SPEAKERS, {
        "t": 1.5, "faces": [{"track_id": "face-1", "seat": "s1", "speaking": 0.1}],
        "screen_talking": False,
    }, ts=1.5)
    await p.publish(MT.SEGMENT, {
        "start": 1.0, "end": 2.0, "text": "My name is Alex, and I'm the project manager today.",
        "voice": "voice-Z", "voice_conf": 0.9,
    }, ts=2.0)
    assert p.utterances()[-1]["speaker"]["name"] == "Alex"
    assert p.utterances()[-1]["speaker"]["guest"] is False


async def test_strong_current_seat_can_bind_when_diarization_is_disabled(make_pipeline):
    p = await make_pipeline(start=False)
    await p.steps(START)
    await p.publish(MT.SPEAKERS, {
        "t": 1.5, "faces": [{"track_id": "face-1", "seat": "s1", "speaking": 0.95}],
        "screen_talking": False,
    }, ts=1.5)
    await p.publish(MT.SEGMENT, {
        "start": 1.0, "end": 2.0, "text": "My name is Alex.", "voice": None, "voice_conf": 0.0,
    }, ts=2.0)
    assert p.utterances()[-1]["speaker"]["name"] == "Alex"
    assert p.utterances()[-1]["speaker"]["seat"] == "s1"


async def test_display_face_is_media_not_a_seated_participant(make_pipeline):
    p = await make_pipeline(scripted("Task", 0.95), start=False)
    await p.steps(START)
    # Real sensors include the display face in faces[] as well as setting screen_talking.
    # The simulator's media_say shortcut does not, so it used to miss this regression.
    await p.publish(MT.SPEAKERS, {
        "t": 1.5, "faces": [{"track_id": "display-1", "seat": "screen", "speaking": 0.99}],
        "screen_talking": True,
    }, ts=1.5)
    await p.publish(MT.SEGMENT, {
        "start": 1.0, "end": 2.0, "text": "Sam, send the deck by Friday.",
        "voice": "voice-Z", "voice_conf": 0.95,
    }, ts=2.0)
    assert p.utterances()[-1]["source"] == "media"
    assert p.state.participants() == []
    assert p.verdicts(action="create_task")[-1]["verdict"] == "deny"
    assert not p.items()


async def _observed_turn(p, text, *, at, voice, confidence, seat="s3", speaking=0.95):
    """Independent sensor-shaped turns; speaking=None supplies no camera evidence."""
    if speaking is not None:
        await p.publish(MT.SPEAKERS, {
            "t": at + 0.5, "faces": [{"track_id": f"face-{seat}", "seat": seat, "speaking": speaking}],
            "screen_talking": seat == "screen",
        }, ts=at + 0.5)
    await p.publish(MT.SEGMENT, {
        "start": at, "end": at + 1, "text": text, "voice": voice, "voice_conf": confidence,
    }, ts=at + 1)
    return p.utterances()[-1]


@pytest.mark.parametrize("weak_text", [
    "I can take the next review.",
    "Very good and as you already know I'm Betsy, I'm the project manager for today.",
])
@pytest.mark.parametrize("later_asd", [0.01, 0.4])
async def test_known_voice_is_anonymous_on_weak_turn_then_recovers(make_pipeline, weak_text, later_asd):
    p = await make_pipeline(start=False)
    await p.steps(START)
    fusion = p.rt.agent("fusion")
    first = await _observed_turn(p, "My name is Francena.", at=1, voice="voice-C", confidence=0.95)
    assert first["speaker"]["name"] == "Francena"
    person = fusion.person_by_voice("voice-C")
    association = (person.name, person.voice, person.seat, person.guest)

    weak = await _observed_turn(p, weak_text, at=11, voice="voice-C", confidence=0.353)
    assert weak["source"] == "person"
    assert weak["speaker"]["name"] is None and weak["speaker"]["guest"] is True
    assert p.state.screen["captions"][-1]["name"] == "Guest"
    assert (person.name, person.voice, person.seat, person.guest) == association
    assert "Betsy" not in p.state.names()

    # The voice is clear even if ASD finds no speaking seat, or only a weak one.
    clear = await _observed_turn(p, "We can continue.", at=21, voice="voice-C",
                                 confidence=0.95, speaking=later_asd)
    assert clear["speaker"]["name"] == "Francena" and clear["speaker"]["guest"] is False
    assert fusion.person_by_voice("voice-C") is person


async def test_known_seat_without_voice_needs_current_strong_seat_evidence(make_pipeline):
    p = await make_pipeline(start=False)
    await p.steps(START)
    await _observed_turn(p, "My name is Alex.", at=1, voice=None, confidence=0.0)
    person = p.rt.agent("fusion").person_by_seat("s3")
    weak = await _observed_turn(p, "We can continue.", at=11, voice=None, confidence=0.0, speaking=0.4)
    assert weak["speaker"]["name"] is None and weak["speaker"]["guest"] is True
    assert person.name == "Alex" and person.guest is False and person.voice is None
    clear = await _observed_turn(p, "We can continue.", at=21, voice=None, confidence=0.0)
    assert clear["speaker"]["name"] == "Alex" and clear["speaker"]["guest"] is False


async def test_weak_voice_cannot_modify_a_named_seat_association(make_pipeline):
    p = await make_pipeline(start=False)
    await p.steps(START)
    await _observed_turn(p, "My name is Alex.", at=1, voice=None, confidence=0.0)
    person = p.rt.agent("fusion").person_by_name("Alex")
    weak = await _observed_turn(p, "We can continue.", at=11, voice="voice-Z", confidence=0.353)
    assert weak["speaker"]["name"] is None and weak["speaker"]["guest"] is True
    assert person.voice is None and person.seat == "s3"
    other = await _observed_turn(p, "We can continue.", at=21, voice="voice-Z",
                                 confidence=0.95, seat="s4")
    assert other["speaker"]["name"] is None and other["speaker"]["guest"] is True
    clear = await _observed_turn(p, "We can continue.", at=31, voice=None, confidence=0.0)
    assert clear["speaker"]["name"] == "Alex"


async def test_new_voice_cannot_inherit_name_from_weak_seat_match(make_pipeline):
    p = await make_pipeline(start=False)
    await p.steps(START)
    await _observed_turn(p, "My name is Alex.", at=1, voice=None, confidence=0.0)
    # Strong clustering confidence does not identify a *new* voice as the named seat's person.
    utt = await _observed_turn(p, "We can continue.", at=11, voice="voice-Z",
                               confidence=0.95, speaking=0.4)
    assert utt["speaker"]["name"] is None and utt["speaker"]["guest"] is True
    assert p.rt.agent("fusion").person_by_name("Alex").voice is None


async def test_unknown_origin_name_pattern_does_not_bootstrap_identity(make_pipeline):
    p = await make_pipeline(start=False)
    await p.steps(START)
    for at in (1, 11):
        utt = await _observed_turn(p, "My name is Alex.", at=at, voice="voice-Z",
                                   confidence=0.99, speaking=None)
        assert utt["source"] == "unknown"
        assert utt["speaker"]["name"] is None and utt["speaker"]["guest"] is True
    assert "Alex" not in p.state.names()


@pytest.mark.parametrize("speaking", [None, 0.95])
async def test_chat_binding_expires_on_first_weak_or_unknown_turn(make_pipeline, speaking):
    p = await make_pipeline(start=False)
    await p.steps(START, {"do": "chat", "command": "here", "user": "Lin"})
    fusion = p.rt.agent("fusion")
    weak = await _observed_turn(p, "We can continue.", at=1, voice="voice-C",
                                confidence=0.353, speaking=speaking)
    assert weak["speaker"]["name"] is None
    assert fusion.pending_chat == []
    other = await _observed_turn(p, "I can take the review.", at=11, voice="voice-D",
                                 confidence=0.95, seat="s4")
    assert other["speaker"]["name"] is None and other["speaker"]["guest"] is True
    assert fusion.person_by_name("Lin").voice is None

    # Retrying is explicit: the same unbound roster entry can arm a fresh next-turn attempt.
    await p.steps({"do": "chat", "command": "here", "user": "Lin"})
    clear = await _observed_turn(p, "I can take the review.", at=21, voice="voice-D",
                                 confidence=0.95, seat="s4")
    assert clear["speaker"]["name"] == "Lin" and clear["speaker"]["voice"] == "voice-D"
    assert fusion.pending_chat == []


async def test_chat_binding_expires_even_when_next_speaker_is_already_known(make_pipeline):
    p = await make_pipeline()
    await p.steps({"do": "chat", "command": "here", "user": "Lin"})
    await p.say("priya", "We can continue.")
    assert p.rt.agent("fusion").pending_chat == []
    other = await _observed_turn(p, "I can take the review.", at=p.rt.world.clock + 10,
                                 voice="voice-C", confidence=0.95)
    assert other["speaker"]["name"] is None and other["speaker"]["guest"] is True


@pytest.mark.parametrize("first_text", ["I can take the review.", "My name is Lin."])
async def test_multiple_chat_registrations_never_guess_remaining_names(make_pipeline, first_text):
    p = await make_pipeline(start=False)
    await p.steps(START, {"do": "chat", "command": "here", "user": "Lin"},
                  {"do": "chat", "command": "here", "user": "Alex"})
    fusion = p.rt.agent("fusion")
    alex_registration = fusion.person_by_name("Alex")
    first = await _observed_turn(p, first_text, at=1, voice="voice-C", confidence=0.95)
    assert first["speaker"]["name"] == ("Lin" if first_text.startswith("My name") else None)
    assert fusion.pending_chat == []
    second = await _observed_turn(p, "I can take the review.", at=11, voice="voice-D",
                                  confidence=0.95, seat="s4")
    assert second["speaker"]["name"] is None and second["speaker"]["guest"] is True
    # Explicit speech still merges the registration, even after the implicit attempt expired.
    explicit = await _observed_turn(p, "My name is Alex.", at=21, voice="voice-D",
                                    confidence=0.95, seat="s4")
    assert explicit["speaker"]["name"] == "Alex"
    assert alex_registration.key not in p.state.people
    assert len([person for person in p.state.people.values() if person.name == "Alex"]) == 1


async def test_media_intro_neither_binds_identity_nor_consumes_chat_attempt(make_pipeline):
    p = await make_pipeline(start=False)
    await p.steps(START, {"do": "chat", "command": "here", "user": "Lin"})
    media = await _observed_turn(p, "My name is Alex.", at=1, voice="voice-screen-A",
                                 confidence=0.99, seat="screen")
    assert media["source"] == "media" and media["speaker"]["name"] is None
    assert "Alex" not in p.state.names()
    assert p.rt.agent("fusion").pending_chat == ["Lin"]
    human = await _observed_turn(p, "I can take the review.", at=11, voice="voice-C", confidence=0.95)
    assert human["speaker"]["name"] == "Lin" and human["speaker"]["guest"] is False
