"""Unit tests for homedemo.decider.questions with FakeDecider: no decider server needed."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from homedemo.contracts import ACTIONS
from homedemo.decider.questions import (
    LIGHT_ACTIONS,
    THIS_ROOM,
    VARIANTS,
    VERSION,
    DeciderQuery,
    Interpretation,
    build_scene_query,
    build_speech_query,
)
from homedemo.decider_client import DeciderAnswer, FakeDecider

ROOT = Path(__file__).resolve().parents[2]
SEED = json.loads((ROOT / "eval" / "seed_cases_2026-10-20.json").read_text(encoding="utf-8"))

ALEX = {"id": "alex", "name": "Alex", "role": "parent", "match": 0.95}
MAYA = {"id": "maya", "name": "Maya", "role": "child", "match": 0.92}


def utt(text: str, room: str = "living", context: str = "Living room. Lights off, evening.", speaker=ALEX):
    return {"room": room, "text": text, "lang": "en", "speaker": speaker, "source": "person",
            "context": context, "wake_word": False}


def scene(text: str = "Time 23:40. Kitchen camera: 1 adult walking in. Kitchen lights: off. Light level: 2 lux.",
          room: str = "kitchen"):
    return {"room": room, "trigger": "person_entered", "text": text}


def choice(probs: dict[str, float]) -> DeciderAnswer:
    top = max(probs, key=probs.get)
    return DeciderAnswer("choice", choice=top, probabilities=probs, confidence=probs[top])


def noul(p: float) -> DeciderAnswer:
    return DeciderAnswer("noul", noul=p, confidence=abs(2 * p - 1), probabilities={"true": p, "false": 1 - p})


def spread(q: DeciderQuery, key: str, top: str, p: float) -> DeciderAnswer:
    """A choice answer putting p on `top` and the rest evenly on the other options of question `key`."""
    opts = list(q.questions[key]["criteria"])
    rest = (1 - p) / (len(opts) - 1)
    return choice({o: (p if o == top else rest) for o in opts})


# --------------------------------------------------------------------------- building
def test_default_version_is_v1_and_v0_selectable():
    assert VERSION == "v1" and set(VARIANTS) >= {"v0", "v1"}
    assert build_speech_query(utt("x")).version == "v1"
    assert build_speech_query(utt("x"), version="v0").version == "v0"
    with pytest.raises(ValueError):
        build_speech_query(utt("x"), version="v99")


def test_v0_speech_matches_measured_wording():
    q = build_speech_query(utt("Can you turn on the lights?"), version="v0")
    intent = q.questions["intent"]
    assert list(q.questions) == ["intent"] and q.gate is None and q.room_key is None
    assert intent["instructions"] == SEED["speech"]["q"]
    measured = SEED["speech"]["opts"]  # the seed omitted "Unlock the door"; everything else must match
    assert {k: v for k, v in intent["criteria"].items() if k != "Unlock the door"} == measured
    assert q.state == 'Living room. Lights off, evening. Heard: "Can you turn on the lights?"'


def test_v0_scene_matches_measured_wording():
    q = build_scene_query(scene(), version="v0")
    assert q.questions["action"]["instructions"] == SEED["scene"]["q"]
    assert q.questions["action"]["criteria"] == SEED["scene"]["opts"]
    assert q.state == scene()["text"]


@pytest.mark.parametrize("version", sorted(VARIANTS))
def test_every_version_builds_well_formed_queries(version):
    for q in (build_speech_query(utt("x"), version=version), build_scene_query(scene(), version=version)):
        assert q.primary in q.questions
        primary = q.questions[q.primary]
        assert primary["type"] == "choice" and len(primary["criteria"]) >= 2
        assert set(primary["criteria"]) == set(q.option_actions)
        assert set(q.option_actions.values()) <= set(ACTIONS) and "none" in q.option_actions.values()
        for key, question in q.questions.items():
            assert question["type"] in {"choice", "noul"} and question["instructions"]
        if q.gate:
            assert q.questions[q.gate]["type"] == "noul"
        if q.room_key:
            assert set(q.questions[q.room_key]["criteria"]) == set(q.option_rooms)
        for line in q.state.splitlines():  # one fact per line, nothing empty
            assert line.strip()


def test_v1_speech_asks_gate_choice_and_room():
    q = build_speech_query(utt("Turn on the kitchen lights."))
    assert list(q.questions) == ["request", "intent", "room"]
    assert q.gate == "request" and q.room_key == "room" and q.speaker_room == "living"
    assert q.option_rooms[THIS_ROOM] is None and q.option_rooms["Kitchen"] == "kitchen"
    assert {"lock_door", "unlock_door", "none"} <= set(q.option_actions.values())
    assert q.state == 'Living room. Lights off, evening. Heard: "Turn on the kitchen lights."'
    labeled = build_speech_query(utt("Turn on the kitchen lights."), layout="labeled")
    assert 'Said: "Turn on the kitchen lights."' in labeled.state and "Speaker: Alex (parent)" in labeled.state


def test_room_question_explains_my_room_for_the_speaker():
    q = build_speech_query(utt("Turn on the light in my room.", speaker=MAYA))
    assert '"my room" when Maya speaks' in q.questions["room"]["criteria"]["Maya's room"]
    q = build_speech_query(utt("Turn on the light in my room.", speaker=ALEX))
    assert '"my room" when Alex speaks' in q.questions["room"]["criteria"]["Parents' bedroom"]
    nobody = {"id": None, "name": None, "role": "unknown", "match": 0.0}
    q = build_speech_query(utt("x", speaker=nobody), layout="labeled", room=True)
    assert "unknown voice" in q.state and "my room" not in json.dumps(q.questions["room"])
    assert "my room" not in json.dumps(build_speech_query(utt("x", speaker=nobody), room=True).questions["room"])


def test_spec_overrides_change_the_query():
    q = build_speech_query(utt("x"), gate=False, room=False)
    assert list(q.questions) == ["intent"] and q.gate is None and q.room_key is None
    q = build_speech_query(utt("x"), none_first=True)
    assert q.option_actions[next(iter(q.questions["intent"]["criteria"]))] == "none"
    q = build_speech_query(utt("x"), split_dim=True)
    assert "lights_dim" in q.option_actions.values()
    q = build_speech_query(utt("x"), extra_actions=False)
    assert "lock_door" not in q.option_actions.values()
    q = build_speech_query(utt("x"), layout="line", facts="state")
    assert q.state.startswith("The house can") and q.state.endswith('Heard: "x"')
    q = build_scene_query(scene(), facts="instructions", gate=False)
    assert q.questions["action"]["instructions"].startswith("The house can")
    q = build_scene_query(scene(), extra_actions=False)
    assert {"lights_off", "cooler"}.isdisjoint(q.option_actions.values())
    q = build_scene_query(scene(), layout="labeled")
    assert q.state.startswith("Room: Kitchen\nSeen: Time 23:40.")


# --------------------------------------------------------------------------- interpreting
@pytest.mark.parametrize("version", sorted(VARIANTS))
@pytest.mark.parametrize("build,inp", [(build_speech_query, utt("I can't see a thing in here.")),
                                       (build_scene_query, scene())])
async def test_probabilities_by_action_sum_to_one_with_fake_decider(version, build, inp):
    q = build(inp, version=version)
    it = q.interpret(await FakeDecider().ask(q.state, q.questions))
    assert isinstance(it, Interpretation)
    assert it.action in ACTIONS and set(it.probabilities) == set(q.option_actions.values())
    assert abs(sum(it.probabilities.values()) - 1) < 1e-9
    assert it.confidence == pytest.approx(it.probabilities[it.action]) and 0 <= it.confidence <= 1
    assert set(it.detail) == set(q.questions)


async def test_scripted_fake_decider_drives_the_action_in_both_versions():
    fake = FakeDecider(script=[("freezing", "intent", ("Warmer", 0.9)), ("freezing", "request", 0.95)])
    for version in ("v0", "v1"):
        q = build_speech_query(utt("It's freezing in here."), version=version)
        it = q.interpret(await fake.ask(q.state, q.questions))
        assert it.action == "warmer" and it.confidence > 0.85 and it.room is None


def test_gate_at_half_is_neutral_and_extremes_flip_the_answer():
    q = build_speech_query(utt("Don't turn off the TV, I'm still watching."), room=True)
    chosen = spread(q, "intent", "TV off", 0.6)
    room = spread(q, "room", THIS_ROOM, 0.9)
    neutral = q.interpret({"request": noul(0.5), "intent": chosen, "room": room})
    assert neutral.action == "tv_off" and neutral.confidence == pytest.approx(0.6)
    vetoed = q.interpret({"request": noul(0.05), "intent": chosen, "room": room})
    assert vetoed.action == "none" and vetoed.probabilities["none"] > vetoed.probabilities["tv_off"]
    backed = q.interpret({"request": noul(0.95), "intent": chosen, "room": room})
    assert backed.action == "tv_off" and backed.confidence > neutral.confidence
    for it in (neutral, vetoed, backed):
        assert abs(sum(it.probabilities.values()) - 1) < 1e-9


def test_gate_mixture_rule_lets_the_gate_own_none():
    q = build_speech_query(utt("x"), combine="mixture", room=True)
    it = q.interpret({"request": noul(0.2), "intent": spread(q, "intent", "Lights on", 0.9),
                      "room": spread(q, "room", THIS_ROOM, 0.9)})
    assert it.action == "none" and it.probabilities["none"] == pytest.approx(0.8)
    assert abs(sum(it.probabilities.values()) - 1) < 1e-9


def test_gate_answer_missing_is_tolerated():
    q = build_speech_query(utt("x"))
    it = q.interpret({"intent": spread(q, "intent", "Lights on", 0.8)})
    assert it.action == "lights_on" and it.confidence == pytest.approx(0.8) and it.room is None


def test_options_sharing_an_action_are_summed():
    q = build_speech_query(utt("x"), version="v0")
    q.option_actions["TV off"] = "none"  # pretend two options mean "none"
    it = q.interpret({"intent": choice({"No action": 0.4, "TV off": 0.3, "Lights on": 0.3})})
    assert it.action == "none" and it.probabilities["none"] == pytest.approx(0.7)


def test_room_mapping_for_light_actions():
    q = build_speech_query(utt("Turn on the kitchen lights."))
    base = {"request": noul(0.95), "intent": spread(q, "intent", "Lights on", 0.95)}
    assert q.interpret({**base, "room": spread(q, "room", "Kitchen", 0.9)}).room == "kitchen"
    assert q.interpret({**base, "room": spread(q, "room", THIS_ROOM, 0.9)}).room is None
    # Naming the room the speaker is already in means "here".
    assert q.interpret({**base, "room": spread(q, "room", "Living room", 0.9)}).room is None
    assert q.interpret({**base, "room": spread(q, "room", "Maya's room", 0.9)}).room == "bedroom_maya"
    it = q.interpret({**base, "room": spread(q, "room", "Front porch", 0.9)})
    assert it.room == "porch" and it.detail["room"]["room"] == "porch"


def test_room_is_none_unless_the_action_is_a_light_action():
    q = build_speech_query(utt("x"), room=True)
    kitchen = spread(q, "room", "Kitchen", 0.9)
    assert set(LIGHT_ACTIONS) == {"lights_on", "lights_off", "lights_dim"}
    for label, action in q.option_actions.items():
        it = q.interpret({"request": noul(0.9), "intent": spread(q, "intent", label, 0.95), "room": kitchen})
        assert it.action == action
        assert (it.room == "kitchen") == (action in LIGHT_ACTIONS)


def test_none_handling_end_to_end():
    q = build_speech_query(utt("My sister said her heater broke last night."), room=True)
    it = q.interpret({"request": noul(0.1), "intent": spread(q, "intent", "Not a request", 0.7),
                      "room": spread(q, "room", "Kitchen", 0.9)})
    assert it.action == "none" and it.room is None
    assert it.confidence == pytest.approx(it.probabilities["none"]) and it.confidence > 0.9
    q0 = build_speech_query(utt("x"), version="v0")
    it0 = q0.interpret({"intent": spread(q0, "intent", "No action", 0.8)})
    assert it0.action == "none" and it0.confidence == pytest.approx(0.8) and it0.room is None


def test_scene_interpret_has_no_room_and_keeps_detail():
    q = build_scene_query(scene())
    it = q.interpret({"act": noul(0.9), "action": spread(q, "action", "Turn on lights", 0.8)})
    assert it.action == "lights_on" and it.room is None
    assert it.detail["act"]["noul"] == 0.9 and it.detail["action"]["choice"] == "Turn on lights"
    hazard = q.interpret({"act": noul(0.97), "action": spread(q, "action", "Alert a parent", 0.9)})
    assert hazard.action == "alert_parent" and hazard.confidence > 0.9


def test_room_auto_asks_only_when_a_place_is_named():
    from homedemo.decider.questions import mentions_room

    assert "room" not in build_speech_query(utt("I can't see a thing.")).questions
    assert "room" in build_speech_query(utt("Turn on the kitchen lights.")).questions
    assert "room" in build_speech_query(utt("Lights on in Maya's room please.")).questions
    assert "room" in build_speech_query(utt("客厅有点暗, turn on the lights")).questions
    assert "room" in build_speech_query(utt("x"), room=True).questions
    assert mentions_room("Porch 的灯开一下") and not mentions_room("好冷啊")
