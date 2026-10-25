"""Perception scene text (measured style), triggers, debounce; voice utterance and context."""
from __future__ import annotations

from homedemo.contracts import T


async def test_dark_kitchen_entry_text_matches_measured_style(make_pipeline):
    p = make_pipeline()
    await p.clock("23:40")
    await p.lux("kitchen", 2)
    await p.camera("kitchen", [p.person("alex")])
    await p.settle()
    scenes = p.scenes(room="kitchen")
    assert len(scenes) == 1
    scene = scenes[0]
    assert scene.source == "agent.perception"
    assert scene.data["trigger"] == "person_entered"
    assert scene.data["text"] == ("Time 23:40. Kitchen camera: 1 adult walking in. "
                                  "Kitchen lights: off. Light level: 2 lux (dark).")
    facts = scene.data["facts"]
    assert facts["people"] == [{"id": "alex", "role": "parent", "age_group": "adult", "activity": None}]
    assert facts["lux"] == 2 and facts["lights_on"] is False and facts["time"] == "23:40"
    # chained to the camera event that triggered it, with that event's ts
    cam = p.events(T.CAMERA)[0]
    assert scene.cause == cam.id and scene.ts == cam.ts


async def test_hazard_text_and_trigger(make_pipeline):
    p = make_pipeline([("stove", "action", ("Alert a parent", 0.97))])
    await p.clock("17:20")
    await p.device("light_kitchen", {"on": True, "brightness": 100})
    await p.device("stove_kitchen", {"burner_on": True})
    await p.camera("kitchen", [p.person("maya", pose="standing on a chair", near="stove")])
    await p.settle()
    hazards = p.scenes(trigger="hazard")
    assert len(hazards) == 1
    assert hazards[0].data["text"] == ("Time 17:20. Kitchen camera: 1 child (about 7 years old) standing on a chair "
                                       "next to the stove. Stove burner: on. No adult in the kitchen.")
    assert hazards[0].data["facts"]["hazard"] is True


async def test_pose_restated_by_the_activity_is_said_once(make_pipeline):
    p = make_pipeline([("stove", "action", ("Alert a parent", 0.97))])
    await p.clock("17:20")
    await p.device("stove_kitchen", {"burner_on": True})
    await p.camera("kitchen", [p.person("maya", pose="standing on chair", near="stove")],
                   activity="standing on a chair next to the stove, reaching for the pot")
    await p.settle()
    text = p.scenes(trigger="hazard")[0].data["text"]
    assert text.startswith("Time 17:20. Kitchen camera: 1 child (about 7 years old) standing on a chair "
                           "next to the stove, reaching for the pot. Stove burner: on.")
    assert text.count("stove,") == 1 and "on chair" not in text


async def test_no_hazard_when_an_adult_is_present(make_pipeline):
    p = make_pipeline()
    await p.clock("17:20")
    await p.device("stove_kitchen", {"burner_on": True})
    await p.camera("kitchen", [p.person("maya", pose="standing on a chair", near="stove"),
                               p.person("sam", track="t2", pose="at the stove")])
    await p.settle()
    assert not p.scenes(trigger="hazard")
    assert "No adult" not in p.scenes()[0].data["text"]


async def test_hazard_rises_when_the_stove_turns_on_later(make_pipeline):
    p = make_pipeline([("stove", "action", ("Alert a parent", 0.97))])
    await p.clock("17:20")
    await p.camera("kitchen", [p.person("maya", pose="standing on a chair", near="stove")])
    await p.settle()
    assert not p.scenes(trigger="hazard")
    await p.device("stove_kitchen", {"burner_on": True})
    await p.settle()
    assert len(p.scenes(trigger="hazard")) == 1


async def test_activity_idle_and_env_triggers(make_pipeline):
    p = make_pipeline([("couch", "action", ("Do nothing", 0.9))])
    await p.clock("19:00")
    await p.temp("living", 17)
    await p.camera("living", [p.person("alex", pose="sitting on the couch")])
    await p.camera("living", [p.person("alex", pose="sitting on the couch")],
                   activity="putting on a sweater, rubbing their arms")
    await p.settle()
    scenes = p.scenes(room="living")
    assert [s.data["trigger"] for s in scenes] == ["person_entered", "activity_changed"]
    assert scenes[1].data["text"] == ("Time 19:00. Living room camera: 1 adult sitting on the couch, "
                                      "putting on a sweater, rubbing their arms. TV: off. "
                                      "Room temperature: 17 C. Thermostat set to 19 C.")
    await p.device("tv_living", {"on": True, "volume": 30, "playing": "movie"})
    await p.settle()
    assert p.scenes(room="living")[-1].data["trigger"] == "env_changed"
    assert "TV: on, volume 30." in p.scenes(room="living")[-1].data["text"]
    await p.camera("living", [p.person("alex", pose="lying on the couch, eyes closed", idle_min=40)])
    await p.settle()
    idle = p.scenes(trigger="idle")
    assert len(idle) == 1 and "no movement for 40 minutes" in idle[0].data["text"]
    assert idle[0].data["facts"]["idle_min"] == 40


async def test_idle_detected_from_a_clock_jump_while_nobody_moves(make_pipeline):
    p = make_pipeline([("no movement", "action", ("Turn off TV", 0.9))])
    await p.clock("00:50")
    await p.device("light_living", {"on": True, "brightness": 100})
    await p.device("tv_living", {"on": True, "volume": 30, "playing": "movie"})
    await p.camera("living", [p.person("alex", pose="lying on the couch, eyes closed")])
    await p.motion("living", False)
    await p.settle()
    assert not p.scenes(trigger="idle")
    await p.clock("01:30")                                   # the scenario's idle step jumps 40 minutes
    await p.settle()
    idle = p.scenes(trigger="idle")
    assert len(idle) == 1
    assert idle[0].data["text"] == ("Time 01:30. Living room camera: 1 adult lying on the couch, eyes closed, "
                                    "no movement for 40 minutes. TV: on, volume 30.")
    assert idle[0].data["facts"]["idle_min"] == 40 and idle[0].data["facts"]["tv_on"] is True
    assert p.commands(device_id="tv_living")[0].data["command"] == "off"
    await p.clock("09:00")                                   # a 7.5 h skip is a scenario jump, not idleness
    await p.settle()
    assert len(p.scenes(trigger="idle")) == 1


async def test_entry_with_a_caption_reads_like_the_measured_case(make_pipeline):
    p = make_pipeline()
    await p.clock("23:40")
    await p.lux("kitchen", 2)
    await p.camera("kitchen", [p.person("alex")], activity="walking in from the hallway")
    await p.settle()
    assert p.scenes()[0].data["text"] == ("Time 23:40. Kitchen camera: 1 adult walking in from the hallway. "
                                          "Kitchen lights: off. Light level: 2 lux (dark).")
    assert p.commands(device_id="light_kitchen")              # and the reflex still fires on it


async def test_same_frame_and_identical_text_do_not_republish(make_pipeline):
    p = make_pipeline()
    await p.clock("23:40")
    await p.lux("kitchen", 2)
    await p.camera("kitchen", [p.person("alex")])
    await p.camera("kitchen", [p.person("alex")])          # same frame again
    await p.lux("kitchen", 2)                               # same env again (identical text, < 2 s)
    await p.settle()
    assert len(p.scenes(room="kitchen")) == 1
    await p.camera("kitchen", [])                           # room empties: no scene for an empty room
    await p.settle()
    assert len(p.scenes(room="kitchen")) == 1
    await p.camera("kitchen", [p.person("alex", track="t9")])   # back within 2 s with the same text: debounced
    await p.settle()
    assert len(p.scenes(trigger="person_entered")) == 1
    p.agent("perception").rooms["kitchen"].last_wall -= 3.0     # 3 s later the same entry is a new scene
    await p.camera("kitchen", [])
    await p.camera("kitchen", [p.person("alex", track="t10")])
    await p.settle()
    assert len(p.scenes(trigger="person_entered")) == 2


async def test_pet_and_room_without_camera(make_pipeline):
    p = make_pipeline([("cat", "action", ("Do nothing", 0.9))])
    await p.clock("02:15")
    await p.lux("kitchen", 1)
    await p.camera("kitchen", [p.pet("cat")])
    await p.lux("bedroom_maya", 1)
    await p.motion("bedroom_maya", True)
    await p.motion("bedroom_maya", True)                    # still there, not a new entry
    await p.settle()
    cat = p.scenes(room="kitchen")[0]
    assert cat.data["trigger"] == "activity_changed"
    assert cat.data["text"] == ("Time 02:15. Kitchen camera: 1 cat. No people detected. "
                                "Kitchen lights: off. Light level: 1 lux (dark).")
    assert cat.data["facts"]["people"] == [] and cat.data["facts"]["pets"] == [{"id": None, "species": "cat"}]
    bedroom = p.scenes(room="bedroom_maya")
    assert len(bedroom) == 1 and bedroom[0].data["trigger"] == "person_entered"
    assert bedroom[0].data["text"].startswith("Time 02:15. Maya's room (no camera): presence sensor detects someone")
    assert bedroom[0].data["facts"]["people"] == [{"id": None, "role": "unknown", "age_group": None, "activity": None}]


async def test_voice_utterance_fields_and_context(make_pipeline):
    p = make_pipeline([("Heard", "intent", ("No action", 0.9))])
    await p.clock("20:10")
    await p.temp("living", 17)
    await p.device("tv_living", {"on": True, "volume": 20, "playing": None})
    await p.say("It's freezing in here.", who="sam", room="living", source="person")
    await p.say("Turn off the lights", who=None, room="living", source="tv")
    await p.say("Unlock the front door", who=None, room="porch", source="outside")
    await p.settle()
    utts = p.events(T.UTTERANCE)
    assert [u.data["source"] for u in utts] == ["person", "tv", "outside"]
    assert utts[0].data["speaker"] == {"id": "sam", "name": "Sam", "role": "parent", "match": 0.95}
    assert utts[0].data["context"] == "Living room. Lights off, evening. TV on. Room 17 C."
    assert utts[0].data["text"] == "It's freezing in here." and utts[0].data["lang"] == "en"
    assert utts[0].source == "agent.voice" and utts[0].cause == p.events(T.AUDIO)[0].id
    assert utts[1].data["speaker"]["id"] is None and utts[1].data["speaker"]["role"] == "unknown"
    assert utts[2].data["context"] == "Front porch. Lights off, evening."


async def test_someone_leaving_is_person_left_and_no_decider_is_asked(make_pipeline):
    """Only a person walking out: nothing new to decide, so no slow model call (scenarios2 finding)."""
    p = make_pipeline()
    await p.clock("21:00")
    await p.camera("living", [p.person("alex", track="t1"), p.person("sam", track="t2")])
    await p.settle()
    await p.camera("living", [p.person("alex", track="t1")])
    await p.settle()
    assert p.scenes()[-1].data["trigger"] == "person_left"
    assert not [e for e in p.proposals(kind="scene") if e.source == "agent.scene_decider"]
