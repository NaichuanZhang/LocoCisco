"""Reflex: dark room at night + person -> lights on; pets and sunny rooms never; no double-proposals."""
from __future__ import annotations

from homedemo.contracts import T


async def test_dark_room_reflex_turns_on_the_light(make_pipeline):
    p = make_pipeline()
    await p.clock("23:40")
    await p.lux("kitchen", 2)
    await p.camera("kitchen", [p.person("alex")])
    await p.settle()
    props = p.proposals(by="reflex")
    assert len(props) == 1
    d = props[0].data
    assert d["mode"] == "act" and d["action"] == "lights_on" and d["target"] == "light_kitchen"
    assert d["confidence"] == 1.0 and d["kind"] == "scene" and d["question"] == "" and d["speaker"] is None
    assert d["latency_ms"] < 50
    assert props[0].source == "agent.reflex" and props[0].cause == p.scenes()[0].id
    verdicts = p.verdicts(proposal_id=props[0].id)
    assert verdicts and verdicts[0].data["verdict"] == "allow"
    cmds = p.commands(device_id="light_kitchen")
    assert len(cmds) == 1 and cmds[0].data["command"] == "on" and cmds[0].data["verdict_id"] == verdicts[0].id
    # only the reflex proposed: the scene decider leaves person_entered alone
    assert {e.data["by"] for e in p.proposals()} == {"reflex"}


async def test_pet_in_a_dark_room_does_not_trigger_the_reflex(make_pipeline):
    p = make_pipeline([("cat", "action", ("Do nothing", 0.9))])
    await p.clock("02:15")
    await p.lux("kitchen", 1)
    await p.camera("kitchen", [p.pet("cat")])
    await p.settle()
    assert p.scenes(room="kitchen")
    assert not p.proposals(by="reflex")
    assert not p.commands()


async def test_sunny_room_does_not_trigger_the_reflex(make_pipeline):
    p = make_pipeline()
    await p.clock("14:10")
    await p.lux("kitchen", 900)
    await p.camera("kitchen", [p.person("alex")])
    await p.settle()
    scene = p.scenes(room="kitchen")[0]
    assert scene.data["trigger"] == "person_entered" and "900 lux (bright sunlight)" in scene.data["text"]
    assert not p.proposals()
    assert not p.commands()


async def test_dark_room_in_daytime_or_with_lights_on_does_not_trigger(make_pipeline):
    p = make_pipeline()
    await p.clock("14:10")
    await p.lux("kitchen", 2)                      # dark but not night (a cellar-dark kitchen at noon)
    await p.camera("kitchen", [p.person("alex")])
    await p.settle()
    assert not p.proposals(by="reflex")
    await p.clock("23:00")
    await p.lux("living", 2)
    await p.device("light_living", {"on": True, "brightness": 100})
    await p.camera("living", [p.person("sam", track="t2")])
    await p.settle()
    assert not p.proposals(by="reflex")


async def test_presence_only_bedroom_reflex(make_pipeline):
    p = make_pipeline()
    await p.clock("23:00")
    await p.lux("bedroom_maya", 1)
    await p.motion("bedroom_maya", True)
    await p.settle()
    cmds = p.commands(device_id="light_bedroom_maya")
    assert len(cmds) == 1 and cmds[0].data["command"] == "on"
    assert p.proposals(by="reflex")[0].data["room"] == "bedroom_maya"


async def test_env_changed_never_proposes(make_pipeline):
    """Lights switched off on purpose must not be switched back on by the reflex."""
    p = make_pipeline()
    await p.clock("23:00")
    await p.lux("living", 2)
    await p.device("light_living", {"on": True, "brightness": 100})
    await p.camera("living", [p.person("alex")])
    await p.device("light_living", {"on": False, "brightness": 0})
    await p.lux("living", 1)
    await p.settle()
    assert p.scenes(trigger="env_changed")
    assert not p.proposals() and not p.events(T.COMMAND)
