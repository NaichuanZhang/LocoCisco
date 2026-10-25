"""Policy rules in order, confidence bands, conflicts; actuator asks, replies and expiry."""
from __future__ import annotations

import pytest

from homedemo.contracts import T

UNLOCK = [("Unlock", "intent", ("Unlock the door", 0.95))]
TV_ON = [("Turn on the TV", "intent", ("TV on", 0.95))]


def decided(p):
    return [e for e in p.proposals(kind="speech") if e.data["by"] != "keyword"]


def verdict_for(p, proposal):
    return p.verdicts(proposal_id=proposal.id)[0].data


async def test_tv_voice_is_denied(make_pipeline):
    p = make_pipeline([("Turn off the lights", "intent", ("Lights off or dimmer", 0.95))])
    await p.device("tv_living", {"on": True, "volume": 30, "playing": "movie"})
    await p.device("light_living", {"on": True, "brightness": 100})
    await p.say("Turn off the lights.", who=None, room="living", source="tv")
    await p.settle()
    v = verdict_for(p, decided(p)[0])
    assert v["verdict"] == "deny" and v["rule"] == "only voices inside the house count; this came from the TV"
    assert not p.commands()
    await p.say("Unlock the front door.", who=None, room="porch", source="outside")
    await p.settle()
    v = verdict_for(p, decided(p)[1])
    assert v["verdict"] == "deny" and "came from outside" in v["rule"]
    assert not p.commands()


async def test_child_unlock_is_denied(make_pipeline):
    p = make_pipeline(UNLOCK)
    await p.say("Unlock the front door.", who="maya", room="living", match=0.93)
    await p.settle()
    v = verdict_for(p, decided(p)[0])
    assert v["verdict"] == "deny" and v["rule"] == "kids may not unlock doors"
    assert v["who"]["id"] == "maya" and v["who"]["role"] == "child"
    assert not p.commands()


@pytest.mark.parametrize("match, expected", [(0.88, "deny"), (0.895, "deny"), (0.90, "allow"), (0.95, "allow")])
async def test_parent_unlock_needs_a_strong_voice_match(make_pipeline, match, expected):
    p = make_pipeline(UNLOCK)
    await p.say("Unlock the front door.", who="alex", room="hallway", match=match)
    await p.settle()
    v = verdict_for(p, decided(p)[0])
    assert v["verdict"] == expected
    cmds = p.commands(device_id="lock_front")
    if expected == "allow":
        assert len(cmds) == 1 and cmds[0].data["command"] == "unlock" and v["rule"].startswith("confident enough")
    else:
        assert not cmds and v["rule"] == "unlocking needs a parent's voice with a strong match (0.90 or more)"


async def test_guest_or_unknown_voice_cannot_unlock(make_pipeline):
    p = make_pipeline(UNLOCK)
    await p.say("Unlock the front door.", who=None, room="hallway")
    await p.settle()
    assert verdict_for(p, decided(p)[0])["verdict"] == "deny"
    assert not p.commands()


async def test_curfew_asks_a_parent_and_yes_executes(make_pipeline):
    p = make_pipeline(TV_ON)
    await p.clock("21:15", "Tuesday")
    await p.say("Turn on the TV.", who="maya", room="living", match=0.91)
    await p.settle()
    proposal = decided(p)[0]
    v = verdict_for(p, proposal)
    assert v["verdict"] == "ask_parent"
    assert v["rule"] == "kids may not turn on the TV after 20:00 on a school night without a parent's OK"
    asks = p.asks()
    assert len(asks) == 1
    ask = asks[0].data
    assert ask["to"] == "parents" and ask["text"] == "Maya wants to turn the TV on. OK?"
    assert ask["options"] == ["Yes", "No"] and ask["proposal_id"] == proposal.id
    assert ask["action"] == "tv_on" and ask["target"] == "tv_living" and ask["room"] == "living"
    assert asks[0].source == "agent.actuator" and asks[0].cause == p.verdicts()[0].id
    assert not p.commands()
    reply = await p.reply(ask["request_id"], "Yes", by="alex")
    await p.settle()
    cmds = p.commands(device_id="tv_living")
    assert len(cmds) == 1 and cmds[0].data["command"] == "on"
    assert cmds[0].data["verdict_id"] == p.verdicts()[0].id and cmds[0].cause == reply.id


async def test_no_curfew_on_a_weekend_or_before_eight(make_pipeline):
    p = make_pipeline(TV_ON)
    await p.clock("21:15", "Saturday")
    await p.say("Turn on the TV.", who="maya", room="living", match=0.91)
    await p.settle()
    assert verdict_for(p, decided(p)[0])["verdict"] == "allow" and len(p.commands(device_id="tv_living")) == 1
    await p.clock("19:30", "Tuesday")
    await p.say("Turn on the TV.", who="maya", room="living", match=0.91)
    await p.settle()
    assert verdict_for(p, decided(p)[1])["verdict"] == "allow"


async def test_mid_confidence_asks_the_speaker(make_pipeline):
    p = make_pipeline([("warmer", "intent", ("Warmer", 0.6))])
    await p.say("Could you make it a bit warmer?", who="sam", room="living")
    await p.settle()
    v = verdict_for(p, decided(p)[0])
    assert v["verdict"] == "ask_speaker" and "unsure" in v["rule"] and "60%" in v["rule"]
    ask = p.asks()[0].data
    assert ask["to"] == "sam" and ask["text"] == "Make it warmer?" and ask["target"] == "thermostat_main"
    assert not p.commands()
    await p.reply(ask["request_id"], "Yes", by="sam")
    await p.settle()
    cmd = p.commands(device_id="thermostat_main")[0].data
    assert cmd["command"] == "set_setpoint" and cmd["args"] == {"delta_c": 1.5}


async def test_mid_confidence_unknown_speaker_asks_the_parents(make_pipeline):
    p = make_pipeline([("cooler", "intent", ("Cooler", 0.6))])
    await p.say("Could you make it cooler?", who=None, room="living")
    await p.settle()
    assert verdict_for(p, decided(p)[0])["verdict"] == "ask_speaker"
    assert p.asks()[0].data["to"] == "parents" and p.asks()[0].data["text"] == "Make it cooler?"


async def test_low_confidence_is_ignored(make_pipeline):
    p = make_pipeline([("heater", "intent", ("Warmer", 0.3))])
    await p.say("My sister said her heater broke last night.", who="alex", room="kitchen")
    await p.settle()
    v = verdict_for(p, decided(p)[0])
    assert v["verdict"] == "ignore" and v["rule"].startswith("too unsure to act (30%")
    assert not p.commands() and not p.asks()


async def test_bands_follow_live_settings(make_pipeline):
    p = make_pipeline([("warmer", "intent", ("Warmer", 0.6))])
    p.settings.act_threshold = 0.5
    await p.say("Make it warmer.", who="sam", room="living")
    await p.settle()
    assert verdict_for(p, decided(p)[0])["verdict"] == "allow"
    p.settings.act_threshold, p.settings.ask_threshold = 0.9, 0.7
    await p.say("Make it warmer please.", who="sam", room="living", ts=100)
    await p.settle()
    assert verdict_for(p, decided(p)[1])["verdict"] == "ignore"


async def test_conflicting_requests_ignore_the_later_one(make_pipeline):
    p = make_pipeline([("turn on the lights", "intent", ("Lights on", 0.95)),
                       ("turn off the lights", "intent", ("Lights off or dimmer", 0.95))])
    await p.say("Can you turn on the lights?", who="alex", room="living", ts=100)
    await p.settle()
    await p.say("No, turn off the lights.", who="sam", room="living", ts=104)
    await p.settle()
    first, second = decided(p)
    assert verdict_for(p, first)["verdict"] == "allow"
    v = verdict_for(p, second)
    assert v["verdict"] == "ignore" and v["rule"] == "conflicting requests: Alex just asked for the opposite"
    assert [c.data["command"] for c in p.commands(device_id="light_living")] == ["on"]
    # 10 s later the same request is fine
    await p.say("Please turn off the lights.", who="sam", room="living", ts=115)
    await p.settle()
    assert verdict_for(p, decided(p)[2])["verdict"] == "allow"
    assert [c.data["command"] for c in p.commands(device_id="light_living")] == ["on", "off"]


async def test_same_resident_changing_their_mind_is_not_a_conflict(make_pipeline):
    p = make_pipeline([("turn on the lights", "intent", ("Lights on", 0.95)),
                       ("turn off the lights", "intent", ("Lights off or dimmer", 0.95))])
    await p.say("Can you turn on the lights?", who="alex", room="living", ts=100)
    await p.settle()
    await p.say("Actually, turn off the lights.", who="alex", room="living", ts=102)
    await p.settle()
    assert [v.data["verdict"] for v in p.verdicts()] == ["allow", "allow"]


async def test_weak_voice_match_counts_as_an_unknown_speaker(make_pipeline):
    p = make_pipeline(TV_ON)
    await p.clock("21:15", "Tuesday")
    await p.say("Turn on the TV.", who="maya", room="living", match=0.5)   # below voice_match 0.85
    await p.settle()
    v = verdict_for(p, decided(p)[0])
    assert v["who"]["id"] is None and v["who"]["role"] == "unknown"
    assert v["verdict"] == "allow"            # nobody identified, so no child rule; confidence 0.95 acts


async def test_reply_no_logs_and_does_nothing(make_pipeline):
    p = make_pipeline([("warmer", "intent", ("Warmer", 0.6))])
    await p.say("Could you make it a bit warmer?", who="sam", room="living")
    await p.settle()
    ask = p.asks()[0].data
    await p.reply(ask["request_id"], "No", by="sam")
    await p.settle()
    assert not p.commands()
    logs = p.logs(agent="actuator")
    assert logs and "said No" in logs[-1].data["text"]
    await p.reply("ask-nope", "Yes", by="sam")
    await p.settle()
    assert not p.commands() and "unknown or expired" in p.logs(agent="actuator")[-1].data["text"]


async def test_ask_expires(make_pipeline):
    p = make_pipeline([("warmer", "intent", ("Warmer", 0.6))])
    p.agent("actuator").ask_ttl_s = 0.0
    await p.say("Could you make it a bit warmer?", who="sam", room="living")
    await p.settle()
    ask = p.asks()[0].data
    await p.reply(ask["request_id"], "Yes", by="sam")
    await p.settle()
    assert not p.commands()
    assert "expired" in p.logs(agent="actuator")[-1].data["text"]
    assert not p.agent("actuator").pending


async def test_command_shapes_for_every_action(make_pipeline):
    actuator = make_pipeline().agent("actuator")
    assert actuator.commands("lights_dim", None, "living") == [{"device_id": "light_living", "command": "dim", "args": {"brightness": 30}}]
    assert actuator.commands("lights_off", "light_kitchen", "living") == [{"device_id": "light_kitchen", "command": "off", "args": {}}]
    assert actuator.commands("tv_off", None, "kitchen") == [{"device_id": "tv_living", "command": "off", "args": {}}]
    assert actuator.commands("cooler", None, None) == [{"device_id": "thermostat_main", "command": "set_setpoint", "args": {"delta_c": -1.5}}]
    assert actuator.commands("lock_door", None, "porch") == [{"device_id": "lock_front", "command": "lock", "args": {}}]
    assert actuator.commands("none", None, "living") == []
    alert = actuator.commands("alert_parent", None, "kitchen")[0]
    assert alert["device_id"] == "phone_parents" and alert["command"] == "notify" and alert["args"]["to"] == "parents"
    assert actuator.ask_text("lights_on", "living", "ask_speaker", None) == "Turn on the living room lights?"
    assert actuator.ask_text("lights_off", "bedroom_maya", "ask_speaker", None) == "Turn off the Maya's room lights?"
    assert actuator.ask_text("unlock_door", "hallway", "ask_speaker", {"name": "Alex"}) == "Unlock the front door?"


async def test_shadow_proposals_get_no_verdict_and_events_chain(make_pipeline):
    p = make_pipeline([("turn on the lights", "intent", ("Lights on", 0.95))])
    await p.say("Can you turn on the lights?", who="alex", room="living")
    await p.settle()
    utt, kw = p.events(T.UTTERANCE)[0], p.proposals(by="keyword")[0]
    assert not p.verdicts(proposal_id=kw.id)
    verdict = p.verdicts()[0]
    assert verdict.source == "agent.policy" and verdict.cause == decided(p)[0].id and verdict.ts == utt.ts
    cmd = p.commands()[0]
    assert cmd.source == "agent.actuator" and cmd.cause == verdict.id and cmd.ts == utt.ts
    assert all(e.source.startswith("agent.") for e in p.bus.history if e.topic in (T.UTTERANCE, T.PROPOSAL, T.VERDICT, T.COMMAND))
