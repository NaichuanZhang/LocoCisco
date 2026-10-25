"""Scene and intent deciders: full ProposalData, the 2B -> 12B cascade (settings.escalate_to="12b"), fallbacks
when a model is down. The default 2B -> Claude on Bedrock cascade is covered in test_bedrock_cascade.py."""
from __future__ import annotations

import asyncio

from homedemo.contracts import ACTIONS, T
from homedemo.decider_client import FakeDecider

from .conftest import BrokenDecider

HAZARD = [("stove", "action", ("Alert a parent", 0.97))]
LIGHTS = [("turn on the lights", "intent", ("Lights on", 0.95))]


async def hazard_frame(p):
    await p.clock("17:20")
    await p.device("light_kitchen", {"on": True, "brightness": 100})
    await p.device("stove_kitchen", {"burner_on": True})
    await p.camera("kitchen", [p.person("maya", pose="standing on a chair", near="stove")])
    await p.settle()


async def test_hazard_scene_alerts_a_parent(make_pipeline):
    p = make_pipeline(HAZARD)
    await hazard_frame(p)
    props = [e for e in p.proposals(kind="scene") if e.source == "agent.scene_decider"]
    assert len(props) == 1
    d, scene = props[0].data, p.scenes(trigger="hazard")[0]
    assert props[0].source == "agent.scene_decider" and props[0].cause == scene.id and props[0].ts == scene.ts
    assert d["action"] == "alert_parent" and d["mode"] == "act" and d["target"] == "phone_parents"
    assert d["confidence"] == 0.97 and abs(sum(d["probabilities"].values()) - 1) < 1e-3   # 4-decimal rounding
    assert set(d["probabilities"]) <= set(ACTIONS) and d["probabilities"]["alert_parent"] == 0.97
    assert d["state_text"] == scene.data["text"] and d["question"].startswith("The home hub sees this")
    assert d["latency_ms"] > 0 and d["speaker"] is None and d["escalated_from"] is None and d["reason"]
    assert d["by"] == "fake"                                    # FakeDecider is labelled honestly
    verdict = p.verdicts(proposal_id=props[0].id)[0].data
    assert verdict["verdict"] == "allow" and verdict["rule"] == "safety alerts to a parent are always allowed"
    cmd = p.commands(device_id="phone_parents")[0].data
    assert cmd["command"] == "notify" and cmd["args"]["to"] == "parents" and "kitchen" in cmd["args"]["text"]


async def test_hazard_rule_alerts_without_the_model(make_pipeline):
    p = make_pipeline([("stove", "action", ("Do nothing", 0.99))])     # even a confidently wrong model
    await hazard_frame(p)
    reflex = [e for e in p.proposals(kind="scene") if e.source == "agent.reflex"]
    assert len(reflex) == 1 and reflex[0].data["action"] == "alert_parent" and reflex[0].data["confidence"] == 1.0
    assert len(p.commands(device_id="phone_parents")) == 1               # one alert, the cooldown holds


async def test_speech_turns_on_the_lights(make_pipeline):
    p = make_pipeline(LIGHTS)
    await p.clock("20:00")
    await p.say("Can you turn on the lights?", who="alex", room="living")
    await p.settle()
    utt = p.events(T.UTTERANCE)[0]
    props = [e for e in p.proposals(kind="speech") if e.data["by"] != "keyword"]
    assert len(props) == 1
    d = props[0].data
    assert props[0].source == "agent.intent_decider" and props[0].cause == utt.id
    assert d["action"] == "lights_on" and d["target"] == "light_living" and d["mode"] == "act"
    assert d["speaker"]["id"] == "alex" and d["state_text"].endswith('Heard: "Can you turn on the lights?"')
    assert d["question"].startswith("The home hub's microphone heard this")
    assert p.commands(device_id="light_living")[0].data["command"] == "on"


async def test_cascade_escalates_an_unsure_2b_to_the_12b(make_pipeline):
    small = FakeDecider("decider-2b", script=[("turn on the lights", "intent", ("Lights on", 0.55))])
    big = FakeDecider("decider-12b", script=[("turn on the lights", "intent", ("Lights on", 0.96))])
    p = make_pipeline(deciders={"2b": small, "12b": big}, decider_mode="cascade", escalate_to="12b")
    await p.clock("20:00")
    await p.say("Can you turn on the lights?", who="alex", room="living")
    await p.settle()
    props = [e for e in p.proposals(kind="speech") if e.data["by"] != "keyword"]
    assert [(e.data["by"], e.data["mode"]) for e in props] == [("decider-2b", "shadow"), ("decider-12b", "act")]
    shadow, act = props
    assert shadow.data["reason"] == "unsure, asking the 12B" and shadow.data["confidence"] == 0.55
    assert act.data["escalated_from"] == shadow.id and act.data["confidence"] == 0.96
    assert shadow.cause == act.cause == p.events(T.UTTERANCE)[0].id
    assert len(small.calls) == 1 and len(big.calls) == 1
    # the shadow proposal gets no verdict; the 12B one is executed
    assert not p.verdicts(proposal_id=shadow.id)
    assert p.verdicts(proposal_id=act.id)[0].data["verdict"] == "allow"
    assert len(p.commands(device_id="light_living")) == 1


async def test_cascade_confident_2b_acts_alone(make_pipeline):
    small = FakeDecider("decider-2b", script=LIGHTS)
    big = FakeDecider("decider-12b", script=LIGHTS)
    p = make_pipeline(deciders={"2b": small, "12b": big}, decider_mode="cascade", escalate_to="12b")
    await p.say("Can you turn on the lights?", who="alex", room="living")
    await p.settle()
    props = [e for e in p.proposals(kind="speech") if e.data["by"] != "keyword"]
    assert len(props) == 1 and props[0].data["by"] == "decider-2b" and props[0].data["mode"] == "act"
    assert len(big.calls) == 0


async def test_cascade_threshold_is_read_live_from_settings(make_pipeline):
    small = FakeDecider("decider-2b", script=[("turn on the lights", "intent", ("Lights on", 0.7))])
    big = FakeDecider("decider-12b", script=LIGHTS)
    p = make_pipeline(deciders={"2b": small, "12b": big}, decider_mode="cascade", escalate_to="12b")
    p.settings.act_threshold = 0.6                       # the UI lowered the bar: 0.7 is now confident
    await p.say("Can you turn on the lights?", who="alex", room="living")
    await p.settle()
    assert len(big.calls) == 0 and p.commands(device_id="light_living")


async def test_2b_failure_falls_back_to_the_12b(make_pipeline):
    broken, big = BrokenDecider("decider-2b"), FakeDecider("decider-12b", script=LIGHTS)
    p = make_pipeline(deciders={"2b": broken, "12b": big}, decider_mode="2b")
    await p.say("Can you turn on the lights?", who="alex", room="living")
    await p.settle()
    logs = p.logs(agent="intent_decider", level="warn")
    assert logs and "2B decider failed" in logs[0].data["text"]
    props = [e for e in p.proposals(kind="speech") if e.data["by"] != "keyword"]
    assert len(props) == 1 and props[0].data["by"] == "decider-12b" and props[0].data["mode"] == "act"
    assert "2B failed, asked the 12B" in props[0].data["reason"]
    assert p.commands(device_id="light_living")
    assert not p.bus.errors                               # the failure was handled, not raised


async def test_12b_failure_in_cascade_uses_the_unsure_2b_answer(make_pipeline):
    small = FakeDecider("decider-2b", script=[("turn on the lights", "intent", ("Lights on", 0.6))])
    p = make_pipeline(deciders={"2b": small, "12b": BrokenDecider("decider-12b")}, decider_mode="cascade", escalate_to="12b")
    await p.say("Can you turn on the lights?", who="alex", room="living")
    await p.settle()
    props = [e for e in p.proposals(kind="speech") if e.data["by"] != "keyword"]
    assert [(e.data["by"], e.data["mode"]) for e in props] == [("decider-2b", "shadow"), ("decider-2b", "act")]
    assert props[1].data["escalated_from"] == props[0].id and "12B unavailable" in props[1].data["reason"]
    assert p.logs(agent="intent_decider", level="warn")
    # 0.6 is in the ask band, so the house asks Alex instead of acting
    assert p.verdicts(proposal_id=props[1].id)[0].data["verdict"] == "ask_speaker"
    assert p.asks()[0].data["to"] == "alex"


async def test_every_decider_down_means_no_proposal(make_pipeline):
    p = make_pipeline(deciders={"2b": BrokenDecider("decider-2b"), "12b": BrokenDecider("decider-12b")},
                      decider_mode="cascade", escalate_to="12b")
    await p.say("Can you turn on the lights?", who="alex", room="living")
    await p.settle()
    assert not [e for e in p.proposals() if e.data["by"] != "keyword"]
    assert len(p.logs(level="warn")) == 2 and not p.commands()


async def test_busy_flag_while_awaiting_the_decider(make_pipeline):
    slow = FakeDecider(script=LIGHTS, latency_ms=60)
    p = make_pipeline(deciders={"fake": slow})
    intent = p.agent("intent_decider")
    await p.say("Can you turn on the lights?", who="alex", room="living")
    await asyncio.sleep(0.02)
    assert intent.busy is True
    await p.settle()
    assert intent.busy is False and all(a.busy is False for a in p.agents)


async def test_none_action_is_still_published_and_ignored(make_pipeline):
    p = make_pipeline([("sister", "intent", ("Not a request", 0.9))])
    await p.say("My sister said her heater broke last night.", who="alex", room="kitchen")
    await p.settle()
    props = [e for e in p.proposals(kind="speech") if e.data["by"] != "keyword"]
    assert props[0].data["action"] == "none" and props[0].data["target"] is None
    verdict = p.verdicts(proposal_id=props[0].id)[0].data
    assert verdict["verdict"] == "ignore" and verdict["rule"] == "nothing to do"
    assert not p.commands() and not p.asks()


async def test_interpretation_room_attribute_overrides_the_event_room(make_pipeline, monkeypatch):
    """If the questions workstream adds Interpretation.room, the agents target that room."""
    import homedemo.agents.intent_decider as mod

    original = mod.build_speech_query

    def patched(utt):
        q = original(utt)
        inner = q.interpret

        def interpret(answers):
            it = inner(answers)
            it.room = "bedroom_maya"
            return it
        q.interpret = interpret
        return q

    monkeypatch.setattr(mod, "build_speech_query", patched)
    p = make_pipeline(LIGHTS)
    await p.say("Can you turn on the lights in my room?", who="alex", room="living")
    await p.settle()
    props = [e for e in p.proposals(kind="speech") if e.data["by"] != "keyword"]
    assert props[0].data["room"] == "bedroom_maya" and props[0].data["target"] == "light_bedroom_maya"
    assert p.commands(device_id="light_bedroom_maya")


async def test_cascade_sends_a_confident_2b_none_from_a_person_to_the_12b(make_pipeline):
    """eval/REPORT.md s3: the 2B's 'none' is weak evidence, so the 12B gets a say."""
    small = FakeDecider("decider-2b", script=[("remote", "intent", ("Not a request", 0.94))])
    big = FakeDecider("decider-12b", script=[("remote", "intent", ("TV on", 0.9))])
    p = make_pipeline(deciders={"2b": small, "12b": big}, decider_mode="cascade", escalate_to="12b")
    await p.say("Where's the remote? The game starts in five minutes.", who="alex", room="living")
    await p.settle()
    props = [e for e in p.proposals(kind="speech") if e.data["by"] != "keyword"]
    assert [(e.data["by"], e.data["mode"], e.data["action"]) for e in props] == [
        ("decider-2b", "shadow", "none"), ("decider-12b", "act", "tv_on")]
    assert len(big.calls) == 1


async def test_cascade_does_not_escalate_a_tv_voice_the_2b_dismisses(make_pipeline):
    small = FakeDecider("decider-2b", script=[("lights", "intent", ("Not a request", 0.92))])
    big = FakeDecider("decider-12b")
    p = make_pipeline(deciders={"2b": small, "12b": big}, decider_mode="cascade", escalate_to="12b")
    await p.say("Turn off the lights.", who=None, room="living", source="tv")
    await p.settle()
    assert len(big.calls) == 0
