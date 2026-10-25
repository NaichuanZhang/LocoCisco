"""Cascade with escalate_to "bedrock": the 2B decides inside the Strands house agent's router; Claude
(a scripted Strands model here) only when the 2B is unsure; fallbacks when Bedrock or the 2B fail."""
from __future__ import annotations

from homedemo.agents.house_agent import HouseAgent
from homedemo.contracts import ACTIONS, Settings, T
from homedemo.decider_client import FakeDecider
from homedemo.patterns.fakes import ScriptedModel

from .conftest import BrokenDecider, Pipeline

LIGHTS_SURE = [("turn on the lights", "intent", ("Lights on", 0.95))]
LIGHTS_UNSURE = [("can't see", "intent", ("Lights on", 0.55))]
FIELDS = {"route", "model", "request", "answers", "threshold", "tokens", "tool_calls", "reply"}


def house(model: ScriptedModel, settings: Settings) -> HouseAgent:
    return HouseAgent(settings, model=model)


def bedrock_pipeline(script, model, *, twelve_b=None, **changes) -> tuple[Pipeline, FakeDecider]:
    settings = Settings(**{"decider_mode": "cascade", "escalate_to": "bedrock", **changes})
    small = FakeDecider("decider-2b", script=script)
    deciders = {"2b": small, "bedrock": house(model, settings)}
    if twelve_b is not None:
        deciders["12b"] = twelve_b
    return Pipeline(settings, deciders), small


def decider_props(p: Pipeline, kind: str = "speech"):
    return [e for e in p.proposals(kind=kind) if e.data["by"] not in ("keyword", "reflex")]


async def test_sure_2b_acts_locally_and_the_model_is_never_called():
    model = ScriptedModel([("lights_off", {"room": "living"})])      # would be wrong; must never run
    p, small = bedrock_pipeline(LIGHTS_SURE, model)
    await p.clock("20:00")
    await p.say("Can you turn on the lights?", who="alex", room="living")
    await p.settle()
    props = decider_props(p)
    assert len(props) == 1 and model.calls == 0 and len(small.calls) == 1
    d = props[0].data
    assert d["by"] == "decider-2b" and d["mode"] == "act" and d["route"] == "local" and d["action"] == "lights_on"
    assert d["model"] == "Strands Decider 2B" and d["tokens"] is None and d["tool_calls"] == [] and d["reply"] is None
    assert d["threshold"] == 0.8 and d["confidence"] == 0.95 and d["escalated_from"] is None
    assert d["request"] == {"state": d["state_text"], "questions": d["request"]["questions"]}
    assert d["request"]["questions"]["intent"]["type"] == "choice"
    assert d["answers"]["intent"]["choice"] == "Lights on" and d["answers"]["intent"]["probabilities"]["Lights on"] == 0.95
    assert FIELDS <= set(d) and d["reason"] == "2B is 95% sure: turn the lights on"
    assert p.commands(device_id="light_living")[0].data["command"] == "on"
    assert not p.bus.errors


async def test_unsure_2b_escalates_to_claude_whose_tool_call_acts():
    model = ScriptedModel([("lights_on", {"room": "living"})], reply="Lights coming on.", usage=(900, 40))
    p, small = bedrock_pipeline(LIGHTS_UNSURE, model)
    await p.clock("20:00")
    await p.say("I can't see a thing in here.", who="alex", room="living")
    await p.settle()
    props = decider_props(p)
    assert [(e.data["by"], e.data["mode"], e.data["route"]) for e in props] == [
        ("decider-2b", "shadow", "local"), ("bedrock", "act", "bedrock")]
    shadow, act = props
    assert shadow.data["reason"] == "unsure, asking Claude on Bedrock" and shadow.data["confidence"] == 0.55
    assert shadow.data["tokens"] is None and shadow.data["escalated_from"] is None
    a = act.data
    assert a["escalated_from"] == shadow.id and a["action"] == "lights_on" and a["confidence"] == 0.9
    assert a["model"] == "Claude Haiku 4.5 on Bedrock" and a["target"] == "light_living" and a["room"] == "living"
    assert a["tokens"] == {"input": 1800, "output": 80}            # two model calls: tool use, then the reply
    assert a["tool_calls"] == [{"name": "lights_on", "input": {"room": "living"}}] and a["reply"] == "Lights coming on."
    assert a["reason"] == "Claude called lights_on: turn the lights on" and a["latency_ms"] >= 0
    assert a["answers"] == shadow.data["answers"] and a["request"] == shadow.data["request"]
    # the pipeline: one utterance event caused both, Claude's call was executed, the 2B asked once
    utt = p.events(T.UTTERANCE)[0]
    assert shadow.cause == act.cause == utt.id and len(small.calls) == 1 and model.calls == 2
    assert not p.verdicts(proposal_id=shadow.id)
    assert p.verdicts(proposal_id=act.id)[0].data["verdict"] == "allow"
    assert len(p.commands(device_id="light_living")) == 1
    # Claude read the overheard sentence with its context, not the decider's question
    prompt = model.requests[0]["messages"][0]["content"][0]["text"]
    assert prompt.startswith("Overheard.") and "Alex (parent)" in prompt and "I can't see a thing in here." in prompt
    assert "lights_on" in model.requests[0]["tools"] and "ask_resident" in model.requests[0]["tools"]
    assert not p.bus.errors


async def test_claude_ask_resident_becomes_a_phone_ask():
    model = ScriptedModel([("ask_resident", {"action": "warmer"})])
    p, _ = bedrock_pipeline([("freezing", "intent", ("Warmer", 0.6))], model)
    await p.say("It's freezing in here.", who="alex", room="living")
    await p.settle()
    act = decider_props(p)[-1].data
    assert act["by"] == "bedrock" and act["action"] == "warmer" and act["confidence"] == 0.6
    assert act["target"] == "thermostat_main" and "asks first" in act["reason"]
    assert p.verdicts(verdict="ask_speaker") and p.asks()[0].data["to"] == "alex"
    assert not p.commands()


async def test_claude_calling_nothing_is_a_none_proposal():
    model = ScriptedModel([], reply="")
    p, _ = bedrock_pipeline([("sister", "intent", ("Not a request", 0.9))], model)   # a 2B "none" escalates
    await p.say("My sister said her heater broke last night.", who="alex", room="kitchen")
    await p.settle()
    props = decider_props(p)
    assert [(e.data["by"], e.data["mode"], e.data["action"]) for e in props] == [
        ("decider-2b", "shadow", "none"), ("bedrock", "act", "none")]
    act = props[1].data
    assert act["tool_calls"] == [] and act["reply"] is None and act["reason"] == "Claude called no tool: nothing to do"
    assert p.verdicts(proposal_id=props[1].id)[0].data["verdict"] == "ignore" and not p.commands()


async def test_claude_no_action_tool_is_none_too():
    model = ScriptedModel([("no_action", {})], reply="Nothing to do.")
    p, _ = bedrock_pipeline([("dollhouse", "intent", ("Not a request", 0.7))], model)
    await p.say("Alexa ordered me a dollhouse.", who="alex", room="living")
    await p.settle()
    act = decider_props(p)[-1].data
    assert act["action"] == "none" and act["tool_calls"] == [{"name": "no_action", "input": {}}]
    assert act["reply"] == "Nothing to do." and not p.commands()


async def test_claude_two_tool_calls_give_two_act_proposals():
    model = ScriptedModel([("tv_off", {}), ("lights_off", {"room": "living"})])
    p, _ = bedrock_pipeline([("asleep", "action", ("Do nothing", 0.6))], model)
    await p.clock("01:30")
    await p.device("tv_living", {"on": True, "volume": 20})
    await p.device("light_living", {"on": True, "brightness": 60})
    await p.camera("living", [p.person("alex", pose="lying on the couch")])
    await p.camera("living", [p.person("alex", pose="lying on the couch")], activity="asleep on the couch, eyes closed")
    await p.settle()
    props = decider_props(p, kind="scene")
    acts = [e.data for e in props if e.data["by"] == "bedrock"]
    assert [(a["action"], a["target"]) for a in acts] == [("tv_off", "tv_living"), ("lights_off", "light_living")]
    assert all(a["escalated_from"] == props[0].id and a["tokens"] for a in acts)
    assert p.commands(device_id="tv_living") and p.commands(device_id="light_living")


async def test_scene_goes_through_the_same_router():
    model = ScriptedModel([("warmer", {})])
    p, small = bedrock_pipeline([("sweater", "action", ("Make warmer", 0.5))], model)
    await p.temp("living", 16.0)
    await p.camera("living", [p.person("alex", pose="sitting on the couch")])
    await p.camera("living", [p.person("alex", pose="sitting on the couch")], activity="putting on a sweater")
    await p.settle()
    props = decider_props(p, kind="scene")
    assert [(e.data["by"], e.data["mode"]) for e in props] == [("decider-2b", "shadow"), ("bedrock", "act")]
    assert props[1].data["action"] == "warmer" and props[1].data["speaker"] is None
    assert model.requests[0]["messages"][0]["content"][0]["text"].startswith("Camera scene.")
    assert len(small.calls) == 1


async def test_tv_voice_dismissed_by_the_2b_is_not_escalated():
    model = ScriptedModel([("lights_off", {"room": "living"})])
    p, _ = bedrock_pipeline([("lights", "intent", ("Not a request", 0.92))], model)
    await p.say("Turn off the lights.", who=None, room="living", source="tv")
    await p.settle()
    props = decider_props(p)
    assert len(props) == 1 and props[0].data["route"] == "local" and props[0].data["action"] == "none"
    assert model.calls == 0


async def test_bedrock_failure_falls_back_to_a_healthy_12b():
    model = ScriptedModel(fail=RuntimeError("ExpiredTokenException: The security token included in the request is expired"))
    big = FakeDecider("decider-12b", script=[("can't see", "intent", ("Lights on", 0.96))])
    p, _ = bedrock_pipeline(LIGHTS_UNSURE, model, twelve_b=big)
    await p.say("I can't see a thing in here.", who="alex", room="living")
    await p.settle()
    props = decider_props(p)
    assert [(e.data["by"], e.data["mode"], e.data["route"]) for e in props] == [
        ("decider-2b", "shadow", "local"), ("decider-12b", "act", "12b")]
    assert props[1].data["escalated_from"] == props[0].id and props[1].data["confidence"] == 0.96
    assert props[1].data["reason"].startswith("Bedrock failed, asked the 12B;")
    warns = [e.data["text"] for e in p.logs(agent="intent_decider", level="warn")]
    assert len(warns) == 1 and warns[0].startswith("Claude on Bedrock failed: RuntimeError: ExpiredTokenException")
    assert p.commands(device_id="light_living") and not p.bus.errors


async def test_bedrock_failure_without_a_12b_uses_the_unsure_2b_answer():
    model = ScriptedModel(fail=ConnectionError("endpoint unreachable"))
    p, _ = bedrock_pipeline([("can't see", "intent", ("Lights on", 0.6))], model, twelve_b=BrokenDecider("decider-12b"))
    await p.say("I can't see a thing in here.", who="alex", room="living")
    await p.settle()
    props = decider_props(p)
    assert [(e.data["by"], e.data["mode"]) for e in props] == [("decider-2b", "shadow"), ("decider-2b", "act")]
    assert props[1].data["escalated_from"] == props[0].id
    assert props[1].data["reason"] == "Claude on Bedrock unavailable, using the unsure 2B answer"
    assert p.verdicts(proposal_id=props[1].id)[0].data["verdict"] == "ask_speaker"   # 0.6 is in the ask band
    assert p.asks()[0].data["to"] == "alex"


async def test_2b_failure_proceeds_to_claude_without_a_shadow():
    model = ScriptedModel([("lights_on", {"room": "living"})])
    settings = Settings(decider_mode="cascade", escalate_to="bedrock")
    p = Pipeline(settings, {"2b": BrokenDecider("decider-2b"), "bedrock": house(model, settings)})
    await p.say("Can you turn on the lights?", who="alex", room="living")
    await p.settle()
    props = decider_props(p)
    assert len(props) == 1 and props[0].data["by"] == "bedrock" and props[0].data["escalated_from"] is None
    assert props[0].data["answers"] == {} and model.calls == 2
    warns = [e.data["text"] for e in p.logs(level="warn")]
    assert warns == ["2B decider failed: ConnectionError: connection refused"]
    assert p.commands(device_id="light_living") and not p.bus.errors


async def test_everything_down_means_no_proposal_and_two_warnings():
    model = ScriptedModel(fail=ConnectionError("endpoint unreachable"))
    settings = Settings(decider_mode="cascade", escalate_to="bedrock")
    p = Pipeline(settings, {"2b": BrokenDecider("decider-2b"), "12b": BrokenDecider("decider-12b"),
                            "bedrock": house(model, settings)})
    await p.say("Can you turn on the lights?", who="alex", room="living")
    await p.settle()
    assert not decider_props(p) and not p.commands()
    assert [e.data["text"].split(":")[0] for e in p.logs(level="warn")] == ["2B decider failed", "Claude on Bedrock failed"]


async def test_escalate_to_12b_keeps_the_old_cascade_even_with_a_house_agent():
    model = ScriptedModel([("lights_on", {"room": "living"})])
    big = FakeDecider("decider-12b", script=[("can't see", "intent", ("Lights on", 0.96))])
    p, _ = bedrock_pipeline(LIGHTS_UNSURE, model, twelve_b=big, escalate_to="12b")
    await p.say("I can't see a thing in here.", who="alex", room="living")
    await p.settle()
    props = decider_props(p)
    assert [e.data["by"] for e in props] == ["decider-2b", "decider-12b"] and model.calls == 0
    assert props[0].data["reason"] == "unsure, asking the 12B" and props[1].data["route"] == "12b"
    assert props[1].data["model"] == "Strands Decider 12B" and FIELDS <= set(props[1].data)


async def test_threshold_is_read_live_by_the_router():
    model = ScriptedModel([("lights_on", {"room": "living"})])
    p, _ = bedrock_pipeline([("can't see", "intent", ("Lights on", 0.7))], model)
    p.settings.act_threshold = 0.6                        # the UI lowered the bar: 0.7 is now confident
    await p.say("I can't see a thing in here.", who="alex", room="living")
    await p.settle()
    props = decider_props(p)
    assert len(props) == 1 and props[0].data["route"] == "local" and props[0].data["threshold"] == 0.6
    assert model.calls == 0 and p.commands(device_id="light_living")


async def test_rule_proposals_carry_the_route_fields():
    p, _ = bedrock_pipeline([("stove", "action", ("Alert a parent", 0.97))], ScriptedModel())
    await p.clock("17:20")
    await p.device("light_kitchen", {"on": True, "brightness": 100})
    await p.device("stove_kitchen", {"burner_on": True})
    await p.camera("kitchen", [p.person("maya", pose="standing on a chair", near="stove")])
    await p.say("Turn on the lights.", who="alex", room="living")
    await p.settle()
    reflex = [e.data for e in p.proposals(by="reflex")]
    keyword = [e.data for e in p.proposals(by="keyword")]
    assert reflex and all(d["route"] == "rule" and d["tokens"] is None and d["model"] for d in reflex)
    assert keyword and keyword[0]["route"] == "rule" and keyword[0]["model"] == "keyword rules"
    assert all(d["action"] in ACTIONS for d in reflex + keyword)
