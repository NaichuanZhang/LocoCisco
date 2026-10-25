"""The home demo takes its deciders from the models registry: slots 2b/12b/llm per profile, the "llm"
escalation through the Strands house agent, registry labels in the X-ray, --profile on the CLIs."""
from __future__ import annotations

import argparse

import pytest

from homedemo.agents.house_agent import HouseAgent
from homedemo.cli import add_decider_args, settings_from_args
from homedemo.contracts import Settings, T
from homedemo.decider_client import FakeDecider, HttpDecider
from homedemo.runtime import BLURBS, DECIDER_SETTINGS, Runtime, build_deciders, registry_overrides
from unspoken.core.fakes import ScriptedModel
from unspoken.models import load_registry

from tests.agents.conftest import Pipeline

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_dev_profile_slots_come_from_the_registry_and_stay_lazy():
    deciders = build_deciders(Settings(bedrock_profile="no-such-profile-for-this-test"))
    assert {"2b", "12b", "fake", "bedrock", "llm"} <= set(deciders)
    assert isinstance(deciders["2b"], HttpDecider) and deciders["2b"].base_url == "http://127.0.0.1:8765"
    assert deciders["2b"].name == "decider-2b" and deciders["2b"].label == "Strands Decider 2B (CPU)"
    assert deciders["12b"].base_url == "http://127.0.0.1:8766" and deciders["12b"].label == "Strands Decider 12B (CPU)"
    llm = deciders["llm"]
    assert isinstance(llm, HouseAgent) and llm.name == "llm" and llm._model is None   # Bedrock untouched
    assert llm.label == "Claude Haiku 4.5 on Bedrock" and llm.route == "bedrock" and llm.short == "Claude"
    assert deciders["bedrock"].name == "bedrock" and deciders["bedrock"].route == "bedrock"


def test_fake_profile_gives_fake_deciders_and_a_scripted_llm():
    deciders = build_deciders(Settings(profile="fake"))
    assert isinstance(deciders["2b"], FakeDecider) and deciders["2b"].name == "decider-2b"
    llm = deciders["llm"]
    assert llm.route == "local" and llm.label == "scripted model" and llm.short == "scripted"
    assert isinstance(llm.model(), ScriptedModel) and llm.model() is llm.model()


def test_disabled_roles_leave_slots_out(tmp_path, monkeypatch):
    p = tmp_path / "m.toml"
    p.write_text("[profiles.p]\n[profiles.p.decider]\nprovider='fake'\n")
    monkeypatch.setenv("UNSPOKEN_MODELS", str(p))
    deciders = build_deciders(Settings(profile="p"))
    assert set(deciders) == {"2b", "fake", "bedrock"}


def test_settings_urls_override_the_profile_only_when_changed():
    assert registry_overrides(Settings()) == {}
    s = Settings(decider_2b_url="http://127.0.0.1:9765", decider_12b_url="http://127.0.0.1:9766")
    assert registry_overrides(s) == {"decider.url": "http://127.0.0.1:9765", "decider_big.url": "http://127.0.0.1:9766"}
    assert build_deciders(s)["2b"].base_url == "http://127.0.0.1:9765"
    assert "profile" in DECIDER_SETTINGS and "llm" in BLURBS


def test_cli_profile_and_llm_escalation():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cascade")
    add_decider_args(ap)
    s = settings_from_args(ap.parse_args(["--profile", "dev-local", "--escalate", "llm"]))
    assert s.profile == "dev-local" and s.escalate_to == "llm" and s.decider_2b_url == Settings().decider_2b_url
    assert settings_from_args(ap.parse_args([])).profile == "dev"
    with pytest.raises(SystemExit):
        ap.parse_args(["--escalate", "gpt"])


# --------------------------------------------------------------------------- cascade through the llm slot
def llm_pipeline(script, model, **changes) -> tuple[Pipeline, FakeDecider, HouseAgent]:
    settings = Settings(**{"decider_mode": "cascade", "escalate_to": "llm", **changes})
    small = FakeDecider("decider-2b", script=script)
    small.label = "Strands Decider 2B (CPU)"
    house = HouseAgent(settings, model=model, name="llm", label="Qwen3 1.7B on llama.cpp (CPU)", route="local")
    bedrock = HouseAgent(settings, model=ScriptedModel(fail=RuntimeError("must not be used")))
    return Pipeline(settings, {"2b": small, "llm": house, "bedrock": bedrock}), small, house


def decider_props(p: Pipeline):
    return [e for e in p.proposals(kind="speech") if e.data["by"] not in ("keyword", "reflex")]


async def test_unsure_2b_escalates_to_the_local_llm_whose_tool_call_acts():
    model = ScriptedModel([("lights_on", {"room": "living"})], reply="Lights coming on.", usage=(700, 30))
    p, small, house = llm_pipeline([("can't see", "intent", ("Lights on", 0.55))], model)
    await p.clock("20:00")
    await p.say("I can't see a thing in here.", who="alex", room="living")
    await p.settle()
    props = decider_props(p)
    assert [(e.data["by"], e.data["mode"], e.data["route"]) for e in props] == [
        ("decider-2b", "shadow", "local"), ("llm", "act", "local")]
    shadow, act = props
    assert shadow.data["reason"] == "unsure, asking Qwen3 1.7B on llama.cpp (CPU)"
    assert shadow.data["model"] == "Strands Decider 2B (CPU)"                       # registry label, not the constant
    a = act.data
    assert a["model"] == "Qwen3 1.7B on llama.cpp (CPU)" and a["escalated_from"] == shadow.id and a["action"] == "lights_on"
    assert a["reason"] == "Qwen3 called lights_on: turn the lights on" and a["tokens"] == {"input": 1400, "output": 60}
    assert a["tool_calls"] == [{"name": "lights_on", "input": {"room": "living"}}] and a["reply"] == "Lights coming on."
    assert model.calls == 2 and len(small.calls) == 1 and house.calls == 1
    assert p.commands(device_id="light_living")[0].data["command"] == "on" and not p.bus.errors


async def test_sure_2b_still_decides_locally_with_a_local_llm_configured():
    model = ScriptedModel([("lights_off", {"room": "living"})])
    p, small, _ = llm_pipeline([("turn on the lights", "intent", ("Lights on", 0.95))], model)
    await p.say("Can you turn on the lights?", who="alex", room="living")
    await p.settle()
    props = decider_props(p)
    assert len(props) == 1 and props[0].data["route"] == "local" and props[0].data["by"] == "decider-2b"
    assert props[0].data["mode"] == "act" and model.calls == 0 and p.commands(device_id="light_living")


async def test_local_llm_failure_falls_back_to_the_unsure_2b_answer():
    model = ScriptedModel(fail=ConnectionError("llama-server is down"))
    p, _, _ = llm_pipeline([("can't see", "intent", ("Lights on", 0.6))], model)
    await p.say("I can't see a thing in here.", who="alex", room="living")
    await p.settle()
    props = decider_props(p)
    assert [(e.data["by"], e.data["mode"]) for e in props] == [("decider-2b", "shadow"), ("decider-2b", "act")]
    assert props[1].data["reason"] == "Qwen3 1.7B on llama.cpp (CPU) unavailable, using the unsure 2B answer"
    warns = [e.data["text"] for e in p.logs(level="warn")]
    assert warns == ["Qwen3 1.7B on llama.cpp (CPU) failed: ConnectionError: llama-server is down"]


async def test_local_llm_failure_prefers_a_healthy_12b():
    model = ScriptedModel(fail=RuntimeError("timeout"))
    big = FakeDecider("decider-12b", script=[("can't see", "intent", ("Lights on", 0.96))])
    big.label = "Strands Decider 12B (CPU)"
    p, _, _ = llm_pipeline([("can't see", "intent", ("Lights on", 0.6))], model)
    p.agent("intent_decider").deciders["12b"] = big
    await p.say("I can't see a thing in here.", who="alex", room="living")
    await p.settle()
    props = decider_props(p)
    assert [e.data["by"] for e in props] == ["decider-2b", "decider-12b"] and props[1].data["route"] == "12b"
    assert props[1].data["model"] == "Strands Decider 12B (CPU)"
    assert props[1].data["reason"].startswith("Qwen3 1.7B on llama.cpp (CPU) failed, asked the 12B;")


async def test_escalate_llm_without_an_llm_slot_uses_the_plain_cascade():
    settings = Settings(decider_mode="cascade", escalate_to="llm")
    small = FakeDecider("decider-2b", script=[("can't see", "intent", ("Lights on", 0.55))])
    big = FakeDecider("decider-12b", script=[("can't see", "intent", ("Lights on", 0.96))])
    p = Pipeline(settings, {"2b": small, "12b": big})
    await p.say("I can't see a thing in here.", who="alex", room="living")
    await p.settle()
    assert [e.data["by"] for e in decider_props(p)] == ["decider-2b", "decider-12b"]


class LoopingModel(ScriptedModel):
    """A small local model stuck in a loop: answers EVERY call with the same tool calls, never with text."""

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        async for ev in super().stream(messages[:1], tool_specs, system_prompt, **kwargs):
            yield ev


async def test_a_looping_local_model_is_stopped_after_two_tool_rounds_and_its_calls_collapsed():
    model = LoopingModel([("warmer", {})])
    p, _, house = llm_pipeline([("freezing", "intent", ("Warmer", 0.6))], model)
    await p.say("It's freezing in here.", who="alex", room="living")
    await p.settle()
    acts = [e.data for e in decider_props(p) if e.data["by"] == "llm"]
    assert model.calls == 2 and house.max_tool_rounds == 2                      # not fourteen
    assert [(a["action"], a["confidence"]) for a in acts] == [("warmer", 0.9)]  # one proposal, not two
    assert acts[0]["tool_calls"] == [{"name": "warmer", "input": {}}, {"name": "warmer", "input": {}}]
    assert acts[0]["reply"] is None and p.commands(device_id="thermostat_main") and not p.bus.errors


async def test_two_different_calls_in_a_round_still_give_two_proposals():
    model = ScriptedModel([("ask_resident", {"action": "lights_on", "room": "living"}), ("ask_resident", {"action": "tv_on"})])
    p, _, _ = llm_pipeline([("remote", "intent", ("Not a request", 0.88))], model)
    await p.say("Where's the remote?", who="alex", room="living")
    await p.settle()
    acts = [e.data for e in decider_props(p) if e.data["by"] == "llm"]
    assert [a["action"] for a in acts] == ["lights_on", "tv_on"] and len(p.asks()) == 2


# --------------------------------------------------------------------------- runtime end to end, fake profile
async def test_runtime_on_the_fake_profile_runs_speech_to_command_through_the_registry():
    rt = Runtime(Settings(profile="fake", escalate_to="llm"), tick_seconds=0)
    assert rt.registry is not None and rt.registry.profile == "fake" and set(rt.deciders) >= {"2b", "12b", "llm"}
    # the profile's scripted llm answers "no_action"; make the 2B sure instead so the demo acts
    rt.deciders["2b"].script = [("can't see", "*", ("Lights on", 0.95))]
    await rt.world.start()
    rt._start_agents()
    await rt.bus.drain()
    await rt.step({"do": "set_time", "time": "21:00", "weekday": "Tuesday"})
    await rt.step({"do": "move", "who": "alex", "to": "living"})
    await rt.step({"do": "say", "who": "alex", "text": "I can't see a thing in here."})
    await rt.bus.drain()
    act = rt.bus.find(T.PROPOSAL, by="decider-2b", mode="act")
    assert act and act[0].data["model"] == "fake decider" and rt.bus.find(T.APPLIED, device_id="light_living")
    blurb = {g["name"]: g["blurb"] for g in rt.agent_graph()}
    assert blurb["intent_decider"] == BLURBS["llm"]["intent_decider"]
    await rt.close()
    assert rt.registry._built == {}


async def test_changing_the_profile_rebuilds_the_slots_from_the_new_registry():
    rt = Runtime(Settings(profile="dev"), tick_seconds=0)
    old_registry = rt.registry
    assert isinstance(rt.deciders["2b"], HttpDecider)
    await rt.update_settings(profile="fake")
    assert rt.registry is not old_registry and rt.registry.profile == "fake"
    assert isinstance(rt.deciders["2b"], FakeDecider) and rt.deciders["llm"].label == "scripted model"
    await rt.close()


def test_registry_llm_model_is_shared_between_the_house_agent_and_the_registry():
    reg = load_registry("fake")
    deciders = build_deciders(Settings(profile="fake"), reg)
    assert deciders["llm"].model() is reg.strands_model("llm")
