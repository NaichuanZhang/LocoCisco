"""Runtime wiring: real sim + real agents on one bus, FakeDecider + a scripted house agent (no servers,
no Bedrock)."""
import pytest

from homedemo.agents.house_agent import HouseAgent
from homedemo.contracts import Settings, T
from homedemo.decider_client import FakeDecider
from homedemo.patterns.fakes import ScriptedModel
from homedemo.runtime import BLURBS, Runtime, build_deciders


def _deciders(settings: Settings | None = None):
    script = [("can't see", "*", ("Lights on", 0.95))]
    return {"2b": FakeDecider("decider-2b", script=script), "12b": FakeDecider("decider-12b", script=script),
            "fake": FakeDecider(script=script),
            "bedrock": HouseAgent(settings or Settings(), model=ScriptedModel([("lights_on", {"room": "living"})]))}


@pytest.fixture
async def rt():
    r = Runtime(deciders=_deciders(), tick_seconds=0)
    await r.world.start()
    r._start_agents()
    await r.bus.drain()
    yield r
    await r.close()


async def test_speech_to_command_through_real_sim(rt):
    await rt.step({"do": "set_time", "time": "21:00", "weekday": "Tuesday"})
    await rt.step({"do": "move", "who": "alex", "to": "living"})
    await rt.step({"do": "say", "who": "alex", "text": "I can't see a thing in here."})
    await rt.bus.drain()
    applied = rt.bus.find(T.APPLIED, device_id="light_living")
    assert applied and applied[-1].data["state"]["on"] is True
    assert rt.snapshot()["devices"][0]["state"]["on"] is True


async def test_reset_rebuilds_agents_and_world(rt):
    first = list(rt.agents)
    await rt.step({"do": "set_device", "device": "tv_living", "state": {"on": True}})
    await rt.reset()
    assert all(a is not b for a, b in zip(first, rt.agents))
    tv = next(d for d in rt.snapshot()["devices"] if d["device_id"] == "tv_living")
    assert tv["state"]["on"] is False
    # Only one set of agents is subscribed: one utterance gives one keyword proposal.
    n = len(rt.bus.find(T.PROPOSAL))
    await rt.step({"do": "say", "who": "alex", "text": "turn on the lights"})
    await rt.bus.drain()
    kw = [e for e in list(rt.bus.history) if e.topic == T.PROPOSAL and e.data["by"] == "keyword"]
    assert len(rt.bus.find(T.PROPOSAL)) > n and len(kw) == 1


async def test_settings_and_graph(rt):
    graph = rt.agent_graph()                                   # cascade -> bedrock: the blurbs say so
    blurb = {g["name"]: g["blurb"] for g in graph}
    assert blurb["intent_decider"] == BLURBS["bedrock"]["intent_decider"] and "Bedrock" in blurb["scene_decider"]
    s = await rt.update_settings(escalate_to="12b")
    assert {g["name"]: g["blurb"] for g in rt.agent_graph()}["scene_decider"] == BLURBS["12b"]["scene_decider"]
    s = await rt.update_settings(decider_mode="fake", act_threshold=0.9)
    assert s.decider_mode == "fake" and rt.settings.act_threshold == 0.9
    with pytest.raises(ValueError):
        await rt.update_settings(nope=1)
    graph = rt.agent_graph()
    assert [g["name"] for g in graph][:2] == ["perception", "voice"] and all("busy" in g for g in graph)


async def test_unsure_2b_goes_through_the_house_agent_to_claude(rt):
    rt.deciders["2b"] = FakeDecider("decider-2b", script=[("can't see", "intent", ("Lights on", 0.55))])
    await rt.step({"do": "set_time", "time": "21:00", "weekday": "Tuesday"})
    await rt.step({"do": "move", "who": "alex", "to": "living"})
    await rt.step({"do": "say", "who": "alex", "text": "I can't see a thing in here."})
    await rt.bus.drain()
    shadow = rt.bus.find(T.PROPOSAL, by="decider-2b", mode="shadow")
    act = rt.bus.find(T.PROPOSAL, by="bedrock")
    assert shadow and act and act[0].data["escalated_from"] == shadow[0].id and act[0].data["route"] == "bedrock"
    assert act[0].data["tokens"] and rt.bus.find(T.APPLIED, device_id="light_living")[-1].data["state"]["on"] is True


def test_build_deciders_has_a_lazy_house_agent():
    deciders = build_deciders(Settings(bedrock_profile="no-such-profile-for-this-test"))
    assert {"2b", "12b", "fake", "bedrock"} <= set(deciders) and isinstance(deciders["bedrock"], HouseAgent)
    assert deciders["bedrock"]._model is None                  # Bedrock is only touched on first use


async def test_runner_steps_are_rejected_by_manual_step(rt):
    with pytest.raises(ValueError):
        await rt.step({"do": "expect", "what": "device"})


async def test_reset_keeps_the_ui_stream():
    from homedemo.web.server import create_app

    r = Runtime(deciders=_deciders(), tick_seconds=0)
    app = create_app(r)                       # same order as python -m homedemo
    await r.start(ticking=False)
    before = [s.name for s in r.bus._subs]
    await r.reset()
    names = [s.name for s in r.bus._subs]
    assert names.count("web.stream") == 1 and sorted(names) == sorted(before)
    assert app.state.hub is not None
    await r.close()
