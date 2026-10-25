"""Keyword baseline: naive substring matching, shadow mode, never executed."""
from __future__ import annotations

import pytest

from homedemo.agents.keyword import keyword_match
from homedemo.contracts import T


@pytest.mark.parametrize("text, action", [
    ("Can you turn on the lights?", "lights_on"),
    ("Don't turn off the TV, I'm still watching.", "tv_off"),       # no negation handling, on purpose
    ("I left the lights on at the office again", "lights_on"),      # substring match, no understanding
    ("把灯关了吧，我要睡了。", "none"),                                # needs the exact token 关灯
    ("关灯", "lights_off"), ("开灯", "lights_on"), ("开电视", "tv_on"), ("关电视", "tv_off"),
    ("Could you make it a bit warmer?", "warmer"), ("It's so stuffy, cooler please", "cooler"),
    ("Unlock the front door.", "unlock_door"),
    ("I can't see a thing in here.", "none"), ("Where's the remote?", "none"),
])
def test_keyword_match(text, action):
    assert keyword_match(text)[0] == action


async def test_keyword_shadow_is_published_but_never_executed(make_pipeline):
    p = make_pipeline([("still watching", "intent", ("No action", 0.9))])
    await p.device("tv_living", {"on": True, "volume": 20, "playing": "game"})
    await p.say("Don't turn off the TV, I'm still watching.", who="alex", room="living")
    await p.settle()
    kw = p.proposals(by="keyword")
    assert len(kw) == 1
    d = kw[0].data
    assert d["mode"] == "shadow" and d["action"] == "tv_off" and d["target"] == "tv_living"
    assert d["confidence"] == 1.0 and d["probabilities"] == {"tv_off": 1.0} and d["kind"] == "speech"
    assert d["reason"] == "keyword match: 'turn off the tv'" and d["speaker"]["id"] == "alex"
    assert kw[0].source == "agent.keyword" and kw[0].cause == p.events(T.UTTERANCE)[0].id
    assert not p.verdicts(proposal_id=kw[0].id)
    assert not p.commands()
    # the decider's own answer (none) is the one that gets a verdict
    decided = [e for e in p.proposals(kind="speech") if e.data["by"] != "keyword"]
    assert p.verdicts(proposal_id=decided[0].id)[0].data["verdict"] == "ignore"


async def test_keyword_none_when_nothing_matches_and_off_switch(make_pipeline):
    p = make_pipeline([("Heard", "intent", ("Lights on", 0.95))])
    await p.say("I can't see a thing in here.", who="alex", room="living")
    await p.settle()
    kw = p.proposals(by="keyword")
    assert len(kw) == 1 and kw[0].data["action"] == "none" and kw[0].data["reason"] == "no keyword matched"
    assert len(p.commands(device_id="light_living")) == 1       # the decider understood it anyway
    p.settings.keyword_shadow = False
    await p.say("Lights off", who="alex", room="living")
    await p.settle()
    assert len(p.proposals(by="keyword")) == 1                   # nothing new while switched off
