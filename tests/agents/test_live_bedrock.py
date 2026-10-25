"""One live run of the Bedrock cascade: the real 2B server and Claude Haiku 4.5 on Bedrock, through the
full agent pipeline. Runs only with HOMEDEMO_LIVE=1 (needs the 2B at HOMEDEMO_2B_URL, default
Settings.decider_2b_url, and AWS credentials for Settings.bedrock_profile)."""
from __future__ import annotations

import os

import httpx
import pytest

from homedemo.agents.house_agent import HouseAgent
from homedemo.contracts import ACTIONS, Settings, T
from homedemo.decider_client import HttpDecider

from .conftest import Pipeline

URL_2B = os.environ.get("HOMEDEMO_2B_URL", Settings().decider_2b_url)
FIELDS = {"route", "model", "request", "answers", "threshold", "tokens", "tool_calls", "reply"}


def _2b_is_up() -> bool:
    try:
        return httpx.get(f"{URL_2B}/health", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


async def test_live_bedrock_cascade_on_an_indirect_request():
    if os.environ.get("HOMEDEMO_LIVE") != "1":
        pytest.skip("set HOMEDEMO_LIVE=1 to run against the 2B server and Bedrock")
    if not _2b_is_up():
        pytest.skip(f"2B decider not running at {URL_2B}")
    settings = Settings(decider_mode="cascade", escalate_to="bedrock", decider_2b_url=URL_2B)
    decider = HttpDecider("decider-2b", URL_2B)
    house = HouseAgent(settings)
    assert await house.healthy()
    try:
        p = Pipeline(settings, {"2b": decider, "bedrock": house})
        await p.clock("21:00")
        await p.say("I can't see a thing in here.", who="alex", room="living")     # 2B call, Claude if unsure
        await p.settle()
    finally:
        await decider.aclose()
    props = [e for e in p.proposals(kind="speech") if e.data["by"] != "keyword"]
    assert props and all(FIELDS <= set(e.data) and e.data["action"] in ACTIONS for e in props)
    first = props[0].data
    assert first["by"] == "decider-2b" and first["route"] == "local" and first["tokens"] is None
    assert first["latency_ms"] > 50 and first["answers"]["intent"]["choice"]
    act = props[-1].data
    assert act["mode"] == "act" and act["action"] == "lights_on"
    if act["by"] == "bedrock":                                  # the 2B was unsure (measured: 67-80%)
        assert first["mode"] == "shadow" and act["escalated_from"] == props[0].id
        assert act["tokens"]["input"] > 0 and act["tool_calls"][0]["name"] in ("lights_on", "ask_resident")
        assert act["model"] == "Claude Haiku 4.5 on Bedrock" and act["latency_ms"] > 50
    else:                                                       # the 2B was sure on its own
        assert act is first and len(props) == 1
    assert p.commands(device_id="light_living") or p.asks(action="lights_on")
    assert not p.bus.errors and not [e for e in p.logs(level="warn") if "failed" in e.data["text"]]
