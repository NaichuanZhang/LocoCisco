"""One live smoke test against the 2B decider (HOMEDEMO_2B_URL, default Settings.decider_2b_url): 2 real
calls. Runs only with HOMEDEMO_LIVE=1 and skips when the server is down."""
from __future__ import annotations

import os

import httpx
import pytest

from homedemo.contracts import ACTIONS, Settings
from homedemo.decider_client import HttpDecider

from .conftest import Pipeline

URL_2B = os.environ.get("HOMEDEMO_2B_URL", Settings().decider_2b_url)


def _2b_is_up() -> bool:
    try:
        return httpx.get(f"{URL_2B}/health", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


async def test_live_2b_hazard_and_speech():
    if os.environ.get("HOMEDEMO_LIVE") != "1":
        pytest.skip("set HOMEDEMO_LIVE=1 to run against the 2B server")
    if not _2b_is_up():
        pytest.skip(f"2B decider not running at {URL_2B}")
    decider = HttpDecider("decider-2b", URL_2B)
    try:
        p = Pipeline(Settings(decider_mode="2b"), {"2b": decider})
        await p.clock("17:20")
        await p.device("light_kitchen", {"on": True, "brightness": 100})
        await p.device("stove_kitchen", {"burner_on": True})
        await p.camera("kitchen", [p.person("maya", pose="standing on a chair", near="stove")])   # call 1
        await p.settle()
        await p.clock("20:00")
        await p.say("Can you turn on the lights?", who="alex", room="living")                       # call 2
        await p.settle()
    finally:
        await decider.aclose()
    scene = [e for e in p.proposals(kind="scene") if e.source == "agent.scene_decider"]
    speech = [e for e in p.proposals(kind="speech") if e.data["by"] != "keyword"]
    assert len(scene) == 1 and len(speech) == 1
    for e in scene + speech:
        d = e.data
        assert d["by"] == "decider-2b" and d["mode"] == "act" and d["action"] in ACTIONS
        assert d["latency_ms"] > 50 and abs(sum(d["probabilities"].values()) - 1) < 1e-3
        assert set(d["probabilities"]) <= set(ACTIONS) and d["state_text"] and d["question"]
    assert scene[0].data["action"] == "alert_parent"            # measured s8: 0.92 on the 2B
    assert speech[0].data["action"] == "lights_on"              # measured p1: 0.87 on the 2B
    assert p.commands(device_id="phone_parents")
    assert not p.bus.errors
