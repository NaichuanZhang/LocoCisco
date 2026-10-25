"""Live checks (UNSPOKEN_LIVE=1): the dev deciders on :8765/:8766 and the dev-local llama.cpp server on :8010
(scripts/dev-llm.sh start). Skipped in the normal suite."""
from __future__ import annotations

import os

import pytest

from homedemo.agents.house_agent import HouseAgent
from homedemo.contracts import Settings
from homedemo.decider.questions import build_speech_query
from homedemo.runtime import build_deciders
from unspoken.models import load_registry

pytestmark = pytest.mark.skipif(os.environ.get("UNSPOKEN_LIVE") != "1", reason="set UNSPOKEN_LIVE=1 for live checks")


async def test_dev_deciders_answer_the_probe():
    reg = load_registry("dev")
    try:
        health = {h.role: h for h in await reg.check(["decider", "decider_big"], timeout=60)}
    finally:
        await reg.aclose()
    assert health["decider"].ok and health["decider_big"].ok
    assert health["decider"].latency_ms < 15000 and health["decider"].detail.startswith("P(true)=")


async def test_dev_local_llm_makes_a_real_tool_call():
    reg = load_registry("dev-local")
    try:
        (llm,) = await reg.check(["llm"], timeout=120)
        assert llm.ok, llm.detail
        settings = Settings(profile="dev-local", escalate_to="llm")
        house = build_deciders(settings, reg)["llm"]
        assert isinstance(house, HouseAgent) and house.route == "local"
        data = {"room": "living", "text": "I can't see a thing in here.", "lang": "en", "source": "person",
                "speaker": {"id": "alex", "name": "Alex", "role": "parent", "match": 0.95},
                "context": "Living room. Lights off, evening."}
        query = build_speech_query(data)
        out = await house.decide(reg.decider(), query, data, router=False)      # straight to the local model
        assert out.error is None, out.error
        names = [c["name"] for c in out.tool_calls]
        assert names and names[0] in ("lights_on", "ask_resident"), names
        assert out.tokens and out.tokens["input"] > 500 and out.latency_ms > 0
    finally:
        await reg.aclose()
