"""The fake meeting runtime behind the room screen: it must play the demo story believably, with the same
event shapes the real pipeline publishes, so the page is built against realistic data."""
from __future__ import annotations

import json

import pytest

from unspoken.meeting.contracts import ACTIONS, MT, MeetingSettings
from unspoken.meeting.web.dev_fake import FakeMeetingRuntime, build_scenarios


@pytest.fixture
def runtime() -> FakeMeetingRuntime:
    return FakeMeetingRuntime(fast=True)


async def _play(runtime: FakeMeetingRuntime, scenario_id: str) -> dict:
    await runtime.run_scenario(scenario_id)
    await runtime._task
    await runtime.bus.drain()
    return runtime.bus.find(MT.SCENARIO, state="finished")[-1].data


async def test_demo_story_passes_its_own_expectations(runtime: FakeMeetingRuntime):
    finished = await _play(runtime, "demo_story")
    assert finished["failed"] == 0 and finished["passed"] >= 10
    assert not runtime.bus.errors
    items = runtime.items()
    assert any(i["kind"] == "task" and i["owner"] == "Sam" and i["due"] == "Friday" for i in items)
    assert any(i["kind"] == "task" and i["owner"] is None for i in items)
    assert sum(i["kind"] == "decision" for i in items) == 2
    assert runtime.devices["fan"]["state"]["on"] is False
    # the recap only left after the owner approved it, and only as text
    channels = [p["channel"] for p in runtime.posts]
    assert channels.count("notes") >= 2 and "dm" in channels and "tasks" in channels
    eg = runtime.snapshot()["egress"]
    assert eg["total_bytes"] == sum(len(p["text"].encode()) for p in runtime.posts) > 0
    assert eg["audio_bytes_out"] == 0 and eg["video_bytes_out"] == 0 and eg["kind"] == "text"
    # every event is JSON-able and carries the envelope the page relies on
    for e in runtime.bus.history:
        j = json.loads(json.dumps(e.to_json(), default=str))
        assert {"topic", "data", "source", "ts", "id", "cause", "wall"} <= set(j)
    # every proposal names an action from the contract
    for p in runtime.bus.find(MT.PROPOSAL):
        assert p.data["action"] in ACTIONS, p.data["action"]


async def test_every_beat_passes(runtime: FakeMeetingRuntime):
    for sc in build_scenarios():
        rt = FakeMeetingRuntime(fast=True)
        finished = await _play(rt, sc["id"])
        assert finished["failed"] == 0, (sc["id"], finished)
        assert not rt.bus.errors, sc["id"]


async def test_proposals_carry_the_xray_fields(runtime: FakeMeetingRuntime):
    await _play(runtime, "beat_answer")
    deciders = [p.data for p in runtime.bus.find(MT.PROPOSAL, by="decider")]
    assert deciders
    for d in deciders:
        assert d["route"] == "local-decider" and d["model"] == "Strands Decider 12B (GPU)"
        assert d["threshold"] == runtime.settings.act_threshold and d["latency_ms"] > 0 and d["tokens"] is None
        assert abs(sum(d["probabilities"].values()) - 1) < 1e-3
        if d["request"]:
            assert set(d["request"]) == {"state", "questions"} and d["request"]["state"]
            for key, q in d["request"]["questions"].items():
                assert q["type"] in ("choice", "noul") and key in d["answers"]
                assert d["answers"][key]["type"] == q["type"]
    # the unsure one is published as a shadow and escalated to the local LLM, which spends tokens and calls tools
    shadow = [d for d in deciders if d["mode"] == "shadow"]
    assert len(shadow) == 1 and shadow[0]["confidence"] < shadow[0]["threshold"] and "Qwen" in shadow[0]["reason"]
    llm = runtime.bus.find(MT.PROPOSAL, by="llm")
    assert len(llm) == 1
    l = llm[0].data
    shadow_ev = [p for p in runtime.bus.find(MT.PROPOSAL, by="decider") if p.data["mode"] == "shadow"][0]
    assert l["escalated_from"] == shadow_ev.id and l["route"] == "local-llm" and "vLLM" in l["model"]
    assert l["tokens"]["input"] > 0 and l["tokens"]["output"] > 0
    assert [t["name"] for t in l["tool_calls"]] == ["search_docs", "answer_question"] and l["reply"]
    # the policy acted on the LLM's proposal and an answer card names the doc
    assert runtime.bus.find(MT.VERDICT, proposal_id=llm[0].id, verdict="act")
    cards = runtime.snapshot()["screen"]["cards"]
    assert cards[0]["card"] == "answer" and cards[0]["doc"] == "notes-2026-09-12.md"


async def test_off_the_record_wipes_and_resumes(runtime: FakeMeetingRuntime):
    await _play(runtime, "beat_checkins")
    await runtime._do_say({"who": "priya", "text": "Sam, send the deck by Friday."})
    await runtime.bus.drain()
    assert len(runtime.snapshot()["screen"]["captions"]) == 3
    await runtime._do_say({"who": "mei", "text": "Off the record for a second, I think Alex is about to resign."})
    await runtime.bus.drain()
    snap = runtime.snapshot()
    assert snap["privacy"]["state"] == "off_record" and snap["privacy"]["by"] == "reflex"
    assert snap["privacy"]["wiped_count"] == 4 and snap["screen"]["captions"] == []
    assert runtime.devices["lamp"]["state"]["scene"] == "off_record"
    reflex = runtime.bus.find(MT.PROPOSAL, by="reflex", action="pause_notes")
    assert reflex and reflex[0].data["route"] == "rule" and reflex[0].data["latency_ms"] <= 1.0
    assert runtime.bus.find(MT.VERDICT, proposal_id=reflex[0].id, verdict="act")
    # the sensitive half of the trigger sentence never reached the bus
    texts = [e.data["text"] for e in runtime.bus.history if e.topic == MT.UTTERANCE]
    assert not any("resign" in t for t in texts) and any(t.endswith("[…]") for t in texts)
    # lines said while off the record are dropped before fusion, not hidden
    await runtime._do_say({"who": "priya", "text": "Noted. Nothing in the notes about that."})
    await runtime.bus.drain()
    assert not any("Noted" in e.data.get("text", "") for e in runtime.bus.history)
    assert runtime.transcript() == []
    await runtime._do_say({"who": "priya", "text": "OK, back on the record."})
    await runtime.bus.drain()
    assert runtime.privacy["state"] == "on" and runtime.devices["lamp"]["state"]["scene"] == "talk"
    assert runtime.bus.find(MT.PROPOSAL, by="reflex", action="resume_notes")


async def test_guest_hides_docs_and_denies_sharing(runtime: FakeMeetingRuntime):
    await _play(runtime, "beat_answer")
    assert runtime.snapshot()["screen"]["cards"][0]["card"] == "answer"
    await runtime._do_enter({"who": "visitor", "seat": "s4"})
    snap = runtime.snapshot()
    assert any(p["guest"] for p in snap["participants"]) and "docs hidden" in snap["screen"]["banner"]
    assert not any(c["card"] == "answer" for c in snap["screen"]["cards"])
    await runtime._do_say({"who": "priya", "text": "Can we pull up the margin sheet?"})
    await runtime.bus.drain()
    denied = runtime.bus.find(MT.VERDICT, verdict="deny")
    assert denied and "guest" in denied[0].data["rule"] and denied[0].data["risk"] == "high"
    await runtime._do_say({"who": "visitor", "text": "Hi, just listening in."})
    await runtime.bus.drain()
    assert runtime.snapshot()["screen"]["captions"][-1]["name"] == "Guest"
    await runtime._do_leave({"who": "visitor"})
    snap = runtime.snapshot()
    assert snap["screen"]["banner"] is None and any(c["card"] == "answer" for c in snap["screen"]["cards"])


async def test_snapshot_banner_and_reset(runtime: FakeMeetingRuntime):
    await _play(runtime, "beat_task")
    await runtime._do_clock({"minutes_left": 2})
    assert runtime.snapshot()["screen"]["banner"] == "2 min left. 1 task has no owner."
    assert runtime.snapshot()["screen"]["agenda"]["minutes_left"] == 2
    await runtime.reset()
    snap = runtime.snapshot()
    assert snap["status"]["state"] == "ended" and snap["items"] == [] and snap["screen"]["banner"] is None
    assert snap["participants"] == [] and snap["devices"][1]["state"]["on"] is True
    with pytest.raises(ValueError):
        await runtime.update_settings(bogus=1)
    assert isinstance(await runtime.update_settings(speed=3.0), MeetingSettings)
    with pytest.raises(KeyError):
        await runtime.run_scenario("nope")


async def test_search_docs_and_recap(runtime: FakeMeetingRuntime):
    await _play(runtime, "beat_task")
    hits = await runtime.search_docs("launch date")
    assert hits and {"doc", "chunk", "score", "text"} <= set(hits[0])
    md = await runtime.recap()
    assert "**Decisions**" in md and "send the deck — Sam, due Friday" in md and "no owner yet" in md
