"""Docs search (keyword and embed paths), recap, the MeetingRuntime API, the scenario library (every scenario
passes in sim with profile fake), the play CLI, and a replay of the sensors fixture WAV through the whole
pipeline with the sensors workstream's fakes (real VAD, scripted ASR)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from unspoken.core.fakes import ScriptedModel
from unspoken.meeting import scenarios as S
from unspoken.meeting.base import MeetingState
from unspoken.meeting.contracts import MT, SCENARIO_STEPS
from unspoken.meeting.docs import DocIndex, answer_from, chunk_text, keyword_score
from unspoken.meeting.fakes import FakeEmbedder
from unspoken.meeting.recap import recap_markdown, recap_template
from unspoken.meeting.runtime import MeetingRuntime

from .conftest import JOIN_PRIYA, JOIN_SAM, START, make_registry, scripted

DOCS = Path(__file__).resolve().parents[2] / "examples" / "docs"


# ------------------------------------------------------------------ docs
async def test_keyword_search_finds_the_right_doc():
    idx = DocIndex(str(DOCS)).load()
    assert len(idx.chunks) >= 3 and {c.doc for c in idx.chunks} == {"launch-plan.md", "pricing-q3.md", "refund-policy.md"}
    hits = await idx.search("what is the refund policy for annual plans", k=2)
    assert hits[0]["doc"] == "refund-policy.md" and hits[0]["method"] == "keyword" and hits[0]["score"] > 0
    answer = answer_from(hits, "refund policy for annual plans")
    assert "annual" in answer.lower() and answer.endswith("(refund-policy.md)") and not answer.startswith("#")
    hits = await idx.search("press briefing date", k=1)
    assert hits[0]["doc"] == "launch-plan.md"
    assert answer_from([], "x").startswith("I found nothing")


async def test_embed_search_uses_the_embedder_with_keyword_fallback_on_failure():
    emb = FakeEmbedder()
    idx = DocIndex(str(DOCS), emb).load()
    hits = await idx.search("Nimbus 2 launch plan press briefing September", k=1)
    assert hits and hits[0]["method"] == "embed" and emb.calls == 2 and idx.embedded

    class Broken:
        async def embed(self, texts):
            raise RuntimeError("embed server down")

    idx2 = DocIndex(str(DOCS), Broken()).load()
    hits = await idx2.search("refund policy", k=1)
    assert hits[0]["method"] == "keyword"


def test_chunking_and_scoring():
    chunks = chunk_text("a\n\n" + "b " * 400 + "\n\nc", size=300)
    assert len(chunks) >= 2 and keyword_score("pricing proposal", "The Q3 pricing proposal moves Starter.") > 0
    assert keyword_score("", "anything") == 0.0 and keyword_score("zzz", "nothing matches") == 0.0


def test_missing_docs_dir_is_empty_not_an_error(tmp_path):
    assert DocIndex(str(tmp_path / "nope")).load().chunks == []


# ------------------------------------------------------------------ recap
def test_recap_template_lists_decisions_tasks_and_owners():
    s = MeetingState()
    s.status.update(title="Pricing sync", owner="Priya")
    s.items["d1"] = {"item_id": "d1", "kind": "decision", "text": "Starter goes to $39", "owner": None, "due": None, "status": "open"}
    s.items["t1"] = {"item_id": "t1", "kind": "task", "text": "Send the deck", "owner": "Sam", "due": "2026-10-30", "status": "open"}
    s.items["t2"] = {"item_id": "t2", "kind": "task", "text": "Update the pricing page", "owner": None, "due": None, "status": "open"}
    s.items["t3"] = {"item_id": "t3", "kind": "task", "text": "gone", "owner": None, "due": None, "status": "undone"}
    text = recap_template(s)
    assert "- Starter goes to $39" in text and "- Send the deck — Sam, Fri Oct 30" in text and "no owner yet" in text
    assert "gone" not in text and "1 task still needs an owner" in text


async def test_recap_with_llm_and_fallback():
    s = MeetingState()
    s.items["t1"] = {"item_id": "t1", "kind": "task", "text": "Send the deck", "owner": "Sam", "due": None, "status": "open"}
    polished = ScriptedModel([], reply="**Recap**\n- Send the deck — Sam")
    assert (await recap_markdown(s, lambda: polished)) == "**Recap**\n- Send the deck — Sam" and polished.calls == 1
    broken = ScriptedModel(fail=RuntimeError("down"))
    assert (await recap_markdown(s, lambda: broken)) == recap_template(s)


# ------------------------------------------------------------------ runtime API
async def test_runtime_snapshot_transcript_chat_and_reset(make_pipeline):
    p = await make_pipeline(scripted("Task", 0.95, owner="Sam"))
    await p.say("priya", "Sam, can you send the deck by Friday?")
    snap = p.rt.snapshot()
    assert set(snap) >= {"status", "privacy", "participants", "screen", "items", "devices", "egress", "models", "settings", "mode"}
    assert snap["status"]["title"] == "Pricing sync" and snap["status"]["owner"] == "Priya" and snap["mode"] == "sim"
    assert [x["name"] for x in snap["participants"]] == ["Priya", "Sam"] and snap["items"][0]["kind"] == "task"
    assert any(m["role"] == "decider" and m["ok"] for m in snap["models"]) and snap["agents"][0]["name"] == "fusion"
    last_start = p.utterances()[-1]["start"]
    assert [u["text"] for u in p.rt.transcript(since=last_start)] == ["Sam, can you send the deck by Friday?"]
    assert p.rt.transcript(since=1e9) == []
    out = await p.rt.chat({"command": "here", "user": "Lin", "args": {"name": "Lin"}})
    assert out["ok"] and "Lin" in out["text"] and "Lin" in p.rt.state.names()
    out = await p.rt.chat({"command": "ask", "user": "Sam", "text": "refund policy for annual plans"})
    assert out["ok"] and "refund-policy.md" in out["text"]
    out = await p.rt.chat({"command": "recap", "user": "Priya"})
    assert out["text"].startswith("**Recap")
    assert (await p.rt.chat({"command": "bogus", "user": "x"}))["ok"] is False
    priv = await p.rt.set_privacy("off_record", "test")
    assert priv["state"] == "off_record" and p.rt.transcript() == []
    await p.rt.update_settings(act_threshold=0.7)
    assert p.rt.settings.act_threshold == 0.7
    with pytest.raises(ValueError):
        await p.rt.update_settings(nope=1)
    await p.rt.reset()
    assert p.rt.snapshot()["items"] == [] and p.rt.snapshot()["participants"] == [] and p.rt.state.privacy["state"] == "on"
    assert not p.bus.errors


async def test_scenario_api_runs_and_stops(make_pipeline):
    p = await make_pipeline(start=False)
    ids = [s["id"] for s in p.rt.scenarios()]
    assert "demo_three_minutes" in ids and ids[0] == "demo_three_minutes"
    p.rt.settings.speed = 100.0
    await p.rt.run_scenario("reg_05_freezing_fan_off")
    await p.rt._scenario_task
    res = p.rt._runner.last_result
    assert res.ok and res.passed == 1 and p.rt.snapshot()["devices"]
    with pytest.raises(KeyError):
        await p.rt.run_scenario("nope")


# ------------------------------------------------------------------ scenario library
def test_every_library_scenario_is_valid_and_steps_match_the_dsl():
    scenarios = S.load_scenarios()
    assert len(scenarios) >= 13 and scenarios[0]["id"] == "demo_three_minutes"
    for sc in scenarios:
        assert S.validate(sc) == [], (sc["id"], S.validate(sc))
        assert all(step["do"] in SCENARIO_STEPS for step in sc["steps"])
    assert sum(1 for s in scenarios if S.REGRESSION_TAG in s["tags"]) >= 12
    assert S.validate({"id": "bad id", "steps": []}) and "no start step" in " ".join(S.validate(
        {"id": "x", "title": "t", "blurb": "b", "tags": [], "steps": [{"do": "expect", "what": "item"}]}))


@pytest.mark.parametrize("scenario_id", [s["id"] for s in S.load_scenarios()])
async def test_scenario_passes_in_sim_with_the_fake_profile(scenario_id, make_pipeline):
    p = await make_pipeline(start=False)          # LocalRegistry("fake"): the keyword decider
    p.rt.settings.speed = 100.0
    await p.rt.run_scenario(scenario_id)
    await p.rt._scenario_task
    res = p.rt._runner.last_result
    assert res.ok, [c.text for c in res.checks if not c.ok]
    assert not p.bus.errors


async def test_play_cli_runs_one_scenario(capsys):
    from unspoken.meeting.play import play

    rc = await play(["reg_07_slides_dim"], profile="fake", speed=100.0, tag=None, out=None, escalate=True, fake_io=True)
    out = capsys.readouterr().out
    assert rc == 0 and "PASS reg_07_slides_dim" in out and "1/1 scenarios passed" in out


# ------------------------------------------------------------------ replay of the sensors fixture
async def test_replay_meeting3_wav_through_the_pipeline():
    """The sensors workstream's fixture (3 synthetic voices, 12 lines) through the real Silero VAD, a scripted ASR
    and the fake diarizer, into fusion -> decider -> policy -> actuator. No model server, no hardware."""
    np = pytest.importorskip("numpy")  # noqa: F841  (sensors need numpy + onnxruntime)
    pytest.importorskip("onnxruntime")
    from unspoken.sensors import start_sensors
    from unspoken.sensors.fakes import FakeASR, FakeDiarizer, FakeRegistry
    from unspoken.sensors.vad import SileroVAD

    fixtures = Path(__file__).resolve().parents[1] / "sensors" / "fixtures"
    truth = json.loads((fixtures / "meeting3.json").read_text())
    sensors_reg = FakeRegistry(vad=SileroVAD(), asr=FakeASR([l["text"] for l in truth["lines"]]), diarizer=FakeDiarizer())
    rt = MeetingRuntime(profile="fake", mode="sim", registry=make_registry(), echo=False, tick_seconds=0)
    await rt.start(ticking=False)
    await rt.bus.publish(__import__("unspoken.core.events", fromlist=["Event"]).Event(
        MT.STATUS, {"state": "started", "title": "Pricing sync", "owner": "Priya", "agenda": ["Q3 pricing"], "minutes": 30},
        source="test", ts=0.0))
    await rt.bus.drain()
    seen: list[dict] = []

    async def collect(ev):                      # the live stream; history is wiped by "off the record"
        seen.append(ev.data)

    rt.bus.subscribe(MT.UTTERANCE, collect, name="test.collect")
    # This is the in-room happy path. Supply explicit fixture camera evidence:
    # audio-only unknown voices now correctly abstain from claiming identity.
    from unspoken.core.events import Event
    await rt.bus.publish(Event(MT.SPEAKERS, {
        "t": 0.0, "screen_talking": False,
        "faces": [{"track_id": "fixture-room", "seat": "room", "speaking": 0.0}],
    }, source="test", ts=0.0))
    await rt.bus.drain()
    handle = await start_sensors(rt.bus, sensors_reg, {"audio": {"path": str(fixtures / "meeting3.wav")}, "replay": {"realtime": False}},
                                 lambda: 0.0)
    await handle.wait(timeout=60)
    await rt.bus.drain(60)
    await handle.stop()
    utts = seen
    assert len(utts) == 12 and [u["text"] for u in utts] == [l["text"] for l in truth["lines"]]
    # three supported check-ins by voice; the same voice label later carries the name
    names = [u["speaker"]["name"] for u in utts]
    assert names[:3] == ["Priya", "Sam", "Lee"] and names[3] == "Priya" and names[4] == "Sam" and names[8] == "Lee"
    assert sorted(rt.state.names()) == ["Lee", "Priya", "Sam"]
    tasks = [i for i in rt.state.items.values() if i["kind"] == "task"]
    assert any("deck" in t["text"] and t["owner"] == "Sam" and t["due"] for t in tasks)
    assert any(i["kind"] == "decision" and "March" in i["text"] for i in rt.state.items.values())
    assert rt.state.privacy["state"] == "off_record" and rt.state.devices["lamp"]["state"]["scene"] == "off_record"
    assert rt.transcript() == [] or all(u["start"] < rt.state.privacy["wiped_from"] for u in rt.transcript())
    # nothing said in the wiped window survives in the bus history (the clock is fixed at 0 here, so that is all of it)
    wiped_from = rt.state.privacy["wiped_from"]
    assert not [e for e in rt.bus.history if e.topic in (MT.SEGMENT, MT.UTTERANCE) and e.ts >= wiped_from]
    assert rt.state.devices["lamp"]["state"]["scene"] == "off_record" and not rt.bus.errors
    await rt.close()


async def test_off_record_words_never_stay_in_bus_history():
    """Words said before "off the record" (inside the wipe window) are deleted from the bus history, and words said
    while off the record or paused are never kept; on-record words before the window stay."""
    from unspoken.core.events import Event

    rt = MeetingRuntime(profile="fake", mode="sim", registry=make_registry(), echo=False, tick_seconds=0)
    await rt.start(ticking=False)
    rt.settings.wipe_seconds = 60.0
    await rt.bus.publish(Event(MT.STATUS, {"state": "started", "title": "Sync", "owner": "Priya", "agenda": ["A"], "minutes": 30},
                               source="test", ts=0.0))

    async def say(t: float, text: str) -> None:
        await rt.bus.publish(Event(MT.UTTERANCE, {"utterance_id": f"u{t}", "start": t, "end": t + 1, "text": text, "lang": "en",
                                                  "speaker": {"name": "Priya"}, "source": "person"}, source="test", ts=t))
        await rt.bus.drain()

    await say(10.0, "Early on-record line about the agenda.")
    await say(200.0, "The layoffs list is ready.")
    await say(230.0, "Let's go off the record.")
    await say(240.0, "Said while off the record.")
    texts = [e.data.get("text") for e in rt.bus.history if e.topic == MT.UTTERANCE]
    assert "Early on-record line about the agenda." in texts
    assert not {"The layoffs list is ready.", "Let's go off the record.", "Said while off the record."} & set(texts)
    assert rt.state.privacy["state"] == "off_record"
    await rt.close()


async def test_profile_fake_means_the_keyword_sim_profile():
    """`python -m unspoken.meeting --profile fake` resolves to config/models.toml profile `sim`: the meeting's keyword
    decider and no LLM (the models package's own `fake` profile has a generic fake decider + scripted LLM)."""
    from unspoken.meeting.fakes import KeywordDecider
    from unspoken.meeting.runtime import build_registry

    reg = build_registry("fake")
    assert isinstance(reg.decider(), KeywordDecider) and not reg.enabled("llm")
    await reg.aclose()


async def test_the_room_never_escalates_to_a_hosted_llm():
    """Profile dev maps llm to Bedrock for the old home demo; the meeting room treats that as no LLM."""
    from unspoken.meeting.base import local_llm_enabled
    from unspoken.models import load_registry

    dev = load_registry("dev")
    assert dev.spec("llm").provider == "bedrock" and not local_llm_enabled(dev)
    local = load_registry("dev-local")
    assert local_llm_enabled(local)
    await dev.aclose(); await local.aclose()
