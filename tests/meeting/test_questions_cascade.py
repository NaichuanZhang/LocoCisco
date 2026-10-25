"""Question library: one request per utterance, interpretation into (action, args, confidence); the cascade:
decider sure -> local-decider, unsure -> the Strands room agent's LLM tool calls, LLM off/failing -> the
unsure decider answer; below ask -> ignored."""
from __future__ import annotations

import datetime as dt
import json

import httpx
import pytest

from unspoken.core.decider import HttpDecider
from unspoken.core.fakes import ScriptedDecider, ScriptedModel
from unspoken.meeting.contracts import ACTIONS, MT
from unspoken.meeting.llm_agent import DIRECT_CONFIDENCE, proposal_from_tool_call
from unspoken.meeting.questions import NO_OWNER, KIND_OPTIONS_FULL, VARIANTS, build_meeting_query, commitment_concern

from .conftest import BrokenDecider, scripted

UTT = {"utterance_id": "u1", "text": "Sam, can you send the deck by Friday?", "lang": "en",
       "speaker": {"name": "Priya", "voice": "voice-A", "seat": "s1", "conf": 0.9, "guest": False}, "source": "person",
       "context": "Pricing sync, topic: Q3 pricing. 22 min left. Here: Priya (owner), Sam."}
NAMES = ["Priya", "Sam"]
MONDAY = dt.date(2026, 10, 26)
XRAY = {"route", "model", "request", "answers", "threshold", "latency_ms", "tokens", "tool_calls", "reply", "by", "mode",
        "kind", "action", "args", "confidence", "probabilities", "reason", "utterance_id", "escalated_from"}


# ------------------------------------------------------------------ build
def test_one_request_with_typed_questions():
    q = build_meeting_query(UTT, NAMES, today=MONDAY)
    assert set(q.questions) == {"kind", "owner", "room", "sensitive"}
    assert q.questions["kind"]["type"] == "choice" and set(q.questions["kind"]["criteria"]) == set(KIND_OPTIONS_FULL)
    assert list(q.questions["owner"]["criteria"]) == ["Priya", "Sam", NO_OWNER]
    assert "the speaker" in q.questions["owner"]["criteria"]["Priya"]
    assert q.questions["room"]["type"] == "choice" and "warmer" in q.questions["room"]["criteria"]
    assert q.questions["sensitive"]["type"] == "noul"
    assert q.state.endswith('Priya said: "Sam, can you send the deck by Friday?"') and q.state.startswith("Pricing sync")


def test_versions_and_overrides():
    v1 = build_meeting_query(UTT, NAMES, "v1")
    assert "sensitive" not in v1.questions and v1.spec is VARIANTS["v1"]
    custom = build_meeting_query(UTT, NAMES, owner_question=False)
    assert "owner" not in custom.questions
    media = build_meeting_query({**UTT, "source": "media"}, NAMES)
    assert "The screen" in media.state


@pytest.mark.parametrize("version", VARIANTS)
@pytest.mark.parametrize("names", [[], ["Alex"]])
async def test_provider_request_has_no_degenerate_owner_choice(version, names):
    """Exercise the HTTP adapter's actual JSON body, without a server or network.

    ScriptedDecider alone accepts a one-option choice; the production provider does not.
    """
    q = build_meeting_query({"text": "I'll prepare a timeline.", "speaker": {}, "source": "person"}, names, version)
    requests = []

    async def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == "/v1/systemone"
        assert ("owner" in body["questions"]) == bool(names)
        for question in body["questions"].values():
            if question["type"] == "choice":
                assert len(question["criteria"]) >= 2
        answers = await scripted("Task", 0.9, owner="Alex").ask(body["state"], body["questions"])
        return httpx.Response(200, json={"answers": answers})

    decider = HttpDecider("request-shape-test", "http://no-network.invalid")
    await decider.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        decider._client = client
        it = q.interpret(await decider.ask(q.state, q.questions))
    assert len(requests) == 1 and it.action == "create_task" and it.owner is None
    assert it.args["owner"] is None
    if names:
        assert set(q.questions["owner"]["criteria"]) == {"Alex", NO_OWNER}


# ------------------------------------------------------------------ interpret
async def test_task_with_owner_and_due_from_code():
    q = build_meeting_query(UTT, NAMES, today=MONDAY)
    answers = await scripted("Task", 0.9, owner="Sam").ask(q.state, q.questions)
    it = q.interpret(answers)
    assert it.action == "create_task" and it.kind == "task" and it.confidence > 0.85
    assert it.args == {"owner": "Sam", "text": UTT["text"], "due": "2026-10-30"}
    assert it.owner == "Sam" and abs(sum(it.probabilities.values()) - 1) < 1e-6 and set(it.probabilities) == set(ACTIONS)


async def test_owner_below_threshold_is_nobody_and_volunteer_is_the_speaker():
    three = NAMES + ["Lin"]
    q = build_meeting_query(UTT, three, today=MONDAY, owner_threshold=0.8)
    answers = await ScriptedDecider({"kind": ("Task", 0.9), "owner": ("Sam", 0.4), "room": ("none", 0.9), "sensitive": 0.05}).ask(q.state, q.questions)
    assert q.interpret(answers).args["owner"] is None
    two = build_meeting_query(UTT, NAMES, today=MONDAY, owner_threshold=0.8)
    answers = await ScriptedDecider({"kind": ("Task", 0.9), "owner": (NO_OWNER, 0.9), "room": ("none", 0.9), "sensitive": 0.05}).ask(two.state, two.questions)
    assert two.interpret(answers).args["owner"] is None        # no two-person inference overriding the model's abstention
    q2 = build_meeting_query({**UTT, "text": "I'll draft the launch email tomorrow.", "speaker": {**UTT["speaker"], "name": "Sam"}}, NAMES, today=MONDAY)
    answers = await scripted("Task", 0.9).ask(q2.state, q2.questions)       # decider says "No one named"
    it = q2.interpret(answers)
    assert it.args["owner"] == "Sam" and it.args["due"] == "2026-10-27"


@pytest.mark.parametrize("text,speaker,names,choice,expected", [
    ("I'll prepare a timeline by Friday.", {}, ["Alex"], "Alex", None),
    ("I'll prepare a timeline by Friday.", {"guest": True}, ["Alex"], "Alex", None),
    ("I'll prepare a timeline by Friday.", {"name": "Alex", "guest": True}, ["Alex"], "Alex", None),
    ("Can you prepare a timeline by Friday?", {"name": "Priya"}, NAMES, "Sam", None),
    ("Can you prepare a timeline by Friday?", {"name": "Priya"}, NAMES, NO_OWNER, None),
    ("Someone should prepare a timeline.", {"name": "Alex"}, ["Alex"], "Alex", None),
    ("Alexandra, prepare a timeline.", {}, ["Alex"], "Alex", None),
    ("ALEX, could you prepare a timeline?", {}, ["Alex"], "Alex", "Alex"),
    ("Sam, would you prepare a timeline?", {"name": "Priya"}, NAMES, "Sam", "Sam"),
    ("I’ll prepare a timeline by Friday.", {"name": "Alex"}, ["Alex"], NO_OWNER, "Alex"),
    ("I'm Alex. I'll prepare a timeline.", {"name": "Alex"}, ["Alex"], NO_OWNER, "Alex"),
])
async def test_owner_needs_utterance_evidence_not_roster_or_context(text, speaker, names, choice, expected):
    q = build_meeting_query({"text": text, "speaker": speaker, "source": "person",
                             "context": "Here: Alex, Priya, Sam. Alex owns the meeting."}, names)
    answers = await scripted("Task", 0.95, owner=choice).ask(q.state, q.questions)
    it = q.interpret(answers)
    assert it.action == "create_task" and it.owner == expected and it.args["owner"] == expected
    if expected is None:
        assert it.owner_confidence == 0.0


async def test_out_of_roster_owner_answer_is_not_trusted_even_when_named():
    q = build_meeting_query({**UTT, "text": "Mallory, prepare a timeline."}, NAMES)
    answers = await scripted("Task", 0.95).ask(q.state, q.questions)
    answers["owner"] = {"choice": "Mallory", "confidence": 0.99, "probabilities": {"Mallory": 0.99, NO_OWNER: 0.01}}
    assert q.interpret(answers).owner is None


async def test_room_request_maps_to_fan_and_light_with_product_confidence():
    q = build_meeting_query({**UTT, "text": "It's freezing in here."}, NAMES)
    it = q.interpret(await scripted("Room request", 0.9, room="warmer").ask(q.state, q.questions))
    assert it.action == "set_fan" and it.args == {"on": False} and 0.75 < it.confidence < 0.9
    it = q.interpret(await scripted("Room request", 0.9, room="dimmer").ask(q.state, q.questions))
    assert it.action == "set_light" and it.args == {"scene": "dim"}
    it = q.interpret(await scripted("Room request", 0.9, room="slides").ask(q.state, q.questions))
    assert it.action == "set_light" and it.args == {"scene": "slides"}
    kind_only = build_meeting_query({**UTT, "text": "It's freezing in here."}, NAMES, room_combine="kind")
    it = kind_only.interpret(await scripted("Room request", 0.9, room="warmer").ask(kind_only.state, kind_only.questions))
    assert it.action == "set_fan" and it.confidence > 0.84      # P(kind=room) minus the small sensitive share


async def test_sensitive_gate_folds_into_pause_notes():
    q = build_meeting_query({**UTT, "text": "Her salary is 120k."}, NAMES, wipe_seconds=45)
    it = q.interpret(await scripted("Nothing", 0.6, sensitive=0.95).ask(q.state, q.questions))
    assert it.action == "pause_notes" and it.args["wipe_seconds"] == 45 and it.confidence > 0.9
    low = q.interpret(await scripted("Nothing", 0.9, sensitive=0.05).ask(q.state, q.questions))
    assert low.action == "none"
    kind_only = build_meeting_query({**UTT, "text": "Her salary is 120k."}, NAMES, sensitive_combine="kind")
    it = kind_only.interpret(await scripted("Nothing", 0.6, sensitive=0.95).ask(kind_only.state, kind_only.questions))
    assert it.action == "none"


async def test_small_talk_and_off_topic_share_the_none_mass():
    q = build_meeting_query({**UTT, "text": "Let's just fire Bob, ha ha."}, NAMES)
    it = q.interpret(await scripted("Small talk or joke", 0.7).ask(q.state, q.questions))
    assert it.action == "none" and it.kind == "small_talk" and it.confidence > 0.7


async def test_docs_sharing_question_is_share_docs():
    q = build_meeting_query({**UTT, "text": "Can you share the pricing doc in the channel?"}, NAMES)
    it = q.interpret(await scripted("Question for the room", 0.9).ask(q.state, q.questions))
    assert it.action == "share_docs" and it.args == {"query": "Can you share the pricing doc in the channel?"}
    q2 = build_meeting_query({**UTT, "text": "What's our refund policy?"}, NAMES)
    assert q2.interpret(await scripted("Question for the room", 0.9).ask(q2.state, q2.questions)).action == "answer_question"


# ------------------------------------------------------------------ cascade
async def test_sure_decider_acts_locally_and_the_llm_is_never_called(make_pipeline):
    model = ScriptedModel([("log_decision", {"text": "wrong"})])      # must never run
    p = await make_pipeline(scripted("Task", 0.95, owner="Sam"), llm_model=model)
    await p.say("priya", "Sam, can you send the deck by Friday?")
    props = p.proposals(by="decider")
    assert len(props) == 1 and model.calls == 0
    d = props[0]
    assert d["mode"] == "act" and d["route"] == "local-decider" and d["action"] == "create_task" and d["args"]["owner"] == "Sam"
    assert d["model"] == "keyword decider (fake)" and d["tokens"] is None and d["tool_calls"] == [] and d["reply"] is None
    assert XRAY <= set(d) and d["threshold"] == 0.6 and d["escalated_from"] is None
    assert d["request"]["questions"]["kind"]["type"] == "choice" and d["answers"]["kind"]["choice"] == "Task"
    assert p.verdicts()[-1]["verdict"] == "act" and p.items()[-1]["owner"] == "Sam"
    assert not p.bus.errors


async def test_unsure_decider_escalates_to_the_local_llm_whose_tool_call_acts(make_pipeline):
    model = ScriptedModel([("create_task", {"text": "Send the deck to the press team", "owner": "Sam", "due": "Friday"})],
                          reply="Task noted.", usage=(700, 30))
    p = await make_pipeline(scripted("Task", 0.5, owner="Sam"), llm_model=model)
    await p.say("priya", "Sam, could you send the deck to the press team by Friday?")
    props = [d for d in p.proposals() if d["by"] in ("decider", "llm")]
    assert [(d["by"], d["mode"], d["route"]) for d in props] == [("decider", "shadow", "local-decider"), ("llm", "act", "local-llm")]
    shadow, act = props
    assert "asking" in shadow["reason"] and shadow["tokens"] is None
    assert act["action"] == "create_task" and act["args"]["owner"] == "Sam" and act["args"]["due"]
    assert act["tokens"] == {"input": 1400, "output": 60} and act["model"] == "scripted LLM" and act["reply"] == "Task noted."
    assert act["confidence"] == DIRECT_CONFIDENCE and act["confidence_source"] == "adapter_constant"
    assert act["confidence_calibrated"] is False and act["probabilities"] == {}
    assert "not a model probability" in act["reason"]
    assert act["tool_calls"] == [{"name": "create_task", "input": {"text": "Send the deck to the press team", "owner": "Sam", "due": "Friday"}}]
    assert act["escalated_from"] == p.events(MT.PROPOSAL, mode="shadow")[0].id and model.calls == 2
    assert p.verdicts(action="create_task")[-1]["verdict"] == "act"
    assert p.items()[-1]["text"] == "Sam, could you send the deck to the press team by Friday?"
    assert p.items()[-1]["owner"] == "Sam"
    prompt = model.requests[0]["messages"][0]["content"][0]["text"]
    assert prompt.startswith("Overheard.") and "Priya" in prompt and "deck" in prompt
    assert "create_task" in model.requests[0]["tools"] and "no_action" in model.requests[0]["tools"]
    assert "Priya, Sam" in model.requests[0]["system_prompt"]


async def test_llm_no_action_is_a_none_proposal_and_ignored(make_pipeline):
    model = ScriptedModel([("no_action", {})], reply="")
    p = await make_pipeline(scripted("Task", 0.5), llm_model=model)
    await p.say("sam", "We sent decks like that last year.")
    act = [d for d in p.proposals() if d["by"] == "llm"][-1]
    assert act["action"] == "none" and act["tool_calls"] == [{"name": "no_action", "input": {}}]
    assert p.verdicts(proposal_id=p.events(MT.PROPOSAL, by="llm")[-1].id)[0]["verdict"] == "ignore" and not p.items()


async def test_llm_off_uses_the_unsure_decider_answer(make_pipeline):
    p = await make_pipeline(scripted("Task", 0.5, owner="Sam"))             # llm role disabled in the fake registry
    await p.say("priya", "Sam, maybe the deck by Friday?")
    props = p.proposals(by="decider")
    assert len(props) == 1 and props[0]["mode"] == "act" and "LLM is off" in props[0]["reason"]
    assert p.verdicts()[-1]["verdict"] == "ask_owner"                     # medium risk at 50% (ask..act band): the owner decides
    assert props[0]["safety"] == {"reason": "tentative commitment", "outcome": "owner_review",
                                  "proposed_action": "create_task", "proposed_args": props[0]["args"]}
    assert p.notifier.dms and "Priya" == p.notifier.dms[0]["user"]


async def test_escalate_false_never_builds_the_room_agent(make_pipeline):
    model = ScriptedModel([("create_task", {"text": "x"})])
    p = await make_pipeline(scripted("Task", 0.5), llm_model=model, escalate=False)
    await p.say("priya", "Sam, maybe the deck by Friday?")
    assert model.calls == 0 and p.proposals(by="decider")[-1]["mode"] == "act"


async def test_llm_failure_falls_back_to_the_unsure_decider(make_pipeline):
    model = ScriptedModel(fail=RuntimeError("vLLM is down"))
    p = await make_pipeline(scripted("Task", 0.5, owner="Sam"), llm_model=model)
    await p.say("priya", "Sam, maybe the deck by Friday?")
    props = [d for d in p.proposals() if d["by"] in ("decider", "llm")]
    assert [d["mode"] for d in props] == ["shadow", "act"] and props[1]["by"] == "decider"
    assert "unavailable" in props[1]["reason"] and any("vLLM is down" in w for w in p.logs("warn"))


@pytest.mark.parametrize("use_llm", [False, True])
async def test_unknown_voice_with_one_roster_member_never_assigns_that_member(make_pipeline, use_llm):
    model = ScriptedModel([("create_task", {"text": "Alex will prepare the timeline", "owner": "Alex"})])
    dec = scripted("Task", 0.5 if use_llm else 0.95, owner="Alex")
    p = await make_pipeline(dec, start=False, llm_model=model if use_llm else None)
    await p.steps({"do": "start", "owner": "Alex", "title": "Planning", "minutes": 30},
                  {"do": "join", "who": "alex", "seat": "s1", "voice": "voice-A", "checkin": "Alex"})
    assert p.state.names() == ["Alex"]
    before = len(dec.calls)
    await p.publish(MT.UTTERANCE, {"utterance_id": "unknown-voice", "text": "I'll prepare a timeline by Friday.",
                                  "speaker": {"guest": True}, "source": "person", "context": "Here: Alex."})
    prop = p.proposals(by="llm" if use_llm else "decider", mode="act")[-1]
    assert len(dec.calls) == before + 1
    assert prop["action"] == "create_task" and prop["args"]["owner"] is None
    assert prop["owner"] is None and prop["owner_evidence"] is None
    assert "owner unassigned" in prop["reason"]
    assert p.items()[-1]["owner"] is None and p.items()[-1]["text"] == "I'll prepare a timeline by Friday."
    assert not p.bus.errors
    if use_llm:
        assert model.calls == 2 and prop["tool_calls"][0]["input"]["owner"] == "Alex"


@pytest.mark.parametrize("text,concern", [
    ("I won't prepare a timeline by Friday.", "negated commitment"),
    ("I will not prepare a timeline by Friday.", "negated commitment"),
    ("I can’t prepare a timeline by Friday.", "negated commitment"),
    ("I'm not going to prepare a timeline.", "negated commitment"),
    ("I'm not preparing a timeline.", "negated commitment"),
    ("Sam, don't prepare a timeline.", "negated commitment"),
    ("Cancel that task to prepare a timeline.", "cancelled commitment"),
    ("My commitment to prepare a timeline is cancelled.", "cancelled commitment"),
    ("Hypothetically, I could prepare a timeline.", "hypothetical commitment"),
    ("If we had time, I would prepare a timeline.", "hypothetical commitment"),
    ("What if Sam prepared a timeline?", "hypothetical commitment"),
    ("Maybe we should prepare a timeline by Friday.", "tentative commitment"),
    ("Sam might prepare a timeline.", "tentative commitment"),
])
async def test_sure_task_classification_is_preserved_but_unsafe_commitment_abstains(make_pipeline, text, concern):
    dec = scripted("Task", 0.95, owner="Sam")
    p = await make_pipeline(dec)
    before = len(dec.calls)
    await p.say("sam", text)
    d = p.proposals(by="decider", mode="act")[-1]
    assert len(dec.calls) == before + 1 and d["answers"]["kind"]["choice"] == "Task"
    assert d["action"] == "none" and d["args"] == {}
    assert d["safety"]["reason"] == concern and d["safety"]["outcome"] == "abstain"
    assert d["safety"]["proposed_action"] == "create_task" and concern in d["reason"]
    assert d["confidence"] > 0.6 and d["confidence"] == d["probabilities"]["create_task"]
    assert d["confidence_action"] == "create_task" and d["confidence_source"] == "decider_probability_mass"
    assert p.verdicts()[-1]["verdict"] == "ignore" and not p.items() and not p.notifier.dms
    assert not p.bus.errors


@pytest.mark.parametrize("text", [
    "I won't prepare a timeline by Friday.",
    "Maybe we should prepare a timeline by Friday.",
    "If we had time, I would prepare a timeline.",
    "The task to prepare a timeline is cancelled.",
])
async def test_llm_cannot_turn_noncommitment_into_a_task_after_decider_none(make_pipeline, text):
    model = ScriptedModel([("create_task", {"text": "Prepare a timeline", "owner": "Sam", "due": "Friday"})])
    # A middle-band NONE still escalates; the guard must inspect the utterance, not just the
    # decider's selected action or the LLM's rewritten (positive) task text.
    dec = scripted("Nothing to do", 0.45)           # v3 label; NONE (including room=none) is in the ask band
    p = await make_pipeline(dec, llm_model=model)
    before = len(dec.calls)
    await p.say("sam", text)
    shadow = p.proposals(by="decider", mode="shadow")[-1]
    act = p.proposals(by="llm", mode="act")[-1]
    assert shadow["action"] == "none" and 0.4 <= shadow["confidence"] < 0.6
    assert len(dec.calls) == before + 1 and model.calls == 2
    assert act["action"] == "none" and act["safety"]["outcome"] == "abstain"
    assert act["safety"]["reason"] == commitment_concern(text)
    assert act["tool_calls"][0]["name"] == "create_task"
    assert act["confidence"] == DIRECT_CONFIDENCE and act["confidence_calibrated"] is False
    assert act["confidence_source"] == "adapter_constant" and act["probabilities"] == {}
    assert p.verdicts()[-1]["verdict"] == "ignore" and not p.items() and not p.notifier.dms
    assert not p.bus.errors


async def test_negated_task_is_still_blocked_when_llm_fails(make_pipeline):
    p = await make_pipeline(scripted("Task", 0.5), llm_model=ScriptedModel(fail=RuntimeError("offline")))
    await p.say("sam", "I won't prepare a timeline.")
    act = p.proposals(by="decider", mode="act")[-1]
    assert act["action"] == "none" and act["safety"]["reason"] == "negated commitment"
    assert "unavailable" in act["reason"] and not p.items()


async def test_tentative_review_cannot_round_up_into_automatic_action(make_pipeline):
    # v3's neutral act gate and 0.05 sensitivity yield P(task)=0.59999. Publication rounds
    # that to 0.6000; the guard must use the same score the policy actually receives.
    p = await make_pipeline(scripted("Task", (0.6 - 0.00001) / 0.95))
    await p.say("sam", "Maybe we should prepare a timeline.")
    act = p.proposals(by="decider", mode="act")[-1]
    assert act["confidence"] == act["threshold"] == 0.6
    assert act["action"] == "none" and act["safety"]["outcome"] == "abstain"
    assert not p.items() and not p.notifier.dms


async def test_negated_sharing_cannot_bypass_guard_via_task_to_docs_mapping(make_pipeline):
    p = await make_pipeline(scripted("Task", 0.95, owner="Sam"))
    await p.say("sam", "I won't share the timeline doc in the channel.")
    act = p.proposals(by="decider", mode="act")[-1]
    assert act["safety"]["proposed_action"] == "share_docs"
    assert act["action"] == "none" and not p.items() and not p.posts("docs")


@pytest.mark.parametrize("text", [
    "Sam, drop that document in the channel.",
    "Sam, cancel that appointment by Friday.",
    "I'll not only prepare a timeline, but also send it.",
])
def test_commitment_guard_is_not_a_blanket_keyword_veto(text):
    assert commitment_concern(text) is None


@pytest.mark.parametrize("use_llm", [False, True])
@pytest.mark.parametrize("text", [
    "Sam, don't forget to prepare a timeline by Friday.",
    "Sam, do not forget to prepare a timeline by Friday.",
    "Sam, could you prepare a timeline by Friday?",
    "Sam, would you please prepare a timeline by Friday?",
    "Sam, could you prepare a timeline if you have time?",
    "Sam, please prepare a timeline if you have time.",
])
async def test_positive_reminders_and_explicit_requests_still_create_tasks(make_pipeline, text, use_llm):
    model = ScriptedModel([("create_task", {"text": "Prepare a timeline", "owner": "Sam", "due": "Friday"})])
    p = await make_pipeline(scripted("Task", 0.5 if use_llm else 0.95, owner="Sam"),
                            llm_model=model if use_llm else None)
    await p.say("priya", text)
    act = p.proposals(by="llm" if use_llm else "decider", mode="act")[-1]
    assert act["action"] == "create_task" and act["args"]["owner"] == "Sam" and "safety" not in act
    assert act["owner_evidence"] == "explicit_name" and p.items()[-1]["owner"] == "Sam"
    assert not p.bus.errors


async def test_below_ask_threshold_is_published_and_ignored_without_the_llm(make_pipeline):
    model = ScriptedModel([("create_task", {"text": "x"})])
    p = await make_pipeline(scripted("Task", 0.3), llm_model=model)
    await p.say("sam", "The six-week test showed no churn change.")
    d = p.proposals(by="decider")[-1]
    assert model.calls == 0 and d["mode"] == "act" and "too unsure" in d["reason"]
    assert p.verdicts()[-1]["verdict"] == "ignore" and not p.items()


async def test_decider_failure_logs_and_the_pipeline_goes_on(make_pipeline):
    p = await make_pipeline(BrokenDecider())
    await p.say("sam", "It's freezing in here.")
    assert any("decider failed" in w for w in p.logs("warn")) and not p.bus.errors
    await p.say("priya", "Off the record for a minute.")                  # reflexes still work
    assert p.state.privacy["state"] == "off_record"


async def test_check_ins_and_reflex_phrases_skip_the_decider(make_pipeline):
    dec = scripted("Task", 0.95, owner="Sam")
    p = await make_pipeline(dec)
    before = len(dec.calls)
    await p.say("priya", "Off the record for a minute.")
    await p.say("priya", "Back on the record.")
    await p.steps({"do": "join", "who": "lin", "seat": "s3", "voice": "voice-C", "checkin": "Lin"})
    assert len(dec.calls) == before


async def test_media_words_are_interpreted_but_denied(make_pipeline):
    p = await make_pipeline(scripted("Task", 0.95, owner="Sam"))
    await p.steps({"do": "media_say", "text": "Sam, send the deck by Friday."})
    v = p.verdicts(action="create_task")[-1]
    assert v["verdict"] == "deny" and "screen" in v["rule"] and not p.items()


async def test_short_intro_with_task_still_reaches_planner(make_pipeline):
    dec = scripted("Task", 0.95, owner="Sam")
    p = await make_pipeline(dec)
    before = len(dec.calls)
    await p.say("sam", "I'm Sam. I'll send it.")
    assert len(dec.calls) == before + 1
    assert p.proposals(action="create_task")


def test_actual_decider_wire_probabilities_do_not_imply_a_task():
    # Observed AMI utterance/answers (public replay); the wire's choice confidence is
    # chance-adjusted, not P(choice). The existing interpreter correctly uses probabilities.
    q = build_meeting_query({
        "text": "Yeah, it's functional yeah, exactly. Okay, so we looking for a working design when we come back.",
        "source": "person", "speaker": {"guest": True},
    }, [], "v3")
    result = q.interpret({
        "act": {"type": "noul", "noul": 0.3039},
        "kind": {"type": "choice", "choice": "Nothing to do", "confidence": 0.7943,
                 "probabilities": {"Task": 0.0273, "Decision": 0.1067, "Question for the room": 0.0131,
                                   "Room request": 0.0133, "Sensitive": 0.0109, "Nothing to do": 0.8286}},
        "sensitive": {"type": "noul", "noul": 0.0433},
    })
    assert result.action == "none" and result.confidence > 0.8
    assert round(result.confidence, 4) == 0.8777
    assert abs(sum(result.probabilities.values()) - 1) < 1e-6


def test_llm_tool_args_parse_string_booleans():
    """Small local models send "false" as a string; bool("false") is True, which once turned the fan ON for
    "it's freezing"."""
    from unspoken.meeting.llm_agent import as_bool

    assert as_bool(False, default=True) is False and as_bool("false", default=True) is False
    assert as_bool("off", default=True) is False and as_bool(0, default=True) is False
    assert as_bool("true", default=False) is True and as_bool(None, default=True) is True


@pytest.mark.parametrize("text,speaker,proposed,expected", [
    ("I'll prepare a timeline.", {}, "Alex", None),
    ("I'll prepare a timeline.", {"name": "Sam"}, "Alex", None),
    ("Can you prepare a timeline?", {"name": "Sam"}, "Alex", None),
    ("Alexandra, prepare a timeline.", {}, "Alex", None),
    ("ALEX, prepare a timeline.", {}, "Alex", "Alex"),
    ("I’ll prepare a timeline.", {"name": "Alex"}, "Alex", "Alex"),
    ("Mallory, prepare a timeline.", {}, "Mallory", None),
])
def test_llm_owner_grounding_uses_original_utterance_not_generated_task(text, speaker, proposed, expected):
    q = build_meeting_query({"text": text, "speaker": speaker, "source": "person"}, ["Alex", "Sam"])
    action, args, score = proposal_from_tool_call(
        {"name": "create_task", "input": {"text": f"{proposed} will prepare the timeline", "owner": proposed}}, q)
    assert action == "create_task" and args["owner"] == expected and score == DIRECT_CONFIDENCE
    if expected is None:
        assert args["text"] == text


@pytest.mark.parametrize("speaker,expected_decider", [({}, None), ({"name": "Sam"}, "Sam")])
async def test_self_commitment_never_assigns_a_named_recipient(speaker, expected_decider):
    q = build_meeting_query({"text": "I'll send the notes to Alex by Friday.", "speaker": speaker,
                             "source": "person"}, ["Alex", "Sam"], today=MONDAY)
    answers = await scripted("Task", 0.95, owner="Alex").ask(q.state, q.questions)
    assert q.interpret(answers).owner == expected_decider
    assert q.owner_evidence("Alex") is None
    _, args, _ = proposal_from_tool_call(
        {"name": "create_task", "input": {"owner": "Alex", "text": "Send notes", "due": "Friday"}}, q)
    assert args["owner"] is None                          # do not trust the LLM's named recipient either
    if expected_decider:
        _, args, _ = proposal_from_tool_call(
            {"name": "create_task", "input": {"owner": "Sam", "text": "Send notes"}}, q)
        assert args["owner"] == "Sam"


@pytest.mark.parametrize("suffix,generated_due,expected", [
    (".", "Friday", None),
    (" by Tuesday.", "Friday", "2026-10-27"),
    (" by Friday.", "", "2026-10-30"),
    (" tomorrow.", "Friday", "2026-10-27"),
    (" in 3 days.", "Friday", "2026-10-29"),
    (" by 2026-11-05.", "Friday", "2026-11-05"),
])
def test_llm_task_text_and_due_are_source_grounded(suffix, generated_due, expected):
    text = "Sam, prepare a timeline" + suffix
    q = build_meeting_query({**UTT, "text": text}, NAMES, today=MONDAY)
    _, args, _ = proposal_from_tool_call(
        {"name": "create_task", "input": {"owner": "Sam", "text": "Prepare a timeline by Friday",
                                         "due": generated_due}}, q)
    assert args == {"owner": "Sam", "text": text, "due": expected}


async def test_llm_cannot_publish_a_hallucinated_deadline_even_with_grounded_owner(make_pipeline):
    text = "Sam, please prepare a timeline."
    model = ScriptedModel([("create_task", {"owner": "Sam", "text": "Prepare a timeline by Friday", "due": "Friday"})])
    p = await make_pipeline(scripted("Task", 0.5, owner="Sam"), llm_model=model)
    await p.say("priya", text)
    prop = p.proposals(by="llm", mode="act")[-1]
    assert prop["args"] == {"owner": "Sam", "text": text, "due": None}
    assert prop["tool_calls"][0]["input"]["due"] == "Friday"     # raw rejected deadline remains diagnostic
    assert p.items()[-1]["due"] is None and p.items()[-1]["text"] == text


@pytest.mark.parametrize("use_llm", [False, True])
@pytest.mark.parametrize("text,room,tool,args,action", [
    ("Brightness.", "brighter", "set_light", {"scene": "bright"}, "set_light"),
    ("Display brightness.", "brighter", "set_light", {"scene": "bright"}, "set_light"),
    ("Very bright.", "brighter", "set_light", {"scene": "bright"}, "set_light"),
    ("Cold.", "warmer", "warmer", {}, "set_fan"),
    ("Fan speed.", "cooler", "cooler", {}, "set_fan"),
    ("Slides.", "slides", "show", {"card": "agenda"}, "set_light"),
])
async def test_room_fragments_abstain_after_both_models_without_hiding_scores(
        make_pipeline, use_llm, text, room, tool, args, action):
    dec = scripted("Room request", 0.5 if use_llm else 0.95, room=room)
    model = ScriptedModel([(tool, args)])
    p = await make_pipeline(dec, llm_model=model if use_llm else None)
    before_calls, before_commands = len(dec.calls), len(p.events(MT.COMMAND))
    await p.say("sam", text)
    prop = p.proposals(by="llm" if use_llm else "decider", mode="act")[-1]
    original_action = "show" if use_llm and tool == "show" else action
    assert prop["action"] == "none" and prop["args"] == {}
    assert prop["safety"]["outcome"] == "abstain"
    assert prop["safety"]["reason"] == "room fragment without request or comfort evidence"
    assert prop["safety"]["proposed_action"] == prop["confidence_action"] == original_action
    assert prop["confidence"] >= 0.6 and "abstained" in prop["reason"]
    assert prop["answers"]["kind"]["choice"] == "Room request" and prop["answers"]["room"]["choice"] == room
    assert len(dec.calls) == before_calls + 1 and len(p.events(MT.COMMAND)) == before_commands
    assert p.verdicts()[-1]["verdict"] == "ignore" and not p.bus.errors
    if use_llm:
        assert model.calls == 2 and prop["tool_calls"] == [{"name": tool, "input": args}]
    else:
        assert prop["confidence"] == prop["probabilities"][action]


@pytest.mark.parametrize("use_llm", [False, True])
@pytest.mark.parametrize("text,room,tool,args", [
    ("It's freezing in here.", "warmer", "warmer", {}),
    ("I'm cold.", "warmer", "warmer", {}),
    ("Too cold.", "warmer", "warmer", {}),
    ("The room is hot.", "cooler", "cooler", {}),
    ("It's getting really stuffy.", "cooler", "cooler", {}),
    ("Turn the fan off.", "warmer", "warmer", {}),
    ("Can we dim the lights?", "dimmer", "set_light", {"scene": "dim"}),
    ("Brighter, please.", "brighter", "set_light", {"scene": "bright"}),
    ("Next slide.", "slides", "set_light", {"scene": "slides"}),
    ("Show slides.", "slides", "show", {"card": "agenda"}),
    ("把灯调亮一点。", "brighter", "set_light", {"scene": "bright"}),
])
async def test_room_requests_comfort_and_slide_cues_still_act(make_pipeline, use_llm, text, room, tool, args):
    p = await make_pipeline(scripted("Room request", 0.5 if use_llm else 0.95, room=room),
                            llm_model=ScriptedModel([(tool, args)]) if use_llm else None)
    await p.say("sam", text)
    prop = p.proposals(by="llm" if use_llm else "decider", mode="act")[-1]
    assert prop["action"] in ("set_fan", "set_light", "show") and "safety" not in prop
    assert p.verdicts()[-1]["verdict"] == "act" and not p.bus.errors


def test_llm_comfort_tools_map_to_set_fan():
    from unspoken.meeting.llm_agent import make_meeting_tools, proposal_from_tool_call

    names = {getattr(t, "tool_name", getattr(t, "__name__", "")) for t in make_meeting_tools()}
    assert {"warmer", "cooler"} <= names and "set_fan" not in names
    q = type("Q", (), {"names": [], "today": None, "text": "", "wipe_seconds": 60.0})()
    assert proposal_from_tool_call({"name": "warmer", "input": {}}, q)[:2] == ("set_fan", {"on": False})
    assert proposal_from_tool_call({"name": "cooler", "input": {}}, q)[:2] == ("set_fan", {"on": True})
