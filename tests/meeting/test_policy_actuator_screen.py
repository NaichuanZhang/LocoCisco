"""Policy table, actuator (items, posts, approvals, undo, devices, egress meter) and screen state."""
from __future__ import annotations

from unspoken.meeting.base import MeetingState
from unspoken.meeting.config import load_config, settings_from
from unspoken.meeting.contracts import ACTIONS, MT
from unspoken.meeting.policy import RISK, PolicyAgent

from .conftest import scripted


# ------------------------------------------------------------------ policy (pure)
def judge(state: MeetingState, action: str, conf: float = 0.9, *, source: str = "person", route: str = "local-decider",
          args: dict | None = None, **changes):
    cfg = load_config()
    settings = settings_from(cfg, **changes)
    policy = PolicyAgent(None, state, settings, cfg)
    p = {"action": action, "args": args or {}, "confidence": conf, "route": route, "mode": "act", "utterance_id": "u1"}
    return policy.judge(p, {"source": source, "speaker": {"name": "Priya"}})


def test_risk_table_covers_every_action():
    assert set(RISK) == set(ACTIONS)
    assert RISK["post_recap"] == "high" and RISK["pause_notes"] == "low" and RISK["create_task"] == "medium"


def test_policy_rules_in_order():
    s = MeetingState()
    assert judge(s, "none")[0] == "ignore"
    assert judge(s, "create_task", source="media")[0] == "deny"
    assert judge(s, "pause_notes", conf=0.3)[0] == "act"                     # privacy first, whatever the confidence
    assert judge(s, "create_task")[0] == "act" and judge(s, "create_task")[1] == "medium"
    assert judge(s, "post_recap")[0] == "ask_owner"
    assert judge(s, "share_docs")[0] == "act"
    s.people["g"] = __import__("unspoken.meeting.base", fromlist=["Person"]).Person(key="g", seat="s4", guest=True)
    v, risk, rule = judge(s, "share_docs")
    assert v == "deny" and "guest" in rule and risk == "high"
    assert judge(s, "post_recap")[0] == "ask_owner"
    s.people.clear()
    s.privacy["state"] = "off_record"
    assert judge(s, "create_task")[0] == "deny" and judge(s, "set_fan")[0] == "deny" and judge(s, "resume_notes")[0] == "act"
    s.privacy["state"] = "paused"
    assert judge(s, "create_task")[0] == "deny" and judge(s, "set_fan")[0] == "act"
    s.sensitive_pause = True
    assert judge(s, "resume_notes")[0] == "ask_owner"
    s.sensitive_pause = False
    assert judge(s, "resume_notes")[0] == "act"
    s.privacy["state"] = "on"
    assert judge(s, "set_light", conf=0.5)[0] == "act"                       # low risk, likely enough
    assert judge(s, "create_task", conf=0.5)[0] == "ask_owner"               # medium risk, unsure: the owner decides
    assert judge(s, "create_task", conf=0.3)[0] == "ignore"
    assert judge(s, "set_light", conf=0.2, route="rule")[0] == "act"         # reflexes bypass the bands
    assert judge(s, "create_task", conf=0.85, act_threshold=0.9)[0] == "ask_owner"


# ------------------------------------------------------------------ actuator
async def test_task_is_posted_and_metered(make_pipeline):
    p = await make_pipeline(scripted("Task", 0.95, owner="Sam"))
    await p.say("priya", "Sam, can you send the deck by Friday?")
    item = p.items()[-1]
    assert item["kind"] == "task" and item["owner"] == "Sam" and item["status"] == "open" and item["due"]
    post = p.posts("tasks")[-1]
    assert "Sam" in post["text"] and post["meta"]["item_id"] == item["item_id"] and item["message"]["id"] == post["id"]
    eg = p.events(MT.EGRESS)[-1].data
    assert eg["kind"] == "text" and eg["total_bytes"] == len(post["text"].encode()) and eg["audio_bytes_out"] == 0 and eg["video_bytes_out"] == 0
    assert p.rt.items(kind="task") == [item] and p.rt.snapshot()["egress"]["total_bytes"] == eg["total_bytes"]


async def test_undo_removes_the_last_task(make_pipeline):
    p = await make_pipeline(scripted("Task", 0.95, owner="Sam"))
    await p.say("priya", "Sam, can you send the deck by Friday?")
    await p.steps({"do": "react", "command": "undo", "user": "Sam", "args": {"item_id": "$last_task"}})
    assert p.items()[-1]["status"] == "undone" and "removed" in p.posts("tasks")[-1]["text"]
    assert not p.state.open_items("task") and not any(c.get("card") == "task" for c in p.state.screen["cards"])


async def test_high_risk_asks_the_owner_and_approve_executes(make_pipeline):
    p = await make_pipeline(scripted("Task", 0.5, owner="Sam"))               # medium risk at 50% -> ask the owner
    await p.say("priya", "Sam, maybe the deck by Friday?")
    assert p.verdicts()[-1]["verdict"] == "ask_owner"
    dm = p.notifier.dms[-1]
    assert dm["user"] == "Priya" and "request" in dm["text"] and dm["meta"]["action"] == "create_task"
    pending = p.items()[-1]
    assert pending["status"] == "pending_approval" and any(c["card"] == "approval" for c in p.state.screen["cards"])
    await p.steps({"do": "chat", "command": "approve", "user": "Priya", "args": {"request_id": dm["meta"]["request_id"]}})
    assert p.state.items[pending["item_id"]]["status"] == "open"
    assert any("approved" in t for t in p.logs())


async def test_deny_drops_the_pending_task(make_pipeline):
    p = await make_pipeline(scripted("Task", 0.5, owner="Sam"))
    await p.say("priya", "Sam, maybe the deck by Friday?")
    rid = p.notifier.dms[-1]["meta"]["request_id"]
    await p.steps({"do": "chat", "command": "deny", "user": "Priya", "args": {"request_id": rid}})
    assert p.items()[-1]["status"] == "undone" and not p.state.open_items("task")


async def test_recap_chat_posts_the_template_recap(make_pipeline):
    p = await make_pipeline(scripted("Decision", 0.95))
    await p.say("priya", "Let's go with the thirty-nine dollar Starter price.")
    await p.steps({"do": "chat", "command": "recap", "user": "Priya"})
    recap = p.posts("notes")[-1]["text"]
    assert recap.startswith("**Recap: Pricing sync**") and "thirty-nine" in recap and "**Decisions**" in recap
    assert any(c["card"] == "posted" for c in p.state.screen["cards"])
    assert (await p.rt.recap()) == recap


async def test_room_requests_drive_fan_and_lamp(make_pipeline):
    p = await make_pipeline(scripted("Room request", 0.95, room="warmer"))
    await p.say("sam", "It's freezing in here.")
    assert p.state.devices["fan"]["state"] == {"on": False}
    cmd = p.events(MT.COMMAND, target="fan")[-1].data
    assert cmd["command"] == "off" and cmd["verdict_id"]
    p2 = await make_pipeline(scripted("Room request", 0.95, room="dimmer"))
    await p2.say("sam", "Can we dim the lights?")
    assert p2.state.devices["lamp"]["state"]["scene"] == "dim"


async def test_question_answers_from_local_docs(make_pipeline):
    p = await make_pipeline(scripted("Question for the room", 0.95))
    await p.say("lin", "What's our refund policy for annual plans?") if False else await p.say("sam", "What's our refund policy for annual plans?")
    card = next(c for c in p.state.screen["cards"] if c["card"] == "answer")
    assert "refund-policy.md" == card["doc"] and "annual" in card["text"].lower()
    assert p.items()[-1]["kind"] == "question" and "refund" in p.posts("notes")[-1]["text"].lower()
    hits = await p.rt.search_docs("refund annual plans")
    assert hits and hits[0]["doc"] == "refund-policy.md" and hits[0]["method"] == "keyword"


async def test_privacy_changes_the_lamp_and_meeting_end_breaks(make_pipeline):
    p = await make_pipeline()
    await p.steps({"do": "chat", "command": "offrecord", "user": "Priya"})
    assert p.state.devices["lamp"]["state"]["scene"] == "off_record"
    await p.steps({"do": "chat", "command": "resume", "user": "Priya"})
    assert p.state.devices["lamp"]["state"]["scene"] == "talk"
    await p.steps({"do": "end"})
    assert p.state.devices["lamp"]["state"]["scene"] == "break"


async def test_notifier_failure_is_logged_not_fatal(make_pipeline):
    p = await make_pipeline(scripted("Task", 0.95, owner="Sam"))
    p.notifier.fail = ConnectionError("openclaw down")
    await p.say("priya", "Sam, can you send the deck by Friday?")
    assert p.items()[-1]["message"] is None and any("openclaw down" in w for w in p.logs("warn")) and not p.bus.errors


# ------------------------------------------------------------------ screen
async def test_captions_are_named_and_bounded(make_pipeline):
    p = await make_pipeline(caption_lines=2)
    await p.say("priya", "One.")
    await p.say("sam", "Two.")
    await p.say("priya", "Three.")
    caps = p.state.screen["captions"]
    assert [(c["name"], c["text"]) for c in caps] == [("Sam", "Two."), ("Priya", "Three.")] and all(c["utterance_id"] for c in caps)
    scr = p.events(MT.SCREEN)[-1].data
    assert scr["privacy"] == "on" and scr["agenda"]["topic"] == "Q3 pricing" and scr["agenda"]["minutes_total"] == 30


async def test_banner_counts_tasks_without_owner_and_catch_up_card(make_pipeline):
    p = await make_pipeline(scripted("Task", 0.95))
    await p.say("priya", "Someone needs to update the pricing page.")
    assert p.items()[-1]["owner"] is None
    await p.steps({"do": "clock", "minutes_left": 2})
    assert p.state.screen["banner"] == "2 min left. 1 task has no owner."
    assert p.state.screen["agenda"]["minutes_left"] == 2
    await p.steps({"do": "chat", "command": "here", "user": "Alex", "args": {"name": "Alex"}})
    card = next(c for c in p.state.screen["cards"] if c["card"] == "catch_up")
    assert card["for"] == "Alex" and card["lines"] and card["topic"] == "Q3 pricing"


async def test_task_and_decision_cards_newest_first(make_pipeline):
    p = await make_pipeline(scripted("Task", 0.95, owner="Sam"))
    await p.say("priya", "Sam, can you send the deck by Friday?")
    await p.say("priya", "Sam, can you also book the room for Monday?")
    cards = p.state.screen["cards"]
    assert [c["card"] for c in cards][:2] == ["task", "task"] and "book the room" in cards[0]["text"] and cards[0]["owner"] == "Sam"
