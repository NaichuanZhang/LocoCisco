"""validate(scenario) -> list of error strings (empty = valid): file shape, every step against
contracts.SCENARIO_STEPS, people introduced before they speak, an expect after something that can act."""
from __future__ import annotations

import re
from typing import Any

from ..contracts import ACTIONS, SCENARIO_STEPS
from .expectations import EXPECT_KINDS
from .runner import RUNNER_STEPS

_ID = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")
PRIVACY_STATES = ("on", "paused", "off_record")
VERDICTS = ("act", "ask_owner", "deny", "ignore")
ROUTES = ("local-decider", "local-llm", "rule", "user")
PROPOSAL_BY = ("reflex", "decider", "decider_big", "llm", "user", "fake")
CHAT_COMMANDS = ("here", "offrecord", "resume", "recap", "agenda", "undo", "approve", "deny", "ask")
SCENE_MODES = ("talk", "slides", "board", "break", "empty")
TRIGGER_STEPS = frozenset(SCENARIO_STEPS) - RUNNER_STEPS


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate(scenario: Any) -> list[str]:
    err: list[str] = []
    if not isinstance(scenario, dict):
        return ["scenario must be a JSON object"]
    sid = scenario.get("id")
    if not isinstance(sid, str) or not _ID.match(sid):
        err.append(f"id must match {_ID.pattern}, got {sid!r}")
    for key in ("title", "blurb"):
        if not isinstance(scenario.get(key), str) or not scenario.get(key, "").strip():
            err.append(f"{key} must be a non-empty string")
    tags = scenario.get("tags")
    if not isinstance(tags, list) or not all(isinstance(t, str) and t for t in tags):
        err.append("tags must be a list of non-empty strings")
    steps = scenario.get("steps")
    if not isinstance(steps, list) or not steps:
        err.append("steps must be a non-empty list")
        return err
    unknown = set(scenario) - {"id", "title", "blurb", "tags", "steps"}
    if unknown:
        err.append(f"unknown top-level keys: {sorted(unknown)}")
    people: set[str] = set()
    triggered = started = False
    for i, step in enumerate(steps):
        where = f"step {i}"
        if not isinstance(step, dict) or "do" not in step:
            err.append(f"{where}: must be an object with a 'do' key")
            continue
        do = step["do"]
        if do not in SCENARIO_STEPS:
            err.append(f"{where}: unknown do {do!r} (known: {', '.join(sorted(SCENARIO_STEPS))})")
            continue
        where = f"step {i} ({do})"
        if "wait" in step and not (_is_num(step["wait"]) and step["wait"] >= 0):
            err.append(f"{where}: wait must be a number >= 0")
        if do == "start":
            started = True
        elif do in ("join", "enter"):
            if not isinstance(step.get("who"), str):
                err.append(f"{where}: 'who' must be a string")
            else:
                people.add(step["who"])
            if do == "join" and "checkin" in step and step["checkin"] is not None and not isinstance(step["checkin"], str):
                err.append(f"{where}: 'checkin' must be a string or null")
        elif do == "say":
            if step.get("who") not in people:
                err.append(f"{where}: {step.get('who')!r} has not joined or entered yet")
            if not isinstance(step.get("text"), str) or not step["text"].strip():
                err.append(f"{where}: 'text' must be a non-empty string")
            if "lang" in step and step["lang"] not in ("en", "zh"):
                err.append(f"{where}: 'lang' must be en or zh")
        elif do == "media_say":
            if not isinstance(step.get("text"), str) or not step["text"].strip():
                err.append(f"{where}: 'text' must be a non-empty string")
        elif do == "leave":
            if step.get("who") not in people:
                err.append(f"{where}: {step.get('who')!r} is not in the room")
        elif do == "scene":
            if step.get("mode") not in SCENE_MODES:
                err.append(f"{where}: 'mode' must be one of {list(SCENE_MODES)}")
        elif do in ("chat", "react"):
            if step.get("command") not in CHAT_COMMANDS:
                err.append(f"{where}: 'command' must be one of {list(CHAT_COMMANDS)}")
            if "args" in step and not isinstance(step["args"], dict):
                err.append(f"{where}: 'args' must be an object")
        elif do == "clock":
            if not _is_num(step.get("minutes_left")):
                err.append(f"{where}: 'minutes_left' must be a number")
        elif do == "wait":
            if not (_is_num(step.get("seconds")) and step["seconds"] >= 0):
                err.append(f"{where}: 'seconds' must be a number >= 0")
        elif do == "note":
            if not isinstance(step.get("text"), str) or not step["text"].strip():
                err.append(f"{where}: 'text' must be a non-empty string")
        elif do == "expect":
            if not triggered:
                err.append(f"{where}: expect before any world step; nothing can have happened yet")
            _validate_expect(step, where, err)
        if do in TRIGGER_STEPS:
            triggered = True
    if not started:
        err.append("scenario has no start step")
    if not any(isinstance(s, dict) and s.get("do") == "expect" for s in steps):
        err.append("scenario has no expect step")
    return err


def _validate_expect(step: dict[str, Any], where: str, err: list[str]) -> None:
    what = step.get("what")
    if what not in EXPECT_KINDS:
        err.append(f"{where}: 'what' must be one of {list(EXPECT_KINDS)}, got {what!r}")
        return
    if "within" in step and not (_is_num(step["within"]) and step["within"] >= 0.1):
        err.append(f"{where}: 'within' must be a number >= 0.1")
    if what == "item" and "kind" in step and step["kind"] not in ("task", "decision", "question", "parking_lot"):
        err.append(f"{where}: unknown item kind {step['kind']!r}")
    if what == "privacy" and step.get("state") not in PRIVACY_STATES:
        err.append(f"{where}: 'state' must be one of {list(PRIVACY_STATES)}")
    if what == "device":
        if not isinstance(step.get("device_id"), str):
            err.append(f"{where}: 'device_id' must be a string")
        if not isinstance(step.get("state"), dict) or not step["state"]:
            err.append(f"{where}: 'state' must be a non-empty object")
    if what == "proposal":
        for key, allowed in (("by", PROPOSAL_BY), ("action", ACTIONS), ("route", ROUTES), ("mode", ("act", "shadow"))):
            if key in step:
                bad = [v for v in (step[key] if isinstance(step[key], list) else [step[key]]) if v not in allowed]
                if bad:
                    err.append(f"{where}: '{key}' has unknown value(s) {bad}")
    if what == "verdict":
        if "verdict" not in step:
            err.append(f"{where}: 'verdict' is required")
        elif step["verdict"] not in VERDICTS:
            err.append(f"{where}: 'verdict' must be one of {list(VERDICTS)}")
        if "action" in step and step["action"] not in ACTIONS:
            err.append(f"{where}: unknown action {step['action']!r}")
    if what == "screen" and not any(k in step for k in ("banner_contains", "card", "caption_contains", "no_caption_contains", "privacy")):
        err.append(f"{where}: screen expect needs banner_contains / card / caption_contains / no_caption_contains / privacy")
