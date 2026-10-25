"""validate(scenario, home) -> list of error strings (empty = valid).

Checks the file shape, every step against contracts.SCENARIO_STEPS, and every id (rooms, devices,
doors, residents, pets) against home.json. People introduced by an earlier `arrive` step count as
known from then on. An `expect` must come after something that can make the pipeline act.
"""
from __future__ import annotations

import re
from typing import Any

from ..contracts import ACTIONS, SCENARIO_STEPS
from .expectations import EXPECT_KINDS, OUTCOME_ACCEPT, PROPOSAL_BY, VERDICTS, DECIDER_WILDCARD
from .homeindex import HomeIndex
from .steps import RUNNER_STEPS, TRIGGER_STEPS, WEEKDAYS

_ID = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")
_TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_LANGS = ("en", "zh")
_MODES = ("act", "shadow")
_KINDS = ("speech", "scene")


class _Ctx:
    def __init__(self, home: HomeIndex) -> None:
        self.home = home
        self.people: set[str] = set(home.residents) | set(home.pets)
        self.triggered = False
        self.errors: list[str] = []


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate(scenario: Any, home: dict[str, Any] | HomeIndex) -> list[str]:
    index = home if isinstance(home, HomeIndex) else HomeIndex(home)
    ctx = _Ctx(index)
    err = ctx.errors
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
    for i, step in enumerate(steps):
        _validate_step(i, step, ctx)
    if not any(isinstance(s, dict) and s.get("do") == "expect" for s in steps):
        err.append("scenario has no expect step")
    return err


def _validate_step(i: int, step: Any, ctx: _Ctx) -> None:
    where = f"step {i}"
    err = ctx.errors
    if not isinstance(step, dict) or "do" not in step:
        err.append(f"{where}: must be an object with a 'do' key")
        return
    do = step["do"]
    if do not in SCENARIO_STEPS:
        err.append(f"{where}: unknown do {do!r} (known: {', '.join(sorted(SCENARIO_STEPS))})")
        return
    where = f"step {i} ({do})"
    if "wait" in step and not (_is_num(step["wait"]) and step["wait"] >= 0):
        err.append(f"{where}: wait must be a number >= 0")
    if do == "expect" and not ctx.triggered:
        err.append(f"{where}: expect before any world or reply step; nothing can have happened yet")
    checker = _CHECKS.get(do)
    if checker:
        checker(step, where, ctx)
    if do in TRIGGER_STEPS:
        ctx.triggered = True


def _req_str(step: dict, key: str, where: str, ctx: _Ctx) -> str | None:
    v = step.get(key)
    if not isinstance(v, str) or not v.strip():
        ctx.errors.append(f"{where}: '{key}' must be a non-empty string")
        return None
    return v


def _req_num(step: dict, key: str, where: str, ctx: _Ctx, *, minimum: float | None = None) -> float | None:
    v = step.get(key)
    if not _is_num(v) or (minimum is not None and v < minimum):
        bound = f" >= {minimum:g}" if minimum is not None else ""
        ctx.errors.append(f"{where}: '{key}' must be a number{bound}")
        return None
    return float(v)


def _opt_in(step: dict, key: str, allowed: Any, where: str, ctx: _Ctx, *, many: bool = False) -> None:
    if key not in step:
        return
    values = step[key] if (many and isinstance(step[key], list)) else [step[key]]
    bad = [v for v in values if v not in allowed]
    if bad:
        ctx.errors.append(f"{where}: '{key}' has unknown value(s) {bad}; allowed: {sorted(allowed)}")


def _person(step: dict, where: str, ctx: _Ctx, *, residents_only: bool = False) -> str | None:
    who = _req_str(step, "who", where, ctx)
    if who is None:
        return None
    pool = set(ctx.home.residents) if residents_only else ctx.people
    if who not in pool:
        kind = "resident" if residents_only else "person or pet"
        ctx.errors.append(f"{where}: unknown {kind} {who!r}; known: {sorted(pool)}")
    return who


def _room(step: dict, key: str, where: str, ctx: _Ctx, *, required: bool = True) -> None:
    if key not in step and not required:
        return
    room = step.get(key)
    if room not in ctx.home.rooms:
        ctx.errors.append(f"{where}: unknown room {room!r}; known: {sorted(ctx.home.rooms)}")


def _device(step: dict, key: str, where: str, ctx: _Ctx, *, kind: str | None = None) -> str | None:
    dev = step.get(key)
    if dev not in ctx.home.devices:
        ctx.errors.append(f"{where}: unknown device {dev!r}; known: {sorted(ctx.home.devices)}")
        return None
    if kind and ctx.home.devices[dev].get("kind") != kind:
        ctx.errors.append(f"{where}: {dev!r} is not a {kind}")
    return dev


def _device_state(step: dict, device: str | None, where: str, ctx: _Ctx) -> None:
    state = step.get("state")
    if not isinstance(state, dict) or not state:
        ctx.errors.append(f"{where}: 'state' must be a non-empty object")
        return
    if device is None:
        return
    known = set(ctx.home.devices[device].get("state", {}))
    unknown = set(state) - known
    if unknown:
        ctx.errors.append(f"{where}: {device!r} has no state key(s) {sorted(unknown)}; known: {sorted(known)}")


# ----------------------------------------------------------------------------- world steps
def _set_time(step: dict, where: str, ctx: _Ctx) -> None:
    t = _req_str(step, "time", where, ctx)
    if t is not None and not _TIME.match(t):
        ctx.errors.append(f"{where}: time must be HH:MM (24 h), got {t!r}")
    _opt_in(step, "weekday", WEEKDAYS, where, ctx)


def _move(step: dict, where: str, ctx: _Ctx) -> None:
    _person(step, where, ctx)
    _room(step, "to", where, ctx)


def _activity(step: dict, where: str, ctx: _Ctx) -> None:
    _person(step, where, ctx)
    _req_str(step, "text", where, ctx)
    if "attrs" in step and not isinstance(step["attrs"], dict):
        ctx.errors.append(f"{where}: 'attrs' must be an object")


def _idle(step: dict, where: str, ctx: _Ctx) -> None:
    _person(step, where, ctx)
    _req_num(step, "minutes", where, ctx, minimum=1)


def _say(step: dict, where: str, ctx: _Ctx) -> None:
    _person(step, where, ctx)
    _req_str(step, "text", where, ctx)
    _opt_in(step, "lang", _LANGS, where, ctx)


def _tv_say(step: dict, where: str, ctx: _Ctx) -> None:
    _req_str(step, "text", where, ctx)
    if "device" in step:
        _device(step, "device", where, ctx, kind="tv")


def _outside_say(step: dict, where: str, ctx: _Ctx) -> None:
    _req_str(step, "text", where, ctx)
    if "door" in step and step["door"] not in ctx.home.doors:
        ctx.errors.append(f"{where}: unknown door {step['door']!r}; known: {sorted(ctx.home.doors)}")


def _set_env(step: dict, where: str, ctx: _Ctx) -> None:
    _room(step, "room", where, ctx)
    if "lux" not in step and "celsius" not in step:
        ctx.errors.append(f"{where}: needs 'lux' and/or 'celsius'")
    if "lux" in step:
        _req_num(step, "lux", where, ctx, minimum=0)
    if "celsius" in step:
        _req_num(step, "celsius", where, ctx)


def _set_device(step: dict, where: str, ctx: _Ctx) -> None:
    _device_state(step, _device(step, "device", where, ctx), where, ctx)


def _phone(step: dict, where: str, ctx: _Ctx) -> None:
    who = _person(step, where, ctx, residents_only=True)
    if who in ctx.home.residents and not ctx.home.residents[who].get("phone"):
        ctx.errors.append(f"{where}: {who!r} has no phone in home.json")
    if not isinstance(step.get("home"), bool):
        ctx.errors.append(f"{where}: 'home' must be true or false")


def _leave(step: dict, where: str, ctx: _Ctx) -> None:
    _person(step, where, ctx)


def _arrive(step: dict, where: str, ctx: _Ctx) -> None:
    who = _req_str(step, "who", where, ctx)
    if who is not None and not _ID.match(who):
        ctx.errors.append(f"{where}: 'who' must be an id like 'guest1', got {who!r}")
    elif who is not None:
        ctx.people.add(who)
    _room(step, "to", where, ctx)


# ----------------------------------------------------------------------------- runner steps
def _wait(step: dict, where: str, ctx: _Ctx) -> None:
    _req_num(step, "seconds", where, ctx, minimum=0)


def _note(step: dict, where: str, ctx: _Ctx) -> None:
    _req_str(step, "text", where, ctx)


def _reply(step: dict, where: str, ctx: _Ctx) -> None:
    _req_str(step, "answer", where, ctx)
    _opt_in(step, "to", {"parents", *ctx.home.residents}, where, ctx)
    if "if_asked" in step and not isinstance(step["if_asked"], bool):
        ctx.errors.append(f"{where}: 'if_asked' must be true or false")


def _expect(step: dict, where: str, ctx: _Ctx) -> None:
    what = step.get("what")
    if what not in EXPECT_KINDS:
        ctx.errors.append(f"{where}: 'what' must be one of {list(EXPECT_KINDS)}, got {what!r}")
        return
    if "within" in step:
        _req_num(step, "within", where, ctx, minimum=0.1)
    _room(step, "room", where, ctx, required=False)
    if what == "device":
        _device_state(step, _device(step, "device", where, ctx), where, ctx)
    elif what == "no_command":
        if "device" in step and step["device"] != "any":
            _device(step, "device", where, ctx)
        if "command" in step and not isinstance(step["command"], str):
            ctx.errors.append(f"{where}: 'command' must be a string")
    elif what == "ask":
        _opt_in(step, "to", {"parents", *ctx.home.residents}, where, ctx)
        _opt_in(step, "action", ACTIONS, where, ctx)
    elif what == "verdict":
        if "verdict" not in step:
            ctx.errors.append(f"{where}: 'verdict' is required")
        _opt_in(step, "verdict", VERDICTS, where, ctx, many=True)
        _opt_in(step, "action", ACTIONS, where, ctx, many=True)
        _opt_in(step, "who", set(ctx.home.residents), where, ctx)
    elif what == "proposal":
        if "by" not in step:
            ctx.errors.append(f"{where}: 'by' is required")
        _opt_in(step, "by", {*PROPOSAL_BY, DECIDER_WILDCARD}, where, ctx, many=True)
        _opt_in(step, "action", ACTIONS, where, ctx, many=True)
        _opt_in(step, "mode", _MODES, where, ctx)
        _opt_in(step, "kind", _KINDS, where, ctx)
        for key in ("min_confidence", "max_confidence"):
            if key in step and not (_is_num(step[key]) and 0 <= step[key] <= 1):
                ctx.errors.append(f"{where}: '{key}' must be between 0 and 1")
    elif what == "outcome":
        action = step.get("action")
        if action not in ACTIONS or action == "none":
            ctx.errors.append(f"{where}: 'action' must be a real action from contracts.ACTIONS, got {action!r}")
        if "accept" in step:
            accept = step["accept"]
            if not isinstance(accept, list) or not accept or any(a not in OUTCOME_ACCEPT for a in accept):
                ctx.errors.append(f"{where}: 'accept' must be a non-empty subset of {list(OUTCOME_ACCEPT)}")


_CHECKS = {
    "set_time": _set_time, "move": _move, "activity": _activity, "idle": _idle, "say": _say,
    "tv_say": _tv_say, "outside_say": _outside_say, "set_env": _set_env, "set_device": _set_device,
    "phone": _phone, "leave": _leave, "arrive": _arrive,
    "wait": _wait, "note": _note, "reply": _reply, "expect": _expect,
}
assert set(_CHECKS) == set(SCENARIO_STEPS), "every DSL step needs a validator"
assert RUNNER_STEPS <= set(_CHECKS)
