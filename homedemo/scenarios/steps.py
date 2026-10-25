"""The scenario step DSL: which steps the world handles, which the runner handles, and how to narrate them.

The step names come from contracts.SCENARIO_STEPS; this module only splits them and turns a step
into one line of text for the scenario.status stream.
"""
from __future__ import annotations

from typing import Any

from ..contracts import SCENARIO_STEPS
from .homeindex import HomeIndex

RUNNER_STEPS: frozenset[str] = frozenset({"wait", "note", "reply", "expect"})
WORLD_STEPS: frozenset[str] = frozenset(SCENARIO_STEPS) - RUNNER_STEPS
# Steps that can make the pipeline do something. An `expect` watches bus events from the most
# recent one of these before it (a reply counts only when a reply was actually published).
TRIGGER_STEPS: frozenset[str] = WORLD_STEPS | {"reply"}

WEEKDAYS: tuple[str, ...] = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def fmt_state(state: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in state.items())


def describe_step(step: dict[str, Any], home: HomeIndex) -> str:
    """One human line per step, in the voice of a narrator ("Alex walks to the Kitchen")."""
    do = step.get("do")
    who = home.name(str(step.get("who", "")))
    text = str(step.get("text", ""))
    if do == "set_time":
        return f"Clock: {step.get('time', '?')} {step.get('weekday', '')}".strip()
    if do == "move":
        return f"{who} walks to the {home.room_name(str(step.get('to', '?')))}"
    if do == "activity":
        return f"{who}: {text}"
    if do == "idle":
        return f"{who} does not move for {step.get('minutes', '?')} minutes"
    if do == "say":
        return f'{who} says "{text}"'
    if do == "tv_say":
        return f'The TV says "{text}"'
    if do == "outside_say":
        door = home.doors.get(str(step.get("door", "")), {}).get("name", "front door")
        return f'A voice outside the {door.lower()} says "{text}"'
    if do == "set_env":
        parts = []
        if "lux" in step:
            parts.append(f"{step['lux']} lux")
        if "celsius" in step:
            parts.append(f"{step['celsius']} °C")
        return f"{home.room_name(str(step.get('room', '?')))}: {', '.join(parts) or 'no change'}"
    if do == "set_device":
        return f"{step.get('device', '?')}: {fmt_state(step.get('state') or {})}"
    if do == "phone":
        return f"{who}'s phone is {'at home' if step.get('home') else 'away'}"
    if do == "leave":
        return f"{who} leaves the house"
    if do == "arrive":
        return f"{who} arrives in the {home.room_name(str(step.get('to', '?')))}"
    if do == "wait":
        return f"Wait {step.get('seconds', 1)} s"
    if do == "note":
        return text
    if do == "reply":
        return f'Phone reply: "{step.get("answer", "")}"'
    if do == "expect":
        from .expectations import describe  # local import: expectations imports homeindex only
        return "Expect " + describe(step, home)
    return f"{do}"
