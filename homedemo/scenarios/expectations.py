"""Matchers for scenario `expect` steps.

Pure functions over the bus events recorded since the step that triggered the check (plus the
world snapshot for `device`), so the runner stays small and every matcher is unit-testable.

Kinds (contracts.SCENARIO_STEPS comment) plus `outcome`, which is not in contracts.py yet:
  {"do":"expect","what":"outcome","action":"lights_on","room":"living","accept":["done","asked"],"within":8}
passes on a device command that amounts to that action in that room ("done") OR on a phone ask
proposing it ("asked", when accepted). Decider-driven expects normally accept both; reflex and
policy expects use exact kinds (device / no_command / verdict / ask).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..contracts import Event, T
from .homeindex import HomeIndex

EXPECT_KINDS: tuple[str, ...] = ("device", "no_command", "ask", "verdict", "proposal", "outcome")
OUTCOME_ACCEPT: tuple[str, ...] = ("done", "asked")
VERDICTS: tuple[str, ...] = ("allow", "ask_speaker", "ask_parent", "deny", "ignore")
PROPOSAL_BY: tuple[str, ...] = ("reflex", "decider-2b", "decider-12b", "bedrock", "keyword", "user", "llm")
DECIDER_WILDCARD = "decider"  # expect "by": "decider" = a model decided: decider-2b, decider-12b or bedrock
MODEL_BY: frozenset[str] = frozenset({"decider-2b", "decider-12b", "bedrock"})
DEFAULT_WITHIN = 5.0

# What a device command amounts to: device kind -> command -> action.
_KIND_COMMAND_ACTION: dict[str, dict[str, str]] = {
    "light": {"on": "lights_on", "off": "lights_off", "dim": "lights_dim"},
    "tv": {"on": "tv_on", "off": "tv_off"},
    "lock": {"unlock": "unlock_door", "lock": "lock_door"},
    "phone": {"notify": "alert_parent"},
}
# An outcome for the key is satisfied by any of these actions ("lights off or dimmer").
OUTCOME_ALIASES: dict[str, frozenset[str]] = {"lights_off": frozenset({"lights_off", "lights_dim"})}
# Device kinds whose room is meaningful for an outcome (a thermostat or lock is house-wide).
_ROOM_BOUND_KINDS: frozenset[str] = frozenset({"light", "tv"})


@dataclass
class Match:
    ok: bool
    text: str
    event_id: str | None = None


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    return a == b


def is_subset(want: dict[str, Any], have: dict[str, Any]) -> bool:
    return all(k in have and values_equal(v, have[k]) for k, v in want.items())


def command_action(cmd: dict[str, Any], home: HomeIndex) -> str | None:
    """The ACTIONS name a CommandData amounts to, or None when it maps to nothing we grade."""
    device = str(cmd.get("device_id", ""))
    kind = home.device_kind(device)
    command = str(cmd.get("command", ""))
    if kind == "thermostat":
        if command != "set_setpoint":
            return None
        delta = (cmd.get("args") or {}).get("delta_c")
        if isinstance(delta, (int, float)) and not isinstance(delta, bool) and delta != 0:
            return "warmer" if delta > 0 else "cooler"
        return None
    return _KIND_COMMAND_ACTION.get(kind or "", {}).get(command)


def _in(want: Any, have: Any) -> bool:
    return want is None or have in as_list(want)


def _fields_match(exp: dict[str, Any], data: dict[str, Any], keys: Sequence[str]) -> bool:
    return all(k not in exp or values_equal(exp[k], data.get(k)) for k in keys)


def _by_matches(want: Any, by: Any) -> bool:
    if want is None:
        return True
    for w in as_list(want):
        if w == by or (w == DECIDER_WILDCARD and by in MODEL_BY):
            return True
    return False


def _device(exp: dict[str, Any], events: Sequence[Event], snapshot: dict[str, Any] | None) -> Match | None:
    device, want = str(exp["device"]), dict(exp["state"])
    for e in events:
        if e.topic in (T.APPLIED, T.DEVICE) and e.data.get("device_id") == device \
                and is_subset(want, e.data.get("state") or {}):
            return Match(True, f"{device} reached {_fmt(want)} ({e.topic})", e.id)
    for d in (snapshot or {}).get("devices") or []:
        if d.get("device_id") == device and is_subset(want, d.get("state") or {}):
            return Match(True, f"{device} is {_fmt(want)} (world state)")
    return None


def _no_command(exp: dict[str, Any], events: Sequence[Event]) -> Match | None:
    device = exp.get("device", "any")
    command = exp.get("command")
    for e in events:
        if e.topic != T.COMMAND:
            continue
        if device not in (None, "any") and e.data.get("device_id") != device:
            continue
        if command and e.data.get("command") != command:
            continue
        return Match(False, f"command {e.data.get('command')} -> {e.data.get('device_id')} was issued", e.id)
    return None


def _ask(exp: dict[str, Any], events: Sequence[Event]) -> Match | None:
    for e in events:
        if e.topic != T.ASK or not _fields_match(exp, e.data, ("to", "action", "room", "target")):
            continue
        if "text" in exp and str(exp["text"]).lower() not in str(e.data.get("text", "")).lower():
            continue
        return Match(True, f"asked {e.data.get('to')}: {e.data.get('text')!r}", e.id)
    return None


def _verdict(exp: dict[str, Any], events: Sequence[Event]) -> Match | None:
    for e in events:
        d = e.data
        if e.topic != T.VERDICT or not _in(exp.get("verdict"), d.get("verdict")) \
                or not _in(exp.get("action"), d.get("action")) or not _fields_match(exp, d, ("room", "target")):
            continue
        if "rule" in exp and str(exp["rule"]).lower() not in str(d.get("rule", "")).lower():
            continue
        if "who" in exp and (d.get("who") or {}).get("id") != exp["who"]:
            continue
        return Match(True, f"verdict {d.get('verdict')} for {d.get('action')} ({d.get('rule', 'no rule given')})", e.id)
    return None


def _proposal(exp: dict[str, Any], events: Sequence[Event]) -> Match | None:
    for e in events:
        d = e.data
        if e.topic != T.PROPOSAL or not _by_matches(exp.get("by"), d.get("by")) \
                or not _in(exp.get("action"), d.get("action")) or not _fields_match(exp, d, ("room", "mode", "kind")):
            continue
        conf = d.get("confidence")
        if "min_confidence" in exp and (conf is None or conf < exp["min_confidence"]):
            continue
        if "max_confidence" in exp and (conf is None or conf > exp["max_confidence"]):
            continue
        shown = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
        return Match(True, f"{d.get('by')} proposed {d.get('action')} ({d.get('mode', 'act')}, confidence {shown})", e.id)
    return None


def _outcome(exp: dict[str, Any], events: Sequence[Event], home: HomeIndex) -> Match | None:
    action = str(exp["action"])
    room = exp.get("room")
    accept = as_list(exp.get("accept")) or list(OUTCOME_ACCEPT)
    wanted = OUTCOME_ALIASES.get(action, frozenset({action}))
    for e in events:
        if "done" in accept and e.topic == T.COMMAND:
            device = str(e.data.get("device_id", ""))
            if command_action(e.data, home) in wanted and _room_ok(room, device, home):
                return Match(True, f"{action}: done ({e.data.get('command')} -> {device})", e.id)
        if "asked" in accept and e.topic == T.ASK:
            if e.data.get("action") in wanted and (room is None or e.data.get("room") in (None, room)):
                return Match(True, f"{action}: asked {e.data.get('to')} {e.data.get('text')!r}", e.id)
    return None


def _room_ok(room: Any, device: str, home: HomeIndex) -> bool:
    if room is None or home.device_kind(device) not in _ROOM_BOUND_KINDS:
        return True
    return home.device_room(device) == room


def check(exp: dict[str, Any], events: Sequence[Event], home: HomeIndex,
          snapshot: dict[str, Any] | None = None) -> Match | None:
    """None = undecided so far. For `no_command` a Match is only ever a failure (a command was seen);
    the runner turns silence until the deadline into the pass."""
    what = exp.get("what")
    if what == "device":
        return _device(exp, events, snapshot)
    if what == "no_command":
        return _no_command(exp, events)
    if what == "ask":
        return _ask(exp, events)
    if what == "verdict":
        return _verdict(exp, events)
    if what == "proposal":
        return _proposal(exp, events)
    if what == "outcome":
        return _outcome(exp, events, home)
    raise ValueError(f"unknown expect kind {what!r}")


def is_negative(exp: dict[str, Any]) -> bool:
    return exp.get("what") == "no_command"


def within(exp: dict[str, Any]) -> float:
    return float(exp.get("within", DEFAULT_WITHIN))


def describe(exp: dict[str, Any], home: HomeIndex) -> str:
    what = exp.get("what")
    w = within(exp)
    if what == "device":
        return f"{exp.get('device')} reaches {_fmt(exp.get('state') or {})} within {w:g} s"
    if what == "no_command":
        device = exp.get("device", "any")
        target = "any device" if device in (None, "any") else device
        cmd = f" {exp['command']}" if exp.get("command") else ""
        return f"no{cmd} command to {target} for {w:g} s"
    if what == "ask":
        who = f" to {exp['to']}" if exp.get("to") else ""
        about = f" about {exp['action']}" if exp.get("action") else ""
        return f"a phone ask{who}{about} within {w:g} s"
    if what == "verdict":
        about = f" for {_alts(exp.get('action'))}" if exp.get("action") else ""
        return f"verdict {_alts(exp.get('verdict'))}{about} within {w:g} s"
    if what == "proposal":
        about = f" proposes {_alts(exp.get('action'))}" if exp.get("action") else " proposes something"
        return f"{_alts(exp.get('by'))}{about} within {w:g} s"
    if what == "outcome":
        where = f" in the {home.room_name(str(exp['room']))}" if exp.get("room") else ""
        accept = as_list(exp.get("accept")) or list(OUTCOME_ACCEPT)
        return f"{exp.get('action')}{where} is {' or '.join(accept)} within {w:g} s"
    return f"{what}"


def summarize(events: Sequence[Event]) -> str:
    """Compact 'what the pipeline did' line for a failure reason."""
    groups: dict[str, list[str]] = {"proposals": [], "verdicts": [], "commands": [], "asks": []}
    for e in events:
        d = e.data
        if e.topic == T.PROPOSAL:
            conf = d.get("confidence")
            shown = f"@{conf:.2f}" if isinstance(conf, (int, float)) else ""
            mode = "" if d.get("mode", "act") == "act" else f"({d.get('mode')})"
            groups["proposals"].append(f"{d.get('by')}:{d.get('action')}{shown}{mode}")
        elif e.topic == T.VERDICT:
            groups["verdicts"].append(f"{d.get('verdict')}:{d.get('action')}")
        elif e.topic == T.COMMAND:
            groups["commands"].append(f"{d.get('command')}->{d.get('device_id')}")
        elif e.topic == T.ASK:
            groups["asks"].append(f"{d.get('to')}:{d.get('action')}")
    parts = [f"{name} [{', '.join(items[:6])}{', ...' if len(items) > 6 else ''}]"
             for name, items in groups.items() if items]
    return "; ".join(parts) if parts else "no proposals, verdicts, commands or asks on the bus"


def _fmt(state: dict[str, Any]) -> str:
    return "{" + ", ".join(f"{k}: {v}" for k, v in state.items()) + "}"


def _alts(value: Any) -> str:
    return " or ".join(str(v) for v in as_list(value)) or "?"
