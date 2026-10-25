"""Matchers for meeting scenario `expect` steps (contracts.SCENARIO_STEPS comment):

  item      kind / owner / text_contains / status           a meeting.item was created or updated
  privacy   state                                          meeting.privacy reached the state
  device    device_id / state (subset)                     room.env shows the state (or the snapshot does)
  post      channel / text_contains / kind                 the Notifier posted (meeting.command target notifier)
  no_post   channel? for `within` seconds                  negative: nothing posted
  no_item   kind? for `within` seconds                     negative: no item created
  proposal  by / action / route / mode / min_confidence
  verdict   verdict / action / rule_contains
  screen    banner_contains / card / caption_contains / no_caption_contains

Pure functions over the bus events recorded since the trigger step, so every matcher is unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ...core.events import Event
from ..contracts import MT

EXPECT_KINDS: tuple[str, ...] = ("item", "privacy", "device", "post", "no_post", "no_item", "proposal", "verdict", "screen")
NEGATIVE_KINDS: frozenset[str] = frozenset({"no_post", "no_item"})
DEFAULT_WITHIN = 5.0


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


def _in(want: Any, have: Any) -> bool:
    return want is None or have in as_list(want)


def _contains(exp: dict[str, Any], key: str, text: Any) -> bool:
    if key not in exp:
        return True
    return str(exp[key]).lower() in str(text or "").lower()


def _item(exp: dict[str, Any], events: Sequence[Event]) -> Match | None:
    for e in events:
        if e.topic != MT.ITEM:
            continue
        d = e.data
        if not _in(exp.get("kind"), d.get("kind")) or not _in(exp.get("status"), d.get("status")):
            continue
        if "owner" in exp and not values_equal(exp["owner"], d.get("owner")):
            continue
        if not _contains(exp, "text_contains", d.get("text")):
            continue
        if "due" in exp and exp["due"] is not None and d.get("due") != exp["due"]:
            continue
        if exp.get("has_due") is True and not d.get("due"):
            continue
        return Match(True, f"{d.get('kind')} {d.get('text')!r} owner={d.get('owner')} due={d.get('due')} ({d.get('status')})", e.id)
    return None


def _privacy(exp: dict[str, Any], events: Sequence[Event]) -> Match | None:
    for e in events:
        if e.topic == MT.PRIVACY and _in(exp.get("state"), e.data.get("state")):
            if "min_wiped" in exp and int(e.data.get("wiped_count") or 0) < int(exp["min_wiped"]):
                continue
            return Match(True, f"privacy {e.data.get('state')} ({e.data.get('reason')}; wiped {e.data.get('wiped_count', 0)})", e.id)
    return None


def _device(exp: dict[str, Any], events: Sequence[Event], snapshot: dict[str, Any] | None) -> Match | None:
    device, want = str(exp["device_id"]), dict(exp["state"])
    for e in events:
        if e.topic == MT.ENV and e.data.get("device_id") == device and is_subset(want, e.data.get("state") or {}):
            return Match(True, f"{device} reached {want}", e.id)
    for d in (snapshot or {}).get("devices") or []:
        if d.get("device_id") == device and is_subset(want, d.get("state") or {}):
            return Match(True, f"{device} is {want} (state)")
    return None


def _post(exp: dict[str, Any], events: Sequence[Event]) -> Match | None:
    for e in events:
        if e.topic != MT.COMMAND or e.data.get("target") != "notifier":
            continue
        args = e.data.get("args") or {}
        if e.data.get("command") != exp.get("command", "post"):
            continue
        if not _in(exp.get("channel"), args.get("channel")) or not _in(exp.get("kind"), args.get("kind")):
            continue
        if not _contains(exp, "text_contains", args.get("text")):
            continue
        return Match(True, f"posted to #{args.get('channel')}: {str(args.get('text'))[:80]!r}", e.id)
    return None


def _no_post(exp: dict[str, Any], events: Sequence[Event]) -> Match | None:
    for e in events:
        if e.topic == MT.COMMAND and e.data.get("target") == "notifier" and e.data.get("command") == "post":
            args = e.data.get("args") or {}
            if _in(exp.get("channel"), args.get("channel")) and _contains(exp, "text_contains", args.get("text")):
                return Match(False, f"posted to #{args.get('channel')}: {str(args.get('text'))[:80]!r}", e.id)
    return None


def _no_item(exp: dict[str, Any], events: Sequence[Event]) -> Match | None:
    for e in events:
        if e.topic == MT.ITEM and _in(exp.get("kind"), e.data.get("kind")) and e.data.get("status") != "undone":
            return Match(False, f"item created: {e.data.get('kind')} {e.data.get('text')!r}", e.id)
    return None


def _proposal(exp: dict[str, Any], events: Sequence[Event]) -> Match | None:
    for e in events:
        d = e.data
        if e.topic != MT.PROPOSAL or not _in(exp.get("by"), d.get("by")) or not _in(exp.get("action"), d.get("action")) \
                or not _in(exp.get("route"), d.get("route")) or not _in(exp.get("mode"), d.get("mode")) \
                or not _in(exp.get("kind"), d.get("kind")):
            continue
        conf = d.get("confidence")
        if "min_confidence" in exp and (conf is None or conf < exp["min_confidence"]):
            continue
        if "max_confidence" in exp and (conf is None or conf > exp["max_confidence"]):
            continue
        shown = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
        return Match(True, f"{d.get('by')} proposed {d.get('action')} via {d.get('route')} ({d.get('mode')}, {shown})", e.id)
    return None


def _verdict(exp: dict[str, Any], events: Sequence[Event]) -> Match | None:
    for e in events:
        d = e.data
        if e.topic != MT.VERDICT or not _in(exp.get("verdict"), d.get("verdict")) or not _in(exp.get("action"), d.get("action")):
            continue
        if not _contains(exp, "rule_contains", d.get("rule")):
            continue
        return Match(True, f"verdict {d.get('verdict')} for {d.get('action')} ({d.get('rule')})", e.id)
    return None


SCREEN_POSITIVE_KEYS = ("banner_contains", "card", "caption_contains", "privacy")


def screen_is_negative(exp: dict[str, Any]) -> bool:
    """A screen expect with only no_caption_contains is negative: it passes when no screen state shows the text."""
    return "no_caption_contains" in exp and not any(k in exp for k in SCREEN_POSITIVE_KEYS)


def _screen(exp: dict[str, Any], events: Sequence[Event]) -> Match | None:
    if screen_is_negative(exp):
        for e in events:
            if e.topic == MT.SCREEN:
                caps = " | ".join(str(c.get("text", "")) for c in e.data.get("captions") or [])
                if str(exp["no_caption_contains"]).lower() in caps.lower():
                    return Match(False, f"caption shows {exp['no_caption_contains']!r}", e.id)
        return None
    for e in events:
        if e.topic != MT.SCREEN:
            continue
        d = e.data
        if "banner_contains" in exp and not _contains(exp, "banner_contains", d.get("banner")):
            continue
        if "card" in exp and not any(c.get("card") == exp["card"] for c in d.get("cards") or []):
            continue
        caps = " | ".join(str(c.get("text", "")) for c in d.get("captions") or [])
        if "caption_contains" in exp and not _contains(exp, "caption_contains", caps):
            continue
        if "no_caption_contains" in exp and str(exp["no_caption_contains"]).lower() in caps.lower():
            continue
        if "privacy" in exp and d.get("privacy") != exp["privacy"]:
            continue
        return Match(True, f"screen: banner={d.get('banner')!r} cards={[c.get('card') for c in d.get('cards') or []]}", e.id)
    return None


def check(exp: dict[str, Any], events: Sequence[Event], snapshot: dict[str, Any] | None = None) -> Match | None:
    """None = undecided so far. Negative kinds only ever return a failure; silence until the deadline passes."""
    what = exp.get("what")
    if what == "item":
        return _item(exp, events)
    if what == "privacy":
        return _privacy(exp, events)
    if what == "device":
        return _device(exp, events, snapshot)
    if what == "post":
        return _post(exp, events)
    if what == "no_post":
        return _no_post(exp, events)
    if what == "no_item":
        return _no_item(exp, events)
    if what == "proposal":
        return _proposal(exp, events)
    if what == "verdict":
        return _verdict(exp, events)
    if what == "screen":
        return _screen(exp, events)
    raise ValueError(f"unknown expect kind {what!r}")


def is_negative(exp: dict[str, Any]) -> bool:
    return exp.get("what") in NEGATIVE_KINDS or (exp.get("what") == "screen" and screen_is_negative(exp))


def within(exp: dict[str, Any]) -> float:
    return float(exp.get("within", DEFAULT_WITHIN))


def describe(exp: dict[str, Any]) -> str:
    what, w = exp.get("what"), within(exp)
    if what == "item":
        bits = [str(exp.get("kind") or "item")]
        if "owner" in exp:
            bits.append(f"owner {exp['owner']}")
        if "text_contains" in exp:
            bits.append(f"containing {exp['text_contains']!r}")
        return f"a {' '.join(bits)} within {w:g} s"
    if what == "privacy":
        return f"privacy {exp.get('state')} within {w:g} s"
    if what == "device":
        return f"{exp.get('device_id')} reaches {exp.get('state')} within {w:g} s"
    if what == "post":
        return f"a post to #{exp.get('channel', 'any')}" + (f" containing {exp['text_contains']!r}" if "text_contains" in exp else "") + f" within {w:g} s"
    if what == "no_post":
        return f"no post to #{exp.get('channel', 'any')} for {w:g} s"
    if what == "no_item":
        return f"no {exp.get('kind') or 'item'} created for {w:g} s"
    if what == "proposal":
        return f"{exp.get('by', 'anyone')} proposes {exp.get('action', 'something')} within {w:g} s"
    if what == "verdict":
        return f"verdict {exp.get('verdict')} for {exp.get('action', 'anything')} within {w:g} s"
    if what == "screen":
        parts = [f"{k}={v!r}" for k, v in exp.items() if k not in ("do", "what", "within")]
        return f"screen {' '.join(parts)} within {w:g} s"
    return str(what)


def summarize(events: Sequence[Event]) -> str:
    groups: dict[str, list[str]] = {"proposals": [], "verdicts": [], "items": [], "posts": [], "privacy": [], "env": []}
    for e in events:
        d = e.data
        if e.topic == MT.PROPOSAL:
            conf = d.get("confidence")
            shown = f"@{conf:.2f}" if isinstance(conf, (int, float)) else ""
            groups["proposals"].append(f"{d.get('by')}:{d.get('action')}{shown}{'' if d.get('mode') == 'act' else '(shadow)'}")
        elif e.topic == MT.VERDICT:
            groups["verdicts"].append(f"{d.get('verdict')}:{d.get('action')}")
        elif e.topic == MT.ITEM:
            groups["items"].append(f"{d.get('kind')}:{str(d.get('text'))[:30]!r}:{d.get('status')}")
        elif e.topic == MT.COMMAND and d.get("target") == "notifier":
            groups["posts"].append(f"{(d.get('args') or {}).get('channel')}:{str((d.get('args') or {}).get('text'))[:30]!r}")
        elif e.topic == MT.PRIVACY:
            groups["privacy"].append(str(d.get("state")))
        elif e.topic == MT.ENV:
            groups["env"].append(f"{d.get('device_id')}={d.get('state')}")
    parts = [f"{name} [{', '.join(items[:6])}{', ...' if len(items) > 6 else ''}]" for name, items in groups.items() if items]
    return "; ".join(parts) if parts else "nothing on the bus"
