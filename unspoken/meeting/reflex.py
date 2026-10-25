"""Reflex rules: plain code, under a millisecond, no model.

* "off the record" / "let's go off the record" / "这个不要记" / "不要记录" -> notes off_record and the last
  settings.wipe_seconds of transcript are DELETED (memory and screen); "back on the record" -> notes on.
* chat /offrecord and /resume do the same; a verdict for pause_notes / resume_notes (decider, LLM) too.
* a slides scene from the camera -> a `set_light slides` proposal (route "rule").
* a "Hi, I'm Priya" / "我是…" check-in is parsed here (fusion calls `checkin_name`).

The ReflexAgent is the ONE owner of the privacy state and the on-record transcript.
"""
from __future__ import annotations

import re
import time
from typing import Any

from ..core.events import Event
from .base import Agent
from .contracts import MT, PRIVATE_TOPICS, PrivacyData, UtteranceData

OFF_RECORD_PATTERNS = [
    re.compile(r"\boff[\s-]+the[\s-]+record\b", re.I),
    re.compile(r"\boff[\s-]+record\b", re.I),
    re.compile(r"\bdon'?t\s+(write|note|record|log)\s+(this|that)\s+(down|one)?\b", re.I),
    re.compile(r"\bstop\s+(taking\s+)?notes\b", re.I),
    re.compile(r"这个不要记|不要记录|不要记|别记|这段不要记|不记录|关掉记录|停止记录"),
]
ON_RECORD_PATTERNS = [
    re.compile(r"\bback\s+on\s+(the\s+)?record\b", re.I),
    re.compile(r"\bon\s+the\s+record\s+again\b", re.I),
    re.compile(r"\bresume\s+(the\s+)?notes\b", re.I),
    re.compile(r"\bstart\s+(taking\s+)?notes\s+again\b", re.I),
    re.compile(r"可以记了|继续记录|恢复记录|可以记录了|开始记录"),
]

# The surrounding words are case-insensitive, but a name is not: otherwise the optional
# surname swallows "and", "here", etc. from natural ASR introductions.
_NAME = r"((?-i:[A-Z][a-z]+(?:[ -][A-Z][a-z]+)?))"
CHECKIN_PATTERNS = [
    re.compile(r"\b(?:hi|hello|hey|good morning|good afternoon)\b[^.!?]{0,25}?[,!]?\s+(?:i'?m|i am|this is|it'?s|my name is)\s+" + _NAME, re.I),
    re.compile(r"^\s*(?:i'?m|i am|this is|my name is)\s+" + _NAME + r"\b[.,!]?\s*(?:here|checking in|joining)?[.!]?\s*$", re.I),
    re.compile(r"\b(?:i'?m|i am|my name is)\s+" + _NAME + r"[,.]?\s+(?:here|checking in|joining)\b", re.I),
    re.compile(r"\b" + _NAME + r"\s+here[.!]?\s*$"),
    re.compile(r"(?:我是|我叫|这是)\s*([A-Za-z][A-Za-z\- ]{1,20}|[\u4e00-\u9fff]{1,4})"),
    # VAD segments may include the role, a preceding question, or the next sentence.
    # Recognize an explicit self-introduction without treating arbitrary "I'm ..." as a name.
    re.compile(r"\b(?:my name is|i['’]?m|i am)\s+" + _NAME
               + r"(?=\s*[,!.?]|$|\s+and\b)", re.I),
]
NOT_NAMES = {"Sorry", "Sure", "Here", "Not", "Just", "Still", "Back", "Done", "Fine", "Good", "Okay", "Ok", "So",
             "Also", "Going", "Afraid", "Happy", "Glad", "Late", "Ready", "Pretty", "Really", "Very", "Thinking",
             "Saying", "Trying", "Told", "Sure", "Right", "Freezing", "Cold", "Hot", "Tired", "Busy"}
_REPORTED_INTRO = re.compile(r'\b(?:said|says|say|saying|if|suppose|pretend|example)\b|["“”]', re.I)


def detect(text: str) -> str | None:
    """'off_record' | 'on' | None for one utterance (any language)."""
    t = (text or "").strip()
    if not t:
        return None
    if any(p.search(t) for p in ON_RECORD_PATTERNS):
        return "on"
    if any(p.search(t) for p in OFF_RECORD_PATTERNS):
        return "off_record"
    return None


def checkin_name(text: str) -> str | None:
    """A self-introduced name, not proof that a particular voice/seat owns it.

    Longer natural introductions are allowed; quoted/hypothetical speech and conflicting
    names abstain. Fusion separately checks speaker evidence before binding an identity.
    """
    t = (text or "").strip()
    if not t:
        return None
    names: set[str] = set()
    for p in CHECKIN_PATTERNS:
        for m in p.finditer(t):
            if _REPORTED_INTRO.search(t[:m.start()]):
                continue
            name = m.group(1).strip(" .,!")
            if name.isascii() and not name[:1].isupper():
                continue                                   # "I'm freezing" is not a name
            if name and name.split()[0] not in NOT_NAMES:
                names.add(name)
    return next(iter(names)) if len(names) == 1 else None


def checkin_only(text: str) -> bool:
    """Skip planning only for a *whole* check-in, not "I'm Sam. I'll send it."."""
    name = checkin_name(text)
    if name is None:
        return False
    return bool(re.fullmatch(
        r"\s*(?:(?:hi|hello|hey|good morning|good afternoon)(?: everyone)?[,!.\s]+)?"
        r"(?:(?:i['’]?m|i am|this is|it['’]?s|my name is)\s+|(?:我是|我叫|这是)\s*)?"
        + re.escape(name) + r"[,!.\s]*(?:(?:here|checking in|joining)[.!]?)?\s*", text, re.I))


class ReflexAgent(Agent):
    name = "reflex"

    def start(self) -> None:
        self.subscribe(MT.UTTERANCE, self._on_utterance)
        self.subscribe(MT.CHAT, self._on_chat)
        self.subscribe(MT.VERDICT, self._on_verdict)
        self.subscribe(MT.SCENE, self._on_scene)
        self.subscribe(MT.STATUS, self._on_status)

    # ------------------------------------------------------------------ transcript + rules
    async def _on_utterance(self, ev: Event) -> None:
        utt: UtteranceData = ev.data  # type: ignore[assignment]
        text = str(utt.get("text") or "")
        rule = detect(text) if utt.get("source") != "media" else None
        if rule == "off_record":
            who = (utt.get("speaker") or {}).get("name") or "someone"
            await self.set_privacy("off_record", f"{who} said 'off the record'", "reflex", ev, wipe=True)
            return
        if rule == "on":
            await self.set_privacy("on", "said 'back on the record'", "reflex", ev)
            return
        if self.state.privacy.get("state") == "on":
            self.state.transcript.append(dict(utt))

    async def _on_chat(self, ev: Event) -> None:
        cmd = ev.data.get("command")
        user = ev.data.get("user") or "someone"
        if cmd == "offrecord":
            await self.set_privacy("off_record", f"/offrecord by {user}", "user", ev, wipe=True)
        elif cmd == "resume":
            await self.set_privacy("on", f"/resume by {user}", "user", ev)

    async def _on_verdict(self, ev: Event) -> None:
        v = ev.data
        if v.get("verdict") != "act":
            return
        action = v.get("action")
        if action == "pause_notes":
            args = v.get("args") or {}
            by = str(v.get("by") or "decider")
            self.state.sensitive_pause = by in ("decider", "decider_big", "llm")
            await self.set_privacy("paused", str(args.get("reason") or "sounded like something private"), by, ev,
                                   wipe=bool(args.get("wipe_seconds")), wipe_seconds=args.get("wipe_seconds"))
        elif action == "resume_notes":
            self.state.sensitive_pause = False
            await self.set_privacy("on", "resumed", str(v.get("by") or "user"), ev)

    async def _on_scene(self, ev: Event) -> None:
        mode = str(ev.data.get("mode") or "talk")
        before = self.state.scene_mode
        self.state.scene_mode = mode
        if mode == before or mode not in ("slides", "talk", "board", "break"):
            return
        scene = self.config["scenes"].get(mode, "talk")
        await self.emit(MT.PROPOSAL, {
            "by": "reflex", "mode": "act", "kind": "room", "action": "set_light", "args": {"scene": scene},
            "confidence": 1.0, "probabilities": {"set_light": 1.0},
            "reason": f"the camera sees {mode}: lamp scene '{scene}'", "utterance_id": None, "escalated_from": None,
            "route": "rule", "model": "reflex rules", "request": {}, "answers": {}, "threshold": 1.0,
            "latency_ms": round((time.time() - ev.wall) * 1000, 2), "tokens": None, "tool_calls": [], "reply": None,
        }, ev)

    async def _on_status(self, ev: Event) -> None:
        if ev.data.get("state") == "started":
            self.state.transcript.clear()
            self.state.sensitive_pause = False
            self.state.privacy = {"state": "on", "reason": "meeting started", "by": "user", "wiped_from": None,
                                  "wiped_count": 0}

    # ------------------------------------------------------------------ the one writer of privacy state
    async def set_privacy(self, state: str, reason: str, by: str, cause: Event | None, *, wipe: bool = False,
                          wipe_seconds: float | None = None) -> PrivacyData:
        wiped_from, wiped_count = None, 0
        if wipe:
            window = float(wipe_seconds if wipe_seconds is not None else self.settings.wipe_seconds)
            now = cause.ts if cause is not None else self.state.clock
            wiped_from = max(0.0, now - window)
            keep = [u for u in self.state.transcript if float(u.get("start", 0.0)) < wiped_from]
            wiped_count = len(self.state.transcript) - len(keep)
            self.state.transcript = keep
            # The wiped minute must not survive in the bus history either (only lines already sent to a
            # trace file stay; keep --trace off in real meetings).
            cutoff = wiped_from
            forget = getattr(self.bus, "forget", None)
            if forget is not None:
                forget(lambda e: e.topic in PRIVATE_TOPICS and float(e.ts or 0.0) >= cutoff)
        data: PrivacyData = {"state": state, "reason": reason, "by": by, "wiped_from": wiped_from,  # type: ignore[typeddict-item]
                             "wiped_count": wiped_count}
        self.state.privacy = data
        await self.emit(MT.PRIVACY, dict(data), cause)
        return data
