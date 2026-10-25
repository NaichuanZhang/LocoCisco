"""Policy: plain code deciding what happens to a mode-act proposal. Rules from home.json
house_rules, applied in this order: action none -> ignore; TV/outside voice -> deny; unlock needs a
parent with a strong voice match (kids denied); kids' TV curfew -> ask_parent; alert_parent -> allow;
two residents asking for opposite things within 10 s -> ignore the later one; then the confidence
bands (act / ask / ignore). Shadow proposals get no verdict. Every verdict names its rule."""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from ..contracts import Event, T
from .base import Agent, ClockState, HomeConfig

CONFLICT_WINDOW_S = 10.0
CURFEW_FROM_MIN = 20 * 60
OPPOSITES: dict[str, set[str]] = {
    "lights_on": {"lights_off", "lights_dim"}, "lights_off": {"lights_on"}, "lights_dim": {"lights_on"},
    "warmer": {"cooler"}, "cooler": {"warmer"},
    "tv_on": {"tv_off"}, "tv_off": {"tv_on"},
    "unlock_door": {"lock_door"}, "lock_door": {"unlock_door"},
}


@dataclass
class _Request:
    ts: float
    wall: float
    speaker_id: str
    speaker_name: str
    action: str
    scope: str   # target device, or the room when no device is known


class PolicyAgent(Agent):
    name = "policy"

    def __init__(self, bus, home: HomeConfig, settings) -> None:
        super().__init__(bus, home, settings)
        self.clock = ClockState(home)
        self._utterances: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._requests: list[_Request] = []

    def start(self) -> None:
        self.subscribe(T.PROPOSAL, self._on_proposal)
        self.subscribe(T.UTTERANCE, self._on_utterance)
        self.subscribe(T.CLOCK, self._on_clock)

    async def _on_clock(self, ev: Event) -> None:
        self.clock.update(ev.data)

    async def _on_utterance(self, ev: Event) -> None:
        self._utterances[ev.id] = ev.data
        while len(self._utterances) > 500:
            self._utterances.popitem(last=False)

    async def _on_proposal(self, ev: Event) -> None:
        p = ev.data
        if p.get("mode") != "act":
            return
        utterance = self._utterances.get(ev.cause or "")
        speaker = self.identify(p.get("speaker") or (utterance or {}).get("speaker"))
        verdict, rule = self.judge(ev, p, utterance, speaker)
        await self.emit(T.VERDICT, {
            "proposal_id": ev.id, "action": p.get("action", "none"), "target": p.get("target"),
            "room": p.get("room"), "verdict": verdict, "rule": rule, "who": speaker,
        }, ev)

    # ------------------------------------------------------------------ rules
    def identify(self, speaker: dict[str, Any] | None) -> dict[str, Any] | None:
        """Normalise a Speaker; a voice match below thresholds.voice_match counts as nobody."""
        if not speaker:
            return None
        sp = dict(speaker)
        match = float(sp.get("match") or 0.0)
        sid = sp.get("id")
        if sid and match < float(self.home.thresholds["voice_match"]):
            return {"id": None, "name": None, "role": "unknown", "match": match}
        return {"id": sid, "name": sp.get("name") or self.home.resident_name(sid),
                "role": sp.get("role") or (self.home.resident_role(sid) if sid else "unknown"), "match": match}

    def judge(self, ev: Event, p: dict[str, Any], utterance: dict | None, speaker: dict | None) -> tuple[str, str]:
        action = p.get("action", "none")
        role = (speaker or {}).get("role", "unknown")
        match = float((speaker or {}).get("match") or 0.0)
        if action == "none":
            return "ignore", "nothing to do"
        source = (utterance or {}).get("source")
        if source in ("tv", "outside"):
            return "deny", f"only voices inside the house count; this came from {'the TV' if source == 'tv' else 'outside'}"
        if action == "unlock_door":
            if role == "child":
                return "deny", "kids may not unlock doors"
            need = float(self.home.thresholds["unlock_voice_match"])
            if role != "parent" or match < need:
                return "deny", f"unlocking needs a parent's voice with a strong match ({need:.2f} or more)"
        if action == "tv_on" and role == "child" and self.clock.minutes >= CURFEW_FROM_MIN and self.clock.school_night:
            return "ask_parent", "kids may not turn on the TV after 20:00 on a school night without a parent's OK"
        if action == "alert_parent":
            return "allow", "safety alerts to a parent are always allowed"
        earlier = self._conflict(ev, p, speaker)
        if earlier is not None:
            return "ignore", f"conflicting requests: {earlier.speaker_name} just asked for the opposite"
        conf = float(p.get("confidence") or 0.0)
        act, ask = float(self.settings.act_threshold), float(self.settings.ask_threshold)
        if conf >= act:
            return "allow", f"confident enough to act ({conf:.0%}, threshold {act:.0%})"
        if conf >= ask:
            who = (speaker or {}).get("name") or "the parents"
            return "ask_speaker", f"unsure what was wanted ({conf:.0%}), asking {who} first"
        return "ignore", f"too unsure to act ({conf:.0%}, below the {ask:.0%} ask threshold)"

    def _conflict(self, ev: Event, p: dict[str, Any], speaker: dict | None) -> _Request | None:
        """Remember speech requests by known residents; report an earlier opposite one within 10 s."""
        if p.get("kind") != "speech" or not (speaker or {}).get("id"):
            return None
        action = p.get("action", "none")
        scope = p.get("target") or p.get("room") or ""
        now_wall = time.time()
        self._requests = [r for r in self._requests if self._within(ev.ts, now_wall, r)]
        for r in self._requests:
            if r.speaker_id != speaker["id"] and r.scope == scope and r.action in OPPOSITES.get(action, set()):
                return r
        self._requests.append(_Request(ev.ts, now_wall, speaker["id"], speaker.get("name") or speaker["id"], action, scope))
        return None

    @staticmethod
    def _within(ts: float, wall: float, r: _Request) -> bool:
        if ts and r.ts:
            return abs(ts - r.ts) <= CONFLICT_WINDOW_S
        return abs(wall - r.wall) <= CONFLICT_WINDOW_S
