"""Policy: plain code deciding what happens to a mode-act proposal. Rules, in this order:

  none -> ignore; words from the room screen (media) -> deny; notes off the record -> only pause / resume
  are allowed; notes paused -> no items (tasks, decisions, questions, parking lot, recap, docs); a guest in the
  room -> share_docs denied and post_recap asks the owner; resume after a sensitive pause asks the owner;
  high risk -> ask the owner; then the confidence bands: at or above act -> act; between ask and act -> low
  risk acts, medium and high ask the owner; below ask -> ignore. pause_notes is always allowed (privacy first).
Every verdict names its rule. Shadow proposals get no verdict.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

from ..core.events import Event
from .base import Agent, MeetingState
from .contracts import ACTIONS, MT, Risk

RISK: dict[str, Risk] = {
    "create_task": "medium", "log_decision": "medium", "answer_question": "low", "park_topic": "low",
    "set_light": "low", "set_fan": "low", "show": "low", "pause_notes": "low", "resume_notes": "medium",
    "post_recap": "high", "share_docs": "medium", "none": "low",
}
ITEM_ACTIONS = frozenset({"create_task", "log_decision", "answer_question", "park_topic", "post_recap", "share_docs"})
PRIVACY_ACTIONS = frozenset({"pause_notes", "resume_notes"})
assert set(RISK) == set(ACTIONS)


class PolicyAgent(Agent):
    name = "policy"

    def __init__(self, bus, state: MeetingState, settings, config) -> None:
        super().__init__(bus, state, settings, config)
        self._utterances: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def start(self) -> None:
        self.subscribe(MT.UTTERANCE, self._on_utterance)
        self.subscribe(MT.PROPOSAL, self._on_proposal)

    async def _on_utterance(self, ev: Event) -> None:
        self._utterances[str(ev.data.get("utterance_id") or ev.id)] = ev.data
        while len(self._utterances) > 500:
            self._utterances.popitem(last=False)

    async def _on_proposal(self, ev: Event) -> None:
        p = ev.data
        if p.get("mode") != "act":
            return
        utt = self._utterances.get(str(p.get("utterance_id") or ""))
        verdict, risk, rule = self.judge(p, utt)
        await self.emit(MT.VERDICT, {
            "proposal_id": ev.id, "action": p.get("action", "none"), "args": dict(p.get("args") or {}),
            "verdict": verdict, "risk": risk, "rule": rule,
            # meeting-workstream additions: who proposed, who spoke (the actuator writes them on items / asks)
            "by": p.get("by"), "utterance_id": p.get("utterance_id"), "speaker": p.get("speaker") or (utt or {}).get("speaker"),
            "reason": p.get("reason"), "confidence": p.get("confidence"),
        }, ev)

    # ------------------------------------------------------------------ rules
    def judge(self, p: dict[str, Any], utt: dict[str, Any] | None) -> tuple[str, Risk, str]:
        action = str(p.get("action") or "none")
        risk: Risk = RISK.get(action, "medium")
        if action == "none":
            return "ignore", risk, "nothing to do"
        if action not in ACTIONS:
            return "deny", "high", f"unknown action {action!r}"
        source = (utt or {}).get("source") or p.get("source")
        if source == "media":
            return "deny", risk, "only people in the room count; this came from the screen"
        if action == "pause_notes":
            return "act", risk, "pausing the notes is always allowed"
        if action == "share_docs" and self.state.guests_present():
            return "deny", "high", "a guest without a check-in is in the room: no company docs"
        privacy = self.state.privacy.get("state", "on")
        if privacy == "off_record":
            if action == "resume_notes":
                return "act", risk, "back on the record"
            return "deny", risk, "off the record: nothing is noted or done until someone says 'back on the record'"
        if privacy == "paused" and action in ITEM_ACTIONS:
            return "deny", risk, "notes are paused: no items until they resume"
        if action == "resume_notes":
            if self.state.sensitive_pause:
                return "ask_owner", risk, "the pause came from a sensitive hint; the owner decides"
            return "act", risk, "resuming the notes"
        if action == "post_recap":
            if self.state.guests_present():
                return "ask_owner", "high", "a guest is in the room; the owner decides what leaves it"
            return "ask_owner", "high", "the recap leaves the box: the owner approves first"
        conf = float(p.get("confidence") or 0.0)
        act, ask = float(self.settings.act_threshold), float(self.settings.ask_threshold)
        if p.get("route") == "rule":
            return "act", risk, "reflex rule"
        if conf >= act:
            if risk == "high":
                return "ask_owner", risk, f"high risk: the owner approves first ({conf:.0%} sure)"
            return "act", risk, f"confident enough to act ({conf:.0%}, threshold {act:.0%})"
        if conf >= ask:
            if risk == "low":
                return "act", risk, f"low risk, likely enough ({conf:.0%})"
            return "ask_owner", risk, f"unsure ({conf:.0%}) and {risk} risk: asking the owner"
        return "ignore", risk, f"too unsure to act ({conf:.0%}, below the {ask:.0%} ask threshold)"
