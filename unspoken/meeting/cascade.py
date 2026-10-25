"""Cascade: meeting.utterance -> meeting.proposal.

1. ONE typed request to the Strands Decider (registry.decider()) per utterance (questions.py).
2. Confidence at or above settings.act_threshold -> act proposal, route "local-decider".
3. Between ask_threshold and act_threshold, when settings.escalate and the llm role is enabled -> the
   decision goes THROUGH the Strands RoomAgent (llm_agent.py): its DeciderToolRouter asks the decider
   once (so the decider is never asked twice), and the local LLM's tool calls become act proposals by
   "llm", route "local-llm", escalated_from = the decider's shadow proposal.
4. Below ask_threshold -> the decider's answer is published anyway (the policy ignores it), so the X-ray
   shows what the decider thought. LLM disabled or failing -> the unsure decider answer stands.
Every proposal carries the X-ray fields (request, answers, model label, latency, tokens, tool calls).
"""
from __future__ import annotations

import time
from typing import Any

from ..core.events import Event
from ..core.router import RouteDecision, raw_answers
from .base import Agent, MeetingState, local_llm_enabled
from .contracts import MT, UtteranceData
from .llm_agent import RoomAgent, interpretation_of, proposal_from_tool_call
from .questions import DeciderQuery, Interpretation, build_meeting_query, commitment_concern, room_action_concern
from .reflex import checkin_only, detect

ACTION_WORDS: dict[str, str] = {
    "create_task": "a task", "log_decision": "a decision", "answer_question": "a question for the room",
    "park_topic": "a parking-lot idea", "set_light": "change the light", "set_fan": "the fan", "show": "show a card",
    "pause_notes": "pause the notes (sounds private)", "resume_notes": "resume the notes", "post_recap": "post the recap",
    "share_docs": "share docs", "none": "nothing to do",
}
HEALTH_CACHE_S = 30.0


class DeciderAgent(Agent):
    name = "decider"

    def __init__(self, bus, state: MeetingState, settings, config, registry: Any) -> None:
        super().__init__(bus, state, settings, config)
        self.registry = registry
        self.version = str(config.get("questions", {}).get("version") or "v2")
        self.owner_threshold = float(config["meeting"].get("owner_threshold", 0.5))
        self._room: RoomAgent | None = None
        self._inflight = 0
        self._health: dict[str, tuple[float, bool]] = {}
        self._warned_llm = False

    def start(self) -> None:
        self.subscribe(MT.UTTERANCE, self._on_utterance)

    # ------------------------------------------------------------------ models
    def room(self) -> RoomAgent | None:
        """The Strands agent around the local LLM; None when the llm role is disabled or unavailable."""
        if self._room is not None:
            return self._room
        try:
            if not local_llm_enabled(self.registry):
                return None
            self._room = RoomAgent(lambda: self.registry.strands_model("llm"), threshold=self.settings.act_threshold,
                                   label=self.registry.label("llm"))
        except Exception:
            return None
        return self._room

    async def _healthy(self, role: str) -> bool:
        now = time.monotonic()
        cached = self._health.get(role)
        if cached and now - cached[0] < HEALTH_CACHE_S:
            return cached[1]
        try:
            ok = self.registry.enabled(role) and bool(await self.registry.decider(role).healthy())
        except Exception:
            ok = False
        self._health[role] = (now, ok)
        return ok

    # ------------------------------------------------------------------ deciding
    async def _on_utterance(self, ev: Event) -> None:
        utt: UtteranceData = ev.data  # type: ignore[assignment]
        text = str(utt.get("text") or "").strip()
        if not text or detect(text) is not None:
            return                                   # reflex rules own "off the record" / "back on the record"
        if checkin_only(text):
            return                                   # "Hi, I'm Priya." is a check-in, fusion handled it
        if self.state.privacy.get("state") == "off_record":
            return                                   # off the record: the words are not even sent to a model
        query = build_meeting_query(utt, self.state.names(), self.version, owner_threshold=self.owner_threshold,
                                    wipe_seconds=self.settings.wipe_seconds)
        self._inflight += 1
        self.busy = True
        try:
            room = self.room() if self.settings.escalate else None
            if room is not None:
                room.threshold = self.settings.act_threshold
                await self._cascade(room, query, utt, ev)
            else:
                await self._decider_only(query, ev)
        finally:
            self._inflight -= 1
            self.busy = self._inflight > 0

    async def _ask(self, role: str, query: DeciderQuery, cause: Event) -> tuple[Interpretation, float, dict] | None:
        t0 = time.perf_counter()
        try:
            answers = await self.registry.decider(role).ask(query.state, query.questions)
            it = query.interpret(answers)
        except Exception as exc:
            await self.log(f"{role} failed: {type(exc).__name__}: {exc}", cause, "warn")
            return None
        return it, (time.perf_counter() - t0) * 1000, raw_answers(answers)

    async def _decider_only(self, query: DeciderQuery, cause: Event) -> None:
        got = await self._ask("decider", query, cause)
        role, prefix = "decider", ""
        if got is None and await self._healthy("decider_big"):
            got = await self._ask("decider_big", query, cause)
            role, prefix = "decider_big", "decider failed, asked the big decider; "
        if got is None:
            return
        it, ms, answers = got
        act, ask = self.settings.act_threshold, self.settings.ask_threshold
        if it.confidence >= act:
            reason = prefix + self._sure(it)
        elif it.confidence >= ask:
            reason = prefix + f"unsure ({it.confidence:.0%}) and the LLM is off: {ACTION_WORDS.get(it.action, it.action)}"
        else:
            reason = prefix + f"too unsure ({it.confidence:.0%}): {ACTION_WORDS.get(it.action, it.action)}"
        await self._propose(query, it, cause, by=role, mode="act", reason=reason, latency_ms=ms, answers=answers)

    async def _cascade(self, room: RoomAgent, query: DeciderQuery, utt: UtteranceData, cause: Event) -> None:
        act, ask = self.settings.act_threshold, self.settings.ask_threshold
        published: dict[str, Event] = {}

        def accept(decision: RouteDecision) -> bool:
            return decision.confidence >= act or decision.confidence < ask      # no LLM for hopeless cases

        async def on_decision(decision: RouteDecision) -> None:
            it = interpretation_of(decision)
            if decision.error is not None or it is None:
                await self.log(f"decider failed: {decision.error}", cause, "warn")
                return
            if decision.route == "local":
                reason = self._sure(it) if it.confidence >= act else \
                    f"too unsure ({it.confidence:.0%}) to bother the LLM: {ACTION_WORDS.get(it.action, it.action)}"
                published["act"] = await self._propose(query, it, cause, by="decider", mode="act", reason=reason,
                                                       latency_ms=decision.latency_ms, answers=decision.answers)
            else:
                published["shadow"] = await self._propose(
                    query, it, cause, by="decider", mode="shadow",
                    reason=f"unsure ({it.confidence:.0%}), asking {room.label}", latency_ms=decision.latency_ms,
                    answers=decision.answers)

        out = await room.decide(self.registry.decider(), query, utt, title=str(self.state.status.get("title") or "Meeting"),
                                owner=self.state.status.get("owner"), accept=accept, on_decision=on_decision)
        if out.route == "local-decider":
            return
        shadow = published.get("shadow")
        dec = out.decision
        it2 = interpretation_of(dec) if dec is not None else None
        if out.error is None:
            calls = out.tool_calls or [{"name": "no_action", "input": {}}]
            extra = {"tokens": out.tokens, "tool_calls": out.tool_calls, "reply": out.reply}
            for call in calls:
                action, args, conf = proposal_from_tool_call(call, query)
                it = Interpretation(action, args, conf, {}, it2.kind if it2 else "none",
                                    it2.kind_probabilities if it2 else {}, args.get("owner"), 0.0, {"tool_call": dict(call)})
                reason = (f"{room.label} called {call['name']}: {ACTION_WORDS.get(action, action)}" if out.tool_calls
                          else f"{room.label} called no tool: nothing to do")
                await self._propose(query, it, cause, by="llm", mode="act", reason=reason, latency_ms=out.latency_ms,
                                    answers=dec.answers if dec else {}, escalated_from=shadow.id if shadow else None,
                                    route="local-llm", model=room.label, extra=extra)
            return
        await self.log(f"{room.label} failed: {out.error}", cause, "warn")
        if it2 is not None:
            await self._propose(query, it2, cause, by="decider", mode="act",
                                reason=f"{room.label} unavailable, using the unsure decider answer",
                                latency_ms=dec.latency_ms, answers=dec.answers,
                                escalated_from=shadow.id if shadow else None)
        elif dec is not None and dec.error:
            await self.log(f"decider failed: {dec.error}", cause, "warn")

    # ------------------------------------------------------------------ proposals
    def _sure(self, it: Interpretation) -> str:
        return f"the decider is {it.confidence:.0%} sure: {ACTION_WORDS.get(it.action, it.action)}"

    def _label(self, role: str) -> str:
        try:
            return str(self.registry.label(role))
        except Exception:
            return role

    async def _propose(self, query: DeciderQuery, it: Interpretation, cause: Event, *, by: str, mode: str, reason: str,
                       latency_ms: float, answers: dict[str, Any] | None, escalated_from: str | None = None,
                       route: str = "local-decider", model: str | None = None, extra: dict[str, Any] | None = None) -> Event:
        data: dict[str, Any] = {
            "by": by, "mode": mode, "kind": it.kind, "action": it.action, "args": dict(it.args),
            "confidence": round(float(it.confidence), 4),
            "confidence_source": "adapter_constant" if by == "llm" else "decider_probability_mass",
            "confidence_action": it.action,
            "probabilities": {k: round(float(v), 4) for k, v in it.probabilities.items() if v > 0.0005},
            "reason": reason, "utterance_id": cause.data.get("utterance_id"), "escalated_from": escalated_from,
            "route": route, "model": model or self._label(by if by in ("decider", "decider_big") else "llm"),
            "request": {"state": query.state, "questions": query.questions}, "answers": answers or {},
            "threshold": float(self.settings.act_threshold), "latency_ms": round(latency_ms, 1),
            "tokens": None, "tool_calls": [], "reply": None,
            # meeting-workstream additions (see the report): who spoke and what the owner question said
            "speaker": cause.data.get("speaker"), "source": cause.data.get("source"),
            "owner": it.owner, "owner_confidence": round(float(it.owner_confidence), 3),
            "owner_evidence": query.owner_evidence(it.owner),
            "kind_probabilities": {k: round(float(v), 4) for k, v in it.kind_probabilities.items()},
        }
        if extra:
            data.update(extra)
        if by == "llm":
            data["confidence_calibrated"] = False
            data["reason"] += "; uncalibrated adapter score, not a model probability"
        concern = commitment_concern(query.text) if it.action in ("create_task", "share_docs") else None
        if it.action in ("set_light", "set_fan", "show"):
            concern = room_action_concern(query.text)
        if concern and mode == "act":
            # Preserve the raw scores/answers/tool calls and the normal model routing. Never turn a
            # semantic veto into pretend model confidence in "none". Tentative tasks already in the
            # policy's owner-review band can still be reviewed; use the published (rounded) score
            # so rounding cannot turn review into automatic action. No threshold is changed.
            review = concern == "tentative commitment" and it.action == "create_task" and \
                self.settings.ask_threshold <= data["confidence"] < self.settings.act_threshold
            data["safety"] = {"reason": concern, "outcome": "owner_review" if review else "abstain",
                              "proposed_action": it.action, "proposed_args": dict(it.args)}
            if review:
                data["reason"] += f"; {concern}: owner review required"
            else:
                data.update(action="none", args={}, owner=None, owner_confidence=0.0, owner_evidence=None)
                data["reason"] = f"abstained: {concern}; original proposal: {data['reason']}"
        elif it.action == "create_task" and it.owner is None:
            data["reason"] += "; owner unassigned: no grounded owner"
        return await self.emit(MT.PROPOSAL, data, cause)
