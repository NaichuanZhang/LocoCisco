"""Shared logic of the two decider agents: which model(s) to ask for settings.decider_mode, the cascade
(2B first; when it is unsure, a house agent or the 12B, settings.escalate_to), the fallbacks when a call
fails, and the ProposalData with every field filled in (route, model, request, answers, tokens...).

Cascade with escalate_to "bedrock" or "llm" (speech AND scenes, one rule): the decision goes THROUGH the
Strands house agent in that deciders slot (agents/house_agent.py). Its DeciderToolRouter asks the 2B once;
if the 2B is sure (and not a doubtful "none", eval/REPORT.md s3) the model is never called and the 2B's
answer is the act proposal (route "local"). Otherwise the 2B's answer is a shadow proposal and the model's
tool calls become act proposals by "bedrock" / "llm" (route = the agent's: "bedrock", or "local" for a model
on the hub; escalated_from = the shadow). Model failure -> the 12B if configured and healthy, else the
unsure 2B answer.

Model names in the X-ray (ProposalData.model) and the routes come from the decider objects when they carry
them (`label`, `route`: set by the models registry and the house agents), else from the SLOT_* defaults.
"""
from __future__ import annotations

import time
from typing import Any

from ..contracts import Event, Settings, T
from ..decider_client import Decider
from ..decider.questions import DeciderQuery, Interpretation
from ..patterns.decider_tool_router import RouteDecision, raw_answers
from .base import ACTION_WORDS, Agent, HomeConfig
from .house_agent import MODEL_LABEL, HouseAgent, interpretation_of, proposal_from_tool_call

HOUSE_SLOTS = ("bedrock", "llm")                     # deciders slots that hold a Strands house agent
SLOT_BY: dict[str, str] = {"2b": "decider-2b", "12b": "decider-12b", "bedrock": "bedrock", "llm": "llm"}
SLOT_LABEL: dict[str, str] = {"2b": "2B", "12b": "12B", "fake": "fake decider", "bedrock": "Claude on Bedrock",
                              "llm": "the LLM"}
SLOT_MODEL: dict[str, str] = {"2b": "Strands Decider 2B", "12b": "Strands Decider 12B", "fake": "fake decider",
                              "bedrock": MODEL_LABEL, "llm": "local LLM"}
SLOT_ROUTE: dict[str, str] = {"2b": "local", "12b": "12b", "fake": "local", "bedrock": "bedrock", "llm": "local"}
PROPOSAL_BY = {"reflex", "decider-2b", "decider-12b", "bedrock", "fake", "keyword", "user", "llm"}
SHADOW_REASON = "unsure, asking the 12B"
SHADOW_REASON_BEDROCK = "unsure, asking Claude on Bedrock"
HEALTH_CACHE_S = 30.0


def accept_rule(decision: RouteDecision, *, from_person: bool, threshold: float) -> bool:
    """When the 2B's answer stands on its own (eval/REPORT.md s3): confidence at or above the act threshold,
    except that a "none" for a person's words or a camera scene is weak evidence and always escalates.
    TV and outside voices do not escalate (the policy denies them whatever a model says)."""
    it = interpretation_of(decision)
    doubtful_none = it is not None and it.action == "none" and from_person
    return decision.confidence >= threshold and not doubtful_none


class DeciderAgent(Agent):
    """Subclasses build a DeciderQuery from their input event and call decide()."""

    def __init__(self, bus, home: HomeConfig, settings: Settings, deciders: dict[str, Any]) -> None:
        super().__init__(bus, home, settings)
        self.deciders = deciders
        self._inflight = 0
        self._warned_mode: str | None = None
        self._health: dict[str, tuple[float, bool]] = {}

    # ------------------------------------------------------------------ model selection
    def _chain(self) -> list[str]:
        """Plain decider slots to try, in order, for the current mode (the house agents are not ones)."""
        mode = self.settings.decider_mode
        available = [s for s in ("2b", "12b", "fake") if s in self.deciders]
        if mode == "cascade":
            chain = [s for s in ("2b", "12b") if s in available]
        elif mode in available:
            chain = [mode]
        else:
            chain = []
        return chain + [s for s in available if s not in chain]

    def _house(self) -> HouseAgent | None:
        """The house agent settings.escalate_to names ("bedrock" or "llm"), if that slot is configured."""
        slot = self.settings.escalate_to
        if slot not in HOUSE_SLOTS:
            return None
        house = self.deciders.get(slot)
        return house if isinstance(house, HouseAgent) or hasattr(house, "decide") else None

    def _by(self, slot: str) -> str:
        if slot in SLOT_BY:
            return SLOT_BY[slot]
        name = getattr(self.deciders[slot], "name", "")
        return name if name in PROPOSAL_BY else "llm"

    def _model_label(self, slot: str) -> str:
        """X-ray model name: what the decider object says about itself (registry label), else the default."""
        return getattr(self.deciders.get(slot), "label", None) or SLOT_MODEL.get(slot, slot)

    def _route(self, slot: str) -> str:
        return getattr(self.deciders.get(slot), "route", None) or SLOT_ROUTE.get(slot, "local")

    # ------------------------------------------------------------------ deciding
    async def decide(self, query: DeciderQuery, cause: Event, *, room: str | None, speaker: dict | None) -> None:
        chain = self._chain()
        if not chain:
            await self.log(f"no decider configured for mode '{self.settings.decider_mode}'", cause, "warn")
            return
        if self.settings.decider_mode not in ("cascade", chain[0]) and self._warned_mode != self.settings.decider_mode:
            self._warned_mode = self.settings.decider_mode
            await self.log(f"decider mode '{self.settings.decider_mode}' is not configured; using {chain[0]}", cause, "warn")
        self._inflight += 1
        self.busy = True
        try:
            cascade = self.settings.decider_mode == "cascade" and chain[0] == "2b"
            house = self._house() if cascade else None
            if house is not None:
                await self._cascade_house(house, query, cause, room, speaker)
            elif cascade and chain[:2] == ["2b", "12b"]:
                await self._cascade(query, cause, room, speaker, fallback=chain[2:])
            else:
                await self._single(chain, query, cause, room, speaker)
        finally:
            self._inflight -= 1
            self.busy = self._inflight > 0

    async def _ask(self, slot: str, query: DeciderQuery, cause: Event) -> tuple[Interpretation, float, dict] | None:
        """One decider call; on failure publish trace.log and return None."""
        t0 = time.perf_counter()
        try:
            answers = await self.deciders[slot].ask(query.state, query.questions)
            interpretation = query.interpret(answers)
        except Exception as exc:  # network, server, or a malformed answer: the pipeline must go on
            await self.log(f"{SLOT_LABEL.get(slot, slot)} decider failed: {type(exc).__name__}: {exc}", cause, "warn")
            return None
        return interpretation, (time.perf_counter() - t0) * 1000, raw_answers(answers)

    async def _healthy(self, slot: str) -> bool:
        """Lazy health check of an optional fallback (e.g. a stopped 12B server), cached 30 s."""
        now = time.monotonic()
        cached = self._health.get(slot)
        if cached and now - cached[0] < HEALTH_CACHE_S:
            return cached[1]
        try:
            ok = bool(await self.deciders[slot].healthy())
        except Exception:
            ok = False
        self._health[slot] = (now, ok)
        return ok

    async def _single(self, chain: list[str], query: DeciderQuery, cause: Event, room, speaker) -> None:
        for i, slot in enumerate(chain):
            got = await self._ask(slot, query, cause)
            if got is None:
                continue
            it, ms, answers = got
            reason = self._sure_reason(slot, it)
            if i:
                reason = f"{SLOT_LABEL.get(chain[i - 1], chain[i - 1])} failed, asked the {SLOT_LABEL.get(slot, slot)}; " + reason
            await self._propose(query, it, cause, slot=slot, mode="act", room=room, speaker=speaker,
                                reason=reason, latency_ms=ms, escalated_from=None, answers=answers)
            return

    def _from_person(self, cause: Event) -> bool:
        """eval/REPORT.md s3: a 2B "none" for a person's words or a scene is weak evidence and escalates;
        a TV or outside voice does not (the policy denies them whatever the model says)."""
        return cause.topic != T.UTTERANCE or cause.data.get("source", "person") == "person"

    async def _cascade(self, query: DeciderQuery, cause: Event, room, speaker, fallback: list[str]) -> None:
        """2B -> 12B (settings.escalate_to == "12b", or no house agent configured)."""
        got = await self._ask("2b", query, cause)
        if got is None:
            await self._single(["12b"] + fallback, query, cause, room, speaker)
            return
        it2b, ms2b, answers = got
        doubtful_none = it2b.action == "none" and self._from_person(cause)
        if it2b.confidence >= self.settings.act_threshold and not doubtful_none:
            await self._propose(query, it2b, cause, slot="2b", mode="act", room=room, speaker=speaker,
                                reason=self._sure_reason("2b", it2b), latency_ms=ms2b, escalated_from=None, answers=answers)
            return
        shadow = await self._propose(query, it2b, cause, slot="2b", mode="shadow", room=room, speaker=speaker,
                                     reason=SHADOW_REASON, latency_ms=ms2b, escalated_from=None, answers=answers)
        got12 = await self._ask("12b", query, cause)
        if got12 is not None:
            it12, ms12, answers12 = got12
            await self._propose(query, it12, cause, slot="12b", mode="act", room=room, speaker=speaker,
                                reason=self._sure_reason("12b", it12), latency_ms=ms12, escalated_from=shadow.id,
                                answers=answers12)
            return
        await self._propose(query, it2b, cause, slot="2b", mode="act", room=room, speaker=speaker,
                            reason="12B unavailable, using the unsure 2B answer", latency_ms=ms2b,
                            escalated_from=shadow.id, answers=answers)

    async def _cascade_house(self, house: HouseAgent, query: DeciderQuery, cause: Event, room, speaker) -> None:
        """2B inside the Strands house agent's router -> its model (Claude on Bedrock, or the registry's llm)
        when unsure -> 12B / unsure 2B. `house.who` / `house.short` / `house.service` word the reasons."""
        slot = house.name
        from_person = self._from_person(cause)
        published: dict[str, Event] = {}

        def accept(decision: RouteDecision) -> bool:
            return accept_rule(decision, from_person=from_person, threshold=self.settings.act_threshold)

        async def on_decision(decision: RouteDecision) -> None:
            """Runs inside before_invocation, right after the 2B answered and before the model is (not) called."""
            it = interpretation_of(decision)
            if decision.error is not None or it is None:
                await self.log(f"2B decider failed: {decision.error}", cause, "warn")
                return
            if decision.route == "local":
                published["act"] = await self._propose(
                    query, it, cause, slot="2b", mode="act", room=room, speaker=speaker,
                    reason=self._sure_reason("2b", it), latency_ms=decision.latency_ms, escalated_from=None,
                    answers=decision.answers)
            else:
                published["shadow"] = await self._propose(
                    query, it, cause, slot="2b", mode="shadow", room=room, speaker=speaker,
                    reason=f"unsure, asking {house.who}", latency_ms=decision.latency_ms, escalated_from=None,
                    answers=decision.answers)

        out = await house.decide(self.deciders["2b"], query, cause.data, accept=accept, on_decision=on_decision)
        if out.by_router:
            return
        shadow = published.get("shadow")
        escalated_from = shadow.id if shadow else None
        it2b = interpretation_of(out.decision) if out.decision is not None else None
        if out.error is None:
            calls = out.tool_calls or [{"name": "no_action", "input": {}}]
            # a small local model sometimes repeats the same call in its next round: one proposal per distinct call
            distinct: list[dict] = [c for i, c in enumerate(calls) if c not in calls[:i]]
            extra = {"tokens": out.tokens, "tool_calls": out.tool_calls, "reply": out.reply}
            for call in distinct:
                action, confidence, call_room = proposal_from_tool_call(call)
                it = Interpretation(action, confidence, {action: confidence}, {"tool_call": dict(call)},
                                    call_room if call_room != room else None)
                if out.tool_calls:
                    reason = f"{house.short} called {call['name']}: {ACTION_WORDS.get(action, action)}"
                    if call["name"] == "ask_resident":
                        reason += " (unsure, so the house asks first)"
                else:
                    reason = f"{house.short} called no tool: nothing to do"
                await self._propose(query, it, cause, slot=slot, mode="act", room=room, speaker=speaker,
                                    reason=reason, latency_ms=out.latency_ms, escalated_from=escalated_from,
                                    answers=out.decision.answers if out.decision else {}, extra=extra)
            return
        await self.log(f"{house.who} failed: {out.error}", cause, "warn")
        if "12b" in self.deciders and await self._healthy("12b"):
            got12 = await self._ask("12b", query, cause)
            if got12 is not None:
                it12, ms12, answers12 = got12
                await self._propose(query, it12, cause, slot="12b", mode="act", room=room, speaker=speaker,
                                    reason=f"{house.service} failed, asked the 12B; " + self._sure_reason("12b", it12),
                                    latency_ms=ms12, escalated_from=escalated_from, answers=answers12)
                return
        if it2b is not None:
            await self._propose(query, it2b, cause, slot="2b", mode="act", room=room, speaker=speaker,
                                reason=f"{house.who} unavailable, using the unsure 2B answer",
                                latency_ms=out.decision.latency_ms, escalated_from=escalated_from,
                                answers=out.decision.answers)

    # ------------------------------------------------------------------ proposals
    def _sure_reason(self, slot: str, it: Interpretation) -> str:
        return f"{SLOT_LABEL.get(slot, slot)} is {it.confidence:.0%} sure: {ACTION_WORDS.get(it.action, it.action)}"

    async def _propose(self, query: DeciderQuery, it: Interpretation, cause: Event, *, slot: str, mode: str,
                       room: str | None, speaker: dict | None, reason: str, latency_ms: float,
                       escalated_from: str | None, answers: dict[str, Any] | None = None,
                       extra: dict[str, Any] | None = None) -> Event:
        target_room = getattr(it, "room", None) or room
        data: dict[str, Any] = {
            "by": self._by(slot), "mode": mode, "room": target_room, "action": it.action,
            "target": self.home.target_for(it.action, target_room),
            "confidence": round(float(it.confidence), 4),
            "probabilities": {k: round(float(v), 4) for k, v in it.probabilities.items()},
            "question": str(query.questions[query.primary].get("instructions", query.headline)),
            "state_text": query.state, "latency_ms": round(latency_ms, 1), "reason": reason,
            "speaker": speaker, "kind": query.kind, "escalated_from": escalated_from,
            # Decider X-ray
            "route": self._route(slot), "model": self._model_label(slot),
            "request": {"state": query.state, "questions": query.questions},
            "answers": answers or {}, "threshold": float(self.settings.act_threshold),
            "tokens": None, "tool_calls": [], "reply": None,
        }
        if extra:
            data.update(extra)
        return await self.emit(T.PROPOSAL, data, cause)
