"""Reflex rules: instant plain code, no model. Handles perception.scene triggers person_entered and
env_changed (the scene decider takes activity_changed / idle / hazard, so they never double-propose).

Rule: a person (never a pet) enters a dark room (lux below thresholds.dark_lux) at night with the
lights off -> lights_on, confidence 1.0, mode act. env_changed scenes publish nothing: turning the
lights back on after someone deliberately switched them off would fight the resident."""
from __future__ import annotations

import time

from ..contracts import Event, T
from .base import Agent


class ReflexAgent(Agent):
    name = "reflex"

    def start(self) -> None:
        self.subscribe(T.SCENE, self._on_scene)

    async def _on_scene(self, ev: Event) -> None:
        d = ev.data
        if d.get("trigger") == "hazard":
            await self._hazard(ev)
            return
        if d.get("trigger") != "person_entered":
            return
        facts = d.get("facts") or {}
        if not facts.get("people"):
            return                                             # pets never trigger the reflex
        if facts.get("lights_on") is not False or not self.home.is_dark(facts.get("lux")):
            return
        if not self.home.is_night(str(facts.get("time") or "")):
            return
        room = d.get("room")
        target = self.home.target_for("lights_on", room)
        if target is None:
            return
        await self.emit(T.PROPOSAL, {
            "by": "reflex", "mode": "act", "room": room, "action": "lights_on", "target": target,
            "confidence": 1.0, "probabilities": {"lights_on": 1.0}, "question": "",
            "state_text": d.get("text", ""), "latency_ms": round((time.time() - ev.wall) * 1000, 2),
            "reason": "Someone walked into a dark room at night with the lights off.",
            "speaker": None, "kind": "scene", "escalated_from": None,
            "route": "rule", "model": "reflex rules", "request": {}, "answers": {}, "threshold": 1.0, "tokens": None,
        }, ev)

    async def _hazard(self, ev: Event) -> None:
        """Safety must not depend on a model cutoff: perception's hazard trigger (a child next to a lit
        stove with no adult in the room) alerts a parent at once. The scene decider still answers the
        same scene; the actuator's per-room cooldown keeps it to one alert."""
        d = ev.data
        room = d.get("room")
        await self.emit(T.PROPOSAL, {
            "by": "reflex", "mode": "act", "room": room, "action": "alert_parent",
            "target": self.home.target_for("alert_parent", room),
            "confidence": 1.0, "probabilities": {"alert_parent": 1.0}, "question": "",
            "state_text": d.get("text", ""), "latency_ms": round((time.time() - ev.wall) * 1000, 2),
            "reason": "A child is next to a lit stove with no adult in the room.",
            "speaker": None, "kind": "scene", "escalated_from": None,
            "route": "rule", "model": "reflex rules", "request": {}, "answers": {}, "threshold": 1.0, "tokens": None,
        }, ev)
