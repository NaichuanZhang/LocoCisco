"""Keyword baseline: what an old-style keyword system would do with the same words, shown next to
the decider for comparison. Deliberately naive: lowercase substring match, no negation handling
("don't turn off the TV" -> tv_off). Always mode shadow, so the policy never executes it."""
from __future__ import annotations

import time

from ..contracts import Event, T
from .base import Agent

# (phrase, action); the phrase found earliest in the text wins, ties go to list order.
KEYWORDS: list[tuple[str, str]] = [
    ("turn on the lights", "lights_on"), ("turn on the light", "lights_on"),
    ("lights on", "lights_on"), ("light on", "lights_on"), ("开灯", "lights_on"),
    ("turn off the lights", "lights_off"), ("turn off the light", "lights_off"),
    ("lights off", "lights_off"), ("light off", "lights_off"), ("关灯", "lights_off"),
    ("turn on the tv", "tv_on"), ("tv on", "tv_on"), ("开电视", "tv_on"),
    ("turn off the tv", "tv_off"), ("tv off", "tv_off"), ("关电视", "tv_off"),
    ("warmer", "warmer"), ("cooler", "cooler"),
    ("unlock", "unlock_door"),
]


def keyword_match(text: str) -> tuple[str, str | None]:
    """-> (action, matched phrase); ("none", None) when nothing matches."""
    low = text.lower()
    best: tuple[int, int, str, str] | None = None
    for order, (phrase, action) in enumerate(KEYWORDS):
        pos = low.find(phrase)
        if pos >= 0 and (best is None or (pos, order) < best[:2]):
            best = (pos, order, action, phrase)
    return (best[2], best[3]) if best else ("none", None)


class KeywordAgent(Agent):
    name = "keyword"

    def start(self) -> None:
        self.subscribe(T.UTTERANCE, self._on_utterance)

    async def _on_utterance(self, ev: Event) -> None:
        if not self.settings.keyword_shadow:
            return
        d = ev.data
        action, phrase = keyword_match(str(d.get("text") or ""))
        room = d.get("room")
        await self.emit(T.PROPOSAL, {
            "by": "keyword", "mode": "shadow", "room": room, "action": action,
            "target": self.home.target_for(action, room), "confidence": 1.0,
            "probabilities": {action: 1.0}, "question": "", "state_text": str(d.get("text") or ""),
            "latency_ms": round((time.time() - ev.wall) * 1000, 2),
            "reason": f"keyword match: '{phrase}'" if phrase else "no keyword matched",
            "speaker": d.get("speaker"), "kind": "speech", "escalated_from": None,
            "route": "rule", "model": "keyword rules", "request": {}, "answers": {}, "threshold": 1.0, "tokens": None,
        }, ev)
