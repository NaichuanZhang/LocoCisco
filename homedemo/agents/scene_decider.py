"""Scene decider: asks the Decider 2B what the house should do about what it sees; Claude on Bedrock (or
the 12B) when it is unsure. Handles the perception.scene triggers activity_changed, idle and hazard
(reflex owns person_entered/env_changed)."""
from __future__ import annotations

from ..contracts import Event, T
from ..decider.questions import build_scene_query
from .cascade import DeciderAgent

SCENE_TRIGGERS = {"activity_changed", "idle", "hazard"}


class SceneDeciderAgent(DeciderAgent):
    name = "scene_decider"

    def start(self) -> None:
        self.subscribe(T.SCENE, self._on_scene)

    async def _on_scene(self, ev: Event) -> None:
        if ev.data.get("trigger") not in SCENE_TRIGGERS or not ev.data.get("text"):
            return
        query = build_scene_query(ev.data)
        await self.decide(query, ev, room=ev.data.get("room"), speaker=None)
