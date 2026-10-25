"""Intent decider: asks the Decider 2B which tool the speaker wants, with no wake word. When the 2B is
unsure, Claude on Bedrock (or the 12B, settings.escalate_to) decides; see agents/cascade.py. TV and
outside voices are still interpreted so the UI can show the policy agent denying them."""
from __future__ import annotations

from ..contracts import Event, T
from ..decider.questions import build_speech_query
from .cascade import DeciderAgent


class IntentDeciderAgent(DeciderAgent):
    name = "intent_decider"

    def start(self) -> None:
        self.subscribe(T.UTTERANCE, self._on_utterance)

    async def _on_utterance(self, ev: Event) -> None:
        if not str(ev.data.get("text") or "").strip():
            return
        query = build_speech_query(ev.data)
        await self.decide(query, ev, room=ev.data.get("room"), speaker=ev.data.get("speaker"))
