"""The meeting agents on one bus, in pipeline order (see contracts.MT):

  bookkeeper  meeting.status / meeting.clock / room.env -> MeetingState (first, so every other agent reads fresh state)
  fusion      audio.segment + video.speakers -> meeting.utterance
  reflex      "off the record", check-ins, slides scene -> meeting.privacy / proposals (plain code)
  decider     meeting.utterance -> meeting.proposal (Strands Decider, local LLM when unsure)
  policy      meeting.proposal -> meeting.verdict (plain code)
  actuator    meeting.verdict -> devices, items, Notifier (egress-metered), screen cards
  screen      everything -> screen.state
"""
from __future__ import annotations

from typing import Any

from ..core.events import BusAPI, Event
from .actuator import ActuatorAgent
from .base import Agent, MeetingState
from .cascade import DeciderAgent
from .contracts import MT, MeetingSettings
from .docs import DocIndex
from .fusion import FusionAgent
from .policy import PolicyAgent
from .reflex import ReflexAgent
from .screen import ScreenAgent

AGENTS: list[dict[str, str]] = [
    {"name": "fusion", "label": "Fusion", "blurb": "Matches each voice to the face that was talking and the name from the check-in."},
    {"name": "reflex", "label": "Reflex rules", "blurb": "Plain code: 'off the record' wipes and pauses the notes in under a millisecond."},
    {"name": "decider", "label": "Decider", "blurb": "One typed request per sentence to the Strands Decider; the local LLM only when it is unsure."},
    {"name": "policy", "label": "Policy", "blurb": "Plain code: risk table, guests, privacy; the owner approves anything high-risk."},
    {"name": "actuator", "label": "Actuator", "blurb": "Lamp, fan, Discord posts (every byte metered), screen cards."},
    {"name": "screen", "label": "Screen", "blurb": "Named captions, agenda and timer, banners, cards."},
]


class Bookkeeper(Agent):
    name = "bookkeeper"

    def start(self) -> None:
        self.subscribe(MT.STATUS, self._on_status)
        self.subscribe(MT.CLOCK, self._on_clock)
        self.subscribe(MT.ENV, self._on_env)
        self.subscribe("*", self._on_any)

    async def _on_any(self, ev: Event) -> None:
        if ev.ts and ev.ts > self.state.clock and ev.topic != MT.LOG:
            self.state.clock = ev.ts

    async def _on_status(self, ev: Event) -> None:
        d = ev.data
        st = d.get("state")
        if st == "started":
            self.state.status = {"state": "started", "title": d.get("title") or "Meeting", "topic": d.get("topic"),
                                 "owner": d.get("owner"), "ends_at": d.get("ends_at")}
            self.state.agenda = list(d.get("agenda") or [])
            if not self.state.status.get("topic") and self.state.agenda:
                self.state.status["topic"] = self.state.agenda[0]
            self.state.items.clear()
            self.state.started_at = ev.ts
            self.state.minutes_left = d.get("minutes")
            self.state.total_minutes = d.get("minutes")
            self.state.scene_mode = "talk"
        elif st == "topic":
            self.state.status["topic"] = d.get("topic")
            if d.get("topic") and d["topic"] not in self.state.agenda:
                self.state.agenda.append(d["topic"])
        elif st == "ended":
            self.state.status["state"] = "ended"
            self.state.status["ends_at"] = None
            self.state.minutes_left = None

    async def _on_clock(self, ev: Event) -> None:
        if "minutes_left" in ev.data:
            self.state.minutes_left = ev.data.get("minutes_left")
        if ev.data.get("ends_at"):
            self.state.status["ends_at"] = ev.data["ends_at"]

    async def _on_env(self, ev: Event) -> None:
        if ev.data.get("device_id"):
            self.state.devices[str(ev.data["device_id"])] = dict(ev.data)


def start_agents(bus: BusAPI, state: MeetingState, settings: MeetingSettings, config: dict[str, Any], *, registry: Any,
                 devices: dict[str, Any], notifier: Any, docs: DocIndex, llm_factory: Any = None) -> list[Agent]:
    actuator = ActuatorAgent(bus, state, settings, config, devices=devices, notifier=notifier, docs=docs, llm_factory=llm_factory)
    agents: list[Agent] = [
        Bookkeeper(bus, state, settings, config),
        FusionAgent(bus, state, settings, config),
        ReflexAgent(bus, state, settings, config),
        DeciderAgent(bus, state, settings, config, registry),
        PolicyAgent(bus, state, settings, config),
        actuator,
        ScreenAgent(bus, state, settings, config),
    ]
    for a in agents:
        a.start()
    return agents
