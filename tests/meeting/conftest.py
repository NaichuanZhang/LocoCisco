"""Shared helpers: a Pipeline = MeetingRuntime in sim mode with fake devices, a recording notifier and a
scripted or keyword decider. No model servers, no hardware."""
from __future__ import annotations

from typing import Any

import pytest

from unspoken.core.events import Event
from unspoken.core.fakes import ScriptedDecider, ScriptedModel
from unspoken.meeting.config import load_config, settings_from
from unspoken.meeting.contracts import MT
from unspoken.meeting.fakes import FakeNotifier, LocalRegistry, MeteredNotifier, fake_devices
from unspoken.meeting.runtime import MeetingRuntime

START = {"do": "start", "title": "Pricing sync", "owner": "Priya", "agenda": ["Q3 pricing", "Launch"], "minutes": 30}
JOIN_PRIYA = {"do": "join", "who": "priya", "seat": "s1", "voice": "voice-A", "checkin": "Priya"}
JOIN_SAM = {"do": "join", "who": "sam", "seat": "s2", "voice": "voice-B", "checkin": "Sam"}


class BrokenDecider:
    name = "broken"

    def __init__(self) -> None:
        self.calls = 0

    async def healthy(self) -> bool:
        return False

    async def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        self.calls += 1
        raise ConnectionError("connection refused")


class Pipeline:
    def __init__(self, rt: MeetingRuntime, notifier: FakeNotifier) -> None:
        self.rt = rt
        self.bus = rt.bus
        self.state = rt.state
        self.notifier = notifier
        # Tests read everything that was delivered. bus.history is NOT used: it deliberately drops what was said
        # off the record (see unspoken/meeting/reflex.py and Bus.retain/forget).
        self.stream: list[Event] = []

        async def _record(ev: Event) -> None:
            self.stream.append(ev)

        rt.bus.subscribe("*", _record, name="test.stream")

    @property
    def devices(self) -> dict[str, Any]:
        return self.rt.devices

    async def steps(self, *steps: dict[str, Any]) -> None:
        for s in steps:
            await self.rt.world.step(s)
            await self.bus.drain(30)
        self.state = self.rt.state

    async def say(self, who: str, text: str, lang: str = "en") -> None:
        await self.steps({"do": "say", "who": who, "text": text, "lang": lang})

    async def publish(self, topic: str, data: dict[str, Any], source: str = "test", ts: float | None = None) -> Event:
        ev = Event(topic, data, source=source, ts=ts if ts is not None else self.rt.world.clock)
        await self.bus.publish(ev)
        await self.bus.drain(30)
        return ev

    def events(self, topic: str, **match: Any) -> list[Event]:
        return [e for e in self.stream if e.topic == topic and all(e.data.get(k) == v for k, v in match.items())]

    def proposals(self, **match: Any) -> list[dict[str, Any]]:
        return [e.data for e in self.events(MT.PROPOSAL, **match)]

    def verdicts(self, **match: Any) -> list[dict[str, Any]]:
        return [e.data for e in self.events(MT.VERDICT, **match)]

    def items(self, **match: Any) -> list[dict[str, Any]]:
        return [e.data for e in self.events(MT.ITEM, **match)]

    def utterances(self) -> list[dict[str, Any]]:
        return [e.data for e in self.events(MT.UTTERANCE)]

    def logs(self, level: str | None = None) -> list[str]:
        return [e.data["text"] for e in self.events(MT.LOG) if level is None or e.data.get("level") == level]

    def posts(self, channel: str | None = None) -> list[dict[str, Any]]:
        return [p for p in self.notifier.posts if channel is None or p["channel"] == channel]

    async def close(self) -> None:
        await self.rt.close()


def make_registry(decider: Any = None, *, llm_model: Any = None, embedder: Any = None) -> LocalRegistry:
    deciders = {"decider": decider} if decider is not None else None
    return LocalRegistry("fake", deciders=deciders, llm_model=llm_model, embedder=embedder)


@pytest.fixture
def config() -> dict[str, Any]:
    return load_config()


@pytest.fixture
async def make_pipeline(config):
    made: list[Pipeline] = []

    async def _make(decider: Any = None, *, llm_model: ScriptedModel | None = None, embedder: Any = None,
                    registry: Any = None, start: bool = True, **setting_changes: Any) -> Pipeline:
        settings = settings_from(config, **setting_changes)
        reg = registry or make_registry(decider, llm_model=llm_model, embedder=embedder)
        notifier = FakeNotifier()
        rt = MeetingRuntime(profile="fake", mode="sim", registry=reg, config=config, settings=settings,
                            devices=fake_devices(config), notifier=None, echo=False, tick_seconds=0)
        rt.notifier = MeteredNotifier(notifier, on_egress=rt._on_egress)
        await rt.start(ticking=False)
        p = Pipeline(rt, notifier)
        if start:
            await p.steps(START, JOIN_PRIYA, JOIN_SAM)
        made.append(p)
        return p

    yield _make
    for p in made:
        await p.close()


def scripted(kind: str = "Nothing", conf: float = 0.9, owner: str | None = None, room: str = "none",
             sensitive: float = 0.05) -> ScriptedDecider:
    """A decider that answers every utterance the same way (kind label, owner, room choice, sensitive P)."""
    script: dict[str, Any] = {"kind": (kind, conf), "room": (room, 0.9), "sensitive": sensitive}
    script["owner"] = (owner, 0.9) if owner else ("No one named", 0.9)
    return ScriptedDecider(script)
