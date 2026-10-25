from __future__ import annotations

from typing import Any

import pytest

from homedemo.bus import Bus
from homedemo.contracts import Event, Settings, T
from homedemo.sim import World, load_home


class Harness:
    """A started World on its own Bus. step()/command() return the events they caused."""

    def __init__(self, *, seed: int = 7, noise: float = 0.0) -> None:
        self.bus = Bus()
        self.world = World(load_home(), self.bus, Settings(), seed=seed, noise=noise)
        self._mark = 0

    async def start(self) -> "Harness":
        await self.world.start()
        await self.bus.drain()
        self._mark = len(self.bus.history)
        return self

    def new(self) -> list[Event]:
        events = list(self.bus.history)[self._mark:]
        self._mark = len(self.bus.history)
        return events

    async def step(self, **step: Any) -> list[Event]:
        self.new()
        await self.world.step(step)
        await self.bus.drain()
        return self.new()

    async def command(self, device_id: str | None, command: str, **args: Any) -> tuple[Event, list[Event]]:
        self.new()
        cmd = Event(T.COMMAND, {"device_id": device_id, "command": command, "args": args, "verdict_id": "v1"},
                    source="agent.actuator", ts=self.world.clock)
        await self.bus.publish(cmd)
        await self.bus.drain()
        return cmd, [e for e in self.new() if e is not cmd]

    async def tick(self, seconds: float = 1.0) -> list[Event]:
        self.new()
        await self.world.tick(seconds)
        await self.bus.drain()
        return self.new()


def only(events: list[Event], topic: str, **match: Any) -> list[Event]:
    return [e for e in events if e.topic == topic and all(e.data.get(k) == v for k, v in match.items())]


def one(events: list[Event], topic: str, **match: Any) -> Event:
    found = only(events, topic, **match)
    assert len(found) == 1, f"want one {topic} {match}, got {[(e.topic, e.data) for e in events]}"
    return found[0]


@pytest.fixture
async def sim() -> Harness:
    return await Harness().start()


@pytest.fixture
def home() -> dict[str, Any]:
    return load_home()
