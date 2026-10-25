"""Event envelope and bus protocol shared by every LocoCisco domain (home demo, meeting room).

Payloads are plain JSON-able dicts so the web UI can stream them unchanged.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Protocol

LOG_TOPIC = "trace.log"   # free-text explanation any agent may emit; the bus reports handler failures here


def _new_id() -> str:
    return uuid.uuid4().hex[:10]


@dataclass
class Event:
    topic: str
    data: dict[str, Any]
    source: str                    # who published: "sim.camera", "agent.policy", "web", ...
    ts: float = 0.0                # simulated clock seconds (World.clock); 0 if unknown
    id: str = field(default_factory=_new_id)
    cause: str | None = None       # id of the event that triggered this one (UI draws the chain)
    wall: float = field(default_factory=time.time)  # real time, for latency display

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


Handler = Callable[[Event], Awaitable[None]]


class BusAPI(Protocol):
    def subscribe(self, pattern: str, handler: Handler, *, name: str = "") -> None:
        """pattern: exact topic ("intent.proposal"), prefix wildcard ("sensor.*"), or "*"."""

    async def publish(self, event: Event) -> None: ...

    async def drain(self, timeout: float = 30.0) -> None:
        """Wait until every handler task started so far (and their children) finished."""
