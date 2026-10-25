"""In-process async pub/sub bus. Shared core (integrator-owned).

Agents are coroutines subscribed to topic patterns. publish() never blocks on handlers:
each matching handler runs as its own task, so a slow decider call cannot stall sensors.
drain() lets tests and the scenario runner wait until the pipeline has settled.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from .events import LOG_TOPIC, Event, Handler

log = logging.getLogger("unspoken.bus")


@dataclass
class _Sub:
    pattern: str
    handler: Handler
    name: str


def topic_matches(pattern: str, topic: str) -> bool:
    if pattern == "*" or pattern == topic:
        return True
    if pattern.endswith(".*"):
        return topic.startswith(pattern[:-1])
    return False


class Bus:
    def __init__(self, history: int = 5000, trace_path: str | None = None) -> None:
        self._subs: list[_Sub] = []
        self._pending: set[asyncio.Task] = set()
        self.history: deque[Event] = deque(maxlen=history)
        self.errors: list[tuple[str, str, BaseException]] = []  # (subscriber, event id, exc)
        self._trace = open(trace_path, "a", encoding="utf-8") if trace_path else None
        # retain(event) -> False keeps an event out of history and the trace file (it is still delivered).
        # The meeting room uses it so nothing said off the record is kept anywhere.
        self.retain: Callable[[Event], bool] | None = None

    def forget(self, predicate: Callable[[Event], bool]) -> int:
        """Delete every history event matching `predicate` (e.g. the wiped minute of transcript); returns the
        count. Lines already written to a trace file are not rewritten: keep tracing off in real meetings."""
        keep = [e for e in self.history if not predicate(e)]
        dropped = len(self.history) - len(keep)
        if dropped:
            self.history = deque(keep, maxlen=self.history.maxlen)
        return dropped

    def subscribe(self, pattern: str, handler: Handler, *, name: str = "") -> None:
        self._subs.append(_Sub(pattern, handler, name or getattr(handler, "__qualname__", "handler")))

    def unsubscribe_all(self) -> None:
        self._subs.clear()

    def subscription_count(self) -> int:
        return len(self._subs)

    def unsubscribe_from(self, index: int) -> None:
        """Drop every subscription made at or after position `index` (see subscription_count()).
        The runtime uses it to rebuild the agents on reset without touching the sim or the UI."""
        del self._subs[index:]

    async def publish(self, event: Event) -> None:
        if self.retain is None or self.retain(event):
            self.history.append(event)
            if self._trace:
                self._trace.write(json.dumps(event.to_json(), ensure_ascii=False, default=str) + "\n")
                self._trace.flush()
        for sub in list(self._subs):
            if topic_matches(sub.pattern, event.topic):
                task = asyncio.create_task(self._run(sub, event), name=f"{sub.name}<-{event.topic}")
                self._pending.add(task)
                task.add_done_callback(self._pending.discard)

    async def _run(self, sub: _Sub, event: Event) -> None:
        try:
            await sub.handler(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # one broken agent must not take the house down
            self.errors.append((sub.name, event.id, exc))
            log.exception("handler %s failed on %s", sub.name, event.topic)
            if event.topic != LOG_TOPIC:
                await self.publish(Event(LOG_TOPIC, {"agent": sub.name, "level": "warn",
                                                 "text": f"{type(exc).__name__}: {exc}"},
                                         source="bus", ts=event.ts, cause=event.id))

    async def drain(self, timeout: float = 30.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while self._pending:
            left = deadline - loop.time()
            if left <= 0:
                raise TimeoutError(f"bus did not settle: {[t.get_name() for t in self._pending]}")
            await asyncio.wait(set(self._pending), timeout=left)

    async def cancel_pending(self) -> None:
        for t in list(self._pending):
            t.cancel()
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)

    def find(self, topic: str, **match: Any) -> list[Event]:
        """History search for tests: events on `topic` whose data contains every key=value."""
        return [e for e in self.history if e.topic == topic
                and all(e.data.get(k) == v for k, v in match.items())]
