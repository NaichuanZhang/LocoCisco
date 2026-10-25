"""ScenarioRunner: plays one scenario against the sim world and the bus, grading `expect` steps.

World steps go to `world.step(step)`; runner steps are wait (real seconds / settings.speed), note (narration),
expect. After every step the runner awaits `bus.drain()`, so an expect watches the events recorded since the
START of the most recent trigger step (not since the expect itself). Positive kinds pass as soon as a
matching event is seen; negative kinds (no_post, no_item) wait the whole window. Progress goes out on
scenario.status as ScenarioStatusData. One scenario at a time.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from ...core.events import BusAPI, Event
from ..contracts import MT, MeetingSettings
from ..sim import WORLD_STEPS
from . import expectations as X

SOURCE = "scenario"
RUNNER_STEPS: frozenset[str] = frozenset({"wait", "note", "expect"})


@dataclass
class CheckResult:
    step_index: int
    ok: bool
    text: str


@dataclass
class RunResult:
    scenario_id: str
    state: str = "finished"          # finished | stopped | error
    passed: int = 0
    failed: int = 0
    checks: list[CheckResult] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.state == "finished" and self.failed == 0


def describe_step(step: dict[str, Any]) -> str:
    do = step.get("do")
    who = str(step.get("who", "")).capitalize()
    if do == "start":
        return f"Meeting starts: {step.get('title', 'Meeting')} (owner {step.get('owner', '?')})"
    if do == "join":
        return f"{who} sits down at {step.get('seat', '?')}" + (f" and checks in as {step['checkin']}" if step.get("checkin") else " (no check-in)")
    if do == "say":
        return f'{who} says "{step.get("text", "")}"'
    if do == "media_say":
        return f'The screen says "{step.get("text", "")}"'
    if do == "enter":
        return f"{who} walks in and takes {step.get('seat', '?')}"
    if do == "leave":
        return f"{who} leaves"
    if do == "scene":
        return f"Camera: room mode {step.get('mode')}"
    if do in ("chat", "react"):
        return f"Discord: {step.get('user', 'someone')} /{step.get('command')} {step.get('args') or ''}".rstrip()
    if do == "clock":
        return f"Clock: {step.get('minutes_left')} min left"
    if do == "end":
        return "Meeting ends"
    if do == "wait":
        return f"Wait {step.get('seconds', 1)} s"
    if do == "note":
        return str(step.get("text", ""))
    if do == "expect":
        return "Expect " + X.describe(step)
    return str(do)


class ScenarioRunner:
    def __init__(self, bus: BusAPI, world: Any, settings: MeetingSettings, *, snapshot: Callable[[], dict[str, Any]] | None = None,
                 drain_timeout: float = 60.0) -> None:
        self.bus = bus
        self.world = world
        self.settings = settings
        self.snapshot_fn = snapshot
        self.drain_timeout = drain_timeout
        self.current: dict[str, Any] | None = None
        self.last_result: RunResult | None = None
        self._events: list[Event] = []
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._stopping = False
        bus.subscribe("*", self._on_event, name="scenario_runner")

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def run(self, scenario: dict[str, Any]) -> RunResult:
        await self.stop()
        async with self._lock:
            self._task = asyncio.current_task()
            self._stopping = False
            try:
                return await self._run(scenario)
            finally:
                self._task = None
                self.current = None

    async def stop(self) -> None:
        task = self._task
        if task is None or task.done() or task is asyncio.current_task():
            return
        self._stopping = True
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    async def _run(self, scenario: dict[str, Any]) -> RunResult:
        result = RunResult(scenario_id=str(scenario.get("id", "?")))
        self.last_result = result
        self.current = scenario
        self._events.clear()
        steps: list[dict[str, Any]] = list(scenario.get("steps") or [])
        index = -1
        try:
            await self._status(result, "started", text=str(scenario.get("title", result.scenario_id)))
            mark = 0
            for index, step in enumerate(steps):
                do = step.get("do")
                await self._status(result, "step", index, describe_step(step), do=do)
                if do == "expect":
                    await self._expect(result, index, step, mark)
                elif do == "wait":
                    await self._sleep(float(step.get("seconds", 1)))
                elif do == "note":
                    pass
                elif do in WORLD_STEPS:
                    mark = len(self._events)
                    await self.world.step({k: v for k, v in step.items() if k != "wait"})
                else:
                    raise ValueError(f"unknown step {do!r}")
                await self._drain()
                if step.get("wait"):
                    await self._sleep(float(step["wait"]))
            await self._status(result, "finished", text=f"{result.passed} passed, {result.failed} failed")
        except asyncio.CancelledError:
            result.state = "stopped"
            await self._status(result, "stopped", max(index, 0), f"stopped at step {max(index, 0)}")
            if not self._stopping:
                raise
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()
        except Exception as exc:
            result.state = "error"
            result.error = f"step {index} ({steps[index].get('do') if 0 <= index < len(steps) else '?'}): {type(exc).__name__}: {exc}"
            await self._status(result, "error", max(index, 0), result.error)
        return result

    async def _expect(self, result: RunResult, index: int, step: dict[str, Any], mark: int) -> None:
        window = X.within(step) / self._speed()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + window
        negative = X.is_negative(step)
        what = X.describe(step)
        while True:
            self._wake.clear()
            seen = self._events[mark:]
            snap = self.snapshot_fn() if (self.snapshot_fn and step.get("what") == "device") else None
            match = X.check(step, seen, snap)
            if match is not None:
                await self._check(result, index, match.ok, f"{what}: {match.text}")
                return
            left = deadline - loop.time()
            if left <= 0:
                if negative:
                    await self._check(result, index, True, f"{what}: nothing seen")
                else:
                    await self._check(result, index, False, f"{what}: not seen; saw {X.summarize(seen)}")
                return
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=left)
            except TimeoutError:
                pass

    async def _on_event(self, event: Event) -> None:
        if self.current is not None:
            self._events.append(event)
            self._wake.set()

    async def _check(self, result: RunResult, index: int, ok: bool, text: str) -> None:
        result.checks.append(CheckResult(index, ok, text))
        if ok:
            result.passed += 1
        else:
            result.failed += 1
        await self._status(result, "expect_pass" if ok else "expect_fail", index, text)

    async def _status(self, result: RunResult, state: str, step_index: int | None = None, text: str = "", **extra: Any) -> None:
        data: dict[str, Any] = {"scenario_id": result.scenario_id, "state": state, "text": text,
                                "passed": result.passed, "failed": result.failed}
        if step_index is not None:
            data["step_index"] = step_index
        data.update({k: v for k, v in extra.items() if v is not None})
        await self.bus.publish(Event(MT.SCENARIO, data, source=SOURCE, ts=float(getattr(self.world, "clock", 0.0) or 0.0)))

    async def _drain(self) -> None:
        try:
            await self.bus.drain(timeout=self.drain_timeout)
        except TimeoutError as exc:
            await self.bus.publish(Event(MT.LOG, {"agent": SOURCE, "level": "warn",
                                                  "text": f"pipeline did not settle in {self.drain_timeout:g} s: {exc}"}, source=SOURCE))

    async def _sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds) / self._speed())

    def _speed(self) -> float:
        return max(float(getattr(self.settings, "speed", 1.0) or 1.0), 0.01)
