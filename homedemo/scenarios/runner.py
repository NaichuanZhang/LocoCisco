"""ScenarioRunner: plays one scenario against the world and the bus, grading `expect` steps.

How a run works
  * WORLD steps go to `world.step(step)` (minus the runner-only "wait" key).
  * RUNNER steps: wait (real seconds / settings.speed), note (narration only), reply (answers the
    newest open phone ask with notify.reply {request_id, answer, by: "scenario"}), expect.
  * After every step the runner awaits `bus.drain()`, so by the time an expect runs the pipeline has
    usually settled. An expect therefore watches the bus events recorded since the START of the most
    recent trigger step (any world step, or a reply that was actually sent) - not since the expect
    itself - otherwise it would miss what the pipeline did during that drain. Positive kinds pass as
    soon as a matching event is seen (or, for `device`, when the world snapshot already shows the
    state); `no_command` waits the whole window (within / speed) and passes if nothing was issued.
  * Progress goes out on T.SCENARIO as ScenarioStatusData: started, step (text = narration, plus
    "do" = the step kind so the UI can style notes), expect_pass / expect_fail (text = the reason),
    finished (passed / failed counts), stopped, error.
  * One scenario at a time: run() stops a running one first; stop() cancels cleanly.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ..contracts import BusAPI, Event, Settings, T, WorldAPI
from . import expectations as X
from .homeindex import HomeIndex, load_home
from .steps import WORLD_STEPS, describe_step

SOURCE = "scenario"


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


class ScenarioRunner:
    def __init__(self, bus: BusAPI, world: WorldAPI, settings: Settings,
                 home: dict[str, Any] | HomeIndex | None = None, *,
                 reset_before_run: bool = True, drain_timeout: float = 30.0) -> None:
        self.bus = bus
        self.world = world
        self.settings = settings
        self.home = home if isinstance(home, HomeIndex) else HomeIndex(home or load_home())
        self.reset_before_run = reset_before_run
        self.drain_timeout = drain_timeout
        self.current: dict[str, Any] | None = None
        self.last_result: RunResult | None = None
        self._events: list[Event] = []            # every bus event during the current run
        self._open_asks: dict[str, Event] = {}    # request_id -> notify.ask not yet replied to
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._stopping = False
        bus.subscribe("*", self._on_event, name="scenario_runner")

    # ------------------------------------------------------------------ public
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

    def open_asks(self) -> list[dict[str, Any]]:
        return [dict(e.data) for e in self._open_asks.values()]

    # ------------------------------------------------------------------ run loop
    async def _run(self, scenario: dict[str, Any]) -> RunResult:
        result = RunResult(scenario_id=str(scenario.get("id", "?")))
        self.last_result = result
        self.current = scenario
        self._events.clear()
        self._open_asks.clear()
        steps: list[dict[str, Any]] = list(scenario.get("steps") or [])
        index = -1
        try:
            if self.reset_before_run:
                await self.world.reset()
                await self._drain()
            await self._status(result, "started", text=str(scenario.get("title", result.scenario_id)))
            mark = 0  # index into self._events where the current trigger step started
            for index, step in enumerate(steps):
                do = step.get("do")
                await self._status(result, "step", index, describe_step(step, self.home), do=do)
                if do == "expect":
                    await self._expect(result, index, step, mark)
                elif do == "wait":
                    await self._sleep(float(step.get("seconds", 1)))
                elif do == "note":
                    pass
                elif do == "reply":
                    start = len(self._events)
                    if await self._reply(result, index, step):
                        mark = start
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
                task.uncancel()  # we asked for this cancellation ourselves; leave the task in a clean state
        except Exception as exc:  # a broken step must not take the runtime down; the UI gets the reason
            result.state = "error"
            result.error = f"step {index} ({steps[index].get('do') if 0 <= index < len(steps) else '?'}): " \
                           f"{type(exc).__name__}: {exc}"
            await self._status(result, "error", max(index, 0), result.error)
        return result

    async def _expect(self, result: RunResult, index: int, step: dict[str, Any], mark: int) -> None:
        window = X.within(step) / self._speed()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + window
        negative = X.is_negative(step)
        what = X.describe(step, self.home)
        while True:
            self._wake.clear()
            seen = self._events[mark:]
            match = X.check(step, seen, self.home, self._snapshot() if step.get("what") == "device" else None)
            if match is not None:
                await self._check(result, index, match.ok, f"{what}: {match.text}")
                return
            left = deadline - loop.time()
            if left <= 0:
                if negative:
                    await self._check(result, index, True, f"{what}: nothing issued")
                else:
                    await self._check(result, index, False, f"{what}: not seen; saw {X.summarize(seen)}")
                return
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=left)
            except TimeoutError:
                pass

    async def _reply(self, result: RunResult, index: int, step: dict[str, Any]) -> bool:
        """Answer the newest open ask (optionally only one addressed `to`). Returns True if sent."""
        answer = str(step.get("answer", "Yes"))
        to = step.get("to")
        asks = [e for e in self._open_asks.values() if to is None or e.data.get("to") == to]
        if not asks:
            if step.get("if_asked"):
                await self._status(result, "step", index, "no open phone ask, nothing to answer", do="reply")
                return False
            await self._check(result, index, False, f'reply "{answer}": no open phone ask to answer')
            return False
        ask = asks[-1]
        options = ask.data.get("options") or []
        if options and answer not in options:
            await self._check(result, index, False,
                              f'reply "{answer}": not one of the ask\'s options {options}')
            return False
        request_id = str(ask.data.get("request_id"))
        self._open_asks.pop(request_id, None)
        await self.bus.publish(Event(T.REPLY, {"request_id": request_id, "answer": answer, "by": SOURCE},
                                     source=SOURCE, ts=self._clock(), cause=ask.id))
        return True

    # ------------------------------------------------------------------ helpers
    async def _on_event(self, event: Event) -> None:
        if event.topic == T.ASK and event.data.get("request_id"):
            self._open_asks[str(event.data["request_id"])] = event
        elif event.topic == T.REPLY:
            self._open_asks.pop(str(event.data.get("request_id")), None)
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

    async def _status(self, result: RunResult, state: str, step_index: int | None = None,
                      text: str = "", **extra: Any) -> None:
        data: dict[str, Any] = {"scenario_id": result.scenario_id, "state": state, "text": text,
                                "passed": result.passed, "failed": result.failed}
        if step_index is not None:
            data["step_index"] = step_index
        data.update({k: v for k, v in extra.items() if v is not None})
        await self.bus.publish(Event(T.SCENARIO, data, source=SOURCE, ts=self._clock()))

    async def _drain(self) -> None:
        try:
            await self.bus.drain(timeout=self.drain_timeout)
        except TimeoutError as exc:
            await self.bus.publish(Event(T.LOG, {"agent": SOURCE, "level": "warn",
                                                 "text": f"pipeline did not settle in {self.drain_timeout:g} s: {exc}"},
                                         source=SOURCE, ts=self._clock()))

    async def _sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds) / self._speed())

    def _speed(self) -> float:
        return max(float(getattr(self.settings, "speed", 1.0) or 1.0), 0.01)

    def _clock(self) -> float:
        try:
            return float(getattr(self.world, "clock", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _snapshot(self) -> dict[str, Any] | None:
        snap = getattr(self.world, "snapshot", None)
        if not callable(snap):
            return None
        try:
            out = snap()
        except Exception:
            return None
        return out if isinstance(out, dict) else None
