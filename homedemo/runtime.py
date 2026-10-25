"""Composition root: one bus, the simulated home, the agents, the deciders and the scenario runner.

Implements contracts.RuntimeAPI for the web UI. Order of bus subscriptions matters for reset():
the sim and the UI subscribe first, the agents last, so reset can drop and rebuild only the agents
(fresh perception memory, policy caches and open asks) while the UI stream keeps running.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, fields
from typing import Any

from .agents import start_agents
from .agents.house_agent import HouseAgent
from .bus import Bus
from .contracts import AGENTS, Event, Settings, T
from .decider_client import FakeDecider
from .sim import World, load_home
from unspoken.models import load_registry

log = logging.getLogger("homedemo.runtime")

WORLD_STEPS = {"set_time", "move", "activity", "idle", "say", "tv_say", "outside_say", "set_env",
               "set_device", "phone", "leave", "arrive"}


DECIDER_SETTINGS = {"profile", "decider_2b_url", "decider_12b_url", "bedrock_model_id", "bedrock_region",
                    "bedrock_profile"}

# What the intent / scene decider agents do, by settings.escalate_to (shown in the UI's agent graph).
BLURBS: dict[str, dict[str, str]] = {
    "bedrock": {
        "scene_decider": "Asks the Decider 2B what the house should do about what it sees; Claude on Bedrock "
                         "decides when the 2B is unsure.",
        "intent_decider": "Asks the Decider 2B which tool the speaker wants, with no wake word. Claude on Bedrock "
                          "only if unsure.",
    },
    "12b": {
        "scene_decider": "Asks the Decider 2B what the house should do about what it sees; the 12B decides when "
                         "the 2B is unsure.",
        "intent_decider": "Asks the Decider 2B which tool the speaker wants, with no wake word. The 12B only if "
                          "unsure.",
    },
    "llm": {
        "scene_decider": "Asks the Decider 2B what the house should do about what it sees; the profile's LLM "
                         "(local Qwen on vLLM / llama.cpp) decides when the 2B is unsure.",
        "intent_decider": "Asks the Decider 2B which tool the speaker wants, with no wake word. The profile's LLM "
                          "only if unsure.",
    },
}

# Settings -> registry overrides: a decider URL that differs from the Settings default (the --decider-*-url
# flags) wins over the profile's config/models.toml block; the default means "whatever the profile says".
_URL_OVERRIDES = (("decider_2b_url", "decider.url"), ("decider_12b_url", "decider_big.url"))


def registry_overrides(settings: Settings) -> dict[str, str]:
    defaults = Settings()
    return {key: getattr(settings, attr) for attr, key in _URL_OVERRIDES
            if getattr(settings, attr) != getattr(defaults, attr)}


def build_registry(settings: Settings):
    """The models registry for settings.profile (config/models.toml), with the Settings URL overrides."""
    return load_registry(settings.profile, overrides=registry_overrides(settings))


def build_deciders(settings: Settings, registry=None) -> dict[str, Any]:
    """The decider slots, from the models registry of settings.profile:
      "2b"  = role decider, "12b" = role decider_big (absent when the profile disables the role),
      "llm" = the Strands house agent on the registry's llm role (its label/route come from the registry),
      "bedrock" = the house agent on Claude from Settings (as it always was), "fake" = FakeDecider.
    Models are built lazily on first use; nothing here touches a server."""
    reg = registry or build_registry(settings)
    deciders: dict[str, Any] = {"fake": FakeDecider(), "bedrock": HouseAgent(settings)}
    if reg.enabled("decider"):
        deciders["2b"] = reg.decider("decider")
    if reg.enabled("decider_big"):
        deciders["12b"] = reg.decider("decider_big")
    if reg.enabled("llm"):
        route = "bedrock" if reg.spec("llm").provider == "bedrock" else "local"
        deciders["llm"] = HouseAgent(settings, model_factory=lambda _s: reg.strands_model("llm"), name="llm",
                                     label=reg.label("llm"), route=route)
    return deciders


class Runtime:
    def __init__(self, settings: Settings | None = None, *, deciders: dict[str, Any] | None = None,
                 seed: int = 7, noise: float = 0.3, trace_path: str | None = None,
                 tick_seconds: float = 1.0) -> None:
        self.settings = settings or Settings()
        self.home = load_home()
        self.bus = Bus(trace_path=trace_path)
        self.world = World(self.home, self.bus, self.settings, seed=seed, noise=noise)
        self.registry = None if deciders is not None else build_registry(self.settings)
        self.deciders = deciders if deciders is not None else build_deciders(self.settings, self.registry)
        self.agents: list = []
        self._agent_sub_index = 0
        self._tick_seconds = tick_seconds
        self._tick_task: asyncio.Task | None = None
        self._scenario_task: asyncio.Task | None = None
        self._runner = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ lifecycle
    async def start(self, *, ticking: bool = True) -> None:
        from .scenarios import ScenarioRunner  # imported late: optional during unit tests

        await self.world.start()
        self._runner = ScenarioRunner(self.bus, self.world, self.settings, self.home, reset_before_run=False)
        self._start_agents()
        await self.bus.drain(60)
        if ticking and self._tick_seconds > 0:
            self._tick_task = asyncio.create_task(self._tick_loop(), name="runtime.tick")

    def _start_agents(self) -> None:
        self._agent_sub_index = self.bus.subscription_count()
        self.agents = start_agents(self.bus, self.home, self.settings, self.deciders)

    async def _tick_loop(self) -> None:
        while True:
            await asyncio.sleep(self._tick_seconds)
            if self._lock.locked():
                continue
            try:
                await self.world.tick(self._tick_seconds * self.settings.speed)
            except Exception:  # physics must never kill the house
                log.exception("tick failed")

    async def close(self) -> None:
        await self.stop_scenario()
        if self._tick_task:
            self._tick_task.cancel()
        await self.bus.cancel_pending()
        for d in self.deciders.values():
            if hasattr(d, "aclose"):
                await d.aclose()
        if self.registry is not None:
            await self.registry.aclose()

    async def _rebuild_deciders(self) -> None:
        """A decider setting changed (profile, URLs, Bedrock ids): new registry, new slots; drop the slots the
        new profile does not have, close what the old registry opened."""
        old = self.registry
        self.registry = build_registry(self.settings)
        fresh = build_deciders(self.settings, self.registry)
        for slot in ("2b", "12b", "llm"):
            if slot not in fresh:
                self.deciders.pop(slot, None)
        self.deciders.update(fresh)
        if old is not None:
            await old.aclose()

    # ------------------------------------------------------------------ RuntimeAPI
    def snapshot(self) -> dict[str, Any]:
        return self.world.snapshot()

    def agent_graph(self) -> list[dict[str, Any]]:
        busy = {a.name: bool(getattr(a, "busy", False)) for a in self.agents}
        blurbs = BLURBS.get(self.settings.escalate_to, {}) if self.settings.decider_mode == "cascade" else {}
        return [{**a, "blurb": blurbs.get(a["name"], a["blurb"]), "busy": busy.get(a["name"], False)} for a in AGENTS]

    def scenarios(self) -> list[dict[str, Any]]:
        from .scenarios import load_scenarios

        return [{"id": s["id"], "title": s["title"], "blurb": s.get("blurb", ""), "tags": s.get("tags", []),
                 "steps": len(s.get("steps", []))} for s in load_scenarios()]

    async def run_scenario(self, scenario_id: str) -> None:
        from .scenarios import get

        scenario = get(scenario_id)
        if scenario is None:
            raise KeyError(scenario_id)
        await self.reset()
        self._scenario_task = asyncio.create_task(self._runner.run(scenario), name=f"scenario:{scenario_id}")

    async def stop_scenario(self) -> None:
        if self._runner is not None and self._scenario_task and not self._scenario_task.done():
            await self._runner.stop()
            self._scenario_task.cancel()
            await asyncio.gather(self._scenario_task, return_exceptions=True)
        self._scenario_task = None

    async def reset(self) -> None:
        async with self._lock:
            await self.stop_scenario()
            await self.bus.cancel_pending()
            self.bus.unsubscribe_from(self._agent_sub_index)
            self._start_agents()
            await self.world.reset()
            await self.bus.drain(60)
            await self.bus.publish(Event(T.LOG, {"agent": "runtime", "level": "info", "text": "House reset."},
                                         source="runtime", ts=self.world.clock))

    async def step(self, step: dict[str, Any]) -> None:
        if step.get("do") not in WORLD_STEPS:
            raise ValueError(f"not a world step: {step.get('do')!r}")
        await self.world.step(step)

    async def phone_reply(self, request_id: str, answer: str) -> None:
        await self.bus.publish(Event(T.REPLY, {"request_id": request_id, "answer": answer, "by": "web"},
                                     source="phone", ts=self.world.clock))

    async def update_settings(self, **changes: Any) -> Settings:
        known = {f.name for f in fields(Settings)}
        bad = set(changes) - known
        if bad:
            raise ValueError(f"unknown settings: {sorted(bad)}")
        for k, v in changes.items():
            setattr(self.settings, k, v)
        if DECIDER_SETTINGS & set(changes):
            await self._rebuild_deciders()
        await self.bus.publish(Event(T.LOG, {"agent": "runtime", "level": "info",
                                             "text": f"Settings changed: {changes}"},
                                     source="runtime", ts=self.world.clock))
        return self.settings

    def settings_dict(self) -> dict[str, Any]:
        return asdict(self.settings)
