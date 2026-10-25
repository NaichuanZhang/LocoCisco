"""The pipeline agents (contracts.AGENTS). They talk only through the bus; home.json is read as
configuration, never as live world state.

    agents = start_agents(bus, home, settings, deciders)   # deciders: {"2b"|"12b"|"fake": Decider, "bedrock": HouseAgent}
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import BusAPI, Settings
from ..decider_client import Decider
from .actuator import ActuatorAgent
from .base import Agent, HomeConfig, load_home
from .house_agent import HouseAgent
from .intent_decider import IntentDeciderAgent
from .keyword import KeywordAgent
from .perception import PerceptionAgent
from .policy import PolicyAgent
from .reflex import ReflexAgent
from .scene_decider import SceneDeciderAgent
from .voice import VoiceAgent

__all__ = ["Agent", "HomeConfig", "load_home", "start_agents", "ActuatorAgent", "HouseAgent", "IntentDeciderAgent",
           "KeywordAgent", "PerceptionAgent", "PolicyAgent", "ReflexAgent", "SceneDeciderAgent", "VoiceAgent"]


def start_agents(bus: BusAPI, home: dict[str, Any] | HomeConfig | str | Path, settings: Settings,
                 deciders: dict[str, Any]) -> list[Agent]:
    """Build every agent in contracts.AGENTS order, subscribe them to the bus and return them."""
    cfg = home if isinstance(home, HomeConfig) else HomeConfig(home)
    agents: list[Agent] = [
        PerceptionAgent(bus, cfg, settings),
        VoiceAgent(bus, cfg, settings),
        ReflexAgent(bus, cfg, settings),
        SceneDeciderAgent(bus, cfg, settings, deciders),
        IntentDeciderAgent(bus, cfg, settings, deciders),
        KeywordAgent(bus, cfg, settings),
        PolicyAgent(bus, cfg, settings),
        ActuatorAgent(bus, cfg, settings),
    ]
    for agent in agents:
        agent.start()
    return agents
