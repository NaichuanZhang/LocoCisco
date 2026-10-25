"""Scenario library + runner (workstream "scenarios").

Public API
  load_scenarios(dir=None) -> list[dict]      every library/*.json, heroes (tag "hero") first,
                                              then file-name order
  get(scenario_id, dir=None) -> dict          KeyError if unknown
  validate(scenario, home) -> list[str]       errors against the DSL and home.json ids ([] = valid)
  summary(scenario) -> dict                   {"id","title","blurb","tags","steps": n} for RuntimeAPI.scenarios()
  ScenarioRunner(bus, world, settings, home)  see runner.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .expectations import EXPECT_KINDS, OUTCOME_ACCEPT
from .homeindex import HomeIndex, load_home
from .runner import CheckResult, RunResult, ScenarioRunner
from .steps import RUNNER_STEPS, WORLD_STEPS
from .validate import validate

LIBRARY_DIR = Path(__file__).resolve().parent / "library"
HERO_TAG = "hero"
TOUR_TAG = "tour"            # the one hero that is the 3-minute run order (hero_00)
REGRESSION_TAG = "regression"
RESEARCH_TAG = "research"    # res_*.json: cases from the f2 research seeds and the eval report's failures

__all__ = [
    "LIBRARY_DIR", "HERO_TAG", "TOUR_TAG", "REGRESSION_TAG", "RESEARCH_TAG", "EXPECT_KINDS", "OUTCOME_ACCEPT",
    "RUNNER_STEPS", "WORLD_STEPS",
    "HomeIndex", "load_home", "load_scenarios", "get", "validate", "summary",
    "ScenarioRunner", "RunResult", "CheckResult",
]


def load_scenarios(dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Read every *.json in the library. Heroes first; within a group, file-name order (the files are
    numbered so the hero story arc is deliberate). Raises ValueError on a file that is not a scenario."""
    folder = Path(dir) if dir else LIBRARY_DIR
    scenarios: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.json")):
        with open(path, encoding="utf-8") as fh:
            try:
                scenario = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}: not valid JSON ({exc})") from exc
        if not isinstance(scenario, dict) or not isinstance(scenario.get("id"), str):
            raise ValueError(f"{path.name}: a scenario file must be an object with an 'id'")
        scenarios.append(scenario)
    heroes = [s for s in scenarios if HERO_TAG in (s.get("tags") or [])]
    others = [s for s in scenarios if HERO_TAG not in (s.get("tags") or [])]
    return heroes + others


def get(scenario_id: str, dir: str | Path | None = None) -> dict[str, Any]:
    for scenario in load_scenarios(dir):
        if scenario["id"] == scenario_id:
            return scenario
    raise KeyError(scenario_id)


def summary(scenario: dict[str, Any]) -> dict[str, Any]:
    return {"id": scenario.get("id"), "title": scenario.get("title", ""), "blurb": scenario.get("blurb", ""),
            "tags": list(scenario.get("tags") or []), "steps": len(scenario.get("steps") or [])}
