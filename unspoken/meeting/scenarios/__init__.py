"""Scenario library + runner for the meeting room.

  load_scenarios(dir=None) -> list[dict]   every library/*.json, demo (tag "demo") first, then file-name order
  get(scenario_id) -> dict                 KeyError if unknown
  validate(scenario) -> list[str]          errors against the DSL ([] = valid)
  summary(scenario) -> dict                for MeetingRuntimeAPI.scenarios()
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .expectations import EXPECT_KINDS
from .runner import RUNNER_STEPS, CheckResult, RunResult, ScenarioRunner, describe_step
from .validate import validate

LIBRARY_DIR = Path(__file__).resolve().parent / "library"
DEMO_TAG = "demo"
REGRESSION_TAG = "regression"

__all__ = ["LIBRARY_DIR", "DEMO_TAG", "REGRESSION_TAG", "EXPECT_KINDS", "RUNNER_STEPS", "CheckResult", "RunResult",
           "ScenarioRunner", "describe_step", "load_scenarios", "get", "validate", "summary"]


def load_scenarios(dir: str | Path | None = None) -> list[dict[str, Any]]:
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
    demos = [s for s in scenarios if DEMO_TAG in (s.get("tags") or [])]
    others = [s for s in scenarios if DEMO_TAG not in (s.get("tags") or [])]
    return demos + others


def get(scenario_id: str, dir: str | Path | None = None) -> dict[str, Any]:
    for scenario in load_scenarios(dir):
        if scenario["id"] == scenario_id:
            return scenario
    raise KeyError(scenario_id)


def summary(scenario: dict[str, Any]) -> dict[str, Any]:
    return {"id": scenario.get("id"), "title": scenario.get("title", ""), "blurb": scenario.get("blurb", ""),
            "tags": list(scenario.get("tags") or []), "steps": len(scenario.get("steps") or [])}
