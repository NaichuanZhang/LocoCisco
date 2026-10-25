"""DeciderToolRouter on a toy Strands agent: the decider picks, the model is skipped or consulted."""
from __future__ import annotations

from typing import Literal

import pytest
from strands import Agent, tool

from homedemo.patterns import NO_TOOL, DeciderToolRouter, RouteDecision
from homedemo.patterns.fakes import ScriptedDecider, ScriptedModel

CALLS: list[tuple] = []


@tool
def set_light(room: Literal["living", "kitchen"], state: Literal["on", "off"]) -> str:
    """Switch the lights in a room on or off.

    Args:
        room: which room
        state: on or off
    """
    CALLS.append(("set_light", room, state))
    return f"lights {state} in the {room}"


@tool
def set_thermostat(direction: Literal["warmer", "cooler"]) -> str:
    """Nudge the thermostat.

    Args:
        direction: warmer or cooler
    """
    CALLS.append(("set_thermostat", direction))
    return f"{direction} by 1.5 C"


@tool
def send_note(text: str) -> str:
    """Leave a free-text note for the family.

    Args:
        text: the note
    """
    CALLS.append(("send_note", text))
    return "noted"


def make_agent(router: DeciderToolRouter, model: ScriptedModel) -> Agent:
    return Agent(model=model, tools=[set_light, set_thermostat, send_note], interventions=[router],
                 system_prompt="You run a house.", callback_handler=None)


@pytest.fixture(autouse=True)
def _clear_calls():
    CALLS.clear()


async def test_confident_pick_runs_the_tool_and_never_calls_the_model():
    decider = ScriptedDecider({"tool": ("set_light", 0.95), "set_light.room": ("kitchen", 0.9),
                               "set_light.state": ("off", 0.98)})
    model = ScriptedModel([("set_thermostat", {"direction": "warmer"})])
    seen: list[RouteDecision] = []
    router = DeciderToolRouter(decider, threshold=0.8, on_decision=seen.append)
    result = await make_agent(router, model).invoke_async("Kitchen lights off.")
    assert model.calls == 0 and dict(result.metrics.accumulated_usage)["totalTokens"] == 0
    assert CALLS == [("set_light", "kitchen", "off")]
    d = router.last
    assert d is seen[0] and d.route == "local" and d.accepted and d.tool == "set_light"
    assert d.args == {"room": "kitchen", "state": "off"} and d.result == "lights off in the kitchen"
    assert d.confidence == pytest.approx(0.95 * 0.9 * 0.98) and d.error is None
    assert str(result).strip() == "DENIED: Decider 2B picked set_light(room=kitchen, state=off) at 84%; model call skipped"
    # one request carried every question, in the wire format
    assert len(decider.calls) == 1
    state, questions = decider.calls[0]
    assert state == "Kitchen lights off." and set(questions) == {"tool", "set_light.room", "set_light.state", "set_thermostat.direction"}
    assert set(questions["tool"]["criteria"]) == {"set_light", "set_thermostat", "send_note", NO_TOOL}
    assert questions["tool"]["criteria"]["set_light"] == "Switch the lights in a room on or off."
    assert all(q["type"] == "choice" for q in questions.values())
    assert d.request == {"state": state, "questions": questions} and d.answers["tool"]["choice"] == "set_light"
    assert d.latency_ms >= 0 and d.probabilities["set_light"] == 0.95


async def test_no_tool_pick_skips_the_model_too():
    decider = ScriptedDecider({"tool": (NO_TOOL, 0.92)})
    model = ScriptedModel([("set_light", {"room": "living", "state": "on"})])
    router = DeciderToolRouter(decider)
    result = await make_agent(router, model).invoke_async("My sister's heater broke last winter.")
    assert model.calls == 0 and CALLS == [] and router.last.tool is None and router.last.route == "local"
    assert "picked no tool at 92%" in str(result)


async def test_unsure_pick_lets_the_model_decide_and_run_tools():
    decider = ScriptedDecider({"tool": ("set_light", 0.55), "set_light.room": ("living", 0.9),
                               "set_light.state": ("on", 0.9)})
    model = ScriptedModel([("set_light", {"room": "living", "state": "on"})], reply="Lights coming on.")
    router = DeciderToolRouter(decider, threshold=0.8)
    result = await make_agent(router, model).invoke_async("I can't see a thing in here.")
    assert model.calls == 2 and str(result).strip() == "Lights coming on."
    assert CALLS == [("set_light", "living", "on")]                 # run by the model's tool call, not the router
    assert dict(result.metrics.accumulated_usage)["inputTokens"] == 240
    assert router.last.route == "llm" and not router.last.accepted and router.last.tool == "set_light"


async def test_free_text_tool_is_never_auto_called():
    decider = ScriptedDecider({"tool": ("send_note", 0.99)})
    model = ScriptedModel([("send_note", {"text": "buy milk"})])
    router = DeciderToolRouter(decider)
    await make_agent(router, model).invoke_async("Remind everyone to buy milk.")
    assert model.calls == 2 and CALLS == [("send_note", "buy milk")]
    assert router.last.route == "llm" and router.last.accepted and router.last.tool == "send_note"


async def test_decider_failure_fails_open_to_the_model():
    decider = ScriptedDecider(fail=ConnectionError("connection refused"))
    model = ScriptedModel([("set_thermostat", {"direction": "warmer"})])
    router = DeciderToolRouter(decider)
    await make_agent(router, model).invoke_async("It's freezing.")
    assert model.calls == 2 and CALLS == [("set_thermostat", "warmer")]
    assert router.last.route == "llm" and router.last.error == "ConnectionError: connection refused"


async def test_tool_failure_fails_open_to_the_model():
    @tool
    def broken(direction: Literal["up", "down"]) -> str:
        """Always breaks.

        Args:
            direction: up or down
        """
        raise RuntimeError("relay stuck")

    decider = ScriptedDecider({"tool": ("broken", 0.99), "broken.direction": ("up", 0.99)})
    model = ScriptedModel(reply="I could not do that.")
    router = DeciderToolRouter(decider)
    agent = Agent(model=model, tools=[broken], interventions=[router], callback_handler=None)
    await agent.invoke_async("Up please.")
    assert model.calls == 1 and router.last.route == "llm" and "relay stuck" in router.last.error


async def test_execute_false_only_reports_the_pick():
    decider = ScriptedDecider({"tool": ("set_thermostat", 0.9), "set_thermostat.direction": ("cooler", 0.95)})
    model = ScriptedModel()
    router = DeciderToolRouter(decider, execute=False)
    result = await make_agent(router, model).invoke_async("It's a sauna in here.")
    assert model.calls == 0 and CALLS == [] and router.last.route == "local"
    assert router.last.tool == "set_thermostat" and router.last.args == {"direction": "cooler"}
    assert "set_thermostat(direction=cooler)" in str(result)


async def test_pluggable_query_interpret_and_accept():
    decider = ScriptedDecider({"intent": ("Warmer", 0.7)})
    model = ScriptedModel()
    questions = {"intent": {"type": "choice", "instructions": "What does the speaker want?",
                            "criteria": {"Warmer": "cold", "Cooler": "hot", "Nothing": "no request"}}}
    router = DeciderToolRouter(
        decider, threshold=0.8,
        build_query=lambda text: (f"Heard: {text}", questions),
        interpret=lambda answers: ({"Warmer": "set_thermostat", "Cooler": "set_thermostat"}.get(answers["intent"]["choice"]),
                                   {"direction": answers["intent"]["choice"].lower()}, answers["intent"]["confidence"],
                                   answers["intent"]["probabilities"]),
        accept=lambda d: d.confidence >= 0.6)         # a looser rule than the threshold
    await make_agent(router, model).invoke_async("Brr.")
    assert decider.calls[0] == ("Heard: Brr.", questions)
    assert model.calls == 0 and CALLS == [("set_thermostat", "warmer")]
    assert router.last.confidence == 0.7 and router.last.route == "local"


async def test_async_on_decision_is_awaited_before_the_action_applies():
    events: list[str] = []

    async def on_decision(d: RouteDecision) -> None:
        events.append(d.route)

    decider = ScriptedDecider({"tool": (NO_TOOL, 0.9)})
    router = DeciderToolRouter(decider, on_decision=on_decision)
    await make_agent(router, ScriptedModel()).invoke_async("Nice weather.")
    assert events == ["local"] and len(router.log) == 1
