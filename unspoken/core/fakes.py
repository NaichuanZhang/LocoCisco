"""Test doubles for the pattern, so the wiring can be unit-tested with no Bedrock and no decider server.

ScriptedModel  a strands.models.Model that answers the first model call with scripted tool calls
               (or text) and every later call (after tool results) with a one-line text reply. It
               counts calls and reports token usage like a real provider, so "the model was never
               called" and "tokens were spent" are both testable. `fail=` makes it raise instead,
               like a Bedrock outage.
ScriptedDecider the decider side: answers every `choice` question with the scripted option (or the
               first option) at the given confidence, every `noul` with P(true), in the wire format.
"""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator, AsyncIterable

from strands.models.model import Model

ToolCallScript = list[tuple[str, dict[str, Any]]]


class ScriptedModel(Model):
    def __init__(self, tool_calls: ToolCallScript | None = None, reply: str = "Done.",
                 usage: tuple[int, int] = (120, 12), fail: Exception | None = None) -> None:
        self.tool_calls = list(tool_calls or [])
        self.reply = reply
        self.usage = usage
        self.fail = fail
        self.calls = 0
        self.requests: list[dict[str, Any]] = []

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> dict[str, Any]:
        return {"model_id": "scripted"}

    async def structured_output(self, output_model: Any, prompt: Any, system_prompt: str | None = None,
                                **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        raise NotImplementedError("ScriptedModel has no structured output")
        yield {}  # pragma: no cover

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: str | None = None,
                     **kwargs: Any) -> AsyncIterable[dict[str, Any]]:
        self.calls += 1
        self.requests.append({"messages": messages, "tools": [t["name"] for t in tool_specs or []],
                              "system_prompt": system_prompt})
        if self.fail is not None:
            raise self.fail
        yield {"messageStart": {"role": "assistant"}}
        last = messages[-1] if messages else {}
        answered_tools = any("toolResult" in b for b in last.get("content", []))
        if self.tool_calls and not answered_tools:
            for i, (name, args) in enumerate(self.tool_calls):
                yield {"contentBlockStart": {"contentBlockIndex": i, "start": {"toolUse": {"toolUseId": f"tool-{i}", "name": name}}}}
                yield {"contentBlockDelta": {"contentBlockIndex": i, "delta": {"toolUse": {"input": json.dumps(args)}}}}
                yield {"contentBlockStop": {"contentBlockIndex": i}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": self.reply}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        inp, out = self.usage
        yield {"metadata": {"usage": {"inputTokens": inp, "outputTokens": out, "totalTokens": inp + out},
                            "metrics": {"latencyMs": 1}}}


class ScriptedDecider:
    """`script` maps a question key (or "*") to (option, confidence) for choice questions / P(true) for noul."""

    def __init__(self, script: dict[str, Any] | None = None, confidence: float = 0.9,
                 fail: Exception | None = None) -> None:
        self.script = script or {}
        self.confidence = confidence
        self.fail = fail
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        self.calls.append((state, questions))
        if self.fail is not None:
            raise self.fail
        out: dict[str, dict[str, Any]] = {}
        for key, q in questions.items():
            scripted = self.script.get(key, self.script.get("*"))
            if q["type"] == "noul":
                p = float(scripted) if isinstance(scripted, (int, float)) else 0.5
                out[key] = {"type": "noul", "noul": p, "confidence": abs(2 * p - 1),
                            "probabilities": {"true": p, "false": 1 - p}}
                continue
            options = list(q["criteria"])
            if isinstance(scripted, tuple):
                pick, conf = scripted
            elif isinstance(scripted, str) and scripted in options:
                pick, conf = scripted, self.confidence
            else:
                pick, conf = options[0], self.confidence
            rest = (1 - conf) / max(1, len(options) - 1)
            out[key] = {"type": "choice", "choice": pick, "confidence": conf,
                        "probabilities": {o: (conf if o == pick else rest) for o in options}}
        return out

    async def healthy(self) -> bool:
        return self.fail is None
