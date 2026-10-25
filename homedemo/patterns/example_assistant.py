"""A tiny standalone Strands agent with the DeciderToolRouter in its path.

    python -m homedemo.patterns.example_assistant [--decider-url http://127.0.0.1:8765] [--profile bedrock]
                                                  [--threshold 0.8] ["Kitchen lights off." ...]

Three tools with enum arguments only, so the Strands Decider 2B can pick a complete call on its own;
Claude Haiku 4.5 on Amazon Bedrock is the model that runs only when the 2B is unsure. Needs
`strands-decider serve <2B checkpoint> --port 8765` (loopback) and AWS credentials for Bedrock.
Prints, per message: who decided (local 2B or Claude), the tool call, the confidence, latency and tokens.

This file imports nothing from the rest of homedemo: copy it next to decider_tool_router.py and go.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any, Literal

import httpx
from strands import Agent, tool
from strands.models import BedrockModel

from .decider_tool_router import DeciderToolRouter

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEMO_MESSAGES = [
    "Kitchen lights off.",
    "It's freezing in here.",
    "Put on some jazz in the living room.",
    "My sister said her heater broke last winter.",
    "I can't see a thing in here.",
]


class LocalDecider:
    """The 20-line client for `strands-decider serve`: POST /v1/systemone with {state, questions}."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=60.0)

    async def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        r = await self._client.post(f"{self.url}/v1/systemone", json={"state": state, "questions": questions})
        r.raise_for_status()
        return r.json()["answers"]            # {key: {"type", "choice", "confidence", "probabilities", ...}}

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------- tools (enum args only)
@tool
def set_lights(room: Literal["living", "kitchen", "bedroom"], state: Literal["on", "off", "dim"]) -> str:
    """Switch or dim the lights in a room."""
    return f"lights {state} in the {room}"


@tool
def set_thermostat(direction: Literal["warmer", "cooler"]) -> str:
    """Nudge the thermostat up or down by 1.5 C."""
    return f"thermostat {direction}"


@tool
def play_music(room: Literal["living", "kitchen", "bedroom"], genre: Literal["jazz", "pop", "classical", "news"]) -> str:
    """Play music or the news on the speaker in a room."""
    return f"{genre} playing in the {room}"


SYSTEM_PROMPT = ("You are a home assistant with no wake word. For one overheard sentence, call at most one tool, or "
                 "none if it is not a request. Then reply with one short sentence the speaker says aloud.")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("messages", nargs="*", default=DEMO_MESSAGES)
    ap.add_argument("--decider-url", default="http://127.0.0.1:8765")
    ap.add_argument("--profile", default="bedrock", help="AWS profile for Bedrock ('' = default chain)")
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()

    import boto3

    decider = LocalDecider(args.decider_url)
    router = DeciderToolRouter(decider, threshold=args.threshold)
    model = BedrockModel(model_id=MODEL_ID, max_tokens=150, temperature=0.0,
                         boto_session=boto3.Session(profile_name=args.profile or None, region_name=args.region))
    try:
        for text in args.messages:
            # A fresh Agent per sentence (0.2 ms): a clean conversation and per-call token metrics.
            agent = Agent(model=model, tools=[set_lights, set_thermostat, play_music], interventions=[router],
                          system_prompt=SYSTEM_PROMPT, callback_handler=None)
            t0 = time.perf_counter()
            result = await agent.invoke_async(text)
            ms = (time.perf_counter() - t0) * 1000
            d = router.last
            usage = result.metrics.accumulated_usage
            tokens = usage["inputTokens"] + usage["outputTokens"] if d.route == "llm" else 0
            who = "2B (local)" if d.route == "local" else "Claude on Bedrock"
            call = f"{d.tool}({', '.join(f'{k}={v}' for k, v in d.args.items())})" if d.tool else "no tool"
            print(f"\n> {text}\n  {who:<18} 2B pick: {call} at {d.confidence:.0%} ({d.latency_ms:.0f} ms) | "
                  f"total {ms:.0f} ms, {tokens} LLM tokens" + (f" | error: {d.error}" if d.error else ""))
            print(f"  reply: {str(result).strip()}")
    finally:
        await decider.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
