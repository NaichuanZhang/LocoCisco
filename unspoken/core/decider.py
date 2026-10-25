"""Decider backends (shared core, integrator-owned).

HttpDecider talks to `strands-decider serve` (POST /v1/systemone) on loopback.
FakeDecider is deterministic for unit tests and for running the demo with no model.

Wire format of a question (same as the server):
  {"type": "choice", "instructions": "...", "criteria": {"Option": "description", ...}}
  {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}}   # criteria optional
  {"type": "score",  "instructions": "...", "criteria": ["low", ..., "high"]}
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass
class DeciderAnswer:
    type: str                              # "choice" | "noul" | "score"
    choice: str | None = None              # choice
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0                # choice/score confidence; for noul = |2p-1|
    noul: float | None = None              # P(true) for noul
    score: float | None = None
    latency_ms: float = 0.0                # whole request, shared by all questions in it
    model: str = ""


class Decider(Protocol):
    name: str  # "decider-2b" | "decider-12b" | "fake"

    async def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, DeciderAnswer]: ...

    async def healthy(self) -> bool: ...


def _parse(name: str, raw: dict[str, Any], latency_ms: float, model: str) -> DeciderAnswer:
    t = raw.get("type")
    if t == "noul":
        p = float(raw["noul"])
        return DeciderAnswer("noul", noul=p, confidence=abs(2 * p - 1),
                             probabilities={"true": p, "false": 1 - p}, latency_ms=latency_ms, model=model)
    if t == "score":
        return DeciderAnswer("score", score=float(raw["score"]), confidence=float(raw.get("confidence", 0)),
                             probabilities=raw.get("probabilities", {}), latency_ms=latency_ms, model=model)
    return DeciderAnswer("choice", choice=raw["choice"], confidence=float(raw["confidence"]),
                         probabilities={k: float(v) for k, v in raw["probabilities"].items()},
                         latency_ms=latency_ms, model=model)


class HttpDecider:
    def __init__(self, name: str, base_url: str, timeout: float = 60.0, max_concurrency: int = 1) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)
        # CPU inference is thread-safe server-side, but one-at-a-time keeps latency numbers honest.
        self._sem = asyncio.Semaphore(max_concurrency)

    async def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, DeciderAnswer]:
        async with self._sem:
            t0 = time.perf_counter()
            r = await self._client.post(f"{self.base_url}/v1/systemone",
                                        json={"state": state, "questions": questions})
            ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        body = r.json()
        model = body.get("model", self.name)
        return {k: _parse(k, v, ms, model) for k, v in body["answers"].items()}

    async def healthy(self) -> bool:
        try:
            r = await self._client.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


_WORD = re.compile(r"[\w']+|[\u4e00-\u9fff]")


class FakeDecider:
    """Deterministic stand-in.

    script: list of (substring_of_state, question_key_or_*, answer) checked in order; answer is
    an option name (choice), a float P(true) (noul), or (option, confidence). Otherwise picks the
    option whose name+description shares the most words with the state, at confidence 0.6.
    """

    def __init__(self, name: str = "fake", script: list[tuple[str, str, Any]] | None = None,
                 latency_ms: float = 5.0) -> None:
        self.name = name
        self.script = script or []
        self.latency_ms = latency_ms
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def healthy(self) -> bool:
        return True

    async def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, DeciderAnswer]:
        self.calls.append((state, questions))
        await asyncio.sleep(self.latency_ms / 1000)
        out: dict[str, DeciderAnswer] = {}
        for key, q in questions.items():
            out[key] = self._answer(state, key, q)
        return out

    def _answer(self, state: str, key: str, q: dict[str, Any]) -> DeciderAnswer:
        scripted = next((a for s, k, a in self.script if s in state and k in (key, "*")), None)
        if q["type"] == "noul":
            # A "*" script line aimed at a choice (e.g. ("Lights on", 0.95)) must not break a yes/no gate.
            p = float(scripted) if isinstance(scripted, (int, float)) else 0.5
            return DeciderAnswer("noul", noul=p, confidence=abs(2 * p - 1),
                                 probabilities={"true": p, "false": 1 - p}, latency_ms=self.latency_ms, model=self.name)
        opts = list(q["criteria"]) if isinstance(q["criteria"], dict) else list(q["criteria"])
        if scripted is not None:
            pick, conf = scripted if isinstance(scripted, tuple) else (scripted, 0.95)
        else:
            words = set(w.lower() for w in _WORD.findall(state))
            def overlap(o: str) -> int:
                desc = q["criteria"][o] if isinstance(q["criteria"], dict) else ""
                return len(words & set(w.lower() for w in _WORD.findall(f"{o} {desc or ''}")))
            pick, conf = max(opts, key=overlap), 0.6
        rest = (1 - conf) / max(1, len(opts) - 1)
        probs = {o: (conf if o == pick else rest) for o in opts}
        if q["type"] == "score":
            idx = opts.index(pick)
            return DeciderAnswer("score", score=float(idx), confidence=conf, probabilities=probs,
                                 latency_ms=self.latency_ms, model=self.name)
        return DeciderAnswer("choice", choice=pick, confidence=conf, probabilities=probs,
                             latency_ms=self.latency_ms, model=self.name)
