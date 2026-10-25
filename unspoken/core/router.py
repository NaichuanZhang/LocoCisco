"""Decider tool router: the Strands Decider picks the tool; the LLM is called only when it is unsure.

Pattern: #tool-selection. A `DeciderToolRouter` is a Strands `InterventionHandler` that sits in front
of an agent's model call (`before_invocation`). It turns the newest user message into ONE typed
request for the Strands Decider (a `choice` question whose options are the agent's tools plus
"No tool", and one `choice` per enum-typed tool argument), reads the answer, and then either:

  * accepted (confident enough)      -> runs the picked tool itself (optional) and returns
                                        `Deny("Decider 2B picked lights_off(room=kitchen) at 91%;
                                        model call skipped")`: Strands ends the turn WITHOUT calling
                                        the model (0 LLM tokens, nothing leaves the machine), or
  * accepted + "No tool"             -> `Deny(...)` as well: the message needed nothing, or
  * not accepted / decider failure   -> `Proceed()`: the LLM decides as it normally would.

Minimal wiring (no other code changes to the agent):

    router = DeciderToolRouter(decider, threshold=0.8)          # decider: async ask(state, questions)
    agent = Agent(model=BedrockModel(...), tools=[...], interventions=[router])
    agent("Kitchen lights off.")
    router.last     # RouteDecision(route="local", tool="lights_off", args={"room": "kitchen"}, confidence=0.91, ...)

Decider contract: any object with `async ask(state: str, questions: dict[str, dict]) -> dict[str, answer]`
in the strands-decider wire format. Questions are `{"type": "choice", "instructions": str,
"criteria": {option: description}}`; each answer is either a wire dict (`{"type", "choice",
"confidence", "probabilities", "noul"}`) or an object with those attributes.

Pluggable: `build_query(text) -> (state, questions)`, `interpret(answers) -> (tool | None, args,
confidence, probabilities[, detail])` and `accept(decision) -> bool` replace the defaults, so a tuned
question set (like the smart home's) can keep its own wording while using this wiring; an optional 5th
item from `interpret` is kept on `RouteDecision.detail`. Limits: the Decider can only
PICK, not write, so a tool whose required arguments are not all enum-typed is never auto-called (the
router proceeds to the LLM when it is picked); tools the router executes itself must be plain callables
(e.g. `@tool` functions without a `tool_context` parameter).
"""
from __future__ import annotations

import inspect
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from strands.interventions import Deny, InterventionHandler, Proceed

NO_TOOL = "No tool"
NO_TOOL_DESCRIPTION = "The message needs none of these tools, or is not a request."
TOOL_QUESTION = "tool"
Query = tuple[str, dict[str, dict[str, Any]]]
Interpreted = tuple[str | None, dict[str, Any], float, dict[str, float]]


class DeciderLike(Protocol):
    async def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]: ...


@dataclass
class RouteDecision:
    """One routing decision; `router.last` and the argument of `on_decision`."""
    route: str                                   # "local" (model skipped) | "llm" (model decides)
    text: str                                    # the user message the decider read
    tool: str | None = None                      # picked tool; None = "No tool"
    args: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0                      # probability of the picked call
    probabilities: dict[str, float] = field(default_factory=dict)  # per option of the tool question
    latency_ms: float = 0.0                      # the decider round trip
    request: dict[str, Any] = field(default_factory=dict)          # {"state": str, "questions": {...}}
    answers: dict[str, Any] = field(default_factory=dict)          # raw answers, JSON-able
    accepted: bool = False
    error: str | None = None                     # decider or tool failure -> the model decides
    result: Any = None                           # the tool's return value when the router ran it
    detail: Any = None                           # whatever a custom `interpret` wants to keep


def raw_answers(answers: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalise decider answers (wire dicts or DeciderAnswer-like objects) to JSON-able dicts."""
    out: dict[str, dict[str, Any]] = {}
    for key, a in answers.items():
        out[key] = {name: _field(a, name) for name in ("type", "choice", "confidence", "probabilities", "noul")}
    return out


def _field(answer: Any, name: str) -> Any:
    value = answer.get(name) if isinstance(answer, dict) else getattr(answer, name, None)
    return dict(value) if isinstance(value, dict) else value


class DeciderToolRouter(InterventionHandler):
    """Let the Strands Decider pick the tool and skip the LLM when it is sure (see module docstring)."""

    name = "decider-tool-router"

    def __init__(self, decider: DeciderLike, *, threshold: float = 0.8,
                 build_query: Callable[[str], Query] | None = None,
                 interpret: Callable[[dict[str, Any]], Interpreted] | None = None,
                 accept: Callable[[RouteDecision], bool] | None = None,
                 execute: bool = True,
                 on_decision: Callable[[RouteDecision], Awaitable[None] | None] | None = None,
                 decider_name: str = "Decider 2B") -> None:
        self.decider = decider
        self.threshold = threshold
        self.build_query = build_query
        self.interpret = interpret
        self.accept = accept
        self.execute = execute
        self.on_decision = on_decision
        self.decider_name = decider_name
        self.last: RouteDecision | None = None
        self.log: deque[RouteDecision] = deque(maxlen=200)
        self._specs: dict[str, dict[str, Any]] = {}

    @property
    def on_error(self) -> str:
        return "proceed"   # a broken router must never block the agent: fail open to the LLM

    # ------------------------------------------------------------------ the seam
    async def before_invocation(self, event: Any, **kwargs: Any) -> Proceed | Deny:
        text = newest_user_text(event.messages) or newest_user_text(event.agent.messages)
        if not text:
            return Proceed(reason="no user text to route")
        self._specs = {s["name"]: s for s in event.agent.tool_registry.get_all_tool_specs()}
        t0 = time.perf_counter()
        decision = RouteDecision(route="llm", text=text)
        try:
            state, questions = self.build_query(text) if self.build_query else self.default_query(text)
            decision.request = {"state": state, "questions": questions}
            answers = await self.decider.ask(state, questions)
            decision.latency_ms = (time.perf_counter() - t0) * 1000
            decision.answers = raw_answers(answers)
            out = self.interpret(answers) if self.interpret else self.default_interpret(answers)
            tool, args, conf, probs = out[:4]
            if len(out) > 4:
                decision.detail = out[4]      # a custom interpret may keep its own object here
            decision.tool, decision.args = tool, dict(args or {})
            decision.confidence, decision.probabilities = float(conf), dict(probs or {})
        except Exception as exc:
            decision.latency_ms = (time.perf_counter() - t0) * 1000
            decision.error = f"{type(exc).__name__}: {exc}"
            return await self._finish(decision, Proceed(reason=decision.error))
        decision.accepted = bool(self.accept(decision) if self.accept else decision.confidence >= self.threshold)
        if not decision.accepted:
            return await self._finish(decision, Proceed(reason="decider unsure; the model decides"))
        if decision.tool is not None and not self.can_call(decision.tool, decision.args):
            return await self._finish(decision, Proceed(reason=f"{decision.tool} needs arguments the decider cannot write"))
        decision.route = "local"
        pct = f"{100 * decision.confidence:.0f}%"
        if decision.tool is None:
            reason = f"{self.decider_name} picked no tool at {pct}; model call skipped"
        else:
            if self.execute:
                try:
                    decision.result = await self.run_tool(event.agent, decision.tool, decision.args)
                except Exception as exc:
                    decision.route, decision.error = "llm", f"{type(exc).__name__}: {exc}"
                    return await self._finish(decision, Proceed(reason=decision.error))
            shown = ", ".join(f"{k}={v}" for k, v in decision.args.items())
            reason = f"{self.decider_name} picked {decision.tool}({shown}) at {pct}; model call skipped"
        return await self._finish(decision, Deny(reason=reason))

    async def _finish(self, decision: RouteDecision, action: Proceed | Deny) -> Proceed | Deny:
        self.last = decision
        self.log.append(decision)
        if self.on_decision is not None:
            out = self.on_decision(decision)
            if inspect.isawaitable(out):
                await out
        return action

    # ------------------------------------------------------------------ defaults (any Strands agent)
    def default_query(self, text: str) -> Query:
        """One choice over the agent's tools (+ "No tool") and one choice per enum-typed argument."""
        criteria = {name: _first_line(spec.get("description", "")) or name for name, spec in self._specs.items()}
        criteria[NO_TOOL] = NO_TOOL_DESCRIPTION
        questions: dict[str, dict[str, Any]] = {TOOL_QUESTION: {
            "type": "choice",
            "instructions": "A user said this to an assistant that has the tools listed as options. Which ONE tool "
                            "should the assistant call to handle it? Pick 'No tool' if the message needs none of them "
                            "or is not a request.",
            "criteria": criteria}}
        for name, spec in self._specs.items():
            for param, prop in _properties(spec).items():
                if "enum" in prop:
                    about = f" ({prop['description']})" if prop.get("description") else ""
                    questions[f"{name}.{param}"] = {
                        "type": "choice",
                        "instructions": f"If the assistant calls {name}, what should its '{param}' argument be{about}?",
                        "criteria": {str(v): f"{param} = {v}" for v in prop["enum"]}}
        return text, questions

    def default_interpret(self, answers: dict[str, Any]) -> Interpreted:
        """Tool = the chosen option; args = the chosen enum values; confidence = the joint probability."""
        pick = answers[TOOL_QUESTION]
        probs = {str(k): float(v) for k, v in (_field(pick, "probabilities") or {}).items()}
        choice = _field(pick, "choice")
        conf = probs.get(choice, float(_field(pick, "confidence") or 0.0))
        if choice == NO_TOOL or choice not in self._specs:
            return None, {}, conf, probs
        args: dict[str, Any] = {}
        for param, prop in _properties(self._specs[choice]).items():
            a = answers.get(f"{choice}.{param}")
            if a is not None and "enum" in prop:
                value = _field(a, "choice")
                args[param] = next((v for v in prop["enum"] if str(v) == str(value)), value)
                conf *= float((_field(a, "probabilities") or {}).get(str(value), _field(a, "confidence") or 0.0))
        return choice, args, conf, probs

    def can_call(self, tool: str, args: dict[str, Any]) -> bool:
        """True when every required argument of the tool is present (the decider cannot write free text)."""
        spec = self._specs.get(tool)
        if spec is None:
            return False
        required = _properties(spec, key="required") or []
        return all(p in args for p in required)

    async def run_tool(self, agent: Any, tool: str, args: dict[str, Any]) -> Any:
        """Call the tool's Python function directly with the picked arguments (plain callables only)."""
        fn = agent.tool_registry.registry.get(tool)
        if fn is None or not callable(fn):
            raise TypeError(f"tool {tool!r} is not a callable the router can run")
        out = fn(**args)
        return await out if inspect.isawaitable(out) else out


# ---------------------------------------------------------------------- helpers
def newest_user_text(messages: Any) -> str:
    """Text of the last user message in a Strands message list ('' if none)."""
    for message in reversed(list(messages or [])):
        if message.get("role") == "user":
            texts = [b["text"] for b in message.get("content", []) if isinstance(b, dict) and "text" in b]
            if texts:
                return "\n".join(texts).strip()
    return ""


def _properties(spec: dict[str, Any], key: str = "properties") -> Any:
    schema = (spec.get("inputSchema") or {}).get("json") or {}
    return schema.get(key) or ({} if key == "properties" else None)


def _first_line(text: str) -> str:
    return (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""
