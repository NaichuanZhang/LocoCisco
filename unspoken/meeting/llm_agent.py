"""The meeting room as ONE Strands agent: the local LLM (registry.strands_model("llm"), Qwen on vLLM on the
GB10, llama.cpp on the dev box) with one tool per contracts.ACTIONS, and the DeciderToolRouter in front
so the Strands Decider picks first and the LLM is only called when it is unsure.

    out = await room.decide(decider, query, utterance, accept=..., on_decision=...)

Tool bodies only RECORD what the LLM asked for; the policy and actuator agents decide and act. The
conversation is fresh on every call. Nothing here ever points at a remote model: the model object comes
from the registry, and the gb10 profile refuses non-local providers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional

from strands import Agent, tool
from strands.models.model import Model

from ..core.router import DeciderToolRouter, RouteDecision
from .contracts import ACTIONS, TokenUsage, ToolCall, UtteranceData
from .questions import DeciderQuery, Interpretation

# Legacy policy scores, NOT probabilities measured or calibrated by the LLM. The cascade emits
# confidence_source=adapter_constant and confidence_calibrated=False for these tool proposals.
ASK_CONFIDENCE, DIRECT_CONFIDENCE = 0.6, 0.9
MAX_TOKENS = 300
LightScene = Literal["talk", "slides", "break", "privacy", "off_record", "bright", "dim"]
Card = Literal["captions", "agenda", "timer", "tasks", "answer", "catch_up"]

SYSTEM_PROMPT = (
    "You are LocoCisco, the assistant of a meeting room. There is no wake word: you overhear ONE sentence said in "
    "the meeting and decide what, if anything, the room should do right now by calling at most two tools.\n"
    "Meeting: {title}. People checked in: {names}. Owner: {owner}.\n"
    "Call create_task only for a real action item (who does what, by when); log_decision when the group settles "
    "something; answer_question for a question the room should look up; set_light for the lamp, warmer when someone is "
    "cold and cooler when someone is hot, for comfort "
    "requests; pause_notes when the talk turns to pay, HR, health, legal matters or layoffs; share_docs / "
    "post_recap only when clearly asked. Jokes, chatter, status reports and off-topic talk need no_action. "
    "Negated, cancelled, hypothetical or tentative work is not a commitment: use no_action, not create_task. "
    "'Don't forget to' is a positive reminder; polite 'could you' requests can be real tasks. "
    "Never invent an owner: a checked-in name must be explicitly said in the utterance, or be the known speaker "
    "making a self-commitment. An unknown voice saying 'I' does not identify anyone in the roster. "
    "Otherwise leave owner empty. Dates: pass the words said (\"Friday\"), "
    "the room converts them.\n"
    "After the tool calls reply with ONE short sentence for the room screen, or nothing when nothing was done."
)


@dataclass
class RoomResult:
    route: str                                   # "local-decider" (model skipped) | "local-llm"
    decision: RouteDecision | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reply: str | None = None
    tokens: TokenUsage | None = None             # local LLM usage; None = no model call
    latency_ms: float = 0.0                      # the LLM's share
    total_ms: float = 0.0
    error: str | None = None


def make_meeting_tools() -> list:
    """One tool per contracts.ACTIONS (plus no_action). Bodies only acknowledge."""

    @tool
    def create_task(text: str, owner: Optional[str] = None, due: Optional[str] = None) -> str:
        """Record an action item: what has to be done, who owns it (a checked-in name) and by when (as said)."""
        return f"Noted task: {text} (owner {owner or 'none'}, due {due or 'none'}). Policy will post it."

    @tool
    def log_decision(text: str) -> str:
        """Record a decision the group just made."""
        return f"Noted decision: {text}."

    @tool
    def answer_question(question: str) -> str:
        """Look up a question in the local company docs and the meeting so far; the answer goes on the screen."""
        return f"Will look up: {question}."

    @tool
    def park_topic(text: str) -> str:
        """Put an idea or tangent in the parking lot for later."""
        return f"Parked: {text}."

    @tool
    def set_light(scene: LightScene) -> str:
        """Change the room lamp: bright / dim, or a mode scene (talk, slides, break)."""
        return f"Lamp scene {scene} requested."

    # Comfort is said the way people feel it, not as a device state: small models mapped "freezing" to
    # set_fan(on=true). Both tools become the set_fan action (warmer = fan off, cooler = fan on).
    @tool
    def warmer() -> str:
        """Someone is cold or it is freezing: make the room warmer (turns the fan / AC off)."""
        return "Warmer requested: fan off."

    @tool
    def cooler() -> str:
        """Someone is hot, it is stuffy or boiling: make the room cooler (turns the fan / AC on)."""
        return "Cooler requested: fan on."

    @tool
    def show(card: Card) -> str:
        """Put a card on the room screen: agenda, timer, tasks, captions, answer, catch_up."""
        return f"Showing {card}."

    @tool
    def pause_notes(reason: str) -> str:
        """Stop taking notes because the talk is private (pay, HR, health, legal, layoffs)."""
        return f"Notes paused: {reason}."

    @tool
    def resume_notes() -> str:
        """Start taking notes again."""
        return "Notes resumed."

    @tool
    def post_recap() -> str:
        """Post the meeting recap to the chat channel (the owner is asked first)."""
        return "Recap requested."

    @tool
    def share_docs(query: str) -> str:
        """Share matching company docs in the chat channel (denied while a guest is in the room)."""
        return f"Docs search requested: {query}."

    @tool
    def no_action() -> str:
        """Nothing to do: chatter, a joke, a status report, off-topic talk, or a question for a person."""
        return "Nothing to do."

    return [create_task, log_decision, answer_question, park_topic, set_light, warmer, cooler, show, pause_notes,
            resume_notes, post_recap, share_docs, no_action]


class RoomAgent:
    """Builds a fresh Strands Agent per call; the model and tools are shared."""

    def __init__(self, model: Model | Callable[[], Model], *, threshold: float = 0.8, label: str = "local LLM") -> None:
        self._model = model
        self.threshold = threshold
        self.label = label
        self.tools = make_meeting_tools()
        self.calls = 0

    def model(self) -> Model:
        if callable(self._model) and not isinstance(self._model, Model):
            self._model = self._model()
        return self._model  # type: ignore[return-value]

    async def decide(self, decider: Any, query: DeciderQuery, utt: UtteranceData, *, title: str, owner: str | None,
                     accept: Callable[[RouteDecision], bool] | None = None,
                     on_decision: Callable[[RouteDecision], Awaitable[None] | None] | None = None) -> RoomResult:
        router = DeciderToolRouter(decider, threshold=self.threshold, execute=False,
                                   build_query=lambda text: (query.state, query.questions),
                                   interpret=lambda answers: interpret_for_router(query, answers),
                                   accept=accept, on_decision=on_decision, decider_name="the decider")
        system = SYSTEM_PROMPT.format(title=title, names=", ".join(query.names) or "nobody yet", owner=owner or "unknown")
        t0 = time.perf_counter()
        self.calls += 1
        try:
            agent = Agent(model=self.model(), tools=self.tools, interventions=[router], system_prompt=system,
                          callback_handler=None)
            result = await agent.invoke_async(prompt_for(query, utt))
        except Exception as exc:
            total = (time.perf_counter() - t0) * 1000
            dec = router.last
            return RoomResult("local-llm", dec, error=f"{type(exc).__name__}: {exc}", total_ms=total,
                              latency_ms=max(0.0, total - (dec.latency_ms if dec else 0.0)))
        total = (time.perf_counter() - t0) * 1000
        dec = router.last
        if dec is not None and dec.route == "local":
            return RoomResult("local-decider", dec, total_ms=total)
        usage = result.metrics.accumulated_usage
        tokens: TokenUsage | None = None
        if usage and (usage.get("inputTokens") or usage.get("outputTokens")):
            tokens = {"input": int(usage.get("inputTokens", 0)), "output": int(usage.get("outputTokens", 0))}
        reply = str(result).strip() or None
        return RoomResult("local-llm", dec, tool_calls=tool_calls_of(agent.messages), reply=reply, tokens=tokens,
                          total_ms=total, latency_ms=max(0.0, total - (dec.latency_ms if dec else 0.0)))


# ---------------------------------------------------------------------- adapters
def interpret_for_router(query: DeciderQuery, answers: dict[str, Any]
                         ) -> tuple[str | None, dict[str, Any], float, dict[str, float], Interpretation]:
    it = query.interpret(answers)
    tool_name = None if it.action == "none" else it.action
    return tool_name, dict(it.args), it.confidence, dict(it.probabilities), it


def interpretation_of(decision: RouteDecision) -> Interpretation | None:
    if isinstance(decision.detail, Interpretation):
        return decision.detail
    return None


def prompt_for(query: DeciderQuery, utt: UtteranceData) -> str:
    sp = utt.get("speaker") or {}
    who = sp.get("name") or ("a guest with no check-in" if sp.get("guest") else "an unknown voice")
    src = {"media": "the room screen (a video or remote call), not a person here"}.get(str(utt.get("source")), "a person in the room")
    return f"Overheard. {utt.get('context', '')} Voice: {who}, from {src}. Heard: \"{utt.get('text', '')}\""


def tool_calls_of(messages: list[dict[str, Any]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for block in message.get("content", []):
            use = block.get("toolUse") if isinstance(block, dict) else None
            if use:
                calls.append({"name": str(use.get("name")), "input": dict(use.get("input") or {})})
    return calls


def as_bool(value: Any, *, default: bool) -> bool:
    """A tool argument as a bool. Small local models often send "false" / "off" / 0 as strings."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in ("false", "off", "no", "0", ""):
        return False
    if text in ("true", "on", "yes", "1"):
        return True
    return default


def proposal_from_tool_call(call: ToolCall, query: DeciderQuery) -> tuple[str, dict[str, Any], float]:
    """(action, args, adapter score) for one LLM tool call; unknown tools count as 'none'.

    The cascade applies the common commitment safety check before publishing an actionable proposal.
    Task text and dates come exclusively from the utterance, never a generated summary/deadline.
    The original tool arguments remain available in the cascade's raw tool_calls for diagnostics.
    """
    name, args = call["name"], dict(call.get("input") or {})
    if name in ("warmer", "cooler"):
        return "set_fan", {"on": name == "cooler"}, DIRECT_CONFIDENCE
    if name == "no_action" or name not in ACTIONS:
        return "none", {}, DIRECT_CONFIDENCE
    if name == "create_task":
        owner = query.grounded_owner(args.get("owner"))
        from .dates import parse_due
        due = parse_due(query.text, query.today)
        return name, {"owner": owner, "text": query.text, "due": due}, DIRECT_CONFIDENCE
    if name == "pause_notes":
        return name, {"reason": str(args.get("reason") or "the LLM heard something private"),
                      "wipe_seconds": query.wipe_seconds}, DIRECT_CONFIDENCE
    if name == "set_fan":
        return name, {"on": as_bool(args.get("on"), default=True)}, DIRECT_CONFIDENCE
    if name == "set_light":
        return name, {"scene": str(args.get("scene") or "talk")}, DIRECT_CONFIDENCE
    return name, args, DIRECT_CONFIDENCE
