"""The house as ONE Strands agent: Claude Haiku 4.5 on Amazon Bedrock with the house's tools, and the
DeciderToolRouter in its path so the Strands Decider 2B picks the tool first.

Every decision goes THROUGH this agent (there is no separate fast path in plain code):

    out = await house.decide(decider_2b, query, event_data, accept=..., on_decision=...)

1. `before_invocation`: the router asks the 2B the house's tuned v1 questions (build_speech_query /
   build_scene_query) in ONE request. Sure -> Deny: the model is never called, `out.route == "local"`,
   `out.tokens is None`, and `out.decision.detail` is the 2B's Interpretation.
2. Unsure -> Proceed: Claude reads one overheard sentence or one camera scene and calls at most two of
   the tools below. Tool bodies only RECORD what Claude asked (the policy and actuator agents still
   decide and act); `out.tool_calls` lists them in order, `out.tokens` the Bedrock usage.

The conversation is fresh on every call (a new Agent object; the Bedrock model object is cached).
"""
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional

from strands import Agent, tool
from strands.hooks import AfterToolsEvent
from strands.models.model import Model

from ..contracts import ACTIONS, Settings, TokenUsage, ToolCall
from ..decider.questions import ROOM_NAMES, DeciderQuery, Interpretation
from ..patterns.decider_tool_router import DeciderToolRouter, RouteDecision
from .base import ACTION_WORDS, HomeConfig, load_home

MODEL_LABEL = "Claude Haiku 4.5 on Bedrock"
MAX_TOOL_ROUNDS = 2        # tool batches per decision; a small local model can otherwise call tools forever
ROOM_ACTIONS = ("lights_on", "lights_off", "lights_dim", "alert_parent")   # tools that take a room
ASK_CONFIDENCE, DIRECT_CONFIDENCE = 0.6, 0.9                              # proposal confidence per tool kind
MAX_TOKENS = 200

Room = Literal[tuple(ROOM_NAMES)]  # type: ignore[valid-type]  # home.json room ids, as an enum
ActionName = Literal[tuple(a for a in ACTIONS if a != "none")]  # type: ignore[valid-type]

SYSTEM_PROMPT = (
    "You are the assistant of a smart home with no wake word. You get ONE overheard sentence or ONE camera "
    "scene and decide what the house should do right now by calling at most two tools.\n"
    "Rooms: {rooms}. Residents: {residents}.{pets} The living room has the only TV; one thermostat heats the "
    "whole house; the front door has the lock.\n"
    "Call no device tool for chatter, a story, something said on TV or radio, or a question about whether "
    "something is on or off: call no_action instead. If you think the person wants something but are not "
    "sure, call ask_resident instead of acting. Only act on what is wanted right now; the house's own policy "
    "code checks who may do what.\n"
    "After the tool calls, reply with ONLY the short sentence the hub speaker says aloud to the people in the "
    "room, in the first person (for example: Turning the kitchen lights off.), or reply with nothing at all when "
    "nothing was done. Never explain your reasoning."
)


@dataclass
class HouseResult:
    route: str                                   # "local" (2B decided, model skipped) | the agent's route ("bedrock" | "local")
    decision: RouteDecision | None               # the router's decision (None if the router never ran)
    tool_calls: list[ToolCall] = field(default_factory=list)   # Claude's Strands tool calls, in order
    reply: str | None = None                     # Claude's spoken sentence, if any
    tokens: TokenUsage | None = None             # Bedrock usage; None = no model call
    latency_ms: float = 0.0                      # Bedrock's share (agent total minus the decider's)
    total_ms: float = 0.0
    error: str | None = None                     # Bedrock / agent failure
    by_router: bool = False                      # True = the decider decided inside the router; the model never ran


def make_house_tools() -> list:
    """The tools Claude sees, mirroring contracts.ACTIONS. Bodies only acknowledge; nothing is actuated."""

    def ack(action: str, room: Optional[str] = None) -> str:
        where = f" in the {ROOM_NAMES.get(room, room)}" if room else ""
        return f"Noted: {ACTION_WORDS.get(action, action)}{where}. The house policy will decide and act."

    @tool
    def lights_on(room: Room) -> str:
        """Turn the lights on in a room (someone wants light or says it is too dark)."""
        return ack("lights_on", room)

    @tool
    def lights_off(room: Room) -> str:
        """Turn the lights off in a room (going to sleep, left on by mistake, nobody there)."""
        return ack("lights_off", room)

    @tool
    def lights_dim(room: Room) -> str:
        """Dim the lights in a room (too bright, movie, winding down)."""
        return ack("lights_dim", room)

    @tool
    def warmer() -> str:
        """Raise the thermostat: someone is cold or asks for heat."""
        return ack("warmer")

    @tool
    def cooler() -> str:
        """Lower the thermostat: someone is hot or asks for it cooler."""
        return ack("cooler")

    @tool
    def tv_on() -> str:
        """Turn the living room TV on: someone wants to watch now."""
        return ack("tv_on")

    @tool
    def tv_off() -> str:
        """Turn the TV off: nobody is watching, someone is asleep, or they ask for quiet."""
        return ack("tv_off")

    @tool
    def unlock_door() -> str:
        """Unlock the front door for someone."""
        return ack("unlock_door")

    @tool
    def lock_door() -> str:
        """Lock the front door."""
        return ack("lock_door")

    @tool
    def alert_parent(room: Room) -> str:
        """Send an urgent alert to a parent's phone: someone may be in danger in this room."""
        return ack("alert_parent", room)

    @tool
    def ask_resident(action: ActionName, room: Optional[Room] = None) -> str:
        """You think the person wants this action but should check first; the house asks them on the phone."""
        return f"Noted: will ask before we {ACTION_WORDS.get(action, action)}."

    @tool
    def no_action() -> str:
        """The house should do nothing: chatter, a story, TV or radio talk, a question about the house, or nothing needed."""
        return "Noted: nothing to do."

    return [lights_on, lights_off, lights_dim, warmer, cooler, tv_on, tv_off, unlock_door, lock_door,
            alert_parent, ask_resident, no_action]


class HouseAgent:
    """Builds a fresh Strands Agent per call; the model and tools are shared. `name` is the deciders slot:
    "bedrock" (Claude from Settings, as always) or "llm" (whatever the models registry's llm role is: Qwen on
    vLLM / llama.cpp, or Claude in the dev profile). `label` is the model name the X-ray shows, `route` where
    the decision is made ("bedrock" or "local"); `short` and `who` are the words the cascade's reasons use
    ("Claude called lights_on", "unsure, asking Claude on Bedrock")."""

    def __init__(self, settings: Settings, home: HomeConfig | None = None, *, model: Model | None = None,
                 model_factory: Callable[[Settings], Model] | None = None, name: str = "bedrock",
                 label: str = MODEL_LABEL, route: str = "bedrock", short: str | None = None,
                 who: str | None = None, max_tool_rounds: int = MAX_TOOL_ROUNDS) -> None:
        self.settings = settings
        self.name = name
        self.max_tool_rounds = max_tool_rounds
        self.label = label
        self.route = route
        words = label.split(" on ")[0].split() if label else []
        self.short = short or (words[0] if words else name)                             # "Claude" / "Qwen3"
        self.who = who or ("Claude on Bedrock" if name == "bedrock" else label)
        self.service = "Bedrock" if name == "bedrock" else self.who                     # "<service> failed, asked the 12B"
        self.home = home or HomeConfig(load_home())
        self.tools = make_house_tools()
        self._model = model
        self._model_factory = model_factory or bedrock_model
        pets = ", ".join(f"{p['name']} ({p.get('species', 'pet')})" for p in self.home.pets.values())
        self.system_prompt = SYSTEM_PROMPT.format(
            rooms=", ".join(f"{rid} = {self.home.room_name(rid)}" for rid in self.home.rooms),
            residents=", ".join(f"{r['name']} ({r.get('role', 'unknown')})" for r in self.home.residents.values()),
            pets=f" Pets: {pets}; a room with only a pet in it needs nothing." if pets else "")
        self.calls = 0

    # ------------------------------------------------------------------ model
    def model(self) -> Model:
        if self._model is None:
            self._model = self._model_factory(self.settings)
        return self._model

    async def healthy(self) -> bool:
        try:
            self.model()
            return True
        except Exception:
            return False

    async def aclose(self) -> None:
        return None

    # ------------------------------------------------------------------ deciding
    async def decide(self, decider: Any, query: DeciderQuery, data: dict[str, Any], *,
                     accept: Callable[[RouteDecision], bool] | None = None,
                     on_decision: Callable[[RouteDecision], Awaitable[None] | None] | None = None,
                     router: bool = True) -> HouseResult:
        """Run one decision through the agent. `router=False` sends every case to Claude (the baseline)."""
        room = data.get("room")
        interventions = []
        if router:
            interventions.append(DeciderToolRouter(
                decider, threshold=self.settings.act_threshold, execute=False,
                build_query=lambda text: (query.state, query.questions),
                interpret=lambda answers: interpret_for_router(query, answers, room),
                accept=accept, on_decision=on_decision))
        t0 = time.perf_counter()
        self.calls += 1
        try:
            agent = Agent(model=self.model(), tools=self.tools, interventions=interventions,
                          system_prompt=self.system_prompt, callback_handler=None)
            rounds = [0]

            def after_tools(event: AfterToolsEvent) -> None:
                """After the N-th tool batch stop before the next model call (strands' cooperative cancel): the
                tool calls so far are the decision; a looping model costs one more call, not fourteen."""
                rounds[0] += 1
                if rounds[0] >= self.max_tool_rounds:
                    agent.cancel(after_current_tools=True)

            agent.add_hook(after_tools, AfterToolsEvent)
            result = await agent.invoke_async(prompt_for(query, data))
        except Exception as exc:
            decision = interventions[0].last if interventions else None
            total = (time.perf_counter() - t0) * 1000
            return HouseResult(self.route, decision, error=f"{type(exc).__name__}: {exc}", total_ms=total,
                               latency_ms=max(0.0, total - (decision.latency_ms if decision else 0.0)))
        total = (time.perf_counter() - t0) * 1000
        decision = interventions[0].last if interventions else None
        if decision is not None and decision.route == "local":
            return HouseResult("local", decision, latency_ms=0.0, total_ms=total, by_router=True)
        usage = result.metrics.accumulated_usage
        tokens: TokenUsage | None = None
        if usage and (usage.get("inputTokens") or usage.get("outputTokens")):
            tokens = {"input": int(usage.get("inputTokens", 0)), "output": int(usage.get("outputTokens", 0))}
        reply = str(result).strip() or None
        return HouseResult(self.route, decision, tool_calls=tool_calls_of(agent.messages), reply=reply,
                           tokens=tokens, total_ms=total,
                           latency_ms=max(0.0, total - (decision.latency_ms if decision else 0.0)))


# ---------------------------------------------------------------------- adapters
def interpret_for_router(query: DeciderQuery, answers: dict[str, Any], room: str | None
                         ) -> tuple[str | None, dict[str, Any], float, dict[str, float], Interpretation]:
    """DeciderQuery.interpret -> (tool, args, confidence, probabilities, detail=the Interpretation)."""
    it: Interpretation = query.interpret(answers)
    tool_name = None if it.action == "none" else it.action
    args: dict[str, Any] = {}
    if tool_name in ROOM_ACTIONS:
        args["room"] = it.room or room
    return tool_name, args, it.confidence, dict(it.probabilities), it


def interpretation_of(decision: RouteDecision) -> Interpretation | None:
    """The 2B's Interpretation behind a router decision (None when the decider failed)."""
    if isinstance(decision.detail, Interpretation):
        return decision.detail
    if decision.error is not None or not decision.probabilities:
        return None
    action = decision.tool or "none"
    return Interpretation(action, decision.confidence, dict(decision.probabilities), {}, decision.args.get("room"))


def prompt_for(query: DeciderQuery, data: dict[str, Any]) -> str:
    """What Claude reads: one overheard sentence with its room context, or one camera scene."""
    if query.kind == "speech":
        sp = data.get("speaker") or {}
        who = f"{sp['name']} ({sp.get('role', 'unknown')})" if sp.get("name") else "an unknown voice"
        source = {"tv": "the TV, not a person", "outside": "outside the front door", "person": "a person in the room"}
        src = source.get(data.get("source", "person"), "a person in the room")
        context = (data.get("context") or "").strip()
        return f"Overheard. {context} Voice: {who}, from {src}. Heard: \"{data.get('text', '')}\""
    return f"Camera scene. {query.state}"


def tool_calls_of(messages: list[dict[str, Any]]) -> list[ToolCall]:
    """Claude's toolUse blocks, in order."""
    calls: list[ToolCall] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for block in message.get("content", []):
            use = block.get("toolUse") if isinstance(block, dict) else None
            if use:
                calls.append({"name": str(use.get("name")), "input": dict(use.get("input") or {})})
    return calls


def proposal_from_tool_call(call: ToolCall) -> tuple[str, float, str | None]:
    """(action, confidence, room) for one of Claude's tool calls; unknown tools count as 'none'."""
    name, args = call["name"], call.get("input") or {}
    if name == "ask_resident":
        action = str(args.get("action") or "none")
        return (action if action in ACTIONS else "none"), ASK_CONFIDENCE, args.get("room")
    if name == "no_action" or name not in ACTIONS:
        return "none", DIRECT_CONFIDENCE, None
    return name, DIRECT_CONFIDENCE, args.get("room")


def bedrock_model(settings: Settings) -> Model:
    """The cached Claude model: boto3 session from settings.bedrock_profile (None = default chain)."""
    import boto3
    from strands.models import BedrockModel

    session = boto3.Session(profile_name=settings.bedrock_profile, region_name=settings.bedrock_region)
    return BedrockModel(model_id=settings.bedrock_model_id, boto_session=session, max_tokens=MAX_TOKENS,
                        temperature=0.0)
