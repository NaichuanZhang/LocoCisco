"""Shared contracts for the smart-home decider demo.

OWNED BY THE INTEGRATOR. Workstream agents must not edit this file; if a contract
needs to change, say so in your final report (what, why, which field).

Every component talks ONLY through the Bus (homedemo/bus.py) with the topics and
payload shapes below. Payloads are plain JSON-able dicts so the web UI can stream
them unchanged. Field names are snake_case. Times are simulated clock seconds
(`Event.ts`) plus human strings inside payloads where a model needs them.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Literal, Protocol, TypedDict


# --------------------------------------------------------------------------- topics
class T:
    # Raw mocked sensors. Published ONLY by homedemo/sim (SensorHub).
    CAMERA = "sensor.camera"      # CameraData
    MOTION = "sensor.motion"      # MotionData
    LUX = "sensor.lux"            # LuxData
    TEMP = "sensor.temp"          # TempData
    DEVICE = "sensor.device"      # DeviceData (device reports its own state)
    DOOR = "sensor.door"          # DoorData
    AUDIO = "sensor.audio"        # AudioData (mic array: ASR text + voice ID + direction)
    PHONE = "sensor.phone"        # PhoneData (resident phone home/away)
    CLOCK = "sim.clock"           # ClockData, published when the scenario sets/jumps time

    # Pipeline agents (homedemo/agents).
    SCENE = "perception.scene"    # SceneData   perception agent -> deciders/reflex
    UTTERANCE = "voice.utterance" # UtteranceData voice agent -> intent agents
    PROPOSAL = "intent.proposal"  # ProposalData reflex / decider / keyword / user
    VERDICT = "policy.verdict"    # VerdictData  policy agent
    COMMAND = "action.command"    # CommandData  actuator agent -> sim
    APPLIED = "action.applied"    # AppliedData  sim, after a command changed a device
    ASK = "notify.ask"            # AskData      actuator -> phone UI / scenario runner
    REPLY = "notify.reply"        # ReplyData    phone UI / scenario runner -> actuator

    # Bookkeeping.
    LOG = "trace.log"             # LogData, free-text explanation any agent may emit
    SCENARIO = "scenario.status"  # ScenarioStatusData, scenario runner progress + verdicts


SENSOR_TOPICS = (T.CAMERA, T.MOTION, T.LUX, T.TEMP, T.DEVICE, T.DOOR, T.AUDIO, T.PHONE)

# Canonical pipeline agents. Event.source for an agent is f"agent.{name}"; the sim publishes as
# "sim.<sensor kind>" (e.g. "sim.camera") and the phone UI as "phone". The web UI draws this graph.
AGENTS: list[dict] = [
    {"name": "perception", "label": "Perception", "in": ["sensor.*", "sim.clock"], "out": [T.SCENE],
     "blurb": "Fuses camera, presence, light and temperature into a short scene per room."},
    {"name": "voice", "label": "Voice", "in": [T.AUDIO, T.DEVICE, T.LUX, T.TEMP, T.CLOCK], "out": [T.UTTERANCE],
     "blurb": "Who is speaking (voice ID) and from where (mic direction: person, TV or outside)."},
    {"name": "reflex", "label": "Reflex rules", "in": [T.SCENE], "out": [T.PROPOSAL],
     "blurb": "Instant plain rules: dark room + person at night -> lights on; child at a lit stove -> alert a parent."},
    {"name": "scene_decider", "label": "Scene decider", "in": [T.SCENE], "out": [T.PROPOSAL],
     "blurb": "Asks the decider what the house should do about what it sees."},
    {"name": "intent_decider", "label": "Intent decider", "in": [T.UTTERANCE], "out": [T.PROPOSAL],
     "blurb": "Asks the Decider 2B which tool the speaker wants, with no wake word. Claude on Bedrock only if unsure."},
    {"name": "keyword", "label": "Keyword baseline", "in": [T.UTTERANCE], "out": [T.PROPOSAL],
     "blurb": "Old-style keyword matching, shown side by side (never executed)."},
    {"name": "policy", "label": "Policy", "in": [T.PROPOSAL], "out": [T.VERDICT],
     "blurb": "Plain code: house rules, who may do what, and the confidence bands."},
    {"name": "actuator", "label": "Actuator", "in": [T.VERDICT, T.REPLY], "out": [T.COMMAND, T.ASK],
     "blurb": "Turns allowed verdicts into device commands, or asks on a phone first."},
]


# --------------------------------------------------------------------------- envelope
# Event, Handler and BusAPI live in unspoken.core.events (shared by every LocoCisco domain).
from unspoken.core.events import BusAPI, Event, Handler, _new_id  # noqa: E402,F401


# --------------------------------------------------------------------------- vocab
Role = Literal["parent", "child", "guest", "unknown"]
VoiceSource = Literal["person", "tv", "outside", "unknown"]

# Everything the house can be asked to do. "none" = deliberately do nothing.
Action = Literal[
    "lights_on", "lights_off", "lights_dim", "warmer", "cooler",
    "tv_on", "tv_off", "unlock_door", "lock_door", "alert_parent", "none",
]
ACTIONS: tuple[str, ...] = (
    "lights_on", "lights_off", "lights_dim", "warmer", "cooler",
    "tv_on", "tv_off", "unlock_door", "lock_door", "alert_parent", "none",
)


class Speaker(TypedDict, total=False):
    id: str | None        # resident id from home.json ("alex"), None if no voice match
    name: str | None
    role: Role
    match: float          # voice-verification score 0..1 (>=0.85 counts as a match)


# --------------------------------------------------------------------------- sensor payloads
class Detection(TypedDict, total=False):
    label: str            # "person", "cat", "dog", "child" is NOT a label: use person + attrs
    conf: float           # detector score 0..1
    track_id: str         # stable per tracked object
    attrs: dict[str, Any] # {"age_group": "child"|"adult", "pose": "standing on chair", "near": "stove", "idle_min": 40,
                          #  "identity": {"id": "maya", "match": 0.9}, "wearing": ["coat"], "holding": ["keys"]}


class CameraData(TypedDict, total=False):
    room: str
    camera_id: str
    detections: list[Detection]
    activity: str | None  # short caption of what people are doing, e.g. "lifting couch cushions"


class MotionData(TypedDict):
    room: str
    motion: bool


class LuxData(TypedDict):
    room: str
    lux: float


class TempData(TypedDict):
    room: str
    celsius: float


class DeviceData(TypedDict, total=False):
    device_id: str        # from home.json
    kind: str             # "light" | "tv" | "thermostat" | "stove" | "lock" | "speaker"
    room: str
    state: dict[str, Any] # light {"on": bool, "brightness": 0-100}; tv {"on": bool, "volume": int, "playing": str|None}
                          # thermostat {"setpoint_c": float}; stove {"burner_on": bool}; lock {"locked": bool}


class DoorData(TypedDict):
    door_id: str
    open: bool
    locked: bool


class AudioData(TypedDict, total=False):
    room: str
    mic_id: str
    text: str             # mocked ASR output
    lang: str             # "en" | "zh"
    asr_conf: float
    wake_word: bool       # always False in the no-wake-word demo unless a scenario sets it
    speaker: Speaker      # mocked voice ID
    doa: dict[str, Any]   # {"azimuth_deg": float, "source": VoiceSource, "conf": float}


class PhoneData(TypedDict):
    person_id: str
    home: bool


class ClockData(TypedDict):
    time: str             # "23:40"
    weekday: str          # "Tuesday"
    school_night: bool    # Sunday..Thursday from 18:00 until sunrise


# --------------------------------------------------------------------------- agent payloads
class SceneData(TypedDict, total=False):
    room: str
    trigger: str          # "person_entered" | "person_left" | "activity_changed" | "idle" | "hazard" | "env_changed"
                          # (person_left: someone only walked out; no agent decides on it)
    text: str             # one short paragraph a decider can read (see tests/fixtures in decider/questions.py)
    facts: dict[str, Any] # {"time","weekday","people":[{"id","role","activity"}],"pets":[...],
                          #  "lux","lights_on","tv_on","celsius","setpoint_c","stove_on","idle_min"}


class UtteranceData(TypedDict, total=False):
    room: str
    text: str
    lang: str
    speaker: Speaker
    source: VoiceSource   # from mic direction: a person in the room, the TV, or outside
    context: str          # one line of room context, e.g. "Living room. Lights off, evening. TV on."
    wake_word: bool


ProposalBy = Literal["reflex", "decider-2b", "decider-12b", "bedrock", "fake", "keyword", "user", "llm"]
ProposalMode = Literal["act", "shadow"]  # shadow = shown in the UI for comparison, never executed
# Where the decision was made: on the home hub by the Decider 2B, by Claude on Amazon Bedrock, by the
# 12B fallback, or by plain code (reflex rules, keyword baseline, a user's phone reply).
ProposalRoute = Literal["local", "bedrock", "12b", "rule"]


class TokenUsage(TypedDict):
    input: int
    output: int


class ToolCall(TypedDict):
    name: str                 # Strands tool name, e.g. "lights_on"
    input: dict[str, Any]     # the arguments Claude passed


class ProposalData(TypedDict, total=False):
    by: ProposalBy
    mode: ProposalMode
    room: str
    action: str           # one of ACTIONS
    target: str | None    # device_id or door_id when known
    confidence: float     # 0..1 (decider confidence; 1.0 for rules)
    probabilities: dict[str, float]  # per option, for the UI bars
    question: str         # the decider question asked (empty for rules)
    state_text: str       # the exact state the decider read
    latency_ms: float
    reason: str           # one human sentence
    speaker: Speaker | None
    kind: Literal["speech", "scene"]
    escalated_from: str | None  # proposal id of the 2B answer when Bedrock (or the 12B) was asked next
    # --- added for the Bedrock path and the UI's Decider X-ray (all optional; old producers omit them)
    route: ProposalRoute        # where this decision was made
    model: str                  # human model name, e.g. "Strands Decider 2B" or "Claude Haiku 4.5 on Bedrock"
    request: dict[str, Any]     # the exact typed request sent to the decider: {"state": str, "questions": {...}}
    answers: dict[str, Any]     # raw per-question answers: {key: {"type","choice","confidence","probabilities","noul"}}
    threshold: float            # the act threshold the confidence was compared with (settings.act_threshold)
    tokens: TokenUsage | None   # Bedrock token usage for this decision; None (or absent) = no LLM call
    tool_calls: list[ToolCall]  # Bedrock route: the Strands tool calls Claude made, in order
    reply: str | None           # Bedrock route: the one-sentence spoken reply Claude wrote, if any


Verdict = Literal["allow", "ask_speaker", "ask_parent", "deny", "ignore"]


class VerdictData(TypedDict, total=False):
    proposal_id: str
    action: str
    target: str | None
    room: str
    verdict: Verdict
    rule: str             # which rule decided, human readable
    who: Speaker | None


class CommandData(TypedDict, total=False):
    device_id: str
    command: str          # "on" | "off" | "dim" | "set_setpoint" | "unlock" | "lock" | "notify"
    args: dict[str, Any]  # {"brightness": 30} | {"delta_c": 1.5} | {"to": "parents", "text": "..."}
    verdict_id: str


class AppliedData(TypedDict, total=False):
    # For "notify" commands: device_id is the command's device_id (e.g. "phone_parents"),
    # state = {"to", "text", "delivered": True}.
    device_id: str
    state: dict[str, Any]
    command_id: str


class AskData(TypedDict, total=False):
    request_id: str
    to: str               # "parents" | resident id
    text: str             # "Turn on the living room lights?"
    options: list[str]    # ["Yes", "No"]
    proposal_id: str
    action: str
    target: str | None
    room: str


class ReplyData(TypedDict):
    request_id: str
    answer: str           # one of AskData.options
    by: str               # resident id or "web"


class LogData(TypedDict, total=False):
    agent: str
    level: Literal["info", "warn"]
    text: str


class ScenarioStatusData(TypedDict, total=False):
    scenario_id: str
    state: Literal["started", "step", "expect_pass", "expect_fail", "finished", "stopped", "error"]
    step_index: int
    text: str
    passed: int
    failed: int


# --------------------------------------------------------------------------- settings
@dataclass
class Settings:
    decider_mode: Literal["2b", "12b", "cascade", "fake"] = "cascade"  # cascade = 2B, then escalate_to if unsure
    escalate_to: Literal["bedrock", "12b", "llm"] = "bedrock"  # who answers when the 2B is unsure ("llm" = the
                                                              # registry's llm role: local Qwen on vLLM/llama.cpp)
    profile: str = "dev"          # config/models.toml profile the deciders and the "llm" slot come from
    bedrock_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    bedrock_region: str = "us-west-2"
    bedrock_profile: str | None = "bedrock"  # AWS profile for boto3; None = default credential chain
    act_threshold: float = 0.80   # >= this: act
    ask_threshold: float = 0.50   # >= this and < act: ask ("Did you want ...?"); below: ignore (eval/REPORT.md s3)
    keyword_shadow: bool = True   # run the keyword baseline next to the decider, for the UI
    speed: float = 1.0            # scenario playback speed (2.0 = twice as fast)
    # Decider URLs: the profile's config/models.toml block is the source of truth; a value here that differs
    # from the default below overrides it (the --decider-2b-url flags), the default means "use the profile".
    decider_2b_url: str = "http://127.0.0.1:8765"
    decider_12b_url: str = "http://127.0.0.1:8766"


# --------------------------------------------------------------------------- component APIs
class WorldAPI(Protocol):
    """Implemented by homedemo/sim. Ground truth of the simulated home."""

    clock: float  # simulated seconds since scenario start

    def snapshot(self) -> dict[str, Any]:
        """JSON for the UI: {"time","weekday","rooms":[{"id","name","x","y","w","h","env":{"lux","celsius"}}],
        "devices":[{"device_id","kind","room","x","y","state"}],"people":[{"id","name","role","room","x","y",
        "activity","home"}],"pets":[...],"doors":[...]}"""

    async def step(self, step: dict[str, Any]) -> None:
        """Apply one WORLD step of the scenario DSL (see SCENARIO_STEPS) and let SensorHub publish
        the sensor events it causes. Unknown "do" values raise ValueError."""

    async def start(self) -> None:
        """Subscribe to action.command and publish the boot report (every device, room sensor, phone)."""

    async def tick(self, seconds: float = 1.0) -> None:
        """Slow physics: clock, temperature drift toward the setpoint, daylight. The runtime calls it."""

    async def reset(self) -> None:
        """Back to home.json initial state."""


class RuntimeAPI(Protocol):
    """Implemented by homedemo/runtime.py (integrator). The web UI codes against this."""

    bus: BusAPI
    settings: Settings

    def snapshot(self) -> dict[str, Any]: ...
    def agent_graph(self) -> list[dict[str, Any]]:
        """contracts.AGENTS plus live status: [{"name","label","in","out","blurb","busy":bool}]"""
    def scenarios(self) -> list[dict[str, Any]]:
        """[{"id","title","blurb","tags":[...],"steps":int}]"""
    async def run_scenario(self, scenario_id: str) -> None:
        """Start in the background; progress arrives on T.SCENARIO. Stops any running one first."""
    async def stop_scenario(self) -> None: ...
    async def reset(self) -> None: ...
    async def step(self, step: dict[str, Any]) -> None:
        """Manual injection from the UI, same DSL as scenario steps (e.g. {"do":"say",...})."""
    async def phone_reply(self, request_id: str, answer: str) -> None: ...
    async def update_settings(self, **changes: Any) -> Settings: ...


# --------------------------------------------------------------------------- scenario DSL
# A scenario file is JSON: {"id","title","blurb","tags":[],"steps":[step,...]}.
# Each step is {"do": <name>, ...args} and may carry "wait": seconds to sleep AFTER it (real
# seconds / Settings.speed). WORLD steps go to WorldAPI.step; RUNNER steps are handled by
# homedemo/scenarios/runner.py.
SCENARIO_STEPS: dict[str, str] = {
    # world
    "set_time": '{"do":"set_time","time":"23:40","weekday":"Tuesday"}',
    "move": '{"do":"move","who":"alex","to":"kitchen"}            # person or pet id',
    "activity": '{"do":"activity","who":"alex","text":"putting on a sweater","attrs":{"wearing":["sweater"]}}',
    "idle": '{"do":"idle","who":"alex","minutes":40}              # jumps clock; person stops moving',
    "say": '{"do":"say","who":"alex","text":"...","lang":"en"}    # person speaks where they are',
    "tv_say": '{"do":"tv_say","text":"Turn off the lights","device":"tv_living"}',
    "outside_say": '{"do":"outside_say","text":"Unlock the front door","door":"front"}',
    "set_env": '{"do":"set_env","room":"kitchen","lux":900,"celsius":17}',
    "set_device": '{"do":"set_device","device":"tv_living","state":{"on":true,"volume":30}}',
    "phone": '{"do":"phone","who":"alex","home":false}',
    "leave": '{"do":"leave","who":"alex"}                        # leaves the house',
    "arrive": '{"do":"arrive","who":"guest1","to":"living"}',
    # runner
    "wait": '{"do":"wait","seconds":2}',
    "reply": '{"do":"reply","answer":"Yes"}                     # answers the newest open phone ask',
    "expect": '{"do":"expect","what":"device","device":"light_kitchen","state":{"on":true},"within":6}',
    "note": '{"do":"note","text":"Narration shown in the UI"}',
}
# expect "what" values: "device" (state subset reached within N s), "no_command" (no command for
# device/any for N s), "ask" (a phone ask to "to" within N s), "verdict" (a verdict with fields),
# "proposal" (a proposal with by/action, e.g. to show the keyword baseline being wrong), and
# "outcome": {"action","room","accept":["done","asked"],"within"} = a command for that action/room
# OR (if "asked" is accepted) a phone ask proposing it. Decider-driven checks accept both.
