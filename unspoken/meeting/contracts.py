"""Meeting-room contracts (INTEGRATOR-OWNED; the meeting workstream may ADD fields or topics and must list
every addition in its final report; nobody renames or removes one without the integrator).

The room is a set of small agents on one Bus (unspoken.core.bus). Sensors publish raw observations; the
pipeline fuses them into named utterances, decides with the Strands Decider first and the local LLM only
when unsure, runs plain-code policy, then acts through devices, the room screen and the Notifier
(OpenClaw -> Discord). Payloads are JSON-able dicts; times are seconds on one monotonic meeting clock
(`Event.ts`), the same clock as SpeechSegment.start and the camera frame times.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict

from ..core.events import BusAPI


# --------------------------------------------------------------------------- topics
class MT:
    # Sensors (unspoken/sensors in live mode, unspoken/meeting/sim in simulated mode)
    SEGMENT = "audio.segment"         # SegmentData   one VAD speech segment, transcribed + voice-labelled
    SPEAKERS = "video.speakers"       # SpeakersData  face tracks with seats and speaking probability (~5 Hz)
    SCENE = "video.scene"             # SceneData     room mode from the camera (slides on, standing at board, empty)
    ENV = "room.env"                  # EnvData       device states (lamp, fan, screen) as devices report them
    CHAT = "chat.command"             # ChatCommandData  Discord slash command / reaction / message, via the bridge

    # Pipeline (unspoken/meeting)
    UTTERANCE = "meeting.utterance"   # UtteranceData  who said what (fusion of voice label + seat + check-in name)
    PRIVACY = "meeting.privacy"       # PrivacyData    notes on / paused / off the record; wipe windows
    PROPOSAL = "meeting.proposal"     # ProposalData   reflex / decider / llm / user proposals (Decider X-ray fields)
    VERDICT = "meeting.verdict"       # VerdictData    policy: act / ask_owner / deny / ignore, with risk level
    COMMAND = "meeting.command"       # CommandData    actuator -> device / screen / notifier
    ITEM = "meeting.item"             # ItemData       a task, decision, question, parking-lot idea (created or updated)
    SCREEN = "screen.state"           # ScreenData     what the room screen should show now
    EGRESS = "egress.meter"           # EgressData     bytes that left the box, by destination and kind
    STATUS = "meeting.status"         # StatusData     meeting started / agenda topic changed / ended
    CLOCK = "meeting.clock"           # ClockData      meeting clock tick (ADDED by the meeting workstream)

    # Bookkeeping
    LOG = "trace.log"                 # LogData (same shape as the home demo)
    SCENARIO = "scenario.status"      # ScenarioStatusData (same shape as the home demo)


SENSOR_TOPICS = (MT.SEGMENT, MT.SPEAKERS, MT.SCENE, MT.ENV, MT.CHAT)
# Topics whose payloads carry what people said. While notes are not on, the bus keeps none of them in history;
# an "off the record" wipe deletes them from history back to PrivacyData.wiped_from.
SPOKEN_TOPICS = (MT.SEGMENT, MT.UTTERANCE)
PRIVATE_TOPICS = (MT.SEGMENT, MT.UTTERANCE, MT.PROPOSAL, MT.VERDICT, MT.SCREEN)


class ClockData(TypedDict, total=False):
    """ADDED by the meeting workstream: the sim (a `clock` step) and live mode (every ~10 s) publish the
    meeting clock so the screen agent can count minutes down without owning a timer of its own."""
    t: float                  # seconds since the meeting started (Event.ts clock)
    minutes_left: float | None
    ends_at: str | None


# --------------------------------------------------------------------------- vocab
Source = Literal["person", "media", "unknown"]          # media = a video or call playing on the room screen
PrivacyState = Literal["on", "paused", "off_record"]    # paused = a human or the box stopped notes; off_record = said so
Kind = Literal["task", "decision", "question", "room", "sensitive", "small_talk", "off_topic", "none"]
Risk = Literal["low", "medium", "high"]
ItemKind = Literal["task", "decision", "question", "parking_lot"]

# Everything the room can be asked to do. Tool names for the LLM match these one to one.
ACTIONS: tuple[str, ...] = (
    "create_task",      # {owner, text, due}            medium risk: acts, a Discord ❌ undoes it
    "log_decision",     # {text}                        medium
    "answer_question",  # {question}                    low: local doc search, answer on screen + thread
    "park_topic",       # {text}                        low
    "set_light",        # {scene: talk|slides|break|privacy|off_record|bright|dim}   low
    "set_fan",          # {on: bool}                    low (stands in for the AC)
    "show",             # {card: captions|agenda|timer|tasks|answer|catch_up, ...}   low
    "pause_notes",      # {reason, wipe_seconds}        low, and always allowed (privacy first)
    "resume_notes",     # {}                            medium: asks the owner when the pause came from a sensitive hint
    "post_recap",       # {}                            high: asks the meeting owner first
    "share_docs",       # {query}                       high when a guest is in the room
    "none",
)


class Speaker(TypedDict, total=False):
    name: str | None          # check-in name ("Priya"), None until someone checks in for this voice/seat
    voice: str | None         # VoiceLabel.label ("voice-B")
    seat: str | None          # seat id from config/room.toml
    conf: float               # fusion confidence 0..1
    guest: bool               # a face or voice with no check-in


# --------------------------------------------------------------------------- sensor payloads
class SegmentData(TypedDict, total=False):
    segment_id: str
    start: float
    end: float
    text: str
    lang: str
    asr_conf: float
    voice: str | None         # VoiceLabel.label
    voice_conf: float
    asr_ms: float             # model latency, for the X-ray
    diar_ms: float


class FaceData(TypedDict, total=False):
    track_id: str
    seat: str | None
    bbox: list[float]         # x, y, w, h in 0..1
    speaking: float


class SpeakersData(TypedDict, total=False):
    t: float
    faces: list[FaceData]
    face_count: int
    screen_talking: bool      # the room screen region shows a talking face (media / remote call)


class SceneData(TypedDict, total=False):
    t: float
    mode: Literal["talk", "slides", "board", "break", "empty"]
    conf: float
    by: str                   # "rule" | "decider" | "vlm"
    text: str                 # one-line caption the decider read


class EnvData(TypedDict, total=False):
    device_id: str
    kind: str
    state: dict[str, Any]


class ChatCommandData(TypedDict, total=False):
    command: Literal["here", "offrecord", "resume", "recap", "agenda", "undo", "approve", "deny", "ask"]
    user: str                 # Discord display name
    args: dict[str, Any]      # here: {"name"}; agenda: {"topics": [...], "ends_at": "15:30"}; undo/approve/deny: {"item_id" | "request_id"}
    text: str                 # free text for "ask"


# --------------------------------------------------------------------------- pipeline payloads
class UtteranceData(TypedDict, total=False):
    utterance_id: str
    segment_id: str
    start: float
    end: float
    text: str
    lang: str
    speaker: Speaker
    source: Source
    context: str              # one line: agenda topic, minutes left, room mode, who is here


class PrivacyData(TypedDict, total=False):
    state: PrivacyState
    reason: str               # "said 'off the record'", "/offrecord by Priya", "sounded like pay talk"
    by: str                   # "reflex" | "decider" | "llm" | "user"
    wiped_from: float | None  # utterances with start >= this were deleted from memory and screen
    wiped_count: int


class TokenUsage(TypedDict):
    input: int
    output: int


class ToolCall(TypedDict):
    name: str
    input: dict[str, Any]


class ProposalData(TypedDict, total=False):
    by: str                   # "reflex" | "decider" | "decider_big" | "llm" | "user" | "fake"
    mode: Literal["act", "shadow"]
    kind: Kind
    action: str               # one of ACTIONS
    args: dict[str, Any]
    confidence: float
    probabilities: dict[str, float]
    reason: str               # one human sentence
    utterance_id: str | None
    escalated_from: str | None
    # Decider X-ray (same meaning as the home demo's ProposalData)
    route: Literal["local-decider", "local-llm", "rule", "user"]
    model: str                # Registry.label(role), never hard-coded
    request: dict[str, Any]   # {"state": str, "questions": {...}} sent to the decider
    answers: dict[str, Any]
    threshold: float
    latency_ms: float
    tokens: TokenUsage | None # local LLM tokens (they never leave the box; shown to prove the LLM was or wasn't called)
    tool_calls: list[ToolCall]
    reply: str | None


class VerdictData(TypedDict, total=False):
    proposal_id: str
    action: str
    args: dict[str, Any]
    verdict: Literal["act", "ask_owner", "deny", "ignore"]
    risk: Risk
    rule: str                 # which policy rule decided, human readable


class CommandData(TypedDict, total=False):
    target: str               # device_id ("lamp", "fan"), "screen", or "notifier"
    command: str              # device command, screen card, or "post" | "dm" | "update"
    args: dict[str, Any]
    verdict_id: str


class ItemData(TypedDict, total=False):
    item_id: str
    kind: ItemKind
    text: str
    owner: str | None
    due: str | None
    status: Literal["open", "done", "undone", "pending_approval"]
    utterance_id: str | None
    message: dict[str, Any] | None    # MessageRef as dict once posted


class ScreenData(TypedDict, total=False):
    privacy: PrivacyState
    captions: list[dict[str, Any]]    # [{"name","text","t"}], newest last, never off-record lines
    agenda: dict[str, Any]            # {"topic", "topics", "minutes_left", "ends_at"}
    cards: list[dict[str, Any]]       # [{"card": "answer"|"tasks"|"catch_up"|..., ...}]
    banner: str | None                # "2 min left. 1 task has no owner."


class EgressData(TypedDict, total=False):
    dest: str                 # "discord" | "openclaw" | ...
    kind: str                 # "text" (the only kind that may ever appear)
    bytes: int
    total_bytes: int
    audio_bytes_out: int      # always 0; shown on screen
    video_bytes_out: int      # always 0


class StatusData(TypedDict, total=False):
    state: Literal["started", "topic", "ended"]
    title: str
    topic: str | None
    owner: str | None         # meeting owner (gets approvals and the recap DM)
    ends_at: str | None


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
class MeetingSettings:
    act_threshold: float = 0.80       # decider confidence at or above: act on the decider's answer
    ask_threshold: float = 0.50       # below act and at or above this: escalate to the local LLM; below: ignore
    escalate: bool = True             # False = decider only (no LLM), for the "how far does the decider get" demo
    wipe_seconds: float = 60.0        # "off the record" wipes this much recent transcript
    caption_lines: int = 6
    speed: float = 1.0                # scenario playback speed
    mode: Literal["sim", "live", "replay"] = "sim"


# --------------------------------------------------------------------------- runtime API
class MeetingRuntimeAPI(Protocol):
    """Implemented by unspoken/meeting/runtime.py. The room screen (unspoken/meeting/web) and the MCP
    server (unspoken/bridge/mcp_server.py) code against this, never against agents directly."""

    bus: BusAPI
    settings: MeetingSettings

    def snapshot(self) -> dict[str, Any]:
        """{"status": StatusData, "privacy": PrivacyData, "participants": [Speaker], "screen": ScreenData,
        "items": [ItemData], "devices": [EnvData], "egress": EgressData, "models": [RoleHealth as dict]}"""
    def transcript(self, since: float | None = None) -> list[UtteranceData]:
        """On-record utterances only: off-record and wiped lines are gone, not hidden."""
    def items(self, kind: ItemKind | None = None) -> list[ItemData]: ...
    async def search_docs(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """[{"doc", "chunk", "score", "text"}] from the local docs folder (config/meeting.toml)."""
    async def recap(self) -> str:
        """Decisions, tasks with owners, open questions, as Discord-ready markdown (local LLM)."""
    async def chat(self, command: ChatCommandData) -> dict[str, Any]:
        """Inject a Discord command (the bridge calls this); returns {"ok": bool, "text": str}."""
    async def set_privacy(self, state: PrivacyState, reason: str, by: str = "user") -> PrivacyData: ...
    def scenarios(self) -> list[dict[str, Any]]: ...
    async def run_scenario(self, scenario_id: str) -> None: ...
    async def stop_scenario(self) -> None: ...
    async def reset(self) -> None: ...
    async def update_settings(self, **changes: Any) -> MeetingSettings: ...


# --------------------------------------------------------------------------- scenario DSL (sim mode)
# {"id","title","blurb","tags":[],"steps":[{"do": ..., ...,"wait": seconds}]}. The simulator turns world
# steps into the SAME sensor events live mode produces (audio.segment + video.speakers), so every agent
# runs unchanged in sim, replay and live mode.
SCENARIO_STEPS: dict[str, str] = {
    "start": '{"do":"start","title":"Pricing sync","owner":"Priya","agenda":["Q3 pricing","Launch"],"minutes":30}',
    "join": '{"do":"join","who":"priya","seat":"s1","voice":"voice-A","checkin":"Priya"}   # checkin null = guest',
    "say": '{"do":"say","who":"priya","text":"Sam, send the deck by Friday.","lang":"en"}',
    "media_say": '{"do":"media_say","text":"...a video playing on the screen speaks..."}',
    "enter": '{"do":"enter","who":"visitor","seat":"s4"}                              # a face with no check-in',
    "leave": '{"do":"leave","who":"visitor"}',
    "scene": '{"do":"scene","mode":"slides"}',
    "chat": '{"do":"chat","command":"here","user":"Sam","args":{"name":"Sam"}}',
    "react": '{"do":"chat","command":"undo","user":"Sam","args":{"item_id":"$last_task"}}',
    "clock": '{"do":"clock","minutes_left":2}',
    "end": '{"do":"end"}',
    "wait": '{"do":"wait","seconds":2}',
    "note": '{"do":"note","text":"Narration shown on the screen"}',
    "expect": '{"do":"expect","what":"item","kind":"task","owner":"Sam","within":8}',
}
# expect "what": "item" (kind/owner/text_contains), "privacy" (state), "device" (device_id, state subset),
# "post" (channel, text_contains), "no_post" (for N s), "proposal" (by/action/route), "screen" (banner_contains).
