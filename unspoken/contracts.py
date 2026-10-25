"""Cross-cutting contracts (INTEGRATOR-OWNED). Every workstream codes against these; do not edit them in a
workstream. If one must change, say what, why and which field in your final report.

Model roles
-----------
Every model the box runs has a ROLE. `config/models.toml` maps each role to a provider, per PROFILE
("dev" = this cloud desktop, "gb10" = the Dell box, "fake" = tests). Code never names a model, a URL or
a vendor: it asks the registry for a role, so switching a model is a config edit, not a code change.

    from unspoken.models import load_registry
    reg = load_registry()                   # profile from UNSPOKEN_PROFILE (default "dev")
    decider = reg.decider()                 # unspoken.core.decider.Decider
    model = reg.strands_model("llm")        # strands.models.Model (vLLM/llama.cpp via OpenAI API; Bedrock only in dev)
    asr = reg.asr()                         # ASR below

A role whose provider is "none" is disabled: `reg.enabled(role)` is False and its getter raises
RoleDisabled. In-process providers (faster-whisper, Silero VAD, Light-ASD, ...) are named in the TOML as
`provider = "python:package.module:Factory"` and built lazily, so the registry never imports torch or
onnxruntime unless that role is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from .core.decider import Decider

# --------------------------------------------------------------------------- roles
ROLES: tuple[str, ...] = (
    "decider",      # fast typed decisions (Strands Decider 12B on the GB10, 2B on the dev CPU)
    "decider_big",  # optional second decider (e.g. 12B when "decider" is the 2B)
    "llm",          # tool-calling chat model for escalations, recaps, doc answers (Qwen3.6 on vLLM)
    "vlm",          # image understanding (whiteboard, room scene); may point at the same server as llm
    "asr",          # speech -> text
    "vad",          # voice activity detection (speech segments)
    "diarization",  # which voice: session-stable voice labels
    "faces",        # face detection / tracking in the room camera
    "asd",          # active speaker detection: which face is talking
    "tts",          # text -> speech (short spoken replies, only when asked)
    "embed",        # text embeddings for local doc search
    "pii",          # personal-detail spans in text (before anything leaves the box)
)
Role = str


class RoleDisabled(RuntimeError):
    """The role's provider is "none" in the active profile."""


class RemoteNotAllowed(RuntimeError):
    """A local_only profile (gb10) refused a provider whose target is not on this machine or its LAN."""


@dataclass
class RoleSpec:
    role: Role
    provider: str                     # "strands-decider" | "openai" | "openai-audio" | "bedrock" | "fake" | "none" | "python:mod:Factory" | ...
    label: str = ""                   # human name for the UI / X-ray, e.g. "Strands Decider 12B (GPU)"
    url: str | None = None            # HTTP base URL for server-backed providers
    model: str | None = None          # model id the server expects (OpenAI "model" field), if any
    options: dict[str, Any] = field(default_factory=dict)   # provider-specific keys from the TOML


@dataclass
class RoleHealth:
    role: Role
    provider: str
    label: str
    target: str                       # URL, model path or "in-process"
    ok: bool
    latency_ms: float | None = None   # one tiny probe request, when the provider supports it
    detail: str = ""                  # error text or a short note ("disabled", "12B, cuda")


class Registry(Protocol):
    profile: str
    local_only: bool

    def spec(self, role: Role) -> RoleSpec: ...
    def enabled(self, role: Role) -> bool: ...
    def label(self, role: Role) -> str: ...

    def decider(self, role: Role = "decider") -> Decider: ...
    def strands_model(self, role: Role = "llm") -> Any: ...       # strands.models.model.Model
    def asr(self) -> "ASR": ...
    def vad(self) -> "VAD": ...
    def diarizer(self) -> "Diarizer": ...
    def faces(self) -> "FaceTracker": ...
    def asd(self) -> "ActiveSpeaker": ...
    def tts(self) -> "TTS": ...
    def embedder(self) -> "Embedder": ...
    def pii(self) -> "PIIDetector": ...

    async def check(self, roles: list[Role] | None = None) -> list[RoleHealth]: ...
    async def aclose(self) -> None: ...


# --------------------------------------------------------------------------- speech
@dataclass
class Word:
    text: str
    start: float                      # seconds from the segment start
    end: float


@dataclass
class Transcript:
    text: str
    lang: str = "en"                  # "en" | "zh" | ...
    conf: float = 1.0
    words: list[Word] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class SpeechSegment:
    pcm16: bytes                      # mono 16-bit little-endian PCM
    sample_rate: int                  # 16000 everywhere in this repo
    start: float                      # seconds since the capture started (wall-clock aligned to video)
    end: float


@runtime_checkable
class ASR(Protocol):
    async def transcribe(self, pcm16: bytes, sample_rate: int, lang: str | None = None) -> Transcript: ...
    async def healthy(self) -> bool: ...


class VADStream(Protocol):
    def feed(self, frame: bytes) -> list[SpeechSegment]:
        """Feed raw PCM16 (any length); returns the speech segments that closed in this call."""
    def flush(self) -> list[SpeechSegment]: ...


@runtime_checkable
class VAD(Protocol):
    def stream(self, sample_rate: int = 16000) -> VADStream: ...


@dataclass
class VoiceLabel:
    label: str                        # session-stable: "voice-A", "voice-B", ... (never a real name)
    conf: float


class DiarSession(Protocol):
    async def label(self, pcm16: bytes, sample_rate: int) -> VoiceLabel: ...
    def reset(self) -> None:
        """Forget every voice (end of meeting). Voices are never stored across meetings."""


@runtime_checkable
class Diarizer(Protocol):
    def session(self) -> DiarSession: ...


@runtime_checkable
class TTS(Protocol):
    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """WAV bytes (mono, 24 kHz or the provider's native rate; the header says which)."""


# --------------------------------------------------------------------------- vision
@dataclass
class FaceTrack:
    track_id: str                     # stable while the face stays in view
    bbox: tuple[float, float, float, float]   # x, y, w, h in 0..1 of the frame
    seat: str | None = None           # seat id from config/sensors.toml when the face sits in a seat zone
    speaking: float = 0.0             # P(this face is talking now), from ASD; 0 when ASD is off


class FaceSession(Protocol):
    def update(self, frame: Any, t: float) -> list[FaceTrack]:
        """frame: HxWx3 uint8 BGR numpy array; t: seconds on the same clock as SpeechSegment.start."""


@runtime_checkable
class FaceTracker(Protocol):
    def session(self) -> FaceSession: ...


class ASDSession(Protocol):
    def update(self, frame: Any, t: float, faces: list[FaceTrack], audio: bytes | None,
               sample_rate: int = 16000) -> list[FaceTrack]:
        """Returns the same tracks with `speaking` filled in. `audio` is the PCM16 since the last call."""


@runtime_checkable
class ActiveSpeaker(Protocol):
    def session(self) -> ASDSession: ...


# --------------------------------------------------------------------------- text
@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class PIISpan:
    start: int
    end: int
    label: str                        # "person", "email", "phone", "salary", "health", "secret", ...
    score: float = 1.0


@runtime_checkable
class PIIDetector(Protocol):
    async def find(self, text: str) -> list[PIISpan]: ...


# --------------------------------------------------------------------------- outputs
@dataclass
class MessageRef:
    channel: str                      # logical channel key from config/bridge.toml ("notes", "tasks", ...)
    id: str                           # transport message id ("" when the transport gives none)
    thread: str | None = None


Transport = Literal["openclaw", "discord-rest", "console"]


@runtime_checkable
class Notifier(Protocol):
    """Everything that leaves the box as text goes through a Notifier, so the egress meter can count it.
    Implementations live in unspoken/bridge (OpenClaw hooks -> Discord; Discord REST fallback; console)."""

    transport: Transport

    async def post(self, channel: str, text: str, *, thread: str | None = None,
                   kind: str = "note", meta: dict[str, Any] | None = None) -> MessageRef: ...
    async def dm(self, user: str, text: str, *, meta: dict[str, Any] | None = None) -> MessageRef: ...
    async def healthy(self) -> bool: ...


DeviceKind = Literal["light", "plug", "screen", "speaker"]


@runtime_checkable
class Device(Protocol):
    """A room appliance. Implementations live in unspoken/devices (WiZ, Kasa, fakes)."""

    device_id: str                    # from config/room.toml, e.g. "lamp", "fan"
    kind: DeviceKind

    async def apply(self, command: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """light: "on" | "off" | "scene" {"scene": "talk"|"slides"|"break"|"privacy"|"paused"|"off_record"|"bright"|"dim"|"off"} |
        "set" {"brightness": 0-100, "color": "#rrggbb"}; plug: "on" | "off". Returns the new state."""
    async def state(self) -> dict[str, Any]: ...
    async def healthy(self) -> bool: ...
