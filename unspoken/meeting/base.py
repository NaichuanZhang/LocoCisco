"""Shared pieces for the meeting agents: the Agent base class and MeetingState, the session-only memory
the agents write and the runtime reads for snapshot()/transcript()/items().

Nothing in MeetingState survives a meeting: reset() empties it, and `end` wipes names, voices and seats.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from ..core.events import BusAPI, Event, Handler
from .contracts import (MT, EgressData, ItemData, MeetingSettings, PrivacyData, ScreenData, Speaker,
                        StatusData, UtteranceData)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@dataclass
class Person:
    """One participant as fusion knows them. All of it is forgotten at meeting end."""
    key: str                          # internal key ("voice-A", "seat:s2", "chat:Sam")
    name: str | None = None
    voice: str | None = None
    seat: str | None = None
    guest: bool = True                # True until a check-in names them
    present: bool = True
    last_seen: float = 0.0
    joined_at: float = 0.0

    def as_speaker(self, conf: float = 1.0) -> Speaker:
        return {"name": self.name, "voice": self.voice, "seat": self.seat, "conf": round(conf, 3),
                "guest": self.guest}


@dataclass
class MeetingState:
    status: StatusData = field(default_factory=lambda: {"state": "ended", "title": "", "topic": None,
                                                        "owner": None, "ends_at": None})
    agenda: list[str] = field(default_factory=list)
    privacy: PrivacyData = field(default_factory=lambda: {"state": "on", "reason": "", "by": "user",
                                                          "wiped_from": None, "wiped_count": 0})
    people: dict[str, Person] = field(default_factory=dict)
    transcript: list[UtteranceData] = field(default_factory=list)   # on-record only
    items: dict[str, ItemData] = field(default_factory=dict)
    screen: ScreenData = field(default_factory=lambda: {"privacy": "on", "captions": [], "agenda": {},
                                                        "cards": [], "banner": None})
    devices: dict[str, dict[str, Any]] = field(default_factory=dict)  # device_id -> EnvData
    egress: EgressData = field(default_factory=lambda: {"dest": "", "kind": "text", "bytes": 0, "total_bytes": 0,
                                                        "audio_bytes_out": 0, "video_bytes_out": 0})
    scene_mode: str = "talk"
    clock: float = 0.0                # latest Event.ts seen
    started_at: float | None = None
    minutes_left: float | None = None
    total_minutes: float | None = None  # planned length (start step / config)
    sensitive_pause: bool = False     # the current pause came from a sensitive hint (resume asks the owner)

    # ---- roster views
    def participants(self) -> list[Speaker]:
        return [p.as_speaker() for p in self.people.values() if p.present]

    def names(self) -> list[str]:
        """Checked-in names, in join order (owner first when present)."""
        names = [p.name for p in self.people.values() if p.present and p.name]
        owner = self.status.get("owner")
        if owner and owner not in names:
            names.insert(0, owner)
        elif owner in names:
            names.remove(owner)
            names.insert(0, owner)
        return list(dict.fromkeys(names))

    def guests_present(self) -> int:
        return sum(1 for p in self.people.values() if p.present and p.guest)

    def wipe_people(self) -> None:
        self.people.clear()

    def open_items(self, kind: str | None = None) -> list[ItemData]:
        return [i for i in self.items.values() if i.get("status") in ("open", "pending_approval")
                and (kind is None or i.get("kind") == kind)]

    def context_line(self) -> str:
        """One line the decider and the LLM read: topic, minutes left, room mode, who is here."""
        parts = []
        title = self.status.get("title") or "Meeting"
        topic = self.status.get("topic")
        parts.append(f"{title}" + (f", topic: {topic}" if topic else "") + ".")
        if self.minutes_left is not None:
            parts.append(f"{max(0, round(self.minutes_left))} min left.")
        names = self.names()
        owner = self.status.get("owner")
        if names:
            shown = [f"{n} (owner)" if n == owner else n for n in names]
            parts.append("Here: " + ", ".join(shown) + ".")
        g = self.guests_present()
        if g:
            parts.append(f"{g} guest{'s' if g > 1 else ''} without a check-in.")
        if self.scene_mode and self.scene_mode != "talk":
            parts.append(f"Room mode: {self.scene_mode}.")
        if self.privacy.get("state") != "on":
            parts.append(f"Notes {self.privacy.get('state')}.")
        return " ".join(parts)


class Agent:
    """One pipeline agent. Subclasses set `name` and implement start()."""

    name: str = ""

    def __init__(self, bus: BusAPI, state: MeetingState, settings: MeetingSettings, config: dict[str, Any]) -> None:
        self.bus = bus
        self.state = state
        self.settings = settings
        self.config = config
        self.busy = False

    @property
    def source(self) -> str:
        return f"agent.{self.name}"

    def start(self) -> None:
        raise NotImplementedError

    def subscribe(self, pattern: str, handler: Handler) -> None:
        self.bus.subscribe(pattern, handler, name=self.name)

    async def emit(self, topic: str, data: dict[str, Any], cause: Event | None, ts: float | None = None) -> Event:
        event = Event(topic, data, source=self.source, ts=ts if ts is not None else (cause.ts if cause else self.state.clock),
                      cause=cause.id if cause else None)
        await self.bus.publish(event)
        return event

    async def log(self, text: str, cause: Event | None, level: str = "info") -> Event:
        return await self.emit(MT.LOG, {"agent": self.name, "level": level, "text": text}, cause)


def local_llm_enabled(registry: Any) -> bool:
    """The meeting room only escalates to a model ON THIS BOX: the llm role must be enabled, not Bedrock or any
    other hosted provider, and its URL (if any) must be loopback / private / *.local. This holds in every profile,
    so `--profile dev` (whose llm is Bedrock for the old home demo) runs the room decider-only."""
    try:
        if not registry.enabled("llm"):
            return False
        spec = registry.spec("llm")
    except Exception:
        return False
    if spec.provider in ("bedrock", "anthropic", "openai-hosted"):
        return False
    if spec.url:
        try:
            from unspoken.models.netpolicy import url_is_local
        except Exception:
            return False
        return bool(url_is_local(spec.url))
    return True
