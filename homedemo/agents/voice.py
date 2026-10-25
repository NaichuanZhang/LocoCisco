"""Voice: sensor.audio -> voice.utterance. Who spoke (voice ID passthrough), from where (mic
direction: person, TV or outside) and one line of room context the agent keeps itself from
device/lux/temp events, e.g. "Living room. Lights off, evening. TV on."."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts import Event, T
from .base import Agent, ClockState, HomeConfig, fmt_num

VOICE_SOURCES = ("person", "tv", "outside", "unknown")


@dataclass
class _RoomCtx:
    lights: dict[str, bool] = field(default_factory=dict)
    tv_on: bool | None = None
    celsius: float | None = None
    lux: float | None = None


class VoiceAgent(Agent):
    name = "voice"

    def __init__(self, bus, home: HomeConfig, settings) -> None:
        super().__init__(bus, home, settings)
        self.clock = ClockState(home)
        self.rooms: dict[str, _RoomCtx] = {}
        for device in home.devices.values():
            ctx = self._ctx(device.get("room"))
            if device.get("kind") == "light":
                ctx.lights[device["device_id"]] = bool((device.get("state") or {}).get("on"))
            elif device.get("kind") == "tv":
                ctx.tv_on = bool((device.get("state") or {}).get("on"))

    def start(self) -> None:
        self.subscribe(T.AUDIO, self._on_audio)
        self.subscribe(T.CLOCK, self._on_clock)
        self.subscribe(T.DEVICE, self._on_device)
        self.subscribe(T.LUX, self._on_lux)
        self.subscribe(T.TEMP, self._on_temp)

    def _ctx(self, room_id: str | None) -> _RoomCtx:
        return self.rooms.setdefault(room_id or "unknown", _RoomCtx())

    async def _on_clock(self, ev: Event) -> None:
        self.clock.update(ev.data)

    async def _on_device(self, ev: Event) -> None:
        d = ev.data
        spec = self.home.devices.get(d.get("device_id", ""), {})
        kind = d.get("kind") or spec.get("kind")
        ctx = self._ctx(d.get("room") or spec.get("room"))
        state = d.get("state") or {}
        if kind == "light":
            ctx.lights[d.get("device_id", "light")] = bool(state.get("on"))
        elif kind == "tv":
            ctx.tv_on = bool(state.get("on"))

    async def _on_lux(self, ev: Event) -> None:
        self._ctx(ev.data.get("room")).lux = float(ev.data.get("lux", 0))

    async def _on_temp(self, ev: Event) -> None:
        self._ctx(ev.data.get("room")).celsius = float(ev.data.get("celsius", 0))

    async def _on_audio(self, ev: Event) -> None:
        d = ev.data
        text = str(d.get("text") or "").strip()
        if not text:
            return
        room = d.get("room") or ""
        speaker: dict[str, Any] = {"id": None, "name": None, "role": "unknown", "match": 0.0}
        speaker.update({k: v for k, v in (d.get("speaker") or {}).items() if v is not None})
        if speaker.get("id") and not d.get("speaker", {}).get("role"):
            speaker["role"] = self.home.resident_role(speaker["id"])
        if speaker.get("id") and not speaker.get("name"):
            speaker["name"] = self.home.resident_name(speaker["id"])
        source = (d.get("doa") or {}).get("source") or "unknown"
        if source not in VOICE_SOURCES:
            source = "unknown"
        await self.emit(T.UTTERANCE, {
            "room": room, "text": text, "lang": d.get("lang") or "en", "speaker": speaker,
            "source": source, "context": self.context(room), "wake_word": bool(d.get("wake_word")),
        }, ev)

    def context(self, room_id: str) -> str:
        """One line in the measured style: '<Room>. Lights off, evening. TV on. Room 17 C.'"""
        ctx = self._ctx(room_id)
        parts = [f"{self.home.room_name(room_id)}."]
        if ctx.lights:
            parts.append(f"Lights {'on' if any(ctx.lights.values()) else 'off'}, {self.clock.daypart()}.")
        else:
            parts.append(f"{self.clock.daypart().capitalize()}.")
        if ctx.tv_on is not None:
            parts.append(f"TV {'on' if ctx.tv_on else 'off'}.")
        if ctx.celsius is not None:
            parts.append(f"Room {fmt_num(ctx.celsius)} C.")
        return " ".join(parts)
