"""Simulated room: people with seats and voices. World steps (contracts.SCENARIO_STEPS) become the SAME
events live sensors publish: audio.segment (transcribed, voice-labelled) and video.speakers (face tracks
with seats and a speaking probability), plus video.scene, chat.command, meeting.status and meeting.clock.
Every agent runs unchanged in sim and live mode; nothing here is audio or video, only their outputs.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ..core.events import BusAPI, Event
from .contracts import MT

FRAME_HZ = 5.0
WORDS_PER_SECOND = 2.6
MEDIA_VOICE = "voice-screen"
WORLD_STEPS = frozenset({"start", "join", "say", "media_say", "enter", "leave", "scene", "chat", "react", "clock", "end"})


@dataclass
class SimPerson:
    who: str
    seat: str | None = None
    voice: str | None = None
    checkin: str | None = None
    present: bool = True
    track_id: str = ""


@dataclass
class World:
    bus: BusAPI
    absent_after: float = 3.0
    people: dict[str, SimPerson] = field(default_factory=dict)
    clock: float = 0.0
    scene_mode: str = "talk"
    started: bool = False
    media_voice_n: int = 0
    frames_per_say: int = 4
    _n: int = 0

    # ------------------------------------------------------------------ lifecycle
    async def reset(self) -> None:
        self.people.clear()
        self.clock = 0.0
        self.scene_mode = "talk"
        self.started = False
        self.media_voice_n = 0

    async def start(self) -> None:
        return None

    async def tick(self, dt: float) -> None:
        """Background frames: faces present, nobody talking (what a quiet room looks like to the camera)."""
        self.clock += dt
        await self._frame(speaking=None, screen_talking=False)

    def snapshot(self) -> dict[str, Any]:
        return {"clock": self.clock, "scene": self.scene_mode,
                "people": [{"who": p.who, "seat": p.seat, "voice": p.voice, "checkin": p.checkin, "present": p.present}
                           for p in self.people.values()]}

    # ------------------------------------------------------------------ steps
    async def step(self, step: dict[str, Any]) -> None:
        do = step.get("do")
        if do == "start":
            await self._start(step)
        elif do == "join":
            await self._join(step)
        elif do == "say":
            await self.say(str(step["who"]), str(step["text"]), str(step.get("lang") or "en"))
        elif do == "media_say":
            await self.media_say(str(step["text"]), str(step.get("lang") or "en"))
        elif do == "enter":
            await self._enter(step)
        elif do == "leave":
            await self._leave(str(step["who"]))
        elif do == "scene":
            self.scene_mode = str(step.get("mode") or "talk")
            await self._publish(MT.SCENE, {"t": self.clock, "mode": self.scene_mode, "conf": 0.95, "by": "rule",
                                           "text": f"camera: {self.scene_mode}"}, "sim.camera")
        elif do in ("chat", "react"):
            await self._publish(MT.CHAT, {"command": step.get("command"), "user": step.get("user") or "someone",
                                          "args": dict(step.get("args") or {}), "text": step.get("text") or ""}, "sim.chat")
        elif do == "clock":
            self.clock += 0.5
            await self._publish(MT.CLOCK, {"t": self.clock, "minutes_left": step.get("minutes_left"),
                                           "ends_at": step.get("ends_at")}, "sim.clock")
        elif do == "end":
            self.started = False
            await self._publish(MT.STATUS, {"state": "ended", "title": step.get("title") or ""}, "sim.meeting")
            self.people.clear()
        else:
            raise ValueError(f"not a world step: {do!r}")

    async def _start(self, step: dict[str, Any]) -> None:
        self.started = True
        agenda = list(step.get("agenda") or [])
        await self._publish(MT.STATUS, {"state": "started", "title": step.get("title") or "Meeting",
                                        "topic": step.get("topic") or (agenda[0] if agenda else None),
                                        "owner": step.get("owner"), "ends_at": step.get("ends_at"), "agenda": agenda,
                                        "minutes": step.get("minutes")}, "sim.meeting")

    async def _join(self, step: dict[str, Any]) -> None:
        who = str(step["who"])
        p = self.people.get(who) or SimPerson(who)
        p.seat, p.voice, p.checkin, p.present = step.get("seat"), step.get("voice"), step.get("checkin"), True
        p.track_id = p.track_id or f"t{len(self.people) + 1}"
        self.people[who] = p
        self.clock += 0.4
        await self._frame(speaking=None, screen_talking=False)
        if p.checkin:
            await self.say(who, f"Hi, I'm {p.checkin}.", "en")

    async def _enter(self, step: dict[str, Any]) -> None:
        who = str(step["who"])
        p = self.people.get(who) or SimPerson(who)
        p.seat, p.present = step.get("seat"), True
        p.voice = step.get("voice") or p.voice
        p.track_id = p.track_id or f"t{len(self.people) + 1}"
        self.people[who] = p
        for _ in range(2):
            self.clock += 0.2
            await self._frame(speaking=None, screen_talking=False)

    async def _leave(self, who: str) -> None:
        p = self.people.get(who)
        if p is None:
            return
        p.present = False
        self.clock += 0.2
        await self._frame(speaking=None, screen_talking=False)
        self.clock += self.absent_after + 0.2         # the camera has not seen the face for a while
        await self._frame(speaking=None, screen_talking=False)

    # ------------------------------------------------------------------ speech
    async def say(self, who: str, text: str, lang: str = "en") -> None:
        p = self.people.get(who)
        if p is None:
            p = SimPerson(who, voice=f"voice-{who[:1].upper()}", track_id=f"t{len(self.people) + 1}")
            self.people[who] = p
        if not p.voice:
            p.voice = f"voice-{who[:1].upper()}"
        start = self.clock
        dur = max(0.8, len(text.split()) / WORDS_PER_SECOND if lang == "en" else len(text) / 4.0)
        for i in range(self.frames_per_say):
            self.clock = start + dur * (i + 0.5) / self.frames_per_say
            await self._frame(speaking=p.seat, screen_talking=False)
        self.clock = start + dur
        await self._segment(text, lang, p.voice, start, self.clock, voice_conf=0.9)
        self.clock += 0.3

    async def media_say(self, text: str, lang: str = "en") -> None:
        start = self.clock
        dur = max(0.8, len(text.split()) / WORDS_PER_SECOND)
        for i in range(self.frames_per_say):
            self.clock = start + dur * (i + 0.5) / self.frames_per_say
            await self._frame(speaking=None, screen_talking=True)
        self.clock = start + dur
        await self._segment(text, lang, MEDIA_VOICE, start, self.clock, voice_conf=0.6)
        self.clock += 0.3

    async def _segment(self, text: str, lang: str, voice: str, start: float, end: float, voice_conf: float) -> None:
        self._n += 1
        await self._publish(MT.SEGMENT, {"segment_id": f"seg-{self._n}", "start": start, "end": end, "text": text,
                                         "lang": lang, "asr_conf": 0.92, "voice": voice, "voice_conf": voice_conf,
                                         "asr_ms": 0.0, "diar_ms": 0.0}, "sim.audio", ts=end)

    async def _frame(self, *, speaking: str | None, screen_talking: bool) -> None:
        faces = []
        for p in self.people.values():
            if not p.present:
                continue
            faces.append({"track_id": p.track_id, "seat": p.seat, "bbox": [0.1, 0.2, 0.15, 0.3],
                          "speaking": 0.92 if (speaking is not None and p.seat == speaking) else 0.04})
        await self._publish(MT.SPEAKERS, {"t": self.clock, "faces": faces, "face_count": len(faces),
                                          "screen_talking": screen_talking}, "sim.camera")

    async def _publish(self, topic: str, data: dict[str, Any], source: str, ts: float | None = None) -> None:
        await self.bus.publish(Event(topic, data, source=source, ts=self.clock if ts is None else ts))
        await asyncio.sleep(0)
