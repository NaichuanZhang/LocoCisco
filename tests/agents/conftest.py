"""Shared helpers: a Pipeline = real Bus + all agents + FakeDecider(s), fed hand-made sensor events."""
from __future__ import annotations

from typing import Any

import pytest

from homedemo.agents import load_home, start_agents
from homedemo.bus import Bus
from homedemo.contracts import Event, Settings, T
from homedemo.decider_client import DeciderAnswer, FakeDecider


class BrokenDecider:
    """A decider whose server is down."""

    def __init__(self, name: str = "decider-2b") -> None:
        self.name = name
        self.calls = 0

    async def healthy(self) -> bool:
        return False

    async def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, DeciderAnswer]:
        self.calls += 1
        raise ConnectionError("connection refused")


class Pipeline:
    def __init__(self, settings: Settings, deciders: dict[str, Any], home: dict | None = None) -> None:
        self.home = home or load_home()
        self.settings = settings
        self.bus = Bus()
        self.agents = start_agents(self.bus, self.home, settings, deciders)
        self._ts = 0.0

    def agent(self, name: str):
        return next(a for a in self.agents if a.name == name)

    # ---- publishing hand-made events
    async def publish(self, topic: str, data: dict[str, Any], source: str, ts: float | None = None) -> Event:
        self._ts = self._ts + 1 if ts is None else ts
        ev = Event(topic, data, source=source, ts=self._ts)
        await self.bus.publish(ev)
        return ev

    async def settle(self) -> None:
        await self.bus.drain()

    async def clock(self, time: str, weekday: str = "Tuesday", school_night: bool | None = None, ts=None) -> Event:
        if school_night is None:
            school_night = weekday in {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"}
        return await self.publish(T.CLOCK, {"time": time, "weekday": weekday, "school_night": school_night}, "sim.clock", ts)

    async def lux(self, room: str, lux: float, ts=None) -> Event:
        return await self.publish(T.LUX, {"room": room, "lux": lux}, "sim.lux", ts)

    async def temp(self, room: str, celsius: float, ts=None) -> Event:
        return await self.publish(T.TEMP, {"room": room, "celsius": celsius}, "sim.temp", ts)

    async def motion(self, room: str, motion: bool, ts=None) -> Event:
        return await self.publish(T.MOTION, {"room": room, "motion": motion}, "sim.motion", ts)

    async def device(self, device_id: str, state: dict[str, Any], ts=None) -> Event:
        spec = next(d for d in self.home["devices"] if d["device_id"] == device_id)
        return await self.publish(T.DEVICE, {"device_id": device_id, "kind": spec["kind"], "room": spec["room"],
                                             "state": state}, "sim.device", ts)

    async def camera(self, room: str, detections: list[dict], activity: str | None = None, ts=None) -> Event:
        return await self.publish(T.CAMERA, {"room": room, "camera_id": f"cam_{room}", "detections": detections,
                                             "activity": activity}, "sim.camera", ts)

    def person(self, who: str | None = None, *, age_group: str | None = None, track: str = "t1", **attrs) -> dict:
        a: dict[str, Any] = dict(attrs)
        if who:
            res = next(r for r in self.home["residents"] if r["id"] == who)
            a["identity"] = {"id": who, "match": 0.9}
            a.setdefault("age_group", "child" if res["role"] == "child" else "adult")
        if age_group:
            a["age_group"] = age_group
        return {"label": "person", "conf": 0.95, "track_id": track, "attrs": a}

    def pet(self, species: str = "cat", track: str = "p1") -> dict:
        return {"label": species, "conf": 0.9, "track_id": track, "attrs": {}}

    async def say(self, text: str, who: str | None = "alex", room: str = "living", match: float = 0.95,
                  source: str = "person", lang: str = "en", ts=None) -> Event:
        speaker: dict[str, Any] = {"id": None, "name": None, "role": "unknown", "match": 0.0}
        if who:
            res = next(r for r in self.home["residents"] if r["id"] == who)
            speaker = {"id": who, "name": res["name"], "role": res["role"], "match": match}
        return await self.publish(T.AUDIO, {"room": room, "mic_id": f"mic_{room}", "text": text, "lang": lang,
                                            "asr_conf": 0.9, "wake_word": False, "speaker": speaker,
                                            "doa": {"azimuth_deg": 0.0, "source": source, "conf": 0.9}}, "sim.audio", ts)

    async def reply(self, request_id: str, answer: str, by: str = "alex", ts=None) -> Event:
        return await self.publish(T.REPLY, {"request_id": request_id, "answer": answer, "by": by}, "phone", ts)

    # ---- reading the history
    def events(self, topic: str, **match: Any) -> list[Event]:
        return self.bus.find(topic, **match)

    def proposals(self, **match: Any) -> list[Event]:
        return self.events(T.PROPOSAL, **match)

    def verdicts(self, **match: Any) -> list[Event]:
        return self.events(T.VERDICT, **match)

    def commands(self, **match: Any) -> list[Event]:
        return self.events(T.COMMAND, **match)

    def asks(self, **match: Any) -> list[Event]:
        return self.events(T.ASK, **match)

    def scenes(self, **match: Any) -> list[Event]:
        return self.events(T.SCENE, **match)

    def logs(self, **match: Any) -> list[Event]:
        return self.events(T.LOG, **match)


@pytest.fixture
def settings() -> Settings:
    return Settings(decider_mode="fake")


@pytest.fixture
def make_pipeline(settings):
    def _make(script: list[tuple[str, str, Any]] | None = None, *, deciders: dict[str, Any] | None = None,
              **setting_changes: Any) -> Pipeline:
        for k, v in setting_changes.items():
            setattr(settings, k, v)
        if deciders is None:
            deciders = {"fake": FakeDecider(script=script or [])}
        return Pipeline(settings, deciders)
    return _make
