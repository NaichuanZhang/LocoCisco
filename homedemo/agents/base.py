"""Shared pieces for the pipeline agents: the Agent base class, a read-only view of home.json
(configuration only, never live world state) and the simulated clock as the agents last heard it."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts import AGENTS, BusAPI, Event, Handler, Settings, T

HOME_PATH = Path(__file__).resolve().parent.parent / "home.json"

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
SCHOOL_NIGHTS = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"}

# What each action means in a sentence; shared by proposals, asks and alerts.
ACTION_WORDS: dict[str, str] = {
    "lights_on": "turn the lights on",
    "lights_off": "turn the lights off",
    "lights_dim": "dim the lights",
    "warmer": "make it warmer",
    "cooler": "make it cooler",
    "tv_on": "turn the TV on",
    "tv_off": "turn the TV off",
    "unlock_door": "unlock the front door",
    "lock_door": "lock the front door",
    "alert_parent": "alert a parent",
    "none": "do nothing",
}


def load_home(path: str | Path = HOME_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def hhmm_to_minutes(hhmm: str) -> int:
    """'23:40' -> 1420. Tolerates junk by returning 0."""
    try:
        h, m = str(hhmm).split(":")[:2]
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return 0


def fmt_num(x: float) -> str:
    """2.0 -> '2', 17.5 -> '17.5'."""
    return f"{x:g}"


class HomeConfig:
    """home.json as configuration: rooms, devices, residents, doors, thresholds, house rules."""

    def __init__(self, home: dict[str, Any] | str | Path) -> None:
        raw = load_home(home) if isinstance(home, (str, Path)) else home
        self.raw = raw
        self.rooms: dict[str, dict] = {r["id"]: r for r in raw.get("rooms", [])}
        self.devices: dict[str, dict] = {d["device_id"]: d for d in raw.get("devices", [])}
        self.residents: dict[str, dict] = {r["id"]: r for r in raw.get("residents", [])}
        self.pets: dict[str, dict] = {p["id"]: p for p in raw.get("pets", [])}
        self.doors: list[dict] = list(raw.get("doors", []))
        self.house_rules: list[dict] = list(raw.get("house_rules", []))
        self.thresholds: dict[str, Any] = {
            "voice_match": 0.85, "unlock_voice_match": 0.90, "dark_lux": 10,
            "night_from": "19:30", "night_to": "06:30", **raw.get("thresholds", {}),
        }
        sensors = raw.get("sensors", {})
        self.camera_rooms: set[str] = {c["room"] for c in sensors.get("cameras", [])}
        start = raw.get("start", {})
        self.start_time: str = start.get("time", "12:00")
        self.start_weekday: str = start.get("weekday", "Monday")

    # ---- lookups
    def room_name(self, room_id: str | None) -> str:
        room = self.rooms.get(room_id or "")
        return room["name"] if room else (room_id or "the house")

    def room_name_in_sentence(self, room_id: str | None) -> str:
        """'Living room' -> 'living room', but keep a resident's name: "Maya's room"."""
        name = self.room_name(room_id)
        first = name.split()[0].rstrip("'s").rstrip("'") if name else ""
        if any(first == r.get("name") for r in self.residents.values()):
            return name
        return name[:1].lower() + name[1:]

    def devices_in(self, room_id: str | None, kind: str) -> list[dict]:
        return [d for d in self.devices.values() if d.get("room") == room_id and d.get("kind") == kind]

    def lights_in(self, room_id: str | None) -> list[str]:
        return [d["device_id"] for d in self.devices_in(room_id, "light")]

    def first_device(self, kind: str) -> str | None:
        return next((d["device_id"] for d in self.devices.values() if d.get("kind") == kind), None)

    def tv_for(self, room_id: str | None) -> str | None:
        in_room = self.devices_in(room_id, "tv")
        return in_room[0]["device_id"] if in_room else self.first_device("tv")

    def thermostat(self) -> str | None:
        return self.first_device("thermostat")

    def door_for(self, room_id: str | None) -> dict | None:
        for door in self.doors:
            if room_id in door.get("between", []):
                return door
        return self.doors[0] if self.doors else None

    def lock_for(self, room_id: str | None) -> str | None:
        door = self.door_for(room_id)
        return door.get("lock") if door else self.first_device("lock")

    def target_for(self, action: str, room_id: str | None) -> str | None:
        """Device the action would touch, or None (action 'none', or nothing fits)."""
        if action in ("lights_on", "lights_off", "lights_dim"):
            lights = self.lights_in(room_id)
            return lights[0] if lights else None
        if action in ("tv_on", "tv_off"):
            return self.tv_for(room_id)
        if action in ("warmer", "cooler"):
            return self.thermostat()
        if action in ("unlock_door", "lock_door"):
            return self.lock_for(room_id)
        if action == "alert_parent":
            return "phone_parents"
        return None

    def resident_role(self, person_id: str | None) -> str:
        res = self.residents.get(person_id or "")
        return res.get("role", "unknown") if res else "unknown"

    def resident_name(self, person_id: str | None) -> str | None:
        res = self.residents.get(person_id or "")
        return res.get("name") if res else None

    # ---- thresholds
    def is_dark(self, lux: float | None) -> bool:
        return lux is not None and lux < float(self.thresholds["dark_lux"])

    def is_night(self, hhmm: str) -> bool:
        m = hhmm_to_minutes(hhmm)
        start, end = hhmm_to_minutes(self.thresholds["night_from"]), hhmm_to_minutes(self.thresholds["night_to"])
        return m >= start or m < end if start > end else start <= m < end

    def rule_text(self, rule_id: str, default: str) -> str:
        return next((r["text"] for r in self.house_rules if r.get("id") == rule_id), default)


class ClockState:
    """The simulated clock as last published on sim.clock; starts at home.json's start time."""

    def __init__(self, home: HomeConfig) -> None:
        self.time = home.start_time
        self.weekday = home.start_weekday
        self.school_night = self.weekday in SCHOOL_NIGHTS
        self.known = False  # True once a sim.clock event arrived

    def update(self, data: dict[str, Any]) -> None:
        self.time = str(data.get("time") or self.time)
        self.weekday = str(data.get("weekday") or self.weekday)
        sn = data.get("school_night")
        self.school_night = bool(sn) if sn is not None else self.weekday in SCHOOL_NIGHTS
        self.known = True

    @property
    def minutes(self) -> int:
        return hhmm_to_minutes(self.time)

    def daypart(self) -> str:
        m = self.minutes
        if 5 * 60 <= m < 12 * 60:
            return "morning"
        if 12 * 60 <= m < 17 * 60:
            return "afternoon"
        if 17 * 60 <= m < 21 * 60:
            return "evening"
        return "night"


class Agent:
    """One pipeline agent. Subclasses set `name` (a contracts.AGENTS name) and implement start()."""

    name: str = ""

    def __init__(self, bus: BusAPI, home: HomeConfig, settings: Settings) -> None:
        self.bus = bus
        self.home = home
        self.settings = settings
        spec = next((a for a in AGENTS if a["name"] == self.name), None)
        self.label: str = spec["label"] if spec else self.name
        self.busy: bool = False

    @property
    def source(self) -> str:
        return f"agent.{self.name}"

    def start(self) -> None:
        raise NotImplementedError

    def subscribe(self, pattern: str, handler: Handler) -> None:
        self.bus.subscribe(pattern, handler, name=self.name)

    async def emit(self, topic: str, data: dict[str, Any], cause: Event) -> Event:
        """Publish as this agent, chained to the event that triggered it (same simulated ts)."""
        event = Event(topic, data, source=self.source, ts=cause.ts, cause=cause.id)
        await self.bus.publish(event)
        return event

    async def log(self, text: str, cause: Event, level: str = "info") -> Event:
        return await self.emit(T.LOG, {"agent": self.name, "level": level, "text": text}, cause)
