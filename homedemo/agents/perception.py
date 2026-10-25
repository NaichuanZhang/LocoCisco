"""Perception: fuses sensor.* and sim.clock into one short scene per room (perception.scene).

The scene text follows the wording the decider was measured on, e.g.
  "Time 23:40. Kitchen camera: 1 adult walking in. Kitchen lights: off. Light level: 2 lux (dark)."
  "Time 17:20. Kitchen camera: 1 child (about 4 years old) standing on a chair next to the stove.
   Stove burner: on. No adult in the kitchen."
Perception is edge-triggered: a camera frame identical to the previous one publishes nothing,
identical text for a room is debounced for 2 s, and empty rooms are not described.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..contracts import Event, T
from .base import Agent, ClockState, HomeConfig, fmt_num

DEBOUNCE_S = 2.0
IDLE_JUMP_MIN = 5          # a clock jump of at least this while nobody moves counts as idle ...
IDLE_JUMP_MAX = 180        # ... up to this; a bigger jump is a scenario skipping ahead, not an idle spell
PERSON_LABELS = {"person", "adult", "child", "human"}
COLD_WORDS = ("sweater", "blanket", "shiver", "rubbing", "jacket", "coat", "cold", "freez")
HOT_WORDS = ("fan", "sweat", "hot", "stuffy")
_IDLE_RE = re.compile(r"no movement for (\d+) minutes")


@dataclass
class _Room:
    id: str
    detections: list[dict] = field(default_factory=list)
    people_keys: set[str] = field(default_factory=set)
    signature: tuple | None = None
    activity: str | None = None
    presence: bool | None = None
    lux: float | None = None
    celsius: float | None = None
    lights: dict[str, bool] = field(default_factory=dict)   # light device -> on
    tv: dict[str, Any] | None = None                         # state of the TV in this room
    stove_on: bool | None = None
    door_open: bool | None = None
    hazard: bool = False
    idle_min: int | None = None
    last_text: str = ""
    last_wall: float = 0.0


def _is_person(det: dict) -> bool:
    label = str(det.get("label", "")).lower()
    return label in PERSON_LABELS or "age_group" in (det.get("attrs") or {})


def _key(det: dict, index: int) -> str:
    attrs = det.get("attrs") or {}
    identity = (attrs.get("identity") or {}).get("id")
    return str(det.get("track_id") or identity or f"{det.get('label', 'person')}#{index}")


class PerceptionAgent(Agent):
    name = "perception"

    def __init__(self, bus, home: HomeConfig, settings) -> None:
        super().__init__(bus, home, settings)
        self.clock = ClockState(home)
        self.setpoint_c: float | None = None
        self.rooms: dict[str, _Room] = {}
        for device in home.devices.values():          # initial device states are configuration
            room = self._room(device.get("room"))
            state = device.get("state") or {}
            kind = device.get("kind")
            if kind == "light":
                room.lights[device["device_id"]] = bool(state.get("on"))
            elif kind == "tv":
                room.tv = dict(state)
            elif kind == "stove":
                room.stove_on = bool(state.get("burner_on"))
            elif kind == "thermostat" and state.get("setpoint_c") is not None:
                self.setpoint_c = float(state["setpoint_c"])
        for door in home.doors:
            for room_id in door.get("between", []):
                self._room(room_id).door_open = bool(door.get("open", False))

    def start(self) -> None:
        self.subscribe(T.CLOCK, self._on_clock)
        self.subscribe(T.CAMERA, self._on_camera)
        self.subscribe(T.MOTION, self._on_motion)
        self.subscribe(T.LUX, self._on_lux)
        self.subscribe(T.TEMP, self._on_temp)
        self.subscribe(T.DEVICE, self._on_device)
        self.subscribe(T.DOOR, self._on_door)

    # ------------------------------------------------------------------ handlers
    def _room(self, room_id: str | None) -> _Room:
        rid = room_id or "unknown"
        if rid not in self.rooms:
            self.rooms[rid] = _Room(rid)
        return self.rooms[rid]

    async def _on_clock(self, ev: Event) -> None:
        before = self.clock.minutes if self.clock.known else None
        self.clock.update(ev.data)
        if before is None:
            return
        jump = self.clock.minutes - before
        if not IDLE_JUMP_MIN <= jump <= IDLE_JUMP_MAX:
            return
        for room in self.rooms.values():   # people who stopped moving while the clock jumped are idle
            if room.id in self.home.camera_rooms and room.presence is False and any(map(_is_person, room.detections)):
                room.idle_min = (room.idle_min or 0) + jump
                await self._publish(room, "idle", ev)

    async def _on_camera(self, ev: Event) -> None:
        d = ev.data
        room = self._room(d.get("room"))
        detections = [dict(x) for x in (d.get("detections") or [])]
        people = [x for x in detections if _is_person(x)]
        new_keys = {_key(x, i) for i, x in enumerate(people)}
        entered = new_keys - room.people_keys
        signature = (
            tuple(sorted((_key(x, i), str(x.get("attrs") or {})) for i, x in enumerate(detections))),
            d.get("activity"),
        )
        changed = signature != room.signature
        left = room.people_keys - new_keys
        same_activity = d.get("activity") == room.activity
        room.detections, room.people_keys, room.signature = detections, new_keys, signature
        room.activity = d.get("activity")
        room.idle_min = self._idle_from_frame(d, detections)
        if not detections:
            room.hazard = False
            return                                     # an empty room is not a scene
        if not changed and not entered:
            return                                     # same frame as before
        hazard_now = self._hazard(room)
        if hazard_now and not room.hazard:
            trigger = "hazard"
        elif entered:
            trigger = "person_entered"
        elif room.idle_min:
            trigger = "idle"
        elif left and same_activity and not hazard_now:
            trigger = "person_left"                    # only someone leaving: nothing new to decide
        else:
            trigger = "activity_changed"
        room.hazard = hazard_now
        await self._publish(room, trigger, ev, entering=entered)

    async def _on_motion(self, ev: Event) -> None:
        room = self._room(ev.data.get("room"))
        was = room.presence
        room.presence = bool(ev.data.get("motion"))
        if room.id in self.home.camera_rooms:
            return                                     # the camera is the truth there
        if room.presence and not was:
            await self._publish(room, "person_entered", ev, entering={"presence"})

    async def _on_lux(self, ev: Event) -> None:
        room = self._room(ev.data.get("room"))
        lux = float(ev.data.get("lux", 0))
        if lux == room.lux:
            return
        room.lux = lux
        await self._env_changed(room, ev)

    async def _on_temp(self, ev: Event) -> None:
        room = self._room(ev.data.get("room"))
        celsius = float(ev.data.get("celsius", 0))
        if celsius == room.celsius:
            return
        room.celsius = celsius
        await self._env_changed(room, ev)

    async def _on_device(self, ev: Event) -> None:
        d = ev.data
        kind = d.get("kind") or (self.home.devices.get(d.get("device_id", ""), {}).get("kind"))
        room_id = d.get("room") or self.home.devices.get(d.get("device_id", ""), {}).get("room")
        room = self._room(room_id)
        state = d.get("state") or {}
        if kind == "light":
            on = bool(state.get("on"))
            if room.lights.get(d.get("device_id", "light")) is on:
                return
            room.lights[d.get("device_id", "light")] = on
        elif kind == "tv":
            if room.tv == dict(state):
                return
            room.tv = dict(state)
        elif kind == "stove":
            burner_on = bool(state.get("burner_on"))
            if burner_on is room.stove_on:
                return
            room.stove_on = burner_on
            hazard_now = self._hazard(room)
            if hazard_now and not room.hazard:
                room.hazard = True
                await self._publish(room, "hazard", ev)
                return
            room.hazard = hazard_now
        elif kind == "thermostat" and state.get("setpoint_c") is not None:
            if float(state["setpoint_c"]) == self.setpoint_c:
                return
            self.setpoint_c = float(state["setpoint_c"])
        else:
            return
        await self._env_changed(room, ev)

    async def _on_door(self, ev: Event) -> None:
        door = next((x for x in self.home.doors if x.get("door_id") == ev.data.get("door_id")), None)
        for room_id in (door or {}).get("between", []):
            room = self._room(room_id)
            is_open = bool(ev.data.get("open"))
            if is_open is room.door_open:
                continue
            room.door_open = is_open
            await self._env_changed(room, ev)

    async def _env_changed(self, room: _Room, ev: Event) -> None:
        if self._occupied(room):
            await self._publish(room, "env_changed", ev)

    # ------------------------------------------------------------------ scene building
    def _occupied(self, room: _Room) -> bool:
        if room.id in self.home.camera_rooms:
            return bool(room.detections)
        return bool(room.presence)

    def _people(self, room: _Room) -> list[dict]:
        return [x for x in room.detections if _is_person(x)]

    def _is_child(self, det: dict) -> bool:
        attrs = det.get("attrs") or {}
        if attrs.get("age_group"):
            return attrs["age_group"] == "child"
        identity = (attrs.get("identity") or {}).get("id")
        return self.home.resident_role(identity) == "child"

    def _hazard(self, room: _Room) -> bool:
        """A child next to a lit stove with no adult in the room."""
        if not room.stove_on:
            return False
        people = self._people(room)
        children = [x for x in people if self._is_child(x)]
        adults = [x for x in people if not self._is_child(x)]
        near_stove = any("stove" in f"{(x.get('attrs') or {}).get('near', '')} {(x.get('attrs') or {}).get('pose', '')}".lower()
                         for x in children)
        return bool(children) and near_stove and not adults

    def _idle_from_frame(self, data: dict, detections: list[dict]) -> int | None:
        candidates = [data.get("idle_min")] + [(x.get("attrs") or {}).get("idle_min") for x in detections]
        for c in candidates:
            if c:
                return int(c)
        m = _IDLE_RE.search(str(data.get("activity") or ""))
        return int(m.group(1)) if m else None

    def _age_phrase(self, det: dict) -> str:
        attrs = det.get("attrs") or {}
        identity = (attrs.get("identity") or {}).get("id")
        age = attrs.get("age") or (self.home.residents.get(identity or "", {}).get("age"))
        return f"child (about {age} years old)" if age else "child"

    def _people_phrase(self, room: _Room, trigger: str, entering: set[str]) -> str:
        groups: dict[str, int] = {}
        bare = True                        # nobody has a pose/attire, so the activity follows without a comma
        for i, det in enumerate(self._people(room)):
            attrs = det.get("attrs") or {}
            if self._is_child(det):
                desc = self._age_phrase(det)
            else:
                desc = "adult" if attrs.get("age_group") == "adult" or attrs.get("identity") else "person"
            activity = str(room.activity or "").lower()
            for item in attrs.get("wearing") or []:
                if item.lower() not in activity:
                    desc += f" in a {item}"
                    bare = False
            pose = str(attrs.get("pose") or "")
            # the activity sentence often restates the pose ("standing on a chair next to the stove"): say it once
            pose_said = bool(pose) and _bare_words(pose) in _bare_words(activity)
            if pose and not pose_said:
                desc += f" {pose}"
                bare = False
            near = str(attrs.get("near") or "")
            if near and near.lower() not in pose.lower() and _bare_words(near) not in _bare_words(activity):
                desc += f" next to the {near}"
                bare = False
            held = [h for h in (attrs.get("holding") or []) if h.lower() not in activity]
            if held:
                desc += ", holding " + " and ".join(held)
                bare = False
            if trigger == "person_entered" and _key(det, i) in entering and not attrs.get("pose") and not room.activity:
                desc += " walking in"
            groups[desc] = groups.get(desc, 0) + 1
        parts = [f"{n} {_plural(desc, n)}" for desc, n in groups.items()]
        pets = [x for x in room.detections if not _is_person(x)]
        pet_groups: dict[str, int] = {}
        for det in pets:
            pet_groups[str(det.get("label", "pet"))] = pet_groups.get(str(det.get("label", "pet")), 0) + 1
        parts += [f"{n} {_plural(label, n)}" for label, n in pet_groups.items()]
        phrase = _join(parts) if parts else "nobody"
        if room.activity and str(room.activity).lower() not in phrase.lower():
            phrase += f" {room.activity}" if bare and len(parts) == 1 else f", {room.activity}"
        if room.idle_min and "no movement" not in str(room.activity or ""):
            phrase += f", no movement for {room.idle_min} minutes"
        if not groups and pets:
            phrase += ". No people detected"
        return phrase

    def _describe(self, room: _Room, trigger: str, entering: set[str]) -> str:
        name = self.home.room_name(room.id)
        parts = [f"Time {self.clock.time}."]
        if room.id in self.home.camera_rooms:
            parts.append(f"{name} camera: {self._people_phrase(room, trigger, entering)}.")
        else:
            parts.append(f"{name} (no camera): presence sensor detects someone in the room.")
        lights_on = any(room.lights.values()) if room.lights else None
        # lighting is worth a line when someone arrives, the environment changed, or it is dark with the lights off
        show_lights = trigger in ("person_entered", "env_changed") or (lights_on is False and self.home.is_dark(room.lux))
        if show_lights and lights_on is not None:
            parts.append(f"{name} lights: {'on' if lights_on else 'off'}.")
        if show_lights and room.lux is not None:
            parts.append(f"Light level: {fmt_num(room.lux)} lux ({self._lux_words(room.lux)}).")
        if room.tv is not None:
            if room.tv.get("on"):
                vol = room.tv.get("volume")
                parts.append(f"TV: on, volume {vol}." if vol is not None else "TV: on.")
            elif trigger in ("activity_changed", "idle"):
                parts.append("TV: off.")
        if room.stove_on:
            parts.append("Stove burner: on.")
        people = self._people(room)
        if people and all(self._is_child(x) for x in people):
            parts.append(f"No adult in the {self.home.room_name_in_sentence(room.id)}.")
        activity = str(room.activity or "").lower()
        notable_temp = room.celsius is not None and abs(room.celsius - 21.0) >= 3
        if room.celsius is not None and (trigger == "env_changed" or notable_temp
                                         or any(w in activity for w in COLD_WORDS + HOT_WORDS)):
            parts.append(f"Room temperature: {fmt_num(room.celsius)} C.")
            if self.setpoint_c is not None:
                parts.append(f"Thermostat set to {fmt_num(self.setpoint_c)} C.")
        if room.door_open is not None and trigger != "idle":
            door = self.home.door_for(room.id)
            parts.append(f"{(door or {}).get('name', 'Door')}: {'open' if room.door_open else 'closed'}.")
        return " ".join(parts)

    def _lux_words(self, lux: float) -> str:
        if self.home.is_dark(lux):
            return "dark"
        if lux < 80:
            return "dim"
        if lux < 400:
            return "lit"
        return "bright sunlight" if 6 * 60 <= self.clock.minutes < 19 * 60 else "bright"

    def _facts(self, room: _Room, trigger: str) -> dict[str, Any]:
        people = []
        for det in self._people(room):
            attrs = det.get("attrs") or {}
            identity = (attrs.get("identity") or {}).get("id")
            role = self.home.resident_role(identity) if identity else ("child" if self._is_child(det) else "unknown")
            people.append({"id": identity, "role": role, "age_group": "child" if self._is_child(det) else attrs.get("age_group"),
                           "activity": attrs.get("pose") or room.activity})
        if room.id not in self.home.camera_rooms and room.presence:
            people.append({"id": None, "role": "unknown", "age_group": None, "activity": None})
        pets = [{"id": ((x.get("attrs") or {}).get("identity") or {}).get("id"), "species": x.get("label")}
                for x in room.detections if not _is_person(x)]
        return {
            "time": self.clock.time, "weekday": self.clock.weekday, "school_night": self.clock.school_night,
            "trigger": trigger, "camera": room.id in self.home.camera_rooms,
            "people": people, "pets": pets,
            "lux": room.lux, "lights_on": any(room.lights.values()) if room.lights else None,
            "tv_on": bool(room.tv.get("on")) if room.tv else None,
            "celsius": room.celsius, "setpoint_c": self.setpoint_c,
            "stove_on": room.stove_on, "idle_min": room.idle_min, "hazard": room.hazard,
        }

    async def _publish(self, room: _Room, trigger: str, cause: Event, entering: set[str] | None = None) -> None:
        text = self._describe(room, trigger, entering or set())
        now = time.monotonic()
        if text == room.last_text and now - room.last_wall < DEBOUNCE_S:
            return
        room.last_text, room.last_wall = text, now
        await self.emit(T.SCENE, {"room": room.id, "trigger": trigger, "text": text,
                                  "facts": self._facts(room, trigger)}, cause)


def _plural(desc: str, n: int) -> str:
    if n == 1:
        return desc
    head, _, rest = desc.partition(" ")
    head = {"person": "people", "child": "children"}.get(head, head + "s")
    return f"{head} {rest}".strip()


def _bare_words(text: str) -> str:
    """Lower-case words without articles or commas, so "standing on chair" matches "standing on a chair"."""
    return " " + " ".join(w for w in re.findall(r"[\w']+", text.lower()) if w not in ("a", "an", "the")) + " "


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
