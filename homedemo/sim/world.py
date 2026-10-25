"""Ground truth of the simulated home: rooms, devices, people, pets, doors, time and light.

World is the only writer of this state. Scenario steps (WorldAPI.step) and action.command
events change it; after each change the SensorHub publishes the sensor events the change
causes. Nothing runs in the background: the runtime calls tick() for drift.
"""
from __future__ import annotations

import copy
import random
from typing import Any

from ..contracts import SCENARIO_STEPS, BusAPI, Event, Settings, T
from .model import (DAY, LIGHT_LUX_FULL, OUTDOOR_CELSIUS, SCHOOL_NIGHTS, WEEKDAYS, Camera, Device, Door, Entity,
                    Mic, Person, Pet, Room, angle_diff, bearing, daylight, fmt_hhmm, parse_hhmm)
from .sensors import TV_CONE_DEG, SensorHub

RUNNER_STEPS = frozenset({"wait", "reply", "expect", "note"})
ROLES = ("parent", "child", "guest", "unknown")
TEMP_STEP_C = 0.5                  # thermal drift toward the setpoint per tick (and per idle minute)
SETPOINT_RANGE = (10.0, 30.0)
NEAR_OFFSET_PX = 30.0              # "near": <device> puts the person this close to it
DOOR_STEP_PX = 30.0                # an outside voice stands this far outside the door
SPOT_MARGIN = 0.15                 # people stand inside the room, away from the walls
LOCATION_ATTRS = ("pose", "near")  # dropped when a person changes rooms


class World:
    """Implements contracts.WorldAPI. Step -> event table: homedemo/sim/README.md."""

    def __init__(self, home: dict[str, Any], bus: BusAPI, settings: Settings | None = None, *,
                 seed: int = 7, noise: float = 0.3) -> None:
        if not 0.0 <= noise <= 1.0:
            raise ValueError("noise must be within 0..1")
        self.home = home
        self.bus = bus
        self.settings = settings or Settings()
        self.seed = seed
        self.noise = noise
        self.rng = random.Random(seed)
        self.hub = SensorHub(self, bus)
        self.clock = 0.0
        self._started = False
        self._build()

    # ------------------------------------------------------------------ construction
    def _build(self) -> None:
        h = self.home
        self.rng = random.Random(self.seed)
        self.clock = 0.0
        start = h.get("start", {})
        self.day_seconds: float = float(parse_hhmm(start.get("time", "12:00")))
        self.weekday: str = start.get("weekday", "Monday")
        if self.weekday not in WEEKDAYS:
            raise ValueError(f"bad weekday {self.weekday!r}")
        th = h.get("thresholds", {})
        self.sunrise = parse_hhmm(th.get("night_to", "06:30"))
        self.sunset = parse_hhmm(th.get("night_from", "19:30"))
        self.rooms: dict[str, Room] = {
            r["id"]: Room(r["id"], r.get("name", r["id"]), float(r["x"]), float(r["y"]), float(r["w"]),
                          float(r["h"]), bool(r.get("indoor", True)))
            for r in h["rooms"]}
        self.devices: dict[str, Device] = {
            d["device_id"]: Device(d["device_id"], d["kind"], d["room"], float(d.get("x", 0)), float(d.get("y", 0)),
                                   copy.deepcopy(d.get("state", {})))
            for d in h.get("devices", [])}
        self.doors: dict[str, Door] = {
            d["door_id"]: Door(d["door_id"], d.get("name", d["door_id"]), list(d.get("between", [])),
                               float(d.get("x", 0)), float(d.get("y", 0)), d.get("lock"), bool(d.get("open", False)))
            for d in h.get("doors", [])}
        sensors = h.get("sensors", {})
        self.cameras: dict[str, Camera] = {c["room"]: Camera(c["camera_id"], c["room"])
                                           for c in sensors.get("cameras", [])}
        self.mics: list[Mic] = [Mic(m["mic_id"], m["room"], float(m["x"]), float(m["y"]))
                                for m in sensors.get("mics", [])]
        first_room = next(iter(self.rooms))
        self.people: dict[str, Person] = {}
        for r in h.get("residents", []):
            role = r.get("role", "unknown")
            p = Person(id=r["id"], name=r.get("name", r["id"].title()), role=role,
                       voice_enrolled=bool(r.get("voice_enrolled", False)), has_phone=bool(r.get("phone", False)),
                       langs=list(r.get("lang", ["en"])),
                       age_group="child" if role == "child" or r.get("age", 99) < 16 else "adult",
                       guest=False, home=True, phone_home=bool(r.get("phone", False)))
            self.people[p.id] = p
            self._place(p, r.get("start_room") or first_room)
        self.pets: dict[str, Pet] = {}
        for r in h.get("pets", []):
            pet = Pet(id=r["id"], name=r.get("name", r["id"].title()), species=r.get("species", "cat"))
            self.pets[pet.id] = pet
            self._place(pet, r.get("start_room") or first_room)
        self.notifications: list[dict[str, Any]] = []
        setpoint = self.setpoint()
        for room in self.rooms.values():
            room.celsius = (20.0 if setpoint is None else setpoint) if room.indoor else OUTDOOR_CELSIUS
        self._recompute_lux()
        self.hub.reset()

    # ------------------------------------------------------------------ lookups
    def _room(self, room_id: Any) -> Room:
        if room_id not in self.rooms:
            raise ValueError(f"unknown room {room_id!r}")
        return self.rooms[room_id]

    def _device(self, device_id: Any) -> Device:
        if device_id not in self.devices:
            raise ValueError(f"unknown device {device_id!r}")
        return self.devices[device_id]

    def _person(self, who: Any) -> Person:
        if who not in self.people:
            raise ValueError(f"unknown person {who!r}")
        return self.people[who]

    def _entity(self, who: Any) -> Entity:
        if who in self.people:
            return self.people[who]
        if who in self.pets:
            return self.pets[who]
        raise ValueError(f"unknown person or pet {who!r}")

    def _door(self, door_id: Any = None) -> Door:
        if door_id is None:
            if not self.doors:
                raise ValueError("the home has no doors")
            return next(iter(self.doors.values()))
        if door_id not in self.doors:
            raise ValueError(f"unknown door {door_id!r}")
        return self.doors[door_id]

    def people_in(self, room_id: str) -> list[Person]:
        return [p for p in self.people.values() if p.home and p.room == room_id]

    def pets_in(self, room_id: str) -> list[Pet]:
        return [p for p in self.pets.values() if p.room == room_id]

    def occupants(self, room_id: str) -> list[Entity]:
        return [*self.people_in(room_id), *self.pets_in(room_id)]

    def devices_in(self, room_id: str, kind: str | None = None) -> list[Device]:
        return [d for d in self.devices.values() if d.room == room_id and (kind is None or d.kind == kind)]

    def setpoint(self) -> float | None:
        for d in self.devices.values():
            if d.kind == "thermostat":
                return float(d.state.get("setpoint_c", 20.0))
        return None

    def door_locked(self, door: Door) -> bool:
        lock = self.devices.get(door.lock or "")
        return bool(lock and lock.state.get("locked"))

    # ------------------------------------------------------------------ geometry
    def _place(self, ent: Entity, room_id: str, x: Any = None, y: Any = None) -> None:
        room = self._room(room_id)
        if x is None or y is None:
            x, y = self._spot(room)
        ent.room, ent.x, ent.y = room.id, float(x), float(y)

    def _spot(self, room: Room) -> tuple[float, float]:
        """A seeded standing spot inside the room, kept out of the TV cone of the room's mic so a
        voice is only mistaken for the TV when a scenario puts someone right in front of it."""
        mx, my = room.w * SPOT_MARGIN, room.h * SPOT_MARGIN
        for _ in range(32):
            x = room.x + mx + self.rng.random() * (room.w - 2 * mx)
            y = room.y + my + self.rng.random() * (room.h - 2 * my)
            if not self._in_tv_cone(room, x, y):
                return round(x, 1), round(y, 1)
        cx, cy = room.center
        return cx, round(cy + room.h * 0.3, 1)

    def _in_tv_cone(self, room: Room, x: float, y: float) -> bool:
        for mic in (m for m in self.mics if m.room == room.id):
            for tv in self.devices_in(room.id, "tv"):
                if angle_diff(bearing(mic.x, mic.y, x, y), bearing(mic.x, mic.y, tv.x, tv.y)) <= TV_CONE_DEG + 5:
                    return True
        return False

    def _snap_near(self, p: Person, near: Any) -> None:
        """attrs.near naming a device in the room (id or kind, not a TV) moves the person next to it."""
        if p.room is None or not isinstance(near, str):
            return
        for dev in self.devices_in(p.room):
            if near in (dev.device_id, dev.kind) and dev.kind != "tv":
                cx, cy = self.rooms[p.room].center
                dx, dy = cx - dev.x, cy - dev.y
                norm = max(1.0, (dx * dx + dy * dy) ** 0.5)
                p.x = round(dev.x + dx / norm * NEAR_OFFSET_PX, 1)
                p.y = round(dev.y + dy / norm * NEAR_OFFSET_PX, 1)
                return

    def _outside_xy(self, door: Door) -> tuple[Room, tuple[float, float]]:
        outdoor = [self.rooms[r] for r in door.between if r in self.rooms and not self.rooms[r].indoor]
        if not outdoor:
            raise ValueError(f"door {door.door_id!r} does not lead outside")
        room = outdoor[0]
        cx, cy = room.center
        dx, dy = cx - door.x, cy - door.y
        norm = max(1.0, (dx * dx + dy * dy) ** 0.5)
        return room, (round(door.x + dx / norm * DOOR_STEP_PX, 1), round(door.y + dy / norm * DOOR_STEP_PX, 1))

    # ------------------------------------------------------------------ time
    @property
    def time_str(self) -> str:
        return fmt_hhmm(self.day_seconds)

    @property
    def school_night(self) -> bool:
        """Sunday..Thursday from 18:00, and the small hours that follow until sunrise."""
        if self.day_seconds >= 18 * 3600:
            return self.weekday in SCHOOL_NIGHTS
        if self.day_seconds < self.sunrise:
            return WEEKDAYS[(WEEKDAYS.index(self.weekday) - 1) % 7] in SCHOOL_NIGHTS
        return False

    def clock_data(self) -> dict[str, Any]:
        return {"time": self.time_str, "weekday": self.weekday, "school_night": self.school_night}

    def _advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time only moves forward")
        self.clock += seconds
        self.day_seconds += seconds
        while self.day_seconds >= DAY:
            self.day_seconds -= DAY
            self.weekday = WEEKDAYS[(WEEKDAYS.index(self.weekday) + 1) % 7]

    def _set_time(self, time: Any = None, weekday: Any = None) -> None:
        if time is not None:
            self._advance((parse_hhmm(time) - self.day_seconds) % DAY)
        if weekday is not None:
            if weekday not in WEEKDAYS:
                raise ValueError(f"bad weekday {weekday!r}")
            self.weekday = weekday
        for room in self.rooms.values():
            room.ambient_override = None
        self._recompute_lux()

    # ------------------------------------------------------------------ light and heat
    def _lights_lux(self, room: Room) -> float:
        return sum(LIGHT_LUX_FULL * float(d.state.get("brightness", 0)) / 100
                   for d in self.devices_in(room.id, "light") if d.state.get("on"))

    def _recompute_lux(self) -> None:
        for room in self.rooms.values():
            ambient = room.ambient_override
            if ambient is None:
                ambient = daylight(self.day_seconds, room.indoor, self.sunrise, self.sunset)
            room.lux = round(ambient + self._lights_lux(room), 1)

    def _drift_temps(self, steps: float) -> None:
        setpoint = self.setpoint()
        if setpoint is None or steps <= 0:
            return
        for room in self.rooms.values():
            if room.indoor:
                diff = setpoint - room.celsius
                move = min(abs(diff), TEMP_STEP_C * steps)
                room.celsius = round(room.celsius + (move if diff > 0 else -move), 2)

    # ------------------------------------------------------------------ lifecycle (WorldAPI)
    async def start(self) -> None:
        """Subscribe to action.command and report the whole initial state."""
        if not self._started:
            self.bus.subscribe(T.COMMAND, self._on_command, name="sim.world")
            self._started = True
        await self.hub.report_all()

    async def reset(self) -> None:
        self._build()
        if self._started:
            await self.hub.report_all()

    async def tick(self, seconds: float = 1.0) -> None:
        """One step of the slow physics: the clock advances, rooms drift 0.5 C toward the setpoint,
        daylight follows the clock. Publishes the changes (clock when the minute changes)."""
        before = self.time_str
        self._advance(seconds)
        self._drift_temps(1)
        self._recompute_lux()
        if self.time_str != before:
            await self.hub.clock()
        await self.hub.sync()

    def snapshot(self) -> dict[str, Any]:
        return {
            "time": self.time_str, "weekday": self.weekday, "school_night": self.school_night,
            "clock": round(self.clock, 3),
            "rooms": [{"id": r.id, "name": r.name, "x": r.x, "y": r.y, "w": r.w, "h": r.h, "indoor": r.indoor,
                       "camera": r.id in self.cameras, "env": {"lux": round(r.lux, 1), "celsius": round(r.celsius, 2)}}
                      for r in self.rooms.values()],
            "devices": [{"device_id": d.device_id, "kind": d.kind, "room": d.room, "x": d.x, "y": d.y,
                         "state": copy.deepcopy(d.state)} for d in self.devices.values()],
            "people": [{"id": p.id, "name": p.name, "role": p.role, "room": p.room, "x": p.x, "y": p.y,
                        "activity": p.activity, "attrs": copy.deepcopy(p.attrs), "idle_min": p.idle_min,
                        "home": p.home, "phone_home": p.phone_home, "guest": p.guest}
                       for p in self.people.values()],
            "pets": [{"id": p.id, "name": p.name, "species": p.species, "room": p.room, "x": p.x, "y": p.y,
                      "activity": p.activity} for p in self.pets.values()],
            "doors": [{"door_id": d.door_id, "name": d.name, "between": list(d.between), "x": d.x, "y": d.y,
                       "open": d.open, "locked": self.door_locked(d)} for d in self.doors.values()],
            "mics": [{"mic_id": m.mic_id, "room": m.room, "x": m.x, "y": m.y} for m in self.mics],
            "notifications": copy.deepcopy(self.notifications),
        }

    # ------------------------------------------------------------------ scenario steps (WorldAPI.step)
    async def step(self, step: dict[str, Any]) -> None:
        if not isinstance(step, dict) or "do" not in step:
            raise ValueError("a step is a dict with a 'do' key")
        do = step["do"]
        if do in RUNNER_STEPS:
            raise ValueError(f"{do!r} is a runner step, not a world step")
        handler = getattr(self, f"_step_{do}", None) if do in SCENARIO_STEPS else None
        if handler is None:
            raise ValueError(f"unknown world step {do!r}")
        await handler(step)

    @staticmethod
    def _text(s: dict[str, Any], key: str = "text") -> str:
        text = s.get(key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"step {s.get('do')!r} needs a non-empty {key!r}")
        return text

    async def _step_set_time(self, s: dict[str, Any]) -> None:
        self._set_time(s.get("time"), s.get("weekday"))
        await self.hub.clock()
        await self.hub.sync()

    async def _step_move(self, s: dict[str, Any]) -> None:
        ent = self._entity(s.get("who"))
        room = self._room(s.get("to"))
        if isinstance(ent, Person) and not ent.home:
            await self._arrive(ent, room, s)  # moving someone who is away = coming home
            return
        self._place(ent, room.id, s.get("x"), s.get("y"))
        ent.activity = None
        if isinstance(ent, Person):
            ent.idle_min = 0.0
            for key in LOCATION_ATTRS:
                ent.attrs.pop(key, None)
        await self.hub.sync()

    async def _step_activity(self, s: dict[str, Any]) -> None:
        ent = self._entity(s.get("who"))
        ent.activity = self._text(s)
        attrs = s.get("attrs") or {}
        if not isinstance(attrs, dict):
            raise ValueError("activity attrs must be an object")
        ent.attrs = copy.deepcopy(attrs)  # replaces: each activity step describes what the camera sees now
        if isinstance(ent, Person):
            ent.idle_min = 0.0
            if "near" in attrs:
                self._snap_near(ent, attrs["near"])
        await self.hub.sync()

    async def _step_idle(self, s: dict[str, Any]) -> None:
        ent = self._entity(s.get("who"))
        minutes = float(s.get("minutes", 0))
        if minutes <= 0:
            raise ValueError("idle needs minutes > 0")
        if isinstance(ent, Person):
            ent.idle_min += minutes
        self._advance(minutes * 60)
        self._drift_temps(minutes)
        self._recompute_lux()
        await self.hub.clock()
        await self.hub.sync()

    async def _step_say(self, s: dict[str, Any]) -> None:
        p = self._person(s.get("who"))
        text = self._text(s)
        if not p.home or p.room is None or p.x is None or p.y is None:
            raise ValueError(f"{p.id} is not at home; use outside_say with who={p.id!r}")
        await self.hub.audio(text=text, lang=s.get("lang"), source_room=p.room, source_xy=(p.x, p.y),
                             speaker=self.hub.speaker_for(p, s.get("match")), asr_conf=s.get("asr_conf"),
                             wake_word=bool(s.get("wake_word", False)))

    async def _step_tv_say(self, s: dict[str, Any]) -> None:
        text = self._text(s)
        tvs = [d for d in self.devices.values() if d.kind == "tv"]
        tv = self._device(s["device"]) if "device" in s else (tvs[0] if tvs else None)
        if tv is None or tv.kind != "tv":
            raise ValueError(f"{s.get('device')!r} is not a TV")
        if not tv.state.get("on"):  # a TV that is talking is on
            tv.state["on"] = True
            await self.hub.sync(force_devices={tv.device_id})
        await self.hub.audio(text=text, lang=s.get("lang"), source_room=tv.room, source_xy=(tv.x, tv.y),
                             speaker=self.hub.speaker_for(None), asr_conf=s.get("asr_conf"))

    async def _step_outside_say(self, s: dict[str, Any]) -> None:
        text = self._text(s)
        room, xy = self._outside_xy(self._door(s.get("door")))
        who = self._person(s["who"]) if s.get("who") else None  # a resident locked out, if named
        await self.hub.audio(text=text, lang=s.get("lang"), source_room=room.id, source_xy=xy,
                             speaker=self.hub.speaker_for(who, s.get("match")), asr_conf=s.get("asr_conf"))

    async def _step_set_env(self, s: dict[str, Any]) -> None:
        room = self._room(s.get("room"))
        if "lux" not in s and "celsius" not in s:
            raise ValueError("set_env needs lux and/or celsius")
        if "lux" in s:
            room.ambient_override = max(0.0, float(s["lux"]) - self._lights_lux(room))
        if "celsius" in s:
            room.celsius = round(float(s["celsius"]), 2)
        self._recompute_lux()
        await self.hub.sync(force_lux={room.id} if "lux" in s else (), force_temp={room.id} if "celsius" in s else ())

    async def _step_set_device(self, s: dict[str, Any]) -> None:
        dev = self._device(s.get("device"))
        state = s.get("state")
        if not isinstance(state, dict) or not state:
            raise ValueError("set_device needs a non-empty state object")
        unknown = [k for k in state if k not in dev.state]
        if unknown:
            raise ValueError(f"{dev.device_id} has no state {unknown}")
        dev.state.update(copy.deepcopy(state))
        if dev.kind == "light":
            self._normalise_light(dev, "on" in state)
        await self._device_changed(dev)

    async def _step_phone(self, s: dict[str, Any]) -> None:
        p = self._person(s.get("who"))
        if not p.has_phone:
            raise ValueError(f"{p.id} has no phone")
        if "home" not in s:
            raise ValueError("phone needs home: true|false")
        p.phone_home = bool(s["home"])
        await self.hub.sync()

    async def _step_leave(self, s: dict[str, Any]) -> None:
        p = self._person(s.get("who"))
        if not p.home:
            return
        door = self._door(s.get("door"))
        door.open = True
        await self.hub.sync()
        p.home, p.room, p.x, p.y = False, None, None, None
        p.activity, p.idle_min = None, 0.0
        for key in LOCATION_ATTRS:
            p.attrs.pop(key, None)
        if p.has_phone:
            p.phone_home = False
        door.open = False
        await self.hub.sync()

    async def _step_arrive(self, s: dict[str, Any]) -> None:
        who = s.get("who")
        if not isinstance(who, str) or not who:
            raise ValueError("arrive needs who")
        room = self._room(s.get("to"))
        p = self.people.get(who)
        if p is None:
            role = s.get("role", "guest")
            if role not in ROLES:
                raise ValueError(f"bad role {role!r}")
            p = Person(id=who, name=s.get("name") or who.replace("_", " ").title(), role=role, guest=True,
                       age_group=s.get("age_group", "adult"), home=False, phone_home=False)
            self.people[who] = p
        if p.home:
            await self._step_move({"do": "move", "who": who, "to": room.id, "x": s.get("x"), "y": s.get("y")})
            return
        await self._arrive(p, room, s)

    async def _arrive(self, p: Person, room: Room, s: dict[str, Any]) -> None:
        door = self._door(s.get("door"))
        door.open = True
        await self.hub.sync()
        p.home = True
        p.phone_home = p.has_phone
        p.activity, p.idle_min = None, 0.0
        self._place(p, room.id, s.get("x"), s.get("y"))
        door.open = False
        await self.hub.sync()

    # ------------------------------------------------------------------ commands (action.command)
    async def _on_command(self, event: Event) -> None:
        data = event.data
        command, args, device_id = data.get("command"), data.get("args") or {}, data.get("device_id")
        if command == "notify":
            note = {"to": str(args.get("to", "parents")), "text": str(args.get("text", "")),
                    "command_id": event.id, "time": self.time_str}
            self.notifications.append(note)
            await self._applied(device_id or "phone", {"to": note["to"], "text": note["text"], "delivered": True},
                                event)
            return
        dev = self._device(device_id)
        self._apply(dev, command, args)
        await self._applied(dev.device_id, copy.deepcopy(dev.state), event)
        await self._device_changed(dev, cause=event.id)

    def _apply(self, dev: Device, command: Any, args: dict[str, Any]) -> None:
        if dev.kind == "light" and command in ("on", "off", "dim"):
            if command == "on":
                dev.state["on"] = True
                dev.state["brightness"] = int(args.get("brightness") or dev.state.get("brightness") or 100)
            elif command == "off":
                dev.state["on"] = False
            else:
                if "brightness" not in args:
                    raise ValueError("dim needs args.brightness")
                dev.state["brightness"] = int(args["brightness"])
                dev.state["on"] = dev.state["brightness"] > 0
            self._normalise_light(dev, True)
        elif dev.kind == "tv" and command in ("on", "off"):
            dev.state["on"] = command == "on"
            if "volume" in args:
                dev.state["volume"] = int(args["volume"])
        elif dev.kind == "thermostat" and command == "set_setpoint":
            current = float(dev.state.get("setpoint_c", 20.0))
            if "setpoint_c" in args:
                target = float(args["setpoint_c"])
            elif "delta_c" in args:
                target = current + float(args["delta_c"])
            else:
                raise ValueError("set_setpoint needs args.setpoint_c or args.delta_c")
            dev.state["setpoint_c"] = round(min(SETPOINT_RANGE[1], max(SETPOINT_RANGE[0], target)), 1)
        elif dev.kind == "lock" and command in ("lock", "unlock"):
            dev.state["locked"] = command == "lock"
        else:
            raise ValueError(f"{dev.device_id} ({dev.kind}) cannot do {command!r}")

    @staticmethod
    def _normalise_light(dev: Device, on_given: bool) -> None:
        st = dev.state
        brightness = int(min(100, max(0, int(st.get("brightness") or 0))))
        if not on_given:
            st["on"] = brightness > 0
        if st.get("on") and brightness == 0:
            brightness = 100
        st["on"] = bool(st.get("on")) and brightness > 0
        st["brightness"] = brightness if st["on"] else 0

    async def _device_changed(self, dev: Device, cause: str | None = None) -> None:
        self._recompute_lux()
        await self.hub.sync(cause=cause, force_devices={dev.device_id})

    async def _applied(self, device_id: str, state: dict[str, Any], cmd: Event) -> None:
        await self.bus.publish(Event(T.APPLIED, {"device_id": device_id, "state": state, "command_id": cmd.id},
                                     source="sim.world", ts=self.clock, cause=cmd.id))
