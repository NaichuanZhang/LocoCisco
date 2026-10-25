"""Test doubles for the scenario runner: no sim, no agents, no model.

FakeWorld   implements WorldAPI: records every step, keeps device state from home.json, publishes
            the sensor events a step implies, and answers action.command with action.applied.
RulePipeline a tiny stand-in for the whole agent pipeline, driven by lookup tables: keyword shadow
            proposals (the real baseline's phrase list), a "decider" answer per utterance / activity
            text (with a target room for requests about another room), the two reflex rules (dark
            room at night; child at a lit stove with no adult -> alert), the house rules from the brief
            (TV and outside voices denied, kids may not unlock, curfew asks a parent, alerts always
            allowed, confidence bands), and an actuator that issues commands or phone asks.
            `delay` publishes the downstream events detached from the bus (loop.call_later), so the
            runner's watch window is exercised after drain() returned.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

from homedemo.bus import Bus
from homedemo.contracts import Event, T
from homedemo.scenarios.homeindex import HomeIndex
from homedemo.scenarios.steps import WORLD_STEPS

SIM = "fake-sim"


class FakeWorld:
    def __init__(self, bus: Bus, home: dict[str, Any]) -> None:
        self.bus = bus
        self.home = HomeIndex(home)
        self.steps: list[dict[str, Any]] = []
        self.resets = 0
        self.notifications: list[dict[str, Any]] = []
        self.clock = 0.0
        self.step_delay = 0.0  # extra sleep inside step(), to test cancellation mid-step
        self._init_state()
        bus.subscribe(T.COMMAND, self._on_command, name="fakeworld")

    def _init_state(self) -> None:
        start = self.home.raw.get("start", {})
        self.time: str = start.get("time", "19:00")
        self.weekday: str = start.get("weekday", "Tuesday")
        self.devices: dict[str, dict[str, Any]] = {d: dict(dev["state"]) for d, dev in self.home.devices.items()}
        self.env: dict[str, dict[str, float]] = {r: {} for r in self.home.rooms}
        self.people: dict[str, str | None] = {r: res.get("start_room") for r, res in self.home.residents.items()}
        self.people.update({p: pet.get("start_room") for p, pet in self.home.pets.items()})
        self.roles: dict[str, str] = {r: res.get("role", "unknown") for r, res in self.home.residents.items()}
        self.species: dict[str, str] = {p: pet.get("species", "cat") for p, pet in self.home.pets.items()}
        self.attrs: dict[str, dict[str, Any]] = {}

    # -------------------------------------------------------------- WorldAPI
    def snapshot(self) -> dict[str, Any]:
        return {"time": self.time, "weekday": self.weekday,
                "devices": [{"device_id": d, "kind": self.home.devices[d]["kind"], "room": self.home.devices[d]["room"],
                             "state": dict(s)} for d, s in self.devices.items()],
                "people": [{"id": p, "room": r} for p, r in self.people.items()]}

    async def reset(self) -> None:
        self.resets += 1
        self.clock = 0.0
        self._init_state()

    async def step(self, step: dict[str, Any]) -> None:
        do = step.get("do")
        if do not in WORLD_STEPS:
            raise ValueError(f"unknown world step {do!r}")
        self.steps.append(step)
        self.clock += 1.0
        if self.step_delay:
            await asyncio.sleep(self.step_delay)
        handler = getattr(self, f"_do_{do}")
        await handler(step)

    # -------------------------------------------------------------- steps
    async def _pub(self, topic: str, data: dict[str, Any], source: str = SIM) -> None:
        await self.bus.publish(Event(topic, data, source=source, ts=self.clock))

    def _school_night(self) -> bool:
        return self.weekday in ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday")

    async def _do_set_time(self, s: dict[str, Any]) -> None:
        self.time = s["time"]
        self.weekday = s.get("weekday", self.weekday)
        await self._pub(T.CLOCK, {"time": self.time, "weekday": self.weekday, "school_night": self._school_night()})

    def _detections(self, room: str, entered: str | None = None) -> list[dict[str, Any]]:
        out = []
        for who, where in self.people.items():
            if where != room:
                continue
            if who in self.species:
                out.append({"label": self.species[who], "conf": 0.9, "track_id": who, "attrs": {}})
                continue
            attrs = {"age_group": "child" if self.roles.get(who) == "child" else "adult",
                     "identity": {"id": who, "match": 0.9}, **self.attrs.get(who, {})}
            if who == entered:
                attrs["entered"] = True
            out.append({"label": "person", "conf": 0.95, "track_id": who, "attrs": attrs})
        return out

    async def _camera(self, room: str, activity: str | None = None, entered: str | None = None) -> None:
        await self._pub(T.CAMERA, {"room": room, "camera_id": f"cam_{room}", "activity": activity,
                                   "detections": self._detections(room, entered)})

    async def _do_move(self, s: dict[str, Any]) -> None:
        self.people[s["who"]] = s["to"]
        self.attrs.pop(s["who"], None)
        await self._pub(T.MOTION, {"room": s["to"], "motion": True})
        await self._camera(s["to"], entered=s["who"])

    async def _do_arrive(self, s: dict[str, Any]) -> None:
        self.people[s["who"]] = s["to"]
        self.roles.setdefault(s["who"], "guest")
        await self._camera(s["to"], entered=s["who"])

    async def _do_leave(self, s: dict[str, Any]) -> None:
        self.people[s["who"]] = None

    async def _do_activity(self, s: dict[str, Any]) -> None:
        self.attrs[s["who"]] = dict(s.get("attrs") or {})
        room = self.people.get(s["who"])
        if room:
            await self._camera(room, activity=s["text"])

    async def _do_idle(self, s: dict[str, Any]) -> None:
        self.clock += float(s["minutes"]) * 60
        room = self.people.get(s["who"])
        if room:
            await self._camera(room, activity=f"no movement for {s['minutes']} minutes")

    async def _audio(self, room: str, text: str, lang: str, speaker: dict[str, Any], source: str) -> None:
        await self._pub(T.AUDIO, {"room": room, "mic_id": f"mic_{room}", "text": text, "lang": lang, "asr_conf": 0.95,
                                  "wake_word": False, "speaker": speaker,
                                  "doa": {"azimuth_deg": 0.0, "source": source, "conf": 0.9}})

    async def _do_say(self, s: dict[str, Any]) -> None:
        who = s["who"]
        room = self.people.get(who) or "living"
        if who in self.home.residents:
            speaker = {"id": who, "name": self.home.name(who), "role": self.roles[who], "match": 0.95}
        else:
            speaker = {"id": None, "name": None, "role": "guest", "match": 0.0}
        await self._audio(room, s["text"], s.get("lang", "en"), speaker, "person")

    async def _do_tv_say(self, s: dict[str, Any]) -> None:
        room = self.home.device_room(s.get("device", "tv_living")) or "living"
        await self._audio(room, s["text"], s.get("lang", "en"), {"id": None, "role": "unknown", "match": 0.0}, "tv")

    async def _do_outside_say(self, s: dict[str, Any]) -> None:
        door = self.home.doors.get(s.get("door", "front"), {})
        outside = next((r for r in door.get("between", []) if not self.home.rooms.get(r, {}).get("indoor", True)), "porch")
        await self._audio(outside, s["text"], s.get("lang", "en"), {"id": None, "role": "unknown", "match": 0.0}, "outside")

    async def _do_set_env(self, s: dict[str, Any]) -> None:
        if "lux" in s:
            self.env[s["room"]]["lux"] = float(s["lux"])
            await self._pub(T.LUX, {"room": s["room"], "lux": float(s["lux"])})
        if "celsius" in s:
            self.env[s["room"]]["celsius"] = float(s["celsius"])
            await self._pub(T.TEMP, {"room": s["room"], "celsius": float(s["celsius"])})

    async def _do_set_device(self, s: dict[str, Any]) -> None:
        self.devices[s["device"]].update(s["state"])
        dev = self.home.devices[s["device"]]
        await self._pub(T.DEVICE, {"device_id": s["device"], "kind": dev["kind"], "room": dev["room"],
                                   "state": dict(self.devices[s["device"]])})

    async def _do_phone(self, s: dict[str, Any]) -> None:
        await self._pub(T.PHONE, {"person_id": s["who"], "home": bool(s["home"])})

    # -------------------------------------------------------------- commands
    async def _on_command(self, e: Event) -> None:
        d = e.data
        device = d.get("device_id", "")
        command = d.get("command")
        args = d.get("args") or {}
        if device not in self.devices:
            if command == "notify":
                self.notifications.append(dict(args))
            return
        state = self.devices[device]
        if command == "on":
            state["on"] = True
            if "brightness" in state:
                state["brightness"] = args.get("brightness", 100)
        elif command == "off":
            state["on"] = False
            if "brightness" in state:
                state["brightness"] = 0
        elif command == "dim":
            state["on"] = True
            state["brightness"] = args.get("brightness", 30)
        elif command == "set_setpoint":
            state["setpoint_c"] = round(state.get("setpoint_c", 19.0) + float(args.get("delta_c", 0)), 1)
        elif command == "unlock":
            state["locked"] = False
        elif command == "lock":
            state["locked"] = True
        await self.bus.publish(Event(T.APPLIED, {"device_id": device, "state": dict(state), "command_id": e.id},
                                     source=SIM, ts=self.clock, cause=e.id))


# ------------------------------------------------------------------ fake pipeline
# The naive baseline's phrase list, mirroring homedemo/agents/keyword.py (substring, no negation; the
# phrase found earliest in the text wins, ties go to list order).
KEYWORDS: tuple[tuple[str, str], ...] = (
    ("turn on the lights", "lights_on"), ("turn on the light", "lights_on"),
    ("lights on", "lights_on"), ("light on", "lights_on"), ("开灯", "lights_on"),
    ("turn off the lights", "lights_off"), ("turn off the light", "lights_off"),
    ("lights off", "lights_off"), ("light off", "lights_off"), ("关灯", "lights_off"),
    ("turn on the tv", "tv_on"), ("tv on", "tv_on"), ("开电视", "tv_on"),
    ("turn off the tv", "tv_off"), ("tv off", "tv_off"), ("关电视", "tv_off"),
    ("warmer", "warmer"), ("cooler", "cooler"),
    ("unlock", "unlock_door"),
)


def keyword_match(text: str) -> str:
    low = text.lower()
    hits = [(low.find(phrase), order, action) for order, (phrase, action) in enumerate(KEYWORDS) if phrase in low]
    return min(hits)[2] if hits else "none"


# What the "decider" answers for each utterance in the library: (action, confidence) or
# (action, confidence, target room) when the words name another room; measured-ish values.
SPEECH: dict[str, tuple] = {
    "Can you turn on the lights?": ("lights_on", 0.87),
    "I can't see a thing in here.": ("lights_on", 0.67),
    "It's freezing in here.": ("warmer", 0.58),
    "Ugh, I'm sweating, it's so stuffy.": ("cooler", 0.89),
    "Where's the remote? The game starts in five minutes.": ("tv_on", 0.78),
    "It's way too bright, my eyes hurt.": ("lights_off", 0.93),
    "好冷啊。": ("warmer", 0.82),
    "把灯关了吧，我要睡了。": ("lights_off", 0.90),
    "太黑了，什么都看不见。": ("lights_on", 0.80),
    "Unlock the front door.": ("unlock_door", 0.95),
    "Could you make it a bit warmer?": ("warmer", 0.90),
    "It's cold in here.": ("warmer", 0.60),
    "Turn on the TV.": ("tv_on", 0.95),
    "Turn on the lights in my room.": ("lights_on", 0.90),
    "Turn off the lights.": ("lights_off", 0.90),
    # research scenarios (res_*.json)
    "Alexa ordered me a dollhouse.": ("none", 0.90),
    "客厅有点暗，turn on the lights.": ("lights_on", 0.90),
    "好暗，我看不清。": ("lights_on", 0.70),
    "Don't turn the hall light on, Maya just fell asleep.": ("none", 0.85),
    "我不在家，don't turn on anything.": ("none", 0.85),
    "Is the porch light on?": ("none", 0.80),
    "Did you lock the front door?": ("none", 0.80),
    "Lights on in Maya's room please.": ("lights_on", 0.90, "bedroom_maya"),
    "Kitchen lights off.": ("lights_off", 0.90, "kitchen"),
    "Man, it's a sauna in here.": ("cooler", 0.85),
    "Unlock the door, my friend is here!": ("unlock_door", 0.95),
    "Brr, is it always this chilly in here?": ("warmer", 0.70),
    "Grandma's heater is always cranked up, it's so much warmer at her place.": ("none", 0.85),
    "If it gets any colder tonight we should turn the heat up.": ("none", 0.80),
}
SCENE: dict[str, tuple[str, float]] = {
    "sitting on the couch, lifting couch cushions and looking around the coffee table": ("tv_on", 0.63),
    "putting on a sweater, rubbing their arms, then sitting back down on the couch with a book": ("warmer", 0.87),
    "no movement for 40 minutes": ("tv_off", 0.60),
}
DEVICE_FOR: dict[str, tuple[str, str, dict[str, Any]]] = {   # action -> (device or kind, command, args)
    "lights_on": ("light", "on", {"brightness": 100}), "lights_off": ("light", "off", {}),
    "lights_dim": ("light", "dim", {"brightness": 30}),
    "tv_on": ("tv_living", "on", {}), "tv_off": ("tv_living", "off", {}),
    "warmer": ("thermostat_main", "set_setpoint", {"delta_c": 1.5}),
    "cooler": ("thermostat_main", "set_setpoint", {"delta_c": -1.5}),
    "unlock_door": ("lock_front", "unlock", {}), "lock_door": ("lock_front", "lock", {}),
    "alert_parent": ("phone_parents", "notify", {"to": "parents", "text": "Maya is next to the lit stove"}),
}


class RulePipeline:
    def __init__(self, bus: Bus, home: dict[str, Any], *, speech: dict[str, tuple] | None = None,
                 scene: dict[str, tuple[str, float]] | None = None, delay: float = 0.0,
                 act_threshold: float = 0.80, ask_threshold: float = 0.45) -> None:
        self.bus = bus
        self.home = HomeIndex(home)
        self.speech = SPEECH if speech is None else speech
        self.scene = SCENE if scene is None else scene
        self.delay = delay
        self.act, self.ask = act_threshold, ask_threshold
        self.time, self.weekday = "19:00", "Tuesday"
        self.lux: dict[str, float] = {}
        self.devices: dict[str, dict[str, Any]] = {d: dict(dev["state"]) for d, dev in self.home.devices.items()}
        self.open: dict[str, tuple[str, str | None]] = {}
        self.commands: list[dict[str, Any]] = []
        self.replies: list[dict[str, Any]] = []
        bus.subscribe(T.CLOCK, self._on_clock, name="fakepipe.clock")
        bus.subscribe(T.LUX, self._on_lux, name="fakepipe.lux")
        bus.subscribe(T.DEVICE, self._on_device, name="fakepipe.device")
        bus.subscribe(T.APPLIED, self._on_device, name="fakepipe.applied")
        bus.subscribe(T.CAMERA, self._on_camera, name="fakepipe.camera")
        bus.subscribe(T.AUDIO, self._on_audio, name="fakepipe.audio")
        bus.subscribe(T.REPLY, self._on_reply, name="fakepipe.reply")

    # -------------------------------------------------------------- state
    async def _on_clock(self, e: Event) -> None:
        self.time, self.weekday = e.data["time"], e.data.get("weekday", self.weekday)

    async def _on_lux(self, e: Event) -> None:
        self.lux[e.data["room"]] = float(e.data["lux"])

    async def _on_device(self, e: Event) -> None:
        self.devices.setdefault(e.data["device_id"], {}).update(e.data.get("state") or {})

    def _night(self) -> bool:
        return self.time >= "19:30" or self.time < "06:30"

    def _after(self, hhmm: str) -> bool:
        return self.time >= hhmm

    def _school_night(self) -> bool:
        return self.weekday in ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday")

    def _light_on(self, room: str) -> bool:
        return any(self.devices.get(d, {}).get("on") for d in self.home.devices_of_kind("light", room))

    # -------------------------------------------------------------- perception + deciders
    async def _on_camera(self, e: Event) -> None:
        room = e.data["room"]
        persons = [d for d in e.data.get("detections", []) if d.get("label") == "person"]
        entered = any((d.get("attrs") or {}).get("entered") for d in persons)
        if entered and self.lux.get(room, 100) < self.home.thresholds.get("dark_lux", 10) and self._night() \
                and not self._light_on(room):
            await self._propose(e, "reflex", "lights_on", 1.0, room, None, "person", "scene", "dark room + person at night")
            return
        activity = e.data.get("activity")
        if not activity:
            return
        attrs = [(d.get("attrs") or {}) for d in persons]
        child_near_stove = any(a.get("age_group") == "child" and a.get("near") == "stove" for a in attrs)
        adult_present = any(a.get("age_group") == "adult" for a in attrs)
        if child_near_stove and not adult_present and self.devices.get("stove_kitchen", {}).get("burner_on"):
            # safety is a plain rule first (reflex), and the scene decider agrees after it
            await self._propose(e, "reflex", "alert_parent", 1.0, room, None, "person", "scene", "child at a lit stove, no adult")
            action, conf = "alert_parent", 0.92
        else:
            action, conf = self.scene.get(activity, ("none", 0.6))
        if action == "tv_off" and not self.devices.get("tv_living", {}).get("on"):
            action = "none"
        await self._propose(e, "decider-2b", action, conf, room, None, "person", "scene", "scene decider")

    async def _on_audio(self, e: Event) -> None:
        text = e.data["text"]
        room = e.data["room"]
        speaker = e.data.get("speaker") or {}
        source = (e.data.get("doa") or {}).get("source", "person")
        kw = keyword_match(text)
        await self.bus.publish(Event(T.PROPOSAL, {"by": "keyword", "mode": "shadow", "room": room, "action": kw,
                                                  "confidence": 1.0, "kind": "speech", "speaker": speaker,
                                                  "reason": "substring match"}, source="agent.keyword", ts=e.ts, cause=e.id))
        action, conf, *target_room = self.speech.get(text, ("none", 0.7))
        await self._propose(e, "decider-2b", action, conf, target_room[0] if target_room else room, speaker, source,
                            "speech", "intent decider")

    # -------------------------------------------------------------- policy + actuator
    async def _propose(self, cause: Event, by: str, action: str, conf: float, room: str, speaker: dict | None,
                       source: str, kind: str, reason: str) -> None:
        prop = Event(T.PROPOSAL, {"by": by, "mode": "act", "room": room, "action": action, "confidence": conf,
                                  "kind": kind, "speaker": speaker, "reason": reason, "question": "", "state_text": "",
                                  "probabilities": {action: conf}, "latency_ms": 5.0},
                     source=f"agent.{by}", ts=cause.ts, cause=cause.id)
        await self.bus.publish(prop)
        verdict, rule, to = self._policy(action, conf, speaker or {}, source)
        downstream = self._downstream(prop, action, room, speaker, verdict, rule, to)
        if self.delay:
            asyncio.get_running_loop().call_later(self.delay, lambda: asyncio.ensure_future(downstream))
        else:
            await downstream

    def _policy(self, action: str, conf: float, speaker: dict[str, Any], source: str) -> tuple[str, str, str]:
        role = speaker.get("role", "unknown")
        to = speaker.get("id") or "parents"
        if action == "none":
            return "ignore", "nothing to do", to
        if source in ("tv", "outside"):
            return "deny", "only voices inside the house count", to
        if action == "unlock_door":
            if role == "parent" and float(speaker.get("match") or 0) >= self.home.thresholds.get("unlock_voice_match", 0.9):
                return "allow", "parent voice match", to
            return "deny", "kids may not unlock doors" if role == "child" else "unlock needs a parent's voice", to
        if action == "tv_on" and role == "child" and self._after("20:00") and self._school_night():
            return "ask_parent", "kids' TV curfew on school nights", "parents"
        if action == "alert_parent":
            return "allow", "safety alerts always allowed", to
        if conf >= self.act:
            return "allow", f"confidence {conf:.2f} >= {self.act:.2f}", to
        if conf >= self.ask:
            return "ask_speaker", f"confidence {conf:.2f} in the ask band", to
        return "ignore", f"confidence {conf:.2f} too low", to

    async def _downstream(self, prop: Event, action: str, room: str, speaker: dict | None, verdict: str,
                          rule: str, to: str) -> None:
        target = self._device_for(action, room)
        v = Event(T.VERDICT, {"proposal_id": prop.id, "action": action, "target": target, "room": room,
                              "verdict": verdict, "rule": rule, "who": speaker}, source="agent.policy", ts=prop.ts, cause=prop.id)
        await self.bus.publish(v)
        if verdict == "allow":
            await self._command(action, room, v)
        elif verdict in ("ask_speaker", "ask_parent"):
            rid = uuid.uuid4().hex[:8]
            self.open[rid] = (action, room)
            await self.bus.publish(Event(T.ASK, {"request_id": rid, "to": to, "text": f"Did you want {action.replace('_', ' ')}?",
                                                 "options": ["Yes", "No"], "proposal_id": prop.id, "action": action,
                                                 "target": target, "room": room}, source="agent.actuator", ts=v.ts, cause=v.id))

    def _device_for(self, action: str, room: str) -> str | None:
        spec = DEVICE_FOR.get(action)
        if not spec:
            return None
        device = spec[0]
        if device == "light":
            lights = self.home.devices_of_kind("light", room)
            return lights[0] if lights else None
        return device

    async def _command(self, action: str, room: str, cause: Event) -> None:
        spec = DEVICE_FOR.get(action)
        device = self._device_for(action, room)
        if not spec or not device:
            return
        data = {"device_id": device, "command": spec[1], "args": dict(spec[2]), "verdict_id": cause.id}
        self.commands.append(data)
        await self.bus.publish(Event(T.COMMAND, data, source="agent.actuator", ts=cause.ts, cause=cause.id))

    async def _on_reply(self, e: Event) -> None:
        self.replies.append(dict(e.data))
        plan = self.open.pop(str(e.data.get("request_id")), None)
        if plan and e.data.get("answer") == "Yes":
            await self._command(plan[0], plan[1] or "living", e)
