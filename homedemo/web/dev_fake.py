"""FakeRuntime: a stand-in for homedemo.runtime so the UI can be built and tested without the sim,
the agents or a model. It drives a real homedemo.bus.Bus with believable event chains (correct
topics, payload shapes, sources and cause ids), a canned world built from home.json, and a few
scenarios written in the real step DSL.

    python -m homedemo.web.dev_fake --port 18750      # UI at http://127.0.0.1:18750
"""
from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from ..bus import Bus
from ..contracts import AGENTS, Event, Settings, T
from ..decider.questions import DeciderQuery, build_scene_query, build_speech_query

HOME_PATH = Path(__file__).resolve().parent.parent / "home.json"

# Fake latencies: the 2B and 12B numbers are the CPU medians from eval/REPORT.md; Bedrock is Claude's
# share of an escalation in the measured tour (eval/BEDROCK.md: 1645 ms, two round trips from us-west-2).
LATENCY_MS = {"decider-2b": 820.0, "decider-12b": 4900.0, "bedrock": 1650.0}
MODEL_NAMES = {"decider-2b": "Strands Decider 2B", "decider-12b": "Strands Decider 12B",
               "bedrock": "Claude Haiku 4.5 on Bedrock"}
ROUTES = {"decider-2b": "local", "decider-12b": "12b", "bedrock": "bedrock"}
ESCALATION_BY = {"bedrock": "bedrock", "12b": "decider-12b"}   # Settings.escalate_to -> ProposalBy
SCHOOL_NIGHTS = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday")

# What Claude says back (one spoken sentence) when Bedrock settles an escalated decision.
BEDROCK_REPLIES = {
    "lights_on": "It's dark in here, so I'm turning the {room} lights on.",
    "lights_off": "Turning the {room} lights off.",
    "lights_dim": "Dimming the {room} lights.",
    "warmer": "Sounds cold. I'll nudge the heating up a little.",
    "cooler": "I'll bring the temperature down a little.",
    "tv_on": "Turning the TV on.",
    "tv_off": "Turning the TV off.",
    "unlock_door": "Unlocking the front door.",
    "lock_door": "Locking the front door.",
    "alert_parent": "Maya is on a chair next to a lit burner. I'm alerting Sam and Alex now.",
    "none": None,
}

# (phrases, action, 2B confidence, 12B confidence). Vague phrasing is where the 2B is unsure.
_SPEECH_GUESSES: list[tuple[tuple[str, ...], str, float, float]] = [
    (("can't see", "cannot see", "so dark", "too dark", "看不见", "好黑"), "lights_on", 0.55, 0.95),
    (("lights on", "light on", "turn on the light", "开灯"), "lights_on", 0.91, 0.97),
    (("lights off", "light off", "turn off the light", "too bright", "关灯"), "lights_off", 0.90, 0.96),
    (("freezing", "cold in here", "warmer", "好冷"), "warmer", 0.78, 0.93),
    (("boiling", "too hot", "cooler", "好热"), "cooler", 0.76, 0.92),
    (("tv off", "turn off the tv", "turn the tv off"), "tv_off", 0.88, 0.95),
    (("watch tv", "tv on", "turn on the tv", "cartoons", "看电视"), "tv_on", 0.88, 0.95),
    (("unlock", "open the door", "let me in", "开门"), "unlock_door", 0.90, 0.96),
]


def load_home() -> dict[str, Any]:
    return json.loads(HOME_PATH.read_text(encoding="utf-8"))


def keyword_baseline(text: str) -> tuple[str, str]:
    """The old-style matcher the demo compares against: (action, reason)."""
    t = text.lower()
    if "light" in t or "灯" in t:
        off = any(w in t for w in ("off", "out", "关"))
        return ("lights_off" if off else "lights_on", "keyword 'light'" + (" + 'off'" if off else ""))
    if "tv" in t or "television" in t or "电视" in t:
        off = "off" in t
        return ("tv_off" if off else "tv_on", "keyword 'tv'" + (" + 'off'" if off else ""))
    if "unlock" in t or "open the door" in t:
        return "unlock_door", "keyword 'unlock'"
    if "lock" in t:
        return "lock_door", "keyword 'lock'"
    if "warm" in t or "heat" in t:
        return "warmer", "keyword 'warm'"
    if "cool" in t:
        return "cooler", "keyword 'cool'"
    return "none", "no keyword matched"


def guess_speech(text: str) -> tuple[str, float, float]:
    t = text.lower()
    for phrases, action, c2, c12 in _SPEECH_GUESSES:
        if any(p in t for p in phrases):
            return action, c2, c12
    return "none", 0.72, 0.90


def spread(options: tuple[str, ...] | list[str], pick: str, conf: float) -> dict[str, float]:
    rest = round((1 - conf) / max(1, len(options) - 1), 4)
    return {o: (round(conf, 4) if o == pick else rest) for o in options}


def fake_answers(query: DeciderQuery, action: str, conf: float) -> dict[str, dict[str, Any]]:
    """Raw per-question answers in the wire shape the Decider server returns, consistent with the
    action/confidence the fake picked: the choice question spreads `conf` over the option labels and
    the yes/no gate says "act" unless the pick is "none"."""
    labels = list(query.option_actions)
    pick = next((lbl for lbl, act in query.option_actions.items() if act == action), labels[-1])
    out: dict[str, dict[str, Any]] = {}
    for key, q in query.questions.items():
        if q.get("type") == "noul":
            p = 0.18 if action == "none" else max(conf, 0.6)
            out[key] = {"type": "noul", "choice": None, "noul": round(p, 4), "confidence": round(abs(2 * p - 1), 4),
                        "probabilities": {"true": round(p, 4), "false": round(1 - p, 4)}}
        elif key == query.primary:
            out[key] = {"type": "choice", "choice": pick, "noul": None, "confidence": round(conf, 4),
                        "probabilities": spread(labels, pick, conf)}
        else:  # the optional room choice: "this room"
            opts = list((q.get("criteria") or {}).keys()) or ["The room the speaker is in"]
            out[key] = {"type": "choice", "choice": opts[0], "noul": None, "confidence": 0.9,
                        "probabilities": spread(opts, opts[0], 0.9)}
    return out


def fake_tokens(state: str, reply: str | None) -> dict[str, int]:
    """Deterministic Bedrock token usage in the measured range (eval/BEDROCK.md: median 3,548 in / 46 out
    per Claude call): the 12 tool schemas and the system prompt dominate the input, two round trips (the
    tool call, then the spoken sentence) make the output."""
    return {"input": 3440 + len(state) // 3, "output": 38 + (len(reply or "") + 3) // 4}


def fake_tool_calls(action: str, room: str, target: str | None) -> list[dict[str, Any]]:
    if action == "none":
        return []
    args: dict[str, Any] = {"room": room}
    if target:
        args["target"] = target
    if action == "warmer":
        args["delta_c"] = 1.5
    elif action == "cooler":
        args["delta_c"] = -1.5
    elif action == "lights_dim":
        args["brightness"] = 30
    elif action == "alert_parent":
        args = {"to": "parents", "room": room, "text": "A child is at the stove with a burner on."}
    return [{"name": action, "input": args}]


def _hhmm(time_s: str) -> int:
    h, m = time_s.split(":")
    return int(h) * 60 + int(m)


class FakeRuntime:
    """Implements contracts.RuntimeAPI. `fast=True` removes every sleep (tests)."""

    def __init__(self, settings: Settings | None = None, *, fast: bool = False) -> None:
        self.home = load_home()
        self.bus = Bus()
        self.settings = settings or Settings()
        self.fast = fast
        self.busy: set[str] = set()
        self._task: asyncio.Task | None = None
        self._open_asks: dict[str, dict[str, Any]] = {}
        self._scenarios = {s["id"]: s for s in build_scenarios()}
        self._load_world()

    # ------------------------------------------------------------------ world state
    def _load_world(self) -> None:
        h = self.home
        self.clock = 0.0
        self._base_min, self._base_clock = _hhmm(h["start"]["time"]), 0.0
        self.weekday = h["start"]["weekday"]
        self.rooms = {r["id"]: {**r, "env": {"lux": 180.0, "celsius": 20.5}} for r in h["rooms"]}
        self.rooms["porch"]["env"] = {"lux": 40.0, "celsius": 12.0}
        self.devices = {d["device_id"]: json.loads(json.dumps(d)) for d in h["devices"]}
        self.doors = {d["door_id"]: dict(d) for d in h["doors"]}
        self.people = {p["id"]: {"id": p["id"], "name": p["name"], "role": p["role"], "room": p["start_room"],
                                 "x": 0.0, "y": 0.0, "activity": "relaxing", "home": True, "phone": p["phone"]}
                       for p in h["residents"]}
        self.pets = {p["id"]: {"id": p["id"], "name": p["name"], "species": p["species"], "room": p["start_room"],
                               "x": 0.0, "y": 0.0, "activity": "napping", "home": True} for p in h["pets"]}
        for who in list(self.people) + list(self.pets):
            self._place(who)
        self._open_asks.clear()

    @property
    def time(self) -> str:
        minutes = self._base_min + int((self.clock - self._base_clock) // 60)
        return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"

    def _set_time(self, hhmm: str) -> None:
        self._base_min, self._base_clock = _hhmm(hhmm), self.clock

    def _advance(self, seconds: float) -> None:
        self.clock += seconds

    def _actor(self, who: str) -> dict[str, Any]:
        if who in self.people:
            return self.people[who]
        if who in self.pets:
            return self.pets[who]
        raise ValueError(f"unknown person or pet {who!r}")

    def _occupants(self, room: str) -> list[dict[str, Any]]:
        return [p for p in self.people.values() if p["home"] and p["room"] == room]

    def _place(self, who: str) -> None:
        """Re-lay out everyone in the actor's room so slots never collide (people first, then pets)."""
        room = self._actor(who)["room"]
        r = self.rooms[room]
        here = [x for x in list(self.people.values()) + list(self.pets.values()) if x["room"] == room and x["home"]]
        for slot, a in enumerate(here):
            a["x"] = round(r["x"] + r["w"] * (0.3 + 0.2 * (slot % 3)), 1)
            a["y"] = round(r["y"] + r["h"] * (0.6 + 0.22 * (slot // 3)), 1)
        for a in (x for x in list(self.people.values()) + list(self.pets.values()) if x["room"] == room and not x["home"]):
            a["x"], a["y"] = round(r["x"] + r["w"] * 0.1, 1), round(r["y"] + r["h"] * 0.85, 1)

    def _is_night(self) -> bool:
        th = self.home["thresholds"]
        now = _hhmm(self.time)
        return now >= _hhmm(th["night_from"]) or now < _hhmm(th["night_to"])

    def _school_night(self) -> bool:
        return self.weekday in SCHOOL_NIGHTS

    def _light_in(self, room: str) -> str | None:
        return next((d for d, v in self.devices.items() if v["kind"] == "light" and v["room"] == room), None)

    def _light_on(self, room: str) -> bool:
        light = self._light_in(room)
        return bool(light and self.devices[light]["state"]["on"])

    def _target_for(self, action: str, room: str) -> str | None:
        if action.startswith("lights"):
            return self._light_in(room)
        if action in ("warmer", "cooler"):
            return "thermostat_main"
        if action.startswith("tv"):
            return "tv_living"
        if action.endswith("_door"):
            return "lock_front"
        return None

    def _relight(self) -> None:
        """Ambient light follows the clock in rooms whose light is off."""
        for r in self.rooms.values():
            if not self._light_on(r["id"]):
                r["env"]["lux"] = (3.0 if r["indoor"] else 1.0) if self._is_night() else (180.0 if r["indoor"] else 400.0)

    # ------------------------------------------------------------------ RuntimeAPI
    def snapshot(self) -> dict[str, Any]:
        return {
            "time": self.time, "weekday": self.weekday, "clock": round(self.clock, 1),
            "rooms": [{"id": r["id"], "name": r["name"], "x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"],
                       "indoor": r["indoor"], "env": dict(r["env"])} for r in self.rooms.values()],
            "devices": [{"device_id": d["device_id"], "kind": d["kind"], "room": d["room"], "x": d["x"],
                         "y": d["y"], "state": dict(d["state"])} for d in self.devices.values()],
            "people": [{k: v for k, v in p.items() if k != "phone"} for p in self.people.values()],
            "pets": [dict(p) for p in self.pets.values()],
            "doors": [{"door_id": d["door_id"], "name": d["name"], "x": d["x"], "y": d["y"],
                       "open": d["open"], "locked": self.devices[d["lock"]]["state"]["locked"]}
                      for d in self.doors.values()],
            "mics": list(self.home["sensors"]["mics"]),
        }

    def agent_graph(self) -> list[dict[str, Any]]:
        return [{**a, "busy": a["name"] in self.busy} for a in AGENTS]

    def scenarios(self) -> list[dict[str, Any]]:
        return [{"id": s["id"], "title": s["title"], "blurb": s["blurb"], "tags": list(s["tags"]),
                 "steps": len(s["steps"])} for s in self._scenarios.values()]

    async def run_scenario(self, scenario_id: str) -> None:
        if scenario_id not in self._scenarios:
            raise KeyError(scenario_id)
        await self.stop_scenario()
        self._task = asyncio.create_task(self._run(self._scenarios[scenario_id]), name=f"scenario:{scenario_id}")

    async def stop_scenario(self) -> None:
        task, self._task = self._task, None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def reset(self) -> None:
        await self.stop_scenario()
        self._load_world()
        self.busy.clear()
        await self._emit(T.CLOCK, self._clock_data(), "sim.clock")
        await self._emit(T.LOG, {"agent": "sim", "level": "info", "text": "World reset to home.json."}, "sim")

    async def step(self, step: dict[str, Any]) -> None:
        do = step.get("do")
        handler = getattr(self, f"_do_{do}", None) if isinstance(do, str) else None
        if handler is None:
            raise ValueError(f"unknown world step {do!r}")
        await handler(step)

    async def phone_reply(self, request_id: str, answer: str) -> None:
        ask = self._open_asks.get(request_id)
        if ask is None:
            raise ValueError(f"no open phone ask {request_id!r}")
        if answer not in ask["data"]["options"]:
            raise ValueError(f"answer must be one of {ask['data']['options']}")
        await self._reply(request_id, answer, by="web")

    async def update_settings(self, **changes: Any) -> Settings:
        for k in changes:
            if not hasattr(self.settings, k):
                raise ValueError(f"unknown setting {k!r}")
        for k, v in changes.items():
            setattr(self.settings, k, v)
        if changes:
            text = ", ".join(f"{k}={v}" for k, v in changes.items())
            await self._emit(T.LOG, {"agent": "runtime", "level": "info", "text": f"Settings changed: {text}"}, "runtime")
        return self.settings

    # ------------------------------------------------------------------ helpers
    async def _emit(self, topic: str, data: dict[str, Any], source: str, cause: Event | None = None) -> Event:
        ev = Event(topic, data, source=source, ts=round(self.clock, 1), cause=cause.id if cause else None)
        await self.bus.publish(ev)
        return ev

    async def _sleep(self, seconds: float) -> None:
        if not self.fast and seconds > 0:
            await asyncio.sleep(seconds / max(0.1, self.settings.speed))

    async def _think(self, agent: str, ms: float) -> None:
        self.busy.add(agent)
        try:
            await self._sleep(ms / 1000)
        finally:
            self.busy.discard(agent)

    def _clock_data(self) -> dict[str, Any]:
        return {"time": self.time, "weekday": self.weekday, "school_night": self._school_night()}

    def _speaker(self, who: str | None) -> dict[str, Any]:
        if who is None or who not in self.people:
            return {"id": None, "name": None, "role": "unknown", "match": 0.0}
        p = self.people[who]
        return {"id": who, "name": p["name"], "role": p["role"], "match": 0.93}

    def _mic_for(self, room: str) -> dict[str, Any]:
        mics = self.home["sensors"]["mics"]
        return next((m for m in mics if m["room"] == room), mics[0])

    def _room_context(self, room: str) -> str:
        r = self.rooms[room]
        tv_s = f" TV {'on' if self.devices['tv_living']['state']['on'] else 'off'}." if room == "living" else ""
        when = "night" if self._is_night() else "daytime"
        return f"{r['name']}. Lights {'on' if self._light_on(room) else 'off'}, lux {r['env']['lux']:.0f}, {when} {self.time}.{tv_s}"

    def _since_beat_start(self) -> list[Event]:
        """Events since the newest scenario note/start (a 'beat'), newest first."""
        out: list[Event] = []
        for e in reversed(self.bus.history):
            if e.topic == T.SCENARIO and e.data.get("state") in ("step", "started"):
                break
            out.append(e)
        return out

    # ------------------------------------------------------------------ sensors + perception
    async def _sense_room(self, room: str, trigger: str, cause: Event | None = None, hazard: bool = False) -> None:
        r = self.rooms[room]
        occupants = self._occupants(room)
        trigger = "hazard" if hazard else trigger
        ev = await self._emit(T.MOTION, {"room": room, "motion": bool(occupants)}, "sim.motion", cause)
        await self._emit(T.LUX, {"room": room, "lux": r["env"]["lux"]}, "sim.lux", cause)
        await self._emit(T.TEMP, {"room": room, "celsius": r["env"]["celsius"]}, "sim.temp", cause)
        cam = next((c for c in self.home["sensors"]["cameras"] if c["room"] == room), None)
        if cam:
            dets = []
            for p in occupants:
                attrs: dict[str, Any] = {"age_group": "child" if p["role"] == "child" else "adult",
                                         "identity": {"id": p["id"], "match": 0.9}}
                if hazard and p["role"] == "child":
                    attrs.update({"pose": "standing on a chair", "near": "stove"})
                dets.append({"label": "person", "conf": 0.94, "track_id": f"t_{p['id']}", "attrs": attrs})
            for pet in self.pets.values():
                if pet["room"] == room:
                    dets.append({"label": pet["species"], "conf": 0.88, "track_id": f"t_{pet['id']}", "attrs": {}})
            ev = await self._emit(T.CAMERA, {"room": room, "camera_id": cam["camera_id"], "detections": dets,
                                             "activity": occupants[0]["activity"] if occupants else None},
                                  "sim.camera", cause)
        await self._perceive(room, trigger, ev)

    async def _perceive(self, room: str, trigger: str, cause: Event) -> None:
        r = self.rooms[room]
        occupants = self._occupants(room)
        people = ", ".join(f"{p['name']} ({p['role']}) {p['activity']}" for p in occupants) or "nobody"
        text = (f"Time {self.time} {self.weekday}, {'night' if self._is_night() else 'day'}. {r['name']}: {people}. "
                f"Lights {'on' if self._light_on(room) else 'off'}, lux {r['env']['lux']:.0f}. {r['env']['celsius']:.0f} °C.")
        if trigger == "hazard":
            text += " The stove burner is on and the child is standing on a chair right next to it."
        facts = {"time": self.time, "weekday": self.weekday,
                 "people": [{"id": p["id"], "role": p["role"], "activity": p["activity"]} for p in occupants],
                 "pets": [p["id"] for p in self.pets.values() if p["room"] == room],
                 "lux": r["env"]["lux"], "lights_on": self._light_on(room),
                 "tv_on": self.devices["tv_living"]["state"]["on"], "celsius": r["env"]["celsius"],
                 "setpoint_c": self.devices["thermostat_main"]["state"]["setpoint_c"],
                 "stove_on": self.devices["stove_kitchen"]["state"]["burner_on"], "idle_min": 0}
        self.busy.add("perception")
        scene = await self._emit(T.SCENE, {"room": room, "trigger": trigger, "text": text, "facts": facts},
                                 "agent.perception", cause)
        self.busy.discard("perception")
        await self._reflex(scene)
        if trigger in ("person_entered", "hazard", "activity_changed") and occupants:
            await self._scene_decider(scene)

    async def _reflex(self, scene: Event) -> None:
        f, room = scene.data["facts"], scene.data["room"]
        if f["people"] and f["lux"] < self.home["thresholds"]["dark_lux"] and not f["lights_on"] and self._is_night():
            prop = await self._emit(T.PROPOSAL, {
                "by": "reflex", "mode": "act", "room": room, "action": "lights_on", "target": self._light_in(room),
                "confidence": 1.0, "probabilities": {"lights_on": 1.0}, "question": "", "state_text": "",
                "latency_ms": 0.4, "reason": "Dark room with a person in it at night.", "speaker": None,
                "kind": "scene", "escalated_from": None, "route": "rule", "model": "plain rule",
                "threshold": self.settings.act_threshold, "tokens": None}, "agent.reflex", scene)
            await self._policy(prop, voice_source="person")

    async def _scene_decider(self, scene: Event) -> None:
        hazard = scene.data["trigger"] == "hazard"
        action, c2, c_up = ("alert_parent", 0.74, 0.97) if hazard else ("none", 0.86, 0.94)
        query = build_scene_query(scene.data)
        await self._decide(scene, query, action, c2, c_up, agent="scene_decider", room=scene.data["room"],
                           speaker=None, voice_source="person")

    # ------------------------------------------------------------------ voice + intent
    async def _hear(self, room: str, text: str, who: str | None, source: str, lang: str = "en") -> None:
        mic = self._mic_for(room)
        azimuth = {"person": 35.0, "tv": 250.0, "outside": 180.0}.get(source, 0.0)
        audio = await self._emit(T.AUDIO, {
            "room": room, "mic_id": mic["mic_id"], "text": text, "lang": lang, "asr_conf": 0.93, "wake_word": False,
            "speaker": self._speaker(who), "doa": {"azimuth_deg": azimuth, "source": source, "conf": 0.9}}, "sim.mic")
        await self._think("voice", 120)
        utt = await self._emit(T.UTTERANCE, {
            "room": room, "text": text, "lang": lang, "speaker": self._speaker(who), "source": source,
            "context": self._room_context(room), "wake_word": False}, "agent.voice", audio)
        if self.settings.keyword_shadow:
            kw_action, kw_reason = keyword_baseline(text)
            await self._emit(T.PROPOSAL, {
                "by": "keyword", "mode": "shadow", "room": room, "action": kw_action,
                "target": self._target_for(kw_action, room), "confidence": 1.0 if kw_action != "none" else 0.0,
                "probabilities": {}, "question": "", "state_text": "", "latency_ms": 0.1, "reason": kw_reason,
                "speaker": self._speaker(who), "kind": "speech", "escalated_from": None,
                "route": "rule", "model": "keyword matcher", "tokens": None}, "agent.keyword", utt)
        action, c2, c_up = guess_speech(text)
        query = build_speech_query(utt.data)
        await self._decide(utt, query, action, c2, c_up, agent="intent_decider", room=room,
                           speaker=self._speaker(who), voice_source=source)

    async def _decide(self, cause: Event, query: DeciderQuery, action: str, c2: float, c_up: float, *,
                      agent: str, room: str, speaker: dict[str, Any] | None, voice_source: str) -> None:
        """The cascade the real DeciderAgent runs, on canned confidences: the 2B answers first; when it
        is at least act_threshold sure the house acts locally (route "local", no tokens). Otherwise the
        2B answer is published as a shadow and Settings.escalate_to (Claude on Bedrock, or the 12B)
        decides, carrying escalated_from. Modes "2b" / "12b" / "fake" ask one model only."""
        mode = self.settings.decider_mode
        threshold = self.settings.act_threshold
        escalation = ESCALATION_BY[self.settings.escalate_to]
        order = {"2b": ["decider-2b"], "12b": ["decider-12b"], "fake": ["decider-2b"],
                 "cascade": ["decider-2b", escalation]}[mode]
        options = list(dict.fromkeys(query.option_actions.values()))
        prev: Event | None = None
        for by in order:
            conf = c2 if by == "decider-2b" else c_up
            latency = 6.0 if mode == "fake" else LATENCY_MS[by]
            escalate = mode == "cascade" and by == "decider-2b" and conf < threshold
            await self._think(agent, latency)
            label = {"decider-2b": "2B", "decider-12b": "12B", "bedrock": "Claude on Bedrock"}[by]
            reason = (f"unsure ({conf:.0%} < {threshold:.0%}), asking {MODEL_NAMES[escalation]}" if escalate
                      else f"{label} is {conf:.0%} sure: {action.replace('_', ' ')}")
            data: dict[str, Any] = {
                "by": by, "mode": "shadow" if escalate else "act", "room": room, "action": action,
                "target": self._target_for(action, room), "confidence": round(conf, 4),
                "probabilities": spread(options, action, conf),
                "question": str(query.questions[query.primary].get("instructions", query.headline)),
                "state_text": query.state, "latency_ms": latency, "reason": reason, "speaker": speaker,
                "kind": query.kind, "escalated_from": prev.id if prev else None,
                "route": ROUTES[by], "model": MODEL_NAMES[by],
                "request": {"state": query.state, "questions": query.questions},
                "answers": fake_answers(query, action, conf), "threshold": threshold, "tokens": None,
            }
            if by == "bedrock":
                reply = BEDROCK_REPLIES.get(action)
                reply = reply.format(room=self.rooms[room]["name"].lower()) if reply else None
                data.update({"probabilities": {action: 1.0}, "confidence": 1.0, "answers": {},
                             "tokens": fake_tokens(query.state, reply),
                             "tool_calls": fake_tool_calls(action, room, data["target"]), "reply": reply})
            prop = await self._emit(T.PROPOSAL, data, f"agent.{agent}", cause)
            if escalate:
                prev = prop
                continue
            await self._policy(prop, voice_source)
            return

    # ------------------------------------------------------------------ policy + actuator
    async def _policy(self, prop: Event, voice_source: str) -> None:
        d = prop.data
        action, who, room = d["action"], d.get("speaker"), d["room"]
        role = (who or {}).get("role", "unknown")
        match = (who or {}).get("match", 0.0)
        th = self.home["thresholds"]
        verdict, rule = "allow", "confidence above the act band"
        if action == "none":
            verdict, rule = "ignore", "nothing to do"
        elif d["kind"] == "speech" and voice_source in ("tv", "outside"):
            verdict, rule = "deny", f"inside_voices_only: the voice came from {'the TV' if voice_source == 'tv' else 'outside'}"
        elif action == "alert_parent":
            verdict, rule = "allow", "alerts_always"
        elif action == "unlock_door" and role == "child":
            verdict, rule = "deny", "kids_no_unlock"
        elif action == "unlock_door" and (role != "parent" or match < th["unlock_voice_match"]):
            verdict, rule = "deny", "unlock_needs_parent (parent voice, match >= 0.90)"
        elif action == "tv_on" and role == "child" and self._school_night() and _hhmm(self.time) >= 20 * 60:
            verdict, rule = "ask_parent", "kids_tv_curfew: school night after 20:00"
        elif action == "lock_door" and role not in ("parent", "child", "guest"):
            verdict, rule = "deny", "unknown voice may not touch doors"
        elif d["by"] == "reflex":
            rule = "reflex rules act on their own; anyone_lights"
        elif d["confidence"] < self.settings.ask_threshold:
            verdict, rule = "ignore", "confidence below the ask band"
        elif d["confidence"] < self.settings.act_threshold:
            verdict, rule = "ask_speaker", "unsure_asks: confidence in the ask band"
        elif action.startswith("lights"):
            rule = "anyone_lights"
        elif action in ("warmer", "cooler"):
            rule = "residents and guests may change the heat"
        self.busy.add("policy")
        ver = await self._emit(T.VERDICT, {"proposal_id": prop.id, "action": action, "target": d.get("target"),
                                           "room": room, "verdict": verdict, "rule": rule, "who": who},
                               "agent.policy", prop)
        self.busy.discard("policy")
        await self._actuate(ver, prop)

    async def _actuate(self, ver: Event, prop: Event) -> None:
        d = ver.data
        if d["verdict"] == "allow":
            await self._command(d["action"], d.get("target"), d["room"], ver)
        elif d["verdict"] in ("ask_parent", "ask_speaker"):
            who = (d.get("who") or {}).get("id")
            to = "parents" if d["verdict"] == "ask_parent" or not who or not self.people[who]["phone"] else who
            name = (d.get("who") or {}).get("name") or "Someone"
            text = (f"{name} wants to turn on the TV ({self.weekday} {self.time}, a school night). Allow?"
                    if d["verdict"] == "ask_parent" else
                    f"Did you want: {d['action'].replace('_', ' ')} in the {self.rooms[d['room']]['name'].lower()}?")
            rid = uuid.uuid4().hex[:8]
            ask = await self._emit(T.ASK, {"request_id": rid, "to": to, "text": text, "options": ["Yes", "No"],
                                           "proposal_id": prop.id, "action": d["action"], "target": d.get("target"),
                                           "room": d["room"]}, "agent.actuator", ver)
            self._open_asks[rid] = {"data": ask.data, "event": ask}
        elif d["verdict"] == "deny":
            await self._emit(T.LOG, {"agent": "actuator", "level": "warn",
                                     "text": f"Denied: {d['action'].replace('_', ' ')} ({d['rule']})."},
                             "agent.actuator", ver)

    async def _reply(self, request_id: str, answer: str, by: str) -> None:
        ask = self._open_asks.pop(request_id)
        rep = await self._emit(T.REPLY, {"request_id": request_id, "answer": answer, "by": by}, "phone", ask["event"])
        if answer == "Yes":
            await self._command(ask["data"]["action"], ask["data"]["target"], ask["data"]["room"], rep)
        else:
            await self._emit(T.LOG, {"agent": "actuator", "level": "info",
                                     "text": f"{by} said {answer}: {ask['data']['action'].replace('_', ' ')} not done."},
                             "agent.actuator", rep)

    async def _command(self, action: str, target: str | None, room: str, cause: Event) -> None:
        if action == "alert_parent":
            text = f"Safety alert from the {self.rooms[room]['name'].lower()}: a child is at the stove with a burner on."
            await self._emit(T.COMMAND, {"device_id": "phone", "command": "notify",
                                         "args": {"to": "parents", "text": text}, "verdict_id": cause.id},
                             "agent.actuator", cause)
            return
        if not target:
            return
        command, args = {
            "lights_on": ("on", {"brightness": 80}), "lights_off": ("off", {}), "lights_dim": ("dim", {"brightness": 30}),
            "warmer": ("set_setpoint", {"delta_c": 1.5}), "cooler": ("set_setpoint", {"delta_c": -1.5}),
            "tv_on": ("on", {}), "tv_off": ("off", {}), "unlock_door": ("unlock", {}), "lock_door": ("lock", {}),
        }[action]
        cmd = await self._emit(T.COMMAND, {"device_id": target, "command": command, "args": args,
                                           "verdict_id": cause.id}, "agent.actuator", cause)
        await self._apply(cmd)

    async def _apply(self, cmd: Event) -> None:
        dev = self.devices[cmd.data["device_id"]]
        st, c, args = dev["state"], cmd.data["command"], cmd.data["args"]
        if dev["kind"] == "light":
            st["on"] = c in ("on", "dim")
            st["brightness"] = args.get("brightness", 0) if st["on"] else 0
            self.rooms[dev["room"]]["env"]["lux"] = 320.0 if st["on"] else min(self.rooms[dev["room"]]["env"]["lux"], 8.0)
        elif dev["kind"] == "tv":
            st["on"] = c == "on"
            st["playing"] = (st.get("playing") or "Evening news") if st["on"] else None
        elif dev["kind"] == "thermostat":
            st["setpoint_c"] = round(st["setpoint_c"] + args.get("delta_c", 0.0), 1)
        elif dev["kind"] == "lock":
            st["locked"] = c == "lock"
        elif dev["kind"] == "stove":
            st["burner_on"] = c == "on"
        applied = await self._emit(T.APPLIED, {"device_id": dev["device_id"], "state": dict(st), "command_id": cmd.id},
                                   "sim", cmd)
        await self._emit(T.DEVICE, {"device_id": dev["device_id"], "kind": dev["kind"], "room": dev["room"],
                                    "state": dict(st)}, "sim.device", applied)

    # ------------------------------------------------------------------ world steps (scenario DSL)
    async def _do_set_time(self, s: dict[str, Any]) -> None:
        self._set_time(s["time"])
        self.weekday = s.get("weekday", self.weekday)
        self._relight()
        await self._emit(T.CLOCK, self._clock_data(), "sim.clock")

    async def _do_move(self, s: dict[str, Any]) -> None:
        a = self._actor(s["who"])
        if s.get("to") not in self.rooms:
            raise ValueError(f"unknown room {s.get('to')!r}")
        prev = a["room"]
        a["room"], a["home"] = s["to"], True
        a["activity"] = s.get("activity", "walking in")
        self._place(s["who"])
        if prev != s["to"]:
            await self._sense_room(prev, "idle")
        await self._sense_room(s["to"], "person_entered", hazard=bool(s.get("hazard")))

    async def _do_activity(self, s: dict[str, Any]) -> None:
        a = self._actor(s["who"])
        a["activity"] = s["text"]
        attrs = s.get("attrs") or {}
        await self._sense_room(a["room"], "activity_changed", hazard=attrs.get("near") == "stove")

    async def _do_idle(self, s: dict[str, Any]) -> None:
        a = self._actor(s["who"])
        a["activity"] = "sitting still"
        self._advance(60 * float(s.get("minutes", 1)))
        self._relight()
        await self._emit(T.CLOCK, self._clock_data(), "sim.clock")
        await self._sense_room(a["room"], "idle")

    async def _do_say(self, s: dict[str, Any]) -> None:
        p = self.people.get(s.get("who", ""))
        if p is None or not p["home"]:
            raise ValueError(f"{s.get('who')!r} is not a person at home")
        if not str(s.get("text", "")).strip():
            raise ValueError("nothing to say")
        await self._hear(p["room"], s["text"], s["who"], "person", s.get("lang", "en"))

    async def _do_tv_say(self, s: dict[str, Any]) -> None:
        tv = self.devices.get(s.get("device", "tv_living"))
        if tv is None or tv["kind"] != "tv":
            raise ValueError("tv_say needs a tv device")
        if not str(s.get("text", "")).strip():
            raise ValueError("nothing to say")
        tv["state"]["on"], tv["state"]["playing"] = True, tv["state"].get("playing") or "Home-makeover show"
        await self._emit(T.DEVICE, {"device_id": tv["device_id"], "kind": "tv", "room": tv["room"],
                                    "state": dict(tv["state"])}, "sim.device")
        await self._hear(tv["room"], s["text"], None, "tv", s.get("lang", "en"))

    async def _do_outside_say(self, s: dict[str, Any]) -> None:
        if s.get("door", "front") not in self.doors:
            raise ValueError("unknown door")
        if not str(s.get("text", "")).strip():
            raise ValueError("nothing to say")
        await self._hear("porch", s["text"], None, "outside", s.get("lang", "en"))

    async def _do_set_env(self, s: dict[str, Any]) -> None:
        r = self.rooms.get(s.get("room", ""))
        if r is None:
            raise ValueError(f"unknown room {s.get('room')!r}")
        if "lux" in s:
            r["env"]["lux"] = float(s["lux"])
        if "celsius" in s:
            r["env"]["celsius"] = float(s["celsius"])
        await self._sense_room(s["room"], "env_changed")

    async def _do_set_device(self, s: dict[str, Any]) -> None:
        dev = self.devices.get(s.get("device", ""))
        if dev is None:
            raise ValueError(f"unknown device {s.get('device')!r}")
        dev["state"].update(s.get("state") or {})
        if dev["kind"] == "light":
            self.rooms[dev["room"]]["env"]["lux"] = 320.0 if dev["state"]["on"] else self.rooms[dev["room"]]["env"]["lux"]
        await self._emit(T.DEVICE, {"device_id": dev["device_id"], "kind": dev["kind"], "room": dev["room"],
                                    "state": dict(dev["state"])}, "sim.device")

    async def _do_phone(self, s: dict[str, Any]) -> None:
        self._actor(s["who"])
        await self._emit(T.PHONE, {"person_id": s["who"], "home": bool(s.get("home", True))}, "sim.phone")

    async def _do_leave(self, s: dict[str, Any]) -> None:
        a = self._actor(s["who"])
        room = a["room"]
        a["home"], a["room"], a["activity"] = False, "porch", "out"
        self._place(s["who"])
        await self._emit(T.PHONE, {"person_id": s["who"], "home": False}, "sim.phone")
        await self._sense_room(room, "idle")

    async def _do_arrive(self, s: dict[str, Any]) -> None:
        who = s["who"]
        if who not in self.people:
            self.people[who] = {"id": who, "name": s.get("name", who.title()), "role": s.get("role", "guest"),
                                "room": "porch", "x": 0.0, "y": 0.0, "activity": "arriving", "home": True, "phone": False}
        await self._do_move({"do": "move", "who": who, "to": s.get("to", "hallway"), "activity": "arriving"})

    # ------------------------------------------------------------------ scenario runner (fake)
    async def _run(self, sc: dict[str, Any]) -> None:
        sid, passed, failed = sc["id"], 0, 0

        async def status(state: str, i: int, text: str) -> None:
            await self._emit(T.SCENARIO, {"scenario_id": sid, "state": state, "step_index": i, "text": text,
                                          "passed": passed, "failed": failed}, "scenario")

        await status("started", 0, sc["title"])
        try:
            for i, st in enumerate(sc["steps"]):
                do = st["do"]
                if do == "note":
                    await status("step", i, st["text"])
                elif do == "wait":
                    self._advance(float(st["seconds"]))
                    await self._sleep(float(st["seconds"]))
                elif do == "reply":
                    await self._auto_reply(st)
                elif do == "expect":
                    ok = await self._expect(st)
                    passed, failed = passed + ok, failed + (not ok)
                    await status("expect_pass" if ok else "expect_fail", i, st.get("text", json.dumps(st)))
                else:
                    await self.step(st)
                if st.get("wait"):
                    self._advance(float(st["wait"]))
                    await self._sleep(float(st["wait"]))
            await self._settle()
            await status("finished", len(sc["steps"]), f"{passed} passed, {failed} failed")
        except asyncio.CancelledError:
            await status("stopped", 0, "stopped")
            raise
        except Exception as exc:  # the UI shows errors instead of a silent stop
            await status("error", 0, f"{type(exc).__name__}: {exc}")

    async def _settle(self) -> None:
        try:
            await self.bus.drain(timeout=5)
        except TimeoutError:
            pass

    async def _auto_reply(self, st: dict[str, Any]) -> None:
        """Give a human on the phone mock `within` seconds, then answer for them."""
        budget = 0.0 if self.fast else float(st.get("within", 0)) / max(0.1, self.settings.speed)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + budget
        while self._open_asks and loop.time() < deadline:
            await asyncio.sleep(0.2)
        for rid in list(self._open_asks):
            await self._reply(rid, st.get("answer", "Yes"), by="scenario")

    async def _expect(self, st: dict[str, Any]) -> bool:
        budget = 0.0 if self.fast else float(st.get("within", 0)) / max(0.1, self.settings.speed)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + budget
        while True:
            await self._settle()
            if self._check(st):
                return True
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(0.1)

    def _check(self, st: dict[str, Any]) -> bool:
        recent = self._since_beat_start()
        what = st.get("what")
        if what == "device":
            dev = self.devices.get(st["device"])
            return bool(dev) and all(dev["state"].get(k) == v for k, v in st["state"].items())
        if what == "no_command":
            return not any(e.topic == T.COMMAND and e.data.get("device_id") == st.get("device") for e in recent)
        if what == "ask":
            return any(e.topic == T.ASK and e.data.get("to") == st.get("to") for e in recent)
        if what == "verdict":
            return any(e.topic == T.VERDICT and e.data.get("verdict") == st.get("verdict") for e in recent)
        if what == "proposal":
            return any(e.topic == T.PROPOSAL and e.data.get("action") == st.get("action")
                       and (st.get("by") in (None, e.data.get("by"))) for e in recent)
        return False


# --------------------------------------------------------------------------- canned scenarios
def _beat_dark_kitchen() -> list[dict[str, Any]]:
    return [
        {"do": "note", "text": "It is 21:30. Sam walks from the living room into the dark kitchen. A plain reflex rule turns the light on; no model is asked."},
        {"do": "move", "who": "sam", "to": "living", "activity": "chatting"},
        {"do": "set_time", "time": "21:30", "weekday": "Tuesday", "wait": 1},
        {"do": "move", "who": "sam", "to": "kitchen", "activity": "looking for a snack", "wait": 2},
        {"do": "expect", "what": "device", "device": "light_kitchen", "state": {"on": True}, "within": 6,
         "text": "kitchen light is on"},
    ]


def _beat_cant_see() -> list[dict[str, Any]]:
    return [
        {"do": "note", "text": "Alex, in the dark living room: \"I can't see a thing in here.\" No wake word, no device named. The keyword baseline hears no command; the 2B on the hub is unsure (0.55) and hands it to Claude on Bedrock inside the same Strands agent."},
        {"do": "say", "who": "alex", "text": "I can't see a thing in here.", "wait": 1},
        {"do": "expect", "what": "device", "device": "light_living", "state": {"on": True}, "within": 12,
         "text": "living room light is on"},
        {"do": "expect", "what": "proposal", "by": "keyword", "action": "none", "text": "keyword baseline proposed nothing"},
    ]


def _beat_tv_voice() -> list[dict[str, Any]]:
    return [
        {"do": "note", "text": "The TV shouts \"Turn off the lights!\". The mic array points at the TV, the voice agent tags it as a TV voice, and policy denies it. The keyword baseline would have obeyed."},
        {"do": "tv_say", "text": "Turn off the lights!", "device": "tv_living", "wait": 1},
        {"do": "expect", "what": "verdict", "verdict": "deny", "within": 8, "text": "TV voice denied"},
        {"do": "expect", "what": "device", "device": "light_living", "state": {"on": True}, "text": "living room light still on"},
    ]


def _beat_stove() -> list[dict[str, Any]]:
    return [
        {"do": "note", "text": "Maya (7) climbs on a chair next to the stove while a burner is on. The scene decider asks for a parent alert; safety alerts are always allowed."},
        {"do": "set_device", "device": "stove_kitchen", "state": {"burner_on": True}, "wait": 1},
        {"do": "move", "who": "maya", "to": "kitchen", "activity": "standing on a chair, reaching over the stove", "hazard": True, "wait": 1},
        {"do": "expect", "what": "proposal", "action": "alert_parent", "within": 12, "text": "parent alert proposed"},
        {"do": "set_device", "device": "stove_kitchen", "state": {"burner_on": False}, "wait": 1},
    ]


def _beat_curfew() -> list[dict[str, Any]]:
    return [
        {"do": "note", "text": "22:05 on a school night. Maya: \"Can we watch TV?\" The decider is sure she wants the TV on; policy says a parent must agree first, so the phone asks. Answer on the phone, or wait 15 s."},
        {"do": "set_time", "time": "22:05", "weekday": "Tuesday", "wait": 1},
        {"do": "move", "who": "maya", "to": "living", "activity": "looking at the TV", "wait": 2},
        {"do": "say", "who": "maya", "text": "Can we watch TV?", "wait": 1},
        {"do": "expect", "what": "ask", "to": "parents", "within": 8, "text": "parents were asked"},
        {"do": "reply", "answer": "Yes", "within": 15},
        {"do": "expect", "what": "device", "device": "tv_living", "state": {"on": True}, "within": 3,
         "text": "TV is on after a parent said yes"},
    ]


def build_scenarios() -> list[dict[str, Any]]:
    tour = (_beat_dark_kitchen() + _beat_cant_see() + _beat_tv_voice() + _beat_stove()
            + [{"do": "set_device", "device": "tv_living", "state": {"on": False, "playing": None}}] + _beat_curfew())
    return [
        {"id": "fake_tour", "title": "Fake tour: five beats", "tags": ["hero", "fake"],
         "blurb": "Reflex, 2B on the hub then Claude on Bedrock when unsure, a TV voice denied, a child at the stove, a curfew ask. Canned events, no model.",
         "steps": tour},
        {"id": "fake_dark_kitchen", "title": "Dark kitchen reflex", "tags": ["fake", "beat"],
         "blurb": "Sam enters the dark kitchen at night; a rule turns the light on.", "steps": _beat_dark_kitchen()},
        {"id": "fake_cant_see", "title": "\"I can't see a thing\"", "tags": ["fake", "beat"],
         "blurb": "No wake word, no device named: 2B unsure, Bedrock decides, lights on.",
         "steps": [{"do": "set_time", "time": "21:30", "weekday": "Tuesday"}] + _beat_cant_see()},
        {"id": "fake_tv_voice", "title": "TV voice denied", "tags": ["fake", "beat"],
         "blurb": "The TV shouts a command; the mic direction says TV; policy denies.",
         "steps": [{"do": "set_time", "time": "21:30", "weekday": "Tuesday"},
                   {"do": "set_device", "device": "light_living", "state": {"on": True, "brightness": 80}}] + _beat_tv_voice()},
        {"id": "fake_stove", "title": "Child at the stove", "tags": ["fake", "beat"],
         "blurb": "Hazard scene -> scene decider -> parent alert (always allowed).", "steps": _beat_stove()},
        {"id": "fake_curfew", "title": "Curfew ask", "tags": ["fake", "beat"],
         "blurb": "Maya wants TV on a school night; the phone asks a parent.", "steps": _beat_curfew()},
    ]


def main(argv: list[str] | None = None) -> None:
    from .server import serve

    ap = argparse.ArgumentParser(description="Serve the demo UI on a FakeRuntime (loopback only).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18750)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args(argv)
    serve(FakeRuntime(Settings(speed=args.speed)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
