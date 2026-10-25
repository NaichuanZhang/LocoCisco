"""Decider question library: how the house phrases its questions to the Strands Decider.

OWNED BY THE "questions" WORKSTREAM. Agents call ONLY build_speech_query / build_scene_query and
DeciderQuery.interpret; everything else here may change freely while tuning.

Versions (eval/REPORT.md has the measurements):
  v0  the wording measured on 2026-10-20: one choice question, short option descriptions.
  v1  full-sentence option descriptions, a short "Not a request" / "Do nothing" option, a yes/no
      "does the speaker want ... changed right now?" gate answered in the same request and folded
      into the action probabilities, a "which room?" choice for speech, and the lock / lights-off /
      cooler actions v0 could not express. The state keeps v0's one-line layout.
Both are selectable: build_speech_query(utt, version="v0"). Any Spec field can be overridden by
keyword for experiments (build_speech_query(utt, gate=False)); callers use the positional form.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..contracts import SceneData, Speaker, UtteranceData
from ..decider_client import DeciderAnswer

VERSION = "v1"

_HOME = json.loads((Path(__file__).resolve().parents[1] / "home.json").read_text(encoding="utf-8"))
ROOM_NAMES: dict[str, str] = {r["id"]: r["name"] for r in _HOME["rooms"]}
_RESIDENTS: dict[str, dict[str, Any]] = {r["id"]: r for r in _HOME["residents"]}
THIS_ROOM = "The room the speaker is in"
LIGHT_ACTIONS = ("lights_on", "lights_off", "lights_dim")


@dataclass(frozen=True)
class Spec:
    """Everything that varies between question versions (see VARIANTS)."""
    descriptions: str = "full"      # "short" = v0 wording | "full" = one sentence with the cues
    gate: bool = True               # add a yes/no "does the house need to act now?" question
    none_option: bool = True        # keep a "nothing" option inside the action choice too
    room: bool | str = True         # speech only: add a "which room?" choice; "auto" = only if a room is named
    layout: str = "line"            # "line" = v0 one-liner | "labeled" = one labeled line per fact
    facts: str | None = None        # house facts: None | "state" | "instructions"
    none_first: bool = False        # put the "nothing" option first instead of last
    split_dim: bool = False         # speech: separate "Lights dimmer" option (else merged into off)
    extra_actions: bool = True      # lock_door (speech); lights_off + cooler (scene)
    gate_text: str = "want"         # which gate wording (keys of SPEECH_GATES / SCENE_GATES)
    none_text: str = "sharp"        # "v0" | "sharp" | "long": wording of the "nothing" option
    combine: str = "product"        # how the gate joins the choice: "product" | "mixture"
    extra_gates: str = ""           # experiments: more gate wordings asked too (comma-separated keys),
                                    # recorded in the answers but ignored by interpret


VARIANTS: dict[str, Spec] = {
    "v0": Spec(descriptions="short", gate=False, room=False, layout="line", extra_actions=False, none_text="v0"),
    "v1": Spec(descriptions="full", layout="line", none_text="sharp", gate=True, gate_text="want", room="auto",
               extra_actions=True, combine="product"),
}


@dataclass
class Interpretation:
    action: str                       # one of contracts.ACTIONS
    confidence: float                 # 0..1, what the policy bands compare against
    probabilities: dict[str, float]   # by ACTION name (UI bars), sums to 1
    detail: dict[str, Any] = field(default_factory=dict)  # raw per-question answers, for the trace
    room: str | None = None           # target room id (home.json); None = the speaker's own room


@dataclass
class DeciderQuery:
    kind: str                         # "speech" | "scene"
    state: str                        # exactly what the decider reads
    questions: dict[str, dict[str, Any]]
    primary: str                      # key of the question whose options map to actions
    option_actions: dict[str, str]    # option label -> action
    headline: str = ""                # short question text for the UI
    version: str = VERSION
    gate: str | None = None           # key of the yes/no "act now?" question, if asked
    room_key: str | None = None       # key of the room choice, if asked
    option_rooms: dict[str, str | None] = field(default_factory=dict)  # room option -> room id
    speaker_room: str | None = None   # where the speaker is (naming it explicitly still means None)
    combine: str = "product"          # see interpret

    def interpret(self, answers: dict[str, DeciderAnswer]) -> Interpretation:
        a = answers[self.primary]
        probs: dict[str, float] = {act: 0.0 for act in dict.fromkeys(self.option_actions.values())}
        for opt, p in a.probabilities.items():
            probs[self.option_actions[opt]] += p
        probs.setdefault("none", 0.0)
        gate = answers.get(self.gate) if self.gate else None
        if gate is not None and gate.noul is not None:
            p_req = gate.noul
            if self.combine == "product":
                # Product of experts: P(a) ∝ choice(a)·P(req), P(none) ∝ choice(none)·(1-P(req)).
                probs = {act: p * (p_req if act != "none" else 1 - p_req) for act, p in probs.items()}
            else:
                # Mixture: the gate owns "nothing"; the choice only ranks the actions.
                # P(a) = P(req)·choice(a | an action), P(none) = 1 - P(req).
                acts = {act: p for act, p in probs.items() if act != "none"}
                z = sum(acts.values()) or 1.0
                probs = {act: p_req * p / z for act, p in acts.items()}
                probs["none"] = 1 - p_req
        total = sum(probs.values()) or 1.0
        probs = {act: p / total for act, p in probs.items()}
        action = max(probs, key=probs.get)
        detail = {k: {"choice": v.choice, "noul": v.noul, "confidence": round(v.confidence, 3)}
                  for k, v in answers.items()}
        room = None
        if self.room_key and self.room_key in answers and action in LIGHT_ACTIONS:
            r = answers[self.room_key]
            room = self.option_rooms.get(r.choice or THIS_ROOM)
            if room == self.speaker_room:
                room = None
            detail[self.room_key]["room"] = room
        return Interpretation(action, probs[action], probs, detail, room)


# --------------------------------------------------------------------------- wording
HOUSE_FACTS = ("The house can switch and dim the lights in any room, change the thermostat, switch the "
               "living room TV, and lock or unlock the front door. It has no wake word: it listens to "
               "normal talk and should act only on what someone wants done right now.")

SPEECH_OPTIONS_V0: dict[str, tuple[str, str]] = {
    "Lights on": ("Turn the lights on.", "lights_on"),
    "Lights off or dimmer": ("Turn the lights off or dim them.", "lights_off"),
    "Warmer": ("Raise the thermostat.", "warmer"),
    "Cooler": ("Lower the thermostat.", "cooler"),
    "TV on": ("Turn the TV on.", "tv_on"),
    "TV off": ("Turn the TV off.", "tv_off"),
    "Unlock the door": ("Unlock the front door.", "unlock_door"),
    "No action": ("This is not a request for the house to do anything; leave everything as it is.", "none"),
}

SPEECH_OPTIONS_FULL: dict[str, tuple[str, str]] = {
    "Lights on": ("The speaker wants the lights turned on, or says it is too dark to see.", "lights_on"),
    "Lights off or dimmer": ("The speaker wants the lights off or dimmer: going to sleep, lights left on "
                             "by mistake, or too bright for their eyes.", "lights_off"),
    "Warmer": ("The speaker is cold or wants the heating turned up.", "warmer"),
    "Cooler": ("The speaker is too hot or wants the room cooler.", "cooler"),
    "TV on": ("The speaker wants the TV on or wants to watch something now.", "tv_on"),
    "TV off": ("The speaker wants the TV off, or the noise from it to stop.", "tv_off"),
    "Unlock the door": ("The speaker wants the front door unlocked or opened for them.", "unlock_door"),
    "Lock the door": ("The speaker wants the front door locked.", "lock_door"),
}
_DIM_SPLIT = {
    "Lights off": ("The speaker wants the lights turned off: going to sleep or lights left on by mistake.",
                   "lights_off"),
    "Lights dimmer": ("The speaker wants the lights dimmed because they are too bright or harsh.",
                      "lights_dim"),
}
# The "nothing" option, by Spec.none_text.
SPEECH_NONE: dict[str, tuple[str, str]] = {
    "v0": ("No action", "This is not a request for the house to do anything; leave everything as it is."),
    "sharp": ("Not a request", "The speaker is not asking the house to change anything right now."),
    "long": ("Not a request", "The speaker is not asking the house to do anything right now: chatting, telling "
             "a story, quoting someone or the TV, asking whether something is on or off, talking about the "
             "past or another place, being sarcastic, or saying NOT to change something. Leave everything "
             "as it is."),
}

SPEECH_GATES: dict[str, dict[str, Any]] = {
    "ask": {
        "type": "noul",
        "instructions": "Is the speaker asking the house to do something right now, directly or by clearly "
                        "implying it (for example complaining that the room is dark or cold)?",
        "criteria": {
            "true": "Yes: a request, command or clear hint aimed at changing the lights, TV, temperature or "
                    "door right now.",
            "false": "No: small talk, a story, a quote from TV or radio, a question about whether something "
                     "is on, a comment about the past or another place, sarcasm, or telling someone NOT to "
                     "change anything.",
        },
    },
    "want": {
        "type": "noul",
        "instructions": "Does the speaker want the lights, TV, temperature or door in this house changed "
                        "right now?",
        "criteria": {
            "true": "Yes: they ask for it, or complain about the room in a way that calls for it (too dark, "
                    "too cold, too loud).",
            "false": "No: they talk about something else, the past, another place, a story or the TV, ask "
                     "a question about the house, joke, or say to leave things as they are.",
        },
    },
    "plain": {
        "type": "noul",
        "instructions": "Is this a request for the house to do something right now?",
    },
}

SCENE_OPTIONS_V0: dict[str, tuple[str, str]] = {
    "Turn on lights": ("Turn on the lights in the room where the person is.", "lights_on"),
    "Turn on TV": ("Turn on the living room TV.", "tv_on"),
    "Make warmer": ("Raise the thermostat a little.", "warmer"),
    "Turn off TV": ("Turn off the TV.", "tv_off"),
    "Alert a parent": ("Send an urgent alert to a parent's phone because someone may be in danger.",
                       "alert_parent"),
    "Do nothing": ("No action is needed; leave everything as it is.", "none"),
}

SCENE_OPTIONS_FULL: dict[str, tuple[str, str]] = {
    "Turn on lights": ("Someone is in the room, it is dark (very low lux) and the lights are off.",
                       "lights_on"),
    "Turn off lights": ("The lights are on but nobody is using the room, or the person there is asleep.",
                        "lights_off"),
    "Turn on TV": ("The TV is off and the person clearly wants to watch now: looking for the remote at "
                   "their usual TV time, or settled in for a planned movie night.", "tv_on"),
    "Turn off TV": ("The TV is on but nobody is watching: the room is empty or the person is asleep.",
                    "tv_off"),
    "Make warmer": ("The room is cold and the person shows it: sweater, blanket, rubbing their arms.",
                    "warmer"),
    "Make cooler": ("The room is hot and the person shows it: fanning, sweating, wiping their forehead.",
                    "cooler"),
    "Alert a parent": ("Someone may be in danger: a child alone near a hot stove, smoke, a person "
                       "collapsed, a stranger inside.", "alert_parent"),
}
SCENE_NONE: dict[str, tuple[str, str]] = {
    "v0": ("Do nothing", "No action is needed; leave everything as it is."),
    "sharp": ("Do nothing", "Nothing needs changing right now."),
    "long": ("Do nothing", "Nothing needs changing: the room is bright or comfortable, an adult is in control "
             "(cooking, supervising), people are leaving the house, only a pet is there, or everyone is "
             "asleep with everything already off."),
}

SCENE_GATES: dict[str, dict[str, Any]] = {
    "ask": {
        "type": "noul",
        "instructions": "Should the house change something right now, without being asked?",
        "criteria": {
            "true": "Yes: a person needs light, warmth, cooling or the TV switched, or someone is in danger.",
            "false": "No: the room is fine as it is, people are leaving, an adult is in control, only a pet "
                     "is present, or everything that matters is already off.",
        },
    },
    "want": {
        "type": "noul",
        "instructions": "Does someone in this room need the house to change the lights, TV or temperature "
                        "right now, or is someone in danger?",
        "criteria": {
            "true": "Yes: a person is stuck in the dark, visibly cold or hot, asleep with the TV or lights "
                    "on, clearly settling in to watch TV, or in danger.",
            "false": "No: the room is fine, people are leaving or busy and in control, only a pet is there, "
                     "or nothing a light, TV or thermostat change would help.",
        },
    },
    "plain": {
        "type": "noul",
        "instructions": "Does the house need to do something about this right now?",
    },
}


def _spec(version: str, overrides: dict[str, Any]) -> Spec:
    try:
        spec = VARIANTS[version]
    except KeyError as exc:
        raise ValueError(f"unknown question version {version!r}; known: {sorted(VARIANTS)}") from exc
    return replace(spec, **overrides) if overrides else spec


def _options(table: dict[str, tuple[str, str]], spec: Spec) -> dict[str, tuple[str, str]]:
    items = list(table.items())
    if spec.none_first:
        items = [kv for kv in items if kv[1][1] == "none"] + [kv for kv in items if kv[1][1] != "none"]
    if not spec.none_option:
        items = [kv for kv in items if kv[1][1] != "none"]
    return dict(items)


def _speaker_line(sp: Speaker | None) -> str:
    if not sp or not sp.get("name"):
        return "an unknown voice"
    return f"{sp['name']} ({sp.get('role', 'unknown')})"


def _own_room(sp: Speaker | None) -> str | None:
    """The bedroom a resident means by "my room" ("Maya's room", "Parents' bedroom" in home.json)."""
    res = _RESIDENTS.get((sp or {}).get("id") or "")
    if not res:
        return None
    for rid, name in ROOM_NAMES.items():
        if res["name"].lower() in name.lower():
            return rid
    if res.get("role") == "parent":
        return next((rid for rid, name in ROOM_NAMES.items() if "parent" in name.lower()), None)
    return None


def _room_question(utt: UtteranceData) -> tuple[dict[str, Any], dict[str, str | None]]:
    sp = utt.get("speaker")
    here = ROOM_NAMES.get(utt.get("room", ""), utt.get("room", "this room"))
    own = _own_room(sp)
    crit: dict[str, str] = {THIS_ROOM: f"No other room is named; the speaker means where they are ({here})."}
    rooms: dict[str, str | None] = {THIS_ROOM: None}
    for rid, name in ROOM_NAMES.items():
        if rid == own and sp and sp.get("name"):
            desc = f"{sp['name']}'s own bedroom: \"my room\" when {sp['name']} speaks."
        elif rid == "porch":
            desc = f"The {name.lower()}, outside the front door."
        else:
            desc = f"{name}." if "'" in name else f"The {name.lower()}."
        crit[name] = desc
        rooms[name] = rid
    q = {"type": "choice", "instructions": "Which room's lights is the speaker talking about?", "criteria": crit}
    return q, rooms


def _speech_table(spec: Spec) -> dict[str, tuple[str, str]]:
    if spec.descriptions == "short":
        table = dict(SPEECH_OPTIONS_V0)
        table.pop("No action")
    else:
        table = dict(SPEECH_OPTIONS_FULL)
        if spec.split_dim:
            table.pop("Lights off or dimmer")
            table.update(_DIM_SPLIT)
        if not spec.extra_actions:
            table.pop("Lock the door")
    label, desc = SPEECH_NONE[spec.none_text]
    table[label] = (desc, "none")
    return table


def _scene_table(spec: Spec) -> dict[str, tuple[str, str]]:
    if spec.descriptions == "short":
        table = dict(SCENE_OPTIONS_V0)
        table.pop("Do nothing")
    else:
        table = dict(SCENE_OPTIONS_FULL)
        if not spec.extra_actions:
            table.pop("Turn off lights")
            table.pop("Make cooler")
    label, desc = SCENE_NONE[spec.none_text]
    table[label] = (desc, "none")
    return table


def build_speech_query(utt: UtteranceData, version: str = VERSION, **overrides: Any) -> DeciderQuery:
    """Words heard with no wake word -> what (if anything) does the speaker want the house to do?"""
    spec = _spec(version, overrides)
    opts = _options(_speech_table(spec), spec)
    context = (utt.get("context") or "").strip()
    text = utt.get("text", "")
    if spec.layout == "line":
        state = f"{context} Heard: \"{text}\"".strip()
    else:
        state = f"Room: {context}\nSpeaker: {_speaker_line(utt.get('speaker'))}, in the room.\nSaid: \"{text}\""
    if spec.facts == "state":
        state = f"{HOUSE_FACTS}\n{state}"
    instructions = "The home hub's microphone heard this. No wake word was used. What should the house do?"
    if spec.facts == "instructions":
        instructions = f"{HOUSE_FACTS} {instructions}"
    questions: dict[str, dict[str, Any]] = {}
    if spec.gate:
        questions["request"] = dict(SPEECH_GATES[spec.gate_text])
        for key in filter(None, spec.extra_gates.split(",")):
            questions[f"request_{key}"] = dict(SPEECH_GATES[key])
    questions["intent"] = {"type": "choice", "instructions": instructions,
                           "criteria": {k: d for k, (d, _) in opts.items()}}
    q = DeciderQuery(kind="speech", state=state, primary="intent", headline="What does the speaker want?",
                     questions=questions, option_actions={k: a for k, (_, a) in opts.items()},
                     version=version, gate="request" if spec.gate else None,
                     speaker_room=utt.get("room"), combine=spec.combine)
    if spec.room == "auto" and not mentions_room(text):
        return q                      # no room named: the answer would be "this room" anyway (~1/4 of the cost)
    if spec.room:
        q.questions["room"], q.option_rooms = _room_question(utt)
        q.room_key = "room"
    return q


_ROOM_WORDS = tuple({n.lower() for n in ROOM_NAMES.values()} | {
    "room", "living", "lounge", "kitchen", "hall", "hallway", "bedroom", "porch", "front door", "upstairs",
    "downstairs", "客厅", "厨房", "走廊", "卧室", "房间", "门廊", "门口", "门外"})


def mentions_room(text: str) -> bool:
    """Cheap check whether the words name a place (any room, "my room", a resident's room)."""
    t = (text or "").lower()
    return any(w in t for w in _ROOM_WORDS)


def build_scene_query(scene: SceneData, version: str = VERSION, **overrides: Any) -> DeciderQuery:
    """What the cameras and sensors see -> should the house do something on its own?"""
    spec = _spec(version, overrides)
    opts = _options(_scene_table(spec), spec)
    state = scene["text"].strip()
    if spec.layout != "line":
        room = ROOM_NAMES.get(scene.get("room", ""), scene.get("room", ""))
        state = f"Room: {room}\nSeen: {state}"
    if spec.facts == "state":
        state = f"{HOUSE_FACTS}\n{state}"
    instructions = "The home hub sees this from its cameras and sensors. What should the house do right now?"
    if spec.facts == "instructions":
        instructions = f"{HOUSE_FACTS} {instructions}"
    questions: dict[str, dict[str, Any]] = {}
    if spec.gate:
        questions["act"] = dict(SCENE_GATES[spec.gate_text])
        for key in filter(None, spec.extra_gates.split(",")):
            questions[f"act_{key}"] = dict(SCENE_GATES[key])
    questions["action"] = {"type": "choice", "instructions": instructions,
                           "criteria": {k: d for k, (d, _) in opts.items()}}
    return DeciderQuery(kind="scene", state=state, primary="action", headline="What should the house do?",
                        questions=questions, option_actions={k: a for k, (_, a) in opts.items()},
                        version=version, gate="act" if spec.gate else None, combine=spec.combine)
