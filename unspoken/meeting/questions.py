"""Decider question library: how the meeting room phrases ONE request to the Strands Decider per utterance.

Questions in the request (all typed, answered together):
  kind       choice   task | decision | question | room | sensitive | small_talk | off_topic | none
  owner      choice   the checked-in names + "No one named"          (omitted without a roster)
  room       choice   brighter | dimmer | warmer | cooler | slides | none (read when kind = room)
  sensitive  noul     pay / HR / health / legal / layoffs hint       (folded into kind = sensitive)

interpret() turns the answers into (action, args, confidence): the action is one of contracts.ACTIONS,
args are written by CODE (owner from the owner choice, due date from dates.parse_due, scene from the room
choice), confidence is the probability mass behind the chosen action. Versions live in VARIANTS so the
eval (eval/run_meeting_eval.py) can compare wordings; agents call build_meeting_query() and interpret().
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from ..core.decider import DeciderAnswer
from .contracts import ACTIONS, UtteranceData
from .dates import parse_due

VERSION = "v2"
NO_OWNER = "No one named"
KINDS = ("task", "decision", "question", "room", "sensitive", "small_talk", "off_topic", "none")
ROOM_CHOICES = ("brighter", "dimmer", "warmer", "cooler", "slides", "none")
ROOM_ACTION: dict[str, tuple[str, dict[str, Any]]] = {
    "brighter": ("set_light", {"scene": "bright"}),
    "dimmer": ("set_light", {"scene": "dim"}),
    "warmer": ("set_fan", {"on": False}),
    "cooler": ("set_fan", {"on": True}),
    "slides": ("set_light", {"scene": "slides"}),
    "none": ("none", {}),
}
NONE_KINDS = ("small_talk", "off_topic", "none")
ACT_KINDS = ("task", "decision", "question", "room")
_VOLUNTEER = re.compile(
    r"(?:^|[.!?;]\s*)\s*(?:ok(?:ay)?[,.]?\s*)?"
    r"(?:(?:i'?ll|i will|i can|i'?m going to|i am going to|let me|leave (?:it|that) (?:to|with) me)\b"
    r"|我来|我会|我去|我负责)", re.I)
# These are vetoes for clear non-commitments, not a replacement task classifier. In particular,
# don't reject a polite "could/would you", or the positive reminder "don't forget to".
_REMINDER = re.compile(r"\b(?:don't|do not)\s+forget\s+to\b", re.I)
_NEGATED = re.compile(
    r"\b(?:won't|can't|cannot|shan't|don't|doesn't|didn't|couldn't|shouldn't|wouldn't"
    r"|(?:will|shall|can|could|should|would|do|does|did)\s+not)\b"
    r"|\b(?:i'm|we're|you're|he's|she's|they're|am|are|is)\s+not\b"
    r"|\b(?:not|no longer)\s+(?:going|planning|intending)\s+to\b"
    r"|\b(?:no (?:longer )?need to|never mind|nevermind)\b", re.I)
_CANCELLED = re.compile(
    r"\b(?:cancel(?:led|ed)?|withdraw|drop|forget)\s+"
    r"(?:(?:that|this)(?:\s+(?:task|commitment|action item)\b|(?=\s*(?:[.!?;,]|$)))"
    r"|(?:my|our|the)\s+(?:task|commitment|action item)\b)"
    r"|\b(?:task|commitment|action item)\b[^.!?;]{0,40}\b(?:cancelled|canceled|withdrawn|called off)\b", re.I)
_HYPOTHETICAL = re.compile(
    r"\b(?:hypothetically|suppose|supposing|what if|imagine if)\b"
    r"|\bif\b[^.!?;]*\b(?:would(?!\s+you\b)|could(?!\s+you\b)|might|i'd|we'd)\b"
    r"|\b(?:would(?!\s+you\b)|could(?!\s+you\b)|might|i'd|we'd)\b[^.!?;]*\bif\b", re.I)
_TENTATIVE = re.compile(r"\b(?:maybe|perhaps|possibly|tentatively|might)\b|\b(?:i|we)\s+(?:may|could|should)\b", re.I)


def commitment_concern(text: str) -> str | None:
    """A narrow English safety check, applied *after* model classification, with an explicit reason.

    Whole-utterance abstention is intentional for mixed/ambiguous clauses; these rules neither parse
    every language nor prove that an otherwise unflagged utterance is a commitment.
    """
    text = (text or "").replace("’", "'")
    negative_text = re.sub(r"\bnot only\b", "", _REMINDER.sub("remember to", text), flags=re.I)
    if _NEGATED.search(negative_text):
        return "negated commitment"
    if _CANCELLED.search(text):
        return "cancelled commitment"
    if _HYPOTHETICAL.search(text):
        return "hypothetical commitment"
    if _TENTATIVE.search(text):
        return "tentative commitment"
    return None


_ROOM_REQUEST_CUE = re.compile(
    r"\b(?:please|(?:can|could|would|will)\s+(?:you|we))\b"
    r"|^(?:turn|switch|brighten|dim|raise|lower)\s+"
    r"|\b(?:warm|cool)\s+(?:it|us|the room)\b"
    r"|\b(?:lights?|lamp|fan|ac)\s+(?:on|off|up|down)\b"
    r"|\b(?:next|previous)\s+slides?\b"
    r"|\b(?:show|display|put up|bring up)\s+(?:(?:the|our)\s+)?(?:slides?|agenda|timer|tasks)\b", re.I)
_ROOM_COMFORT_CUE = re.compile(
    r"\b(?:i'm|i am|it's|it is|we're|we are|(?:the room|this room|room|here)\s+is"
    r"|too|so|feel(?:s|ing)?|getting)\s+"
    r"(?:(?:very|really|quite|too|so)\s+)?(?:cold|hot|warm|freezing|chilly|stuffy|boiling|shivering|dark|bright)\b"
    r"|\b(?:cold|hot|warm|freezing|chilly|stuffy|dark|bright)\s+(?:in\s+)?here\b", re.I)


def room_action_concern(text: str) -> str | None:
    """Veto short English fragments without request/comfort evidence, not full room-intent NLP.

    Product-feature nouns/adjectives do not authorize device changes. Longer clauses and other
    languages remain model-classified; camera/reflex scene cues never go through this model guard.
    """
    text = (text or "").strip().replace("’", "'")
    words = re.findall(r"\b[\w']+\b", text)
    if not words or (len(words) <= 4 and all(w.isascii() for w in words) and
                     not _ROOM_REQUEST_CUE.search(text) and not _ROOM_COMFORT_CUE.search(text)):
        return "room fragment without request or comfort evidence"
    return None


# A "question for the room" that asks to put a document in front of people is share_docs (high risk with a guest).
_DOC = r"(?:doc|docs|document|file|deck|policy|plan|proposal|spec|slides|notes)"
_SHARE_DOCS = re.compile(r"\b(?:share|pull up|put up|paste|drop|post|link)\b.{0,40}\b" + _DOC + r"\b"
                         r"|\b" + _DOC + r"\b.{0,30}\b(?:in|to|into|on) the (?:channel|chat|thread|discord)\b"
                         r"|把.{0,10}(?:文档|文件|方案).{0,6}(?:发到|贴到|分享到|发群里|发到群)", re.I)


def wants_docs_shared(text: str) -> bool:
    return bool(_SHARE_DOCS.search(text or ""))


@dataclass(frozen=True)
class Spec:
    """Everything that varies between question versions (see VARIANTS)."""
    descriptions: str = "full"         # "short" | "full" wording of the kind options
    sensitive_gate: bool = True        # ask the yes/no sensitive question too
    sensitive_combine: str = "or"      # "or" (noisy-or with the kind) | "kind" (ignore the gate) | "product"
    room_combine: str = "product"      # confidence of a room action: P(kind=room) * P(room choice) | "kind"
    owner_question: bool = True        # ask who owns a task (else owner = plain-code heuristics only)
    layout: str = "line"               # "line" = context + quote on one line | "labeled" = one labeled line per fact
    speaker_in_owner: bool = True      # tell the owner question who is speaking ("I'll do it" = the speaker)
    merge_none: bool = True            # small_talk / off_topic / none share the "none" action mass
    instructions: str = "v2"           # which wording of the kind instructions
    gate: bool = False                 # ask a yes/no "did they give a to-do / decide / ask / want the room changed?" too
    gate_combine: str = "product"      # product of experts with the kind choice | "mixture"
    kinds: str = "8"                   # "8" = every contracts.Kind as an option | "6" = chatter/off-topic/none merged
    room: str = "always"               # "always" | "auto" = ask the room question only when the words mention the room
    question_text: str = "v2"          # wording of the "question" option and the gate: "v2" | "v4" (factual questions count)


VARIANTS: dict[str, Spec] = {
    "v1": Spec(descriptions="short", sensitive_gate=False, sensitive_combine="kind", room_combine="kind",
               speaker_in_owner=False, instructions="v1"),
    "v2": Spec(),
    "v3": Spec(gate=True, kinds="6", room="auto", instructions="v3"),
    "v4": Spec(gate=True, kinds="6", room="auto", instructions="v3", question_text="v4"),
}


@dataclass
class Interpretation:
    action: str                               # one of contracts.ACTIONS
    args: dict[str, Any]
    confidence: float                         # probability mass behind `action`
    probabilities: dict[str, float]           # by action
    kind: str                                 # the most likely kind
    kind_probabilities: dict[str, float]
    owner: str | None = None
    owner_confidence: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeciderQuery:
    state: str
    questions: dict[str, dict[str, Any]]
    names: list[str]
    speaker: str | None
    text: str
    spec: Spec
    version: str = VERSION
    owner_threshold: float = 0.5
    wipe_seconds: float = 60.0
    today: Any = None                          # datetime.date for the due parser (None = today)

    @property
    def headline(self) -> str:
        return "What did the speaker just do?"

    def owner_evidence(self, candidate: Any) -> str | None:
        """Roster membership alone (including the only other 'you') is not identity evidence."""
        if not isinstance(candidate, str) or candidate not in self.names:
            return None
        if not commitment_concern(self.text) and _VOLUNTEER.search(self.text.replace("’", "'")):
            # "I'll send it to Alex" names a recipient, not an owner. A self-commitment takes
            # precedence over mere name presence, even when the speaker's identity is unknown.
            return "known_speaker_self_commitment" if candidate == self.speaker else None
        if re.search(r"(?<!\w)" + re.escape(candidate) + r"(?!\w)", self.text, re.I):
            return "explicit_name"
        return None

    def grounded_owner(self, candidate: Any) -> str | None:
        return candidate if self.owner_evidence(candidate) else None

    def interpret(self, answers: dict[str, Any]) -> Interpretation:
        kind_a = _answer(answers["kind"])
        kinds = {k: 0.0 for k in KINDS}
        for label, p in kind_a["probabilities"].items():
            kinds[KIND_BY_LABEL.get(label, "none")] += float(p)
        if self.spec.gate and "act" in answers and _answer(answers["act"]).get("noul") is not None:
            p_act = float(_answer(answers["act"])["noul"])
            if self.spec.gate_combine == "product":
                # Product of experts: P(k) ∝ choice(k)·P(act) for the four actionable kinds, ·(1-P(act)) otherwise.
                kinds = {k: v * (p_act if k in ACT_KINDS else 1 - p_act) for k, v in kinds.items()}
            else:
                acts = {k: v for k, v in kinds.items() if k in ACT_KINDS}
                z = sum(acts.values()) or 1.0
                rest = {k: v for k, v in kinds.items() if k not in ACT_KINDS}
                zr = sum(rest.values()) or 1.0
                kinds = {**{k: p_act * v / z for k, v in acts.items()}, **{k: (1 - p_act) * v / zr for k, v in rest.items()}}
            total = sum(kinds.values()) or 1.0
            kinds = {k: v / total for k, v in kinds.items()}
        if self.spec.sensitive_gate and "sensitive" in answers:
            p_s = _answer(answers["sensitive"]).get("noul")
            if p_s is not None:
                ks = kinds["sensitive"]
                if self.spec.sensitive_combine == "or":
                    new = 1 - (1 - ks) * (1 - float(p_s))
                elif self.spec.sensitive_combine == "product":
                    new = ks * float(p_s)
                else:
                    new = ks
                rest = max(1e-9, 1 - ks)
                scale = (1 - new) / rest
                kinds = {k: (new if k == "sensitive" else v * scale) for k, v in kinds.items()}
        total = sum(kinds.values()) or 1.0
        kinds = {k: v / total for k, v in kinds.items()}
        kind = max(kinds, key=kinds.get)

        probs: dict[str, float] = {a: 0.0 for a in ACTIONS}
        probs["create_task"] += kinds["task"]
        probs["log_decision"] += kinds["decision"]
        probs["answer_question"] += kinds["question"]
        probs["pause_notes"] += kinds["sensitive"]
        for k in NONE_KINDS:
            probs["none"] += kinds[k]
        room_choice, room_probs = "none", {}
        if "room" in answers:
            ra = _answer(answers["room"])
            room_probs = {str(k): float(v) for k, v in ra["probabilities"].items()}
            room_choice = str(ra.get("choice") or "none")
            if self.spec.room_combine == "product":
                for choice, p in room_probs.items():
                    probs[ROOM_ACTION.get(choice, ("none", {}))[0]] += kinds["room"] * p
            else:
                probs[ROOM_ACTION.get(room_choice, ("none", {}))[0]] += kinds["room"]
        else:
            probs["none"] += kinds["room"]
        total = sum(probs.values()) or 1.0
        probs = {a: p / total for a, p in probs.items()}
        if wants_docs_shared(self.text) and (probs["answer_question"] + probs["create_task"]) > probs["none"]:
            # "share the pricing doc in the channel": the decider sees a question or a task; the room shares docs
            probs["share_docs"] += probs["answer_question"] + probs["create_task"]
            probs["answer_question"] = probs["create_task"] = 0.0
        action = max(probs, key=probs.get)
        if action in ("set_light", "set_fan"):
            best = max((c for c in room_probs if ROOM_ACTION[c][0] == action), key=room_probs.get, default=room_choice)
            room_choice = best

        owner, owner_conf = None, 0.0
        if "owner" in answers:
            oa = _answer(answers["owner"])
            choice = str(oa.get("choice") or NO_OWNER)
            candidate_conf = float(oa["probabilities"].get(choice, oa.get("confidence") or 0.0))
            if choice != NO_OWNER and candidate_conf >= self.owner_threshold:
                owner = self.grounded_owner(choice)
                if owner:
                    owner_conf = candidate_conf
        if owner is None and self.owner_evidence(self.speaker) == "known_speaker_self_commitment":
            owner = self.speaker                       # code evidence, not a measured owner probability

        args = self.args_for(action, owner, room_choice)
        detail = {k: {"choice": v.get("choice"), "noul": v.get("noul"), "confidence": round(float(v.get("confidence") or 0), 3)}
                  for k, v in ((k, _answer(a)) for k, a in answers.items())}
        return Interpretation(action, args, probs[action], probs, kind, kinds, owner, owner_conf, detail)

    def args_for(self, action: str, owner: str | None, room_choice: str) -> dict[str, Any]:
        if action == "create_task":
            return {"owner": owner, "text": self.text, "due": parse_due(self.text, self.today)}
        if action == "log_decision":
            return {"text": self.text}
        if action == "answer_question":
            return {"question": self.text}
        if action == "share_docs":
            return {"query": self.text}
        if action == "park_topic":
            return {"text": self.text}
        if action in ("set_light", "set_fan"):
            return dict(ROOM_ACTION.get(room_choice, ("none", {}))[1])
        if action == "pause_notes":
            return {"reason": "sounded like something private", "wipe_seconds": self.wipe_seconds}
        return {}


# --------------------------------------------------------------------------- wording
KIND_OPTIONS_SHORT: dict[str, tuple[str, str]] = {
    "Task": ("Someone is asked, or offers, to do something.", "task"),
    "Decision": ("The group decides or agrees on something.", "decision"),
    "Question for the room": ("A question the room assistant should answer.", "question"),
    "Room request": ("A wish about the lights, temperature, fan or screen.", "room"),
    "Sensitive": ("Pay, HR, health, legal or layoffs talk that should not be noted.", "sensitive"),
    "Small talk or joke": ("Chatter, a joke, a greeting.", "small_talk"),
    "Off topic": ("Not about the meeting.", "off_topic"),
    "Nothing": ("Normal discussion; nothing to note or do.", "none"),
}

KIND_OPTIONS_FULL: dict[str, tuple[str, str]] = {
    "Task": ("An action item: someone is asked to do a piece of work, or offers to do it, often with a deadline. "
             "'Sam, send the deck by Friday.' 'I'll draft the proposal tomorrow.' '你周三前把报告发我。'", "task"),
    "Decision": ("The group settles a choice or commits to a plan. 'Let's go with option B.' 'We agreed to ship on the "
                 "14th.' '那就定这个方案。'", "decision"),
    "Question for the room": ("A question the room assistant should look up and answer: what was decided before, "
                              "what a company doc or policy says, a number. 'What did we decide about pricing last "
                              "time?' 'What's our refund policy?'", "question"),
    "Room request": ("The speaker wants the room itself changed: more or less light, it is too cold or too hot, the "
                     "fan, or put the slides, agenda or timer on the screen. 'It's freezing in here.' 'Can we dim "
                     "the lights?' '把灯调亮一点。'", "room"),
    "Sensitive": ("Private matters that must stay out of the notes: someone's pay or salary, a performance or HR "
                  "complaint, health, legal exposure, layoffs or firing a real person. 'Her salary is 120k.' 'We may "
                  "have to let two people go.'", "sensitive"),
    "Small talk or joke": ("Chatter, a joke or sarcasm, a greeting or a check-in ('Hi, I'm Priya'), a laugh about "
                           "firing someone that is clearly not serious. Nothing for the assistant to do.", "small_talk"),
    "Off topic": ("Talk unrelated to the meeting: lunch, the weekend, sports, the commute. Nothing to do.", "off_topic"),
    "Nothing": ("Ordinary discussion of the topic: explaining, reporting status, thinking aloud. No task, no decision, "
                "no question for the assistant, nothing to change in the room.", "none"),
}
KIND_OPTIONS_6: dict[str, tuple[str, str]] = {
    "Task": KIND_OPTIONS_FULL["Task"],
    "Decision": KIND_OPTIONS_FULL["Decision"],
    "Question for the room": KIND_OPTIONS_FULL["Question for the room"],
    "Room request": KIND_OPTIONS_FULL["Room request"],
    "Sensitive": KIND_OPTIONS_FULL["Sensitive"],
    "Nothing to do": ("Chatter, a joke, a greeting, small talk, off-topic talk, or ordinary discussion of the topic "
                      "with no to-do, no decision and no request in it.", "none"),
}
KIND_BY_LABEL: dict[str, str] = {label: kind for label, (_, kind) in KIND_OPTIONS_FULL.items()}
KIND_BY_LABEL.update({label: kind for label, (_, kind) in KIND_OPTIONS_SHORT.items()})
KIND_BY_LABEL.update({label: kind for label, (_, kind) in KIND_OPTIONS_6.items()})

ACT_GATE: dict[str, Any] = {
    "type": "noul",
    "instructions": "Did the speaker just give someone a to-do, settle a decision for the group, ask the room assistant "
                    "to look something up, or ask for a change in the room (light, temperature, fan, screen)?",
    "criteria": {
        "true": "Yes: an action item for a named or implied person, a decision, a question for the assistant or the "
                "docs, or a wish about the room itself.",
        "false": "No: chatter, a joke, a greeting or check-in, a status report, ordinary discussion, off-topic talk, "
                 "a question to a colleague, or private matters (pay, HR, health, legal, layoffs).",
    },
}
ACT_GATE_V4: dict[str, Any] = {
    "type": "noul",
    "instructions": "Does this sentence need the meeting-room assistant to do something: write down a to-do or a "
                    "decision, answer a factual question (a date, a policy, a number, what was decided before), or "
                    "change the room (light, temperature, fan, screen)?",
    "criteria": {
        "true": "Yes: someone is given a to-do or takes one on, the group settles something, a factual question is "
                "asked aloud that the room's documents or notes could answer, or someone wants the room changed.",
        "false": "No: chatter, a joke, a greeting or check-in, a status report, an opinion, ordinary back-and-forth "
                 "discussion, off-topic talk, or private matters (pay, HR, health, legal, layoffs).",
    },
}
QUESTION_V4 = ("A factual question asked aloud that the room's documents or notes could answer: a date, a policy, a "
               "number, what was decided before. 'When is the press briefing again?' 'What's our refund policy?' "
               "'Do we know which regions the test ran in?'")

_ROOM_WORDS = ("cold", "freez", "chilly", "hot", "warm", "stuffy", "boiling", "air", "fan", " ac", "a/c", "light", "dark",
               "bright", "dim", "glare", "lamp", "slide", "screen", "agenda", "timer", "projector", "heat", "冷", "热", "灯",
               "风扇", "暗", "亮", "幻灯", "屏幕", "空调", "议程")


def mentions_room(text: str) -> bool:
    """Cheap check whether the words could be about the room (temperature, light, fan, screen)."""
    t = f" {(text or '').lower()}"
    return any(w in t for w in _ROOM_WORDS)

KIND_INSTRUCTIONS: dict[str, str] = {
    "v1": "A meeting-room assistant heard this. What kind of thing was said?",
    "v2": "The meeting room's microphone heard this sentence; nobody addressed the assistant and there is no wake "
          "word. What is it? Pick the one option that fits best.",
    "v3": "The meeting room's assistant overheard this sentence (no wake word, nobody addressed it). Which one "
          "option describes it best?",
}

ROOM_OPTIONS: dict[str, str] = {
    "brighter": "More light: it is too dark, turn the lights up or on.",
    "dimmer": "Less light: too bright, dim the lights, glare on the screen.",
    "warmer": "The speaker is cold: freezing, chilly, turn the fan or AC off, warm it up.",
    "cooler": "The speaker is hot: stuffy, warm, turn the fan or AC on, cool it down.",
    "slides": "Put the slides, the agenda, the timer or the task list on the room screen.",
    "none": "Not a request about the lights, temperature, fan or screen.",
}

SENSITIVE_GATE: dict[str, Any] = {
    "type": "noul",
    "instructions": "Does this sentence touch something that should stay out of the meeting notes: a real person's "
                    "pay or salary, a performance or HR complaint about someone, someone's health, legal exposure, "
                    "or layoffs?",
    "criteria": {
        "true": "Yes: it names or discusses pay, HR trouble, health, legal risk or job cuts of real people.",
        "false": "No: ordinary work talk, a joke, product or pricing numbers, or a general remark.",
    },
}


def _spec(version: str, overrides: dict[str, Any]) -> Spec:
    try:
        spec = VARIANTS[version]
    except KeyError as exc:
        raise ValueError(f"unknown question version {version!r}; known: {sorted(VARIANTS)}") from exc
    return replace(spec, **overrides) if overrides else spec


def _who(utt: UtteranceData) -> str:
    sp = utt.get("speaker") or {}
    if utt.get("source") == "media":
        return "The screen (a video or remote call, not a person in the room)"
    if sp.get("name"):
        return str(sp["name"])
    return "A guest without a check-in" if sp.get("guest") else "An unknown voice"


def build_meeting_query(utt: UtteranceData, names: list[str], version: str = VERSION, *,
                        owner_threshold: float = 0.5, wipe_seconds: float = 60.0, today: Any = None,
                        **overrides: Any) -> DeciderQuery:
    """ONE decider request for one utterance. `names` = checked-in names (owner options)."""
    spec = _spec(version, overrides)
    names = list(dict.fromkeys(n for n in names if n and n != NO_OWNER))
    text = str(utt.get("text") or "")
    context = (utt.get("context") or "").strip()
    who = _who(utt)
    sp = utt.get("speaker") or {}
    speaker = sp.get("name") if not sp.get("guest") and utt.get("source") != "media" else None
    speaker = speaker if speaker in names else None
    if spec.layout == "line":
        state = f'{context} {who} said: "{text}"'.strip()
    else:
        state = f"Meeting: {context}\nSpeaker: {who}\nSaid: \"{text}\""
    table = KIND_OPTIONS_6 if spec.kinds == "6" else (KIND_OPTIONS_FULL if spec.descriptions == "full" else KIND_OPTIONS_SHORT)
    questions: dict[str, dict[str, Any]] = {}
    if spec.gate:
        questions["act"] = dict(ACT_GATE_V4 if spec.question_text == "v4" else ACT_GATE)
    criteria = {label: desc for label, (desc, _) in table.items()}
    if spec.question_text == "v4" and "Question for the room" in criteria:
        criteria["Question for the room"] = QUESTION_V4
    questions["kind"] = {"type": "choice", "instructions": KIND_INSTRUCTIONS[spec.instructions], "criteria": criteria}
    if spec.owner_question and names:
        crit: dict[str, str] = {}
        for n in names:
            if spec.speaker_in_owner and n == speaker:
                crit[n] = f"{n} (the speaker) takes it on: 'I'll do it', 'I can take that', 'leave it with me'."
            else:
                crit[n] = f"{n} is explicitly named in the utterance as the person asked to do it."
        crit[NO_OWNER] = "No named person is asked and no known speaker volunteers, or this is not a task. " \
                         "An unknown speaker saying 'I' and an unnamed 'you' do not identify a roster member."
        questions["owner"] = {"type": "choice",
                              "instructions": "If this is a task, who is responsible for doing it?", "criteria": crit}
    if spec.room == "always" or mentions_room(text):
        questions["room"] = {"type": "choice",
                             "instructions": "If this is about the room, what does the speaker want?",
                             "criteria": dict(ROOM_OPTIONS)}
    if spec.sensitive_gate:
        questions["sensitive"] = dict(SENSITIVE_GATE)
    return DeciderQuery(state=state, questions=questions, names=list(names), speaker=speaker, text=text, spec=spec,
                        version=version, owner_threshold=owner_threshold, wipe_seconds=wipe_seconds, today=today)


def _answer(a: Any) -> dict[str, Any]:
    """DeciderAnswer or wire dict -> plain dict with choice / probabilities / noul / confidence."""
    if isinstance(a, DeciderAnswer):
        return {"type": a.type, "choice": a.choice, "probabilities": dict(a.probabilities), "noul": a.noul,
                "confidence": a.confidence}
    if isinstance(a, dict):
        return {"type": a.get("type"), "choice": a.get("choice"), "probabilities": dict(a.get("probabilities") or {}),
                "noul": a.get("noul"), "confidence": a.get("confidence")}
    return {"type": getattr(a, "type", None), "choice": getattr(a, "choice", None),
            "probabilities": dict(getattr(a, "probabilities", {}) or {}), "noul": getattr(a, "noul", None),
            "confidence": getattr(a, "confidence", None)}
