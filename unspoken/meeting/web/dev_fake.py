"""A fake MeetingRuntimeAPI for building and rehearsing the room screen without the real runtime.

    python -m unspoken.meeting.web.dev_fake --port 18751            # loopback only
    python -m unspoken.meeting.web.dev_fake --port 18751 --speed 8 --autoplay demo_story

It plays the 3-minute demo story as the SAME bus events the real pipeline publishes (meeting.utterance,
meeting.proposal with the Decider X-ray fields, meeting.verdict, meeting.command, meeting.item,
meeting.privacy, screen.state, egress.meter, room.env, trace.log, scenario.status) on canned
confidences, so the page can be built against realistic data. Model labels below are placeholders in the
shape of RoleHealth; the real page reads Registry labels from the runtime's snapshot.
"""
from __future__ import annotations

import argparse
import asyncio
import re
from dataclasses import asdict
from typing import Any

from ...core.bus import Bus
from ...core.events import Event
from ..contracts import MT, MeetingSettings

# --------------------------------------------------------------------------- canned model world
MODEL_LABELS = {"decider": "Strands Decider 12B (GPU)", "llm": "Qwen3.6 35B-A3B on vLLM"}
LATENCY_MS = {"reflex": 1.0, "decider": 118.0, "llm": 1840.0}
ROUTES = {"reflex": "rule", "decider": "local-decider", "llm": "local-llm", "user": "user", "rule": "rule"}

FAKE_MODELS: list[dict[str, Any]] = [
    {"role": "decider", "provider": "strands-decider", "label": MODEL_LABELS["decider"],
     "target": "http://127.0.0.1:8765", "ok": True, "latency_ms": 118.0, "detail": "12B, cuda"},
    {"role": "decider_big", "provider": "none", "label": "", "target": "disabled", "ok": True, "detail": "disabled"},
    {"role": "llm", "provider": "openai", "label": MODEL_LABELS["llm"], "target": "http://127.0.0.1:8000/v1",
     "ok": True, "latency_ms": 410.0, "detail": "vLLM, qwen3_xml tools"},
    {"role": "vlm", "provider": "openai", "label": "Qwen3.6 (vision) on vLLM", "target": "http://127.0.0.1:8000/v1",
     "ok": True, "latency_ms": 650.0, "detail": "same server as llm"},
    {"role": "asr", "provider": "python:unspoken.sensors.asr:FasterWhisper", "label": "faster-whisper large-v3-turbo",
     "target": "in-process", "ok": True, "latency_ms": 210.0, "detail": "cuda, int8"},
    {"role": "vad", "provider": "python:unspoken.sensors.vad:Silero", "label": "Silero VAD", "target": "in-process",
     "ok": True, "latency_ms": 2.0, "detail": "onnx"},
    {"role": "diarization", "provider": "python:unspoken.sensors.diar:WeSpeaker", "label": "WeSpeaker voice labels",
     "target": "in-process", "ok": True, "latency_ms": 35.0, "detail": "session-only"},
    {"role": "faces", "provider": "python:unspoken.sensors.faces:YuNet", "label": "YuNet face tracker",
     "target": "in-process", "ok": True, "latency_ms": 9.0, "detail": "no recognition"},
    {"role": "asd", "provider": "python:unspoken.sensors.asd:LightASD", "label": "Light-ASD", "target": "in-process",
     "ok": True, "latency_ms": 14.0, "detail": "who is talking"},
    {"role": "tts", "provider": "none", "label": "", "target": "disabled", "ok": True, "detail": "disabled"},
    {"role": "embed", "provider": "python:unspoken.sensors.embed:BgeM3", "label": "bge-m3 (local docs)",
     "target": "in-process", "ok": True, "latency_ms": 22.0, "detail": "docs folder"},
    {"role": "pii", "provider": "python:unspoken.sensors.pii:Gliner", "label": "GLiNER PII", "target": "in-process",
     "ok": True, "latency_ms": 31.0, "detail": "before anything leaves"},
]

INTENT_OPTIONS: dict[str, tuple[str, str]] = {   # label -> (action, description) for the typed question
    "A task for someone": ("create_task", "someone is asked to do something, maybe by a date"),
    "A decision": ("log_decision", "the group settled something"),
    "A question for the notes": ("answer_question", "asks what was decided or what a document says"),
    "Park it": ("park_topic", "an idea to come back to later"),
    "Room: light": ("set_light", "lighting, slides, brightness"),
    "Room: temperature": ("set_fan", "too hot, too cold, air"),
    "Show something": ("show", "asks to see the agenda, timer, tasks or what they missed"),
    "Share a document": ("share_docs", "asks to pull up a file or sheet"),
    "Nothing to do": ("none", "conversation, small talk, or none of the above"),
}
ACTION_TO_LABEL = {a: lbl for lbl, (a, _) in INTENT_OPTIONS.items()}

# phrase fragments -> (kind, action, args, decider confidence). Confidence below act_threshold escalates.
_GUESSES: list[tuple[tuple[str, ...], str, str, dict[str, Any], float]] = [
    (("off the record",), "sensitive", "pause_notes", {"reason": "said 'off the record'", "wipe_seconds": 60}, 1.0),
    (("send the deck", "by friday", "by monday", "can you send", "please send"), "task", "create_task",
     {"owner": "Sam", "text": "send the deck", "due": "Friday"}, 0.92),
    (("someone should", "we need someone to"), "task", "create_task",
     {"owner": None, "text": "update the pricing page", "due": None}, 0.84),
    (("let's wrap", "sam owns the deck"), "decision", "log_decision",
     {"text": "Enterprise stays at $48; Sam owns the deck"}, 0.85),
    (("we agreed", "decision:", "we decided", "stays at"), "decision", "log_decision",
     {"text": "Enterprise tier stays at $48 a seat"}, 0.83),
    (("what did we decide", "what did we say about", "last month"), "question", "answer_question",
     {"question": "What did we decide about the launch date?"}, 0.58),
    (("what did i miss", "catch me up"), "question", "show", {"card": "catch_up"}, 0.81),
    (("freezing", "too cold", "it's cold"), "room", "set_fan", {"on": False}, 0.88),
    (("too hot", "stuffy", "warm in here"), "room", "set_fan", {"on": True}, 0.87),
    (("share my slides", "my slides", "dim the lights"), "room", "set_light", {"scene": "slides"}, 0.9),
    (("pull up the", "margin sheet", "open the sheet"), "sensitive", "share_docs", {"query": "margin sheet"}, 0.86),
    (("park that", "parking lot", "come back to"), "parking_lot", "park_topic", {"text": "revisit the SMB discount"}, 0.8),
]
RESUME_PHRASES = ("back on the record", "on the record again", "resume notes", "you can take notes")

DOCS = [
    {"doc": "notes-2026-09-12.md", "chunk": 3, "score": 0.91,
     "text": "Launch moved from Oct 28 to Nov 18 so pricing and the deck ship together."},
    {"doc": "pricing-2026.md", "chunk": 1, "score": 0.74,
     "text": "Enterprise tier: $48 per seat per month, annual only."},
]
LLM_REPLIES = {
    "answer_question": "The notes from 2026-09-12 say the launch moved to Nov 18 so pricing and the deck ship together.",
    "show": "Here is what you missed: one decision and one task so far.",
}
NAME_RE = re.compile(r"^\s*([A-Z][a-z]+),")


def spread(options: list[str], pick: str, conf: float) -> dict[str, float]:
    rest = round((1 - conf) / max(1, len(options) - 1), 4)
    return {o: (round(conf, 4) if o == pick else rest) for o in options}


def fake_tokens(state: str, reply: str | None) -> dict[str, int]:
    """Local LLM usage in a believable range: 11 tool schemas + system prompt dominate the input."""
    return {"input": 1720 + len(state) // 3, "output": 42 + (len(reply or "") + 3) // 4}


def _fmt_bytes(n: int) -> str:
    return f"{n} B" if n < 1024 else f"{n / 1024:.1f} KB"


# --------------------------------------------------------------------------- the fake runtime
class FakeMeetingRuntime:
    """Implements unspoken.meeting.contracts.MeetingRuntimeAPI on canned data."""

    def __init__(self, settings: MeetingSettings | None = None, *, fast: bool = False) -> None:
        self.bus = Bus()
        self.settings = settings or MeetingSettings()
        self.fast = fast
        self._task: asyncio.Task | None = None
        self._scenario_id: str | None = None
        self._scenarios = build_scenarios()
        self._reset_state()

    # ------------------------------------------------------------------ state
    def _reset_state(self) -> None:
        self.clock = 0.0
        self.status: dict[str, Any] = {"state": "ended", "title": "", "topic": None, "owner": None, "ends_at": None}
        self.privacy: dict[str, Any] = {"state": "on", "reason": "meeting start", "by": "user",
                                        "wiped_from": None, "wiped_count": 0}
        self.participants: dict[str, dict[str, Any]] = {}
        self.captions: list[dict[str, Any]] = []
        self._transcript: list[dict[str, Any]] = []
        self.agenda: dict[str, Any] = {"topic": None, "topics": [], "minutes_left": None, "ends_at": None,
                                       "minutes_total": None}
        self.cards: list[dict[str, Any]] = []
        self.banner: str | None = None
        self._items: dict[str, dict[str, Any]] = {}
        self.devices: dict[str, dict[str, Any]] = {
            "lamp": {"device_id": "lamp", "kind": "light", "state": {"on": True, "scene": "talk", "brightness": 80}},
            "fan": {"device_id": "fan", "kind": "plug", "state": {"on": True}},
        }
        self.egress: dict[str, Any] = {"dest": "discord", "kind": "text", "bytes": 0, "total_bytes": 0,
                                       "audio_bytes_out": 0, "video_bytes_out": 0}
        self.posts: list[dict[str, Any]] = []
        self.scene_mode = "talk"
        self._pending_recap: str | None = None
        self._last_task: str | None = None
        self._seq = 0

    def _nid(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self._seq:03d}"

    @property
    def guests_present(self) -> bool:
        return any(p.get("guest") for p in self.participants.values())

    # ------------------------------------------------------------------ MeetingRuntimeAPI
    def snapshot(self) -> dict[str, Any]:
        return {
            "status": dict(self.status),
            "privacy": dict(self.privacy),
            "participants": [dict(p) for p in self.participants.values()],
            "screen": self._screen(),
            "items": self.items(),
            "devices": [dict(d) for d in self.devices.values()],
            "egress": dict(self.egress),
            "models": [dict(m) for m in FAKE_MODELS],
        }

    def transcript(self, since: float | None = None) -> list[dict[str, Any]]:
        return [dict(u) for u in self._transcript if since is None or u["start"] >= since]

    def items(self, kind: str | None = None) -> list[dict[str, Any]]:
        return [dict(i) for i in self._items.values() if kind is None or i["kind"] == kind]

    async def search_docs(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        if self.guests_present:
            return []
        return [dict(d) for d in DOCS[:k]]

    async def recap(self) -> str:
        decisions = [i for i in self._items.values() if i["kind"] == "decision" and i["status"] != "undone"]
        tasks = [i for i in self._items.values() if i["kind"] == "task" and i["status"] != "undone"]
        questions = [i for i in self._items.values() if i["kind"] == "question"]
        lines = [f"**Recap: {self.status.get('title') or 'meeting'}**", "", "**Decisions**"]
        lines += [f"- {d['text']}" for d in decisions] or ["- none"]
        lines += ["", "**Tasks**"]
        lines += [f"- {t['text']} — {t['owner'] or 'no owner yet'}" + (f", due {t['due']}" if t['due'] else "")
                  for t in tasks] or ["- none"]
        lines += ["", "**Open questions**"]
        lines += [f"- {q['text']}" for q in questions] or ["- none"]
        return "\n".join(lines)

    async def chat(self, command: dict[str, Any]) -> dict[str, Any]:
        cmd, user, args, text = command.get("command"), command.get("user", "?"), command.get("args") or {}, command.get("text", "")
        ev = await self._emit(MT.CHAT, {"command": cmd, "user": user, "args": args, "text": text}, "bridge.discord")
        if cmd == "here":
            return await self._checkin(args.get("name") or user, ev)
        if cmd == "offrecord":
            await self.set_privacy("off_record", f"/offrecord by {user}", by="user")
            return {"ok": True, "text": "Off the record. Notes paused and the last minute wiped."}
        if cmd == "resume":
            await self.set_privacy("on", f"/resume by {user}", by="user")
            return {"ok": True, "text": "Back on the record."}
        if cmd == "recap":
            md = await self._post("notes", await self.recap(), kind="recap", cause=ev)
            return {"ok": True, "text": md}
        if cmd == "agenda":
            topics = list(args.get("topics") or self.agenda["topics"])
            self.agenda.update({"topics": topics, "topic": topics[0] if topics else None,
                                "ends_at": args.get("ends_at", self.agenda["ends_at"])})
            await self._screen_changed(ev)
            return {"ok": True, "text": f"Agenda: {', '.join(topics)}"}
        if cmd == "undo":
            item_id = args.get("item_id") or self._last_task
            item = self._items.get(item_id or "")
            if not item:
                return {"ok": False, "text": "Nothing to undo."}
            item["status"] = "undone"
            self.cards = [c for c in self.cards if c.get("item_id") != item_id]
            await self._emit(MT.ITEM, dict(item), "agent.items", ev)
            await self._log("items", f"{user} undid: {item['text']}", ev)
            await self._screen_changed(ev)
            return {"ok": True, "text": f"Undone: {item['text']}"}
        if cmd == "approve":
            if self._pending_recap is None:
                return {"ok": False, "text": "Nothing waiting for approval."}
            self._pending_recap = None
            self.cards = [c for c in self.cards if c.get("card") != "approval"]
            md = await self._post("notes", await self.recap(), kind="recap", cause=ev)
            self.cards.insert(0, {"card": "posted", "title": "Recap posted", "channel": "#notes",
                                  "text": f"{_fmt_bytes(len(md.encode()))} of text to Discord. Nothing else left the box."})
            await self._screen_changed(ev)
            return {"ok": True, "text": "Recap posted to #notes."}
        if cmd == "deny":
            self._pending_recap = None
            self.cards = [c for c in self.cards if c.get("card") != "approval"]
            await self._screen_changed(ev)
            return {"ok": True, "text": "Recap not posted."}
        if cmd == "ask":
            await self._hear(None, text, "en", source="person", via_user=user, cause=ev)
            return {"ok": True, "text": "Asked."}
        raise ValueError(f"unknown command {cmd!r}")

    async def set_privacy(self, state: str, reason: str, by: str = "user") -> dict[str, Any]:
        wiped_from, wiped = None, 0
        if state == "off_record":
            wiped_from = max(0.0, self.clock - self.settings.wipe_seconds)
            before = len(self.captions)
            self.captions = [c for c in self.captions if c["t"] < wiped_from]
            self._transcript = [u for u in self._transcript if u["start"] < wiped_from]
            wiped = before - len(self.captions)
        self.privacy = {"state": state, "reason": reason, "by": by, "wiped_from": wiped_from, "wiped_count": wiped}
        ev = await self._emit(MT.PRIVACY, dict(self.privacy), f"agent.privacy")
        scene = {"on": "talk", "paused": "privacy", "off_record": "off_record"}[state]
        if state == "on" and self.scene_mode == "slides":
            scene = "slides"
        await self._device("lamp", "scene", {"scene": scene}, ev)
        await self._log("privacy", {"on": "Notes on.", "paused": f"Notes paused: {reason}.",
                                    "off_record": f"Off the record ({reason}). {wiped} lines wiped."}[state], ev)
        await self._screen_changed(ev)
        return dict(self.privacy)

    def scenarios(self) -> list[dict[str, Any]]:
        return [dict(s) for s in self._scenarios]

    async def run_scenario(self, scenario_id: str) -> None:
        sc = next((s for s in self._scenarios if s["id"] == scenario_id), None)
        if sc is None:
            raise KeyError(scenario_id)
        await self.stop_scenario()
        self._scenario_id = scenario_id
        self._task = asyncio.create_task(self._run(sc), name=f"scenario:{scenario_id}")

    async def stop_scenario(self) -> None:
        task, self._task = self._task, None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await self._emit(MT.SCENARIO, {"scenario_id": self._scenario_id, "state": "stopped", "step_index": -1,
                                           "text": "stopped", "passed": 0, "failed": 0}, "scenario")

    async def reset(self) -> None:
        await self.stop_scenario()
        self._reset_state()
        await self._emit(MT.STATUS, dict(self.status), "runtime")
        await self._screen_changed(None)

    async def update_settings(self, **changes: Any) -> MeetingSettings:
        for k, v in changes.items():
            if not hasattr(self.settings, k):
                raise ValueError(f"unknown setting {k!r}")
            setattr(self.settings, k, v)
        return self.settings

    # ------------------------------------------------------------------ plumbing
    async def _emit(self, topic: str, data: dict[str, Any], source: str, cause: Event | None = None) -> Event:
        ev = Event(topic, data, source=source, ts=round(self.clock, 2), cause=cause.id if cause else None)
        await self.bus.publish(ev)
        return ev

    async def _log(self, agent: str, text: str, cause: Event | None = None, level: str = "info") -> None:
        await self._emit(MT.LOG, {"agent": agent, "level": level, "text": text}, f"agent.{agent}", cause)

    async def _sleep(self, seconds: float) -> None:
        self.clock += seconds
        if not self.fast and seconds > 0:
            await asyncio.sleep(seconds / max(0.01, self.settings.speed))

    async def _think(self, ms: float) -> None:
        await self._sleep(ms / 1000.0)

    def _screen(self) -> dict[str, Any]:
        n = self.settings.caption_lines
        cards = self.cards if not self.guests_present else [c for c in self.cards if not c.get("docs")]
        return {"privacy": self.privacy["state"], "captions": [dict(c) for c in self.captions[-n:]],
                "agenda": dict(self.agenda), "cards": [dict(c) for c in cards[:4]], "banner": self.banner}

    async def _screen_changed(self, cause: Event | None) -> None:
        await self._emit(MT.SCREEN, self._screen(), "agent.screen", cause)

    async def _device(self, device_id: str, command: str, args: dict[str, Any], cause: Event | None) -> None:
        cmd = await self._emit(MT.COMMAND, {"target": device_id, "command": command, "args": args,
                                            "verdict_id": cause.id if cause else ""}, "agent.actuator", cause)
        dev = self.devices[device_id]
        if command == "scene":
            dev["state"].update({"on": True, "scene": args["scene"],
                                 "brightness": {"slides": 25, "off_record": 60, "privacy": 60}.get(args["scene"], 80)})
        elif command in ("on", "off"):
            dev["state"]["on"] = command == "on"
        await self._emit(MT.ENV, dict(dev), f"device.{device_id}", cmd)

    async def _post(self, channel: str, text: str, *, kind: str = "note", cause: Event | None = None) -> str:
        n = len(text.encode("utf-8"))
        cmd = await self._emit(MT.COMMAND, {"target": "notifier", "command": "post",
                                            "args": {"channel": channel, "kind": kind, "text": text},
                                            "verdict_id": cause.id if cause else ""}, "agent.actuator", cause)
        self.posts.append({"channel": channel, "kind": kind, "text": text})
        self.egress["bytes"] = n
        self.egress["total_bytes"] += n
        await self._emit(MT.EGRESS, dict(self.egress), "bridge.notifier", cmd)
        return text

    def _speaker(self, who: str | None) -> dict[str, Any]:
        p = self.participants.get(who or "")
        if p is None:
            return {"name": None, "voice": None, "seat": None, "conf": 0.3, "guest": True}
        return dict(p)

    def _context(self) -> str:
        who = ", ".join(p["name"] or "a guest" for p in self.participants.values())
        left = self.agenda.get("minutes_left")
        return (f"topic: {self.agenda.get('topic') or 'none'}; {left if left is not None else '?'} min left; "
                f"room: {self.scene_mode}; here: {who or 'nobody'}; notes {self.privacy['state']}")

    # ------------------------------------------------------------------ the pipeline, faked
    async def _hear(self, who: str | None, text: str, lang: str = "en", *, source: str = "person",
                    via_user: str | None = None, cause: Event | None = None) -> None:
        speaker = self._speaker(who) if via_user is None else {"name": via_user, "voice": None, "seat": None,
                                                                "conf": 1.0, "guest": False}
        uid = self._nid("utt")
        start, end = self.clock, self.clock + max(1.0, len(text) / 14)
        low = text.lower()
        if self.privacy["state"] == "off_record" and not any(p in low for p in RESUME_PHRASES):
            # off the record: the line is dropped before fusion publishes it; no text reaches the bus or the screen
            await self._log("privacy", "off the record: a line was heard and dropped", cause)
            return
        shown = text
        if "off the record" in low and self.privacy["state"] != "off_record":
            cut = low.index("off the record") + len("off the record")      # the trigger is kept, the rest never is
            shown = text[:cut] + (" […]" if cut < len(text.rstrip(". ")) else "")
        utt = {"utterance_id": uid, "segment_id": self._nid("seg"), "start": round(start, 2), "end": round(end, 2),
               "text": shown, "lang": lang, "speaker": speaker, "source": source, "context": self._context()}
        ev = await self._emit(MT.UTTERANCE, utt, "agent.fusion", cause)
        if self.privacy["state"] == "off_record":
            await self._reflex_resume(ev, uid, text)
            return
        if source == "person":
            self.captions.append({"name": speaker.get("name") or "Guest", "text": shown, "t": round(start, 2),
                                  "utterance_id": uid, "voice": speaker.get("voice")})
            self._transcript.append(utt)
            await self._screen_changed(ev)
        else:
            await self._emit(MT.PROPOSAL, {"by": "reflex", "mode": "act", "kind": "none", "action": "none", "args": {},
                                           "confidence": 1.0, "probabilities": {"none": 1.0}, "utterance_id": uid,
                                           "reason": "voice came from the room screen (media), not a person",
                                           "route": "rule", "model": "rule: media voice", "request": {}, "answers": {},
                                           "threshold": self.settings.act_threshold, "latency_ms": 0.4,
                                           "tokens": None, "escalated_from": None}, "agent.reflex", ev)
            return
        guess = next(((k, a, dict(args), c) for phrases, k, a, args, c in _GUESSES if any(p in low for p in phrases)),
                     ("small_talk", "none", {}, 0.86))
        kind, action, args, conf = guess
        if action == "create_task" and (m := NAME_RE.match(text)):
            args["owner"] = m.group(1)
        if action == "answer_question" and via_user:
            args["question"] = text
        if action == "pause_notes":
            await self._reflex_off_record(ev, uid, args)
            return
        await self._decide(ev, uid, text, speaker, kind, action, args, conf)

    async def _reflex_off_record(self, cause: Event, uid: str, args: dict[str, Any]) -> None:
        prop = await self._emit(MT.PROPOSAL, {
            "by": "reflex", "mode": "act", "kind": "sensitive", "action": "pause_notes", "args": args,
            "confidence": 1.0, "probabilities": {"pause_notes": 1.0}, "utterance_id": uid, "escalated_from": None,
            "reason": "reflex: the words 'off the record' always pause notes, no model asked",
            "route": "rule", "model": "rule: off_the_record", "request": {}, "answers": {},
            "threshold": self.settings.act_threshold, "latency_ms": LATENCY_MS["reflex"], "tokens": None}, "agent.reflex", cause)
        ver = await self._emit(MT.VERDICT, {"proposal_id": prop.id, "action": "pause_notes", "args": args,
                                            "verdict": "act", "risk": "low",
                                            "rule": "privacy first: pause_notes is always allowed"}, "agent.policy", prop)
        await self.set_privacy("off_record", "said 'off the record'", by="reflex")
        await self._log("policy", "off the record: notes paused, last minute wiped, lamp red", ver)

    async def _reflex_resume(self, cause: Event, uid: str, text: str) -> None:
        prop = await self._emit(MT.PROPOSAL, {
            "by": "reflex", "mode": "act", "kind": "sensitive", "action": "resume_notes", "args": {},
            "confidence": 1.0, "probabilities": {"resume_notes": 1.0}, "utterance_id": uid, "escalated_from": None,
            "reason": "reflex: 'back on the record' resumes notes, no model asked",
            "route": "rule", "model": "rule: back_on_record", "request": {}, "answers": {},
            "threshold": self.settings.act_threshold, "latency_ms": LATENCY_MS["reflex"], "tokens": None}, "agent.reflex", cause)
        ver = await self._emit(MT.VERDICT, {"proposal_id": prop.id, "action": "resume_notes", "args": {},
                                            "verdict": "act", "risk": "medium",
                                            "rule": "the people in the room said so; the paused line itself is not kept"},
                               "agent.policy", prop)
        await self.set_privacy("on", f"said '{text.strip().rstrip('.')}'", by="reflex")
        await self._log("policy", "back on the record: notes resume, lamp back to the room scene", ver)

    def _request(self, text: str, speaker: dict[str, Any], kind_hint: str) -> dict[str, Any]:
        name = speaker.get("name") or "a guest"
        state = (f"Meeting '{self.status.get('title') or 'untitled'}', {self._context()}. "
                 f"{name} ({'checked in' if speaker.get('name') else 'no check-in'}) said: \"{text}\"")
        return {"state": state, "questions": {
            "intent": {"type": "choice", "instructions": "What, if anything, should the room do with this sentence?",
                       "criteria": {lbl: desc for lbl, (_, desc) in INTENT_OPTIONS.items()}},
            "actionable": {"type": "noul",
                           "instructions": "Is this a request or a fact the notes should keep, rather than conversation?"},
        }}

    def _answers(self, action: str, conf: float) -> dict[str, dict[str, Any]]:
        labels = list(INTENT_OPTIONS)
        pick = ACTION_TO_LABEL.get(action, "Nothing to do")
        p = 0.15 if action == "none" else max(conf, 0.6)
        return {"intent": {"type": "choice", "choice": pick, "noul": None, "confidence": round(conf, 4),
                           "probabilities": spread(labels, pick, conf)},
                "actionable": {"type": "noul", "choice": None, "noul": round(p, 4), "confidence": round(abs(2 * p - 1), 4),
                               "probabilities": {"true": round(p, 4), "false": round(1 - p, 4)}}}

    async def _decide(self, cause: Event, uid: str, text: str, speaker: dict[str, Any], kind: str, action: str,
                      args: dict[str, Any], conf: float) -> None:
        threshold = self.settings.act_threshold
        request = self._request(text, speaker, kind)
        labels = list(INTENT_OPTIONS)
        pick = ACTION_TO_LABEL.get(action, "Nothing to do")
        unsure = conf < threshold
        if unsure and conf < self.settings.ask_threshold:
            action, kind, args, unsure = "none", "none", {}, False
        escalate = unsure and self.settings.escalate
        await self._think(LATENCY_MS["decider"])
        reason = (f"unsure ({conf:.0%} < {threshold:.0%}), asking {MODEL_LABELS['llm']}" if escalate else
                  f"decider is {conf:.0%} sure: {pick.lower()}" if action != "none" else
                  f"decider is {conf:.0%} sure there is nothing to do")
        data: dict[str, Any] = {
            "by": "decider", "mode": "shadow" if escalate else "act", "kind": kind, "action": action, "args": args,
            "confidence": round(conf, 4), "probabilities": spread(labels, pick, conf), "reason": reason,
            "utterance_id": uid, "escalated_from": None, "route": "local-decider", "model": MODEL_LABELS["decider"],
            "request": request, "answers": self._answers(action, conf), "threshold": threshold,
            "latency_ms": LATENCY_MS["decider"], "tokens": None,
        }
        prop = await self._emit(MT.PROPOSAL, data, "agent.decider", cause)
        if escalate:
            await self._think(LATENCY_MS["llm"])
            reply = LLM_REPLIES.get(action, f"I'll {action.replace('_', ' ')}.")
            tool_calls = [{"name": action, "input": args}]
            if action == "answer_question":
                tool_calls = [{"name": "search_docs", "input": {"query": "launch date decision", "k": 3}},
                              {"name": "answer_question", "input": {"question": args["question"], "doc": DOCS[0]["doc"]}}]
            llm = dict(data)
            llm.update({"by": "llm", "mode": "act", "confidence": 1.0, "probabilities": {action: 1.0}, "answers": {},
                        "reason": f"local LLM read the transcript and the docs: {action.replace('_', ' ')}",
                        "escalated_from": prop.id, "route": "local-llm", "model": MODEL_LABELS["llm"],
                        "latency_ms": LATENCY_MS["llm"], "tokens": fake_tokens(request["state"], reply),
                        "tool_calls": tool_calls, "reply": reply})
            prop = await self._emit(MT.PROPOSAL, llm, "agent.llm", prop)
        await self._policy(prop, speaker)

    async def _policy(self, prop: Event, speaker: dict[str, Any]) -> None:
        d = prop.data
        action, args = d["action"], d.get("args") or {}
        risk = {"create_task": "medium", "log_decision": "medium", "resume_notes": "medium", "post_recap": "high",
                "share_docs": "high" if self.guests_present else "medium"}.get(action, "low")
        if action == "none":
            verdict, rule = "ignore", "nothing to do"
        elif self.guests_present and action in ("share_docs", "answer_question"):
            verdict, rule = "deny", "guest in the room: local docs stay hidden"
        elif action == "post_recap":
            verdict, rule = "ask_owner", "high risk: the meeting owner approves anything that leaves as a summary"
        elif action == "share_docs":
            verdict, rule = "act", "no guest present: docs may be shown on the room screen"
        else:
            verdict, rule = "act", f"{risk} risk, confidence above threshold" if d["by"] != "llm" else f"{risk} risk, local LLM decided"
        ver = await self._emit(MT.VERDICT, {"proposal_id": prop.id, "action": action, "args": args, "verdict": verdict,
                                            "risk": risk, "rule": rule}, "agent.policy", prop)
        if verdict == "deny":
            await self._log("policy", f"denied {action}: {rule}", ver, level="warn")
            self.banner = "Guest in the room · local docs hidden"
            await self._screen_changed(ver)
        elif verdict == "act":
            await self._actuate(ver, prop, speaker)
        elif verdict == "ask_owner":
            await self._ask_owner(ver, prop)

    async def _actuate(self, ver: Event, prop: Event, speaker: dict[str, Any]) -> None:
        action, args, uid = ver.data["action"], ver.data["args"], prop.data.get("utterance_id")
        if action == "create_task":
            item = {"item_id": self._nid("task"), "kind": "task", "text": args.get("text", ""), "owner": args.get("owner"),
                    "due": args.get("due"), "status": "open", "utterance_id": uid, "message": None}
            text = f"✅ Task: {item['text']} — {item['owner'] or 'no owner yet'}" + (f", due {item['due']}" if item['due'] else "") + "  (react ❌ to undo)"
            await self._post("tasks", text, kind="task", cause=ver)
            item["message"] = {"channel": "tasks", "id": self._nid("msg"), "thread": None}
            self._items[item["item_id"]] = item
            self._last_task = item["item_id"]
            await self._emit(MT.ITEM, dict(item), "agent.items", ver)
            self.cards.insert(0, {"card": "task", "item_id": item["item_id"], "text": item["text"], "owner": item["owner"],
                                  "due": item["due"], "undo": "undo in Discord ❌"})
        elif action == "log_decision":
            item = {"item_id": self._nid("dec"), "kind": "decision", "text": args.get("text", ""), "owner": None, "due": None,
                    "status": "open", "utterance_id": uid, "message": None}
            await self._post("notes", f"📌 Decision: {item['text']}", kind="decision", cause=ver)
            item["message"] = {"channel": "notes", "id": self._nid("msg"), "thread": None}
            self._items[item["item_id"]] = item
            await self._emit(MT.ITEM, dict(item), "agent.items", ver)
            self.cards.insert(0, {"card": "decision", "item_id": item["item_id"], "text": item["text"]})
        elif action == "answer_question":
            hits = await self.search_docs(args.get("question", ""))
            answer = prop.data.get("reply") or (hits[0]["text"] if hits else "Nothing in the local docs.")
            self.cards.insert(0, {"card": "answer", "docs": True, "question": args.get("question", ""),
                                  "text": answer, "doc": hits[0]["doc"] if hits else None})
            await self._post("notes", f"💬 {args.get('question', '')}\n{answer}", kind="answer", cause=ver)
        elif action == "show" and args.get("card") == "catch_up":
            name = speaker.get("name") or "Guest"
            lines = [f"Decision: {i['text']}" for i in self._items.values() if i["kind"] == "decision"] + \
                    [f"Task: {i['text']} ({i['owner'] or 'no owner'})" for i in self._items.values() if i["kind"] == "task"]
            self.cards.insert(0, {"card": "catch_up", "for": name, "topic": self.agenda.get("topic"),
                                  "lines": lines[:4] or ["Nothing decided yet."]})
        elif action == "set_fan":
            await self._device("fan", "on" if args.get("on") else "off", {}, ver)
            await self._log("actuator", f"fan {'on' if args.get('on') else 'off'} (stands in for the AC)", ver)
        elif action == "set_light":
            await self._device("lamp", "scene", {"scene": args.get("scene", "talk")}, ver)
            await self._log("actuator", f"lamp scene {args.get('scene')}", ver)
        elif action == "park_topic":
            item = {"item_id": self._nid("park"), "kind": "parking_lot", "text": args.get("text", ""), "owner": None,
                    "due": None, "status": "open", "utterance_id": uid, "message": None}
            self._items[item["item_id"]] = item
            await self._emit(MT.ITEM, dict(item), "agent.items", ver)
        elif action == "share_docs":
            self.cards.insert(0, {"card": "answer", "docs": True, "question": args.get("query", ""),
                                  "text": DOCS[1]["text"], "doc": DOCS[1]["doc"]})
        await self._screen_changed(ver)

    async def _ask_owner(self, ver: Event, prop: Event) -> None:
        owner = self.status.get("owner") or "the owner"
        rid = self._nid("req")
        self._pending_recap = rid
        await self._post("dm", f"Recap ready. ✅ to post it to #notes, ❌ to drop it. (request {rid})", kind="approval", cause=ver)
        self.cards.insert(0, {"card": "approval", "request_id": rid, "text": f"Recap ready · waiting for {owner}'s ✅ in Discord"})
        await self._log("policy", f"recap waits for {owner}'s approval (request {rid})", ver)
        await self._screen_changed(ver)

    async def _checkin(self, name: str, cause: Event) -> dict[str, Any]:
        target = next((k for k, p in self.participants.items() if p.get("name") is None and k != "visitor"), None)
        if target is None:
            target = name.lower()
            self.participants[target] = {"name": None, "voice": None, "seat": None, "conf": 0.5, "guest": False}
        p = self.participants[target]
        p.update({"name": name, "guest": False, "conf": 0.97})
        for c in self.captions:            # the voice now has a name: relabel what it already said
            if c.get("voice") and c["voice"] == p.get("voice") and c["name"] == "Guest":
                c["name"] = name
        for c in self.cards:
            if c.get("card") == "catch_up" and c.get("for") == "Guest":
                c["for"] = name
        await self._log("fusion", f"{name} checked in (seat {p.get('seat') or '?'}, {p.get('voice') or 'no voice yet'})", cause)
        await self._screen_changed(cause)
        return {"ok": True, "text": f"Welcome, {name}."}

    # ------------------------------------------------------------------ scenario steps
    async def _do_start(self, s: dict[str, Any]) -> None:
        topics = list(s.get("agenda") or [])
        minutes = int(s.get("minutes", 30))
        self.status = {"state": "started", "title": s.get("title", "Meeting"), "topic": topics[0] if topics else None,
                       "owner": s.get("owner"), "ends_at": s.get("ends_at", "15:30")}
        self.agenda = {"topic": topics[0] if topics else None, "topics": topics, "minutes_left": minutes,
                       "ends_at": self.status["ends_at"], "minutes_total": minutes}
        self.privacy = {"state": "on", "reason": "meeting start", "by": "user", "wiped_from": None, "wiped_count": 0}
        ev = await self._emit(MT.STATUS, dict(self.status), "runtime")
        await self._emit(MT.PRIVACY, dict(self.privacy), "agent.privacy", ev)
        await self._device("lamp", "scene", {"scene": "talk"}, ev)
        await self._screen_changed(ev)

    async def _do_join(self, s: dict[str, Any]) -> None:
        name = s.get("checkin")
        self.participants[s["who"]] = {"name": name, "voice": s.get("voice"), "seat": s.get("seat"),
                                       "conf": 0.95 if name else 0.4, "guest": name is None}
        await self._emit(MT.SPEAKERS, {"t": self.clock, "faces": [{"track_id": s["who"], "seat": s.get("seat"),
                                                                   "bbox": [0.2, 0.3, 0.1, 0.2], "speaking": 0.0}],
                                       "face_count": len(self.participants), "screen_talking": False}, "sensors.video")
        await self._log("fusion", f"{name or 'someone'} sat down at {s.get('seat')}" + ("" if name else " (no check-in yet)"))
        await self._screen_changed(None)

    async def _do_say(self, s: dict[str, Any]) -> None:
        await self._hear(s["who"], s["text"], s.get("lang", "en"))

    async def _do_media_say(self, s: dict[str, Any]) -> None:
        await self._hear(None, s["text"], s.get("lang", "en"), source="media")

    async def _do_enter(self, s: dict[str, Any]) -> None:
        self.participants[s["who"]] = {"name": None, "voice": None, "seat": s.get("seat"), "conf": 0.4, "guest": True}
        ev = await self._emit(MT.SPEAKERS, {"t": self.clock, "faces": [], "face_count": len(self.participants),
                                            "screen_talking": False}, "sensors.video")
        self.banner = "Guest in the room · local docs hidden"
        await self._log("policy", "a face with no check-in: docs leave the screen, nothing is shared", ev)
        await self._screen_changed(ev)

    async def _do_leave(self, s: dict[str, Any]) -> None:
        self.participants.pop(s["who"], None)
        if not self.guests_present:
            self.banner = None
            await self._log("policy", "guest left: local docs available again")
        await self._screen_changed(None)

    async def _do_scene(self, s: dict[str, Any]) -> None:
        self.scene_mode = s["mode"]
        ev = await self._emit(MT.SCENE, {"t": self.clock, "mode": s["mode"], "conf": 0.9, "by": "decider",
                                         "text": f"camera: {s['mode']}"}, "sensors.video")
        if s["mode"] in ("slides", "talk"):
            scene = "slides" if s["mode"] == "slides" else "talk"
            prop = await self._emit(MT.PROPOSAL, {
                "by": "decider", "mode": "act", "kind": "room", "action": "set_light", "args": {"scene": scene},
                "confidence": 0.93, "probabilities": {"slides": 0.93, "talk": 0.04, "board": 0.02, "break": 0.01},
                "reason": f"camera scene is '{s['mode']}' (93%): lamp scene {scene}", "utterance_id": None,
                "escalated_from": None, "route": "local-decider", "model": MODEL_LABELS["decider"],
                "request": {"state": f"Room camera, one frame: {'a projected slide fills the wall, people face it' if scene == 'slides' else 'people face each other, no projection'}",
                            "questions": {"mode": {"type": "choice", "instructions": "What is the room doing?",
                                                   "criteria": {"slides": "a presentation is on", "talk": "a conversation",
                                                                "board": "someone writes at the board", "break": "nobody is talking"}}}},
                "answers": {"mode": {"type": "choice", "choice": scene, "confidence": 0.93,
                                     "probabilities": {"slides": 0.93, "talk": 0.04, "board": 0.02, "break": 0.01}}},
                "threshold": self.settings.act_threshold, "latency_ms": LATENCY_MS["decider"], "tokens": None}, "agent.decider", ev)
            ver = await self._emit(MT.VERDICT, {"proposal_id": prop.id, "action": "set_light", "args": {"scene": scene},
                                                "verdict": "act", "risk": "low", "rule": "low risk, confidence above threshold"},
                                   "agent.policy", prop)
            await self._device("lamp", "scene", {"scene": scene}, ver)

    async def _do_chat(self, s: dict[str, Any]) -> None:
        args = dict(s.get("args") or {})
        if args.get("item_id") == "$last_task":
            args["item_id"] = self._last_task
        await self.chat({"command": s["command"], "user": s.get("user", "?"), "args": args, "text": s.get("text", "")})

    async def _do_clock(self, s: dict[str, Any]) -> None:
        self.agenda["minutes_left"] = s["minutes_left"]
        if s["minutes_left"] <= 2:
            unowned = [i for i in self._items.values() if i["kind"] == "task" and not i["owner"] and i["status"] == "open"]
            self.banner = f"{s['minutes_left']} min left." + (f" {len(unowned)} task has no owner." if len(unowned) == 1 else
                                                              f" {len(unowned)} tasks have no owner." if unowned else "")
        await self._screen_changed(None)

    async def _do_end(self, s: dict[str, Any]) -> None:
        self.status["state"] = "ended"
        self.banner = None
        ev = await self._emit(MT.STATUS, dict(self.status), "runtime")
        prop = await self._emit(MT.PROPOSAL, {
            "by": "reflex", "mode": "act", "kind": "none", "action": "post_recap", "args": {}, "confidence": 1.0,
            "probabilities": {"post_recap": 1.0}, "reason": "meeting ended: offer the recap", "utterance_id": None,
            "escalated_from": None, "route": "rule", "model": "rule: meeting_end", "request": {}, "answers": {},
            "threshold": self.settings.act_threshold, "latency_ms": 0.6, "tokens": None}, "agent.reflex", ev)
        await self._policy(prop, {"name": self.status.get("owner")})

    async def _do_note(self, s: dict[str, Any]) -> None:
        await self._log("scenario", s["text"])

    # ------------------------------------------------------------------ runner
    async def _run(self, sc: dict[str, Any]) -> None:
        passed = failed = 0
        sid = sc["id"]

        async def status(state: str, i: int, text: str) -> None:
            await self._emit(MT.SCENARIO, {"scenario_id": sid, "state": state, "step_index": i, "text": text,
                                           "passed": passed, "failed": failed}, "scenario")

        await status("started", -1, sc["title"])
        for i, st in enumerate(sc["steps"]):
            do = st["do"]
            await status("step", i, st.get("text") or do)
            if do == "wait":
                await self._sleep(float(st.get("seconds", 1)))
            elif do == "expect":
                ok = await self._expect(st)
                passed, failed = passed + ok, failed + (not ok)
                await status("expect_pass" if ok else "expect_fail", i, _describe_expect(st))
            else:
                handler = getattr(self, f"_do_{do}", None)
                if handler is None:
                    raise ValueError(f"unknown step {do!r}")
                await handler(st)
                await self._sleep(0.0 if self.fast else 0.4)
        await self.bus.drain()
        await status("finished", len(sc["steps"]), f"{passed} passed, {failed} failed")
        self._task = None

    async def _expect(self, st: dict[str, Any]) -> bool:
        deadline = self.clock + float(st.get("within", 5))
        while True:
            await self.bus.drain()
            if self._check(st):
                return True
            if self.fast or self.clock >= deadline:
                return self._check(st)
            await self._sleep(0.25)

    def _check(self, st: dict[str, Any]) -> bool:
        what = st["what"]
        if what == "item":
            return any(i["kind"] == st.get("kind", i["kind"]) and (st.get("owner") is None or i["owner"] == st["owner"])
                       and st.get("text_contains", "").lower() in i["text"].lower() for i in self._items.values())
        if what == "privacy":
            return self.privacy["state"] == st["state"]
        if what == "device":
            dev = self.devices.get(st["device_id"])
            return bool(dev) and all(dev["state"].get(k) == v for k, v in (st.get("state") or {}).items())
        if what == "post":
            return any(p["channel"] == st.get("channel", p["channel"]) and st.get("text_contains", "") in p["text"] for p in self.posts)
        if what == "no_post":
            return not self.posts
        if what == "proposal":
            return any(all(e.data.get(k) == st[k] for k in ("by", "action", "route") if k in st)
                       for e in self.bus.history if e.topic == MT.PROPOSAL)
        if what == "screen":
            return bool(self.banner) and st.get("banner_contains", "") in (self.banner or "")
        raise ValueError(f"unknown expectation {what!r}")


def _describe_expect(st: dict[str, Any]) -> str:
    bits = [f"{k}={v}" for k, v in st.items() if k not in ("do", "what", "within")]
    return f"expect {st['what']}: " + ", ".join(bits)


# --------------------------------------------------------------------------- the 3-minute story, in beats
def _beat_checkins() -> list[dict[str, Any]]:
    return [
        {"do": "start", "title": "Pricing sync", "owner": "Priya", "agenda": ["Q3 pricing", "Launch date", "Open questions"],
         "minutes": 30, "ends_at": "15:30"},
        {"do": "note", "text": "No wake word. LocoCisco hears the room, sees who talks, and writes named notes."},
        {"do": "join", "who": "priya", "seat": "s1", "voice": "voice-A", "checkin": "Priya"},
        {"do": "join", "who": "mei", "seat": "s3", "voice": "voice-C", "checkin": "Mei"},
        {"do": "wait", "seconds": 2},
        {"do": "say", "who": "priya", "text": "Let's start with Q3 pricing. Mei, where did we land on the enterprise tier?"},
        {"do": "wait", "seconds": 4},
        {"do": "say", "who": "mei", "text": "We agreed last week to hold enterprise at 48 a seat."},
        {"do": "expect", "what": "item", "kind": "decision", "text_contains": "48", "within": 6},
        {"do": "wait", "seconds": 7},
    ]


def _beat_task() -> list[dict[str, Any]]:
    return [
        {"do": "say", "who": "priya", "text": "Sam, send the deck by Friday."},
        {"do": "expect", "what": "item", "kind": "task", "owner": "Sam", "within": 6},
        {"do": "expect", "what": "post", "channel": "tasks", "text_contains": "deck"},
        {"do": "wait", "seconds": 6},
        {"do": "say", "who": "mei", "text": "Someone should update the pricing page too."},
        {"do": "wait", "seconds": 7},
    ]


def _beat_room() -> list[dict[str, Any]]:
    return [
        {"do": "say", "who": "mei", "text": "It's freezing in here."},
        {"do": "expect", "what": "device", "device_id": "fan", "state": {"on": False}, "within": 6},
        {"do": "wait", "seconds": 7},
    ]


def _beat_late_joiner() -> list[dict[str, Any]]:
    return [
        {"do": "join", "who": "sam", "seat": "s2", "voice": "voice-B", "checkin": None},
        {"do": "wait", "seconds": 2},
        {"do": "say", "who": "sam", "text": "Sorry I'm late. What did I miss?"},
        {"do": "wait", "seconds": 4},
        {"do": "chat", "command": "here", "user": "Sam", "args": {"name": "Sam"}},
        {"do": "wait", "seconds": 5},
        {"do": "say", "who": "sam", "text": "Let me share my slides."},
        {"do": "scene", "mode": "slides"},
        {"do": "expect", "what": "device", "device_id": "lamp", "state": {"scene": "slides"}, "within": 6},
        {"do": "wait", "seconds": 6},
    ]


def _beat_answer() -> list[dict[str, Any]]:
    return [
        {"do": "say", "who": "priya", "text": "What did we decide about the launch date last month?"},
        {"do": "expect", "what": "proposal", "by": "llm", "action": "answer_question", "route": "local-llm", "within": 8},
        {"do": "wait", "seconds": 10},
    ]


def _beat_guest() -> list[dict[str, Any]]:
    return [
        {"do": "enter", "who": "visitor", "seat": "s4"},
        {"do": "wait", "seconds": 3},
        {"do": "say", "who": "visitor", "text": "Hi all, I'm from the vendor side, just listening in."},
        {"do": "wait", "seconds": 4},
        {"do": "say", "who": "priya", "text": "Can we pull up the margin sheet?"},
        {"do": "expect", "what": "screen", "banner_contains": "docs hidden", "within": 6},
        {"do": "wait", "seconds": 5},
        {"do": "leave", "who": "visitor"},
        {"do": "scene", "mode": "talk"},
        {"do": "wait", "seconds": 3},
    ]


def _beat_off_record() -> list[dict[str, Any]]:
    return [
        {"do": "say", "who": "mei", "text": "Off the record for a second, I think Alex is about to resign."},
        {"do": "expect", "what": "privacy", "state": "off_record", "within": 4},
        {"do": "wait", "seconds": 4},
        {"do": "say", "who": "priya", "text": "Noted. Nothing in the notes about that."},
        {"do": "wait", "seconds": 7},
        {"do": "say", "who": "priya", "text": "OK, back on the record."},
        {"do": "expect", "what": "privacy", "state": "on", "within": 4},
        {"do": "wait", "seconds": 4},
    ]


def _beat_wrap() -> list[dict[str, Any]]:
    return [
        {"do": "clock", "minutes_left": 2},
        {"do": "expect", "what": "screen", "banner_contains": "2 min left", "within": 3},
        {"do": "wait", "seconds": 5},
        {"do": "say", "who": "priya", "text": "Let's wrap. Decision: enterprise stays at 48, Sam owns the deck."},
        {"do": "wait", "seconds": 5},
        {"do": "end"},
        {"do": "wait", "seconds": 5},
        {"do": "chat", "command": "approve", "user": "Priya", "args": {}},
        {"do": "expect", "what": "post", "channel": "notes", "text_contains": "Recap", "within": 4},
        {"do": "note", "text": "Audio out 0 B. Video out 0 B. Only text reached Discord, and the owner approved it."},
        {"do": "wait", "seconds": 4},
    ]


def build_scenarios() -> list[dict[str, Any]]:
    beats = [
        ("beat_checkins", "Check-ins and a decision", "Two people check in; the decider logs a decision.", _beat_checkins),
        ("beat_task", "A task with an owner and a due date", "'Sam, send the deck by Friday' becomes a Discord task.", _beat_task),
        ("beat_room", "The room", "'It's freezing' turns the fan off.", _beat_room),
        ("beat_late_joiner", "Late joiner and slides", "Sam is a Guest until /here; slides dim the lamp.", _beat_late_joiner),
        ("beat_answer", "Answer from local docs", "The decider is unsure, the local LLM reads the docs.", _beat_answer),
        ("beat_guest", "A guest enters", "Docs hide, sharing is denied, the guest leaves.", _beat_guest),
        ("beat_off_record", "Off the record", "Notes pause, the last minute is wiped, then resume.", _beat_off_record),
        ("beat_wrap", "Two minutes left and the recap", "Banner, final decision, owner-approved recap.", _beat_wrap),
    ]
    story: list[dict[str, Any]] = []
    for _, _, _, fn in beats:
        story += fn()
    out = [{"id": "demo_story", "title": "The 3-minute demo", "blurb": "Every beat, in order, about three minutes at 1x.",
            "tags": ["demo"], "steps": story}]
    for sid, title, blurb, fn in beats:
        steps = fn() if sid == "beat_checkins" else _beat_checkins()[:4] + fn()
        out.append({"id": sid, "title": title, "blurb": blurb, "tags": ["beat"], "steps": steps})
    return out


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    from .server import create_app, require_loopback

    ap = argparse.ArgumentParser(description="Serve the room screen on a fake meeting runtime (loopback only).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18751)
    ap.add_argument("--speed", type=float, default=1.0, help="scenario playback speed")
    ap.add_argument("--autoplay", default=None, help="scenario id to start right away (e.g. demo_story)")
    args = ap.parse_args(argv)
    require_loopback(args.host)
    runtime = FakeMeetingRuntime(MeetingSettings(speed=args.speed))
    app = create_app(runtime)
    if args.autoplay:
        if args.autoplay not in {s["id"] for s in runtime.scenarios()}:
            ap.error(f"unknown scenario {args.autoplay!r}")

        @app.on_event("startup")
        async def _autoplay() -> None:
            await runtime.run_scenario(args.autoplay)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
