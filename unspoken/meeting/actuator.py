"""Actuator: turns `act` verdicts into device commands, items, Discord posts and screen cards; turns
`ask_owner` verdicts into a DM to the meeting owner and executes on approve. Chat `undo` removes a task,
`approve` / `deny` answer a pending ask, `recap` posts the recap, `ask` answers from the local docs.

Everything that leaves the box goes through the Notifier wrapped in the egress meter (fakes.MeteredNotifier),
which publishes egress.meter after every post; audio and video bytes out are 0 by construction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..core.events import Event
from .base import Agent, MeetingState, new_id
from .contracts import MT, ItemData
from .dates import humanize
from .docs import DocIndex, answer_from
from .recap import recap_markdown

BRIGHT, DIM = 100, 25
ITEM_KIND = {"create_task": "task", "log_decision": "decision", "answer_question": "question", "park_topic": "parking_lot"}


@dataclass
class _Pending:
    request_id: str
    verdict: dict[str, Any]
    created: float
    item_id: str | None = None


class ActuatorAgent(Agent):
    name = "actuator"

    def __init__(self, bus, state: MeetingState, settings, config, *, devices: dict[str, Any], notifier: Any,
                 docs: DocIndex, llm_factory: Any = None) -> None:
        super().__init__(bus, state, settings, config)
        self.devices = devices
        self.notifier = notifier
        self.docs = docs
        self.llm_factory = llm_factory
        self.channels = dict(config.get("channels") or {})
        self.lamp_id = str(config["devices"].get("lamp", "lamp"))
        self.fan_id = str(config["devices"].get("fan", "fan"))
        self.ttl = float(config["meeting"].get("approval_ttl_seconds", 300.0))
        self.pending: dict[str, _Pending] = {}
        self.last_task_id: str | None = None

    def start(self) -> None:
        self.subscribe(MT.VERDICT, self._on_verdict)
        self.subscribe(MT.CHAT, self._on_chat)
        self.subscribe(MT.PRIVACY, self._on_privacy)
        self.subscribe(MT.STATUS, self._on_status)

    # ------------------------------------------------------------------ verdicts
    async def _on_verdict(self, ev: Event) -> None:
        v = ev.data
        if v.get("verdict") == "act":
            await self.execute(v, ev)
        elif v.get("verdict") == "ask_owner":
            await self._ask_owner(v, ev)

    async def execute(self, v: dict[str, Any], cause: Event, *, approved_by: str | None = None) -> None:
        action, args = str(v.get("action")), dict(v.get("args") or {})
        if action == "create_task":
            await self._create_item("task", str(args.get("text") or ""), cause, v, owner=args.get("owner"), due=args.get("due"))
        elif action == "log_decision":
            await self._create_item("decision", str(args.get("text") or ""), cause, v)
        elif action == "park_topic":
            await self._create_item("parking_lot", str(args.get("text") or ""), cause, v)
        elif action == "answer_question":
            await self._answer(str(args.get("question") or ""), cause, v)
        elif action == "set_light":
            await self._light(str(args.get("scene") or "talk"), cause, v.get("proposal_id"))
        elif action == "set_fan":
            await self._device(self.fan_id, "on" if args.get("on", True) else "off", {}, cause, v.get("proposal_id"))
        elif action == "show":
            await self._command("screen", "show", {"card": args.get("card", "agenda"), **{k: x for k, x in args.items() if k != "card"}}, cause, v)
        elif action == "post_recap":
            text = await recap_markdown(self.state, self.llm_factory)
            await self._post(self.channels.get("notes", "notes"), text, cause, v, kind="recap")
        elif action == "share_docs":
            hits = await self.docs.search(str(args.get("query") or ""), k=3)
            if hits:
                body = "\n".join(f"**{h['doc']}** — {h['text'][:200].strip()}…" for h in hits)
                await self._post(self.channels.get("notes", "notes"), f"Docs for “{args.get('query')}”:\n{body}", cause, v, kind="docs")
            else:
                await self._command("screen", "show", {"card": "answer", "question": args.get("query"),
                                                       "text": "Nothing in the local docs matches."}, cause, v)
        elif action in ("pause_notes", "resume_notes", "none"):
            return                                   # the reflex agent owns the privacy state
        else:
            await self.log(f"no actuator for {action!r}", cause, "warn")

    # ------------------------------------------------------------------ items
    async def _create_item(self, kind: str, text: str, cause: Event, v: dict[str, Any], *, owner: str | None = None,
                           due: str | None = None, status: str = "open") -> ItemData:
        item: ItemData = {"item_id": new_id(kind[:4]), "kind": kind, "text": text, "owner": owner, "due": due,  # type: ignore[typeddict-item]
                          "status": status, "utterance_id": v.get("utterance_id"), "message": None}
        self.state.items[item["item_id"]] = item
        if kind == "task":
            self.last_task_id = item["item_id"]
            line = f"☐ **{text}** — {owner or 'no owner yet'}, due {humanize(due)} (task {item['item_id']})"
            channel = self.channels.get("tasks", "tasks")
        elif kind == "decision":
            line, channel = f"✅ Decision: {text}", self.channels.get("notes", "notes")
        elif kind == "parking_lot":
            line, channel = f"🅿️ Parked: {text}", self.channels.get("notes", "notes")
        else:
            line, channel = f"❓ {text}", self.channels.get("notes", "notes")
        ref = await self._post(channel, line, cause, v, kind=kind, meta={"item_id": item["item_id"]})
        if ref is not None:
            item["message"] = {"channel": ref.channel, "id": ref.id, "thread": ref.thread}
        await self.emit(MT.ITEM, dict(item), cause)
        return item

    async def _answer(self, question: str, cause: Event, v: dict[str, Any]) -> None:
        hits = await self.docs.search(question, k=3)
        answer = answer_from(hits, question)
        item = await self._create_item("question", question, cause, v)
        await self._command("screen", "show", {"card": "answer", "question": question, "text": answer,
                                               "sources": [h["doc"] for h in hits], "item_id": item["item_id"]}, cause, v)
        if hits:
            await self._post(self.channels.get("notes", "notes"), f"Q: {question}\nA: {answer}", cause, v, kind="answer",
                             thread=(item.get("message") or {}).get("id"))

    # ------------------------------------------------------------------ devices + screen
    async def _light(self, scene: str, cause: Event, verdict_id: str | None) -> None:
        if scene == "bright":
            await self._device(self.lamp_id, "set", {"brightness": BRIGHT, "scene": "bright"}, cause, verdict_id)
        elif scene == "dim":
            await self._device(self.lamp_id, "set", {"brightness": DIM, "scene": "dim"}, cause, verdict_id)
        else:
            await self._device(self.lamp_id, "scene", {"scene": scene}, cause, verdict_id)

    async def _device(self, device_id: str, command: str, args: dict[str, Any], cause: Event, verdict_id: str | None) -> None:
        dev = self.devices.get(device_id)
        await self.emit(MT.COMMAND, {"target": device_id, "command": command, "args": dict(args), "verdict_id": verdict_id or ""}, cause)
        if dev is None:
            await self.log(f"no device {device_id!r}", cause, "warn")
            return
        try:
            new_state = await dev.apply(command, args)
        except Exception as exc:
            await self.log(f"{device_id} {command} failed: {type(exc).__name__}: {exc}", cause, "warn")
            return
        env = {"device_id": device_id, "kind": getattr(dev, "kind", "light"), "state": dict(new_state)}
        self.state.devices[device_id] = env
        await self.emit(MT.ENV, env, cause)

    async def _command(self, target: str, command: str, args: dict[str, Any], cause: Event, v: dict[str, Any] | None) -> None:
        await self.emit(MT.COMMAND, {"target": target, "command": command, "args": args,
                                     "verdict_id": (v or {}).get("proposal_id") or ""}, cause)

    # ------------------------------------------------------------------ notifier + egress
    async def _post(self, channel: str, text: str, cause: Event, v: dict[str, Any] | None, *, kind: str = "note",
                    thread: str | None = None, meta: dict[str, Any] | None = None):
        await self._command("notifier", "post", {"channel": channel, "text": text, "kind": kind}, cause, v)
        try:
            ref = await self.notifier.post(channel, text, thread=thread, kind=kind, meta=meta)
        except Exception as exc:
            await self.log(f"post to #{channel} failed: {type(exc).__name__}: {exc}", cause, "warn")
            return None
        await self._egress(cause)
        return ref

    async def _dm(self, user: str, text: str, cause: Event, meta: dict[str, Any] | None = None):
        await self._command("notifier", "dm", {"user": user, "text": text}, cause, None)
        try:
            ref = await self.notifier.dm(user, text, meta=meta)
        except Exception as exc:
            await self.log(f"dm to {user} failed: {type(exc).__name__}: {exc}", cause, "warn")
            return None
        await self._egress(cause)
        return ref

    async def _egress(self, cause: Event) -> None:
        await self.emit(MT.EGRESS, dict(self.state.egress), cause)

    def on_egress(self, dest: str, kind: str, nbytes: int) -> None:
        """Callback for the egress meter (sync): remembers the running total; _egress() publishes it."""
        e = self.state.egress
        e.update(dest=dest, kind=kind, bytes=int(nbytes), total_bytes=int(e.get("total_bytes", 0)) + int(nbytes))
        e.setdefault("audio_bytes_out", 0)
        e.setdefault("video_bytes_out", 0)

    # ------------------------------------------------------------------ approvals
    async def _ask_owner(self, v: dict[str, Any], cause: Event) -> None:
        owner = self.state.status.get("owner") or "owner"
        self._expire()
        rid = new_id("req")
        action, args = str(v.get("action")), dict(v.get("args") or {})
        what = describe(action, args)
        pend = _Pending(rid, dict(v), time.monotonic())
        if action == "create_task":
            item = await self._create_item("task", str(args.get("text") or ""), cause, v, owner=args.get("owner"),
                                           due=args.get("due"), status="pending_approval")
            pend.item_id = item["item_id"]
        self.pending[rid] = pend
        await self._dm(owner, f"LocoCisco wants to {what}. Approve with ✅ or deny with ❌ (request {rid}). Why: {v.get('rule', '')}",
                       cause, meta={"request_id": rid, "action": action})

    async def _on_chat(self, ev: Event) -> None:
        cmd = ev.data.get("command")
        args = dict(ev.data.get("args") or {})
        user = str(ev.data.get("user") or "someone")
        if cmd in ("approve", "deny"):
            rid = str(args.get("request_id") or "")
            pend = self.pending.pop(rid, None) if rid else (self.pending.popitem()[1] if self.pending else None)
            if pend is None:
                await self.log(f"{cmd} from {user}: no pending request", ev, "warn")
                return
            if time.monotonic() - pend.created > self.ttl:
                await self.log(f"request {pend.request_id} expired before {user}'s {cmd}", ev, "warn")
                return
            if cmd == "approve":
                if pend.item_id and pend.item_id in self.state.items:
                    item = self.state.items[pend.item_id]
                    item["status"] = "open"
                    await self.emit(MT.ITEM, dict(item), ev)
                else:
                    await self.execute(pend.verdict, ev, approved_by=user)
                await self.log(f"{user} approved {describe(str(pend.verdict.get('action')), dict(pend.verdict.get('args') or {}))}", ev)
            else:
                if pend.item_id and pend.item_id in self.state.items:
                    self.state.items[pend.item_id]["status"] = "undone"
                    await self.emit(MT.ITEM, dict(self.state.items[pend.item_id]), ev)
                await self.log(f"{user} denied {describe(str(pend.verdict.get('action')), dict(pend.verdict.get('args') or {}))}", ev)
        elif cmd == "undo":
            item_id = str(args.get("item_id") or "")
            if item_id in ("", "$last_task"):
                item_id = self.last_task_id or ""
            item = self.state.items.get(item_id)
            if item is None:
                await self.log(f"undo from {user}: no item {item_id!r}", ev, "warn")
                return
            item["status"] = "undone"
            await self._post(self.channels.get("tasks" if item.get("kind") == "task" else "notes", "notes"),
                             f"↩️ {user} removed: {item.get('text')}", ev, None, kind="undo", meta={"item_id": item_id})
            await self.emit(MT.ITEM, dict(item), ev)
        elif cmd == "recap":
            text = await recap_markdown(self.state, self.llm_factory)
            await self._post(self.channels.get("notes", "notes"), text, ev, None, kind="recap")
        elif cmd == "ask":
            question = str(ev.data.get("text") or args.get("text") or "")
            if question:
                await self._answer(question, ev, {"utterance_id": None, "proposal_id": ev.id})

    def _expire(self) -> None:
        now = time.monotonic()
        for rid in [r for r, p in self.pending.items() if now - p.created > self.ttl]:
            del self.pending[rid]

    # ------------------------------------------------------------------ privacy -> lamp
    async def _on_privacy(self, ev: Event) -> None:
        state = ev.data.get("state")
        scenes = self.config["scenes"]
        if state == "off_record":
            scene = scenes.get("off_record", "off_record")
        elif state == "paused":
            scene = scenes.get("paused", "privacy")
        else:
            scene = scenes.get(self.state.scene_mode, "talk")
        await self._device(self.lamp_id, "scene", {"scene": scene}, ev, None)

    async def _on_status(self, ev: Event) -> None:
        if ev.data.get("state") == "started":
            self.pending.clear()
            self.last_task_id = None
            self.state.egress.update(bytes=0, total_bytes=0, audio_bytes_out=0, video_bytes_out=0)
            await self._device(self.lamp_id, "scene", {"scene": self.config["scenes"].get("start", "talk")}, ev, None)
        elif ev.data.get("state") == "ended":
            await self._device(self.lamp_id, "scene", {"scene": "break"}, ev, None)


def describe(action: str, args: dict[str, Any]) -> str:
    if action == "create_task":
        return f"post a task: “{args.get('text', '')}” ({args.get('owner') or 'no owner'}, due {humanize(args.get('due'))})"
    if action == "log_decision":
        return f"log a decision: “{args.get('text', '')}”"
    if action == "post_recap":
        return "post the recap to the channel"
    if action == "share_docs":
        return f"share docs about “{args.get('query', '')}”"
    if action == "resume_notes":
        return "resume the notes"
    return f"{action} {args}".strip()
