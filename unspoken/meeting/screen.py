"""Screen state: what the room screen shows now (screen.state). Named captions (never an off-record line;
wiped lines disappear), agenda topic and minutes left, banners ("2 min left. 1 task has no owner."), and
cards: answer (from the actuator), tasks (whenever items change), catch_up for a late joiner.
The room screen page (unspoken/meeting/web, screen workstream) renders ScreenData; this agent only decides it.
"""
from __future__ import annotations

from typing import Any

from ..core.events import Event
from .base import Agent, MeetingState
from .contracts import MT, ScreenData
from .dates import humanize
from .reflex import detect

MAX_CARDS = 6


class ScreenAgent(Agent):
    name = "screen"

    def __init__(self, bus, state: MeetingState, settings, config) -> None:
        super().__init__(bus, state, settings, config)
        self.banner_minutes = float(config["meeting"].get("banner_minutes", 2.0))
        self.catch_up_after = float(config["meeting"].get("catch_up_after_seconds", 120.0))

    def start(self) -> None:
        self.subscribe(MT.UTTERANCE, self._on_utterance)
        self.subscribe(MT.PRIVACY, self._on_privacy)
        self.subscribe(MT.STATUS, self._on_status)
        self.subscribe(MT.CLOCK, self._on_clock)
        self.subscribe(MT.ITEM, self._on_item)
        self.subscribe(MT.COMMAND, self._on_command)
        self.subscribe(MT.CHAT, self._on_chat)
        self.subscribe(MT.VERDICT, self._on_verdict)

    async def _on_verdict(self, ev: Event) -> None:
        d = ev.data
        if d.get("verdict") == "deny" and d.get("action") == "share_docs":
            self._add_card({"card": "notice", "title": "Docs hidden",
                            "text": "A guest without a check-in is in the room: company docs stay off the screen and the channel."
                            if "guest" in str(d.get("rule")) else f"Docs not shared: {d.get('rule')}"})
            await self._publish(ev)
        elif d.get("verdict") == "ask_owner":
            owner = self.state.status.get("owner") or "the owner"
            what = {"post_recap": "Recap ready", "create_task": "Task", "share_docs": "Docs", "resume_notes": "Resume notes"}.get(
                str(d.get("action")), str(d.get("action")))
            self._add_card({"card": "approval", "request_id": d.get("proposal_id"),
                            "text": f"{what} · waiting for {owner}'s ✅ in Discord"})
            await self._publish(ev)

    # ------------------------------------------------------------------ inputs
    async def _on_utterance(self, ev: Event) -> None:
        d = ev.data
        text = str(d.get("text") or "")
        if self.state.privacy.get("state") != "on" or detect(text) is not None:
            return                                      # off the record / paused: nothing on the screen
        sp = d.get("speaker") or {}
        if d.get("source") == "media":
            name = "Screen"
        else:
            name = sp.get("name") or ("Guest" if sp.get("guest") else "Someone")
        caps = list(self.state.screen.get("captions") or [])
        caps.append({"name": name, "text": text, "t": float(d.get("start", ev.ts)), "utterance_id": d.get("utterance_id")})
        self.state.screen["captions"] = caps[-int(self.settings.caption_lines):]
        await self._publish(ev)

    async def _on_privacy(self, ev: Event) -> None:
        self.state.screen["privacy"] = ev.data.get("state", "on")
        wiped_from = ev.data.get("wiped_from")
        if wiped_from is not None:
            self.state.screen["captions"] = [c for c in self.state.screen.get("captions") or []
                                             if float(c.get("t", 0.0)) < float(wiped_from)]
        await self._publish(ev)

    async def _on_status(self, ev: Event) -> None:
        d = ev.data
        st = d.get("state")
        if st == "started":
            self.state.screen = {"privacy": "on", "captions": [], "cards": [], "banner": None,
                                 "agenda": {"topic": d.get("topic"), "topics": list(self.state.agenda),
                                            "minutes_left": self.state.minutes_left, "ends_at": d.get("ends_at"),
                                            "minutes_total": int(d["minutes"]) if d.get("minutes") is not None else None}}
        elif st == "topic":
            self.state.screen.setdefault("agenda", {})["topic"] = d.get("topic")
            self.state.screen["agenda"]["topics"] = list(self.state.agenda)
        elif st == "ended":
            self.state.screen["banner"] = "Meeting ended. Voices and faces forgotten."
            self.state.screen["captions"] = []
            self.state.screen.setdefault("agenda", {})["minutes_left"] = None
        await self._publish(ev)

    async def _on_clock(self, ev: Event) -> None:
        left = ev.data.get("minutes_left")
        agenda = self.state.screen.setdefault("agenda", {})
        agenda["minutes_left"] = left
        if ev.data.get("ends_at"):
            agenda["ends_at"] = ev.data["ends_at"]
        self.state.screen["banner"] = self.banner(left)
        await self._publish(ev)

    async def _on_item(self, ev: Event) -> None:
        item = ev.data
        if item.get("status") in ("undone", "done"):
            self.state.screen["cards"] = [c for c in self.state.screen.get("cards") or [] if c.get("item_id") != item.get("item_id")]
        elif item.get("kind") == "task":
            self._add_card({"card": "task", "item_id": item.get("item_id"), "text": item.get("text"), "owner": item.get("owner"),
                            "due": humanize(item.get("due")) if item.get("due") else None,
                            "undo": "waiting for the owner's ✅" if item.get("status") == "pending_approval" else "undo in Discord ❌"})
        elif item.get("kind") == "decision":
            self._add_card({"card": "decision", "item_id": item.get("item_id"), "text": item.get("text")})
        if self.state.minutes_left is not None and self.state.minutes_left <= self.banner_minutes:
            self.state.screen["banner"] = self.banner(self.state.minutes_left)
        await self._publish(ev)

    async def _on_command(self, ev: Event) -> None:
        d = ev.data
        args = dict(d.get("args") or {})
        if d.get("target") == "notifier" and d.get("command") == "post" and args.get("kind") in ("recap", "docs"):
            title = "Recap posted" if args.get("kind") == "recap" else "Docs shared"
            self.state.screen["cards"] = [c for c in self.state.screen.get("cards") or [] if c.get("card") != "approval"]
            self._add_card({"card": "posted", "title": title, "channel": f"#{args.get('channel')}",
                            "text": str(args.get("text") or "").split("\n")[0].replace("**", "").strip()[:120]})   # the screen shows plain text
            await self._publish(ev)
            return
        if d.get("target") != "screen":
            return
        card = str(args.pop("card", "agenda"))
        if card == "answer":
            sources = args.pop("sources", None) or []
            args["doc"] = sources[0] if sources else None
        self._add_card({"card": card, **args})
        await self._publish(ev)

    async def _on_chat(self, ev: Event) -> None:
        d = ev.data
        if d.get("command") == "here":
            if self.elapsed_seconds(ev.ts) >= self.catch_up_after and self.state.items:
                self._add_card(self.catch_up_card(str((d.get("args") or {}).get("name") or d.get("user") or "you")))
                await self._publish(ev)
        elif d.get("command") == "agenda":
            args = d.get("args") or {}
            topics = list(args.get("topics") or [])
            if topics:
                self.state.agenda = topics
                self.state.screen.setdefault("agenda", {})["topics"] = topics
            if args.get("ends_at"):
                self.state.screen.setdefault("agenda", {})["ends_at"] = args["ends_at"]
            await self._publish(ev)

    # ------------------------------------------------------------------ cards + banners
    def elapsed_seconds(self, ts: float) -> float:
        """Meeting time so far: from the planned length and minutes left when known, else the event clock."""
        total = self.state.total_minutes
        if total is not None and self.state.minutes_left is not None:
            return max(0.0, (float(total) - float(self.state.minutes_left)) * 60.0)
        return ts - (self.state.started_at or 0.0) if self.state.started_at is not None else 0.0

    def banner(self, minutes_left: Any) -> str | None:
        if minutes_left is None:
            return None
        left = float(minutes_left)
        if left > self.banner_minutes:
            return None
        parts = [f"{max(0, round(left))} min left."]
        no_owner = [t for t in self.state.open_items("task") if not t.get("owner")]
        if no_owner:
            parts.append(f"{len(no_owner)} task{'s have' if len(no_owner) > 1 else ' has'} no owner.")
        open_q = self.state.open_items("question")
        if open_q:
            parts.append(f"{len(open_q)} open question{'s' if len(open_q) > 1 else ''}.")
        return " ".join(parts)

    def _add_card(self, card: dict[str, Any]) -> None:
        """Newest first (the page shows the top 3-4). One card per item; one per kind for notices/answers."""
        same = (lambda c: c.get("item_id") == card.get("item_id")) if card.get("item_id") else (lambda c: c.get("card") == card.get("card"))
        cards = [c for c in self.state.screen.get("cards") or [] if not same(c)]
        self.state.screen["cards"] = ([card] + cards)[:MAX_CARDS]

    def catch_up_card(self, name: str) -> dict[str, Any]:
        decisions = [i["text"] for i in self.state.items.values() if i.get("kind") == "decision" and i.get("status") == "open"]
        tasks = [f"{i['text']} ({i.get('owner') or 'no owner'})" for i in self.state.items.values()
                 if i.get("kind") == "task" and i.get("status") == "open"]
        lines = [f"Decided: {t}" for t in decisions[-3:]] + [f"Task: {t}" for t in tasks[-3:]]
        return {"card": "catch_up", "for": name, "topic": self.state.status.get("topic"), "lines": lines,
                "decisions": decisions[-3:], "tasks": tasks[-3:],
                "text": f"Welcome {name}. " + (f"{len(tasks)} task{'s' if len(tasks) != 1 else ''} so far." if tasks else "No tasks yet.")}

    async def _publish(self, cause: Event) -> None:
        snap: ScreenData = {k: v for k, v in self.state.screen.items()}  # type: ignore[assignment]
        await self.emit(MT.SCREEN, dict(snap), cause)
