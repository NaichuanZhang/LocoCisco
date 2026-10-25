"""Actuator: turns allow verdicts into action.command(s) resolved via home.json, turns ask_* verdicts
into a friendly notify.ask on a phone, and executes the stored action when the notify.reply is Yes.
Asks expire after ask_ttl_s (60 s); a late or unknown reply only logs."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from ..contracts import Event, T
from .base import ACTION_WORDS, Agent, HomeConfig

DIM_BRIGHTNESS = 30
THERMOSTAT_STEP_C = 1.5


@dataclass
class _PendingAsk:
    verdict_id: str
    action: str
    target: str | None
    room: str | None
    to: str
    created: float


class ActuatorAgent(Agent):
    name = "actuator"

    def __init__(self, bus, home: HomeConfig, settings, ask_ttl_s: float = 60.0,
                 alert_cooldown_s: float = 60.0) -> None:
        super().__init__(bus, home, settings)
        self.ask_ttl_s = ask_ttl_s
        self.alert_cooldown_s = alert_cooldown_s   # one parent alert per room per minute, not a stream
        self._last_alert: dict[str, float] = {}
        self.pending: dict[str, _PendingAsk] = {}

    def start(self) -> None:
        self.subscribe(T.VERDICT, self._on_verdict)
        self.subscribe(T.REPLY, self._on_reply)

    async def _on_verdict(self, ev: Event) -> None:
        v = ev.data
        verdict = v.get("verdict")
        if verdict == "allow":
            await self._execute(v.get("action", "none"), v.get("target"), v.get("room"), ev, verdict_id=ev.id)
        elif verdict in ("ask_speaker", "ask_parent"):
            await self._ask(v, ev)

    async def _on_reply(self, ev: Event) -> None:
        rid = str(ev.data.get("request_id") or "")
        answer = str(ev.data.get("answer") or "").strip().lower()
        pend = self.pending.pop(rid, None)
        if pend is None:
            await self.log(f"reply to an unknown or expired ask ({rid or 'no id'}); nothing done", ev, "warn")
            return
        if time.monotonic() - pend.created > self.ask_ttl_s:
            await self.log(f"the ask about '{ACTION_WORDS.get(pend.action, pend.action)}' expired before the reply; nothing done", ev)
            return
        if answer == "yes":
            await self._execute(pend.action, pend.target, pend.room, ev, verdict_id=pend.verdict_id)
        else:
            await self.log(f"{ev.data.get('by') or pend.to} said {ev.data.get('answer') or 'no'}: "
                           f"not going to {ACTION_WORDS.get(pend.action, pend.action)}", ev)

    # ------------------------------------------------------------------ commands
    def commands(self, action: str, target: str | None, room: str | None) -> list[dict[str, Any]]:
        """CommandData (without verdict_id) for one action, resolved through home.json."""
        if action in ("lights_on", "lights_off", "lights_dim"):
            lights = [target] if target in self.home.devices else self.home.lights_in(room)
            cmd = {"lights_on": ("on", {"brightness": 100}), "lights_off": ("off", {}),
                   "lights_dim": ("dim", {"brightness": DIM_BRIGHTNESS})}[action]
            return [{"device_id": light, "command": cmd[0], "args": dict(cmd[1])} for light in lights]
        if action in ("tv_on", "tv_off"):
            tv = target if target in self.home.devices else self.home.tv_for(room)
            return [{"device_id": tv, "command": "on" if action == "tv_on" else "off", "args": {}}] if tv else []
        if action in ("warmer", "cooler"):
            thermostat = target if target in self.home.devices else self.home.thermostat()
            delta = THERMOSTAT_STEP_C if action == "warmer" else -THERMOSTAT_STEP_C
            return [{"device_id": thermostat, "command": "set_setpoint", "args": {"delta_c": delta}}] if thermostat else []
        if action in ("unlock_door", "lock_door"):
            lock = target if target in self.home.devices else self.home.lock_for(room)
            return [{"device_id": lock, "command": "unlock" if action == "unlock_door" else "lock", "args": {}}] if lock else []
        if action == "alert_parent":
            where = self.home.room_name_in_sentence(room) if room else "the house"
            return [{"device_id": "phone_parents", "command": "notify",
                     "args": {"to": "parents", "text": f"Safety alert: someone may be in danger in the {where}. Please check now."}}]
        return []

    async def _execute(self, action: str, target: str | None, room: str | None, cause: Event, *, verdict_id: str) -> None:
        if action == "alert_parent":
            last = self._last_alert.get(room or "")
            if last is not None and time.monotonic() - last < self.alert_cooldown_s:
                await self.log(f"parents were already alerted about {self.home.room_name(room)} just now", cause)
                return
            self._last_alert[room or ""] = time.monotonic()
        cmds = self.commands(action, target, room)
        if not cmds:
            await self.log(f"no device for '{action}' in {self.home.room_name(room)}", cause, "warn")
            return
        for cmd in cmds:
            await self.emit(T.COMMAND, {**cmd, "verdict_id": verdict_id}, cause)

    # ------------------------------------------------------------------ asks
    def ask_text(self, action: str, room: str | None, verdict: str, who: dict | None) -> str:
        room_words = self.home.room_name_in_sentence(room) if room else "the"
        texts = {
            "lights_on": f"Turn on the {room_words} lights?", "lights_off": f"Turn off the {room_words} lights?",
            "lights_dim": f"Dim the {room_words} lights?", "warmer": "Make it warmer?", "cooler": "Make it cooler?",
            "tv_on": "Turn on the TV?", "tv_off": "Turn off the TV?",
            "unlock_door": "Unlock the front door?", "lock_door": "Lock the front door?", "alert_parent": "Alert a parent?",
        }
        name = (who or {}).get("name")
        if verdict == "ask_parent" and name:
            return f"{name} wants to {ACTION_WORDS.get(action, action)}. OK?"
        return texts.get(action, f"{ACTION_WORDS.get(action, action).capitalize()}?")

    async def _ask(self, v: dict[str, Any], ev: Event) -> None:
        who = v.get("who") or {}
        to = "parents" if v["verdict"] == "ask_parent" or not who.get("id") else str(who["id"])
        action, target, room = v.get("action", "none"), v.get("target"), v.get("room")
        self._expire()
        rid = f"ask-{uuid.uuid4().hex[:8]}"
        self.pending[rid] = _PendingAsk(ev.id, action, target, room, to, time.monotonic())
        await self.emit(T.ASK, {
            "request_id": rid, "to": to, "text": self.ask_text(action, room, v["verdict"], who),
            "options": ["Yes", "No"], "proposal_id": v.get("proposal_id"),
            "action": action, "target": target, "room": room,
        }, ev)

    def _expire(self) -> None:
        now = time.monotonic()
        for rid in [r for r, p in self.pending.items() if now - p.created > self.ask_ttl_s]:
            del self.pending[rid]
