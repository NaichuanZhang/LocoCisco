"""Dev fake of MeetingRuntimeAPI for the bridge: `python -m unspoken.bridge.mcp_server --runtime fake` and the tests.
What it does: real Bus, in-memory state, a two-line stand-in policy that
turns MT.PROPOSAL into MT.VERDICT (so create_task / set_light get a verdict), and a record of every chat command.
No model servers, no hardware."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from unspoken.core.bus import Bus
from unspoken.core.events import Event
from unspoken.meeting.contracts import (MT, ChatCommandData, ItemData, ItemKind, MeetingSettings, PrivacyData,
                                        PrivacyState, UtteranceData)


class FakeMeetingRuntime:
    """Implements MeetingRuntimeAPI. `policy` maps action -> verdict for the stand-in policy ("skip" = no verdict)."""

    def __init__(self, *, policy: dict[str, str] | None = None, verdict_delay: float = 0.0) -> None:
        self.bus = Bus()
        self.settings = MeetingSettings()
        self.clock = 600.0                     # seconds on the meeting clock
        self.policy = {"create_task": "act", "set_light": "act", **(policy or {})}
        self.verdict_delay = verdict_delay
        self.chats: list[ChatCommandData] = []
        self.privacy: PrivacyData = {"state": "on", "reason": "", "by": "user", "wiped_from": None, "wiped_count": 0}
        self._items: list[ItemData] = [
            {"item_id": "t1", "kind": "task", "text": "Send the deck", "owner": "Sam", "due": "Friday",
             "status": "open", "utterance_id": "u2", "message": None},
            {"item_id": "d1", "kind": "decision", "text": "Ship pricing v2 in Q3", "owner": None, "due": None,
             "status": "open", "utterance_id": "u3", "message": None},
        ]
        self._transcript: list[UtteranceData] = [
            {"utterance_id": "u1", "segment_id": "s1", "start": 10.0, "end": 12.0, "text": "Let's start.",
             "lang": "en", "speaker": {"name": "Priya", "voice": "voice-A", "seat": "s1", "conf": 0.9, "guest": False},
             "source": "person", "context": ""},
            {"utterance_id": "u2", "segment_id": "s2", "start": 300.0, "end": 304.0,
             "text": "Sam, send the deck by Friday.", "lang": "en",
             "speaker": {"name": "Priya", "voice": "voice-A", "seat": "s1", "conf": 0.9, "guest": False},
             "source": "person", "context": ""},
            {"utterance_id": "u3", "segment_id": "s3", "start": 580.0, "end": 583.0, "text": "Agreed, v2 in Q3.",
             "lang": "en", "speaker": {"name": None, "voice": "voice-B", "seat": "s2", "conf": 0.6, "guest": True},
             "source": "person", "context": ""},
        ]
        self.bus.subscribe(MT.PROPOSAL, self._policy, name="fake.policy")

    # ------------------------------------------------------------------ stand-in policy
    async def _policy(self, event: Event) -> None:
        verdict = self.policy.get(event.data.get("action", ""), "deny")
        if verdict == "skip":
            return
        if self.verdict_delay:
            await asyncio.sleep(self.verdict_delay)
        risk = "medium" if event.data.get("action") == "create_task" else "low"
        await self.bus.publish(Event(MT.VERDICT, {"proposal_id": event.id, "action": event.data.get("action"),
                                                  "args": event.data.get("args", {}), "verdict": verdict,
                                                  "risk": risk, "rule": f"fake policy: {verdict}"},
                                     source="fake.policy", ts=event.ts, cause=event.id))

    # ------------------------------------------------------------------ MeetingRuntimeAPI
    def snapshot(self) -> dict[str, Any]:
        return {
            "status": {"state": "started", "title": "Pricing sync", "topic": "Q3 pricing", "owner": "Priya",
                       "ends_at": "15:30"},
            "privacy": dict(self.privacy),
            "participants": [{"name": "Priya", "voice": "voice-A", "seat": "s1", "conf": 0.9, "guest": False},
                             {"name": None, "voice": "voice-B", "seat": "s2", "conf": 0.6, "guest": True}],
            "screen": {"privacy": self.privacy["state"], "captions": [], "agenda": {}, "cards": [],
                       "banner": "2 min left. 1 task has no owner."},
            "items": [dict(i) for i in self._items],
            "devices": [{"device_id": "lamp", "kind": "light", "state": {"on": True, "scene": "talk"}}],
            "egress": {"dest": "openclaw", "kind": "text", "bytes": 0, "total_bytes": 0,
                       "audio_bytes_out": 0, "video_bytes_out": 0},
            "models": [{"role": "decider", "label": "Strands Decider 2B (fake)", "ok": True}],
        }

    def transcript(self, since: float | None = None) -> list[UtteranceData]:
        if self.privacy["state"] == "off_record":
            return []
        return [u for u in self._transcript if since is None or float(u["start"]) >= since]

    def items(self, kind: ItemKind | None = None) -> list[ItemData]:
        return [i for i in self._items if kind is None or i["kind"] == kind]

    async def search_docs(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        return [{"doc": "pricing.md", "chunk": 0, "score": 0.9, "text": f"...{query}..."}][:k]

    async def recap(self) -> str:
        return "**Decisions**\n- Ship pricing v2 in Q3\n**Tasks**\n- Sam: send the deck (Friday)"

    async def chat(self, command: ChatCommandData) -> dict[str, Any]:
        self.chats.append(command)
        c = command.get("command")
        if c == "offrecord":
            self.privacy = {"state": "off_record", "reason": str(command.get("args", {}).get("reason", "")),
                            "by": "user", "wiped_from": self.clock - 60, "wiped_count": 1}
            return {"ok": True, "text": "Notes paused. The last minute is gone."}
        if c == "resume":
            self.privacy = {"state": "on", "reason": "resumed", "by": "user", "wiped_from": None, "wiped_count": 0}
            return {"ok": True, "text": "Notes back on."}
        if c == "here":
            return {"ok": True, "text": f"Welcome, {command.get('args', {}).get('name')}."}
        if c == "undo":
            iid = command.get("args", {}).get("item_id")
            for i in self._items:
                if i["item_id"] == iid:
                    i["status"] = "undone"
                    return {"ok": True, "text": f"Undid {iid}."}
            return {"ok": False, "text": f"No item {iid}."}
        if c in ("approve", "deny"):
            return {"ok": True, "text": f"{c}: {command.get('args', {}).get('request_id')}"}
        if c == "boom":
            raise RuntimeError("kaboom")
        return {"ok": True, "text": f"{c} ok"}

    async def set_privacy(self, state: PrivacyState, reason: str, by: str = "user") -> PrivacyData:
        self.privacy = {"state": state, "reason": reason, "by": by, "wiped_from": None, "wiped_count": 0}
        return self.privacy

    def scenarios(self) -> list[dict[str, Any]]:
        return []

    async def run_scenario(self, scenario_id: str) -> None:
        pass

    async def stop_scenario(self) -> None:
        pass

    async def reset(self) -> None:
        self.chats.clear()

    async def update_settings(self, **changes: Any) -> MeetingSettings:
        for k, v in changes.items():
            if not hasattr(self.settings, k):
                raise ValueError(k)
            setattr(self.settings, k, v)
        return self.settings

    def settings_dict(self) -> dict[str, Any]:
        return asdict(self.settings)
