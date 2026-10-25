"""Read-only lookup over home.json (rooms, devices, doors, residents, pets) shared by the scenario tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HOME_PATH = Path(__file__).resolve().parent.parent / "home.json"

# Devices whose effect is not tied to the room they hang in; an "outcome" expect ignores `room` for them.
HOUSE_WIDE_KINDS: frozenset[str] = frozenset({"thermostat"})
# Pseudo device the actuator addresses for parent alerts (not in home.json).
PHONE_DEVICE_PREFIX = "phone"


def load_home(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or HOME_PATH, encoding="utf-8") as fh:
        return json.load(fh)


class HomeIndex:
    def __init__(self, home: dict[str, Any]) -> None:
        self.raw = home
        self.rooms: dict[str, dict[str, Any]] = {r["id"]: r for r in home.get("rooms", [])}
        self.devices: dict[str, dict[str, Any]] = {d["device_id"]: d for d in home.get("devices", [])}
        self.doors: dict[str, dict[str, Any]] = {d["door_id"]: d for d in home.get("doors", [])}
        self.residents: dict[str, dict[str, Any]] = {r["id"]: r for r in home.get("residents", [])}
        self.pets: dict[str, dict[str, Any]] = {p["id"]: p for p in home.get("pets", [])}
        self.thresholds: dict[str, Any] = dict(home.get("thresholds", {}))

    def name(self, who: str) -> str:
        for table in (self.residents, self.pets):
            if who in table:
                return str(table[who].get("name", who))
        return who.capitalize()

    def room_name(self, room: str) -> str:
        return str(self.rooms.get(room, {}).get("name", room))

    def device_kind(self, device_id: str) -> str | None:
        d = self.devices.get(device_id)
        if d is not None:
            return str(d.get("kind"))
        if device_id.startswith(PHONE_DEVICE_PREFIX):
            return "phone"
        return None

    def device_room(self, device_id: str) -> str | None:
        d = self.devices.get(device_id)
        return None if d is None else d.get("room")

    def devices_of_kind(self, kind: str, room: str | None = None) -> list[str]:
        return [i for i, d in self.devices.items() if d.get("kind") == kind and (room is None or d.get("room") == room)]
