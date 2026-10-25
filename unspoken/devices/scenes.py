"""Light scenes: one scene name -> one WiZ "pilot" (the JSON the bulb takes in setPilot / returns from
getPilot). Scenes are OUR names (what the meeting needs), not WiZ's built-in sceneId numbers, so the
same colours show up on any bulb and a fake can check them exactly.

Pilot keys (WiZ local protocol, UDP 38899): state (bool), dimming (10..100; bulbs reject lower values),
r/g/b (0..255, colour mode) or temp (Kelvin, 2200..6500, white mode).
Reference: https://github.com/sbidy/pywizlight (protocol notes in the README and pywizlight/bulb.py).
"""
from __future__ import annotations

from typing import Any

Pilot = dict[str, Any]

DIMMING_MIN = 10          # WiZ bulbs answer "Invalid params" below 10
DIMMING_MAX = 100
TEMP_MIN = 2200
TEMP_MAX = 6500

SCENES: dict[str, Pilot] = {
    "talk":       {"state": True, "temp": 4000, "dimming": 80},                  # neutral white, people talking
    "slides":     {"state": True, "temp": 2700, "dimming": 25},                  # warm and dim so the screen reads
    "break":      {"state": True, "temp": 2200, "dimming": 50},                  # cosy warm white
    "privacy":    {"state": True, "r": 60, "g": 200, "b": 90, "dimming": 50},    # notes ON: soft green
    "paused":     {"state": True, "r": 255, "g": 160, "b": 0, "dimming": 60},    # notes paused: amber
    "off_record": {"state": True, "r": 255, "g": 0, "b": 0, "dimming": 70},      # off the record: red
    "bright":     {"state": True, "temp": 5000, "dimming": 100},
    "dim":        {"state": True, "temp": 2700, "dimming": 15},
    "off":        {"state": False},
}
SCENE_NAMES: tuple[str, ...] = tuple(SCENES)


def clamp_dimming(brightness: int | float) -> int:
    """Our 1..100 brightness -> a dimming value the bulb accepts (10..100). 0 is "off", handled by the caller."""
    return max(DIMMING_MIN, min(DIMMING_MAX, int(round(float(brightness)))))


def clamp_temp(kelvin: int | float) -> int:
    return max(TEMP_MIN, min(TEMP_MAX, int(round(float(kelvin)))))


def parse_color(color: str) -> tuple[int, int, int]:
    """'#rrggbb' or 'rrggbb' -> (r, g, b). Raises ValueError on anything else."""
    s = str(color).strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"color must be #rrggbb, got {color!r}")
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError as e:
        raise ValueError(f"color must be #rrggbb, got {color!r}") from e


def format_color(r: int, g: int, b: int) -> str:
    return f"#{int(r) & 255:02x}{int(g) & 255:02x}{int(b) & 255:02x}"


def scene_pilot(scene: str) -> Pilot:
    """A fresh copy of the scene's pilot. Raises KeyError for an unknown scene."""
    return dict(SCENES[scene])


def pilot_for_set(brightness: int | float | None = None, color: str | None = None) -> Pilot:
    """The "set" command: {"brightness": 0-100, "color": "#rrggbb"} (either may be missing).
    brightness 0 turns the light off; 1..9 become 10 (the bulb's floor)."""
    if brightness is None and color is None:
        raise ValueError("set needs brightness and/or color")
    if brightness is not None and int(round(float(brightness))) <= 0:
        return {"state": False}
    pilot: Pilot = {"state": True}
    if color is not None:
        r, g, b = parse_color(color)
        pilot.update({"r": r, "g": g, "b": b})
    if brightness is not None:
        pilot["dimming"] = clamp_dimming(brightness)
    return pilot


def pilot_to_state(pilot: Pilot) -> dict[str, Any]:
    """Bulb pilot (getPilot result or what we sent) -> the Device state dict:
    {"on", "brightness", "color" ('#rrggbb' or None), "temp" (K or None), "scene" (our name or None)}."""
    on = bool(pilot.get("state", False))
    state: dict[str, Any] = {
        "on": on,
        "brightness": int(pilot.get("dimming", 0) or 0),
        "color": None,
        "temp": None,
        "scene": None,
    }
    if "temp" in pilot and pilot["temp"] is not None:
        state["temp"] = int(pilot["temp"])
    elif any(k in pilot for k in ("r", "g", "b")):
        state["color"] = format_color(pilot.get("r", 0) or 0, pilot.get("g", 0) or 0, pilot.get("b", 0) or 0)
    state["scene"] = scene_for_pilot(pilot)
    return state


def scene_for_pilot(pilot: Pilot) -> str | None:
    """Which of OUR scenes this pilot is (exact match on the keys the scene sets), or None."""
    if not pilot.get("state", False):
        return "off"
    for name, want in SCENES.items():
        if all(pilot.get(k) == v for k, v in want.items()):
            # a white scene must not also carry colour, and vice versa, or "set" results would mislabel
            if ("temp" in want) == ("temp" in pilot and pilot.get("temp") is not None):
                return name
    return None


def merge_pilot(current: Pilot, update: Pilot) -> Pilot:
    """What the bulb holds after setPilot(update) on top of `current` (a colour update drops white mode
    and vice versa; a dimming-only update keeps the colour; "off" keeps everything but the state)."""
    out = dict(current)
    if "temp" in update:
        for k in ("r", "g", "b", "c", "w", "sceneId"):
            out.pop(k, None)
    if any(k in update for k in ("r", "g", "b")):
        out.pop("temp", None)
        out.pop("sceneId", None)
    out.update(update)
    return out
