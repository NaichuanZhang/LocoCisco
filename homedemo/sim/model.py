"""Static shape of the home (parsed from home.json), time helpers, the daylight model and geometry."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HOME_PATH = Path(__file__).resolve().parents[1] / "home.json"
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
SCHOOL_NIGHTS = frozenset({"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"})
DAY = 86_400

LIGHT_LUX_FULL = 300.0       # one light at brightness 100, measured in its room
OUTDOOR_PEAK_LUX = 10_000.0  # clear sky at solar noon
WINDOW_FACTOR = 0.08         # share of outdoor daylight that reaches an indoor room
NIGHT_LUX_INDOOR = 2.0       # standby LEDs and street light through the curtains
NIGHT_LUX_OUTDOOR = 1.0
OUTDOOR_CELSIUS = 12.0       # porch temperature until a scenario sets it


def load_home(path: str | Path | None = None) -> dict[str, Any]:
    """The home configuration (homedemo/home.json unless a path is given)."""
    with open(path or HOME_PATH, encoding="utf-8") as f:
        return json.load(f)


def parse_hhmm(text: Any) -> int:
    """'23:40' -> seconds since midnight."""
    try:
        hours, minutes = str(text).split(":")
        h, m = int(hours), int(minutes)
    except ValueError:
        raise ValueError(f"bad time {text!r}, want 'HH:MM'") from None
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"bad time {text!r}, want 'HH:MM'")
    return h * 3600 + m * 60


def fmt_hhmm(day_seconds: float) -> str:
    s = int(day_seconds) % DAY
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"


def daylight(day_seconds: float, indoor: bool, sunrise: int, sunset: int) -> float:
    """Sunlight alone: a half sine between sunrise and sunset, a small floor at night."""
    floor = NIGHT_LUX_INDOOR if indoor else NIGHT_LUX_OUTDOOR
    if not sunrise < day_seconds < sunset:
        return floor
    sun = math.sin(math.pi * (day_seconds - sunrise) / (sunset - sunrise))
    return max(floor, OUTDOOR_PEAK_LUX * sun * (WINDOW_FACTOR if indoor else 1.0))


def bearing(x0: float, y0: float, x1: float, y1: float) -> float:
    """Direction from (x0, y0) to (x1, y1) in canvas degrees: 0 = +x (right), 90 = +y (down)."""
    return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 360


def angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def distance(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.hypot(x1 - x0, y1 - y0)


@dataclass
class Room:
    id: str
    name: str
    x: float
    y: float
    w: float
    h: float
    indoor: bool = True
    lux: float = 0.0
    celsius: float = 19.0
    ambient_override: float | None = None  # set_env pins the daylight term until the next set_time/reset

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


@dataclass
class Device:
    device_id: str
    kind: str
    room: str
    x: float
    y: float
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class Door:
    door_id: str
    name: str
    between: list[str]
    x: float
    y: float
    lock: str | None = None  # device_id of the lock on this door
    open: bool = False


@dataclass
class Mic:
    mic_id: str
    room: str
    x: float
    y: float


@dataclass
class Camera:
    camera_id: str
    room: str


@dataclass
class Entity:
    """Something that moves around: a person or a pet."""
    id: str
    name: str
    room: str | None = None
    x: float | None = None
    y: float | None = None
    activity: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Person(Entity):
    role: str = "guest"
    voice_enrolled: bool = False
    has_phone: bool = False
    langs: list[str] = field(default_factory=lambda: ["en"])
    age_group: str = "adult"
    guest: bool = True
    idle_min: float = 0.0
    home: bool = True
    phone_home: bool = True


@dataclass
class Pet(Entity):
    species: str = "cat"
