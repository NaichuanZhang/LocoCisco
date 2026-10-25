"""config/sensors.toml: audio/video devices, seat zones, pipeline knobs. Everything has a default so the
pipeline runs with an empty dict (tests) and the TOML only overrides.

    [audio]   device = "Jabra"   alsa_device = "plughw:CARD=Jabra,DEV=0"   chunk_ms = 32
    [video]   device = 0 | "/dev/video0" | "C920"   fps = 5   width = 640   height = 360
    [speakers] hz = 5
    [seats]   s1 = [0.00, 0.20, 0.30, 0.80]        # x, y, w, h in 0..1 of the frame
              screen = [0.70, 0.00, 0.30, 0.45]    # a zone named "screen" is the room display
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "config" / "sensors.toml"

Zone = tuple[float, float, float, float]   # x, y, w, h normalised to the frame

DEFAULTS: dict[str, Any] = {
    "audio": {"device": "", "alsa_device": "default", "sample_rate": 16000, "chunk_ms": 32, "backend": "auto"},
    "video": {"device": 0, "fps": 5, "width": 640, "height": 360},
    "speakers": {"hz": 5.0, "speaking_threshold": 0.5},
    "seats": {},
}


def merge_options(spec: Any, options: dict[str, Any]) -> dict[str, Any]:
    """RoleSpec.options, then constructor kwargs; an `options={...}` kwarg is merged too (so both
    `Factory(spec)` and `Factory(options={...})` / `Factory(threshold=0.7)` work in code and tests)."""
    out: dict[str, Any] = dict(getattr(spec, "options", None) or {}) if spec is not None else {}
    extra = options.pop("options", None)
    if isinstance(extra, dict):
        out.update(extra)
    out.update(options)
    return out


def config_path() -> Path:
    env = os.environ.get("UNSPOKEN_SENSORS_CONFIG")
    return Path(env) if env else DEFAULT_PATH


def load_sensors_config(path: str | Path | None = None) -> dict[str, Any]:
    """Defaults merged with the TOML (missing file = defaults only)."""
    p = Path(path) if path else config_path()
    data: dict[str, Any] = {}
    if p.exists():
        with open(p, "rb") as f:
            data = tomllib.load(f)
    return merge_defaults(data)


def merge_defaults(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for section, defaults in DEFAULTS.items():
        merged = dict(defaults)
        merged.update(data.get(section) or {})
        out[section] = merged
    for k, v in data.items():
        out.setdefault(k, v)
    return out


def seat_zones(config: dict[str, Any] | None) -> dict[str, Zone]:
    """{seat_id: (x, y, w, h)} from config["seats"]; tolerates lists, dicts {x,y,w,h} and bad rows (skipped)."""
    zones: dict[str, Zone] = {}
    for seat, raw in ((config or {}).get("seats") or {}).items():
        try:
            if isinstance(raw, dict):
                vals = (raw["x"], raw["y"], raw["w"], raw["h"])
            else:
                vals = tuple(raw)
            x, y, w, h = (float(v) for v in vals)
        except (KeyError, TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        zones[str(seat)] = (min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0), min(w, 1.0), min(h, 1.0))
    return zones


def seat_for(bbox: tuple[float, float, float, float], zones: dict[str, Zone], min_overlap: float = 0.3) -> str | None:
    """The zone holding the face: largest share of the face box inside a zone (>= min_overlap), else the
    zone containing the face centre, else None."""
    if not zones:
        return None
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    best, best_share = None, 0.0
    for seat, (zx, zy, zw, zh) in zones.items():
        ix = max(0.0, min(x + w, zx + zw) - max(x, zx))
        iy = max(0.0, min(y + h, zy + zh) - max(y, zy))
        share = (ix * iy) / (w * h) if w > 0 and h > 0 else 0.0
        if share > best_share:
            best, best_share = seat, share
    if best is not None and best_share >= min_overlap:
        return best
    for seat, (zx, zy, zw, zh) in zones.items():
        if zx <= cx <= zx + zw and zy <= cy <= zy + zh:
            return seat
    return None
