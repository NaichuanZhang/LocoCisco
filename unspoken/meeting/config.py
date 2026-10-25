"""config/meeting.toml -> one plain dict, plus MeetingSettings built from it.

Everything the meeting brain can be tuned with lives in that file (thresholds, wipe window, device ids,
Discord channel keys, docs folder). Secrets never do. `load_config()` tolerates a missing file so tests
and other workstreams can run with the defaults below.
"""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

from .contracts import MeetingSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "meeting.toml"

DEFAULTS: dict[str, Any] = {
    "meeting": {
        "title": "Pricing sync",
        "owner": "Priya",
        "docs_dir": "examples/docs",
        "act_threshold": 0.60,
        "ask_threshold": 0.40,
        "escalate": True,
        "wipe_seconds": 60.0,
        "caption_lines": 6,
        "owner_threshold": 0.50,
        "approval_ttl_seconds": 300.0,
        "catch_up_after_seconds": 120.0,
        "absent_after_seconds": 3.0,
        "clock_tick_seconds": 10.0,
        "banner_minutes": 2.0,
    },
    "devices": {"lamp": "lamp", "fan": "fan"},
    "channels": {"tasks": "tasks", "notes": "notes"},
    "scenes": {"start": "talk", "slides": "slides", "board": "talk", "break": "break",
               "off_record": "off_record", "paused": "privacy"},
    "screen": {"host": "127.0.0.1", "port": 8780},
    "questions": {"version": "v3"},
}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Defaults overlaid with config/meeting.toml (or `path`). A missing file is not an error."""
    p = Path(path) if path else CONFIG_PATH
    raw: dict[str, Any] = {}
    if p.exists():
        with open(p, "rb") as f:
            raw = tomllib.load(f)
    cfg = _merge(DEFAULTS, raw)
    docs = Path(cfg["meeting"]["docs_dir"])
    cfg["meeting"]["docs_dir"] = str(docs if docs.is_absolute() else REPO_ROOT / docs)
    return cfg


def settings_from(cfg: dict[str, Any], **changes: Any) -> MeetingSettings:
    m = cfg["meeting"]
    s = MeetingSettings(act_threshold=float(m["act_threshold"]), ask_threshold=float(m["ask_threshold"]),
                        escalate=bool(m["escalate"]), wipe_seconds=float(m["wipe_seconds"]),
                        caption_lines=int(m["caption_lines"]))
    for k, v in changes.items():
        if not hasattr(s, k):
            raise ValueError(f"unknown setting {k!r}")
        setattr(s, k, v)
    return s
