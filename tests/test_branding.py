from pathlib import Path
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_COPY = (
    "README.md",
    "DESCRIPTION.md",
    "docs/HACKDAY.md",
    "docs/MEETING.md",
    "docs/NEMOCLAW.md",
    "docs/PLAN.md",
    "homedemo/web/static/index.html",
    "homedemo/web/static/app.js",
    "homedemo/web/static/present.js",
    "nemoclaw/workspace/AGENTS.md",
    "nemoclaw/workspace/IDENTITY.md",
    "nemoclaw/workspace/SOUL.md",
    "nemoclaw/workspace/TOOLS.md",
    "unspoken/bridge/mcp_server.py",
    "unspoken/meeting/actuator.py",
    "unspoken/meeting/llm_agent.py",
    "unspoken/meeting/scenarios/library/demo_three_minutes.json",
    "unspoken/meeting/web/dev_fake.py",
    "unspoken/meeting/web/server.py",
    "unspoken/meeting/web/static/index.html",
    "unspoken/transcribe/live/web.py",
)


@pytest.mark.parametrize("path", PUBLIC_COPY)
def test_public_copy_uses_lococisco(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    assert "LocoCisco" in text
    assert "Unspoken" not in text
    assert "默契" not in text


def test_room_display_name():
    room = tomllib.loads((ROOT / "config/room.toml").read_text())
    assert room["room"]["name"] == "LocoCisco meeting room"


def test_technical_identifiers_remain_compatible():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["name"] == "unspoken"
    bridge = tomllib.loads((ROOT / "config/bridge.toml").read_text())
    assert bridge["notifier"]["openclaw"]["sandbox"] == "unspoken"
    assert (ROOT / "unspoken/meeting/__main__.py").is_file()
