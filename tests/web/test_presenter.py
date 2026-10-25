"""Static checks for the video-ready UI: the Decider X-ray panel, the scoreboard, presenter mode and the code
card (homedemo/web/static/{index.html,xray.js,present.js,style.css}). No browser is needed: these guard the
page scaffold the scripts rely on, the one project-name constant, the labels-from-payload rule (no
hard-coded escalation wording), and that the plain scripts parse."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from homedemo.web import server as web

STATIC = Path(web.__file__).resolve().parent / "static"
INDEX = (STATIC / "index.html").read_text(encoding="utf-8")
APP = (STATIC / "app.js").read_text(encoding="utf-8")
XRAY = (STATIC / "xray.js").read_text(encoding="utf-8")
PRESENT = (STATIC / "present.js").read_text(encoding="utf-8")
STYLE = (STATIC / "style.css").read_text(encoding="utf-8")
OVERLAYS = (STATIC / "3d" / "overlays.js").read_text(encoding="utf-8")


def test_index_scaffold_for_xray_scoreboard_and_presenter():
    for needle in ('id="xray"', 'id="xr-body"', 'id="xr-route"', 'id="xr-clock"', 'id="xr-history"',
                   'id="scoreboard"', 'data-sb="decisions"', 'data-sb="local"', 'data-sb="escalated"',
                   'data-sb="avoided"', 'data-sb="kw"', 'id="present-banner"', 'data-project-name', 'data-intent',
                   'id="present-caption"', 'id="code-card"', 'id="cc-python"', 'id="cc-json"', 'id="present-toggle"',
                   'id="set-escalate"'):
        assert needle in INDEX, needle
    # the banner starts hidden (presenter mode is opt-in) and the plain scripts load after app.js, before the 3D module
    assert re.search(r'id="present-banner"[^>]*\bhidden\b', INDEX)
    order = [INDEX.index(s) for s in ('src="static/app.js"', 'src="static/xray.js"', 'src="static/present.js"',
                                      'src="static/3d/main.js"')]
    assert order == sorted(order)


def test_one_project_name_constant_and_the_intent_line():
    names = re.findall(r"const PROJECT_NAME = '([^']+)'", PRESENT)
    assert names == ["LocoCisco"]
    assert PRESENT.count("'LocoCisco'") == 1, "the project name is one constant"
    assert re.findall(r"const PROJECT_NAME_ZH = '([^']*)'", PRESENT) == [""]
    assert "el.hidden = !PROJECT_NAME_ZH" in PRESENT
    assert 'data-project-name-zh' in INDEX and 'lang="zh-Hans"' in INDEX
    assert "No wake word. A 2B Strands Decider on the hub picks the tool, so what you say stays home." in PRESENT
    assert "present=1" in PRESENT or "'present'" in PRESENT
    for key in ("'p'", "'x'", "'c'"):
        assert key in PRESENT, f"key {key} is wired"


def test_labels_come_from_the_payload_not_hard_coded_models():
    assert "window.homeLabels = labels" in APP
    for src, name in ((APP, "app.js"), (OVERLAYS, "overlays.js"), (XRAY, "xray.js")):
        assert "asking 12B" not in src and "asking the 12B…" not in src, f"{name}: hard-coded escalation wording"
        assert "'12B answer'" not in src and "'2B shadow answer'" not in src, name
    # every ProposalBy the contract knows has a default route and model name for older producers
    for by in ("decider-2b", "decider-12b", "bedrock", "fake", "keyword", "user", "llm", "reflex"):
        assert re.search(rf"['\"]?{re.escape(by)}['\"]?:\s*\{{ route:", APP), by
    # the 3D card reads the shared labels (with a stand-alone fallback) and shows Claude's reply at the hub
    assert "window.homeLabels" in OVERLAYS and "hubSays" in OVERLAYS


def test_bedrock_route_says_what_left_the_house():
    # the privacy claim on screen must match house_agent.prompt_for: Claude reads text the hub wrote
    assert "class: 'xr-privacy'" in XRAY and ".xr-privacy" in STYLE
    assert "the audio stayed home" in XRAY and "the video stayed home" in XRAY


def test_xray_covers_the_spec():
    for needle in ("renderJson", "qBadge", "'yes / no'", "latency", "threshold", "tool_calls", "reply", "judgeKeyword",
                   "renderHistory", "scoreboard", "animateNumber", "median", "prefers-reduced-motion", "escalated_from"):
        assert needle in XRAY, needle
    # presenter-mode layout: house ~62%, X-ray ~38%, debug panes hidden, large type
    assert "minmax(0, 62fr) minmax(0, 38fr)" in STYLE
    assert re.search(r"body\.present \.top,.*\.timeline-card.*\{ display: none; \}", STYLE, re.S)
    assert "@media (prefers-reduced-motion: reduce)" in STYLE
    # the code card carries the router wiring and the typed question
    assert "DeciderToolRouter(decider, threshold=0.8)" in PRESENT and "interventions=[router]" in PRESENT
    assert "'No tool'" in PRESENT and "'set_lights.room'" in PRESENT


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_plain_scripts_parse():
    for name in ("xray.js", "present.js"):
        proc = subprocess.run(["node", "--check", str(STATIC / name)], capture_output=True, timeout=30)
        assert proc.returncode == 0, f"{name}: {proc.stderr.decode(errors='replace')}"
