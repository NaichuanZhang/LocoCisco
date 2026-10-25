"""Run the actual replay client through autoplay and capture setup scenarios."""
from pathlib import Path
import shutil
import subprocess

import pytest

from unspoken.meeting.record_demo import recording_page_url
from unspoken.meeting.web.server import STATIC_DIR


def test_replay_autoplay_contract():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for browser behavior checks")
    result = subprocess.run(
        [node, str(Path(__file__).with_name("replay_autoplay_contract.cjs")), str(STATIC_DIR / "replay.js")],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_recorder_disables_page_autoplay_without_losing_other_options():
    assert recording_page_url("http://127.0.0.1:8780/") == "http://127.0.0.1:8780/?autoplay=0"
    assert recording_page_url("http://127.0.0.1:8780/?present=1&autoplay=1#room") == (
        "http://127.0.0.1:8780/?present=1&autoplay=0"
    )
