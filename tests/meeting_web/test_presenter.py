"""Exercise the recording view's state boundaries with the actual JavaScript module."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_presenter_clears_private_content_and_uses_observed_model_results():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for the presenter state-boundary checks")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [node, str(Path(__file__).with_name("presenter_contract.cjs")),
         str(root / "unspoken/meeting/web/static/presenter.js")],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
