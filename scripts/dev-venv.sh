#!/bin/bash
# Creates .venv next to the repo root of the CURRENT worktree. Each worktree may share one venv:
#   VENV=/path/to/venv scripts/dev-venv.sh
set -e
cd "$(dirname "$0")/.."
V="${VENV:-.venv}"
[ -x "$V/bin/python" ] || uv venv --python 3.12 "$V"
uv pip install --python "$V/bin/python" -r requirements-dev.txt
echo "venv ready: $V"
