#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec /home/suzhenjie2026/world-model/deployments/wm-safetylab/.venv/bin/python "$ROOT/scripts/team_workspace.py" "$@"
