#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[run_demo] venv python not found at $PYTHON_BIN"
  echo "[run_demo] bootstrap first: bash $ROOT_DIR/scripts/setup_venv.sh"
  exit 1
fi

exec "$PYTHON_BIN" "$ROOT_DIR/simulation/demo.py" "$@"
