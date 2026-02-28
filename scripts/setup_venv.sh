#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

echo "[setup] root: $ROOT_DIR"
echo "[setup] python: $PYTHON_BIN"
echo "[setup] venv: $VENV_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[setup] ERROR: Python executable '$PYTHON_BIN' not found"
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT_DIR/requirements.txt"

echo "[setup] complete"
echo "[setup] run demo with:"
echo "        bash $ROOT_DIR/scripts/run_demo.sh --interactive"
