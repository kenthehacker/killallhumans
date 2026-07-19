#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$ROOT_DIR/requirements/development-test.lock.txt}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[setup] ERROR: Python executable '$PYTHON_BIN' not found"
  exit 1
fi

resolve_path() {
  "$PYTHON_BIN" -c \
    'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' \
    "$1"
}

# Compare canonical paths so aliases such as ./scripts/../.venv cannot bypass
# profile isolation. Resolution is read-only and works before the venv exists.
VENV_DIR="$(resolve_path "$VENV_DIR")"
REQUIREMENTS_FILE="$(resolve_path "$REQUIREMENTS_FILE")"
DEFAULT_VENV="$(resolve_path "$ROOT_DIR/.venv")"
LEGACY_REQUIREMENTS="$(resolve_path "$ROOT_DIR/requirements/legacy-simulation.txt")"

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  echo "[setup] ERROR: requirements file not found: $REQUIREMENTS_FILE"
  exit 1
fi
if [[ "$REQUIREMENTS_FILE" == "$LEGACY_REQUIREMENTS" && "$VENV_DIR" == "$DEFAULT_VENV" ]]; then
  echo "[setup] ERROR: refusing to mix legacy simulation dependencies into $DEFAULT_VENV"
  echo "[setup] Set an explicit distinct path, for example VENV_DIR='$ROOT_DIR/.venv-legacy'."
  exit 1
fi

PROFILE_MARKER="$VENV_DIR/.aigp-environment-profile"

# Bind the destination before creating the venv or invoking pip.  If either
# operation is interrupted, a later run may resume only with this same exact
# profile instead of silently layering a different profile over a partial
# install.  Do not guess the provenance of an already-populated, unmarked
# environment: require a fresh path for the first managed setup.
if [[ ! -e "$PROFILE_MARKER" && ! -L "$PROFILE_MARKER" ]]; then
  if [[ -e "$VENV_DIR" && ! -d "$VENV_DIR" ]]; then
    echo "[setup] ERROR: venv destination is not a directory: $VENV_DIR"
    exit 1
  fi
  if [[ -d "$VENV_DIR" ]]; then
    VENV_HAS_ENTRIES=false
    for entry in "$VENV_DIR"/* "$VENV_DIR"/.[!.]* "$VENV_DIR"/..?*; do
      if [[ -e "$entry" || -L "$entry" ]]; then
        VENV_HAS_ENTRIES=true
        break
      fi
    done
    if [[ "$VENV_HAS_ENTRIES" == true ]]; then
      echo "[setup] ERROR: refusing to adopt populated unbound environment: $VENV_DIR"
      echo "[setup] Choose a fresh VENV_DIR; profile provenance cannot be inferred safely."
      exit 1
    fi
  else
    mkdir -p "$VENV_DIR"
  fi

  # Publish a fully-written marker with an atomic create-if-absent hard link.
  # A concurrent reserver can win the link race, after which the exact content
  # check below decides whether this invocation may proceed.
  "$PYTHON_BIN" - "$PROFILE_MARKER" "$REQUIREMENTS_FILE" <<'PY'
import os
import pathlib
import stat
import sys
import tempfile

marker = pathlib.Path(sys.argv[1])
profile = sys.argv[2]
fd, temporary_name = tempfile.mkstemp(
    dir=marker.parent, prefix=f".{marker.name}.", suffix=".tmp"
)
temporary = pathlib.Path(temporary_name)
try:
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(profile + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, marker)
    except FileExistsError:
        pass
finally:
    temporary.unlink(missing_ok=True)
PY
fi

if [[ -L "$PROFILE_MARKER" || ! -f "$PROFILE_MARKER" ]]; then
  echo "[setup] ERROR: environment profile marker is not a regular file: $PROFILE_MARKER"
  exit 1
fi
EXISTING_PROFILE="$(<"$PROFILE_MARKER")"
if [[ "$EXISTING_PROFILE" != "$REQUIREMENTS_FILE" ]]; then
  echo "[setup] ERROR: $VENV_DIR is already bound to a different requirements profile:"
  echo "[setup]   $EXISTING_PROFILE"
  echo "[setup] Choose a distinct VENV_DIR instead of mixing dependency profiles."
  exit 1
fi

echo "[setup] root: $ROOT_DIR"
echo "[setup] python: $PYTHON_BIN"
echo "[setup] venv: $VENV_DIR"
echo "[setup] requirements: $REQUIREMENTS_FILE"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install -r "$REQUIREMENTS_FILE"

echo "[setup] complete"
if [[ "$REQUIREMENTS_FILE" == "$LEGACY_REQUIREMENTS" ]]; then
  echo "[setup] run the legacy demo with:"
  echo "        bash $ROOT_DIR/scripts/run_demo.sh --interactive"
else
  echo "[setup] run the default non-live tests with:"
  echo "        python -m pytest -q"
fi
