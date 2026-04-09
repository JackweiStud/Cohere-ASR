#!/bin/zsh
set -euo pipefail

ROOT="/Users/jackwl/Code/test"
VENV="$ROOT/.venv"
PYTHON="/opt/homebrew/bin/python3.11"

if [ ! -x "$PYTHON" ]; then
  echo "python3.11 not found at $PYTHON" >&2
  exit 1
fi

if [ ! -d "$VENV" ]; then
  "$PYTHON" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install --upgrade --force-reinstall -r "$ROOT/requirements.txt"

echo "Virtual environment ready at $VENV"
