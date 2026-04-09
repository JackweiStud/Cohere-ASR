#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
PYTHON="${PYTHON:-$(command -v python3.11 2>/dev/null || command -v python3 2>/dev/null || true)}"

if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
  echo "python3.11 (or python3) not found. Install it or set the PYTHON env var." >&2
  exit 1
fi

if [ ! -d "$VENV" ]; then
  "$PYTHON" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install --upgrade --force-reinstall -r "$ROOT/requirements.txt"

echo "Virtual environment ready at $VENV"
