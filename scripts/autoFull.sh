#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"

if [ ! -x "$VENV/bin/python" ]; then
  echo "Virtual environment not found at $VENV" >&2
  echo "Run ./scripts/setup_venv.sh first." >&2
  exit 1
fi

exec "$VENV/bin/python" "$ROOT/scripts/autoFull.py" "$@"
