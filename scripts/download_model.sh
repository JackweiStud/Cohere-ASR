#!/bin/zsh
set -euo pipefail

ROOT="/Users/jackwl/Code/test"
VENV="$ROOT/.venv"
MODEL_ID="CohereLabs/cohere-transcribe-03-2026"
TARGET_DIR="$ROOT/models/CohereLabs/cohere-transcribe-03-2026"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://hf-mirror.com}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "Virtual environment not found: $VENV" >&2
  echo "Run ./scripts/setup_venv.sh first." >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"

echo "Using HF endpoint: $HF_ENDPOINT_VALUE"

HF_ENDPOINT="$HF_ENDPOINT_VALUE" "$VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download

model_id = "CohereLabs/cohere-transcribe-03-2026"
target_dir = "/Users/jackwl/Code/test/models/CohereLabs/cohere-transcribe-03-2026"

snapshot_download(
    repo_id=model_id,
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)

print(f"Downloaded model to: {target_dir}")
PY
