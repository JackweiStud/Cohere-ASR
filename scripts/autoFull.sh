#!/bin/zsh
# autoFull.sh — transcribe → analysis pack in one shot (cleanup step is skipped)
#
# Usage:
#   ./scripts/autoFull.sh [OPTIONS]
#
# Options:
#   -i, --input     <file>   Audio file to transcribe (default: ./input/extracted_audio.wav)
#   -o, --output    <dir>    Output directory (default: ./output)
#   -m, --model     <dir>    Local model directory (optional; auto-detected when absent)
#   -l, --language  <code>   ISO 639-1 language code (default: en)
#   -e, --env       <file>   Path to .env file (default: ./.env)
#   -h, --help               Show this help message

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
SCRIPTS="$ROOT/scripts"

# ---------- defaults ----------
INPUT="$ROOT/input/extracted_audio.wav"
OUTPUT_DIR="$ROOT/output"
MODEL_PATH=""
LANGUAGE="en"
ENV_PATH="$ROOT/.env"

# ---------- arg parse ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)    INPUT="$2";       shift 2 ;;
    -o|--output)   OUTPUT_DIR="$2";  shift 2 ;;
    -m|--model)    MODEL_PATH="$2";  shift 2 ;;
    -l|--language) LANGUAGE="$2";    shift 2 ;;
    -e|--env)      ENV_PATH="$2";    shift 2 ;;
    -h|--help)
      sed -n '2,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ---------- sanity checks ----------
if [ ! -x "$VENV/bin/python" ]; then
  echo "Virtual environment not found at $VENV" >&2
  echo "Run ./scripts/setup_venv.sh first." >&2
  exit 1
fi

if [ ! -f "$INPUT" ]; then
  echo "Input audio file not found: $INPUT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

PYTHON="$VENV/bin/python"
TRANSCRIPT="$OUTPUT_DIR/transcript.txt"
ANALYSIS="$OUTPUT_DIR/transcript_analysis.md"

# ---------- step 1: transcribe ----------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 1/2 — Transcribe"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TRANSCRIBE_ARGS=(
  "$SCRIPTS/poc_cohere_local_transcribe.py"
  --input      "$INPUT"
  --output-dir "$OUTPUT_DIR"
  --language   "$LANGUAGE"
)
[[ -n "$MODEL_PATH" ]] && TRANSCRIBE_ARGS+=(--model-path "$MODEL_PATH")

"$PYTHON" "${TRANSCRIBE_ARGS[@]}"

if [ ! -f "$TRANSCRIPT" ]; then
  echo "Transcription failed: $TRANSCRIPT not found." >&2
  exit 2
fi

# ---------- step 2: summary ----------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 2/2 — Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

"$PYTHON" "$SCRIPTS/transcript_summary.py" \
  --input    "$TRANSCRIPT" \
  --output   "$ANALYSIS" \
  --env-path "$ENV_PATH"

# ---------- done ----------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Done"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " transcript : $TRANSCRIPT"
echo " analysis   : $ANALYSIS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
