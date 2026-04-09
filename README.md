# Cohere Local PoC Workspace

This workspace is isolated from the main project and is only for local feasibility testing of `CohereLabs/cohere-transcribe-03-2026` on `Mac mini M4 16GB`.

It now contains one local test track:

- `transformers + torch + MPS`

## Layout

```text
/Users/jackwl/Code/test/
├── .env
├── .env.example
├── .venv/
├── input/
│   └── extracted_audio.wav
├── logs/
├── models/
│   └── CohereLabs/
├── output/
├── requirements.txt
├── scripts/
│   ├── download_model.sh
│   ├── setup_venv.sh
│   ├── poc_cohere_local_transcribe.py
│   ├── transcript_cleanup.py
│   └── transcript_summary.py
└── README.md
```

## Python

This workspace uses Python `3.11`, not the system default `3.14`, because the local ASR stack is more likely to be compatible on `3.11`.

## Input

Default input directory:

```text
/Users/jackwl/Code/test/input
```

Current default test file:

```text
/Users/jackwl/Code/test/input/extracted_audio.wav
```

## Setup

```bash
cd /Users/jackwl/Code/test
./scripts/setup_venv.sh
```

## Model Download

Recommended local model path:

```text
/Users/jackwl/Code/test/models/CohereLabs/cohere-transcribe-03-2026
```

Download with:

```bash
cd /Users/jackwl/Code/test
./scripts/download_model.sh
```

The PoC script will prefer this local path automatically.

## Local Env

This test workspace no longer reads the main project's `.env`.

Local cleanup settings live in:

```text
/Users/jackwl/Code/test/.env
```

Template:

```text
/Users/jackwl/Code/test/.env.example
```

## Run

```bash
cd /Users/jackwl/Code/test
./.venv/bin/python ./scripts/poc_cohere_local_transcribe.py \
  --input /Users/jackwl/Code/test/input/extracted_audio.wav \
  --output-dir /Users/jackwl/Code/test/output
```

Optional explicit env path:

```bash
cd /Users/jackwl/Code/test
./.venv/bin/python ./scripts/poc_cohere_local_transcribe.py \
  --env-path /Users/jackwl/Code/test/.env \
  --input /Users/jackwl/Code/test/input/extracted_audio.wav \
  --output-dir /Users/jackwl/Code/test/output
```

## Cleanup Only

You can run transcript cleanup independently:

```bash
cd /Users/jackwl/Code/test
./.venv/bin/python ./scripts/transcript_cleanup.py \
  --input /Users/jackwl/Code/test/output/transcript.txt \
  --output /Users/jackwl/Code/test/output/transcript_cleaned.txt
```

## Summary Only

You can run the structured Chinese summary independently:

```bash
cd /Users/jackwl/Code/test
./.venv/bin/python ./scripts/transcript_summary.py \
  --input /Users/jackwl/Code/test/output/transcript_cleaned.txt \
  --output /Users/jackwl/Code/test/output/transcript_summary.md
```

Optional explicit model path:

```bash
cd /Users/jackwl/Code/test
./.venv/bin/python ./scripts/poc_cohere_local_transcribe.py \
  --model-path /Users/jackwl/Code/test/models/CohereLabs/cohere-transcribe-03-2026 \
  --input /Users/jackwl/Code/test/input/extracted_audio.wav \
  --output-dir /Users/jackwl/Code/test/output
```

## What the script writes

- `output/transcript.txt`
- `output/transcript_cleaned.txt`
- `output/transcript_summary.md`
- `output/report.json`

The report captures basic runtime facts for this machine:

- selected model path
- python version
- torch version
- selected device
- input duration
- model load time
- transcription time
- rough process RSS before and after

The scripts also write:

- `logs/poc_cohere_local_transcribe.log`
- `logs/transcript_cleanup.log`
- `logs/transcript_summary.log`

## Notes

- This PoC uses `transformers` in the version range recommended by the model card.
- If you see multilingual garbage output, the first thing to check is whether the environment is using an unsupported `transformers` version.
- The LLM cleanup step is an independent script and module, reused by the local Cohere PoC.
- The LLM summary step is also independent and uses a fixed prompt for URL video transcript summarization.

## Scope

This workspace only answers one question:

Can this Mac run the Cohere model locally well enough to justify integrating `summary-only-fast` later?
