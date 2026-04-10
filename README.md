# Cohere Local PoC — Audio Transcription + LLM Cleanup + Summary

> **Language / 语言：** English | [中文](README.zh.md)

A self-contained local feasibility workspace for running
[`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
on Apple Silicon (tested on Mac mini M4 16 GB) and piping the output through an
optional LLM-based cleanup plus one structured post-processing output:
an analysis pack.

The workspace answers one concrete question:

> Can a Mac run the Cohere model locally well enough to justify integrating a
> `summary-only-fast` pipeline later?

---

## Pipeline overview

```
input/audio.wav
    │
    ▼
poc_cohere_local_transcribe.py   ← local ASR (transformers + MPS)
    │  output/transcript.txt
    ▼
transcript_cleanup.py            ← optional LLM dedup / punctuation pass
    │  output/transcript_cleaned.txt
    ▼
transcript_summary.py            ← LLM analysis pack (can read transcript.txt or transcript_cleaned.txt)
       output/transcript_cleaned_analysis.md
```

Each stage is an independent script and can be run in isolation.

---

## Layout

```text
.
├── .env                   ← local secrets (git-ignored)
├── .env.example           ← copy this, fill in your keys
├── .venv/                 ← created by setup_venv.sh (git-ignored)
├── input/                 ← put your audio files here (git-ignored)
├── logs/                  ← runtime logs (git-ignored)
├── models/                ← downloaded model weights (git-ignored)
│   └── CohereLabs/
├── output/                ← generated transcripts and summaries (git-ignored)
├── requirements.txt
├── scripts/
│   ├── autoFull.py
│   ├── autoFull.sh
│   ├── download_model.sh
│   ├── setup_venv.sh
│   ├── poc_cohere_local_transcribe.py
│   ├── transcript_cleanup.py
│   └── transcript_summary.py
├── README.md
└── README.zh.md
```

---

## Requirements

- **Python 3.11** — the local ASR stack is more compatible on 3.11 than on
  newer versions.
- An LLM API key for the cleanup and summary steps (see [Local env](#local-env)).

---

## Setup

```bash
./scripts/setup_venv.sh
```

The script auto-detects `python3.11` (or falls back to `python3`). You can
override the interpreter:

```bash
PYTHON=/opt/homebrew/bin/python3.11 ./scripts/setup_venv.sh
```

---

## Model download

```bash
./scripts/download_model.sh
```

The model is saved to `./models/CohereLabs/cohere-transcribe-03-2026`.
The PoC script detects this path automatically.

To use a mirror (e.g. inside mainland China):

```bash
HF_ENDPOINT=https://hf-mirror.com ./scripts/download_model.sh
```

---

## Local env

Copy the template and fill in your API key:

```bash
cp .env.example .env
```

`.env.example`:

```dotenv
LLM_API_KEY=
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=deepseek-ai/DeepSeek-V3
```

`LLM_BASE_URL` and `LLM_MODEL` are optional — the defaults above are used when
they are absent.

---

## Run

### Full pipeline (transcribe → optional cleanup → summary)

The quickest way to run everything in one shot is the Python CLI:

```bash
./scripts/autoFull.py
```

The shell wrapper is kept for compatibility:

```bash
./scripts/autoFull.sh
```

Cleanup is skipped by default. To enable step 2, pass:

```bash
./scripts/autoFull.sh --enCleanUp 1
```

All options are optional — defaults are picked up automatically:

```bash
./scripts/autoFull.sh \
  --input    ./input/extracted_audio.wav \
  --output   ./output \
  --language en \
  --model    ./models/CohereLabs/cohere-transcribe-03-2026 \
  --env      ./.env
```

`-h` / `--help` prints the full usage.

### Step by step

If you need finer control, run each stage individually.

#### Transcribe only

```bash
./.venv/bin/python ./scripts/poc_cohere_local_transcribe.py \
  --input ./input/extracted_audio.wav \
  --output-dir ./output
```

#### Cleanup only

```bash
./.venv/bin/python ./scripts/transcript_cleanup.py \
  --input ./output/transcript.txt \
  --output ./output/transcript_cleaned.txt
```

### Summary only

```bash
./.venv/bin/python ./scripts/transcript_summary.py \
  --input ./output/transcript_cleaned.txt
```

Optional: explicitly set the analysis output path:

```bash
./.venv/bin/python ./scripts/transcript_summary.py \
  --input ./output/transcript_cleaned.txt \
  --output ./output/transcript_cleaned_analysis.md
```

---

## Output files

| File | Description |
|------|-------------|
| `output/transcript.txt` | Raw ASR output |
| `output/transcript_cleaned.txt` | LLM-deduped and punctuated transcript |
| `output/transcript_analysis.md` | Analysis pack generated directly from the raw transcript |
| `output/transcript_cleaned_analysis.md` | Claim/evidence/risk analysis pack + X thread draft |
| `output/report.json` | Runtime benchmark — model path, Python/torch version, device, audio duration, load time, transcription time, RSS |
| `logs/poc_cohere_local_transcribe.log` | ASR run log |
| `logs/transcript_cleanup.log` | Cleanup run log |
| `logs/transcript_summary.log` | Summary run log |

---

## Notes

- If you see multilingual garbage in the transcription output, the most likely
  cause is an unsupported `transformers` version. Check `requirements.txt` for
  the tested version range.
- The LLM cleanup and summary steps are independent modules — they can be
  imported or called without the ASR step.
- The summary stage now produces one Markdown file:
  an analysis pack for `claim / evidence / risk` review.
- Final Markdown is post-processed for readability, with sentence-first line
  breaks after `. ? !` and `。？！` where possible.
