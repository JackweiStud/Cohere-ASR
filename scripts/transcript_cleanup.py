#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import httpx


DEFAULT_ENV_PATH = "/Users/jackwl/Code/test/.env"
DEFAULT_LOG_PATH = "/Users/jackwl/Code/test/logs/transcript_cleanup.log"


def configure_logging(log_path: str) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def load_env_file(env_path: str) -> dict[str, str]:
    data: dict[str, str] = {}
    path = Path(env_path)
    if not path.exists():
        return data

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            data[key] = value
    return data


def resolve_llm_settings(env_path: str = DEFAULT_ENV_PATH) -> tuple[str, str, str]:
    file_env = load_env_file(env_path)
    api_key = os.environ.get("LLM_API_KEY") or file_env.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL") or file_env.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL") or file_env.get("LLM_MODEL", "gpt-4o-mini")
    return api_key.strip(), base_url.rstrip("/"), model.strip()


def strip_cleanup_preamble(text: str) -> str:
    if not text:
        return text

    prefixes = (
        "Here's the lightly cleaned transcript:",
        "Here is the lightly cleaned transcript:",
        "Here's the cleaned transcript:",
        "Here is the cleaned transcript:",
        "Cleaned transcript:",
        "Lightly cleaned transcript:",
    )

    stripped = text.strip()
    for prefix in prefixes:
        if stripped.lower().startswith(prefix.lower()):
            stripped = stripped[len(prefix):].lstrip()
            break
    return stripped


def cleanup_transcript_with_llm(
    transcript: str,
    env_path: str = DEFAULT_ENV_PATH,
) -> tuple[str, str, float]:
    api_key, base_url, model = resolve_llm_settings(env_path)
    if not api_key or not model:
        logging.warning("Skipping transcript cleanup because LLM settings are incomplete")
        return transcript, "", 0.0

    system_prompt = (
        "You are an ASR transcript cleanup assistant. "
        "Your task is minimum-edit cleanup only. "
        "Stay as close as possible to the original transcript. "
        "Only fix obvious ASR recognition errors when the correction is highly confident. "
        "Only merge obviously broken sentences when necessary for readability. "
        "Preserve original wording, order, tone, meaning, names, numbers, timings, claims, and examples. "
        "Do not summarize. "
        "Do not rewrite for style. "
        "Do not paraphrase for smoothness. "
        "Do not add titles, notes, explanations, labels, or any introductory sentence. "
        "If uncertain, keep the original text. "
        "Return plain text only."
    )
    user_prompt = (
        "Clean this ASR transcript with a minimum-edit pass.\n\n"
        "Rules:\n"
        "1. Make the smallest possible number of edits.\n"
        "2. Correct only obvious ASR mistakes when confidence is high.\n"
        "3. Preserve wording and sentence order whenever possible.\n"
        "4. Preserve all numbers, examples, names, and factual claims.\n"
        "5. Merge only clearly broken sentence fragments.\n"
        "6. Remove repeated fragments only if they are clearly accidental ASR duplication.\n"
        "7. Do not add headings, notes, quotes, introductions, or explanations.\n"
        "8. Do not summarize or rewrite stylistically.\n"
        "9. If uncertain, keep the original wording.\n"
        "10. Return only the cleaned transcript text.\n\n"
        f"Transcript:\n{transcript}"
    )

    started = time.time()
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    cleaned = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    elapsed = round(time.time() - started, 3)
    cleaned = strip_cleanup_preamble(cleaned)
    if not cleaned:
        logging.warning("Cleanup LLM returned empty content; keeping original transcript")
        return transcript, model, elapsed
    return cleaned, model, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Light cleanup for ASR transcript text")
    parser.add_argument("--input", required=True, help="Path to transcript txt file")
    parser.add_argument("--output", default="", help="Optional cleaned transcript path")
    parser.add_argument("--env-path", default=DEFAULT_ENV_PATH, help="Path to local .env file")
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH, help="Path to log file")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output).expanduser().resolve() if args.output else input_path.with_name(f"{input_path.stem}_cleaned.txt")
    configure_logging(str(Path(args.log_path).expanduser().resolve()))

    logging.info("Starting transcript cleanup")
    logging.info("Input path: %s", input_path)
    logging.info("Output path: %s", output_path)
    logging.info("Env path: %s", Path(args.env_path).expanduser().resolve())

    transcript = input_path.read_text(encoding="utf-8").strip()
    cleaned, model, elapsed = cleanup_transcript_with_llm(transcript, env_path=str(Path(args.env_path).expanduser().resolve()))
    output_path.write_text(cleaned + "\n", encoding="utf-8")

    logging.info("Transcript cleanup finished in %.3fs", elapsed)
    if model:
        logging.info("Cleanup model: %s", model)

    print(f"Cleaned transcript written to: {output_path}")
    if model:
        print(f"Cleanup model: {model}")
        print(f"Cleanup seconds: {elapsed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
