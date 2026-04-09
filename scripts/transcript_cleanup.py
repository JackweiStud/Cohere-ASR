#!/usr/bin/env python3
"""对 ASR 转写文本做最小改动清洗。

职责范围
    只负责读取已有的转写文本，调用配置好的 LLM 做 minimum-edit cleanup，
    输出清洗后的 ``*_cleaned.txt``。不负责语音转写；语音转写请使用
    ``poc_cohere_local_transcribe.py``。

使用示例（建议在项目虚拟环境中执行）::
    /path/to/.venv/bin/python scripts/transcript_cleanup.py \\
        --input output/transcript.txt \\
        --output output/transcript_cleaned.txt

主要参数
    --input     输入转写文本路径。当前建议使用 ``.txt``，例如 ``output/transcript.txt``。
    --output    可选；输出路径。默认与输入同目录，文件名自动加 ``_cleaned`` 后缀。
    --log-path  日志文件路径。默认 ``logs/transcript_cleanup.log``。

LLM 配置
    默认读取项目根目录下的 ``.env``（与 ``scripts/`` 同级），**无需在命令行传任何 env 参数**。
    仅在需要使用非默认路径时，才传入 ``--env-path`` 覆盖。

注意事项
    - 需要在环境变量或项目根 ``.env`` 中提供 ``LLM_API_KEY``；``LLM_BASE_URL`` 和
      ``LLM_MODEL`` 可选，不填则使用默认值。
    - 若输入不是 ``.txt``，脚本会记录错误日志并提示先提供文本转写结果。
    - 该脚本使用 ``temperature=0``，尽量保持最小修改。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import httpx


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "transcript_cleanup.log"


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


def resolve_llm_settings(env_path: str | Path = DEFAULT_ENV_PATH) -> tuple[str, str, str]:
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
    env_path: str | Path = DEFAULT_ENV_PATH,
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
    parser.add_argument(
        "--env-path",
        default=str(DEFAULT_ENV_PATH),
        help="可选。覆盖 LLM 配置 .env；默认使用项目根目录 .env，通常不必传入。",
    )
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH), help="Path to log file")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    configure_logging(str(Path(args.log_path).expanduser().resolve()))

    if not input_path.exists():
        logging.error("Input file not found: %s", input_path)
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    if input_path.suffix.lower() != ".txt":
        logging.error("Unsupported input format: %s", input_path.suffix or "<no suffix>")
        logging.error("This script currently accepts only .txt transcript input: %s", input_path)
        print(
            "Unsupported input format. This script currently accepts only .txt transcript input.\n"
            f"Input path: {input_path}\n"
            "Suggestion: provide the transcript text file first, such as output/transcript.txt.",
            file=sys.stderr,
        )
        return 1

    output_path = Path(args.output).expanduser().resolve() if args.output else input_path.with_name(f"{input_path.stem}_cleaned.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Starting transcript cleanup")
    logging.info("Input path: %s", input_path)
    logging.info("Output path: %s", output_path)
    logging.info("Env path: %s", Path(args.env_path).expanduser().resolve())

    transcript = input_path.read_text(encoding="utf-8").strip()
    if not transcript:
        logging.warning("Input transcript is empty: %s", input_path)
        print(f"Input transcript is empty: {input_path}", file=sys.stderr)
        return 1

    try:
        cleaned, model, elapsed = cleanup_transcript_with_llm(
            transcript,
            env_path=str(Path(args.env_path).expanduser().resolve()),
        )
    except httpx.HTTPError as exc:
        logging.error("Transcript cleanup request failed: %s", exc)
        print(
            "Transcript cleanup request failed. Please check LLM_API_KEY, LLM_BASE_URL, "
            "LLM_MODEL, network connectivity, and server availability.",
            file=sys.stderr,
        )
        return 2

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
