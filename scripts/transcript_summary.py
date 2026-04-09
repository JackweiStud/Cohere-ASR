#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import httpx

from transcript_cleanup import DEFAULT_ENV_PATH, configure_logging, resolve_llm_settings


DEFAULT_LOG_PATH = "/Users/jackwl/Code/test/logs/transcript_summary.log"

SUMMARY_SYSTEM_PROMPT = (
    "你是一位双语信息架构师。请阅读用户提供的完整转写文本。"
    "转写文本可能来自 ASR，可能含口误或听写错误。"
    "请严格按照用户要求的结构输出简体中文结果，不要增加额外前言或说明。"
)

SUMMARY_USER_PROMPT_TEMPLATE = """URL 视频快速总结

你是一位双语信息架构师。请阅读用户粘贴的【完整转写文本】（可能来自 ASR，含口误/听写错误）。

请用简体中文输出两部分，结构严格如下：

### A. 内容总结与“主张—依据—中译”
1) 先用 5–10 句中文概括全文主线（说明这是演讲/口述/推广/教程中的哪一种语气也可）。
2) 提取 5–8 条“核心观点/主张”。每条必须包含：
   - 主张标题（简短）
   - 主张说明（1–3 句中文）
   - 依据：1–2 段【转写原文英文摘录】（尽量是连续短语/句子，不要编造）
   - 对应【中文翻译】（忠实直译为主，专有名词不确定则标注“疑似：…”）

要求：
- 不要把营销数字（例如节省 X 小时）当成事实；若出现，标注为“演讲者声称/需核验”。
- 若你发现明显的 ASR 误差，在段末用括注提示“疑似听写：…→…”不要展开全文纠错表（除非用户另要求）。

### B. X 汇报（演讲者主张框架版）
把上述观点整理成适合在 X（Twitter）发布的“串帖/Thread”体例（10 条以内），要求：
- 中性语气：这是“演讲者主张框架”，不是你已经验证的结论。
- 每条 1–3 句中文，便于 thread 阅读；第一条是总览，最后一条写“风险：口误/听写、个人经历叙事、商业转化段落需分拆引用”等。
- 不要输出 hashtag；不要编造转写里没有的步骤或产品功能。

【完整转写文本】：
{transcript}
"""


def summarize_transcript_with_llm(
    transcript: str,
    env_path: str = DEFAULT_ENV_PATH,
) -> tuple[str, str, float]:
    api_key, base_url, model = resolve_llm_settings(env_path)
    if not api_key or not model:
        raise RuntimeError("LLM settings are incomplete in local .env")

    started = time.time()
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": SUMMARY_USER_PROMPT_TEMPLATE.format(transcript=transcript),
                },
            ],
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    content = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    elapsed = round(time.time() - started, 3)
    if not content:
        raise RuntimeError("LLM returned empty summary content")
    return content, model, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Structured Chinese summary from transcript text")
    parser.add_argument(
        "--input",
        default="/Users/jackwl/Code/test/output/transcript_cleaned.txt",
        help="Path to transcript txt file",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional markdown output path",
    )
    parser.add_argument(
        "--env-path",
        default=DEFAULT_ENV_PATH,
        help="Path to local .env file",
    )
    parser.add_argument(
        "--log-path",
        default=DEFAULT_LOG_PATH,
        help="Path to log file",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_summary.md")
    )
    configure_logging(str(Path(args.log_path).expanduser().resolve()))

    logging.info("Starting transcript summary")
    logging.info("Input path: %s", input_path)
    logging.info("Output path: %s", output_path)
    logging.info("Env path: %s", Path(args.env_path).expanduser().resolve())

    transcript = input_path.read_text(encoding="utf-8").strip()
    summary, model, elapsed = summarize_transcript_with_llm(
        transcript,
        env_path=str(Path(args.env_path).expanduser().resolve()),
    )
    output_path.write_text(summary + "\n", encoding="utf-8")

    logging.info("Transcript summary finished in %.3fs", elapsed)
    logging.info("Summary model: %s", model)

    print(f"Summary written to: {output_path}")
    print(f"Summary model: {model}")
    print(f"Summary seconds: {elapsed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
