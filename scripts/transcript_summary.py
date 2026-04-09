#!/usr/bin/env python3
"""基于转写文本生成结构化中文总结。

职责范围
    只负责读取已有的转写文本，调用配置好的 LLM 输出结构化中文总结与 X 汇报草稿。
    不负责语音转写；语音转写请使用 ``poc_cohere_local_transcribe.py``。
    不负责 minimum-edit cleanup；转写清洗请使用 ``transcript_cleanup.py``。

使用示例（建议在项目虚拟环境中执行）::
    /path/to/.venv/bin/python scripts/transcript_summary.py \\
        --input output/transcript_cleaned.txt \\
        --output output/transcript_cleaned_summary.md

主要参数
    --input     输入转写文本路径。默认 ``output/transcript_cleaned.txt``。
    --output    可选；输出 Markdown 路径。默认与输入同目录，文件名自动加 ``_summary.md`` 后缀。
    --log-path  日志文件路径。默认 ``logs/transcript_summary.log``。

LLM 配置
    默认读取项目根目录下的 ``.env``（与 ``scripts/`` 同级），通常不必传入 ``--env-path``。
    仅在需要使用非默认路径时，才传入 ``--env-path`` 覆盖。

注意事项
    - 需要在环境变量或项目根 ``.env`` 中提供 ``LLM_API_KEY``；``LLM_BASE_URL`` 和
      ``LLM_MODEL`` 可选，不填则使用默认值。
    - 当前仅接受 ``.txt`` 输入；若输入不是文本转写结果，脚本会记录错误日志并给出建议。
    - 本脚本使用 ``temperature=0``，尽量让总结结构稳定、可复现。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import httpx

from transcript_cleanup import DEFAULT_ENV_PATH, configure_logging, resolve_llm_settings


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "output" / "transcript_cleaned.txt"
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "transcript_summary.log"

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
- 当原文同时出现“工具隐喻/宣传口号”和“具体操作链路”时，优先总结具体操作链路，不要只保留抽象比喻。

### B. X 汇报（演讲者主张框架版）
把上述观点整理成适合在 X（Twitter）发布的“串帖/Thread”体例（10 条以内），要求：
- 中性语气：这是“演讲者主张框架”，不是你已经验证的结论。
- 每条 1–3 句中文，便于 thread 阅读；第一条是总览，最后一条写“风险：口误/听写、个人经历叙事、商业转化段落需分拆引用”等。
- 不要输出 hashtag；不要编造转写里没有的步骤或产品功能。
- 若某条观点涉及工具协同、知识库接入、资料比对，thread 中必须写清“原文中提供的信息是怎么结合”，避免只写成抽象金句或宣传口号。

【完整转写文本】：
{transcript}
"""


def summarize_transcript_with_llm(
    transcript: str,
    env_path: str | Path = DEFAULT_ENV_PATH,
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
            "temperature": 0,
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
        default=str(DEFAULT_INPUT_PATH),
        help="Path to transcript txt file",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional markdown output path",
    )
    parser.add_argument(
        "--env-path",
        default=str(DEFAULT_ENV_PATH),
        help="可选。覆盖 LLM 配置 .env；默认使用项目根目录 .env，通常不必传入。",
    )
    parser.add_argument(
        "--log-path",
        default=str(DEFAULT_LOG_PATH),
        help="Path to log file",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    configure_logging(str(Path(args.log_path).expanduser().resolve()))

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    if input_path.suffix.lower() != ".txt":
        logging.error("Unsupported input format: %s", input_path.suffix or "<no suffix>")
        logging.error("This script currently accepts only .txt transcript input: %s", input_path)
        print(
            "Unsupported input format. This script currently accepts only .txt transcript input.\n"
            f"Input path: {input_path}\n"
            "Suggestion: provide transcript_cleaned.txt or another transcript text file first.",
            file=sys.stderr,
        )
        return 1

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_summary.md")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Starting transcript summary")
    logging.info("Input path: %s", input_path)
    logging.info("Output path: %s", output_path)
    logging.info("Env path: %s", Path(args.env_path).expanduser().resolve())

    transcript = input_path.read_text(encoding="utf-8").strip()
    if not transcript:
        logging.warning("Input transcript is empty: %s", input_path)
        print(f"Input transcript is empty: {input_path}", file=sys.stderr)
        return 1

    try:
        summary, model, elapsed = summarize_transcript_with_llm(
            transcript,
            env_path=str(Path(args.env_path).expanduser().resolve()),
        )
    except httpx.HTTPError as exc:
        logging.error("Transcript summary request failed: %s", exc)
        print(
            "Transcript summary request failed. Please check LLM_API_KEY, LLM_BASE_URL, "
            "LLM_MODEL, network connectivity, and server availability.",
            file=sys.stderr,
        )
        return 2
    except RuntimeError as exc:
        logging.error("Transcript summary failed: %s", exc)
        print(str(exc), file=sys.stderr)
        return 2

    output_path.write_text(summary + "\n", encoding="utf-8")

    logging.info("Transcript summary finished in %.3fs", elapsed)
    logging.info("Summary model: %s", model)

    print(f"Summary written to: {output_path}")
    print(f"Summary model: {model}")
    print(f"Summary seconds: {elapsed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
