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

内部机制原理
    脚本按「全文字符数」做粗粒度预算（近似 token，非 tokenizer 精确值），分两条主路径：

    1) 短文本直出：长度不超过 ``DIRECT_SUMMARY_CHAR_BUDGET`` 时，将整段转写填入用户
       prompt，一次 ``/chat/completions`` 请求直接产出最终 A/B 结构 Markdown。

    2) 长文本分段汇总：超过直出阈值时，先用 ``CHUNK_CHAR_BUDGET`` 在标点与空白等
       ``CHUNK_BREAK_MARKERS`` 优先处切分，再对每段单独请求「分段摘要」；若某段仍
       被服务端判为超长（如 413、或 400 响应体含典型上下文超限文案），则在该段内
       递归减半拆分，直至低于 ``MIN_CHUNK_CHAR_BUDGET`` 仍失败则向上抛出。
       所有分段摘要拼成上下文后，再发一次「汇总」请求，仅允许使用各段里已出现的
       依据与翻译，输出与短路径相同的 A/B 最终结构。

    请求与容错：``_call_llm_with_retry`` 对超时、连接类错误及 ``RETRYABLE_STATUS_CODES``
    中的状态码做有限次指数退避重试；非可重试的 HTTP 错误或配置缺失则立即失败。
    日志会记录分段数、重试与退化情况，便于区分「 prompt 过长」与「网络不稳定」。
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import httpx

from transcript_cleanup import DEFAULT_ENV_PATH, configure_logging, resolve_llm_settings


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "output" / "transcript_cleaned.txt"
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "transcript_summary.log"
# 单次 chat/completions 请求超时（秒）
DEFAULT_REQUEST_TIMEOUT = 300
# 网络抖动或服务端瞬时错误时的最大重试次数（含首次请求共 N 次尝试）
DEFAULT_MAX_RETRIES = 3
# 全文字符数不超过此值时走单次总结；针对 SiliconFlow DeepSeek-V3（约 164K 上下文）的保守线
DIRECT_SUMMARY_CHAR_BUDGET = 48000
# 超过直出阈值后，按该目标长度分段做「分段摘要」（按字符近似，非精确 token）
CHUNK_CHAR_BUDGET = 24000
# 分段仍超长需再拆时，单段不得低于此长度，避免语义切碎
MIN_CHUNK_CHAR_BUDGET = 6000
# 这些状态码视为可退避重试（401/403/400 等一般不重试）
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
# 切分 transcript 时在此类边界后断开，元组顺序即优先顺序（越靠前越先尝试）
CHUNK_BREAK_MARKERS = (
    "\n\n",
    "\n",
    "。", "！", "？",
    ". ", "! ", "? ",
    "；", "; ",
    "，", ", ",
    "、", " ",
)

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

CHUNK_SUMMARY_USER_PROMPT_TEMPLATE = """以下是长转写文本的一个分段，标识：{chunk_label}。

请只提炼这一段最重要的信息，供后续总汇总使用。请用简体中文输出以下结构，不要增加额外前言：

### 分段概览
用 2-4 句概括这段主要在讲什么。

### 关键主张
提取 1-3 条本段最重要的观点。每条必须包含：
- 标题：
- 说明：
- 依据原文：
- 中文直译：

### 风险与不确定性
- 若存在营销口径、个人叙事、口误/听写疑点、证据不足、产品能力未被充分证明等，逐条列出。

要求：
- 只总结当前这一段，不要补写本段没有出现的事实。
- “依据原文”必须直接摘录本段原文中的连续短语或句子，不要编造。
- 若本段信息有限，也要如实说明，不要为了凑条数而扩写。

【当前分段原文】：
{transcript}
"""

MERGE_SUMMARY_USER_PROMPT_TEMPLATE = """你将收到同一份长转写文本的多段摘要草稿。每段草稿已经包含该段的主张、原文摘录和中文直译。

请基于这些分段摘要，输出最终结果，结构严格如下：

### A. 内容总结与“主张—依据—中译”
1) 先用 5-10 句中文概括全文主线（也可说明其语气更像演讲/口述/推广/教程中的哪一种）。
2) 提取 5-8 条“核心观点/主张”。每条必须包含：
   - 主张标题（简短）
   - 主张说明（1-3 句中文）
   - 依据：1-2 段【转写原文英文摘录】（只能使用分段摘要中已经出现的摘录）
   - 对应【中文翻译】（忠实直译为主，专有名词不确定则标注“疑似：...”）

要求：
- 不要把营销数字（例如节省 X 小时）当成事实；若出现，标注为“演讲者声称/需核验”。
- 若你发现明显的 ASR 误差，在段末用括注提示“疑似听写：...->...”。
- 当原文同时出现“工具隐喻/宣传口号”和“具体操作链路”时，优先总结具体操作链路。
- 只能使用分段摘要里已有信息，不要补写新的事实或新的原文引文。
- 若证据不足，请明确标注“需核验”。

### B. X 汇报（演讲者主张框架版）
把上述观点整理成适合在 X（Twitter）发布的“串帖/Thread”体例（10 条以内），要求：
- 中性语气：这是“演讲者主张框架”，不是你已经验证的结论。
- 每条 1-3 句中文；第一条是总览，最后一条写风险提示。
- 不要输出 hashtag；不要编造转写里没有的步骤或产品功能。
- 若某条观点涉及工具协同、知识库接入、资料比对，thread 中必须写清“原文中提供的信息是怎么结合”的。

【分段摘要】：
{chunk_summaries}
"""

CONTEXT_LIMIT_PATTERNS = (
    "context length",
    "maximum context",
    "max context",
    "too many tokens",
    "token limit",
    "prompt is too long",
    "input is too long",
    "request too large",
    "context_window_exceeded",
    "maximum length",
)


def _extract_llm_content(payload: dict) -> str:
    return (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )


def _response_body_excerpt(response: httpx.Response) -> str:
    try:
        body = response.text.strip()
    except Exception:
        return ""
    if not body:
        return ""
    body = re.sub(r"\s+", " ", body)
    return body[:240]


def _looks_like_context_limit_error(response: httpx.Response) -> bool:
    if response.status_code == 413:
        return True
    body_excerpt = _response_body_excerpt(response).lower()
    return response.status_code in {400, 414} and any(
        pattern in body_excerpt for pattern in CONTEXT_LIMIT_PATTERNS
    )


def _split_transcript_into_chunks(transcript: str, max_chars: int) -> list[str]:
    normalized = transcript.strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    start = 0
    total_length = len(normalized)

    while start < total_length:
        remaining = total_length - start
        if remaining <= max_chars:
            tail = normalized[start:].strip()
            if tail:
                chunks.append(tail)
            break

        lower_bound = min(start + MIN_CHUNK_CHAR_BUDGET, total_length)
        upper_bound = min(start + max_chars, total_length)
        split_idx = -1

        for marker in CHUNK_BREAK_MARKERS:
            candidate = normalized.rfind(marker, lower_bound, upper_bound)
            if candidate != -1:
                split_idx = candidate + len(marker)
                break

        if split_idx <= start:
            split_idx = upper_bound

        chunk = normalized[start:split_idx].strip()
        if chunk:
            chunks.append(chunk)
        start = split_idx

    return chunks


def _call_llm_with_retry(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
) -> tuple[str, float]:
    started = time.time()

    for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
        try:
            response = httpx.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "temperature": 0,
                    "messages": messages,
                },
                timeout=DEFAULT_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            content = _extract_llm_content(payload)
            if not content:
                raise RuntimeError("LLM returned empty summary content")
            return content, round(time.time() - started, 3)

        except httpx.HTTPStatusError as exc:
            response = exc.response
            if _looks_like_context_limit_error(response):
                raise RuntimeError(
                    "LLM request was rejected because the transcript or prompt is too long for the configured model."
                ) from exc

            message = f"LLM request failed with HTTP {response.status_code}"
            body_excerpt = _response_body_excerpt(response)
            if body_excerpt:
                message = f"{message}: {body_excerpt}"
            retryable = response.status_code in RETRYABLE_STATUS_CODES
            logging.warning("LLM request attempt %s/%s failed: %s", attempt, DEFAULT_MAX_RETRIES, message)
            if not retryable or attempt == DEFAULT_MAX_RETRIES:
                raise RuntimeError(message) from exc

        except (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError) as exc:
            message = f"LLM request failed due to network instability: {exc}"
            logging.warning("LLM request attempt %s/%s failed: %s", attempt, DEFAULT_MAX_RETRIES, message)
            if attempt == DEFAULT_MAX_RETRIES:
                raise RuntimeError(message) from exc

        except httpx.HTTPError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        if attempt < DEFAULT_MAX_RETRIES:
            time.sleep(2 ** (attempt - 1))

    raise RuntimeError("LLM request failed after retries")


def _summarize_chunk_recursive(
    *,
    transcript: str,
    api_key: str,
    base_url: str,
    model: str,
    chunk_label: str,
    depth: int = 0,
) -> tuple[list[str], float]:
    try:
        content, elapsed = _call_llm_with_retry(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": CHUNK_SUMMARY_USER_PROMPT_TEMPLATE.format(
                        chunk_label=chunk_label,
                        transcript=transcript,
                    ),
                },
            ],
        )
        logging.info("Chunk %s summarized in %.3fs (%s chars)", chunk_label, elapsed, len(transcript))
        return [f"## Chunk {chunk_label}\n\n{content}"], elapsed
    except RuntimeError as exc:
        if (
            "too long for the configured model" in str(exc)
            and len(transcript) > MIN_CHUNK_CHAR_BUDGET
            and depth < 6
        ):
            smaller_budget = max(MIN_CHUNK_CHAR_BUDGET, len(transcript) // 2)
            sub_chunks = _split_transcript_into_chunks(transcript, smaller_budget)
            if len(sub_chunks) <= 1:
                raise

            logging.warning(
                "Chunk %s is still too large; degrading into %s sub-chunks",
                chunk_label,
                len(sub_chunks),
            )
            nested_summaries: list[str] = []
            total_elapsed = 0.0
            for index, sub_chunk in enumerate(sub_chunks, start=1):
                summaries, elapsed = _summarize_chunk_recursive(
                    transcript=sub_chunk,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    chunk_label=f"{chunk_label}.{index}",
                    depth=depth + 1,
                )
                nested_summaries.extend(summaries)
                total_elapsed += elapsed
            return nested_summaries, total_elapsed
        raise


def summarize_transcript_with_llm(
    transcript: str,
    env_path: str | Path = DEFAULT_ENV_PATH,
) -> tuple[str, str, float]:
    api_key, base_url, model = resolve_llm_settings(env_path)
    if not api_key or not model:
        raise RuntimeError("LLM settings are incomplete in local .env")

    transcript = transcript.strip()
    if len(transcript) <= DIRECT_SUMMARY_CHAR_BUDGET:
        content, elapsed = _call_llm_with_retry(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": SUMMARY_USER_PROMPT_TEMPLATE.format(transcript=transcript),
                },
            ],
        )
        return content, model, elapsed

    chunks = _split_transcript_into_chunks(transcript, CHUNK_CHAR_BUDGET)
    logging.info(
        "Transcript exceeds direct summary budget (%s chars); split into %s chunks",
        len(transcript),
        len(chunks),
    )

    chunk_summaries: list[str] = []
    total_elapsed = 0.0
    for index, chunk in enumerate(chunks, start=1):
        summaries, elapsed = _summarize_chunk_recursive(
            transcript=chunk,
            api_key=api_key,
            base_url=base_url,
            model=model,
            chunk_label=f"{index}/{len(chunks)}",
        )
        chunk_summaries.extend(summaries)
        total_elapsed += elapsed

    merged_content, merge_elapsed = _call_llm_with_retry(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": MERGE_SUMMARY_USER_PROMPT_TEMPLATE.format(
                    chunk_summaries="\n\n".join(chunk_summaries)
                ),
            },
        ],
    )
    total_elapsed += merge_elapsed
    return merged_content, model, round(total_elapsed, 3)


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
