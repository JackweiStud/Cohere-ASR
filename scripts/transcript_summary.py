#!/usr/bin/env python3
"""基于转写文本生成分析包。

职责范围
    只负责读取已有的转写文本，调用配置好的 LLM 输出分析包：
    主张 / 证据 / 风险提示 / X 汇报
    不负责语音转写；语音转写请使用 ``poc_cohere_local_transcribe.py``。
    不负责 minimum-edit cleanup；转写清洗请使用 ``transcript_cleanup.py``。

使用示例（建议在项目虚拟环境中执行）::
    /path/to/.venv/bin/python scripts/transcript_summary.py \
        --input output/transcript_cleaned.txt

主要参数
    --input     输入转写文本路径。默认 ``output/transcript_cleaned.txt``。
    --output    可选；分析包 Markdown 路径。
    --log-path  日志文件路径。默认 ``logs/transcript_summary.log``。

LLM 配置
    默认读取项目根目录下的 ``.env``（与 ``scripts/`` 同级），通常不必传入 ``--env-path``。
    仅在需要使用非默认路径时，才传入 ``--env-path`` 覆盖。

注意事项
    - 需要在环境变量或项目根 ``.env`` 中提供 ``LLM_API_KEY``；``LLM_BASE_URL`` 和
      ``LLM_MODEL`` 可选，不填则使用默认值。
    - 当前仅接受 ``.txt`` 输入；若输入不是文本转写结果，脚本会记录错误日志并给出建议。
    - 本脚本使用 ``temperature=0``，尽量让输出结构稳定、可复现。
    - 最终 Markdown 会在后处理阶段优先按 ``. ? ! 。？！`` 做断句换行，提升可读性。
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
DEFAULT_REQUEST_TIMEOUT = 300
DEFAULT_MAX_RETRIES = 3
MAX_GENERATION_ATTEMPTS = 2
DIRECT_SUMMARY_CHAR_BUDGET = 48000
CHUNK_CHAR_BUDGET = 24000
MIN_CHUNK_CHAR_BUDGET = 6000
MIN_CLAIM_ITEMS = 5
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
CHUNK_BREAK_MARKERS = (
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    ". ",
    "! ",
    "? ",
    "；",
    "; ",
    "，",
    ", ",
    "、",
    " ",
)
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
SENTENCE_BREAK_PATTERN = re.compile(r"(?<=[.?!。？！])\s+")

SUMMARY_SYSTEM_PROMPT = (
    "你是一位双语信息架构师。"
    "请阅读用户提供的完整转写文本。"
    "转写文本可能来自 ASR，可能含口误或听写错误。"
    "请严格按照用户要求的 Markdown 结构输出简体中文结果，不要增加额外前言或说明。"
)

FORMAT_RULES = """
Markdown 格式要求：
- 所有输出必须是干净的 Markdown，不要写前言或额外说明。
- 为了提升可读性，句末优先按 `. ? ! 。？！` 断行。
- 不要输出超长自然段；一段里若有多句，请逐句换行。
- 英文引文若包含多句，请放在同一个引用块中逐句换行。
"""

ANALYSIS_DIRECT_PROMPT_TEMPLATE = """你将阅读一份完整的视频转写文本，并输出“分析包”。

目标：
帮助用户快速判断作者到底在主张什么、拿了什么证据、哪些部分值得信、哪些部分需要警惕。

请严格输出以下结构：

### A. 内容主线总结
用 5-8 句中文概括全文主线。
每句单独成行。
可点明这是演讲 / 推广 / 教程 / 经验分享中的哪一种语气。

### B. 核心主张拆解
提取 5-8 条最重要的主张。
每条严格使用下面的字段格式：

**主张1：标题**
- 主张说明：
- 证据原文：
  > 英文原文摘录
- 中文直译：
  > 忠实直译
- 类型判断：事实 / 观点 / 混合
- 依据强度：强 / 中 / 弱
- 风险标签：营销话术 / 夸大表达 / 听写疑点 / 需外部核验 / 个人经验叙事 / 无明显风险
- 警惕原因：

要求：
- “证据原文”必须直接摘录转写里的连续短语或句子，不要编造。
- 若原文只有个人经验或宣传口号，明确指出证据强度偏弱。
- 不要把营销数字或效果承诺直接当成事实。
- 对未经外部验证的效果、数字、因果、优劣结论，一律使用“作者声称 / 演讲者主张 / 原文展示 / 演示中展示”等归因措辞，不要改写成客观事实。
- 若发现明显 ASR 疑点，可在中文直译或警惕原因中标注“疑似听写：A -> B”。
- 风险提示只能基于 transcript 中已出现的信息，不得脑补隐藏动机。
- 除非原文明确提到，否则不要推断“商业合作倾向 / 收钱推荐 / 广告植入”等隐藏关系。

### C. 风险提示与核验点
至少包含以下 4 组内容：
- 事实性内容：哪些内容更接近可直接采信的信息
- 需二次核验：哪些数字、因果、效果结论需要外部验证
- 营销与夸大表达：哪些句子更像宣传、包装或情绪推动
- ASR / 术语疑点：哪些专有名词、工具名、数据点可能听错

### D. X 汇报
把上述观点整理成适合在 X（Twitter）发布的“串帖 / Thread”体例，10 条以内：
- 中性语气：这是“演讲者主张框架”，不是你已经验证的结论。
- 每条 1-3 句。
- 第一条做总览，最后一条做风险提示。
- 不要输出 hashtag。
- 若涉及工具协同、知识库接入、资料比对，必须写清原文中的组合方式。

{format_rules}

【完整转写文本】：
{transcript}
"""

CHUNK_SUMMARY_USER_PROMPT_TEMPLATE = """以下是长转写文本的一个分段，标识：{chunk_label}。

请只提炼这一段最重要的信息，供后续总汇总使用。
请严格输出以下结构，不要增加额外前言：

### 分段概览
用 2-4 句概括这段在讲什么。
每句单独成行。

### 分段主张证据库
提取 1-3 条本段最重要的主张。
每条使用以下字段：

**主张1：标题**
- 主张说明：
- 证据原文：
- 中文直译：
- 类型判断：事实 / 观点 / 混合
- 依据强度：强 / 中 / 弱
- 风险标签：营销话术 / 夸大表达 / 听写疑点 / 需外部核验 / 个人经验叙事 / 无明显风险
- 警惕原因：

### 分段风险与不确定性
- 若存在营销口径、个人叙事、口误 / 听写疑点、证据不足、产品能力未被充分证明等，逐条列出。

要求：
- 只总结当前这一段，不要补写本段没有出现的事实。
- “证据原文”必须直接摘录本段原文中的连续短语或句子，不要编造。
- 若本段信息有限，也要如实说明，不要为了凑条数而扩写。
- 对未经验证的效果、数字、因果，一律保留“作者声称 / 演讲者主张”的归因语气。
- 不要推断隐藏动机或商业关系。
- 句末优先按 `. ? ! 。？！` 断行，避免长段。

【当前分段原文】：
{transcript}
"""

ANALYSIS_MERGE_PROMPT_TEMPLATE = """你将收到同一份长转写文本的多段摘要草稿。
每段草稿已经包含分段主张、原文摘录、中文直译、类型判断和风险标签。

请基于这些分段摘要，输出最终“分析包”，结构严格如下：

### A. 内容主线总结
用 5-8 句中文概括全文主线。
每句单独成行。

### B. 核心主张拆解
提取 5-8 条最重要的主张。
每条严格使用下面的字段格式：

**主张1：标题**
- 主张说明：
- 证据原文：
  > 英文原文摘录
- 中文直译：
  > 忠实直译
- 类型判断：事实 / 观点 / 混合
- 依据强度：强 / 中 / 弱
- 风险标签：营销话术 / 夸大表达 / 听写疑点 / 需外部核验 / 个人经验叙事 / 无明显风险
- 警惕原因：

### C. 风险提示与核验点
- 事实性内容：
- 需二次核验：
- 营销与夸大表达：
- ASR / 术语疑点：

### D. X 汇报
把上述观点整理成适合在 X（Twitter）发布的“串帖 / Thread”体例，10 条以内：
- 中性语气：这是“演讲者主张框架”，不是你已经验证的结论。
- 每条 1-3 句。
- 第一条做总览，最后一条做风险提示。
- 不要输出 hashtag。
- 若涉及工具协同、知识库接入、资料比对，必须写清原文中的组合方式。

要求：
- 只能使用分段摘要里已有信息，不要补写新的事实或新的原文引文。
- 若证据不足，请明确标注“需核验”。
- 不要把营销数字或效果承诺当成事实。
- 对未经外部验证的效果、数字、因果、优劣结论，一律使用“作者声称 / 演讲者主张 / 原文展示 / 演示中展示”等归因措辞。
- 风险提示只能基于 transcript 已出现的信息，不得脑补隐藏动机。
- 除非原文明确提到，否则不要推断“商业合作倾向 / 收钱推荐 / 广告植入”等隐藏关系。
- 句末优先按 `. ? ! 。？！` 断行，避免长段。

【分段摘要】：
{chunk_summaries}
"""

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


def _request_markdown_output(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
) -> tuple[str, float]:
    return _call_llm_with_retry(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )


def _count_pattern(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def _section_lines(markdown: str, heading: str) -> list[str]:
    pattern = re.compile(
        rf"(?ms)^###\s+{re.escape(heading)}\n(.*?)(?=^###\s+|\Z)"
    )
    match = pattern.search(markdown)
    if not match:
        return []
    return [line.strip() for line in match.group(1).splitlines() if line.strip()]


def _validate_analysis_markdown(markdown: str) -> list[str]:
    issues: list[str] = []
    claim_count = _count_pattern(r"^\*\*主张\d+：", markdown)
    if claim_count < MIN_CLAIM_ITEMS:
        issues.append(f"核心主张数量不足，至少需要 {MIN_CLAIM_ITEMS} 条。")

    summary_lines = _section_lines(markdown, "A. 内容主线总结")
    if summary_lines:
        attributed_lines = sum(
            1 for line in summary_lines if any(token in line for token in ("作者", "演讲者", "原文", "演示"))
        )
        if attributed_lines < max(2, min(3, len(summary_lines))):
            issues.append("内容主线总结对未经验证的结论归因不足，应更多使用“作者声称/演讲者主张/原文展示”等措辞。")

    forbidden_hidden_motive_phrases = (
        "商业合作倾向",
        "收钱推荐",
        "广告植入",
        "恰饭",
        "利益输送",
    )
    if any(phrase in markdown for phrase in forbidden_hidden_motive_phrases):
        issues.append("出现了 transcript 未明确支持的隐藏动机推断，应删除这类脑补风险。")

    return issues


def _generate_with_validation(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    validator,
    label: str,
) -> tuple[str, float]:
    total_elapsed = 0.0
    effective_prompt = prompt
    last_issues: list[str] = []

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        content, elapsed = _request_markdown_output(
            api_key=api_key,
            base_url=base_url,
            model=model,
            prompt=effective_prompt,
        )
        total_elapsed += elapsed
        issues = validator(content)
        if not issues:
            return content, round(total_elapsed, 3)

        last_issues = issues
        logging.warning("%s validation failed on attempt %s: %s", label, attempt, "; ".join(issues))
        if attempt == MAX_GENERATION_ATTEMPTS:
            break

        effective_prompt = (
            f"{prompt}\n\n"
            "上一次输出未达标，请只修正以下问题后重新完整输出，不要解释：\n"
            + "\n".join(f"- {issue}" for issue in issues)
        )

    raise RuntimeError(f"{label} generation failed validation: {'; '.join(last_issues)}")


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
        content, elapsed = _request_markdown_output(
            api_key=api_key,
            base_url=base_url,
            model=model,
            prompt=CHUNK_SUMMARY_USER_PROMPT_TEMPLATE.format(
                chunk_label=chunk_label,
                transcript=transcript,
            ),
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


def _split_sentence_like(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    return [part.strip() for part in SENTENCE_BREAK_PATTERN.split(normalized) if part.strip()]


def _format_markdown_line(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped:
        return [""]
    if stripped.startswith("#") or stripped.startswith("|"):
        return [line.rstrip()]

    quote_match = re.match(r"^(\s*>\s?)(.*)$", line)
    if quote_match:
        prefix, content = quote_match.groups()
        parts = _split_sentence_like(content)
        return [f"{prefix}{part}" for part in parts] or [line.rstrip()]

    list_match = re.match(r"^(\s*(?:[-*+]|\d+\.)\s+)(.*)$", line)
    if list_match:
        prefix, content = list_match.groups()
        parts = _split_sentence_like(content)
        if len(parts) <= 1:
            return [line.rstrip()]
        continuation_prefix = " " * len(prefix)
        return [f"{prefix}{parts[0]}"] + [f"{continuation_prefix}{part}" for part in parts[1:]]

    parts = _split_sentence_like(line)
    return parts or [line.rstrip()]


def _format_markdown_sentences(markdown: str) -> str:
    lines = markdown.strip().splitlines()
    if not lines:
        return ""

    formatted_lines: list[str] = []
    in_code_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            formatted_lines.append(line.rstrip())
            continue
        if in_code_block:
            formatted_lines.append(line.rstrip())
            continue
        formatted_lines.extend(_format_markdown_line(line.rstrip()))

    return "\n".join(formatted_lines).strip() + "\n"


def _derive_output_path(input_path: Path, output_arg: str) -> Path:
    if not output_arg:
        return input_path.with_name(f"{input_path.stem}_analysis.md")
    return Path(output_arg).expanduser().resolve()


def summarize_transcript_with_llm(
    transcript: str,
    env_path: str | Path = DEFAULT_ENV_PATH,
) -> tuple[str, str, float]:
    api_key, base_url, model = resolve_llm_settings(env_path)
    if not api_key or not model:
        raise RuntimeError("LLM settings are incomplete in local .env")

    transcript = transcript.strip()
    if len(transcript) <= DIRECT_SUMMARY_CHAR_BUDGET:
        analysis_content, analysis_elapsed = _generate_with_validation(
            api_key=api_key,
            base_url=base_url,
            model=model,
            prompt=ANALYSIS_DIRECT_PROMPT_TEMPLATE.format(
                transcript=transcript,
                format_rules=FORMAT_RULES.strip(),
            ),
            validator=_validate_analysis_markdown,
            label="analysis pack",
        )
        return (
            _format_markdown_sentences(analysis_content),
            model,
            round(analysis_elapsed, 3),
        )

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

    chunk_summary_text = "\n\n".join(chunk_summaries)
    analysis_content, analysis_elapsed = _generate_with_validation(
        api_key=api_key,
        base_url=base_url,
        model=model,
        prompt=ANALYSIS_MERGE_PROMPT_TEMPLATE.format(chunk_summaries=chunk_summary_text),
        validator=_validate_analysis_markdown,
        label="analysis pack",
    )
    total_elapsed += analysis_elapsed
    return (
        _format_markdown_sentences(analysis_content),
        model,
        round(total_elapsed, 3),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Structured analysis pack from transcript text")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to transcript txt file",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional analysis markdown output path",
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

    analysis_path = _derive_output_path(input_path, args.output)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Starting transcript summary")
    logging.info("Input path: %s", input_path)
    logging.info("Output path: %s", analysis_path)
    logging.info("Env path: %s", Path(args.env_path).expanduser().resolve())

    transcript = input_path.read_text(encoding="utf-8").strip()
    if not transcript:
        logging.warning("Input transcript is empty: %s", input_path)
        print(f"Input transcript is empty: {input_path}", file=sys.stderr)
        return 1

    try:
        analysis_md, model, elapsed = summarize_transcript_with_llm(
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

    analysis_path.write_text(analysis_md, encoding="utf-8")

    logging.info("Transcript summary finished in %.3fs", elapsed)
    logging.info("Summary model: %s", model)

    print(f"Analysis written to: {analysis_path}")
    print(f"Summary model: {model}")
    print(f"Summary seconds: {elapsed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
