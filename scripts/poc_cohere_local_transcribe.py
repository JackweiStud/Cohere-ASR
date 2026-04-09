#!/usr/bin/env python3
"""本地 Cohere Transcribe PoC：加载模型、可选按体积分片、转写、拼接与报告。

职责范围
    只负责语音识别与分片转写管线。不在此脚本中做文稿清洗；清洗请单独运行
    ``transcript_cleanup.py``（对已生成的 ``transcript.txt`` 等进行处理）。

使用示例（建议在项目虚拟环境中执行）::
    /path/to/.venv/bin/python scripts/poc_cohere_local_transcribe.py \\
        --input input/extracted_audio.wav \\
        --output-dir output \\
        --language en \\
        --split-threshold-mb 25 \\
        --chunk-target-mb 15

主要参数
    --input              单个音频文件路径。默认 ``input/extracted_audio.wav``。当前仅接受 ``.wav``；
                         若输入为其他格式，脚本会记录错误日志并提示先转换为 WAV。
    --output-dir         输出目录。默认 ``output/``。
    --model-path         可选；本地模型目录。默认优先使用
                         ``models/CohereLabs/cohere-transcribe-03-2026``；
                         若该目录不存在，则回退到 Hugging Face ID
                         ``CohereLabs/cohere-transcribe-03-2026``。
    --language           ISO 639-1 语言码。默认 ``en``。
    --no-punctuation     关闭标点。默认关闭该开关，即保留标点。
    --split-threshold-mb 输入文件大小（MiB）超过该值则切分；否则整段转写。默认 ``25``。
    --chunk-target-mb    切分时，按「源文件字节/时长」估算每片时长，使分片体积大致接近该目标
                         （近似值）。默认 ``15``。
    --log-path           日志文件路径。默认 ``logs/poc_cohere_local_transcribe.log``。

输出
    transcript.txt        全文拼接结果（分片之间用换行连接）。
    report.json           运行与资源等指标。
    chunks_manifest.json  各分片路径、时间范围、体积与耗时等。
    chunks/               切分得到的 WAV 分片（仅当发生切分时写入）。
    transcripts/          各分片对应的纯文本。

分片实现
    使用 ``soundfile``（libsndfile）按采样率与目标秒数读帧、写 WAV；不使用 FFmpeg。

注意事项
    - ``chunk-target-mb`` 是按**源文件**平均码率估算的“约 15MB”类目标；若源为压缩格式而分片
      落盘为 WAV，单文件实际 MiB 可能与直觉不完全一致；PCM WAV 输入时通常更接近期望。
    - 硬切分点在时间上连续，句中截断时拼接处可能需靠后续 ``transcript_cleanup`` 微调。
    - 当前仅接受 ``.wav`` 输入；若源文件不是 WAV，请先在上游完成抽音频/转码，再运行本脚本。
    - 依赖 torch、transformers、soundfile、psutil 等；缺依赖时脚本会提示先配置虚拟环境。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "input" / "extracted_audio.wav"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "CohereLabs" / "cohere-transcribe-03-2026"
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "poc_cohere_local_transcribe.log"


@dataclass
class ChunkResult:
    index: int
    chunk_path: str
    transcript_path: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    file_size_mb: float
    transcribe_seconds: float
    transcript_chars: int
    transcript_words: int


@dataclass
class Report:
    model_id: str
    model_path: str
    input_path: str
    output_dir: str
    language: str
    punctuation: bool
    split_threshold_mb: float
    chunk_target_mb: float
    used_split: bool
    chunk_count: int
    python_version: str
    torch_version: str
    transformers_version: str
    device: str
    duration_seconds: float
    file_size_mb: float
    rss_before_mb: float
    rss_after_load_mb: float
    rss_after_transcribe_mb: float
    load_seconds: float
    transcribe_seconds: float
    transcript_chars: int
    transcript_words: int
    transcript_path: str
    manifest_path: str


def _rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def _get_audio_info(audio_path: Path) -> dict[str, float | int | str]:
    import soundfile as sf

    info = sf.info(str(audio_path))
    duration_seconds = round(float(info.frames) / float(info.samplerate), 3)
    return {
        "duration_seconds": duration_seconds,
        "samplerate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


def _resolve_device(torch_module) -> str:
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def _configure_logging(log_path: str) -> None:
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


def _resolve_model_path(explicit_model_path: str | None) -> str:
    if explicit_model_path:
        return str(Path(explicit_model_path).expanduser().resolve())

    default_path = Path(DEFAULT_LOCAL_MODEL_PATH)
    if default_path.exists():
        return str(default_path.resolve())

    return MODEL_ID


def _load_model(model_path: str, device: str):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    return processor, model


def _estimate_chunk_seconds(file_size_bytes: int, duration_seconds: float, target_mb: float) -> float:
    if duration_seconds <= 0:
        return 1.0

    bytes_per_second = file_size_bytes / duration_seconds
    if bytes_per_second <= 0:
        return duration_seconds

    target_seconds = (target_mb * 1024 * 1024) / bytes_per_second
    return max(1.0, round(target_seconds, 3))


def _write_audio_chunks(
    input_path: Path,
    chunks_dir: Path,
    chunk_seconds: float,
) -> list[dict[str, float | int | str]]:
    import soundfile as sf

    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = sorted(chunks_dir.glob("chunk_*.wav"))
    for existing_file in chunk_paths:
        existing_file.unlink()

    manifests: list[dict[str, float | int | str]] = []
    with sf.SoundFile(str(input_path)) as source:
        samplerate = int(source.samplerate)
        total_frames = len(source)
        chunk_frames = max(1, int(chunk_seconds * samplerate))
        chunk_index = 1
        start_frame = 0

        while start_frame < total_frames:
            frames_to_read = min(chunk_frames, total_frames - start_frame)
            data = source.read(frames_to_read)
            end_frame = start_frame + frames_to_read
            chunk_path = chunks_dir / f"chunk_{chunk_index:04d}.wav"
            sf.write(str(chunk_path), data, samplerate)
            manifests.append(
                {
                    "index": chunk_index,
                    "chunk_path": str(chunk_path),
                    "start_seconds": round(start_frame / samplerate, 3),
                    "end_seconds": round(end_frame / samplerate, 3),
                    "duration_seconds": round(frames_to_read / samplerate, 3),
                    "file_size_mb": round(chunk_path.stat().st_size / (1024 * 1024), 2),
                }
            )
            chunk_index += 1
            start_frame = end_frame

    return manifests


def _build_chunk_plan(
    input_path: Path,
    output_dir: Path,
    duration_seconds: float,
    file_size_bytes: int,
    split_threshold_mb: float,
    chunk_target_mb: float,
) -> tuple[list[dict[str, float | int | str]], bool]:
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb <= split_threshold_mb:
        return [
            {
                "index": 1,
                "chunk_path": str(input_path),
                "start_seconds": 0.0,
                "end_seconds": duration_seconds,
                "duration_seconds": duration_seconds,
                "file_size_mb": round(file_size_mb, 2),
            }
        ], False

    chunk_seconds = _estimate_chunk_seconds(
        file_size_bytes=file_size_bytes,
        duration_seconds=duration_seconds,
        target_mb=chunk_target_mb,
    )
    logging.info(
        "Input exceeds %.2fMB, splitting audio into chunks of about %.3fs",
        split_threshold_mb,
        chunk_seconds,
    )
    chunk_manifests = _write_audio_chunks(
        input_path=input_path,
        chunks_dir=output_dir / "chunks",
        chunk_seconds=chunk_seconds,
    )
    return chunk_manifests, True


def _transcribe_one(
    model,
    processor,
    torch_module,
    audio_path: str,
    language: str,
    punctuation: bool,
) -> str:
    with torch_module.inference_mode():
        texts = model.transcribe(
            processor=processor,
            audio_files=[audio_path],
            language=language,
            punctuation=punctuation,
        )
    if not texts:
        return ""
    return str(texts[0]).strip()


def _transcribe_chunks(
    model,
    processor,
    torch_module,
    chunk_manifests: list[dict[str, float | int | str]],
    transcripts_dir: Path,
    language: str,
    punctuation: bool,
) -> list[ChunkResult]:
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    results: list[ChunkResult] = []

    for chunk_manifest in chunk_manifests:
        chunk_index = int(chunk_manifest["index"])
        chunk_path = str(chunk_manifest["chunk_path"])
        transcript_path = transcripts_dir / f"chunk_{chunk_index:04d}.txt"
        logging.info(
            "Transcribing chunk %s: %s (%.3fs -> %.3fs)",
            chunk_index,
            chunk_path,
            float(chunk_manifest["start_seconds"]),
            float(chunk_manifest["end_seconds"]),
        )
        started = time.time()
        transcript = _transcribe_one(
            model=model,
            processor=processor,
            torch_module=torch_module,
            audio_path=chunk_path,
            language=language,
            punctuation=punctuation,
        )
        elapsed = round(time.time() - started, 3)
        transcript_path.write_text(transcript + "\n", encoding="utf-8")
        results.append(
            ChunkResult(
                index=chunk_index,
                chunk_path=chunk_path,
                transcript_path=str(transcript_path),
                start_seconds=float(chunk_manifest["start_seconds"]),
                end_seconds=float(chunk_manifest["end_seconds"]),
                duration_seconds=float(chunk_manifest["duration_seconds"]),
                file_size_mb=float(chunk_manifest["file_size_mb"]),
                transcribe_seconds=elapsed,
                transcript_chars=len(transcript),
                transcript_words=len(transcript.split()),
            )
        )
        logging.info(
            "Chunk %s finished in %.3fs, chars=%s, words=%s",
            chunk_index,
            elapsed,
            len(transcript),
            len(transcript.split()),
        )

    return results


def _merge_transcripts(chunk_results: list[ChunkResult]) -> str:
    parts: list[str] = []
    for chunk_result in chunk_results:
        text = Path(chunk_result.transcript_path).read_text(encoding="utf-8").strip()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Cohere Transcribe PoC runner")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to one local audio file",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Optional local model directory. Defaults to the test workspace model path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for transcript and report",
    )
    parser.add_argument(
        "--log-path",
        default=DEFAULT_LOG_PATH,
        help="Path to the log file",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="ISO 639-1 language code. Cohere requires an explicit language.",
    )
    parser.add_argument(
        "--split-threshold-mb",
        type=float,
        default=25.0,
        help="Split the input audio when its size exceeds this threshold.",
    )
    parser.add_argument(
        "--chunk-target-mb",
        type=float,
        default=15.0,
        help="Approximate target size for each audio chunk after splitting.",
    )
    parser.add_argument(
        "--no-punctuation",
        action="store_true",
        help="Disable punctuation in model output.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(Path(args.log_path).expanduser().resolve())
    _configure_logging(log_path)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    if input_path.suffix.lower() != ".wav":
        logging.error("Unsupported input format: %s", input_path.suffix or "<no suffix>")
        logging.error("This script currently accepts only .wav input: %s", input_path)
        print(
            "Unsupported input format. This script currently accepts only .wav input.\n"
            f"Input path: {input_path}\n"
            "Suggestion: convert the source audio to WAV first, then rerun this script.",
            file=sys.stderr,
        )
        return 1

    if args.split_threshold_mb <= 0 or args.chunk_target_mb <= 0:
        print("split-threshold-mb and chunk-target-mb must be positive numbers", file=sys.stderr)
        return 1

    try:
        import torch
        import transformers
    except Exception as exc:
        print(
            "Missing dependencies. Run ./scripts/setup_venv.sh first.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 2

    logging.info("Starting local Cohere PoC")
    logging.info("Input path: %s", input_path)
    logging.info("Output dir: %s", output_dir)
    logging.info("Requested language: %s", args.language)
    logging.info("Split threshold: %.2fMB", args.split_threshold_mb)
    logging.info("Chunk target: %.2fMB", args.chunk_target_mb)

    device = _resolve_device(torch)
    model_path = _resolve_model_path(args.model_path or None)
    audio_info = _get_audio_info(input_path)
    duration_seconds = float(audio_info["duration_seconds"])
    file_size_bytes = input_path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    rss_before_mb = _rss_mb()
    logging.info("Resolved model path: %s", model_path)
    logging.info("Device: %s", device)
    logging.info("Audio duration: %.3fs", duration_seconds)
    logging.info("Audio size: %.2fMB", file_size_mb)
    logging.info("RSS before load: %.2fMB", rss_before_mb)

    if transformers.__version__.startswith(("5.3", "5.4", "5.5", "5.6")):
        logging.warning(
            "transformers %s is outside the version range recommended by the model card",
            transformers.__version__,
        )

    chunk_manifests, used_split = _build_chunk_plan(
        input_path=input_path,
        output_dir=output_dir,
        duration_seconds=duration_seconds,
        file_size_bytes=file_size_bytes,
        split_threshold_mb=args.split_threshold_mb,
        chunk_target_mb=args.chunk_target_mb,
    )
    logging.info("Prepared %s chunk(s)", len(chunk_manifests))

    load_start = time.time()
    processor, model = _load_model(model_path, device)
    load_seconds = round(time.time() - load_start, 3)
    rss_after_load_mb = _rss_mb()
    logging.info("Model loaded in %.3fs", load_seconds)
    logging.info("RSS after load: %.2fMB", rss_after_load_mb)

    transcribe_start = time.time()
    chunk_results = _transcribe_chunks(
        model=model,
        processor=processor,
        torch_module=torch,
        chunk_manifests=chunk_manifests,
        transcripts_dir=output_dir / "transcripts",
        language=args.language,
        punctuation=not args.no_punctuation,
    )
    transcript = _merge_transcripts(chunk_results)
    transcribe_seconds = round(time.time() - transcribe_start, 3)
    rss_after_transcribe_mb = _rss_mb()
    logging.info("Transcription finished in %.3fs", transcribe_seconds)
    logging.info("RSS after transcribe: %.2fMB", rss_after_transcribe_mb)
    logging.info("Transcript chars: %s", len(transcript))
    logging.info("Transcript words: %s", len(transcript.split()))

    transcript_path = output_dir / "transcript.txt"
    transcript_path.write_text(transcript + "\n", encoding="utf-8")

    manifest_path = output_dir / "chunks_manifest.json"
    manifest_payload = {
        "input_path": str(input_path),
        "used_split": used_split,
        "split_threshold_mb": args.split_threshold_mb,
        "chunk_target_mb": args.chunk_target_mb,
        "chunk_count": len(chunk_results),
        "chunks": [asdict(chunk_result) for chunk_result in chunk_results],
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = Report(
        model_id=MODEL_ID,
        model_path=model_path,
        input_path=str(input_path),
        output_dir=str(output_dir),
        language=args.language,
        punctuation=not args.no_punctuation,
        split_threshold_mb=args.split_threshold_mb,
        chunk_target_mb=args.chunk_target_mb,
        used_split=used_split,
        chunk_count=len(chunk_results),
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        device=device,
        duration_seconds=duration_seconds,
        file_size_mb=file_size_mb,
        rss_before_mb=rss_before_mb,
        rss_after_load_mb=rss_after_load_mb,
        rss_after_transcribe_mb=rss_after_transcribe_mb,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        transcript_chars=len(transcript),
        transcript_words=len(transcript.split()),
        transcript_path=str(transcript_path),
        manifest_path=str(manifest_path),
    )

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Transcript written to: {transcript_path}")
    print(f"Chunk manifest written to: {manifest_path}")
    print(f"Report written to: {report_path}")
    print(f"Log written to: {log_path}")
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    print(f"Used split: {used_split}")
    print(f"Chunk count: {len(chunk_results)}")
    print(f"Load seconds: {load_seconds}")
    print(f"Transcribe seconds: {transcribe_seconds}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
