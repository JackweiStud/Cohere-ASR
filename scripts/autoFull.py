#!/usr/bin/env python3
"""Transcribe -> optional cleanup -> summary pipeline runner.

This Python CLI is the real orchestrator for the local Cohere ASR workflow.
The sibling autoFull.sh script is kept as a thin compatibility wrapper.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import httpx

from poc_cohere_local_transcribe import _build_chunk_plan, _get_audio_info, _load_model
from poc_cohere_local_transcribe import _merge_transcripts, _resolve_device, _resolve_model_path, _rss_mb
from poc_cohere_local_transcribe import _transcribe_chunks
from transcript_cleanup import configure_logging, cleanup_transcript_in_chunks, DEFAULT_ENV_PATH
from transcript_summary import summarize_transcript_with_llm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "input" / "extracted_audio.wav"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "autoFull.log"
DEFAULT_SPLIT_THRESHOLD_MB = 25.0
DEFAULT_CHUNK_TARGET_MB = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe -> optional cleanup -> summary pipeline")
    parser.add_argument("-i", "--input", default=str(DEFAULT_INPUT_PATH), help="Audio file to transcribe")
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("-m", "--model", default="", help="Local model directory")
    parser.add_argument("-l", "--language", default="en", help="ISO 639-1 language code")
    parser.add_argument("--enCleanUp", type=int, choices=(0, 1), default=0, help="Enable cleanup step (0/1)")
    parser.add_argument("-e", "--env", default=str(DEFAULT_ENV_PATH), help="Path to .env file")
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH), help="Path to the log file")
    return parser.parse_args()


def _validate_input_path(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")
    if input_path.suffix.lower() != ".wav":
        raise ValueError(f"Unsupported input format: {input_path.suffix or '<no suffix>'}. This script currently accepts only .wav input.")


def run_pipeline() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    env_path = Path(args.env).expanduser().resolve()
    log_path = str(Path(args.log_path).expanduser().resolve())

    configure_logging(log_path)
    logging.info("Starting autoFull pipeline")
    logging.info("Input path: %s", input_path)
    logging.info("Output dir: %s", output_dir)
    logging.info("Language: %s", args.language)
    logging.info("Cleanup enabled: %s", bool(args.enCleanUp))
    logging.info("Env path: %s", env_path)

    try:
        _validate_input_path(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
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

    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(torch)
    model_path = _resolve_model_path(args.model or None)
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
        split_threshold_mb=DEFAULT_SPLIT_THRESHOLD_MB,
        chunk_target_mb=DEFAULT_CHUNK_TARGET_MB,
    )
    logging.info("Prepared %s chunk(s)", len(chunk_manifests))

    print("")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" Step 1/2 — Transcribe")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    import time

    load_begin = time.time()
    processor, model = _load_model(model_path, device)
    load_seconds = round(time.time() - load_begin, 3)
    rss_after_load_mb = _rss_mb()
    logging.info("Model loaded in %.3fs", load_seconds)
    logging.info("RSS after load: %.2fMB", rss_after_load_mb)

    transcribe_begin = time.time()
    chunk_results = _transcribe_chunks(
        model=model,
        processor=processor,
        torch_module=torch,
        chunk_manifests=chunk_manifests,
        transcripts_dir=output_dir / "transcripts",
        language=args.language,
        punctuation=True,
    )
    transcript = _merge_transcripts(chunk_results)
    transcribe_seconds = round(time.time() - transcribe_begin, 3)
    rss_after_transcribe_mb = _rss_mb()
    logging.info("Transcription finished in %.3fs", transcribe_seconds)
    logging.info("RSS after transcribe: %.2fMB", rss_after_transcribe_mb)

    transcript_path = output_dir / "transcript.txt"
    transcript_path.write_text(transcript + "\n", encoding="utf-8")

    manifest_path = output_dir / "chunks_manifest.json"
    manifest_payload = {
        "input_path": str(input_path),
        "used_split": used_split,
        "split_threshold_mb": DEFAULT_SPLIT_THRESHOLD_MB,
        "chunk_target_mb": DEFAULT_CHUNK_TARGET_MB,
        "chunk_count": len(chunk_results),
        "chunks": [],
    }
    from dataclasses import asdict
    import json

    manifest_payload["chunks"] = [asdict(chunk_result) for chunk_result in chunk_results]
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_path = output_dir / "report.json"
    report_payload = {
        "model_id": "CohereLabs/cohere-transcribe-03-2026",
        "model_path": model_path,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "language": args.language,
        "punctuation": True,
        "split_threshold_mb": DEFAULT_SPLIT_THRESHOLD_MB,
        "chunk_target_mb": DEFAULT_CHUNK_TARGET_MB,
        "used_split": used_split,
        "chunk_count": len(chunk_results),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": device,
        "duration_seconds": duration_seconds,
        "file_size_mb": file_size_mb,
        "rss_before_mb": rss_before_mb,
        "rss_after_load_mb": rss_after_load_mb,
        "rss_after_transcribe_mb": rss_after_transcribe_mb,
        "load_seconds": load_seconds,
        "transcribe_seconds": transcribe_seconds,
        "transcript_chars": len(transcript),
        "transcript_words": len(transcript.split()),
        "transcript_path": str(transcript_path),
        "manifest_path": str(manifest_path),
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.enCleanUp == 1:
        cleaned_transcript_path = output_dir / "transcript_cleaned.txt"
        analysis_path = output_dir / "transcript_cleaned_analysis.md"

        print("")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(" Step 2/3 — Cleanup")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        cleaned_text, cleanup_model, cleanup_elapsed, cleanup_chunk_count, failed_cleanup_chunk_count = cleanup_transcript_in_chunks(
            transcript,
            env_path=env_path,
        )
        cleaned_transcript_path.write_text(cleaned_text + "\n", encoding="utf-8")
        logging.info("Transcript cleanup finished in %.3fs", cleanup_elapsed)
        logging.info("Cleanup chunks: %s", cleanup_chunk_count)
        logging.info("Failed cleanup chunks: %s", failed_cleanup_chunk_count)
        if cleanup_model:
            logging.info("Cleanup model: %s", cleanup_model)
        print(f"Cleaned transcript written to: {cleaned_transcript_path}")
        if cleanup_model:
            print(f"Cleanup model: {cleanup_model}")
            print(f"Cleanup seconds: {cleanup_elapsed}")
            print(f"Cleanup chunks: {cleanup_chunk_count}")
            print(f"Failed cleanup chunks: {failed_cleanup_chunk_count}")

        summary_input = cleaned_transcript_path
    else:
        analysis_path = output_dir / "transcript_analysis.md"
        summary_input = transcript_path

    print("")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if args.enCleanUp == 1:
        print(" Step 3/3 — Summary")
    else:
        print(" Step 2/2 — Summary")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    summary_text = summary_input.read_text(encoding="utf-8").strip()
    if not summary_text:
        print(f"Input transcript is empty: {summary_input}", file=sys.stderr)
        return 1

    try:
        analysis_md, summary_model, summary_elapsed = summarize_transcript_with_llm(summary_text, env_path=env_path)
    except httpx.HTTPError as exc:
        print(
            "Transcript summary request failed. Please check LLM_API_KEY, LLM_BASE_URL, "
            "LLM_MODEL, network connectivity, and server availability.",
            file=sys.stderr,
        )
        logging.error("Transcript summary request failed: %s", exc)
        return 2
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        logging.error("Transcript summary failed: %s", exc)
        return 2

    analysis_path.write_text(analysis_md, encoding="utf-8")
    logging.info("Transcript summary finished in %.3fs", summary_elapsed)
    logging.info("Summary model: %s", summary_model)

    print("")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" Done")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Transcript written to: {transcript_path}")
    if args.enCleanUp == 1:
        print(f"Cleaned transcript written to: {cleaned_transcript_path}")
    print(f"Analysis written to: {analysis_path}")
    print(f"Summary model: {summary_model}")
    print(f"Summary seconds: {summary_elapsed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_pipeline())
