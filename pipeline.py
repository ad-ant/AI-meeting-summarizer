import logging
from pathlib import Path

from config import INPUT_DIR, OUTPUT_DIR
from audio.transcribe import transcribe_audio
from processing.chunking import split_text
from processing.summarize import merge_summaries, run_agentic_analysis, summarize_chunk

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_FORMATS = (".mp3", ".mp4", ".wav")


def get_audio_file(input_dir: str) -> Path:
    """Return first supported audio file from input directory."""
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    for file_path in sorted(input_path.iterdir()):
        if file_path.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
            logger.info("Using audio file: %s", file_path.name)
            return file_path

    available = sorted(p.name for p in input_path.iterdir())
    raise FileNotFoundError(
        f"No supported audio file in {input_dir}. "
        f"Expected one of {SUPPORTED_AUDIO_FORMATS}. Found: {available}"
    )


def run_pipeline(input_dir: str = str(INPUT_DIR), output_dir: str = str(OUTPUT_DIR)) -> None:
    """Run full flow: transcription, summarization, action items."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Pipeline started")

    audio_file = get_audio_file(input_dir)

    transcript_path = output_path / "transcript.txt"
    summary_path = output_path / "summary.txt"
    action_items_path = output_path / "action_items.md"

    logger.info("Transcribing audio")
    transcribe_audio(str(audio_file), str(transcript_path))

    transcript_text = transcript_path.read_text(encoding="utf-8")
    chunks = split_text(transcript_text)
    logger.info("Chunks created: %s", len(chunks))

    summaries: list[str] = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        logger.info("Summarizing chunk %s/%s", idx, total)
        try:
            summaries.append(summarize_chunk(chunk))
        except Exception as err:
            logger.warning("Skipping chunk %s due to error: %s", idx, err)

    if not summaries:
        raise RuntimeError("Could not summarize any chunk. Check Ollama and timeout settings.")

    logger.info("Merging chunk summaries")
    try:
        full_summary = merge_summaries(summaries)
    except (RuntimeError, ValueError) as err:
        logger.warning("Could not merge summaries, falling back to concatenation: %s", err)
        full_summary = "\n\n".join(summaries)

    summary_path.write_text(full_summary, encoding="utf-8")
    logger.info("Summary saved: %s", summary_path)

    logger.info("Extracting action items")
    run_agentic_analysis(full_summary, str(action_items_path))

    logger.info("Pipeline finished")