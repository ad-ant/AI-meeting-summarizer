import logging
from pathlib import Path

from faster_whisper import WhisperModel

from config import WHISPER_DEVICE, WHISPER_LANGUAGE, WHISPER_MODEL

logger = logging.getLogger(__name__)


def transcribe_audio(audio_path: str, output_path: str) -> None:
    """Transcribe audio to text and save transcript to a file."""
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info("Loading Whisper model: %s", WHISPER_MODEL)
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE)

    logger.info("Transcribing: %s", audio_path)
    segments, _ = model.transcribe(
        str(audio_file),
        language=WHISPER_LANGUAGE,
        beam_size=5,
    )

    transcript = "\n".join(segment.text for segment in segments)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(transcript, encoding="utf-8")

    logger.info("Transcript saved: %s", output_path)