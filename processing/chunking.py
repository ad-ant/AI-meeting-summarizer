import logging
from config import CHUNK_MAX_WORDS

logger = logging.getLogger(__name__)


def split_text(text: str, max_words: int = CHUNK_MAX_WORDS) -> list[str]:
    """Split text into word-based chunks."""
    if max_words <= 0:
        raise ValueError("max_words must be greater than 0")

    if not text or not text.strip():
        logger.warning("split_text got empty input")
        return []

    words = text.split()
    chunks = [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]

    logger.debug("Created %s chunks", len(chunks))
    return chunks