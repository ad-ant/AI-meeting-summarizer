import logging
from pathlib import Path

from config import LOG_FORMAT, LOG_LEVEL, LOGS_DIR


def setup_logging() -> None:
    """Set up console and file logging for the app."""
    log_dir = Path(LOGS_DIR)
    log_dir.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)