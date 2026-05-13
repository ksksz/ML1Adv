"""Logging helpers for the project."""

import logging
from pathlib import Path

from ml1adv_cirrhosis.config import LOG_FILE


_LOGGER_NAME = "ml1adv_cirrhosis"


def get_logger() -> logging.Logger:
    """Create or return the singleton project logger."""
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
