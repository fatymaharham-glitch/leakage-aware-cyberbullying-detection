"""Consistent console logging for command-line scripts."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes concise messages to standard error."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
