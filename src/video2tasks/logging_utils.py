"""Shared logging utilities for the video2tasks package."""

from __future__ import annotations

import logging
import sys


PACKAGE_LOGGER_NAME = "video2tasks"
_MESSAGE_FORMAT = logging.Formatter("%(message)s")


def configure_logging(level: str) -> logging.Logger:
    """Configure the shared package logger to emit plain messages to stdout."""
    normalized_level = str(level or "INFO").upper()
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    package_logger.setLevel(getattr(logging, normalized_level, logging.INFO))
    package_logger.propagate = False

    for handler in list(package_logger.handlers):
        package_logger.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(_MESSAGE_FORMAT)
    package_logger.addHandler(handler)
    return package_logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger rooted under the shared video2tasks namespace."""
    logger_name = str(name or PACKAGE_LOGGER_NAME)
    if logger_name != PACKAGE_LOGGER_NAME and not logger_name.startswith(f"{PACKAGE_LOGGER_NAME}."):
        logger_name = f"{PACKAGE_LOGGER_NAME}.{logger_name}"
    return logging.getLogger(logger_name)
