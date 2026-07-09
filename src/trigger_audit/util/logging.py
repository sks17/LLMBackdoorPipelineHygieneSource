"""Minimal logging configuration shared by the CLI and runners."""

from __future__ import annotations

import logging

_CONFIGURED = False


def configure_logging(level: str | int = "INFO") -> None:
    """Configure root logging once with a concise, cluster-log-friendly format."""
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (configuring logging with defaults on first use)."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
