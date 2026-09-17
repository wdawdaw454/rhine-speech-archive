"""Structured logging helpers."""
from __future__ import annotations

import logging
from pathlib import Path


_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"


def setup_logging(
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    console: bool = True,
    name: str | None = None,
) -> logging.Logger:
    """Configure root (or named) logger with optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter = logging.Formatter(_DEFAULT_FORMAT)

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger