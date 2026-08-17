"""
STREEVA — Area Risk Scoring Engine
app/utils/logger.py

Structured JSON logging using structlog.
Every module should import `get_logger` from here — never use `print()`.

Why structlog?
  - Outputs machine-parseable JSON lines (production-ready)
  - Supports key-value context binding (e.g. bind lat, lng, h3_cell)
  - Compatible with cloud log aggregators (GCP, AWS CloudWatch, Datadog)
"""

from __future__ import annotations

import logging
import sys

import structlog

from app.config import get_settings


def configure_logging() -> None:
    """
    Configure structlog and stdlib logging. Call once at app startup.
    In development, outputs human-readable coloured logs.
    In production, outputs JSON lines.
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Shared processors for all environments
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.environment == "development":
        # Pretty coloured console output for local dev
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # JSON lines for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Return a named structlog logger.

    Usage:
        logger = get_logger(__name__)
        logger.info("risk_computed", score=74.2, h3_cell="89283082837ffff")
    """
    return structlog.get_logger(name)
