"""
STREEVA — Area Risk Scoring Engine
app/utils/retry.py

Exponential-backoff retry decorator using tenacity.

Why retry?
  External APIs (Google Places, Overpass) can transiently fail due to:
  - Rate limiting (HTTP 429)
  - Temporary server errors (HTTP 500/503)
  - Network timeouts

  Retrying with exponential backoff (e.g., 1s → 2s → 4s) handles
  transient failures without hammering the server — industry standard.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Sequence, Type

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Retryable exceptions ─────────────────────────────────────────────────────
# These are transient errors where retrying makes sense.
RETRYABLE_EXCEPTIONS: tuple[Type[Exception], ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


class RateLimitError(Exception):
    """Raised when an API returns HTTP 429 Too Many Requests."""
    pass


class ExternalAPIError(Exception):
    """Raised when an API returns a non-retryable error (4xx other than 429)."""
    pass


def with_retry(
    max_attempts: int = 3,
    wait_min: float = 1.0,
    wait_max: float = 10.0,
    reraise: bool = True,
) -> Callable:
    """
    Decorator factory that adds exponential-backoff retry logic.

    Args:
        max_attempts: Maximum number of attempts (including first).
        wait_min: Minimum seconds between retries.
        wait_max: Maximum seconds between retries.
        reraise: If True, re-raises the final exception after all attempts fail.

    Usage:
        @with_retry(max_attempts=3)
        def fetch_places(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @retry(
            retry=retry_if_exception_type((*RETRYABLE_EXCEPTIONS, RateLimitError)),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=wait_min, max=wait_max),
            before_sleep=before_sleep_log(logger, "warning"),  # type: ignore[arg-type]
            reraise=reraise,
        )
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        return wrapper
    return decorator


def check_response(response: requests.Response, api_name: str = "API") -> None:
    """
    Raise appropriate exceptions based on HTTP status code.

    Args:
        response: The requests.Response object.
        api_name: Human-readable API name for error messages.

    Raises:
        RateLimitError: On HTTP 429.
        ExternalAPIError: On other 4xx/5xx errors.
    """
    if response.status_code == 429:
        raise RateLimitError(
            f"{api_name} rate limit exceeded (HTTP 429). Will retry."
        )
    if response.status_code >= 500:
        raise requests.exceptions.ConnectionError(
            f"{api_name} server error (HTTP {response.status_code}). Will retry."
        )
    if response.status_code >= 400:
        raise ExternalAPIError(
            f"{api_name} client error (HTTP {response.status_code}): {response.text[:200]}"
        )
