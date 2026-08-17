"""
STREEVA — Area Risk Scoring Engine
app/core/time_multiplier.py

Explicit time-of-day risk multiplier.

Design philosophy:
  The same physical location carries different risk at 2am vs 2pm.
  Instead of hiding this as a hidden bias term in the scoring formula,
  we make it an explicit, named, configurable multiplier.

  This allows:
  - Transparent reporting in API responses ("your score is 74 because
    it's late night, adding a 1.40× multiplier")
  - Easy adjustment via city_config.yaml without code changes
  - Clear justification in research reports

Time bands are loaded from city_config.yaml so they can be
tuned per-city (e.g., Mumbai's peak risk hours differ from Chennai's).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import get_city_config
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TimeBandResult:
    """Result of time-of-day classification."""
    band_name: str
    multiplier: float
    label: str
    description: str


def get_time_multiplier(hour: int) -> TimeBandResult:
    """
    Return the time-of-day risk multiplier for a given hour.

    Args:
        hour: Hour of day in 24-hour format (0–23).

    Returns:
        TimeBandResult with band name, multiplier, and human-readable label.

    Raises:
        ValueError: If hour is outside [0, 23].

    Examples:
        >>> get_time_multiplier(14)
        TimeBandResult(band_name='day', multiplier=1.0, ...)

        >>> get_time_multiplier(23)
        TimeBandResult(band_name='late_night', multiplier=1.4, ...)
    """
    if not 0 <= hour <= 23:
        raise ValueError(f"hour must be in [0, 23], got {hour}")

    city_config = get_city_config()
    time_bands = city_config.time_bands

    for band_name, band_data in time_bands.items():
        if hour in band_data["hours"]:
            result = TimeBandResult(
                band_name=band_name,
                multiplier=band_data["multiplier"],
                label=band_data["label"],
                description=band_data["description"],
            )
            logger.debug(
                "time_band_selected",
                hour=hour,
                band=band_name,
                multiplier=band_data["multiplier"],
            )
            return result

    # Fallback: should never happen if config covers all 24 hours
    logger.warning("time_band_not_found", hour=hour, fallback="day")
    return TimeBandResult(
        band_name="day",
        multiplier=1.0,
        label="Daytime (fallback)",
        description="No matching time band found; using baseline multiplier.",
    )


def get_all_multipliers() -> dict[str, float]:
    """
    Return a mapping of {band_name: multiplier} for all configured time bands.
    Useful for documentation and the notebook's sensitivity analysis.

    Returns:
        Dict mapping band names to their multiplier values.
    """
    city_config = get_city_config()
    return {
        band_name: band_data["multiplier"]
        for band_name, band_data in city_config.time_bands.items()
    }
