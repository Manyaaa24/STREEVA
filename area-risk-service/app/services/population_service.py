"""
STREEVA — Area Risk Scoring Engine
app/services/population_service.py

Population density extraction from WorldPop GeoTIFF raster.

Data source:
  WorldPop Global High Resolution Population Denominators Project (2020)
  - 100m resolution population count raster
  - Values represent estimated number of people per 100m × 100m pixel
  - Download: scripts/download_worldpop.py
  - License: Creative Commons Attribution 4.0 International (CC BY 4.0)
  - Citation: WorldPop (www.worldpop.org) School of Geography and Environmental
    Science, University of Southampton

Why population density predicts safety:
  High population density correlates with:
  - More potential witnesses to incidents (natural surveillance)
  - Higher probability of informal social control
  - Better-lit streets and more foot traffic
  This is a component of Jane Jacobs' "eyes on the street" theory.

  NOTE: This relationship is not linear — very dense informal settlements
  can have high crime. The model treats this as one of many features,
  not the sole predictor.

Why rasterio (not geopandas/Census)?
  - WorldPop provides the most granular open population data for India (100m)
  - Rasterio samples a single pixel in milliseconds — no spatial join needed
  - The raster covers all of Tamil Nadu, ready for multi-city extension
  - Census ward-level data would require a spatial join and is far less granular

Normalization:
  Min-max across the Chennai bounding box, computed once at startup.
  Inverted: dense → 0 risk pts, sparse → 100 risk pts.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.cache.disk_cache import get_cache
from app.config import get_city_config, get_settings
from app.core.h3_utils import h3_to_centroid
from app.utils.logger import get_logger

logger = get_logger(__name__)

_CACHE_TYPE = "population"
_NEUTRAL_SCORE = 50.0  # Used when raster is unavailable


def _get_ttl_seconds() -> float:
    city_config = get_city_config()
    return city_config.cache_ttl["population_ttl_days"] * 86400


@lru_cache(maxsize=1)
def _load_raster_stats() -> dict[str, Any] | None:
    """
    Load WorldPop raster and compute Chennai bounding-box min/max.
    Called once at startup; result cached in memory via lru_cache.

    Returns:
        Dict with 'dataset_path', 'chennai_min', 'chennai_max', 'nodata_value'
        or None if raster file is unavailable.
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        logger.error("rasterio_not_installed")
        return None

    settings = get_settings()
    city_config = get_city_config()
    raster_path = settings.worldpop_raster_abs

    if not raster_path.exists():
        logger.warning(
            "worldpop_raster_missing",
            path=str(raster_path),
            hint="Run scripts/download_worldpop.py to download the raster.",
        )
        return None

    try:
        bbox = city_config.bbox
        with rasterio.open(str(raster_path)) as dataset:
            # Read only the Chennai bounding box window
            window = from_bounds(
                left=bbox["west"],
                bottom=bbox["south"],
                right=bbox["east"],
                top=bbox["north"],
                transform=dataset.transform,
            )
            data = dataset.read(1, window=window)
            nodata = dataset.nodata

            # Mask out nodata values
            if nodata is not None:
                valid_data = data[data != nodata]
            else:
                valid_data = data[data > 0]

            if len(valid_data) == 0:
                logger.warning("worldpop_raster_no_valid_data", bbox=bbox)
                return None

            pop_min = float(np.percentile(valid_data, 5))   # 5th percentile (not extreme outliers)
            pop_max = float(np.percentile(valid_data, 95))  # 95th percentile

            logger.info(
                "worldpop_raster_loaded",
                path=str(raster_path),
                chennai_pop_min=round(pop_min, 2),
                chennai_pop_max=round(pop_max, 2),
                valid_pixels=len(valid_data),
            )
            return {
                "dataset_path": str(raster_path),
                "chennai_min": pop_min,
                "chennai_max": pop_max,
                "nodata_value": nodata,
            }
    except Exception as exc:
        logger.error("worldpop_raster_load_failed", error=str(exc), path=str(raster_path))
        return None


def _sample_population(lat: float, lng: float) -> float | None:
    """
    Sample the WorldPop raster at a given coordinate.

    Args:
        lat: Latitude.
        lng: Longitude.

    Returns:
        Population count for the 100m pixel at (lat, lng), or None if unavailable.
    """
    try:
        import rasterio
    except ImportError:
        return None

    settings = get_settings()
    raster_path = settings.worldpop_raster_abs

    if not raster_path.exists():
        return None

    try:
        with rasterio.open(str(raster_path)) as dataset:
            # Convert lat/lng to pixel row/col
            row, col = dataset.index(lng, lat)  # Note: rasterio uses (x=lng, y=lat)
            # Read the single pixel
            data = dataset.read(1, window=((row, row + 1), (col, col + 1)))
            if data.size == 0:
                return None
            value = float(data[0][0])
            nodata = dataset.nodata
            if nodata is not None and abs(value - nodata) < 1e-6:
                return None
            return max(0.0, value)
    except Exception as exc:
        logger.warning("worldpop_sample_failed", lat=lat, lng=lng, error=str(exc))
        return None


def compute_population_density_score(
    h3_cell: str,
) -> tuple[float, dict[str, Any], bool]:
    """
    Compute population density risk score (0–100) for an H3 cell.

    Higher score = lower population density = higher risk.
    (Population density is INVERTED: dense areas are safer.)

    Args:
        h3_cell: H3 cell index string.

    Returns:
        Tuple of (score_0_to_100, raw_features_dict, is_fallback).
        is_fallback=True when raster is unavailable.
    """
    cache = get_cache()
    ttl = _get_ttl_seconds()

    # Check cache first
    cached_data, is_stale = cache.get(h3_cell, _CACHE_TYPE, ttl)
    if cached_data and not is_stale:
        score = _pop_to_score(
            cached_data.get("population_per_pixel", None),
            _load_raster_stats(),
        )
        return score, cached_data, False

    lat, lng = h3_to_centroid(h3_cell)

    raster_stats = _load_raster_stats()
    pop_value = _sample_population(lat, lng)
    is_fallback = raster_stats is None or pop_value is None

    raw_features: dict[str, Any] = {
        "population_per_pixel": pop_value,
        "lat": lat,
        "lng": lng,
        "raster_available": raster_stats is not None,
        "source": "worldpop_100m" if not is_fallback else "unavailable",
    }

    if raster_stats is not None:
        raw_features["normalization_min"] = raster_stats["chennai_min"]
        raw_features["normalization_max"] = raster_stats["chennai_max"]

    if not is_fallback:
        cache.set(h3_cell, _CACHE_TYPE, raw_features)

    if is_fallback:
        logger.warning(
            "population_fallback_neutral",
            h3_cell=h3_cell,
            raster_available=raster_stats is not None,
            pop_value=pop_value,
        )

    score = _pop_to_score(pop_value, raster_stats) if not is_fallback else _NEUTRAL_SCORE
    return score, raw_features, is_fallback


def _pop_to_score(
    pop_value: float | None,
    raster_stats: dict[str, Any] | None,
) -> float:
    """
    Convert population pixel count to a 0–100 risk score (inverted min-max).

    Formula:
        normalised = clamp((pop - min) / (max - min), 0, 1)
        score = (1 - normalised) × 100    ← inverted: dense = safer = low score

    Returns 50.0 (neutral) when data is unavailable.
    """
    if pop_value is None or raster_stats is None:
        return _NEUTRAL_SCORE

    pop_min = raster_stats["chennai_min"]
    pop_max = raster_stats["chennai_max"]

    if pop_max <= pop_min:
        return _NEUTRAL_SCORE

    normalised = (pop_value - pop_min) / (pop_max - pop_min)
    normalised = max(0.0, min(1.0, normalised))
    score = (1.0 - normalised) * 100.0  # Invert: dense = safer = lower risk score
    return round(score, 2)
