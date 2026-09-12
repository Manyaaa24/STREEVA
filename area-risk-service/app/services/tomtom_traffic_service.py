"""
STREEVA — TomTom Traffic Flow Service
app/services/tomtom_traffic_service.py

Fetches real-time traffic flow data from TomTom Traffic Flow API v4
and computes the traffic_congestion_ratio (currentSpeed / freeFlowSpeed).

Note on Signal Design:
    traffic_congestion_ratio measures road speed relative to free-flow capacity.
    It is kept SEPARATE from structural OSM isolation_score (road/junction density).
    - High ratio (~1.0): Traffic moving freely at expected speeds.
    - Low ratio (< 0.5): Congestion/delay on the road segment.
    - -1.0 (No road / N/A): Off-road location, campus, or API unavailable.

Caching:
    TomTom flow data is cached per H3 cell for 5 minutes (300 s) using DiskCache.

Fallback hierarchy:
    1. Live TomTom API   -> confidence = "live"
    2. Cache (stale)     -> confidence = "stale_cache"
    3. No road in range  -> confidence = "no_road",   traffic_congestion_ratio = -1.0
    4. API error         -> confidence = "api_error",  traffic_congestion_ratio = -1.0
    5. No API key        -> confidence = "no_key",     traffic_congestion_ratio = -1.0
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.config import get_settings
from app.cache.disk_cache import get_cache
from app.utils.logger import get_logger

logger = get_logger(__name__)

# TomTom Traffic Flow Segment Data endpoint (v4)
# zoom=10 gives ~500m segment granularity — good match for H3 res-9 (~174m hex)
_TOMTOM_FLOW_URL = (
    "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{zoom}/json"
    "?point={lat},{lng}&unit=KMPH&key={key}"
)
_ZOOM = 10

# Cache TTL for traffic data — 5 minutes (traffic updates every 2 min on TomTom)
_TRAFFIC_CACHE_TTL = 300
_CACHE_DATA_TYPE = "tomtom_traffic"

# Request timeout — keep tight so a slow TomTom response doesn't block scoring
_REQUEST_TIMEOUT_S = 2.5


def compute_congestion_ratio(current_speed: float, free_flow_speed: float) -> float:
    """
    Compute traffic congestion ratio: current_speed / free_flow_speed.

    Args:
        current_speed: Current measured speed (km/h).
        free_flow_speed: Free-flow baseline speed (km/h).

    Returns:
        float ratio in [0.0, 1.0], or -1.0 if free_flow_speed <= 0.
    """
    if free_flow_speed <= 0:
        return -1.0
    return round(min(current_speed / free_flow_speed, 1.0), 4)


async def get_traffic_signal(lat: float, lng: float, h3_cell: str = "") -> dict[str, Any]:
    """
    Fetch TomTom traffic flow for (lat, lng) and return traffic congestion metrics.

    Args:
        lat: Latitude of the query point.
        lng: Longitude of the query point.
        h3_cell: H3 cell string (used as cache key).

    Returns:
        {
            "traffic_congestion_ratio": float,   # currentSpeed / freeFlowSpeed (0-1, -1 if N/A)
            "current_speed_kmh":        float,
            "free_flow_speed_kmh":      float,
            "confidence":               str,     # "live"|"live_cached"|"stale_cache"|"no_road"|"api_error"|"no_key"
            "source":                   str,
        }
    """
    settings = get_settings()

    # ── 1. No key configured ──────────────────────────────────────────────────
    if not settings.has_tomtom_key:
        logger.info("tomtom_no_key_configured")
        return _fallback_result("no_key")

    cache = get_cache()
    cache_key = h3_cell or f"{round(lat, 4)}_{round(lng, 4)}"

    # ── 2. Check disk cache (fresh) ───────────────────────────────────────────
    cached_data, is_stale = cache.get(cache_key, _CACHE_DATA_TYPE, _TRAFFIC_CACHE_TTL)
    if cached_data and not is_stale:
        logger.debug("tomtom_cache_hit", h3_cell=cache_key)
        return {**cached_data, "confidence": "live_cached"}

    # ── 3. Call TomTom API ────────────────────────────────────────────────────
    url = _TOMTOM_FLOW_URL.format(
        zoom=_ZOOM,
        lat=round(lat, 6),
        lng=round(lng, 6),
        key=settings.tomtom_api_key,
    )

    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S) as client:
            response = await client.get(url)

        if response.status_code == 404:
            # TomTom returns 404 when no road segment is within range (e.g. campus / off-road)
            logger.info("tomtom_no_road_segment", lat=lat, lng=lng)
            result = _no_road_result()
            cache.set(cache_key, _CACHE_DATA_TYPE, result)
            return result

        response.raise_for_status()
        payload = response.json()

        flow = payload.get("flowSegmentData", {})
        current_speed = float(flow.get("currentSpeed", 0))
        free_flow_speed = float(flow.get("freeFlowSpeed", 0))

        congestion_ratio = compute_congestion_ratio(current_speed, free_flow_speed)

        result = {
            "traffic_congestion_ratio": congestion_ratio,
            "current_speed_kmh":        round(current_speed, 1),
            "free_flow_speed_kmh":      round(free_flow_speed, 1),
            "confidence":               "live",
            "source":                   "tomtom_flow_v4",
        }

        logger.info(
            "tomtom_traffic_fetched",
            lat=lat,
            lng=lng,
            current_speed=current_speed,
            free_flow_speed=free_flow_speed,
            traffic_congestion_ratio=congestion_ratio,
        )

        cache.set(cache_key, _CACHE_DATA_TYPE, result)
        return result

    except httpx.TimeoutException:
        logger.warning("tomtom_timeout", lat=lat, lng=lng, timeout_s=_REQUEST_TIMEOUT_S)
        if cached_data:
            return {**cached_data, "confidence": "stale_cache"}
        return _fallback_result("api_error")

    except httpx.HTTPStatusError as exc:
        logger.warning("tomtom_http_error", status=exc.response.status_code, lat=lat, lng=lng)
        if cached_data:
            return {**cached_data, "confidence": "stale_cache"}
        return _fallback_result("api_error")

    except Exception as exc:
        logger.error("tomtom_unexpected_error", error=str(exc), lat=lat, lng=lng)
        if cached_data:
            return {**cached_data, "confidence": "stale_cache"}
        return _fallback_result("api_error")


def _no_road_result() -> dict[str, Any]:
    """Result when TomTom finds no road segment within range (e.g., campus / off-road)."""
    return {
        "traffic_congestion_ratio": -1.0,
        "current_speed_kmh":        0.0,
        "free_flow_speed_kmh":      0.0,
        "confidence":               "no_road",
        "source":                   "tomtom_flow_v4",
    }


def _fallback_result(confidence: str) -> dict[str, Any]:
    """Result when TomTom API is unconfigured or encounters an error."""
    return {
        "traffic_congestion_ratio": -1.0,
        "current_speed_kmh":        -1.0,
        "free_flow_speed_kmh":      -1.0,
        "confidence":               confidence,
        "source":                   "fallback",
    }
