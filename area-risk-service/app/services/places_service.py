"""
STREEVA — Area Risk Scoring Engine
app/services/places_service.py

Google Places API (New) integration for POI-based features.
OSMnx Overpass serves as an automatic resilience fallback.

Features provided:
  - Commercial density score: count of OPERATIONAL businesses within search radius
  - Police distance score: distance to nearest police station
  - Hospital distance score: distance to nearest hospital

Why Google Places as primary (not OSM amenities)?
  - Real-time business status: Places returns OPERATIONAL/CLOSED_PERMANENTLY/CLOSED_TEMPORARILY
    OSM has no concept of "is this shop still open in 2024?"
  - Coverage: Google maintains Places for India with high completeness
  - Accuracy: crowd-sourced updates keep Places current; OSM edits are sporadic in India

Why field mask?
  Google Places (New) bills per field requested. We only need:
    places.id, places.location, places.types, places.businessStatus, places.displayName
  Excluded: reviews, photos, rating, editorialSummary, priceLevel, etc.
  These add nothing to the risk formula and push into higher billing tiers.

Fallback to OSM:
  If Places returns HTTP 429, 503, or network error, OSMnx amenity queries
  are used silently. This is resilience engineering, not a cost shortcut.
  Both sources are supported in production.

Cache TTL: 1 day (businesses change more frequently than road networks)
"""

from __future__ import annotations

import math
from typing import Any

import requests

from app.cache.disk_cache import get_cache
from app.config import get_city_config, get_settings
from app.core.h3_utils import h3_to_centroid, haversine_distance_m
from app.utils.logger import get_logger
from app.utils.retry import RateLimitError, check_response, with_retry

logger = get_logger(__name__)

# Cache data type identifiers
_CACHE_TYPE_COMMERCIAL = "places_commercial"
_CACHE_TYPE_POLICE = "places_police"
_CACHE_TYPE_HOSPITAL = "places_hospital"

# Places API (New) endpoint
_PLACES_NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"


def _get_ttl_seconds() -> float:
    """Return cache TTL for Places data in seconds."""
    city_config = get_city_config()
    return city_config.cache_ttl["places_ttl_days"] * 86400


# ── Google Places API calls ───────────────────────────────────────────────────

@with_retry(max_attempts=3, wait_min=1.0, wait_max=8.0)
def _places_nearby_search(
    lat: float,
    lng: float,
    included_types: list[str],
    radius_m: int,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """
    Call the Google Places API (New) Nearby Search endpoint.

    Args:
        lat: Center latitude.
        lng: Center longitude.
        included_types: List of Place types to search for.
        radius_m: Search radius in meters.
        max_results: Maximum number of results to return.

    Returns:
        List of place dicts from the API response.

    Raises:
        RateLimitError: On HTTP 429.
        ExternalAPIError: On non-retryable API errors.
    """
    settings = get_settings()
    city_config = get_city_config()

    payload = {
        "includedTypes": included_types,
        "maxResultCount": min(max_results, 20),  # API max is 20
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": float(radius_m),
            }
        },
    }

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": settings.google_places_api_key,
        "X-Goog-FieldMask": city_config.places_field_mask,
    }

    response = requests.post(
        _PLACES_NEARBY_URL,
        json=payload,
        headers=headers,
        timeout=10,
    )

    check_response(response, api_name="Google Places")

    data = response.json()
    return data.get("places", [])


# ── OSM fallback queries ──────────────────────────────────────────────────────

def _osm_fallback_amenity(
    lat: float, lng: float, amenity_tag: str, radius_m: int = 500
) -> list[dict[str, Any]]:
    """
    Query Overpass API for amenity tags as a fallback when Places is unavailable.

    Args:
        lat: Center latitude.
        lng: Center longitude.
        amenity_tag: OSM amenity value (e.g., 'police', 'hospital', 'restaurant').
        radius_m: Search radius in meters.

    Returns:
        List of simplified place dicts with location and type.
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:15];
    (
      node["amenity"="{amenity_tag}"](around:{radius_m},{lat},{lng});
      way["amenity"="{amenity_tag}"](around:{radius_m},{lat},{lng});
    );
    out center;
    """
    try:
        headers = {"User-Agent": "STREEVA_Safety_App/1.0 (contact@streeva.org)"}
        response = requests.post(overpass_url, data={"data": query}, headers=headers, timeout=20)
        response.raise_for_status()
        elements = response.json().get("elements", [])
        results = []
        for el in elements:
            if el.get("type") == "node":
                elat, elng = el.get("lat", lat), el.get("lon", lng)
            elif el.get("type") == "way" and "center" in el:
                elat, elng = el["center"]["lat"], el["center"]["lon"]
            else:
                continue
            results.append({
                "location": {"latitude": elat, "longitude": elng},
                "types": [amenity_tag],
                "businessStatus": "OPERATIONAL",
                "displayName": {"text": el.get("tags", {}).get("name", amenity_tag)},
            })
        return results
    except Exception as exc:
        logger.warning("osm_fallback_failed", amenity=amenity_tag, error=str(exc))
        return []


def _osm_fallback_commercial(lat: float, lng: float, radius_m: int) -> list[dict]:
    """Run OSM fallback for commercial POIs (multiple amenity types)."""
    amenity_types = [
        "restaurant", "cafe", "pharmacy", "supermarket",
        "bus_station", "fuel", "bank", "atm",
    ]
    results = []
    for amenity in amenity_types:
        results.extend(_osm_fallback_amenity(lat, lng, amenity, radius_m))
    return results


# ── Main public functions ─────────────────────────────────────────────────────

def compute_commercial_density_score(
    h3_cell: str,
) -> tuple[float, dict[str, Any], bool]:
    """
    Compute commercial density score (0–100) for an H3 cell.

    Higher score = less commercial activity = higher risk.
    (Commercial density is INVERTED: more shops → safer → lower risk score.)

    Algorithm:
      1. Check disk cache (TTL: 1 day)
      2. Fetch from Google Places (primary)
      3. On failure: fall back to OSM amenity query
      4. Count OPERATIONAL establishments
      5. Normalise: 0 POIs → 100 risk, max_pois+ → 0 risk

    Args:
        h3_cell: H3 cell index string.

    Returns:
        Tuple of (score_0_to_100, raw_features_dict, is_fallback).
    """
    cache = get_cache()
    settings = get_settings()
    city_config = get_city_config()
    ttl = _get_ttl_seconds()
    weights_cfg_norm = __import__("app.config", fromlist=["get_scoring_weights"]).get_scoring_weights().normalization

    cached_data, is_stale = cache.get(h3_cell, _CACHE_TYPE_COMMERCIAL, ttl)
    if cached_data and not is_stale:
        score = _density_to_score(
            cached_data.get("operational_count", 0),
            weights_cfg_norm["commercial_density_max_pois"],
        )
        return score, cached_data, False

    lat, lng = h3_to_centroid(h3_cell)
    is_fallback = False

    try:
        if not settings.has_places_key:
            raise ValueError("No Google Places API key configured.")
        places = _places_nearby_search(
            lat, lng,
            included_types=city_config.commercial_poi_types,
            radius_m=city_config.places_search_radius_m,
            max_results=20,
        )
        source = "google_places"
    except Exception as exc:
        logger.warning(
            "places_commercial_failed_using_osm",
            h3_cell=h3_cell,
            error=str(exc),
        )
        places = _osm_fallback_commercial(lat, lng, city_config.places_search_radius_m)
        source = "osm_fallback"
        is_fallback = True

    operational = [
        p for p in places
        if p.get("businessStatus", "OPERATIONAL") in ("OPERATIONAL", "")
    ]

    raw_features = {
        "source": source,
        "total_results": len(places),
        "operational_count": len(operational),
        "lat": lat,
        "lng": lng,
        "search_radius_m": city_config.places_search_radius_m,
    }

    if not is_fallback and (cached_data is None or is_stale):
        cache.set(h3_cell, _CACHE_TYPE_COMMERCIAL, raw_features)

    max_pois = weights_cfg_norm["commercial_density_max_pois"]
    score = _density_to_score(len(operational), max_pois)
    return score, raw_features, is_fallback


def compute_police_distance_score(
    h3_cell: str,
) -> tuple[float, dict[str, Any], bool]:
    """
    Compute police distance score (0–100) for an H3 cell.

    Higher score = farther from police = higher risk.
    Uses sigmoid normalisation centred at 2500m.

    Args:
        h3_cell: H3 cell index string.

    Returns:
        Tuple of (score_0_to_100, raw_features_dict, is_fallback).
    """
    return _compute_emergency_distance_score(
        h3_cell=h3_cell,
        cache_type=_CACHE_TYPE_POLICE,
        places_types=["police"],
        osm_amenity="police",
        feature_name="police",
    )


def compute_hospital_distance_score(
    h3_cell: str,
) -> tuple[float, dict[str, Any], bool]:
    """
    Compute hospital distance score (0–100) for an H3 cell.

    Higher score = farther from hospital = higher risk.
    Uses sigmoid normalisation centred at 3000m.

    Args:
        h3_cell: H3 cell index string.

    Returns:
        Tuple of (score_0_to_100, raw_features_dict, is_fallback).
    """
    return _compute_emergency_distance_score(
        h3_cell=h3_cell,
        cache_type=_CACHE_TYPE_HOSPITAL,
        places_types=["hospital", "emergency_room_doctor"],
        osm_amenity="hospital",
        feature_name="hospital",
    )


def _compute_emergency_distance_score(
    h3_cell: str,
    cache_type: str,
    places_types: list[str],
    osm_amenity: str,
    feature_name: str,
) -> tuple[float, dict[str, Any], bool]:
    """Shared implementation for police and hospital distance scoring."""
    from app.config import get_scoring_weights

    cache = get_cache()
    settings = get_settings()
    city_config = get_city_config()
    weights_cfg = get_scoring_weights()
    ttl = _get_ttl_seconds()

    cached_data, is_stale = cache.get(h3_cell, cache_type, ttl)
    if cached_data and not is_stale:
        score = _distance_to_score(
            cached_data.get("nearest_distance_m", 5000.0),
            feature_name,
            weights_cfg.normalization,
        )
        return score, cached_data, False

    lat, lng = h3_to_centroid(h3_cell)
    is_fallback = False

    try:
        if not settings.has_places_key:
            raise ValueError("No Google Places API key configured.")
        # Search within 5km for emergency services
        places = _places_nearby_search(
            lat, lng,
            included_types=places_types,
            radius_m=5000,
            max_results=5,
        )
        source = "google_places"
    except Exception as exc:
        logger.warning(
            f"places_{feature_name}_failed_using_osm",
            h3_cell=h3_cell,
            error=str(exc),
        )
        places = _osm_fallback_amenity(lat, lng, osm_amenity, radius_m=5000)
        source = "osm_fallback"
        is_fallback = True

    # Find nearest
    nearest_dist = _nearest_distance(lat, lng, places)

    raw_features = {
        "source": source,
        "feature": feature_name,
        "nearest_distance_m": round(nearest_dist, 1),
        "count_found": len(places),
        "lat": lat,
        "lng": lng,
    }

    if not is_fallback and (cached_data is None or is_stale):
        cache.set(h3_cell, cache_type, raw_features)

    score = _distance_to_score(nearest_dist, feature_name, weights_cfg.normalization)
    return score, raw_features, is_fallback


# ── Helper functions ──────────────────────────────────────────────────────────

def _nearest_distance(
    lat: float, lng: float, places: list[dict[str, Any]]
) -> float:
    """Return the distance (m) to the nearest place. Returns 5000.0 if no places found."""
    if not places:
        return 5000.0  # Max assumed distance when none found

    min_dist = float("inf")
    for place in places:
        loc = place.get("location", {})
        plat = loc.get("latitude", lat)
        plng = loc.get("longitude", lng)
        d = haversine_distance_m(lat, lng, plat, plng)
        min_dist = min(min_dist, d)

    return min(min_dist, 5000.0)


def _distance_to_score(
    distance_m: float,
    feature_name: str,
    normalization: dict,
) -> float:
    """
    Convert a distance (meters) to a 0–100 risk score using sigmoid.

    Formula: score = 100 / (1 + exp(-k * (d - midpoint)))
    - d = 0: score ≈ 0 (right next to the station → safe)
    - d = midpoint: score = 50
    - d = large: score → 100 (very far → risky)
    """
    if feature_name == "police":
        midpoint = normalization["police_sigmoid_midpoint_m"]
        k = normalization["police_sigmoid_k"]
    else:
        midpoint = normalization["hospital_sigmoid_midpoint_m"]
        k = normalization["hospital_sigmoid_k"]

    score = 100.0 / (1.0 + math.exp(-k * (distance_m - midpoint)))
    return round(min(max(score, 0.0), 100.0), 2)


def _density_to_score(operational_count: int, max_pois: int) -> float:
    """
    Convert commercial POI count to a 0–100 risk score (inverted linear).

    Formula: score = 100 - clamp(count / max_pois * 100, 0, 100)
    - 0 POIs → 100 (no activity → risky)
    - max_pois+ → 0 (dense commercial → safe)
    """
    score = 100.0 - min(operational_count / max_pois * 100.0, 100.0)
    return round(max(score, 0.0), 2)
