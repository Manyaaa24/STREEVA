"""
STREEVA — Area Risk Scoring Engine
app/services/osm_service.py

Road network feature extraction from OpenStreetMap via OSMnx.

What this service provides:
  - Road type score (based on OSM highway tag classification)
  - Road density (total road length per km²)
  - Junction density (intersections per km²)
  - Composite isolation score (weighted combination of above)

Why road features predict safety:
  The "eyes on the street" theory (Jane Jacobs, 1961) argues that
  busy, well-connected streets deter crime through natural surveillance.
  Empirical criminology supports this:
  - High road connectivity → more pedestrians, more witnesses
  - Major arterial roads → better lighting, more traffic, faster police response
  - Service lanes, tracks, footpaths → isolated, unlit, minimal activity

Why OSMnx?
  - Fetches and models the OSM road network as a networkx DiGraph
  - Handles coordinate system conversions automatically
  - The networkx graph is directly reusable for route-scoring in Phase 2
  - Well-maintained, widely used in transport research

Data source: OpenStreetMap (© OpenStreetMap contributors, ODbL license)
Fetch method: Overpass API via OSMnx (no API key required)
Cache TTL: 7 days (road network changes slowly)
"""

from __future__ import annotations

from typing import Any

import osmnx as ox

from app.cache.disk_cache import get_cache
from app.config import get_city_config, get_scoring_weights
from app.core.h3_utils import h3_to_centroid, h3_to_polygon, haversine_distance_m
from app.utils.logger import get_logger
from app.utils.retry import with_retry

logger = get_logger(__name__)

# Cache data type identifier
_CACHE_TYPE = "osm"

# OSMnx network type: 'drive' for vehicle-accessible roads
# 'all' would include footways/paths, which are relevant for isolation
_NETWORK_TYPE = "all"


def _get_ttl_seconds() -> float:
    """Return cache TTL for OSM data in seconds."""
    city_config = get_city_config()
    return city_config.cache_ttl["osm_ttl_days"] * 86400


@with_retry(max_attempts=3, wait_min=2.0, wait_max=15.0)
def _fetch_osm_features(lat: float, lng: float, radius_m: float = 500) -> dict[str, Any]:
    """
    Fetch road network features from OSM for a point.

    Args:
        lat: Latitude of center point.
        lng: Longitude of center point.
        radius_m: Search radius in meters.

    Returns:
        Dict with raw OSM road network statistics.
    """
    # Configure OSMnx to use caching and be less verbose
    ox.settings.log_console = False
    ox.settings.use_cache = False  # We handle our own caching

    try:
        G = ox.graph_from_point(
            (lat, lng),
            dist=radius_m,
            network_type=_NETWORK_TYPE,
            simplify=True,
        )
    except ox.InsufficientResponseError:
        # No OSM data for this location (extremely rare in Chennai)
        logger.warning("osm_no_data", lat=lat, lng=lng, radius_m=radius_m)
        return _empty_osm_result()

    # Extract edges (road segments) and nodes (junctions)
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    nodes_gdf, _ = ox.graph_to_gdfs(G)

    # Calculate area of the search disk (km²)
    import math
    area_km2 = math.pi * (radius_m / 1000) ** 2

    # Road type: get the dominant highway tag
    highway_tags = edges["highway"].dropna()
    # highway can be a string or a list; flatten
    all_tags = []
    for tag in highway_tags:
        if isinstance(tag, list):
            all_tags.extend(tag)
        else:
            all_tags.append(str(tag))

    dominant_highway = _most_common(all_tags) if all_tags else "unclassified"

    # Road density: total edge length / area
    total_length_m = edges["length"].sum() if not edges.empty else 0.0
    road_density_m_per_km2 = total_length_m / area_km2

    # Junction density: number of nodes with degree > 1 / area
    if len(G.nodes) > 0:
        intersection_nodes = [
            n for n, d in G.degree() if d > 2
        ]
        junction_density = len(intersection_nodes) / area_km2
    else:
        junction_density = 0.0

    return {
        "dominant_highway": dominant_highway,
        "all_highway_tags": list(set(all_tags)),
        "total_road_length_m": round(total_length_m, 2),
        "road_density_m_per_km2": round(road_density_m_per_km2, 2),
        "junction_density_per_km2": round(junction_density, 2),
        "node_count": len(G.nodes),
        "edge_count": len(G.edges),
        "search_radius_m": radius_m,
        "area_km2": round(area_km2, 4),
    }


def _most_common(lst: list[str]) -> str:
    """Return the most common element in a list."""
    from collections import Counter
    if not lst:
        return "unclassified"
    return Counter(lst).most_common(1)[0][0]


def _empty_osm_result() -> dict[str, Any]:
    """Return a zero-data OSM result when no road network is found."""
    return {
        "dominant_highway": "unclassified",
        "all_highway_tags": [],
        "total_road_length_m": 0.0,
        "road_density_m_per_km2": 0.0,
        "junction_density_per_km2": 0.0,
        "node_count": 0,
        "edge_count": 0,
        "search_radius_m": 500,
        "area_km2": 0.785,
    }


def compute_isolation_score(h3_cell: str) -> tuple[float, dict[str, Any], bool]:
    """
    Compute the isolation score (0–100) for an H3 cell.

    Higher score = more isolated = higher risk contribution.

    Algorithm:
      1. Check disk cache (TTL: 7 days)
      2. On cache miss: fetch OSM road network for cell centroid
      3. Compute three sub-scores:
         - road_type_score: based on dominant highway tag (lookup table from config)
         - road_density_score: normalised inverse of road density
         - junction_density_score: normalised inverse of junction density
      4. Weighted average of sub-scores → isolation score
      5. Write to cache

    Args:
        h3_cell: H3 cell index string.

    Returns:
        Tuple of (isolation_score_0_to_100, raw_features_dict, is_fallback).
        is_fallback=True means cached data was stale/unavailable.
    """
    cache = get_cache()
    ttl = _get_ttl_seconds()
    weights_cfg = get_scoring_weights()

    # Check cache
    cached_data, is_stale = cache.get(h3_cell, _CACHE_TYPE, ttl)
    if cached_data and not is_stale:
        score = _compute_isolation_from_features(cached_data, weights_cfg)
        return score, cached_data, False

    # Fetch from OSM
    lat, lng = h3_to_centroid(h3_cell)
    is_fallback = False
    try:
        raw_features = _fetch_osm_features(lat, lng, radius_m=500)
        cache.set(h3_cell, _CACHE_TYPE, raw_features)
    except Exception as exc:
        logger.error(
            "osm_fetch_failed",
            h3_cell=h3_cell,
            lat=lat,
            lng=lng,
            error=str(exc),
        )
        if cached_data:
            # Use stale cache as fallback
            logger.warning("osm_using_stale_cache", h3_cell=h3_cell)
            raw_features = cached_data
            is_fallback = True
        else:
            logger.warning("osm_using_default_fallback", h3_cell=h3_cell)
            raw_features = _empty_osm_result()
            is_fallback = True

    score = _compute_isolation_from_features(raw_features, weights_cfg)
    return score, raw_features, is_fallback


def _compute_isolation_from_features(
    features: dict[str, Any],
    weights_cfg: Any,
) -> float:
    """
    Compute a 0–100 isolation score from raw OSM feature dict.

    Sub-scores (all 0–100, higher = more isolated = riskier):
      - road_type_score: from config lookup table
      - road_density_score: inverse normalised (0 density → 100 risk)
      - junction_density_score: inverse normalised

    Final score: weighted average of three sub-scores.
    """
    norm = weights_cfg.normalization
    road_type_scores: dict[str, int] = norm["road_type_scores"]

    # 1. Road type score (lookup)
    highway = features.get("dominant_highway", "unclassified")
    road_type_score = float(
        road_type_scores.get(highway, road_type_scores["default"])
    )

    # 2. Road density score (inverse normalised)
    # Reference: Chennai arterial roads ~10,000–20,000 m/km², service lanes <2,000
    density = features.get("road_density_m_per_km2", 0.0)
    MAX_DENSITY = 25_000.0  # saturates at this density
    road_density_score = max(0.0, 100.0 - min(density / MAX_DENSITY * 100.0, 100.0))

    # 3. Junction density score (inverse normalised)
    # Reference: dense urban Chennai ~20–50 junctions/km²
    junc_density = features.get("junction_density_per_km2", 0.0)
    MAX_JUNC_DENSITY = 60.0
    junction_density_score = max(0.0, 100.0 - min(junc_density / MAX_JUNC_DENSITY * 100.0, 100.0))

    # Weighted average of sub-scores
    isolation_score = (
        0.50 * road_type_score +
        0.30 * road_density_score +
        0.20 * junction_density_score
    )

    logger.debug(
        "isolation_computed",
        highway=highway,
        road_type_score=round(road_type_score, 1),
        road_density_score=round(road_density_score, 1),
        junction_density_score=round(junction_density_score, 1),
        isolation_score=round(isolation_score, 1),
    )

    return round(min(max(isolation_score, 0.0), 100.0), 2)
