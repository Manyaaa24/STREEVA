"""
STREEVA — Area Risk Scoring Engine
app/api/routes.py

FastAPI route handlers for the area risk scoring API.

This module is responsible for:
  1. Parsing and validating query parameters
  2. Orchestrating feature service calls
  3. Running the risk scorer
  4. Building the structured JSON response
  5. Logging every request with full context

Design: The router knows nothing about HOW scores are computed.
It delegates to service modules and the scorer, keeping concerns separated.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import get_city_config, get_scoring_weights
from app.core.h3_utils import latlng_to_h3, h3_to_centroid
from app.core.time_multiplier import get_time_multiplier
from app.models.schemas import (
    AreaRiskResponse,
    ContributingFactors,
    ErrorResponse,
    HealthResponse,
    TimeBandInfo,
)
from app.scoring.base_scorer import FeatureSet
from app.scoring.weighted_scorer import WeightedRiskScorer
from app.services.macro_baseline_service import get_macro_baseline_score
from app.services.osm_service import compute_isolation_score
from app.services.places_service import (
    compute_commercial_density_score,
    compute_hospital_distance_score,
    compute_police_distance_score,
)
from app.services.population_service import compute_population_density_score
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Module-level scorer singleton (constructed once at import time)
_scorer = WeightedRiskScorer()


@router.get(
    "/area-risk",
    response_model=AreaRiskResponse,
    summary="Compute area risk score for a location",
    description="""
    Given a latitude, longitude, and hour of day, returns a hyperlocal safety
    risk score (0–100) for that location in Chennai, India.

    **Score interpretation:**
    - 0–25: **Low** — Busy, well-connected area near emergency services
    - 26–50: **Medium** — Moderate activity and connectivity
    - 51–75: **High** — Isolated or low-activity area
    - 76–100: **Critical** — Very isolated, far from help, high-risk time

    **Data sources:**
    - Road network: OpenStreetMap via OSMnx
    - POI density & emergency proximity: Google Places API (New)
    - Population density: WorldPop 2020 (100m GeoTIFF)
    - Crime baseline: NCRB 2014 district-level IPC data (data.gov.in)

    **Validation note:** No ground-truth crime-incident dataset exists for Chennai
    at point level. Scores are validated qualitatively — busy roads should score
    lower than isolated service lanes. See README.md for full methodology.
    """,
    responses={
        200: {"description": "Risk score computed successfully."},
        400: {"model": ErrorResponse, "description": "Invalid query parameters."},
        500: {"model": ErrorResponse, "description": "Internal scoring error."},
    },
    tags=["Risk Scoring"],
)
async def get_area_risk(
    lat: Annotated[
        float,
        Query(ge=-90.0, le=90.0, description="Latitude (WGS-84 decimal degrees).", example=12.9141),
    ],
    lng: Annotated[
        float,
        Query(ge=-180.0, le=180.0, description="Longitude (WGS-84 decimal degrees).", example=80.1408),
    ],
    hour: Annotated[
        int,
        Query(ge=0, le=23, description="Hour of day in 24-hour format.", example=22),
    ],
) -> AreaRiskResponse:
    """Compute and return the area risk score for (lat, lng, hour)."""
    start_time = time.perf_counter()

    city_config = get_city_config()

    # 1. Convert to H3 cell
    h3_cell = latlng_to_h3(lat, lng, city_config.h3_resolution)
    cell_lat, cell_lng = h3_to_centroid(h3_cell)

    logger.info(
        "risk_request_received",
        lat=lat,
        lng=lng,
        hour=hour,
        h3_cell=h3_cell,
        h3_resolution=city_config.h3_resolution,
    )

    # 2. Get time multiplier
    time_band = get_time_multiplier(hour)

    # 3. Compute all features (independently, order doesn't matter)
    fallback_features: list[str] = []

    try:
        crime_score, crime_meta, crime_fallback = get_macro_baseline_score()
        if crime_fallback:
            fallback_features.append("crime_baseline")
    except Exception as exc:
        logger.error("feature_crime_failed", error=str(exc))
        crime_score, crime_meta, crime_fallback = 50.0, {}, True
        fallback_features.append("crime_baseline")

    try:
        isolation_score, osm_meta, iso_fallback = compute_isolation_score(h3_cell)
        if iso_fallback:
            fallback_features.append("isolation")
    except Exception as exc:
        logger.error("feature_isolation_failed", error=str(exc))
        isolation_score, osm_meta, iso_fallback = 50.0, {}, True
        fallback_features.append("isolation")

    try:
        commercial_score, comm_meta, comm_fallback = compute_commercial_density_score(h3_cell)
        if comm_fallback:
            fallback_features.append("commercial_density")
    except Exception as exc:
        logger.error("feature_commercial_failed", error=str(exc))
        commercial_score, comm_meta, comm_fallback = 50.0, {}, True
        fallback_features.append("commercial_density")

    try:
        police_score, police_meta, police_fallback = compute_police_distance_score(h3_cell)
        if police_fallback:
            fallback_features.append("police_distance")
    except Exception as exc:
        logger.error("feature_police_failed", error=str(exc))
        police_score, police_meta, police_fallback = 50.0, {}, True
        fallback_features.append("police_distance")

    try:
        hospital_score, hosp_meta, hosp_fallback = compute_hospital_distance_score(h3_cell)
        if hosp_fallback:
            fallback_features.append("hospital_distance")
    except Exception as exc:
        logger.error("feature_hospital_failed", error=str(exc))
        hospital_score, hosp_meta, hosp_fallback = 50.0, {}, True
        fallback_features.append("hospital_distance")

    try:
        pop_score, pop_meta, pop_fallback = compute_population_density_score(h3_cell)
        if pop_fallback:
            fallback_features.append("population_density")
    except Exception as exc:
        logger.error("feature_population_failed", error=str(exc))
        pop_score, pop_meta, pop_fallback = 50.0, {}, True
        fallback_features.append("population_density")

    # 4. Build FeatureSet
    feature_set = FeatureSet(
        h3_cell=h3_cell,
        lat=cell_lat,
        lng=cell_lng,
        hour=hour,
        crime_baseline_score=crime_score,
        isolation_score=isolation_score,
        commercial_density_score=commercial_score,
        police_distance_score=police_score,
        hospital_distance_score=hospital_score,
        population_density_score=pop_score,
        time_multiplier=time_band.multiplier,
        time_band_name=time_band.band_name,
        time_band_label=time_band.label,
        fallback_features=fallback_features,
    )

    # 5. Score
    result = _scorer.score(feature_set)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

    logger.info(
        "risk_request_complete",
        lat=lat,
        lng=lng,
        hour=hour,
        h3_cell=h3_cell,
        risk_score=result.risk_score,
        classification=result.classification,
        data_completeness=result.data_completeness,
        execution_ms=elapsed_ms,
    )

    return AreaRiskResponse(
        risk_score=result.risk_score,
        classification=result.classification,
        h3_cell=h3_cell,
        contributing_factors=ContributingFactors(**result.contributing_factors),
        time_band=TimeBandInfo(**result.time_band_info),
        data_completeness=result.data_completeness,
        computed_at=datetime.now(tz=timezone.utc),
        query={"lat": lat, "lng": lng, "hour": hour},
        scorer=f"{_scorer.scorer_name} {_scorer.scorer_version}",
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["Operations"],
)
async def health_check() -> HealthResponse:
    """Liveness check for the service."""
    return HealthResponse(scorer=f"{_scorer.scorer_name} {_scorer.scorer_version}")


@router.get(
    "/cache/stats",
    summary="Disk cache statistics",
    tags=["Operations"],
)
async def cache_stats() -> dict:
    """Return current disk cache statistics."""
    from app.cache.disk_cache import get_cache
    return get_cache().stats()
