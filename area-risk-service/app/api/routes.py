"""
STREEVA — Area Risk Scoring Engine
app/api/routes.py

FastAPI route handlers for the area risk scoring API.

Changes vs. v1:
  • Default scorer is now MLRiskScorer (LightGBM, trained on proxy labels).
  • WeightedRiskScorer is available via ?scorer=weighted for comparison.
  • Feature resolution is now OFFLINE-FIRST:
      - isolation, commercial_density, police_distance, hospital_distance,
        population_density, nightlight_mean → served from pre-computed
        Parquet index (h3_hex_features.parquet) via offline_feature_service.
      - crime_baseline → still served from NCRB macro_baseline_service
        (static district-level JSON, fully offline).
      - Runtime OSM / Google Places API calls are NO LONGER MADE here.
  • ?scorer=ml  → MLRiskScorer  (default)
  • ?scorer=weighted → WeightedRiskScorer (legacy, for A/B comparison)

Design: The router knows nothing about HOW scores are computed.
It delegates to service modules and the scorer, keeping concerns separated.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException, Query

from app.config import get_city_config
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
from app.scoring.ml_risk_scorer import MLRiskScorer
from app.scoring.weighted_scorer import WeightedRiskScorer
from app.services.macro_baseline_service import get_macro_baseline_score
from app.services.offline_feature_service import get_precomputed_features
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# -------------------------------------------------------------------------
# Scorer singletons — initialised once at import time.
# -------------------------------------------------------------------------
_ml_scorer = MLRiskScorer()           # Default: LightGBM
_weighted_scorer = WeightedRiskScorer()  # Fallback / comparison


def _get_scorer(scorer_param: str):
    """Return the appropriate scorer singleton based on the query parameter."""
    if scorer_param == "weighted":
        return _weighted_scorer
    return _ml_scorer


@router.get(
    "/area-risk",
    response_model=AreaRiskResponse,
    summary="Compute area risk score for a location",
    description="""
    Given a latitude, longitude, and optional ISO timestamp or hour, returns a hyperlocal safety
    risk score (0–100) for that location.

    **Dynamic Location & Time Handling:**
    - Coordinates (`lat`, `lng`) are obtained dynamically from the user's browser/GPS.
    - If `timestamp` is provided, the local time and hour are extracted in `Asia/Kolkata` (IST) timezone.
    - If neither `timestamp` nor `hour` is provided, the current time in `Asia/Kolkata` is used automatically.

    **Score interpretation:**
    - 0–25: **Low** — Busy, well-connected area near emergency services
    - 26–50: **Medium** — Moderate activity and connectivity
    - 51–75: **High** — Isolated or low-activity area
    - 76–100: **Critical** — Very isolated, far from help, high-risk time

    **Scorer selection (v2):**
    - `?scorer=ml` *(default)* — LightGBM model trained on Empirical Bayes proxy labels
    - `?scorer=weighted` — Legacy weighted formula (for comparison)

    **Data sources (offline-first):**
    - All features served from pre-computed Parquet index (h3_hex_features.parquet)
    """,
    responses={
        200: {"description": "Risk score computed successfully."},
        400: {"model": ErrorResponse, "description": "Invalid query parameters."},
        503: {"model": ErrorResponse, "description": "ML model not loaded."},
        500: {"model": ErrorResponse, "description": "Internal scoring error."},
    },
    tags=["Risk Scoring"],
)
async def get_area_risk(
    lat: Annotated[
        float,
        Query(ge=-90.0, le=90.0, description="Latitude (WGS-84 decimal degrees)."),
    ],
    lng: Annotated[
        float,
        Query(ge=-180.0, le=180.0, description="Longitude (WGS-84 decimal degrees)."),
    ],
    timestamp: Annotated[
        str | None,
        Query(description="Optional ISO 8601 timestamp string (e.g. 2026-09-11T18:02:00+05:30)."),
    ] = None,
    hour: Annotated[
        int | None,
        Query(ge=0, le=23, description="Optional hour of day in 24-hour format (0–23)."),
    ] = None,
    timezone_name: Annotated[
        str,
        Query(alias="timezone", description="Target timezone name (default: Asia/Kolkata)."),
    ] = "Asia/Kolkata",
    scorer: Annotated[
        Literal["ml", "weighted"],
        Query(description="Scorer implementation to use. 'ml' = LightGBM (default), 'weighted' = legacy formula."),
    ] = "ml",
) -> AreaRiskResponse:
    """Compute and return the area risk score for (lat, lng, timestamp/hour)."""
    start_time = time.perf_counter()

    # Determine target local timezone
    try:
        target_tz = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, Exception):
        target_tz = ZoneInfo("Asia/Kolkata")
        timezone_name = "Asia/Kolkata"

    # Derive local datetime and local hour dynamically
    if timestamp:
        try:
            clean_ts = timestamp.replace("Z", "+00:00") if timestamp.endswith("Z") else timestamp
            parsed_dt = datetime.fromisoformat(clean_ts)
            if parsed_dt.tzinfo is None:
                dt_local = parsed_dt.replace(tzinfo=target_tz)
            else:
                dt_local = parsed_dt.astimezone(target_tz)
        except Exception as exc:
            logger.warning("invalid_timestamp_parsing_fallback", timestamp=timestamp, error=str(exc))
            dt_local = datetime.now(target_tz)
    elif hour is not None:
        now_in_tz = datetime.now(target_tz)
        dt_local = now_in_tz.replace(hour=hour, minute=0, second=0, microsecond=0)
    else:
        dt_local = datetime.now(target_tz)

    calculated_hour = dt_local.hour
    formatted_timestamp = dt_local.isoformat()

    city_config = get_city_config()

    # 1. Convert to H3 cell
    h3_cell = latlng_to_h3(lat, lng, city_config.h3_resolution)
    cell_lat, cell_lng = h3_to_centroid(h3_cell)

    logger.info(
        "risk_request_received",
        lat=lat,
        lng=lng,
        timestamp=formatted_timestamp,
        timezone=timezone_name,
        extracted_hour=calculated_hour,
        h3_cell=h3_cell,
        h3_resolution=city_config.h3_resolution,
        scorer=scorer,
    )

    # 2. Select scorer
    active_scorer = _get_scorer(scorer)

    # 3. Guard: ML scorer requires the model artifact to be loaded
    if scorer == "ml" and not _ml_scorer.model_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "ML model artifact not found. "
                "Run `python -m app.scoring.train_model` to train and save the model, "
                "then restart the service. Use ?scorer=weighted to fall back to the "
                "legacy weighted scorer without restarting."
            ),
        )

    # 4. Get time band
    time_band = get_time_multiplier(calculated_hour)

    # 5. Resolve features — OFFLINE-FIRST via Parquet index
    fallback_features: list[str] = []

    # 5a. Offline spatial features (all 6 in one lookup)
    offline_features = get_precomputed_features(h3_cell)
    if offline_features.get("fallback", False):
        # Cell is outside the pre-built index (outside Chennai bbox)
        fallback_features += [
            "isolation", "commercial_density",
            "police_distance", "hospital_distance",
            "population_density", "nightlight_mean",
        ]
        logger.warning(
            "offline_feature_fallback",
            h3_cell=h3_cell,
            reason="cell_not_in_parquet_index",
        )

    # 5b. Macro crime baseline (static NCRB district JSON — always offline)
    try:
        crime_score, crime_meta = get_macro_baseline_score(lat=cell_lat, lng=cell_lng)
        crime_fallback = crime_meta.get("source") == "fallback_neutral"
        if crime_fallback:
            fallback_features.append("crime_baseline")
    except Exception as exc:
        logger.error("feature_crime_failed", error=str(exc))
        crime_score, crime_meta, crime_fallback = 50.0, {}, True
        fallback_features.append("crime_baseline")

    # 5c. Live Traffic Flow Signal (TomTom API v4) — separate from structural OSM isolation
    from app.services.tomtom_traffic_service import get_traffic_signal
    traffic_info = await get_traffic_signal(cell_lat, cell_lng, h3_cell=h3_cell)

    if traffic_info["confidence"] in ("no_key", "api_error"):
        fallback_features.append("traffic_congestion_ratio")

    # Structural OSM isolation score (from parquet index)
    osm_isolation_score = offline_features["isolation_score"]

    # Check for campus-like model blindspots
    model_limitations = []
    if offline_features.get("commercial_density_score", 100) < 60 and \
       offline_features.get("nightlight_mean_score", 100) < 10 and \
       osm_isolation_score < 50:
        model_limitations.append(
            "This area exhibits characteristics of a gated institutional campus "
            "(high lighting and activity, but sparse public roads). The risk score "
            "may be inflated as the model cannot account for private security."
        )

    # 6. Build FeatureSet
    feature_set = FeatureSet(
        h3_cell=h3_cell,
        lat=cell_lat,
        lng=cell_lng,
        hour=calculated_hour,
        crime_baseline_score=crime_score,
        isolation_score=osm_isolation_score,
        commercial_density_score=offline_features["commercial_density_score"],
        police_distance_score=offline_features["police_distance_score"],
        hospital_distance_score=offline_features["hospital_distance_score"],
        population_density_score=offline_features["population_density_score"],
        nightlight_mean_score=offline_features["nightlight_mean_score"],
        time_multiplier=time_band.multiplier,
        time_band_name=time_band.band_name,
        time_band_label=time_band.label,
        fallback_features=fallback_features,
        traffic_congestion_ratio=traffic_info["traffic_congestion_ratio"],
        traffic_confidence=traffic_info["confidence"],
    )

    # 7. Score
    try:
        result = active_scorer.score(feature_set)
        result.crime_meta = crime_meta
        result.model_limitations = model_limitations
        result.traffic_signal = traffic_info
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("scorer_unexpected_error", error=str(exc))
        raise HTTPException(status_code=500, detail="Internal scoring error.")

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

    logger.info(
        "risk_request_complete",
        lat=lat,
        lng=lng,
        timestamp=formatted_timestamp,
        timezone=timezone_name,
        extracted_hour=calculated_hour,
        h3_cell=h3_cell,
        risk_score=result.risk_score,
        classification=result.classification,
        data_completeness=result.data_completeness,
        scorer_used=active_scorer.scorer_name,
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
        query={
            "lat": lat,
            "lng": lng,
            "timestamp": formatted_timestamp,
            "timezone": timezone_name,
            "hour": calculated_hour,
        },
        scorer=f"{active_scorer.scorer_name} {active_scorer.scorer_version}",
        input_signals=result.input_signals,
        risk_factors=result.risk_factors,
        crime_meta=result.crime_meta,
        model_limitations=result.model_limitations,
        traffic_signal=result.traffic_signal,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["Operations"],
)
async def health_check() -> HealthResponse:
    """Liveness check for the service. Reports both scorers."""
    ml_status = "loaded" if _ml_scorer.model_loaded else "not_loaded (use ?scorer=weighted)"
    return HealthResponse(
        scorer=f"{_ml_scorer.scorer_name} {_ml_scorer.scorer_version} [{ml_status}]"
    )


@router.get(
    "/cache/stats",
    summary="Disk cache statistics",
    tags=["Operations"],
)
async def cache_stats() -> dict:
    """Return current disk cache statistics."""
    from app.cache.disk_cache import get_cache
    return get_cache().stats()
