"""
STREEVA — Area Risk Scoring Engine
app/models/schemas.py

Pydantic request/response models for the FastAPI API.

These models define the stable API contract that the Risk Fusion Engine
(and mobile app) will depend on. Changes to internal implementation
must not change the shape of these models without versioning.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ── Request model ─────────────────────────────────────────────────────────────

class AreaRiskRequest(BaseModel):
    """Query parameters for the /area-risk endpoint."""

    lat: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude in WGS-84 decimal degrees.",
        examples=[12.9141],
    )
    lng: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude in WGS-84 decimal degrees.",
        examples=[80.1408],
    )
    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of day in 24-hour format (0–23).",
        examples=[22],
    )


# ── Response sub-models ───────────────────────────────────────────────────────

class ContributingFactors(BaseModel):
    """Per-feature risk contributions that sum to explain the final score."""

    crime_baseline: float = Field(
        description="Risk contribution from NCRB district crime baseline (0–20 pts)."
    )
    isolation: float = Field(
        description="Risk contribution from road network isolation score (0–25 pts)."
    )
    commercial_density: float = Field(
        description="Risk contribution from commercial POI density — inverted (0–15 pts)."
    )
    police_distance: float = Field(
        description="Risk contribution from distance to nearest police station (0–20 pts)."
    )
    hospital_distance: float = Field(
        description="Risk contribution from distance to nearest hospital (0–10 pts)."
    )
    population_density: float = Field(
        description="Risk contribution from population density — inverted (0–10 pts)."
    )
    time_multiplier_boost: float = Field(
        description="Additional risk added by the time-of-day multiplier."
    )


class TimeBandInfo(BaseModel):
    """Explainability metadata for the time-of-day multiplier."""
    band_name: str = Field(description="Internal band name (e.g. 'late_night').")
    label: str = Field(description="Human-readable time band label.")
    multiplier: float = Field(description="Multiplier applied to base score (e.g. 1.40).")


# ── Main response model ───────────────────────────────────────────────────────

class AreaRiskResponse(BaseModel):
    """
    Complete risk assessment response.

    This schema is the stable contract consumed by the Risk Fusion Engine.
    Adding new optional fields is backwards-compatible; removing fields is not.
    """

    risk_score: float = Field(
        description="Final risk score from 0 (safest) to 100 (most dangerous).",
        ge=0.0,
        le=100.0,
        examples=[74.2],
    )
    classification: Literal["Low", "Medium", "High", "Critical"] = Field(
        description="Risk classification label based on configured thresholds.",
        examples=["High"],
    )
    h3_cell: str = Field(
        description="H3 hexagonal cell index containing the queried coordinate.",
        examples=["89283082837ffff"],
    )
    contributing_factors: ContributingFactors = Field(
        description="Per-feature breakdown explaining the risk score."
    )
    time_band: TimeBandInfo = Field(
        description="Time-of-day multiplier details for explainability."
    )
    data_completeness: int = Field(
        description="Percentage of features computed from primary data sources (0–100).",
        ge=0,
        le=100,
        examples=[94],
    )
    computed_at: datetime = Field(
        description="UTC timestamp when this score was computed.",
    )
    query: dict[str, Any] = Field(
        description="Echo of the original query parameters.",
        examples=[{"lat": 12.9141, "lng": 80.1408, "hour": 22}],
    )
    scorer: str = Field(
        description="Name and version of the scorer that produced this result.",
        examples=["WeightedRiskScorer v1.0.0"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "risk_score": 74.2,
                "classification": "High",
                "h3_cell": "89283082837ffff",
                "contributing_factors": {
                    "crime_baseline": 14.8,
                    "isolation": 22.1,
                    "commercial_density": 8.6,
                    "police_distance": 13.4,
                    "hospital_distance": 5.2,
                    "population_density": 4.1,
                    "time_multiplier_boost": 6.0,
                },
                "time_band": {
                    "band_name": "late_night",
                    "label": "Late Night (10pm–5am)",
                    "multiplier": 1.40,
                },
                "data_completeness": 94,
                "computed_at": "2026-07-22T22:00:00+05:30",
                "query": {"lat": 12.9141, "lng": 80.1408, "hour": 22},
                "scorer": "WeightedRiskScorer v1.0.0",
            }
        }
    }


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""
    status: Literal["ok"] = "ok"
    service: str = "streeva-area-risk"
    version: str = "1.0.0"
    scorer: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str | None = None
