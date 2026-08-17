"""
STREEVA — Area Risk Scoring Engine
app/scoring/base_scorer.py

Abstract base class for all risk scorers.

Why an abstract interface?
  SOLID principle: Dependency Inversion.
  The FastAPI route depends on BaseRiskScorer, NOT WeightedRiskScorer.
  This means:
  - V1 uses WeightedRiskScorer (interpretable weighted formula)
  - V2 can use MLRiskScorer (trained model)
  - The API contract (/area-risk response shape) NEVER changes
  - Swapping the scorer requires only changing the dependency injection,
    not any API or routing code.

This is the same pattern used by scikit-learn (BaseEstimator) and
production ML systems (MLflow models implement a common interface).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class FeatureSet:
    """
    All computed feature values for a single H3 cell at a given hour.

    All raw scores are normalised to [0, 100] before being placed here.
    Higher raw score → higher risk contribution for each feature.
    """
    h3_cell: str
    lat: float
    lng: float
    hour: int

    # Normalised feature scores (0–100 each)
    crime_baseline_score: float      # From NCRB district data
    isolation_score: float           # From OSM road network
    commercial_density_score: float  # From Google Places (inverted)
    police_distance_score: float     # From Google Places (sigmoid distance)
    hospital_distance_score: float   # From Google Places (sigmoid distance)
    population_density_score: float  # From WorldPop raster (inverted)

    # Time-of-day multiplier (explicit, not hidden in formula)
    time_multiplier: float
    time_band_name: str
    time_band_label: str

    # Data quality metadata
    fallback_features: list[str]     # Features that used fallback data


@dataclass
class RiskResult:
    """
    The complete risk assessment result for a single (lat, lng, hour) query.
    This is the canonical output contract for all scorer implementations.
    """
    risk_score: float                # Final score: 0–100
    classification: str              # Low | Medium | High | Critical
    h3_cell: str
    contributing_factors: dict[str, float]  # Per-feature risk contribution
    data_completeness: int           # % of features with primary data (0–100)
    feature_set: FeatureSet          # Raw feature values (for logging/debug)
    time_band_info: dict[str, Any]   # Time band metadata for explainability


class BaseRiskScorer(ABC):
    """
    Abstract interface that all risk scorers must implement.

    Usage:
        scorer = WeightedRiskScorer()
        result = scorer.score(feature_set)
        # result.risk_score → 0–100
        # result.classification → 'Low' | 'Medium' | 'High' | 'Critical'
    """

    @abstractmethod
    def score(self, feature_set: FeatureSet) -> RiskResult:
        """
        Compute a risk score from a pre-computed FeatureSet.

        Args:
            feature_set: All normalised feature values for the target cell.

        Returns:
            RiskResult with final score, classification, and breakdown.
        """
        ...

    @property
    @abstractmethod
    def scorer_name(self) -> str:
        """Human-readable name of this scorer implementation."""
        ...

    @property
    @abstractmethod
    def scorer_version(self) -> str:
        """Version string for this scorer (e.g. 'v1.0.0')."""
        ...
