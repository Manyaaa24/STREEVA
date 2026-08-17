"""
STREEVA — Area Risk Scoring Engine
app/scoring/weighted_scorer.py

V1 weighted linear risk scorer.

Scoring formula (full derivation):
  ─────────────────────────────────────────────────────────────
  RawScore = Σ weight_i × feature_score_i

  Where each feature_score_i ∈ [0, 100] (higher = riskier) and
  weights sum to 1.0 (loaded from configs/scoring_weights.yaml).

  FinalScore = clamp(RawScore × TimeMultiplier, 0, 100)
  ─────────────────────────────────────────────────────────────

  Expanded:
  RawScore =
    w_crime    × crime_baseline_score   +   (default: 0.20)
    w_iso      × isolation_score        +   (default: 0.25)
    w_com      × commercial_density_score + (default: 0.15, inverted in places_service)
    w_police   × police_distance_score  +   (default: 0.20)
    w_hospital × hospital_distance_score +  (default: 0.10)
    w_pop      × population_density_score   (default: 0.10, inverted in population_service)

  TimeMultiplier ∈ [1.0, 1.40] (from city_config.yaml, applied multiplicatively)

Why weighted linear for V1?
  - Every weight can be justified and explained in a project report
  - Changing a weight requires only editing scoring_weights.yaml
  - The contributing_factors breakdown shows exactly WHY a score was given
  - Weights can be tuned by a domain expert (e.g. a criminologist)
    without requiring ML training data or re-training

Why NOT a trained ML model for V1?
  - No ground-truth "this location is unsafe" labeled dataset exists for Chennai
  - A model trained on proxy signals without validation would give false confidence
  - The weighted formula is fully auditable and explainable — critical for a
    capstone project that must defend its methodology

Future extension:
  Replace WeightedRiskScorer with MLRiskScorer(BaseRiskScorer) without
  changing any API code. The FeatureSet dataclass is the interface contract.
"""

from __future__ import annotations

from typing import Any

from app.config import get_scoring_weights
from app.scoring.base_scorer import BaseRiskScorer, FeatureSet, RiskResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


class WeightedRiskScorer(BaseRiskScorer):
    """
    Explicit weighted linear risk scorer.

    Weights are loaded from configs/scoring_weights.yaml at construction.
    No weights or thresholds are hardcoded in this file.
    """

    def __init__(self) -> None:
        self._weights_cfg = get_scoring_weights()
        self._weights = self._weights_cfg.weights

        logger.info(
            "weighted_scorer_initialized",
            weights=self._weights,
            thresholds=self._weights_cfg.thresholds,
        )

    @property
    def scorer_name(self) -> str:
        return "WeightedRiskScorer"

    @property
    def scorer_version(self) -> str:
        return "v1.0.0"

    def score(self, feature_set: FeatureSet) -> RiskResult:
        """
        Compute weighted risk score from the feature set.

        Args:
            feature_set: All normalised feature values for the target cell.

        Returns:
            RiskResult with final score, per-feature contributions, and metadata.
        """
        w = self._weights

        # ── Compute per-feature contributions ─────────────────────────────
        # Each contribution = weight × feature_score
        # Sum of all contributions = raw_score (before time multiplier)
        contributions: dict[str, float] = {
            "crime_baseline": round(w["crime_baseline"] * feature_set.crime_baseline_score, 2),
            "isolation": round(w["isolation"] * feature_set.isolation_score, 2),
            "commercial_density": round(w["commercial_density"] * feature_set.commercial_density_score, 2),
            "police_distance": round(w["police_distance"] * feature_set.police_distance_score, 2),
            "hospital_distance": round(w["hospital_distance"] * feature_set.hospital_distance_score, 2),
            "population_density": round(w["population_density"] * feature_set.population_density_score, 2),
        }

        # ── Raw score (before time multiplier) ────────────────────────────
        raw_score = sum(contributions.values())

        # ── Apply time multiplier ──────────────────────────────────────────
        time_contribution = round(
            raw_score * (feature_set.time_multiplier - 1.0), 2
        )
        contributions["time_multiplier_boost"] = time_contribution

        final_score_raw = raw_score * feature_set.time_multiplier
        final_score = round(max(0.0, min(final_score_raw, 100.0)), 1)

        # ── Classify ──────────────────────────────────────────────────────
        classification = self._weights_cfg.classify(final_score)

        # ── Data completeness ──────────────────────────────────────────────
        # Each fallback feature reduces completeness by ~16.7% (6 features)
        completeness = max(0, 100 - len(feature_set.fallback_features) * 17)

        logger.info(
            "risk_scored",
            h3_cell=feature_set.h3_cell,
            lat=feature_set.lat,
            lng=feature_set.lng,
            hour=feature_set.hour,
            raw_score=round(raw_score, 2),
            time_multiplier=feature_set.time_multiplier,
            final_score=final_score,
            classification=classification,
            data_completeness=completeness,
            fallback_features=feature_set.fallback_features,
        )

        return RiskResult(
            risk_score=final_score,
            classification=classification,
            h3_cell=feature_set.h3_cell,
            contributing_factors=contributions,
            data_completeness=completeness,
            feature_set=feature_set,
            time_band_info={
                "band_name": feature_set.time_band_name,
                "label": feature_set.time_band_label,
                "multiplier": feature_set.time_multiplier,
            },
        )
