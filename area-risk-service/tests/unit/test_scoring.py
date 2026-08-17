"""
STREEVA — Area Risk Scoring Engine
tests/unit/test_scoring.py

Unit tests for the WeightedRiskScorer.

These tests use fully mocked FeatureSets — no external API calls.
"""

from __future__ import annotations

import pytest

from app.scoring.base_scorer import FeatureSet, RiskResult
from app.scoring.weighted_scorer import WeightedRiskScorer


def _make_feature_set(**overrides) -> FeatureSet:
    """Helper: create a default FeatureSet with optional field overrides."""
    defaults = dict(
        h3_cell="89283082837ffff",
        lat=13.08,
        lng=80.27,
        hour=14,
        crime_baseline_score=50.0,
        isolation_score=50.0,
        commercial_density_score=50.0,
        police_distance_score=50.0,
        hospital_distance_score=50.0,
        population_density_score=50.0,
        time_multiplier=1.0,
        time_band_name="day",
        time_band_label="Daytime (8am–6pm)",
        fallback_features=[],
    )
    defaults.update(overrides)
    return FeatureSet(**defaults)


class TestWeightedRiskScorer:

    def setup_method(self):
        self.scorer = WeightedRiskScorer()

    def test_scorer_name_and_version(self):
        assert self.scorer.scorer_name == "WeightedRiskScorer"
        assert self.scorer.scorer_version == "v1.0.0"

    def test_neutral_features_give_mid_score(self):
        """All features at 50 with day multiplier → score ~50."""
        fs = _make_feature_set()
        result = self.scorer.score(fs)
        assert isinstance(result, RiskResult)
        assert 45.0 <= result.risk_score <= 55.0

    def test_all_zero_features_give_low_score(self):
        """All features at 0 → very low risk score."""
        fs = _make_feature_set(
            crime_baseline_score=0.0,
            isolation_score=0.0,
            commercial_density_score=0.0,
            police_distance_score=0.0,
            hospital_distance_score=0.0,
            population_density_score=0.0,
            time_multiplier=1.0,
        )
        result = self.scorer.score(fs)
        assert result.risk_score < 10.0
        assert result.classification == "Low"

    def test_all_max_features_give_critical(self):
        """All features at 100 with late-night multiplier → Critical."""
        fs = _make_feature_set(
            crime_baseline_score=100.0,
            isolation_score=100.0,
            commercial_density_score=100.0,
            police_distance_score=100.0,
            hospital_distance_score=100.0,
            population_density_score=100.0,
            time_multiplier=1.40,
        )
        result = self.scorer.score(fs)
        assert result.risk_score == 100.0
        assert result.classification == "Critical"

    def test_score_is_clamped_to_0_100(self):
        """Score must never exceed 100 or go below 0."""
        fs = _make_feature_set(
            crime_baseline_score=100.0,
            isolation_score=100.0,
            commercial_density_score=100.0,
            police_distance_score=100.0,
            hospital_distance_score=100.0,
            population_density_score=100.0,
            time_multiplier=2.0,  # Artificially high
        )
        result = self.scorer.score(fs)
        assert 0.0 <= result.risk_score <= 100.0

    def test_night_scores_higher_than_day(self):
        """Same location at night (×1.40) must score higher than at day (×1.0)."""
        base = dict(
            crime_baseline_score=50.0,
            isolation_score=60.0,
            commercial_density_score=40.0,
            police_distance_score=55.0,
            hospital_distance_score=45.0,
            population_density_score=50.0,
        )
        day_fs = _make_feature_set(**base, time_multiplier=1.0, hour=14)
        night_fs = _make_feature_set(**base, time_multiplier=1.40, hour=22)

        day_result = self.scorer.score(day_fs)
        night_result = self.scorer.score(night_fs)

        assert night_result.risk_score > day_result.risk_score

    def test_contributing_factors_keys(self):
        """Response must contain all required factor keys."""
        fs = _make_feature_set()
        result = self.scorer.score(fs)
        required_keys = {
            "crime_baseline", "isolation", "commercial_density",
            "police_distance", "hospital_distance", "population_density",
            "time_multiplier_boost",
        }
        assert required_keys == set(result.contributing_factors.keys())

    def test_data_completeness_100_with_no_fallbacks(self):
        """No fallback features → 100% completeness."""
        fs = _make_feature_set(fallback_features=[])
        result = self.scorer.score(fs)
        assert result.data_completeness == 100

    def test_data_completeness_reduced_with_fallbacks(self):
        """Each fallback reduces completeness."""
        fs = _make_feature_set(fallback_features=["isolation", "police_distance"])
        result = self.scorer.score(fs)
        assert result.data_completeness < 100

    def test_classification_boundaries(self):
        """Test all four classification thresholds."""
        cases = [
            (0.0, "Low"),
            (25.0, "Low"),
            (26.0, "Medium"),
            (50.0, "Medium"),
            (51.0, "High"),
            (75.0, "High"),
            (76.0, "Critical"),
            (100.0, "Critical"),
        ]
        from app.config import get_scoring_weights
        weights_cfg = get_scoring_weights()
        for score, expected_class in cases:
            assert weights_cfg.classify(score) == expected_class, \
                f"Score {score} → expected {expected_class}"

    def test_isolation_weight_dominates_when_others_zero(self):
        """
        Isolation has the highest weight (0.25).
        With isolation=100 and all others=0, isolation should dominate.
        """
        fs = _make_feature_set(
            crime_baseline_score=0.0,
            isolation_score=100.0,
            commercial_density_score=0.0,
            police_distance_score=0.0,
            hospital_distance_score=0.0,
            population_density_score=0.0,
            time_multiplier=1.0,
        )
        result = self.scorer.score(fs)
        # isolation contribution = 0.25 × 100 = 25 pts
        assert abs(result.risk_score - 25.0) < 1.0
        assert result.contributing_factors["isolation"] == pytest.approx(25.0, abs=0.1)
