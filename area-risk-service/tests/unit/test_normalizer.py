"""
STREEVA — Area Risk Scoring Engine
tests/unit/test_normalizer.py

Unit tests for normalization formulas used in feature services.
"""

from __future__ import annotations

import math
import pytest


class TestDistanceToScore:
    """Tests for sigmoid distance-to-score normalization."""

    def _sigmoid_score(self, distance_m: float, midpoint: float, k: float) -> float:
        return 100.0 / (1.0 + math.exp(-k * (distance_m - midpoint)))

    def test_at_midpoint_is_50(self):
        """At the sigmoid midpoint, score should be exactly 50."""
        score = self._sigmoid_score(2500, midpoint=2500, k=0.001)
        assert abs(score - 50.0) < 0.01

    def test_zero_distance_is_near_zero(self):
        """At distance 0, police station is right next to you — very safe."""
        score = self._sigmoid_score(0, midpoint=2500, k=0.001)
        assert score < 10.0

    def test_far_distance_is_near_100(self):
        """At 5000m+, station is far away — high risk contribution."""
        score = self._sigmoid_score(5000, midpoint=2500, k=0.001)
        assert score > 88.0

    def test_score_is_monotonically_increasing(self):
        """Greater distance → higher risk score."""
        scores = [self._sigmoid_score(d, 2500, 0.001) for d in [0, 500, 1000, 2500, 4000, 5000]]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]


class TestDensityToScore:
    """Tests for inverted commercial density scoring."""

    def _density_score(self, count: int, max_pois: int) -> float:
        return max(0.0, 100.0 - min(count / max_pois * 100.0, 100.0))

    def test_zero_pois_is_100(self):
        """No businesses → maximum risk (no eyes on street)."""
        assert self._density_score(0, 20) == 100.0

    def test_max_pois_is_zero(self):
        """max_pois or more businesses → minimum risk."""
        assert self._density_score(20, 20) == 0.0
        assert self._density_score(25, 20) == 0.0  # Clamped

    def test_half_max_is_50(self):
        """Half of max POI count → 50 risk."""
        assert abs(self._density_score(10, 20) - 50.0) < 0.01

    def test_score_is_monotonically_decreasing(self):
        """More businesses → lower risk score."""
        scores = [self._density_score(c, 20) for c in [0, 5, 10, 15, 20]]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]


class TestPopulationNormalization:
    """Tests for inverted population density scoring."""

    def _pop_score(self, pop: float, pop_min: float, pop_max: float) -> float:
        if pop_max <= pop_min:
            return 50.0
        normalised = (pop - pop_min) / (pop_max - pop_min)
        normalised = max(0.0, min(1.0, normalised))
        return (1.0 - normalised) * 100.0

    def test_minimum_density_is_100(self):
        """Lowest population density → highest risk (least surveillance)."""
        assert self._pop_score(0, 0, 1000) == 100.0

    def test_maximum_density_is_zero(self):
        """Highest population density → lowest risk."""
        assert self._pop_score(1000, 0, 1000) == 0.0

    def test_midpoint_is_50(self):
        assert abs(self._pop_score(500, 0, 1000) - 50.0) < 0.01

    def test_out_of_range_clamped(self):
        """Values beyond range should be clamped to 0 or 100."""
        assert self._pop_score(-100, 0, 1000) == 100.0
        assert self._pop_score(2000, 0, 1000) == 0.0


class TestRoadTypeScoring:
    """Tests for OSM highway tag risk scores."""

    def test_motorway_is_safest(self):
        from app.config import get_scoring_weights
        scores = get_scoring_weights().normalization["road_type_scores"]
        assert scores["motorway"] < scores["residential"]
        assert scores["motorway"] < scores["service"]
        assert scores["motorway"] < scores["track"]

    def test_service_lane_is_riskier_than_primary(self):
        from app.config import get_scoring_weights
        scores = get_scoring_weights().normalization["road_type_scores"]
        assert scores["service"] > scores["primary"]

    def test_track_is_riskiest(self):
        from app.config import get_scoring_weights
        scores = get_scoring_weights().normalization["road_type_scores"]
        assert scores["track"] > scores["residential"]
        assert scores["track"] > scores["tertiary"]


class TestTimeMultiplier:
    """Tests for time-of-day multiplier logic."""

    def test_late_night_has_highest_multiplier(self):
        from app.core.time_multiplier import get_time_multiplier
        late_night = get_time_multiplier(23)
        day = get_time_multiplier(14)
        assert late_night.multiplier > day.multiplier

    def test_day_multiplier_is_1_0(self):
        from app.core.time_multiplier import get_time_multiplier
        result = get_time_multiplier(12)
        assert result.multiplier == 1.0

    def test_all_hours_covered(self):
        """Every hour 0–23 must map to a time band."""
        from app.core.time_multiplier import get_time_multiplier
        for h in range(24):
            result = get_time_multiplier(h)
            assert result.multiplier >= 1.0
            assert result.band_name != ""

    def test_invalid_hour_raises(self):
        from app.core.time_multiplier import get_time_multiplier
        with pytest.raises(ValueError):
            get_time_multiplier(24)
        with pytest.raises(ValueError):
            get_time_multiplier(-1)
