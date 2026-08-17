"""
STREEVA — Area Risk Scoring Engine
tests/api/test_routes.py

API tests using FastAPI TestClient (no live server needed).
All external API calls are mocked to avoid network dependency.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)



class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "streeva-area-risk"
        assert "scorer" in data


class TestAreaRiskEndpoint:

    def test_valid_request_returns_200(self):
        """Valid (lat, lng, hour) → 200 with expected schema."""
        with patch("app.api.routes.get_macro_baseline_score",
                   return_value=(60.0, {}, False)), \
             patch("app.api.routes.compute_isolation_score",
                   return_value=(70.0, {}, False)), \
             patch("app.api.routes.compute_commercial_density_score",
                   return_value=(40.0, {}, False)), \
             patch("app.api.routes.compute_police_distance_score",
                   return_value=(55.0, {}, False)), \
             patch("app.api.routes.compute_hospital_distance_score",
                   return_value=(45.0, {}, False)), \
             patch("app.api.routes.compute_population_density_score",
                   return_value=(50.0, {}, False)):
            response = client.get("/area-risk", params={"lat": 13.0827, "lng": 80.2707, "hour": 14})

        assert response.status_code == 200

    def test_response_schema_keys(self):
        """Response must contain all required top-level keys."""
        with (
            patch("app.api.routes.get_macro_baseline_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_isolation_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_commercial_density_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_police_distance_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_hospital_distance_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_population_density_score", return_value=(50.0, {}, False)),
        ):
            response = client.get("/area-risk", params={"lat": 13.08, "lng": 80.27, "hour": 10})

        data = response.json()
        required_keys = {
            "risk_score", "classification", "h3_cell",
            "contributing_factors", "time_band", "data_completeness",
            "computed_at", "query", "scorer",
        }
        assert required_keys.issubset(set(data.keys()))

    def test_risk_score_in_valid_range(self):
        with (
            patch("app.api.routes.get_macro_baseline_score", return_value=(75.0, {}, False)),
            patch("app.api.routes.compute_isolation_score", return_value=(80.0, {}, False)),
            patch("app.api.routes.compute_commercial_density_score", return_value=(90.0, {}, False)),
            patch("app.api.routes.compute_police_distance_score", return_value=(85.0, {}, False)),
            patch("app.api.routes.compute_hospital_distance_score", return_value=(70.0, {}, False)),
            patch("app.api.routes.compute_population_density_score", return_value=(60.0, {}, False)),
        ):
            response = client.get("/area-risk", params={"lat": 12.88, "lng": 80.08, "hour": 23})

        data = response.json()
        assert 0.0 <= data["risk_score"] <= 100.0

    def test_classification_is_valid_label(self):
        with (
            patch("app.api.routes.get_macro_baseline_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_isolation_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_commercial_density_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_police_distance_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_hospital_distance_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_population_density_score", return_value=(50.0, {}, False)),
        ):
            response = client.get("/area-risk", params={"lat": 13.05, "lng": 80.28, "hour": 14})

        data = response.json()
        assert data["classification"] in ("Low", "Medium", "High", "Critical")

    def test_query_echo(self):
        """Response must echo back the query parameters."""
        with (
            patch("app.api.routes.get_macro_baseline_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_isolation_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_commercial_density_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_police_distance_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_hospital_distance_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_population_density_score", return_value=(50.0, {}, False)),
        ):
            response = client.get("/area-risk", params={"lat": 12.9141, "lng": 80.1408, "hour": 22})

        data = response.json()
        assert data["query"]["lat"] == pytest.approx(12.9141)
        assert data["query"]["lng"] == pytest.approx(80.1408)
        assert data["query"]["hour"] == 22

    def test_missing_lat_returns_422(self):
        response = client.get("/area-risk", params={"lng": 80.27, "hour": 14})
        assert response.status_code == 422

    def test_missing_lng_returns_422(self):
        response = client.get("/area-risk", params={"lat": 13.08, "hour": 14})
        assert response.status_code == 422

    def test_missing_hour_returns_422(self):
        response = client.get("/area-risk", params={"lat": 13.08, "lng": 80.27})
        assert response.status_code == 422

    def test_invalid_lat_returns_422(self):
        response = client.get("/area-risk", params={"lat": 999, "lng": 80.27, "hour": 14})
        assert response.status_code == 422

    def test_invalid_hour_returns_422(self):
        response = client.get("/area-risk", params={"lat": 13.08, "lng": 80.27, "hour": 25})
        assert response.status_code == 422

    def test_h3_cell_in_response(self):
        """Response must include a valid H3 cell index."""
        import h3
        with (
            patch("app.api.routes.get_macro_baseline_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_isolation_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_commercial_density_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_police_distance_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_hospital_distance_score", return_value=(50.0, {}, False)),
            patch("app.api.routes.compute_population_density_score", return_value=(50.0, {}, False)),
        ):
            response = client.get("/area-risk", params={"lat": 13.08, "lng": 80.27, "hour": 14})

        data = response.json()
        assert h3.is_valid_cell(data["h3_cell"])


class TestCacheStatsEndpoint:

    def test_cache_stats_returns_200(self):
        response = client.get("/cache/stats")
        assert response.status_code == 200

    def test_cache_stats_has_required_keys(self):
        response = client.get("/cache/stats")
        data = response.json()
        assert "total_entries" in data
        assert "total_size_bytes" in data
