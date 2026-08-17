import pytest
import os
import tempfile
import yaml
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.db.database import Base, get_db
from app.db import models
from app.scoring.fusion_scorer import RiskFusionScorer

# 1. Unit Tests for Scorer
def test_fusion_scorer_math():
    # Create a temporary yaml config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({"fusion_weights": {"area_risk": 0.6, "motion_anomaly": 0.4, "audio_distress": 0.0}}, f)
        temp_path = f.name
        
    try:
        scorer = RiskFusionScorer(config_path=temp_path)
        # 50 * 0.6 + 50 * 0.4 = 50
        score, classification = scorer.compute_risk(50.0, 50.0)
        assert score == 50.0
        assert classification == "High"
        
        # 100 * 0.6 + 0 * 0.4 = 60
        score, classification = scorer.compute_risk(100.0, 0.0)
        assert score == 60.0
        assert classification == "High"
        
        # Clamping
        score, classification = scorer.compute_risk(150.0, 100.0)
        assert score == 100.0
        assert classification == "Critical"
    finally:
        os.unlink(temp_path)

# 2. Integration Tests for API
from sqlalchemy.pool import StaticPool
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def test_log_trip_endpoint():
    payload = {
        "trip_id": "test_trip_1",
        "label": "normal",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "points": [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "lat": 12.9,
                "lng": 80.1,
                "accel_x": 0.0,
                "accel_y": 9.8,
                "accel_z": 0.0
            }
        ]
    }
    
    response = client.post("/log-trip", json=payload)
    assert response.status_code == 201
    assert "Successfully logged trip" in response.json()["message"]

def test_journey_risk_endpoint(mocker):
    # Mock httpx.AsyncClient to return a fixed area risk score
    class MockResponse:
        status_code = 200
        def json(self):
            return {"risk_score": 80.0}

    class MockAsyncClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, *args, **kwargs):
            return MockResponse()

    mocker.patch("httpx.AsyncClient", return_value=MockAsyncClient())
    
    # Send a small window of dummy accel data
    points = []
    for _ in range(10):
        points.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lat": 12.9,
            "lng": 80.1,
            "accel_x": 0.0,
            "accel_y": 9.8,
            "accel_z": 0.0
        })
        
    response = client.post("/journey-risk", json={"points": points})
    assert response.status_code == 200
    data = response.json()
    
    # Area risk was mocked to 80.0
    assert data["fusion_factors"]["area_risk"] == 80.0
    # Motion risk will be fallback 50.0 (since no real model loaded in test or predicting 0.5)
    # The actual output depends on the model. If no model, it defaults to 50.0
    assert "journey_risk_score" in data
    assert "classification" in data
