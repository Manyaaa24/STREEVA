from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class TripPointCreate(BaseModel):
    timestamp: datetime
    lat: float
    lng: float
    accel_x: float
    accel_y: float
    accel_z: float

class TripCreate(BaseModel):
    trip_id: str
    label: str
    start_time: datetime
    end_time: datetime
    points: List[TripPointCreate]

class JourneyRiskRequest(BaseModel):
    points: List[TripPointCreate]

class FusionFactors(BaseModel):
    area_risk: float
    motion_anomaly: float
    audio_distress: Optional[float] = None

class JourneyRiskResponse(BaseModel):
    journey_risk_score: float
    classification: str
    fusion_factors: FusionFactors
    timestamp: datetime
