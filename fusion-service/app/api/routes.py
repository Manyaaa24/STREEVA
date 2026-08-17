import os
import httpx
import asyncio
import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from ..db import database, models
from .. import schemas
from ..ml.motion_model import ActivityStateClassifier
from ..scoring.fusion_scorer import RiskFusionScorer

router = APIRouter()

# Initialize ML model and Scorer
motion_model = ActivityStateClassifier()
fusion_scorer = RiskFusionScorer()

AREA_RISK_URL = os.getenv("AREA_RISK_URL", "http://localhost:8000/area-risk")

@router.post("/log-trip", status_code=201)
def log_trip(trip: schemas.TripCreate, db: Session = Depends(database.get_db)):
    """
    Saves a real sensor-logged trip from the logger tool into SQLite.
    """
    db_trip = models.Trip(
        id=trip.trip_id,
        label=trip.label,
        start_time=trip.start_time,
        end_time=trip.end_time
    )
    db.add(db_trip)
    
    db_points = [
        models.TripPoint(
            trip_id=trip.trip_id,
            timestamp=p.timestamp,
            lat=p.lat,
            lng=p.lng,
            accel_x=p.accel_x,
            accel_y=p.accel_y,
            accel_z=p.accel_z
        ) for p in trip.points
    ]
    db.bulk_save_objects(db_points)
    db.commit()
    
    return {"message": f"Successfully logged trip {trip.trip_id} with {len(db_points)} points."}


@router.post("/journey-risk", response_model=schemas.JourneyRiskResponse)
async def compute_journey_risk(request: schemas.JourneyRiskRequest):
    """
    Computes a live fused risk score based on recent GPS & accelerometer window.
    """
    if not request.points:
        raise HTTPException(status_code=400, detail="No points provided")
        
    latest_point = request.points[-1]
    
    # 1. Compute Motion Risk Score (ActivityStateClassifier + Jerk)
    # Extract raw x, y, z tuples for the model
    accel_data = [(p.accel_x, p.accel_y, p.accel_z) for p in request.points]
    motion_score = motion_model.predict_motion_score(accel_data)
    
    # 2. Fetch Area Risk Score (HTTP Call to Area Risk Service)
    area_score = 50.0 # Default if service fails
    async with httpx.AsyncClient() as client:
        try:
            # We use the timestamp of the latest point to determine the hour
            # Fallback to current time if unavailable
            hour = latest_point.timestamp.hour
            resp = await client.get(
                AREA_RISK_URL, 
                params={"lat": latest_point.lat, "lng": latest_point.lng, "hour": hour},
                timeout=5.0
            )
            if resp.status_code == 200:
                area_score = resp.json().get("risk_score", 50.0)
        except httpx.RequestError as e:
            print(f"Warning: Failed to reach Area Risk Service: {e}")

    # 3. Fuse Scores
    fused_score, classification = fusion_scorer.compute_risk(
        area_risk=area_score,
        motion_anomaly=motion_score,
        audio_distress=0.0 # Placeholder
    )
    
    return schemas.JourneyRiskResponse(
        journey_risk_score=round(fused_score, 1),
        classification=classification,
        fusion_factors=schemas.FusionFactors(
            area_risk=round(area_score, 1),
            motion_anomaly=round(motion_score, 1),
            audio_distress=0.0
        ),
        timestamp=datetime.utcnow()
    )


@router.websocket("/replay/{trip_id}")
async def replay_trip(websocket: WebSocket, trip_id: str, db: Session = Depends(database.get_db)):
    """
    Replays a real stored trip over WebSocket, calculating dynamic risk.
    """
    await websocket.accept()
    
    trip = db.query(models.Trip).filter(models.Trip.id == trip_id).first()
    if not trip:
        await websocket.send_text("Trip not found")
        await websocket.close()
        return
        
    points = sorted(trip.points, key=lambda p: p.timestamp)
    if not points:
        await websocket.send_text("Trip has no points")
        await websocket.close()
        return

    # Sliding window logic to match our model's expectation
    window_size = 30 # Just an example moving window for smooth motion evaluation
    recent_points = []
    
    try:
        for p in points:
            # Construct a TripPointCreate equivalent for the function
            point_data = schemas.TripPointCreate(
                timestamp=p.timestamp,
                lat=p.lat,
                lng=p.lng,
                accel_x=p.accel_x,
                accel_y=p.accel_y,
                accel_z=p.accel_z
            )
            
            recent_points.append(point_data)
            if len(recent_points) > window_size:
                recent_points.pop(0)
                
            # Compute risk
            req = schemas.JourneyRiskRequest(points=recent_points)
            risk_resp = await compute_journey_risk(req)
            
            # Stream to client
            await websocket.send_text(risk_resp.model_dump_json())
            
            # Simulate real-time delay
            await asyncio.sleep(0.5)
            
        await websocket.send_text("Replay complete")
        await websocket.close()
    except WebSocketDisconnect:
        print("Client disconnected")
