"""
STREEVA — Area Risk Scoring Engine
app/main.py

FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Then open:
    http://localhost:8000/docs     ← Swagger UI
    http://localhost:8000/redoc    ← ReDoc documentation
    http://localhost:8000/health   ← Liveness check
    http://localhost:8000/area-risk?lat=13.0827&lng=80.2707&hour=14
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.utils.logger import configure_logging, get_logger

# Configure structured logging at import time
configure_logging()
logger = get_logger(__name__)


def create_app() -> FastAPI:
    """
    Application factory pattern.
    Creates and configures the FastAPI instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="STREEVA — Area Risk Scoring Engine",
        description="""
## Overview
Hyperlocal area safety risk scoring service for Chennai, India.

Given a **latitude**, **longitude**, and **hour of day**, returns a risk score
(0–100) with a full explainable breakdown of contributing factors.

## Architecture
- **Layer 1 — Macro Baseline**: NCRB district-level IPC crime data (2014)
- **Layer 2 — Hyperlocal Intelligence**: OpenStreetMap road network + Google Places POIs + WorldPop population density

## Data Sources
| Source | Used For |
|--------|----------|
| NCRB 2014 (data.gov.in) | Crime baseline per district |
| OpenStreetMap (OSMnx) | Road type, density, isolation |
| Google Places API (New) | Commercial density, police/hospital proximity |
| WorldPop 2020 (100m) | Population density |

## Validation
No geocoded crime incident data exists for India at point level.
This service uses proxy signals to estimate safety. Scores are validated
qualitatively: busy commercial roads should score lower than isolated lanes.
See README.md for the full methodology and limitations.

## Future Extensions
The `/area-risk` endpoint consumes `BaseRiskScorer`. Replace `WeightedRiskScorer`
with an ML model by implementing the same interface — no API changes required.
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[
            {
                "name": "Risk Scoring",
                "description": "Area risk score computation endpoints.",
            },
            {
                "name": "Operations",
                "description": "Health, cache, and operational endpoints.",
            },
        ],
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    # Allows the mobile app / risk fusion engine to call this service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict to specific origins in production
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = round((time.perf_counter() - start) * 1000, 1)
        response.headers["X-Process-Time-Ms"] = str(elapsed)
        return response

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "unhandled_exception",
            path=str(request.url),
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(router)

    # ── Startup / Shutdown events ──────────────────────────────────────────────
    @app.on_event("startup")
    async def startup_event():
        logger.info(
            "service_started",
            service="streeva-area-risk",
            version="1.0.0",
            environment=settings.environment,
            places_key_configured=settings.has_places_key,
        )
        # Pre-warm: load NCRB lookup and WorldPop raster stats at startup
        try:
            from app.services.macro_baseline_service import _load_district_lookup
            _load_district_lookup()
        except Exception as exc:
            logger.warning("startup_ncrb_preload_failed", error=str(exc))

        try:
            from app.services.population_service import _load_raster_stats
            _load_raster_stats()
        except Exception as exc:
            logger.warning("startup_worldpop_preload_failed", error=str(exc))

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("service_stopped", service="streeva-area-risk")

    return app


# ── Application instance ──────────────────────────────────────────────────────
app = create_app()
