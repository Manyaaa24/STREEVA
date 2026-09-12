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
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings
from app.utils.logger import configure_logging, get_logger

# Configure structured logging at import time
configure_logging()
logger = get_logger(__name__)


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "service_started",
        service="streeva-area-risk",
        version="1.0.0",
        environment=settings.environment,
        places_key_configured=settings.has_places_key,
    )
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

    yield

    logger.info("service_stopped", service="streeva-area-risk")


def create_app() -> FastAPI:
    """
    Application factory pattern.
    Creates and configures the FastAPI instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="STREEVA — Area Risk Scoring Engine",
        lifespan=lifespan,
        description="""
## Overview
Hyperlocal area safety risk scoring service for Chennai, India.
""",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
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

    # ── Serve Web UI at / ─────────────────────────────────────────────────────
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_ui():
            return FileResponse(static_dir / "index.html")

    return app


# ── Application instance ──────────────────────────────────────────────────────
app = create_app()
