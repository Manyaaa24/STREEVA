from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from .db.database import engine, Base
from .api import routes

# Ensure the DB schema is created
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="STREEVA Risk Fusion Engine",
    description="Fuses Base Area Risk, Motion Anomaly Risk, and Distress Audio Risk",
    version="1.0.0"
)

# Allow CORS for the logger.html tool
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router)

@app.get("/")
def root():
    return {"status": "Fusion Service is running"}
