"""
STREEVA — Area Risk Scoring Engine
app/config.py

Centralised configuration loader using pydantic-settings.
All values come from environment variables (.env) or YAML config files.
No secrets or magic numbers are hardcoded here.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Project root (area-risk-service/) ────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    All fields map directly to keys in .env / system env.
    """

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Google Places API ────────────────────────────────────────────────────
    google_places_api_key: str = Field(
        default="",
        description="Google Places API (New) key. Required for live POI data.",
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Logging level.")
    environment: str = Field(default="development", description="App environment.")

    # ── City ─────────────────────────────────────────────────────────────────
    city_name: str = Field(default="Chennai", description="Target city name.")

    # ── File Paths ───────────────────────────────────────────────────────────
    cache_dir: str = Field(
        default="data/cache",
        description="Disk cache directory (relative to project root).",
    )
    worldpop_raster_path: str = Field(
        default="data/raw/worldpop_tamilnadu_2020_100m.tif",
        description="Path to WorldPop GeoTIFF. Run scripts/download_worldpop.py.",
    )
    ncrb_lookup_path: str = Field(
        default="data/processed/ncrb_district_lookup.json",
        description="Path to NCRB district lookup JSON. Run scripts/ingest_ncrb.py.",
    )

    @property
    def cache_dir_abs(self) -> Path:
        """Absolute path to cache directory."""
        p = Path(self.cache_dir)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def worldpop_raster_abs(self) -> Path:
        """Absolute path to WorldPop raster."""
        p = Path(self.worldpop_raster_path)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def ncrb_lookup_abs(self) -> Path:
        """Absolute path to NCRB district lookup JSON."""
        p = Path(self.ncrb_lookup_path)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def has_places_key(self) -> bool:
        """True if a non-empty Google Places API key is configured."""
        return bool(self.google_places_api_key.strip())


class ScoringWeights:
    """
    Loads scoring weights and normalisation parameters from
    configs/scoring_weights.yaml. Changing weights requires
    only a YAML edit — no code changes needed.
    """

    _WEIGHTS_FILE = ROOT_DIR / "configs" / "scoring_weights.yaml"

    def __init__(self) -> None:
        with open(self._WEIGHTS_FILE, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        self.weights: dict[str, float] = raw["weights"]
        self.thresholds: dict[str, float] = raw["thresholds"]
        self.normalization: dict = raw["normalization"]

        self._validate_weights()

    def _validate_weights(self) -> None:
        """Ensure weights sum to 1.0 (within floating-point tolerance)."""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"scoring_weights.yaml: weights must sum to 1.0, got {total:.4f}. "
                f"Adjust the values and restart the service."
            )

    def classify(self, score: float) -> str:
        """
        Convert a numeric risk score (0–100) to a classification label.

        Args:
            score: Risk score in [0, 100].

        Returns:
            One of: 'Low', 'Medium', 'High', 'Critical'.
        """
        if score <= self.thresholds["low"]:
            return "Low"
        elif score <= self.thresholds["medium"]:
            return "Medium"
        elif score <= self.thresholds["high"]:
            return "High"
        else:
            return "Critical"


class CityConfig:
    """
    Loads city-specific configuration from configs/city_config.yaml.
    To add a new city, create a new YAML file and point CITY_CONFIG_PATH to it.
    """

    _CONFIG_FILE = ROOT_DIR / "configs" / "city_config.yaml"

    def __init__(self, config_path: Path | None = None) -> None:
        path = config_path or Path(
            os.environ.get("CITY_CONFIG_PATH", str(self._CONFIG_FILE))
        )
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        self._data: dict = raw["city"]

    @property
    def name(self) -> str:
        return self._data["name"]

    @property
    def state(self) -> str:
        return self._data["state"]

    @property
    def ncrb_district_key(self) -> str:
        return self._data["ncrb_district_key"]

    @property
    def bbox(self) -> dict[str, float]:
        """Bounding box dict with keys: south, west, north, east."""
        return self._data["bbox"]

    @property
    def h3_resolution(self) -> int:
        return self._data["h3_resolution"]

    @property
    def places_search_radius_m(self) -> int:
        return self._data["places_search_radius_m"]

    @property
    def cache_ttl(self) -> dict[str, int]:
        return self._data["cache"]

    @property
    def time_bands(self) -> dict:
        return self._data["time_bands"]

    @property
    def commercial_poi_types(self) -> list[str]:
        return self._data["commercial_poi_types"]

    @property
    def places_field_mask(self) -> str:
        return self._data["places_field_mask"]


# ── Singleton accessors (cached for performance) ──────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    return Settings()


@lru_cache(maxsize=1)
def get_scoring_weights() -> ScoringWeights:
    """Return the singleton ScoringWeights instance."""
    return ScoringWeights()


@lru_cache(maxsize=1)
def get_city_config() -> CityConfig:
    """Return the singleton CityConfig instance."""
    return CityConfig()
