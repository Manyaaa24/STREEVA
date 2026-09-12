"""
STREEVA — Offline Feature Service
app/services/offline_feature_service.py

Replaces runtime OSM/Google Places calls by looking up pre-computed hex values.
Loads in-memory parquet table.
"""

import pandas as pd
from pathlib import Path
import logging
import math

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "h3_hex_features.parquet"

_df = None

def get_h3_index():
    global _df
    if _df is None:
        if FEATURES_PATH.exists():
            # Load and index by h3_cell for O(1) lookups
            _df = pd.read_parquet(FEATURES_PATH).set_index("h3_cell")
        else:
            logger.error(f"Missing {FEATURES_PATH}")
            # Mock empty DataFrame
            _df = pd.DataFrame()
    return _df

def get_precomputed_features(h3_cell: str) -> dict:
    df = get_h3_index()
    
    if h3_cell in df.index:
        row = df.loc[h3_cell]
        
        # We perform the same normalizations to 0-100 here as we did in train_model.py
        max_rd = df['road_density'].max() + 1e-5
        iso = (1.0 - (row['road_density'] / max_rd)) * 100.0
        
        max_poi = df['poi_count'].max() + 1e-5
        com = (1.0 - (row['poi_count'] / max_poi)) * 100.0
        
        pol_dist = row['dist_police_m']
        pol = 100.0 / (1.0 + math.exp(-0.001 * (pol_dist - 2500)))
        
        hosp_dist = row['dist_hospital_m']
        hosp = 100.0 / (1.0 + math.exp(-0.001 * (hosp_dist - 3000)))
        
        max_pop = df['population'].max() + 1e-5
        pop = (1.0 - (row['population'] / max_pop)) * 100.0
        
        max_nl = df['nightlight_mean'].max() + 1e-5
        nl = (1.0 - (row['nightlight_mean'] / max_nl)) * 100.0
        
        crime = float(row.get('crime_baseline_score', 50.0))
        
        return {
            "crime_baseline_score": crime,
            "isolation_score": iso,
            "commercial_density_score": com,
            "police_distance_score": pol,
            "hospital_distance_score": hosp,
            "population_density_score": pop,
            "nightlight_mean_score": nl,
            "fallback": False
        }
    else:
        # H3 cell completely outside known Chennai bbox
        return {
            "crime_baseline_score": 50.0,
            "isolation_score": 50.0,
            "commercial_density_score": 50.0,
            "police_distance_score": 50.0,
            "hospital_distance_score": 50.0,
            "population_density_score": 50.0,
            "nightlight_mean_score": 50.0,
            "fallback": True
        }
