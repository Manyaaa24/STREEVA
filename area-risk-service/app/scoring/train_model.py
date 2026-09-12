"""
STREEVA — Train ML Risk Model
app/scoring/train_model.py

Trains a LightGBM regressor on the shrinkage proxy targets.
Saves the artifact to `models/` directory for use by MLRiskScorer.
"""

import pandas as pd
import lightgbm as lgb
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TARGETS_PATH = BASE_DIR / "data" / "processed" / "h3_training_target.parquet"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT_PATH = MODEL_DIR / "lgb_risk_scorer.txt"

def main():
    if not TARGETS_PATH.exists():
        logger.error(f"Missing {TARGETS_PATH}. Run build_training_target.py first.")
        return
        
    logger.info("Loading training data...")
    df = pd.read_parquet(TARGETS_PATH)
    
    # We will simulate test-features processing how it would look in inference
    # Normalize features 0-100 like the API does for BaseScorer:
    
    # isolation: higher road density -> lower isolation risk (invert)
    max_rd = df['road_density'].max() + 1e-5
    df['isolation_score'] = (1.0 - (df['road_density'] / max_rd)) * 100.0
    
    # commercial: higher POI -> lower risk (invert)
    max_poi = df['poi_count'].max() + 1e-5
    df['commercial_density_score'] = (1.0 - (df['poi_count'] / max_poi)) * 100.0
    
    # police/hospital: higher dist = higher risk
    max_pol = df['dist_police_m'].max() + 1e-5
    df['police_distance_score'] = (df['dist_police_m'] / max_pol) * 100.0
    
    max_hosp = df['dist_hospital_m'].max() + 1e-5
    df['hospital_distance_score'] = (df['dist_hospital_m'] / max_hosp) * 100.0
    
    # population: higher = lower risk
    max_pop = df['population'].max() + 1e-5
    df['population_density_score'] = (1.0 - (df['population'] / max_pop)) * 100.0
    
    # nightlight: higher light = lower risk
    max_nl = df['nightlight_mean'].max() + 1e-5
    df['nightlight_mean_score'] = (1.0 - (df['nightlight_mean'] / max_nl)) * 100.0
    
    # Add hour to simulate time variants explicitly
    # To train the model with hour variance, we duplicate samples with different hours
    # Since our proxy didn't incorporate time directly, we will let the model learn 
    # to maintain base target but we will inject a slight time penalty so it learns 
    # the time dependence.
    
    dfs = []
    for hr in range(24):
        tdf = df.copy()
        import math
        tdf['hour_sin'] = math.sin(2 * math.pi * hr / 24.0)
        tdf['hour_cos'] = math.cos(2 * math.pi * hr / 24.0)
        # Shift target slightly based on night time manually so model learns the time curve 
        # (Since our proxy labels had no 'time' column)
        time_penalty = 1.0
        if 22 <= hr or hr <= 4:
            time_penalty = 1.4
        elif 18 <= hr <= 21:
            time_penalty = 1.2
        elif 5 <= hr <= 7:
            time_penalty = 1.15
            
        tdf['risk_target'] = (tdf['risk_target'] * time_penalty).clip(0, 100)
        dfs.append(tdf)
        
    train_df = pd.concat(dfs, ignore_index=True)
    
    if 'crime_baseline_score' not in df.columns:
        df['crime_baseline_score'] = 50.0

    features = [
        'crime_baseline_score', 'isolation_score', 'commercial_density_score', 
        'police_distance_score', 'hospital_distance_score', 'population_density_score', 
        'nightlight_mean_score', 'hour_sin', 'hour_cos'
    ]
    target = 'risk_target'
    
    X = train_df[features]
    y = train_df[target]
    
    logger.info(f"Training LightGBM model on {len(X)} samples with {len(features)} features...")
    
    train_data = lgb.Dataset(X, label=y)
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }
    
    # Train
    bst = lgb.train(param, train_data, num_boost_round=100)
    
    bst.save_model(str(MODEL_OUT_PATH))
    logger.info(f"Model saved to {MODEL_OUT_PATH}.")
    
    # Feature importances
    importances = bst.feature_importance(importance_type='split')
    logger.info("\nFeature Importances:")
    for fn, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
        logger.info(f"  {fn}: {imp}")

if __name__ == "__main__":
    main()
