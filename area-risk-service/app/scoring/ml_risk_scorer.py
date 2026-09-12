"""
STREEVA — ML Risk Scorer
app/scoring/ml_risk_scorer.py

Implements BaseRiskScorer using a gradient-boosted tree (LightGBM).
Returns predicted risk along with raw feature transparency.
"""

from typing import Any
import lightgbm as lgb
from pathlib import Path
import numpy as np
import shap

from app.scoring.base_scorer import BaseRiskScorer, FeatureSet, RiskResult

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgb_risk_scorer.txt"

class MLRiskScorer(BaseRiskScorer):
    """
    Risk Scorer powered by a LightGBM regressor trained via empirical Bayes proxy labels.
    """
    
    def __init__(self):
        # We load the model synchronously at startup
        # In a very high throughput environment, this might be a singleton lazy load
        if MODEL_PATH.exists():
            self.model = lgb.Booster(model_file=str(MODEL_PATH))
            self.model_loaded = True
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.model = None
            self.model_loaded = False
            self.explainer = None
            
    def _get_classification(self, score: float) -> str:
        if score < 25.0: return "Low"
        if score < 50.0: return "Medium"
        if score < 75.0: return "High"
        return "Critical"

    def score(self, feature_set: FeatureSet) -> RiskResult:
        if not self.model_loaded:
            # Degrade gracefully or fail loud
            raise RuntimeError("ML model artifact not found.")

        # Match exactly the 9-feature array from train_model.py:
        # 'crime_baseline_score', 'isolation_score', 'commercial_density_score', 
        # 'police_distance_score', 'hospital_distance_score', 'population_density_score', 
        # 'nightlight_mean_score', 'hour_sin', 'hour_cos'
        
        feature_array = np.array([[
            feature_set.crime_baseline_score,
            feature_set.isolation_score,
            feature_set.commercial_density_score,
            feature_set.police_distance_score,
            feature_set.hospital_distance_score,
            feature_set.population_density_score,
            feature_set.nightlight_mean_score,
            feature_set.hour_sin,
            feature_set.hour_cos
        ]])
        
        pred_score = float(self.model.predict(feature_array)[0])
        pred_cli = max(0.0, min(100.0, pred_score))
        
        shap_vals = self.explainer.shap_values(feature_array)
        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]
            shap_vals = shap_vals[0]
            
        # Calculate data completeness
        total_features = 7
        primary_count = total_features - len(feature_set.fallback_features)
        completeness_pct = int((primary_count / total_features) * 100)

        # For backwards schema compatibility (contributing_factors is required):
        dummy_raw = (
            feature_set.crime_baseline_score * 0.20 +
            feature_set.isolation_score * 0.25 +
            feature_set.commercial_density_score * 0.15 +
            feature_set.police_distance_score * 0.20 +
            feature_set.hospital_distance_score * 0.10 +
            feature_set.population_density_score * 0.10
        )
        time_boost = round(dummy_raw * (feature_set.time_multiplier - 1.0), 2)

        return RiskResult(
            risk_score=round(pred_cli, 1),
            classification=self._get_classification(pred_cli),
            h3_cell=feature_set.h3_cell,
            contributing_factors={
                "crime_baseline": round(feature_set.crime_baseline_score, 1),
                "isolation": round(feature_set.isolation_score, 1),
                "commercial_density": round(feature_set.commercial_density_score, 1),
                "police_distance": round(feature_set.police_distance_score, 1),
                "hospital_distance": round(feature_set.hospital_distance_score, 1),
                "population_density": round(feature_set.population_density_score, 1),
                "time_multiplier_boost": time_boost,
            },
            input_signals={
                "crime_baseline": round(feature_set.crime_baseline_score, 1),
                "isolation": round(feature_set.isolation_score, 1),
                "traffic_congestion_ratio": round(feature_set.traffic_congestion_ratio, 4),
                "commercial_density": round(feature_set.commercial_density_score, 1),
                "police_distance": round(feature_set.police_distance_score, 1),
                "hospital_distance": round(feature_set.hospital_distance_score, 1),
                "population_density": round(feature_set.population_density_score, 1),
                "nightlight_mean": round(feature_set.nightlight_mean_score, 1),
            },
            risk_factors={
                "base_value": round(float(base_value), 2),
                "crime_baseline": round(float(shap_vals[0][0]), 2),
                "isolation": round(float(shap_vals[0][1]), 2),
                "commercial_density": round(float(shap_vals[0][2]), 2),
                "police_distance": round(float(shap_vals[0][3]), 2),
                "hospital_distance": round(float(shap_vals[0][4]), 2),
                "population_density": round(float(shap_vals[0][5]), 2),
                "nightlight_mean": round(float(shap_vals[0][6]), 2),
                "time_variance_sin": round(float(shap_vals[0][7]), 2),
                "time_variance_cos": round(float(shap_vals[0][8]), 2),
            },
            data_completeness=completeness_pct,
            feature_set=feature_set,
            time_band_info={
                "band_name": feature_set.time_band_name,
                "label": f"{feature_set.time_band_label} (reference only, model uses cyclical sin/cos features)",
                "multiplier": feature_set.time_multiplier,
                "applied_in_ml": False,
            }
        )

    @property
    def scorer_name(self) -> str:
        return "LGBBProxyScorer"

    @property
    def scorer_version(self) -> str:
        return "v1.1.0-ml"
