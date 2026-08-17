import os
import yaml
from ..schemas import FusionFactors

class RiskFusionScorer:
    def __init__(self, config_path: str = None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(base_dir, "fusion_weights.yaml")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.weights = config.get("fusion_weights", {})
            
    def compute_risk(self, area_risk: float, motion_anomaly: float, audio_distress: float = 0.0) -> tuple[float, str]:
        """
        Computes the blended journey risk score based on configured weights.
        Clamps the final score to [0, 100].
        """
        w_area = self.weights.get("area_risk", 0.6)
        w_motion = self.weights.get("motion_anomaly", 0.4)
        w_audio = self.weights.get("audio_distress", 0.0)
        
        # Ensure weights sum to 1.0 (or normalize them if they don't, but assuming they do)
        total_weight = w_area + w_motion + w_audio
        
        blended_score = (
            (area_risk * w_area) + 
            (motion_anomaly * w_motion) + 
            (audio_distress * w_audio)
        ) / total_weight
        
        # Clamp to 0-100
        final_score = max(0.0, min(100.0, blended_score))
        
        # Classification
        if final_score >= 75:
            classification = "Critical"
        elif final_score >= 50:
            classification = "High"
        elif final_score >= 25:
            classification = "Medium"
        else:
            classification = "Low"
            
        return final_score, classification
