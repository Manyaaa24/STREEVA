"""
STREEVA Risk Fusion Engine
app/ml/motion_model.py

Loads the exported 1D-CNN TFLite model and runs inference on raw accelerometer data.
Extracts a jerk heuristic and combines it with the CNN output to formulate a motion risk score.
"""
import os
import numpy as np
import logging

try:
    # Use TensorFlow Lite runtime if available, else fallback to full TF
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
    except ImportError:
        tflite = None

logger = logging.getLogger(__name__)

class ActivityStateClassifier:
    """
    Distinguishes steady state (stillness) from dynamic state (movement) using a 1D-CNN.
    Combines this with a jerk heuristic (rate of change of acceleration) to capture 
    sudden violent movement.
    
    NOTE Limitation: This classifier is trained on UCI HAR. It cannot distinguish normal
    walking from erratic walking, struggling, or panic running. It only detects stillness 
    vs movement. True anomaly/fall detection is future work.
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "models", "motion_anomaly.tflite")
            
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        self.load_model()

    def load_model(self):
        if tflite is None:
            logger.warning("ActivityStateClassifier: TensorFlow/TFLite runtime not installed. Fallback active.")
            return

        if not os.path.exists(self.model_path):
            logger.warning(f"ActivityStateClassifier: Model file not found at {self.model_path}. Fallback active.")
            return

        try:
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info("ActivityStateClassifier: real model loaded successfully.")
        except Exception as e:
            logger.error(f"ActivityStateClassifier: Failed to load TFLite model: {e}")

    def predict_motion_score(self, accel_data: list[tuple[float, float, float]]) -> float:
        """
        Predicts a motion risk score for a given sequence of raw accelerometer data.
        
        Args:
            accel_data: List of (x, y, z) tuples.
        
        Returns:
            Risk score (0 to 100).
        """
        if not accel_data:
            return 50.0

        # Calculate Jerk Heuristic
        # Jerk is rate of change of acceleration magnitude.
        data_array = np.array(accel_data, dtype=np.float32)
        mags = np.linalg.norm(data_array, axis=1)
        if len(mags) > 1:
            # We use absolute difference per sample as a proxy for jerk
            jerks = np.abs(np.diff(mags))
            peak_jerk = np.max(jerks)
        else:
            peak_jerk = 0.0
            
        JERK_CEILING = 15.0 # configurable ceiling for maximum jerk scaling
        jerk_norm = np.clip(peak_jerk / JERK_CEILING, 0.0, 1.0)
        
        # CNN Inference
        cnn_prob = 0.5 # Default fallback
        if self.interpreter is not None:
            expected_timesteps = self.input_details[0]['shape'][1]
            
            # Pad or truncate
            if len(data_array) > expected_timesteps:
                cnn_input = data_array[-expected_timesteps:]
            elif len(data_array) < expected_timesteps:
                pad_length = expected_timesteps - len(data_array)
                padding = np.zeros((pad_length, 3), dtype=np.float32)
                cnn_input = np.vstack([padding, data_array])
            else:
                cnn_input = data_array
                
            input_data = np.expand_dims(cnn_input, axis=0)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            cnn_prob = float(output_data[0][0])
            
        # Combine CNN prob and Jerk heuristic
        motion_score = np.clip((0.5 * cnn_prob * 100.0) + (0.5 * jerk_norm * 100.0), 0.0, 100.0)
        return float(motion_score)
