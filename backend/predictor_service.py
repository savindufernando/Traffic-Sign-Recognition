"""
Traffic Sign Recognition - Predictor Service.
Singleton wrapper around TrafficSignPredictor for FastAPI.
"""

import sys
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
import io

# Add parent directory to path for imports
backend_dir = Path(__file__).parent
project_root = backend_dir.parent
sys.path.insert(0, str(project_root))


class PredictorService:
    """
    Singleton service for traffic sign prediction.
    Loads model once and reuses for all requests.
    """
    
    _instance: Optional['PredictorService'] = None
    _predictor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._predictor is None:
            self._load_model()
    
    def _load_model(self):
        """Load the prediction model."""
        try:
            from src.inference.predictor import load_predictor
            
            model_path = project_root / "models_sri_lanka" / "best_model.pth"
            config_path = project_root / "config_sri_lanka.yaml"
            
            print(f"Loading model from: {model_path}")
            self._predictor = load_predictor(str(model_path), str(config_path))
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise e
    
    def predict(self, image: Image.Image) -> Dict:
        """
        Predict traffic sign class from PIL Image.
        
        Returns:
            dict with class_name, class_id, confidence, is_confident, top_k
        """
        if self._predictor is None:
            raise RuntimeError("Model not loaded")
        
        # Get prediction
        result = self._predictor.predict(image, return_all_probs=False)
        
        # Get top-k predictions
        top_k = self._predictor.get_top_k_predictions(image, k=5)
        result['top_k'] = top_k
        
        return result
    
    def predict_from_bytes(self, image_bytes: bytes) -> Dict:
        """Predict from raw image bytes."""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return self.predict(image)
    
    def get_classes(self) -> list:
        """Return list of all traffic sign classes."""
        if self._predictor is None:
            return []
        return list(self._predictor.class_names.values()) if hasattr(self._predictor, 'class_names') else []


# Global service instance
predictor_service = PredictorService()
