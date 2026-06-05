"""
Inference Engine for Traffic Sign Recognition.
Provides single image and batch prediction with confidence scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from ..data.transforms import get_val_transforms, IMAGENET_MEAN, IMAGENET_STD
from ..models.hybrid_model import TrafficSignModel, create_model


class TrafficSignPredictor:
    """
    Inference engine for traffic sign recognition.
    
    Provides:
    - Single image prediction
    - Batch prediction
    - Confidence thresholding
    - Temperature-scaled calibration
    - Uncertainty estimation (MC Dropout)
    
    Args:
        model_path: Path to trained model weights
        config: Model configuration
        device: Inference device
        temperature: Calibration temperature
        confidence_threshold: Minimum confidence for valid prediction
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config: dict = None,
        device: torch.device = None,
        temperature: float = 1.0,
        confidence_threshold: float = 0.5
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        
        # Default config
        if config is None:
            config = {
                'dataset': {'num_classes': 43, 'image_size': 224},
                'model': {
                    'backbone': 'convnext_tiny',
                    'backbone_pretrained': False,
                    'transformer_layers': 2,
                    'transformer_heads': 8,
                    'transformer_dim': 384,
                    'dropout': 0.2,
                    'use_attention': True
                }
            }
        
        self.config = config
        self.image_size = config.get('dataset', {}).get('image_size', 224)
        
        # Load model
        self.model = create_model(config)
        self._load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = get_val_transforms(self.image_size)
        
        # Class names - load from dataset
        dataset_path = config.get('dataset', {}).get('path', '')
        num_classes = config.get('dataset', {}).get('num_classes', 122)
        self.class_names = self._load_class_names(dataset_path, num_classes)
    
    def _load_class_names(self, dataset_path: str, num_classes: int) -> list:
        """Load class names from dataset classes.txt file."""
        from pathlib import Path
        
        # Try to find classes.txt in various locations
        possible_paths = [
            Path(dataset_path) / "labels_yolo" / "classes.txt",
            Path(dataset_path) / "classes.txt",
            Path("sri_lankan_traffc_signs/synthetic/labels_yolo/classes.txt"),
            Path("d:/APIIT/FYP/Traffic-Sign-Recognition/sri_lankan_traffc_signs/synthetic/labels_yolo/classes.txt"),
        ]
        
        for path in possible_paths:
            if path.exists():
                with open(path, 'r') as f:
                    classes = [line.strip() for line in f if line.strip()]
                if len(classes) >= num_classes:
                    print(f"Loaded {len(classes)} class names from {path}")
                    return classes
        
        # Fallback: generate generic class names
        print(f"Warning: Could not find classes.txt, using generic class names")
        return [f"Class_{i}" for i in range(num_classes)]
    
    def _load_weights(self, model_path: Union[str, Path]):
        """Load model weights from checkpoint or state dict."""
        model_path = Path(model_path)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)
    
    def preprocess(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # ─── Night-Time/Low-Light Enhancement ─────────────────────────────
        # If mean intensity is lower than 40 (~15%), boost it slightly
        avg_intensity = np.mean(image)
        if avg_intensity < 40:
            # Shift pixel values up by the difference to reach a minimum baseline
            boost = int(45 - avg_intensity)
            image = np.clip(image.astype(np.int32) + boost, 0, 255).astype(np.uint8)
        
        # Apply transforms (now includes CLAHE as of recent update)
        transformed = self.transform(image=image)
        tensor = transformed['image']
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        return_all_probs: bool = False
    ) -> Dict:
        """
        Predict traffic sign class for a single image.
        
        Args:
            image: Input image
            return_all_probs: Return probabilities for all classes
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            tensor = image.to(self.device)
        else:
            tensor = self.preprocess(image).to(self.device)
        
        # Forward pass
        logits = self.model(tensor)
        
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=1)
        
        # Get prediction
        confidence, pred_class = probs.max(dim=1)
        confidence = confidence.item()
        pred_class = pred_class.item()
        
        # Check confidence threshold
        is_confident = confidence >= self.confidence_threshold
        
        result = {
            'class_id': pred_class,
            'class_name': self.class_names[pred_class],
            'confidence': confidence,
            'is_confident': is_confident,
            'temperature': self.temperature
        }
        
        if return_all_probs:
            result['all_probabilities'] = probs.cpu().numpy()[0]
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[np.ndarray, Image.Image]]
    ) -> List[Dict]:
        """
        Predict traffic sign classes for a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction dictionaries
        """
        # Preprocess all images
        tensors = [self.preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0).to(self.device)
        
        # Forward pass
        logits = self.model(batch)
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=1)
        
        # Get predictions
        confidences, pred_classes = probs.max(dim=1)
        
        results = []
        for i in range(len(images)):
            confidence = confidences[i].item()
            pred_class = pred_classes[i].item()
            
            results.append({
                'class_id': pred_class,
                'class_name': self.class_names[pred_class],
                'confidence': confidence,
                'is_confident': confidence >= self.confidence_threshold
            })
        
        return results
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        n_samples: int = 30
    ) -> Dict:
        """
        Predict with MC Dropout uncertainty estimation.
        
        Args:
            image: Input image
            n_samples: Number of MC samples
            
        Returns:
            Dictionary with prediction and uncertainty
        """
        # Preprocess
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            tensor = image.to(self.device)
        else:
            tensor = self.preprocess(image).to(self.device)
        
        # Enable dropout for MC sampling
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        
        # MC sampling
        predictions = []
        for _ in range(n_samples):
            logits = self.model(tensor)
            probs = F.softmax(logits / self.temperature, dim=1)
            predictions.append(probs)
        
        # Stack and compute statistics
        predictions = torch.stack(predictions)  # (n_samples, 1, num_classes)
        mean_probs = predictions.mean(dim=0)
        std_probs = predictions.std(dim=0)
        
        # Get prediction
        confidence, pred_class = mean_probs.max(dim=1)
        confidence = confidence.item()
        pred_class = pred_class.item()
        
        # Epistemic uncertainty (mean std)
        epistemic_unc = std_probs.mean().item()
        
        # Predictive entropy
        entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1).item()
        
        return {
            'class_id': pred_class,
            'class_name': self.class_names[pred_class],
            'confidence': confidence,
            'epistemic_uncertainty': epistemic_unc,
            'predictive_entropy': entropy,
            'is_confident': confidence >= self.confidence_threshold,
            'n_samples': n_samples
        }
    
    def get_top_k_predictions(
        self,
        image: Union[np.ndarray, Image.Image],
        k: int = 5
    ) -> List[Dict]:
        """
        Get top-k predictions for an image.
        
        Args:
            image: Input image
            k: Number of top predictions
            
        Returns:
            List of top-k predictions
        """
        result = self.predict(image, return_all_probs=True)
        probs = result['all_probabilities']
        
        # Get top-k indices
        top_k_indices = np.argsort(probs)[::-1][:k]
        
        predictions = []
        for idx in top_k_indices:
            predictions.append({
                'rank': len(predictions) + 1,
                'class_id': int(idx),
                'class_name': self.class_names[idx],
                'confidence': float(probs[idx])
            })
        
        return predictions
    
    def set_temperature(self, temperature: float):
        """Update calibration temperature."""
        self.temperature = temperature
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold."""
        self.confidence_threshold = threshold


def load_predictor(
    model_path: str,
    config_path: str = None,
    device: str = None
) -> TrafficSignPredictor:
    """
    Convenience function to load predictor.
    
    Args:
        model_path: Path to model weights
        config_path: Path to config file (optional)
        device: Device string ('cuda' or 'cpu')
        
    Returns:
        Initialized TrafficSignPredictor
    """
    import yaml
    
    config = None
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    if device:
        device = torch.device(device)
    
    return TrafficSignPredictor(
        model_path=model_path,
        config=config,
        device=device
    )
