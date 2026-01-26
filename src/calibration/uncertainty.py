"""
MC Dropout Uncertainty Estimation for Traffic Sign Recognition.
Provides epistemic uncertainty estimates via Monte Carlo Dropout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class MCDropoutPredictor:
    """
    Monte Carlo Dropout predictor for uncertainty estimation.
    
    Performs multiple forward passes with dropout enabled to estimate
    epistemic (model) uncertainty.
    
    Paper: "Dropout as a Bayesian Approximation" (Gal & Ghahramani, 2016)
    
    Args:
        model: Neural network model with dropout layers
        n_samples: Number of MC samples
        device: Device to use
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 30,
        device: torch.device = None
    ):
        self.model = model
        self.n_samples = n_samples
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            mean_probs: Mean predicted probabilities (B, num_classes)
            epistemic_uncertainty: Standard deviation of predictions (B, num_classes)
            predictive_entropy: Predictive entropy (B,)
        """
        self.model.eval()
        self._enable_dropout()
        
        x = x.to(self.device)
        predictions = []
        
        for _ in range(self.n_samples):
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            predictions.append(probs)
        
        # Stack: (n_samples, B, num_classes)
        predictions = torch.stack(predictions)
        
        # Mean prediction
        mean_probs = predictions.mean(dim=0)
        
        # Epistemic uncertainty (std across samples)
        epistemic_uncertainty = predictions.std(dim=0)
        
        # Predictive entropy
        predictive_entropy = self._compute_entropy(mean_probs)
        
        return mean_probs, epistemic_uncertainty, predictive_entropy
    
    @staticmethod
    def _compute_entropy(probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distribution."""
        return -(probs * torch.log(probs + 1e-10)).sum(dim=1)
    
    def predict_with_rejection(
        self,
        x: torch.Tensor,
        entropy_threshold: float = 1.0,
        uncertainty_threshold: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with rejection option for uncertain predictions.
        
        Args:
            x: Input tensor
            entropy_threshold: Reject if entropy > threshold
            uncertainty_threshold: Reject if mean uncertainty > threshold
            
        Returns:
            predictions: Predicted class labels (-1 for rejected)
            confidences: Confidence scores
            should_reject: Boolean mask for rejected predictions
        """
        mean_probs, epistemic_unc, entropy = self.predict(x)
        
        # Get predictions and confidences
        confidences, predictions = mean_probs.max(dim=1)
        
        # Mean uncertainty across classes
        mean_unc = epistemic_unc.mean(dim=1)
        
        # Rejection criteria
        should_reject = (entropy > entropy_threshold) | (mean_unc > uncertainty_threshold)
        
        # Set rejected predictions to -1
        predictions[should_reject] = -1
        
        return predictions, confidences, should_reject
    
    def get_calibrated_probabilities(
        self,
        x: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Get temperature-scaled MC dropout probabilities.
        
        Args:
            x: Input tensor
            temperature: Scaling temperature
            
        Returns:
            Calibrated probabilities
        """
        self.model.eval()
        self._enable_dropout()
        
        x = x.to(self.device)
        predictions = []
        
        for _ in range(self.n_samples):
            logits = self.model(x)
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=1)
            predictions.append(probs)
        
        return torch.stack(predictions).mean(dim=0)


def uncertainty_decomposition(
    predictions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose total uncertainty into epistemic and aleatoric components.
    
    Args:
        predictions: MC sample predictions (n_samples, B, num_classes)
        
    Returns:
        total_uncertainty: Total predictive uncertainty
        epistemic_uncertainty: Model uncertainty (reducible with more data)
        aleatoric_uncertainty: Data uncertainty (irreducible)
    """
    mean_probs = predictions.mean(dim=0)
    
    # Total uncertainty (entropy of mean)
    total_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)
    
    # Expected entropy (mean of individual entropies) - aleatoric
    individual_entropies = -(predictions * torch.log(predictions + 1e-10)).sum(dim=-1)
    aleatoric = individual_entropies.mean(dim=0)
    
    # Epistemic = Total - Aleatoric (mutual information)
    epistemic = total_entropy - aleatoric
    
    return total_entropy, epistemic, aleatoric
