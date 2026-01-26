"""
Loss Functions for Traffic Sign Recognition.
Includes Focal Loss and Label Smoothing for handling class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reduces the relative loss for well-classified examples, 
    focusing on hard, misclassified examples.
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Args:
        alpha: Class weights (tensor of shape [num_classes])
        gamma: Focusing parameter. Higher gamma = more focus on hard examples
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits (B, num_classes)
            targets: Ground truth labels (B,)
        """
        # Compute cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute pt (probability of correct class)
        pt = torch.exp(-ce_loss)
        
        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight
        
        # Compute focal loss
        focal_loss = focal_weight * ce_loss
        
        # Reduce
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-Entropy with Label Smoothing.
    
    Prevents overconfidence by smoothing the target distribution.
    
    Args:
        smoothing: Label smoothing factor (0.0 to 1.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits (B, num_classes)
            targets: Ground truth labels (B,)
        """
        num_classes = inputs.size(-1)
        
        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute log softmax
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Compute loss
        loss = (-smooth_targets * log_probs).sum(dim=-1)
        
        # Reduce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined Focal Loss + Label Smoothing.
    
    Best of both worlds: handles class imbalance and prevents overconfidence.
    
    Args:
        alpha: Class weights for Focal Loss
        gamma: Focusing parameter for Focal Loss
        smoothing: Label smoothing factor
        focal_weight: Weight for Focal Loss component
        smooth_weight: Weight for Label Smoothing component
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        smoothing: float = 0.1,
        focal_weight: float = 0.5,
        smooth_weight: float = 0.5
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.smooth_loss = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.focal_weight = focal_weight
        self.smooth_weight = smooth_weight
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        focal = self.focal_loss(inputs, targets)
        smooth = self.smooth_loss(inputs, targets)
        return self.focal_weight * focal + self.smooth_weight * smooth


def create_loss_function(
    config: dict,
    class_weights: Optional[torch.Tensor] = None
) -> nn.Module:
    """
    Create loss function from configuration.
    
    Args:
        config: Training configuration
        class_weights: Optional class weights for imbalance handling
        
    Returns:
        Loss function module
    """
    training_config = config.get('training', {})
    
    gamma = training_config.get('focal_loss_gamma', 2.0)
    smoothing = training_config.get('label_smoothing', 0.1)
    
    return CombinedLoss(
        alpha=class_weights,
        gamma=gamma,
        smoothing=smoothing,
        focal_weight=0.5,
        smooth_weight=0.5
    )
