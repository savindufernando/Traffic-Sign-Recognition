"""
Hybrid CNN-Transformer Model for Traffic Sign Recognition.
Combines ConvNeXt backbone with Transformer encoder for high accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple, Dict

from .attention import SEBlock, CBAM, TransformerEncoder


class TrafficSignModel(nn.Module):
    """
    Hybrid ConvNeXt + Transformer model for traffic sign recognition.
    
    Architecture:
        Input Image (224×224×3)
            ↓
        ConvNeXt Backbone (pretrained)
            ↓
        SE-Block Attention
            ↓
        Patch Embedding + Position Encoding
            ↓
        Transformer Encoder (2-4 layers)
            ↓
        [CLS] Token → MLP Head
            ↓
        Predictions (43 classes)
    
    Args:
        num_classes: Number of output classes (43 for GTSRB)
        backbone: Backbone model name (convnext_tiny, efficientnet_b0, resnet34)
        pretrained: Use pretrained backbone weights
        transformer_layers: Number of transformer encoder layers
        transformer_heads: Number of attention heads
        transformer_dim: Transformer model dimension
        dropout: Dropout probability
        use_attention: Use SE-Block attention after backbone
    """
    
    def __init__(
        self,
        num_classes: int = 43,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_dim: int = 384,
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.transformer_dim = transformer_dim
        self.use_attention = use_attention
        
        # Create backbone (without classification head)
        self.backbone_name = backbone
        self.is_lightweight = 'mobilevit' in backbone or 'fastvit' in backbone
        
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Get backbone output channels
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_out = self.backbone(dummy_input)
            if isinstance(backbone_out, tuple):
                backbone_out = backbone_out[0]
            
            # Handle different output formats
            if len(backbone_out.shape) == 4:
                # (B, C, H, W) format
                self.backbone_channels = backbone_out.shape[1]
                self.feature_size = backbone_out.shape[2]  # H = W
            else:
                # (B, N, C) format (already flattened)
                self.backbone_channels = backbone_out.shape[-1]
                self.feature_size = int(backbone_out.shape[1] ** 0.5)
        
        # SE-Block attention (optional)
        if use_attention:
            self.attention = SEBlock(self.backbone_channels, reduction=16)
        else:
            self.attention = nn.Identity()
        
        # Project backbone features to transformer dimension
        self.proj = nn.Sequential(
            nn.Conv2d(self.backbone_channels, transformer_dim, 1),
            nn.BatchNorm2d(transformer_dim),
            nn.GELU()
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))
        
        # Position embedding
        num_patches = self.feature_size * self.feature_size
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, transformer_dim) * 0.02)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout
        )
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize learnable parameters."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, H, W)
            return_features: If True, return features before classification head
            
        Returns:
            Logits (B, num_classes) or features if return_features=True
        """
        B = x.shape[0]
        
        # Backbone feature extraction
        features = self.backbone(x)
        
        # Handle different backbone output formats
        if isinstance(features, tuple):
            features = features[0]
        
        # Ensure (B, C, H, W) format
        if len(features.shape) == 3:
            # (B, N, C) -> (B, C, H, W)
            H = W = int(features.shape[1] ** 0.5)
            features = features.permute(0, 2, 1).reshape(B, -1, H, W)
        
        # SE-Block attention
        features = self.attention(features)
        
        # Project to transformer dimension
        features = self.proj(features)  # (B, transformer_dim, H, W)
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        features = features.flatten(2).permute(0, 2, 1)  # (B, num_patches, transformer_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        features = torch.cat([cls_tokens, features], dim=1)  # (B, num_patches+1, transformer_dim)
        
        # Add position embedding
        features = features + self.pos_embed
        
        # Transformer encoding (Skip for inherently lightweight vision transformers)
        if not self.is_lightweight:
            features = self.transformer(features)
        else:
            # For lightweight models, just pass through the custom layers as a pooling mechanism
            # or apply a simpler pooling to save compute.
            pass
        
        # Extract CLS token
        cls_output = features[:, 0]  # (B, transformer_dim)
        
        if return_features:
            return cls_output
        
        # Classification
        logits = self.head(cls_output)
        
        return logits
    
    def get_attention_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention weights for visualization.
        
        Note: This requires modifying the attention layers to return weights.
        For now, returns dummy weights.
        """
        # TODO: Implement proper attention weight extraction
        return {}
    
    def enable_mc_dropout(self):
        """Enable dropout layers for MC Dropout uncertainty estimation."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        n_samples: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with MC Dropout uncertainty estimation.
        
        Args:
            x: Input images
            n_samples: Number of MC samples
            
        Returns:
            mean_probs: Mean predicted probabilities
            epistemic_uncertainty: Epistemic uncertainty (std of predictions)
            predictive_entropy: Predictive entropy
        """
        self.eval()
        self.enable_mc_dropout()
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        # Stack predictions: (n_samples, B, num_classes)
        predictions = torch.stack(predictions)
        
        # Mean prediction
        mean_probs = predictions.mean(dim=0)
        
        # Epistemic uncertainty (standard deviation)
        epistemic_uncertainty = predictions.std(dim=0)
        
        # Predictive entropy
        predictive_entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=1)
        
        return mean_probs, epistemic_uncertainty, predictive_entropy


def create_model(config: dict) -> TrafficSignModel:
    """
    Create model from configuration dictionary.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized model
    """
    model_config = config.get('model', {})
    
    return TrafficSignModel(
        num_classes=config.get('dataset', {}).get('num_classes', 43),
        backbone=model_config.get('backbone', 'convnext_tiny'),
        pretrained=model_config.get('backbone_pretrained', True),
        transformer_layers=model_config.get('transformer_layers', 2),
        transformer_heads=model_config.get('transformer_heads', 8),
        transformer_dim=model_config.get('transformer_dim', 384),
        dropout=model_config.get('dropout', 0.2),
        use_attention=model_config.get('use_attention', True)
    )
