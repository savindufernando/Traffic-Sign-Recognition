"""
Unit Tests for Traffic Sign Recognition Model.
Tests model architecture, forward pass, and gradient flow.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.attention import SEBlock, CBAM, TransformerEncoder
from src.models.hybrid_model import TrafficSignModel, create_model


class TestSEBlock:
    """Test Squeeze-and-Excitation block."""
    
    def test_output_shape(self):
        """SE block should preserve input shape."""
        se = SEBlock(channels=64, reduction=16)
        x = torch.randn(2, 64, 14, 14)
        
        out = se(x)
        
        assert out.shape == x.shape
    
    def test_values_scaled(self):
        """SE block should scale input values."""
        se = SEBlock(channels=64, reduction=16)
        x = torch.randn(2, 64, 14, 14)
        
        out = se(x)
        
        # Output should be different from input (scaled)
        assert not torch.allclose(out, x)


class TestCBAM:
    """Test CBAM attention module."""
    
    def test_output_shape(self):
        """CBAM should preserve input shape."""
        cbam = CBAM(channels=64, reduction=16)
        x = torch.randn(2, 64, 14, 14)
        
        out = cbam(x)
        
        assert out.shape == x.shape
    
    def test_gradient_flow(self):
        """Gradient should flow through CBAM."""
        cbam = CBAM(channels=64, reduction=16)
        x = torch.randn(2, 64, 14, 14, requires_grad=True)
        
        out = cbam(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestTransformerEncoder:
    """Test custom Transformer encoder."""
    
    def test_output_shape(self):
        """Transformer should output correct shape."""
        encoder = TransformerEncoder(
            d_model=384,
            nhead=8,
            num_layers=2,
            dim_feedforward=1536,
            dropout=0.1
        )
        
        x = torch.randn(2, 50, 384)  # (batch, seq_len, d_model)
        out = encoder(x)
        
        assert out.shape == x.shape
    
    def test_gradient_flow(self):
        """Gradient should flow through transformer."""
        encoder = TransformerEncoder(d_model=384, nhead=8, num_layers=2)
        x = torch.randn(2, 50, 384, requires_grad=True)
        
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None


class TestTrafficSignModel:
    """Test hybrid CNN-Transformer model."""
    
    @pytest.fixture
    def model(self):
        """Create a lightweight model for testing."""
        return TrafficSignModel(
            num_classes=43,
            backbone='resnet18',  # Lighter backbone for testing
            pretrained=False,
            transformer_layers=1,
            transformer_heads=4,
            transformer_dim=256,
            dropout=0.1,
            use_attention=True
        )
    
    def test_forward_pass_shape(self, model):
        """Model output should have correct shape."""
        x = torch.randn(2, 3, 224, 224)
        
        out = model(x)
        
        assert out.shape == (2, 43)
    
    def test_forward_pass_no_nan(self, model):
        """Model output should not contain NaN."""
        x = torch.randn(2, 3, 224, 224)
        
        out = model(x)
        
        assert not torch.isnan(out).any()
    
    def test_gradient_flow_full_model(self, model):
        """Gradients should flow through entire model."""
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        # Check gradients exist for key parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_return_features(self, model):
        """Model should be able to return features."""
        x = torch.randn(2, 3, 224, 224)
        
        features = model(x, return_features=True)
        
        # Features should be transformer dimension
        assert features.shape == (2, 256)
    
    def test_eval_mode(self, model):
        """Model should work in eval mode."""
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (2, 43)


class TestModelCreation:
    """Test model creation from config."""
    
    def test_create_from_config(self):
        """Model should be created from config dict."""
        config = {
            'dataset': {'num_classes': 43},
            'model': {
                'backbone': 'resnet18',
                'backbone_pretrained': False,
                'transformer_layers': 1,
                'transformer_heads': 4,
                'transformer_dim': 256,
                'dropout': 0.1,
                'use_attention': True
            }
        }
        
        model = create_model(config)
        
        assert isinstance(model, TrafficSignModel)
        assert model.num_classes == 43


class TestModelSaveLoad:
    """Test model save and load functionality."""
    
    def test_state_dict_save_load(self, tmp_path):
        """Model weights should be saveable and loadable."""
        model1 = TrafficSignModel(
            num_classes=43,
            backbone='resnet18',
            pretrained=False,
            transformer_layers=1
        )
        
        # Save
        save_path = tmp_path / 'model.pth'
        torch.save(model1.state_dict(), save_path)
        
        # Load into new model
        model2 = TrafficSignModel(
            num_classes=43,
            backbone='resnet18',
            pretrained=False,
            transformer_layers=1
        )
        model2.load_state_dict(torch.load(save_path))
        
        # Compare outputs
        x = torch.randn(1, 3, 224, 224)
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)
        
        assert torch.allclose(out1, out2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
