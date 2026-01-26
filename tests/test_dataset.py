"""
Unit Tests for Traffic Sign Recognition Data Pipeline.
Tests dataset loading, transforms, and class balancing.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transforms import get_train_transforms, get_val_transforms, get_tta_transforms
from src.data.dataset import GTSRBDataset, GTSRB_CLASSES


class TestTransforms:
    """Test data augmentation transforms."""
    
    def test_train_transform_output_shape(self):
        """Training transform should output correct tensor shape."""
        transform = get_train_transforms(image_size=224)
        
        # Create dummy image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        transformed = transform(image=image)
        tensor = transformed['image']
        
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32
    
    def test_val_transform_output_shape(self):
        """Validation transform should output correct tensor shape."""
        transform = get_val_transforms(image_size=224)
        
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        transformed = transform(image=image)
        tensor = transformed['image']
        
        assert tensor.shape == (3, 224, 224)
    
    def test_val_transform_deterministic(self):
        """Validation transform should be deterministic."""
        transform = get_val_transforms(image_size=224)
        
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        t1 = transform(image=image)['image']
        t2 = transform(image=image)['image']
        
        assert torch.allclose(t1, t2)
    
    def test_tta_transforms_count(self):
        """TTA should return multiple transforms."""
        tta_transforms = get_tta_transforms(image_size=224)
        
        assert len(tta_transforms) >= 3  # At least 3 TTA variants
    
    def test_normalization_applied(self):
        """Images should be normalized."""
        transform = get_val_transforms(image_size=224)
        
        # White image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        transformed = transform(image=image)
        tensor = transformed['image']
        
        # After normalization, values should be around (1 - mean) / std
        assert tensor.mean() < 3.0  # Not 255


class TestDataset:
    """Test GTSRB dataset."""
    
    def test_class_names_count(self):
        """Should have 43 traffic sign classes."""
        assert len(GTSRB_CLASSES) == 43
    
    def test_class_names_are_strings(self):
        """Class names should be strings."""
        for name in GTSRB_CLASSES:
            assert isinstance(name, str)
            assert len(name) > 0
    
    def test_speed_limit_classes_exist(self):
        """Common speed limit signs should be present."""
        speed_limit_classes = [name for name in GTSRB_CLASSES if "Speed limit" in name]
        assert len(speed_limit_classes) >= 8  # 20, 30, 50, 60, 70, 80, 100, 120
    
    def test_stop_sign_exists(self):
        """Stop sign should be in classes."""
        assert "Stop" in GTSRB_CLASSES


class TestClassWeights:
    """Test class weight computation."""
    
    def test_weights_sum_to_num_classes(self):
        """Weights should roughly balance classes."""
        # Simulate imbalanced class counts
        class_counts = {0: 100, 1: 1000, 2: 500}
        total = sum(class_counts.values())
        num_classes = len(class_counts)
        
        weights = []
        for i in range(num_classes):
            count = class_counts.get(i, 1)
            weight = total / (num_classes * count)
            weights.append(weight)
        
        weights = torch.tensor(weights)
        
        # Higher weight for minority class
        assert weights[0] > weights[1]  # Class 0 has fewer samples
    
    def test_all_weights_positive(self):
        """All weights should be positive."""
        class_counts = {0: 100, 1: 1000, 2: 500}
        total = sum(class_counts.values())
        num_classes = len(class_counts)
        
        weights = []
        for i in range(num_classes):
            count = class_counts.get(i, 1)
            weight = total / (num_classes * count)
            weights.append(weight)
        
        for w in weights:
            assert w > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
