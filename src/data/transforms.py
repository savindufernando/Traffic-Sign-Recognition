"""
Data Augmentation Transforms for Traffic Sign Recognition.
Includes weather effects, motion blur, and occlusion simulation.
Supports fallback to torchvision if albumentations has issues.
"""

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Try to import albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except Exception as e:
    print(f"Warning: albumentations not fully available: {e}")
    print("Using torchvision transforms instead")
    ALBUMENTATIONS_AVAILABLE = False


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================================
# Torchvision Fallback Transforms
# ============================================================================

class TorchvisionTrainTransform:
    """Torchvision-based training transforms."""
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Random augmentations
        image = T.Resize((self.image_size, self.image_size))(image)
        
        # Random rotation
        if torch.rand(1) < 0.5:
            angle = torch.randint(-15, 15, (1,)).item()
            image = TF.rotate(image, angle)
        
        # Random brightness/contrast
        if torch.rand(1) < 0.4:
            brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.6
            image = TF.adjust_brightness(image, brightness)
        
        if torch.rand(1) < 0.4:
            contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.6
            image = TF.adjust_contrast(image, contrast)
        
        # Random color jitter
        if torch.rand(1) < 0.3:
            image = T.ColorJitter(hue=0.1, saturation=0.2)(image)
        
        # To tensor and normalize
        tensor = TF.to_tensor(image)
        tensor = self.normalize(tensor)
        
        return {'image': tensor}


class TorchvisionValTransform:
    """Torchvision-based validation transforms."""
    
    def __init__(self, image_size: int = 224):
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return {'image': self.transform(image)}


# ============================================================================
# Albumentations Transforms (if available)
# ============================================================================

def _get_albumentations_train_transforms(
    image_size: int = 224,
    motion_blur_prob: float = 0.3,
    weather_prob: float = 0.2,
    occlusion_prob: float = 0.3,
    affine_prob: float = 0.5,
    brightness_contrast_prob: float = 0.4
):
    """Get albumentations training transforms."""
    return A.Compose([
        A.Resize(image_size, image_size),
        
        # Motion blur
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 7)),
            A.GaussianBlur(blur_limit=(3, 5)),
        ], p=motion_blur_prob),
        
        # Geometric transforms
        A.Affine(
            scale=(0.85, 1.15),
            rotate=(-15, 15),
            shear=(-10, 10),
            mode=0,
            p=affine_prob
        ),
        
        # Color augmentation
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=brightness_contrast_prob
        ),
        
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
        ], p=0.3),
        
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
        
        # Real-world conditions for Sri Lankan roads
        A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20,
                     drop_width=1, blur_value=3, p=0.15),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.1),
        
        # Partial occlusion (tree branches, stickers, dirt)
        A.CoarseDropout(max_holes=3, max_height=40, max_width=40,
                        min_height=10, min_width=10, fill_value=0, p=0.2),
        
        # Adaptive histogram equalization (poor lighting, shadows)
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
        
        # Viewing angle variation
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _get_albumentations_val_transforms(image_size: int = 224):
    """Get albumentations validation transforms."""
    return A.Compose([
        A.Resize(image_size, image_size),
        # Add CLAHE for consistent contrast in various lighting (night/glare)
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ============================================================================
# Public API
# ============================================================================

def get_train_transforms(image_size: int = 224, **kwargs):
    """
    Get training augmentation pipeline.
    Uses albumentations if available, otherwise torchvision.
    """
    if ALBUMENTATIONS_AVAILABLE:
        return _get_albumentations_train_transforms(image_size, **kwargs)
    else:
        return TorchvisionTrainTransform(image_size)


def get_val_transforms(image_size: int = 224):
    """
    Get validation/test transform pipeline.
    Uses albumentations if available, otherwise torchvision.
    """
    if ALBUMENTATIONS_AVAILABLE:
        return _get_albumentations_val_transforms(image_size)
    else:
        return TorchvisionValTransform(image_size)


def get_tta_transforms(image_size: int = 224):
    """Get Test-Time Augmentation transforms."""
    # Use torchvision for TTA as it's simpler
    return [
        TorchvisionValTransform(image_size),
        # Add more variants if needed
    ]
