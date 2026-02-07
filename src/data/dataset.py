"""
GTSRB Dataset Implementation.
Handles data loading, class balancing, and train/val splits.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from .transforms import get_train_transforms, get_val_transforms


# Traffic sign class names (GTSRB)
# Default GTSRB classes (fallback)
GTSRB_CLASSES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield",
    "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited",
    "No entry", "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road",
    "Slippery road", "Road narrows on the right", "Road work",
    "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]

def load_class_names(data_dir: Path) -> List[str]:
    """Load class names from classes.txt if available, else use default."""
    names_file = data_dir / "classes.txt"
    if names_file.exists():
        with open(names_file, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    
    # Try dataset.yaml (YOLO format)
    yaml_file = data_dir / "dataset.yaml"
    if yaml_file.exists():
        import yaml
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    return list(data['names'].values())
        except Exception:
            pass
            
    return GTSRB_CLASSES


class GTSRBDataset(Dataset):
    """
    GTSRB (German Traffic Sign Recognition Benchmark) Dataset.
    
    Args:
        data_dir: Path to dataset root
        csv_file: CSV file with image paths and labels
        transform: Albumentations transform pipeline
        is_train: Whether this is training set
    """
    
    def __init__(
        self,
        data_dir: str,
        csv_file: str,
        transform=None,
        is_train: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        
        # Load CSV
        self.df = pd.read_csv(csv_file)
        
        # Get image paths and labels
        self.image_paths = self.df['Path'].values
        self.labels = self.df['ClassId'].values
        
        # Compute class weights for imbalance handling
        self.class_counts = Counter(self.labels)
        # Load class names dynamically
        self.classes = load_class_names(self.data_dir)
        self.num_classes = len(self.classes)
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image
        img_path = self.data_dir / self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Get label
        label = int(self.labels[idx])
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for loss function."""
        total_samples = len(self.labels)
        weights = []
        for i in range(self.num_classes):
            count = self.class_counts.get(i, 1)
            weight = total_samples / (self.num_classes * count)
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_sample_weights(self) -> List[float]:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label].item() for label in self.labels]
        return sample_weights
    
    def get_class_name(self, class_id: int) -> str:
        """Get human-readable class name."""
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return str(class_id)


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    image_size: int = 224,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Path to GTSRB dataset root
        batch_size: Batch size
        val_split: Validation split ratio
        image_size: Target image size
        num_workers: DataLoader workers
        use_weighted_sampler: Use class-balanced sampling
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_path = Path(data_dir)
    
    # Load training CSV
    train_csv = data_path / 'Train.csv'
    train_df = pd.read_csv(train_csv)
    
    # Stratified train/val split
    train_indices, val_indices = train_test_split(
        np.arange(len(train_df)),
        test_size=val_split,
        stratify=train_df['ClassId'].values,
        random_state=seed
    )
    
    # Create split DataFrames
    train_split_df = train_df.iloc[train_indices].reset_index(drop=True)
    val_split_df = train_df.iloc[val_indices].reset_index(drop=True)
    
    # Save temporary split CSVs
    train_split_csv = data_path / 'Train_split.csv'
    val_split_csv = data_path / 'Val_split.csv'
    train_split_df.to_csv(train_split_csv, index=False)
    val_split_df.to_csv(val_split_csv, index=False)
    
    # Get transforms
    train_transform = get_train_transforms(image_size=image_size)
    val_transform = get_val_transforms(image_size=image_size)
    
    # Create datasets
    train_dataset = GTSRBDataset(
        data_dir=data_dir,
        csv_file=str(train_split_csv),
        transform=train_transform,
        is_train=True
    )
    
    val_dataset = GTSRBDataset(
        data_dir=data_dir,
        csv_file=str(val_split_csv),
        transform=val_transform,
        is_train=False
    )
    
    test_dataset = GTSRBDataset(
        data_dir=data_dir,
        csv_file=str(data_path / 'Test.csv'),
        transform=val_transform,
        is_train=False
    )
    
    # Create samplers
    train_sampler = None
    shuffle = True
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
