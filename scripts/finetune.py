"""
Fine-Tune the TSR Model with Real Dashcam Data.

This is a SEPARATE script from train.py — it never modifies the original.

What it does:
1. Loads the EXISTING best_model.pth (read-only)
2. Mixes synthetic data (80%) + real dashcam crops (20%)
3. Fine-tunes for 15 epochs with a very low learning rate
4. Saves the result as best_model_finetuned.pth (NEW file)

The original model is NEVER overwritten.

Usage:
    python scripts/finetune.py --config config_finetune.yaml
"""

import os
import sys
import yaml
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from PIL import Image
from sklearn.metrics import f1_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.hybrid_model import create_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Dataset for real dashcam crops (class-organized folders) ────────────

class RealCropsDataset(Dataset):
    """Dataset for real dashcam crops organized in class folders."""

    def __init__(self, root_dir: str, class_names: list, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.samples = []

        if not self.root_dir.exists():
            print(f"Warning: Real crops directory not found: {self.root_dir}")
            return

        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir() and class_dir.name in self.class_to_idx:
                class_idx = self.class_to_idx[class_dir.name]
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), class_idx))

        print(f"Loaded {len(self.samples)} real crops from {len(set(s[1] for s in self.samples))} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


# ─── Main Fine-Tuning Logic ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune TSR model with real dashcam data")
    parser.add_argument("--config", type=str, default="config_finetune.yaml",
                        help="Path to fine-tuning config")
    args = parser.parse_args()

    # Load config
    config_path = project_root / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))

    # Setup directories (all NEW — never overwrites existing)
    outputs_dir = project_root / config["paths"]["outputs_dir"]
    logs_dir = project_root / config["paths"]["logs_dir"]
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = logs_dir / f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ─── Load existing model (READ-ONLY) ─────────────────────────────
    base_model_path = project_root / config["base_model"]
    output_model_path = project_root / config["output_model"]

    logger.info(f"Loading base model from: {base_model_path}")
    logger.info(f"Fine-tuned model will be saved to: {output_model_path}")
    logger.info(f"[SAFE] Original model will NOT be modified.")

    # Create model architecture
    model = create_model(config)

    # Load existing weights
    checkpoint = torch.load(base_model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    logger.info("Base model loaded successfully.")

    # ─── Load class names ────────────────────────────────────────────
    classes_file = project_root / "sri_lankan_traffc_signs" / "synthetic" / "labels_yolo" / "classes.txt"
    if classes_file.exists():
        with open(classes_file) as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        class_names = [f"Class_{i}" for i in range(config["dataset"]["num_classes"])]
    logger.info(f"Loaded {len(class_names)} class names")

    # ─── Prepare datasets ────────────────────────────────────────────
    image_size = config["dataset"]["image_size"]
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    # Load real dashcam crops
    real_crops_path = project_root / config["dataset"]["real_crops_path"]
    real_dataset = RealCropsDataset(str(real_crops_path), class_names, transform=train_transform)

    if len(real_dataset) == 0:
        logger.error("No real crops found! Run crop_sign_regions.py first.")
        logger.error(f"Expected crops at: {real_crops_path}")
        sys.exit(1)

    # Load synthetic data (existing, read-only)
    from src.data.dataset import GTSRBDataset
    synthetic_path = config["dataset"]["synthetic_path"]
    synthetic_csv = str(Path(synthetic_path) / "Train.csv")
    synthetic_dataset = GTSRBDataset(
        data_dir=synthetic_path,
        csv_file=synthetic_csv,
        transform=train_transform,
        is_train=True
    )
    logger.info(f"Synthetic dataset: {len(synthetic_dataset)} images")

    # Combine datasets
    combined_dataset = ConcatDataset([synthetic_dataset, real_dataset])

    # Create weighted sampler to oversample real data
    real_ratio = config["dataset"].get("real_data_ratio", 0.2)
    synthetic_weight = 1.0
    real_weight = (len(synthetic_dataset) / max(len(real_dataset), 1)) * (real_ratio / (1 - real_ratio))

    weights = [synthetic_weight] * len(synthetic_dataset) + [real_weight] * len(real_dataset)
    sampler = WeightedRandomSampler(weights, num_samples=len(combined_dataset), replacement=True)

    train_loader = DataLoader(
        combined_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    logger.info(f"Combined dataset: {len(combined_dataset)} images")
    logger.info(f"Real data weight: {real_weight:.2f}x (targeting {real_ratio*100:.0f}% of batches)")

    # ─── Fine-tuning setup ───────────────────────────────────────────
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    freeze_epochs = config["training"].get("freeze_backbone_epochs", 3)

    # Optimizer (low learning rate to preserve existing knowledge)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config["training"]["weight_decay"]
    )

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=config["training"].get("label_smoothing", 0.1))

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Mixed precision
    use_amp = config["training"].get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    # ─── Training loop ───────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"  FINE-TUNING (preserving original model)")
    logger.info(f"  Epochs: {epochs} | LR: {lr} | Freeze backbone: {freeze_epochs} epochs")
    logger.info(f"{'='*60}\n")

    best_loss = float("inf")

    for epoch in range(epochs):
        # Freeze/unfreeze backbone
        if epoch < freeze_epochs:
            for name, param in model.named_parameters():
                if "backbone" in name or "feature_extractor" in name:
                    param.requires_grad = False
            if epoch == 0:
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"Backbone FROZEN — {trainable:,} trainable params (classifier head only)")
        elif epoch == freeze_epochs:
            for param in model.parameters():
                param.requires_grad = True
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Backbone UNFROZEN — {trainable:,} trainable params (full model)")

        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        logger.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Save best model (as NEW file — never overwrites original)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "loss": epoch_loss,
                "accuracy": epoch_acc,
                "config": config,
                "finetuned_from": str(base_model_path),
                "finetuned_at": datetime.now().isoformat()
            }, output_model_path)
            logger.info(f"  ★ Saved best model to {output_model_path}")

    # ─── Done ────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"  FINE-TUNING COMPLETE!")
    logger.info(f"  Original model: {base_model_path} (UNCHANGED)")
    logger.info(f"  Fine-tuned model: {output_model_path} (NEW)")
    logger.info(f"  Best loss: {best_loss:.4f}")
    logger.info(f"{'='*60}\n")

    print(f"\nTo switch to the fine-tuned model, update predictor_service.py:")
    print(f'  model_path = project_root / "models_sri_lanka" / "best_model_finetuned.pth"')
    print(f"\nTo revert, change it back to:")
    print(f'  model_path = project_root / "models_sri_lanka" / "best_model.pth"')


if __name__ == "__main__":
    main()
