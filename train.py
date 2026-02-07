"""
Main Training Script for Traffic Sign Recognition.
Orchestrates data loading, model creation, and training.
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import get_dataloaders
from src.models import TrafficSignModel
from src.models.hybrid_model import create_model
from src.training import Trainer, CombinedLoss, FocalLoss
from src.training.losses import create_loss_function


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    log_file = output_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train Traffic Sign Recognition Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode (5 epochs)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply overrides
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.quick_test:
        config['training']['epochs'] = 5
        print("\n⚡ Quick test mode: Training for 5 epochs only\n")
    
    # Create directories
    models_dir = Path(config['paths']['models_dir'])
    outputs_dir = Path(config['paths']['outputs_dir'])
    logs_dir = Path(config['paths']['logs_dir'])
    
    for d in [models_dir, outputs_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(logs_dir)
    
    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data
    logger.info("Loading data...")
    data_dir = config['dataset']['path']
    batch_size = config['training']['batch_size']
    image_size = config['dataset']['image_size']
    val_split = config['dataset']['val_split']
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=val_split,
        image_size=image_size,
        num_workers=4,
        use_weighted_sampler=True,
        seed=seed
    )
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Get class weights from training dataset
    class_weights = train_loader.dataset.get_class_weights()
    logger.info(f"Class weights computed (range: {class_weights.min():.2f} - {class_weights.max():.2f})")
    
    # Model
    logger.info("Creating model...")
    model = create_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = create_loss_function(config, class_weights)
    logger.info("Loss function: Focal + Label Smoothing")
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from checkpoint: {args.resume} (next epoch {start_epoch + 1})")
    
    # Train
    print("\n" + "="*60)
    print("  TRAFFIC SIGN RECOGNITION - TRAINING")
    print("="*60)
    print(f"  Model: {config['model']['backbone']} + Transformer")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"  Device: {device}")
    print("="*60 + "\n")
    
    history = trainer.train()
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("  TEST SET EVALUATION")
    print("="*60)
    
    # Load best model
    best_model_path = models_dir / 'best_model.pth'
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    trainer.val_loader = test_loader  # Swap to test loader
    test_loss, test_acc, test_f1 = trainer.validate()
    
    print(f"\n  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1 Score: {test_f1:.4f}")
    print("\n" + "="*60)
    print("  TRAINING COMPLETE!")
    print(f"  Best model saved to: {best_model_path}")
    print("="*60 + "\n")
    
    # Save test metrics
    import json
    test_metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_f1': test_f1
    }
    with open(outputs_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == '__main__':
    main()
