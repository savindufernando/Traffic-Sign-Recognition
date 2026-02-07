"""
Training Loop for Traffic Sign Recognition.
Includes AMP, learning rate scheduling, early stopping, and checkpointing.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


class Trainer:
    """
    Training loop for traffic sign recognition.
    
    Features:
    - Mixed-precision training (AMP)
    - Cosine annealing with warm restarts
    - Early stopping
    - Model checkpointing
    - Comprehensive metrics logging
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss function
        config: Training configuration
        device: Training device
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config: dict,
        device: torch.device = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        
        # Device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        
        # Training config
        training_config = config.get('training', {})
        self.epochs = training_config.get('epochs', 50)
        self.lr = training_config.get('learning_rate', 1e-4)
        self.weight_decay = training_config.get('weight_decay', 1e-4)
        self.warmup_epochs = training_config.get('warmup_epochs', 5)
        self.use_amp = training_config.get('use_amp', True) and torch.cuda.is_available()
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=10)
        
        # Paths
        paths_config = config.get('paths', {})
        self.models_dir = Path(paths_config.get('models_dir', 'models'))
        self.outputs_dir = Path(paths_config.get('outputs_dir', 'outputs'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Best metrics
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'lr': []
        }

        # Resume support
        self.start_epoch = 0
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Train]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with scaled gradients
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy_score(all_labels, all_preds):.4f}'
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc='Validating')
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, accuracy, f1
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'early_stopping': {
                'best_score': self.early_stopping.best_score,
                'counter': self.early_stopping.counter,
                'should_stop': self.early_stopping.should_stop
            },
            'config': self.config
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest
        torch.save(checkpoint, self.models_dir / 'latest_checkpoint.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.models_dir / 'best_model.pth')
            # Also save just model weights for easier loading
            torch.save(self.model.state_dict(), self.models_dir / 'best_model_weights.pth')

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint and return the next epoch to run."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.best_val_acc = checkpoint.get('best_val_acc', self.best_val_acc)
        self.best_epoch = checkpoint.get('best_epoch', checkpoint.get('epoch', self.best_epoch))

        if 'history' in checkpoint and isinstance(checkpoint['history'], dict):
            self.history = checkpoint['history']

        early_state = checkpoint.get('early_stopping')
        if isinstance(early_state, dict):
            self.early_stopping.best_score = early_state.get('best_score', self.early_stopping.best_score)
            self.early_stopping.counter = early_state.get('counter', self.early_stopping.counter)
            self.early_stopping.should_stop = early_state.get('should_stop', self.early_stopping.should_stop)
        elif self.best_val_acc:
            self.early_stopping.best_score = self.best_val_acc
            self.early_stopping.counter = 0
            self.early_stopping.should_stop = False

        last_epoch = int(checkpoint.get('epoch', -1))
        self.start_epoch = max(0, last_epoch + 1)
        return self.start_epoch
    
    def train(self) -> Dict:
        """
        Full training loop.
        
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        logger.info(f"AMP enabled: {self.use_amp}")
        
        start_time = time.time()
        
        if self.start_epoch >= self.epochs:
            logger.info(
                f"Checkpoint epoch {self.start_epoch} is already at or beyond total epochs {self.epochs}."
            )
            return self.history

        if self.start_epoch > 0:
            logger.info(f"Resuming from epoch {self.start_epoch + 1}/{self.epochs}")

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc, val_f1 = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(current_lr)
            
            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Logging
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
            )
            
            # Print to console
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"  LR: {current_lr:.6f} | Best Acc: {self.best_val_acc:.4f} (epoch {self.best_epoch+1})")
            
            # Early stopping
            if self.early_stopping(val_acc):
                logger.info(f"Early stopping at epoch {epoch+1}")
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Training complete
        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time/60:.1f} minutes")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        
        # Save history
        with open(self.outputs_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save final metrics
        final_metrics = {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch + 1,
            'total_epochs': len(self.history['train_loss']),
            'training_time_minutes': total_time / 60,
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'final_val_f1': self.history['val_f1'][-1]
        }
        
        with open(self.outputs_dir / 'final_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        return self.history
