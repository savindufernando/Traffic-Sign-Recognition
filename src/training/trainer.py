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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR, LinearLR
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


logger = logging.getLogger(__name__)


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM).
    Seeks parameters in flatter loss regions for better generalization.
    Foret et al., 2021 - https://arxiv.org/abs/2010.01412
    """
    
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Ascend to the sharpest point in the neighborhood."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)  # Move to sharpest point
                self.state[p]['e_w'] = e_w
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Descend from the sharpest point."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]['e_w'])  # Move back from sharp point
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def step(self, closure=None):
        """Standard step (calls base optimizer)."""
        return self.base_optimizer.step(closure)
    
    def state_dict(self):
        return self.base_optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)


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
        self.use_sam = training_config.get('use_sam', True)
        self.use_swa = training_config.get('use_swa', True)
        self.swa_start_pct = training_config.get('swa_start_pct', 0.85)  # Start SWA at 85% of training
        
        # SAM is incompatible with AMP GradScaler (causes NaN loss)
        # Auto-disable SAM when AMP is active; SWA still provides generalization benefits
        if self.use_sam and self.use_amp:
            logger.warning("SAM disabled: incompatible with AMP mixed precision. SWA remains active.")
            self.use_sam = False
        
        # Optimizer: SAM wraps AdamW for flatter minima
        if self.use_sam:
            self.optimizer = SAM(
                self.model.parameters(),
                base_optimizer=AdamW,
                rho=0.05,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
            logger.info("Using SAM optimizer (Sharpness-Aware Minimization)")
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        
        # Scheduler: Linear warmup → Cosine annealing with warm restarts
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=self.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs]
        )
        
        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # SWA (Stochastic Weight Averaging)
        if self.use_swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_start_epoch = int(self.epochs * self.swa_start_pct)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.lr * 0.5)
            logger.info(f"SWA enabled: averaging starts at epoch {self.swa_start_epoch + 1}")
        else:
            self.swa_model = None
            self.swa_start_epoch = self.epochs + 1  # Never triggers
        
        # Early stopping (monitors F1, more sensitive than accuracy at 99%+)
        self.early_stopping = EarlyStopping(patience=15, min_delta=0.0005)
        
        # Paths
        paths_config = config.get('paths', {})
        self.models_dir = Path(paths_config.get('models_dir', 'models'))
        self.outputs_dir = Path(paths_config.get('outputs_dir', 'outputs'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Best metrics
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
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
            
            # Forward pass with optional Mixup augmentation
            self.optimizer.zero_grad()
            
            # Mixup: blend samples 50% of the time for regularization
            use_mixup = torch.rand(1).item() < 0.5
            if use_mixup:
                lam = np.random.beta(0.2, 0.2)
                rand_idx = torch.randperm(images.size(0)).to(self.device)
                mixed_images = lam * images + (1 - lam) * images[rand_idx]
                labels_b = labels[rand_idx]
            
            # --- SAM dual-step optimization ---
            if self.use_sam:
                # First forward-backward (compute gradient at current point)
                if self.use_amp:
                    with autocast():
                        if use_mixup:
                            outputs = self.model(mixed_images)
                            loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(outputs, labels_b)
                        else:
                            outputs = self.model(images)
                            loss = self.criterion(outputs, labels)
                    # SAM + AMP: manually handle gradient scaling to avoid
                    # double unscale_ calls (GradScaler only allows one per cycle)
                    self.scaler.scale(loss).backward()
                    # Unscale for gradient clipping, then SAM first step
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    # Skip scaler.step — SAM handles the parameter update
                    self.optimizer.first_step(zero_grad=True)
                    # Must call update() to reset scaler state for next cycle
                    self.scaler.update()
                    
                    # Second forward-backward (compute gradient at sharpest point)
                    with autocast():
                        if use_mixup:
                            loss2 = lam * self.criterion(self.model(mixed_images), labels) + (1 - lam) * self.criterion(self.model(mixed_images), labels_b)
                        else:
                            loss2 = self.criterion(self.model(images), labels)
                    self.scaler.scale(loss2).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.second_step(zero_grad=True)
                    self.scaler.update()
                else:
                    if use_mixup:
                        outputs = self.model(mixed_images)
                        loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(outputs, labels_b)
                    else:
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.first_step(zero_grad=True)
                    
                    if use_mixup:
                        loss2 = lam * self.criterion(self.model(mixed_images), labels) + (1 - lam) * self.criterion(self.model(mixed_images), labels_b)
                    else:
                        loss2 = self.criterion(self.model(images), labels)
                    loss2.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.second_step(zero_grad=True)
            else:
                # Standard optimization (no SAM)
                if self.use_amp:
                    with autocast():
                        if use_mixup:
                            outputs = self.model(mixed_images)
                            loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(outputs, labels_b)
                        else:
                            outputs = self.model(images)
                            loss = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if use_mixup:
                        outputs = self.model(mixed_images)
                        loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(outputs, labels_b)
                    else:
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
            'best_val_f1': self.best_val_f1, # Changed from best_val_acc
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

        self.best_val_acc = checkpoint.get('best_val_acc', self.best_val_acc) # Keep for backward compatibility
        self.best_val_f1 = checkpoint.get('best_val_f1', self.best_val_f1) # New best metric
        self.best_epoch = checkpoint.get('best_epoch', checkpoint.get('epoch', self.best_epoch))

        if 'history' in checkpoint and isinstance(checkpoint['history'], dict):
            self.history = checkpoint['history']

        early_state = checkpoint.get('early_stopping')
        if isinstance(early_state, dict):
            self.early_stopping.best_score = early_state.get('best_score', self.early_stopping.best_score)
            self.early_stopping.counter = early_state.get('counter', self.early_stopping.counter)
            self.early_stopping.should_stop = early_state.get('should_stop', self.early_stopping.should_stop)
        elif self.best_val_f1: # Use best_val_f1 for early stopping initialization
            self.early_stopping.best_score = self.best_val_f1
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
            
            # Update scheduler (use SWA scheduler after swa_start_epoch)
            if self.use_swa and epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                current_lr = self.swa_scheduler.get_last_lr()[0]
            else:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(current_lr)
            
            # Check for best model (track F1 — more sensitive at 99%+ accuracy)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # SWA status indicator
            swa_status = " [SWA]" if (self.use_swa and epoch >= self.swa_start_epoch) else ""
            
            # Logging
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{self.epochs}{swa_status} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
            )
            
            # Print to console
            print(f"\nEpoch {epoch+1}/{self.epochs}{swa_status}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"  LR: {current_lr:.6f} | Best F1: {self.best_val_f1:.4f} (epoch {self.best_epoch+1})")
            
            # Early stopping (on F1) — disabled during SWA phase
            if not (self.use_swa and epoch >= self.swa_start_epoch):
                if self.early_stopping(val_f1):
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        # SWA: Update batch normalization statistics
        if self.use_swa and self.swa_model is not None:
            logger.info("Updating SWA batch normalization statistics...")
            print("\nUpdating SWA model batch normalization...")
            torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, device=self.device)
            
            # Save SWA model
            swa_path = self.models_dir / 'best_model_swa.pth'
            torch.save(self.swa_model.module.state_dict(), swa_path)
            logger.info(f"SWA model saved to {swa_path}")
            print(f"SWA model saved to {swa_path}")
        
        # Training complete
        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time/60:.1f} minutes")
        logger.info(f"Best validation F1: {self.best_val_f1:.4f} at epoch {self.best_epoch+1}")
        
        # Save history
        with open(self.outputs_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save final metrics
        final_metrics = {
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch + 1,
            'total_epochs': len(self.history['train_loss']),
            'training_time_minutes': total_time / 60,
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'final_val_f1': self.history['val_f1'][-1],
            'sam_enabled': self.use_sam,
            'swa_enabled': self.use_swa,
            'swa_start_epoch': self.swa_start_epoch + 1 if self.use_swa else None
        }
        
        with open(self.outputs_dir / 'final_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        return self.history
