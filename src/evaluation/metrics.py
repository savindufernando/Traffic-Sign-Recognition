"""
Evaluation Metrics for Traffic Sign Recognition.
Computes comprehensive classification metrics and generates reports.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, top_k_accuracy_score
)
from tqdm import tqdm

from ..data.dataset import GTSRB_CLASSES


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities (optional, for top-k)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Top-k accuracy if probabilities provided
    if y_probs is not None:
        metrics['top3_accuracy'] = top_k_accuracy_score(y_true, y_probs, k=3)
        metrics['top5_accuracy'] = top_k_accuracy_score(y_true, y_probs, k=5)
    
    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class'] = {
        'precision': per_class_precision.tolist(),
        'recall': per_class_recall.tolist(),
        'f1': per_class_f1.tolist()
    }
    
    return metrics


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    output_path: Optional[Path] = None
) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Class names
        output_path: Path to save report
        
    Returns:
        Classification report string
    """
    if class_names is None:
        class_names = GTSRB_CLASSES
    
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (20, 16),
    normalize: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Class names
        save_path: Path to save figure
        figsize: Figure size
        normalize: Normalize confusion matrix
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = GTSRB_CLASSES
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=False,  # Too many classes for annotations
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        ax=ax,
        xticklabels=range(len(class_names)),
        yticklabels=range(len(class_names))
    )
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix (Normalized)' if normalize else 'Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_per_class_metrics(
    metrics: Dict,
    class_names: List[str] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot per-class precision, recall, F1.
    
    Args:
        metrics: Dictionary with per_class metrics
        class_names: Class names
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(metrics['per_class']['f1']))]
    
    per_class = metrics['per_class']
    n_classes = len(per_class['f1'])
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(n_classes)
    width = 0.25
    
    ax.bar(x - width, per_class['precision'], width, label='Precision', alpha=0.8)
    ax.bar(x, per_class['recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, per_class['f1'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        test_loader: Test dataloader
        device: Device
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary of evaluation results
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in tqdm(test_loader, desc='Evaluating'):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_probs)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate report
        report = generate_classification_report(
            y_true, y_pred,
            output_path=output_dir / 'classification_report.txt'
        )
        
        # Plot confusion matrix
        plot_confusion_matrix(
            y_true, y_pred,
            save_path=output_dir / 'confusion_matrix.png'
        )
        
        # Plot per-class metrics
        plot_per_class_metrics(
            metrics,
            save_path=output_dir / 'per_class_metrics.png'
        )
        
        print(f"\nEvaluation results saved to: {output_dir}")
    
    return metrics
