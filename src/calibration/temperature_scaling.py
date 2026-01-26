"""
Temperature Scaling Calibration for Traffic Sign Recognition.
Implements post-hoc calibration for reliable confidence scores.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for model calibration.
    
    A simple post-hoc calibration method that scales logits by a learned temperature.
    
    Paper: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature
    
    def fit(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 0.01,
        max_iter: int = 50
    ) -> float:
        """
        Fit temperature on validation set.
        
        Args:
            model: Trained model
            val_loader: Validation dataloader
            device: Device to use
            lr: Learning rate for temperature optimization
            max_iter: Maximum optimization iterations
            
        Returns:
            Final temperature value
        """
        model.eval()
        self.to(device)
        
        # Collect all logits and labels
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels)
        
        logits = torch.cat(all_logits, dim=0).to(device)
        labels = torch.cat(all_labels, dim=0).to(device)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return self.temperature.item()


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probs: Predicted probabilities (N, num_classes)
        labels: Ground truth labels (N,)
        n_bins: Number of bins
        
    Returns:
        ece: Expected Calibration Error
        bin_accs: Accuracy in each bin
        bin_confs: Confidence in each bin
        bin_counts: Number of samples in each bin
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_counts[i] = in_bin.sum()
        
        if bin_counts[i] > 0:
            bin_accs[i] = accuracies[in_bin].mean()
            bin_confs[i] = confidences[in_bin].mean()
    
    # Compute ECE
    ece = np.sum(np.abs(bin_accs - bin_confs) * bin_counts) / np.sum(bin_counts)
    
    return ece, bin_accs, bin_confs, bin_counts


def compute_mce(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    """
    _, bin_accs, bin_confs, bin_counts = compute_ece(probs, labels, n_bins)
    
    # Only consider bins with samples
    valid_bins = bin_counts > 0
    if not valid_bins.any():
        return 0.0
    
    return np.max(np.abs(bin_accs[valid_bins] - bin_confs[valid_bins]))


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot reliability diagram for calibration visualization.
    
    Args:
        probs: Predicted probabilities
        labels: Ground truth labels
        n_bins: Number of bins
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    ece, bin_accs, bin_confs, bin_counts = compute_ece(probs, labels, n_bins)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    ax1 = axes[0]
    bin_centers = np.linspace(1/(2*n_bins), 1 - 1/(2*n_bins), n_bins)
    
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Bars for actual accuracy
    bar_width = 1.0 / n_bins
    valid_bins = bin_counts > 0
    
    ax1.bar(
        bin_centers[valid_bins],
        bin_accs[valid_bins],
        width=bar_width,
        alpha=0.7,
        label='Accuracy',
        edgecolor='black'
    )
    
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Reliability Diagram (ECE = {ece:.4f})')
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Confidence histogram
    ax2 = axes[1]
    confidences = np.max(probs, axis=1)
    ax2.hist(confidences, bins=n_bins, range=(0, 1), alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def calibrate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path
) -> Tuple[TemperatureScaling, float, float]:
    """
    Calibrate model and compute calibration metrics.
    
    Args:
        model: Trained model
        val_loader: Validation dataloader
        device: Device
        output_dir: Directory to save outputs
        
    Returns:
        calibrator: Fitted TemperatureScaling module
        ece_before: ECE before calibration
        ece_after: ECE after calibration
    """
    model.eval()
    
    # Collect predictions
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    
    # Before calibration
    probs_before = F.softmax(logits, dim=1).numpy()
    ece_before, _, _, _ = compute_ece(probs_before, labels)
    
    # Fit temperature scaling
    calibrator = TemperatureScaling()
    temperature = calibrator.fit(model, val_loader, device)
    
    # After calibration
    scaled_logits = calibrator(logits.to(device))
    probs_after = F.softmax(scaled_logits, dim=1).cpu().numpy()
    ece_after, _, _, _ = compute_ece(probs_after, labels)
    
    # Plot reliability diagrams
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_reliability_diagram(probs_before, labels, save_path=output_dir / 'reliability_before.png')
    plot_reliability_diagram(probs_after, labels, save_path=output_dir / 'reliability_after.png')
    
    # Save calibrator
    torch.save(calibrator.state_dict(), output_dir / 'calibrator.pth')
    
    print(f"\nCalibration Results:")
    print(f"  Temperature: {temperature:.4f}")
    print(f"  ECE before: {ece_before:.4f}")
    print(f"  ECE after:  {ece_after:.4f}")
    
    return calibrator, ece_before, ece_after
