"""
Training Curves Visualization
Generates training evidence plots from training_history.json
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load your REAL training history
history_path = Path(__file__).parent / 'outputs_sri_lanka' / 'training_history.json'

with open(history_path, 'r') as f:
    history = json.load(f)

epochs = range(1, len(history['train_loss']) + 1)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('TSR Model Training Progress', fontsize=14, fontweight='bold')

# 1. Loss Curve
axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Accuracy Curve
train_acc_pct = [x * 100 for x in history['train_acc']]
val_acc_pct = [x * 100 for x in history['val_acc']]
axes[1].plot(epochs, train_acc_pct, 'b-', label='Train Accuracy', linewidth=2)
axes[1].plot(epochs, val_acc_pct, 'g-', label='Val Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([95, 100])  # Focus on high accuracy range

# 3. F1 Score Curve
val_f1_pct = [x * 100 for x in history['val_f1']]
axes[2].plot(epochs, val_f1_pct, 'g-', label='Val F1 Score', linewidth=2, marker='o')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('F1 Score (%)')
axes[2].set_title('Validation F1 Score')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([95, 100])

# Print final metrics
print("=" * 50)
print("FINAL TRAINING METRICS (Real Values)")
print("=" * 50)
print(f"Final Train Accuracy: {history['train_acc'][-1]*100:.2f}%")
print(f"Final Val Accuracy:   {history['val_acc'][-1]*100:.2f}%")
print(f"Final Val F1 Score:   {history['val_f1'][-1]*100:.2f}%")
print(f"Final Train Loss:     {history['train_loss'][-1]:.4f}")
print(f"Final Val Loss:       {history['val_loss'][-1]:.4f}")
print("=" * 50)

plt.tight_layout()

# Save the figure
output_path = Path(__file__).parent / 'outputs_sri_lanka' / 'training_curves.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved to: {output_path}")

plt.show()
