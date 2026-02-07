"""
Training Dialog for Traffic Sign Recognition GUI.
Allows configuring and monitoring model training.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import yaml
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
    QProgressBar, QTextEdit, QFileDialog, QCheckBox, QTabWidget,
    QWidget
)
from PySide6.QtCore import Qt, Signal, QThread

from .styles import COLORS


class TrainingWorker(QThread):
    """
    Background worker for model training.
    Runs training in separate thread to keep GUI responsive.
    """
    
    progress = Signal(int, float, float)  # epoch, loss, accuracy
    log = Signal(str)
    finished = Signal(dict)  # final metrics
    error = Signal(str)
    
    def __init__(self, config_path: str, epochs: int = None, parent=None):
        super().__init__(parent)
        self.config_path = config_path
        self.epochs_override = epochs
        self._stop_requested = False
    
    def run(self):
        """Execute training in background."""
        try:
            import sys
            from pathlib import Path
            
            # Add project root to path
            project_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(project_root))
            
            import yaml
            import torch
            
            from src.data import get_dataloaders
            from src.models.hybrid_model import create_model
            from src.training import Trainer
            from src.training.losses import create_loss_function
            
            # Load config
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if self.epochs_override:
                config['training']['epochs'] = self.epochs_override
            
            self.log.emit(f"Loaded config: {self.config_path}")
            self.log.emit(f"Training for {config['training']['epochs']} epochs")
            
            # Device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.log.emit(f"Using device: {device}")
            
            # Data
            self.log.emit("Loading data...")
            train_loader, val_loader, _ = get_dataloaders(
                data_dir=config['dataset']['path'],
                batch_size=config['training']['batch_size'],
                val_split=config['dataset']['val_split'],
                image_size=config['dataset']['image_size'],
                num_workers=2,
                use_weighted_sampler=True
            )
            self.log.emit(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
            
            # Model
            self.log.emit("Creating model...")
            model = create_model(config)
            
            # Loss
            class_weights = train_loader.dataset.get_class_weights()
            criterion = create_loss_function(config, class_weights)
            
            # Trainer with callback
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                config=config,
                device=device
            )
            
            # Training loop with progress updates
            epochs = config['training']['epochs']
            for epoch in range(epochs):
                if self._stop_requested:
                    self.log.emit("Training stopped by user.")
                    break
                
                train_loss, train_acc = trainer.train_epoch()
                val_loss, val_acc, val_f1 = trainer.validate()
                
                self.progress.emit(epoch + 1, val_loss, val_acc)
                self.log.emit(f"Epoch {epoch+1}/{epochs} - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                
                trainer.scheduler_step(val_loss)
                trainer.save_checkpoint(epoch, val_loss, val_acc)
            
            final_metrics = {
                'epochs_completed': epoch + 1,
                'final_loss': val_loss,
                'final_accuracy': val_acc,
                'final_f1': val_f1
            }
            self.finished.emit(final_metrics)
            
        except Exception as e:
            import traceback
            self.error.emit(f"Training error: {str(e)}\n{traceback.format_exc()}")
    
    def stop(self):
        """Request training stop."""
        self._stop_requested = True


class TrainingDialog(QDialog):
    """
    Dialog for configuring and monitoring model training.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎓 Model Training")
        self.setMinimumSize(700, 600)
        self.worker: Optional[TrainingWorker] = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Tabs
        tabs = QTabWidget()
        
        # Configuration Tab
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        
        # Config file selection
        file_group = QGroupBox("Configuration File")
        file_layout = QHBoxLayout(file_group)
        
        self.config_path_label = QLabel("config_sri_lanka.yaml")
        self.config_path_label.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold;")
        
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(100)
        browse_btn.clicked.connect(self.browse_config)
        
        file_layout.addWidget(self.config_path_label, stretch=1)
        file_layout.addWidget(browse_btn)
        config_layout.addWidget(file_group)
        
        # Hyperparameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout(params_group)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 200)
        self.epochs_spin.setValue(50)
        
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(4, 128)
        self.batch_spin.setValue(32)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setValue(0.0001)
        self.lr_spin.setSingleStep(0.00001)
        
        self.backbone_combo = QComboBox()
        self.backbone_combo.addItems(['convnext_tiny', 'efficientnet_b0', 'resnet34'])
        
        self.amp_check = QCheckBox("Mixed Precision (AMP)")
        self.amp_check.setChecked(True)
        
        params_layout.addRow("Epochs:", self.epochs_spin)
        params_layout.addRow("Batch Size:", self.batch_spin)
        params_layout.addRow("Learning Rate:", self.lr_spin)
        params_layout.addRow("Backbone:", self.backbone_combo)
        params_layout.addRow("", self.amp_check)
        
        config_layout.addWidget(params_group)
        config_layout.addStretch()
        
        tabs.addTab(config_tab, "⚙️ Configuration")
        
        # Progress Tab
        progress_tab = QWidget()
        progress_layout = QVBoxLayout(progress_tab)
        
        # Progress bar
        progress_group = QGroupBox("Training Progress")
        pg_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.status_label = QLabel("Ready to train")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        metrics_layout = QHBoxLayout()
        self.loss_label = QLabel("Loss: ---")
        self.acc_label = QLabel("Accuracy: ---")
        metrics_layout.addWidget(self.loss_label)
        metrics_layout.addWidget(self.acc_label)
        
        pg_layout.addWidget(self.progress_bar)
        pg_layout.addWidget(self.status_label)
        pg_layout.addLayout(metrics_layout)
        
        progress_layout.addWidget(progress_group)
        
        # Log output
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                padding: 10px;
            }}
        """)
        log_layout.addWidget(self.log_text)
        
        progress_layout.addWidget(log_group, stretch=1)
        
        tabs.addTab(progress_tab, "📈 Progress")
        
        layout.addWidget(tabs)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶️ Start Training")
        self.start_btn.setObjectName("successBtn")
        self.start_btn.clicked.connect(self.start_training)
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.setStyleSheet(f"background-color: {COLORS['error']};")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
        self._config_path = "config_sri_lanka.yaml"
    
    def browse_config(self):
        """Open file dialog to select config file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Configuration File",
            "",
            "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self._config_path = file_path
            self.config_path_label.setText(Path(file_path).name)
            self.load_config(file_path)
    
    def load_config(self, path: str):
        """Load config values into UI."""
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.epochs_spin.setValue(config.get('training', {}).get('epochs', 50))
            self.batch_spin.setValue(config.get('training', {}).get('batch_size', 32))
            self.lr_spin.setValue(config.get('training', {}).get('learning_rate', 0.0001))
            
            backbone = config.get('model', {}).get('backbone', 'convnext_tiny')
            idx = self.backbone_combo.findText(backbone)
            if idx >= 0:
                self.backbone_combo.setCurrentIndex(idx)
            
            self.amp_check.setChecked(config.get('training', {}).get('use_amp', True))
            
            self.log_text.append(f"Loaded config: {path}")
        except Exception as e:
            self.log_text.append(f"Error loading config: {e}")
    
    def start_training(self):
        """Start training in background thread."""
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting training...")
        self.log_text.clear()
        
        epochs = self.epochs_spin.value()
        
        self.worker = TrainingWorker(self._config_path, epochs)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.on_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
    
    def stop_training(self):
        """Stop training."""
        if self.worker:
            self.worker.stop()
            self.status_label.setText("Stopping...")
            self.stop_btn.setEnabled(False)
    
    def on_progress(self, epoch: int, loss: float, accuracy: float):
        """Handle training progress update."""
        total_epochs = self.epochs_spin.value()
        percent = int((epoch / total_epochs) * 100)
        self.progress_bar.setValue(percent)
        self.status_label.setText(f"Epoch {epoch}/{total_epochs}")
        self.loss_label.setText(f"Loss: {loss:.4f}")
        self.acc_label.setText(f"Accuracy: {accuracy:.2%}")
    
    def on_log(self, message: str):
        """Append message to log."""
        self.log_text.append(message)
    
    def on_finished(self, metrics: dict):
        """Handle training completion."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("✅ Training Complete!")
        
        self.log_text.append("\n" + "="*50)
        self.log_text.append("TRAINING COMPLETE")
        self.log_text.append(f"Final Loss: {metrics.get('final_loss', 0):.4f}")
        self.log_text.append(f"Final Accuracy: {metrics.get('final_accuracy', 0):.2%}")
        self.log_text.append(f"Final F1: {metrics.get('final_f1', 0):.4f}")
    
    def on_error(self, error_msg: str):
        """Handle training error."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("❌ Error occurred")
        self.log_text.append(f"\nERROR: {error_msg}")
    
    def closeEvent(self, event):
        """Handle dialog close - stop training if running."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()
