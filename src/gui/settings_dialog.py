"""
Settings Dialog for Traffic Sign Recognition GUI.
Allows configuring model path, confidence threshold, and other preferences.
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QLineEdit, QFileDialog, QDoubleSpinBox,
    QSpinBox, QCheckBox
)
from PySide6.QtCore import Signal

from .styles import COLORS


class SettingsDialog(QDialog):
    """
    Settings dialog for application preferences.
    """
    
    settings_changed = Signal(dict)
    
    def __init__(self, current_settings: dict = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Settings")
        self.setMinimumWidth(500)
        
        self.settings = current_settings or {}
        self.setup_ui()
        self.load_settings()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Model Settings
        model_group = QGroupBox("Model Configuration")
        model_layout = QFormLayout(model_group)
        
        # Model path
        model_path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to best_model.pth")
        
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self.browse_model)
        
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(browse_btn)
        
        # Config path
        config_path_layout = QHBoxLayout()
        self.config_path_edit = QLineEdit()
        self.config_path_edit.setPlaceholderText("Path to config.yaml")
        
        config_browse_btn = QPushButton("Browse")
        config_browse_btn.setFixedWidth(80)
        config_browse_btn.clicked.connect(self.browse_config)
        
        config_path_layout.addWidget(self.config_path_edit)
        config_path_layout.addWidget(config_browse_btn)
        
        model_layout.addRow("Model Path:", model_path_layout)
        model_layout.addRow("Config Path:", config_path_layout)
        
        layout.addWidget(model_group)
        
        # Inference Settings
        inference_group = QGroupBox("Inference Settings")
        inference_layout = QFormLayout(inference_group)
        
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.5)
        
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.1, 5.0)
        self.temperature_spin.setDecimals(2)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(1.0)
        
        self.mc_samples_spin = QSpinBox()
        self.mc_samples_spin.setRange(1, 100)
        self.mc_samples_spin.setValue(30)
        
        self.mc_enabled_check = QCheckBox("Enable MC Dropout Uncertainty")
        
        inference_layout.addRow("Confidence Threshold:", self.confidence_spin)
        inference_layout.addRow("Temperature:", self.temperature_spin)
        inference_layout.addRow("MC Dropout Samples:", self.mc_samples_spin)
        inference_layout.addRow("", self.mc_enabled_check)
        
        layout.addWidget(inference_group)
        
        # Camera Settings
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QFormLayout(camera_group)
        
        self.camera_id_spin = QSpinBox()
        self.camera_id_spin.setRange(0, 10)
        self.camera_id_spin.setValue(0)
        
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(30)
        
        camera_layout.addRow("Camera ID:", self.camera_id_spin)
        camera_layout.addRow("Target FPS:", self.fps_spin)
        
        layout.addWidget(camera_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        save_btn = QPushButton("💾 Save")
        save_btn.setObjectName("successBtn")
        save_btn.clicked.connect(self.save_settings)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_defaults)
        
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        
        layout.addLayout(btn_layout)
    
    def browse_model(self):
        """Open file dialog to select model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "PyTorch Models (*.pth *.pt)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
    
    def browse_config(self):
        """Open file dialog to select config file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Config File",
            "",
            "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.config_path_edit.setText(file_path)
    
    def load_settings(self):
        """Load current settings into UI."""
        self.model_path_edit.setText(self.settings.get('model_path', ''))
        self.config_path_edit.setText(self.settings.get('config_path', ''))
        self.confidence_spin.setValue(self.settings.get('confidence_threshold', 0.5))
        self.temperature_spin.setValue(self.settings.get('temperature', 1.0))
        self.mc_samples_spin.setValue(self.settings.get('mc_samples', 30))
        self.mc_enabled_check.setChecked(self.settings.get('mc_enabled', False))
        self.camera_id_spin.setValue(self.settings.get('camera_id', 0))
        self.fps_spin.setValue(self.settings.get('target_fps', 30))
    
    def save_settings(self):
        """Save settings and emit signal."""
        self.settings = {
            'model_path': self.model_path_edit.text(),
            'config_path': self.config_path_edit.text(),
            'confidence_threshold': self.confidence_spin.value(),
            'temperature': self.temperature_spin.value(),
            'mc_samples': self.mc_samples_spin.value(),
            'mc_enabled': self.mc_enabled_check.isChecked(),
            'camera_id': self.camera_id_spin.value(),
            'target_fps': self.fps_spin.value()
        }
        self.settings_changed.emit(self.settings)
        self.accept()
    
    def reset_defaults(self):
        """Reset all settings to defaults."""
        self.model_path_edit.setText('models_sri_lanka/best_model.pth')
        self.config_path_edit.setText('config_sri_lanka.yaml')
        self.confidence_spin.setValue(0.5)
        self.temperature_spin.setValue(1.0)
        self.mc_samples_spin.setValue(30)
        self.mc_enabled_check.setChecked(False)
        self.camera_id_spin.setValue(0)
        self.fps_spin.setValue(30)
    
    def get_settings(self) -> dict:
        """Return current settings."""
        return self.settings
