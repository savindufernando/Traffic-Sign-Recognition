"""
Main Window for Traffic Sign Recognition GUI.
Central hub with image panel, results panel, and menu actions.
"""

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QMenuBar, QMenu, QStatusBar, QMessageBox, QFileDialog,
    QSplitter, QFrame
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence

from .styles import DARK_THEME, COLORS
from .image_panel import ImagePanel
from .results_panel import ResultsPanel
from .training_dialog import TrainingDialog
from .settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    """
    Main application window for Traffic Sign Recognition.
    """
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("🚦 Traffic Sign Recognition")
        self.setMinimumSize(1200, 800)
        
        # Apply dark theme
        self.setStyleSheet(DARK_THEME)
        
        # Predictor (loaded on demand)
        self.predictor = None
        self.settings = {
            'model_path': 'models_sri_lanka/best_model.pth',
            'config_path': 'config_sri_lanka.yaml',
            'confidence_threshold': 0.5,
            'temperature': 1.0,
            'mc_enabled': False,
            'mc_samples': 30
        }
        
        self.setup_ui()
        self.setup_menu()
        self.setup_statusbar()
        
        # Load model on startup
        QTimer.singleShot(100, self.load_model)
    
    def setup_ui(self):
        """Initialize the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Header
        header = QLabel("🚦 Sri Lankan Traffic Sign Recognition")
        header.setStyleSheet(f"""
            font-size: 28px;
            font-weight: bold;
            color: {COLORS['text']};
            padding: 10px;
            background: transparent;
        """)
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)
        
        # Subtitle
        subtitle = QLabel("Upload an image or use your camera to identify traffic signs")
        subtitle.setStyleSheet(f"font-size: 14px; color: {COLORS['text_secondary']}; background: transparent;")
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)
        
        # Content splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        
        # Left panel - Image
        left_frame = QFrame()
        left_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border-radius: 15px;
                padding: 15px;
            }}
        """)
        left_layout = QVBoxLayout(left_frame)
        
        self.image_panel = ImagePanel()
        self.image_panel.image_loaded.connect(self.on_image_loaded)
        left_layout.addWidget(self.image_panel)
        
        # Right panel - Results
        right_frame = QFrame()
        right_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border-radius: 15px;
                padding: 15px;
            }}
        """)
        right_layout = QVBoxLayout(right_frame)
        
        self.results_panel = ResultsPanel()
        right_layout.addWidget(self.results_panel)
        
        splitter.addWidget(left_frame)
        splitter.addWidget(right_frame)
        splitter.setSizes([600, 500])
        
        main_layout.addWidget(splitter, stretch=1)
    
    def setup_menu(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("📂 Open Image...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.image_panel.browse_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("💾 Export Results...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("🚪 Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Model menu
        model_menu = menubar.addMenu("&Model")
        
        train_action = QAction("🎓 Train Model...", self)
        train_action.setShortcut("Ctrl+T")
        train_action.triggered.connect(self.show_training_dialog)
        model_menu.addAction(train_action)
        
        model_menu.addSeparator()
        
        reload_action = QAction("🔄 Reload Model", self)
        reload_action.setShortcut("Ctrl+R")
        reload_action.triggered.connect(self.load_model)
        model_menu.addAction(reload_action)
        
        # Settings menu
        settings_menu = menubar.addMenu("&Settings")
        
        settings_action = QAction("⚙️ Preferences...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.show_settings_dialog)
        settings_menu.addAction(settings_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("ℹ️ About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_statusbar(self):
        """Create status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.model_status = QLabel("Model: Not loaded")
        self.model_status.setStyleSheet("padding: 5px;")
        self.status_bar.addPermanentWidget(self.model_status)
        
        self.fps_label = QLabel("")
        self.fps_label.setStyleSheet("padding: 5px;")
        self.status_bar.addPermanentWidget(self.fps_label)
    
    def load_model(self):
        """Load the prediction model."""
        try:
            self.status_bar.showMessage("Loading model...", 2000)
            
            # Add project root to path
            project_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(project_root))
            
            from src.inference.predictor import load_predictor
            
            model_path = self.settings['model_path']
            config_path = self.settings['config_path']
            
            # Check if paths exist
            if not Path(model_path).exists():
                # Try relative to project root
                model_path = project_root / model_path
                config_path = project_root / config_path
            
            self.predictor = load_predictor(
                str(model_path),
                str(config_path)
            )
            
            # Apply settings
            self.predictor.set_temperature(self.settings['temperature'])
            self.predictor.set_confidence_threshold(self.settings['confidence_threshold'])
            
            self.model_status.setText(f"✅ Model: {Path(model_path).name}")
            self.model_status.setStyleSheet(f"color: {COLORS['success']}; padding: 5px;")
            self.status_bar.showMessage("Model loaded successfully!", 3000)
            
        except Exception as e:
            self.model_status.setText("❌ Model: Failed to load")
            self.model_status.setStyleSheet(f"color: {COLORS['error']}; padding: 5px;")
            QMessageBox.warning(
                self,
                "Model Load Error",
                f"Failed to load model:\n{str(e)}\n\nPlease check settings."
            )
    
    def on_image_loaded(self, image):
        """Handle image loaded - run prediction."""
        if self.predictor is None:
            self.status_bar.showMessage("Model not loaded. Please load model first.", 3000)
            return
        
        try:
            # Get prediction with top-k
            result = self.predictor.predict(image, return_all_probs=False)
            
            # Get top-k predictions
            top_k = self.predictor.get_top_k_predictions(image, k=5)
            result['top_k'] = top_k
            
            # Optionally add uncertainty
            if self.settings.get('mc_enabled', False):
                uncertainty_result = self.predictor.predict_with_uncertainty(
                    image, 
                    n_samples=self.settings.get('mc_samples', 30)
                )
                result['uncertainty'] = uncertainty_result.get('total_uncertainty')
                result['epistemic_uncertainty'] = uncertainty_result.get('epistemic_uncertainty')
                result['aleatoric_uncertainty'] = uncertainty_result.get('aleatoric_uncertainty')
            
            # Update results panel
            self.results_panel.update_results(result)
            
            # Update status
            self.status_bar.showMessage(
                f"Predicted: {result['class_name']} ({result['confidence']:.1%})", 
                5000
            )
            
        except Exception as e:
            self.status_bar.showMessage(f"Prediction error: {str(e)}", 5000)
    
    def show_training_dialog(self):
        """Open training dialog."""
        dialog = TrainingDialog(self)
        dialog.exec()
    
    def show_settings_dialog(self):
        """Open settings dialog."""
        dialog = SettingsDialog(self.settings, self)
        dialog.settings_changed.connect(self.apply_settings)
        dialog.exec()
    
    def apply_settings(self, new_settings: dict):
        """Apply new settings."""
        self.settings = new_settings
        
        # Reload model if path changed
        if self.predictor:
            self.predictor.set_temperature(new_settings['temperature'])
            self.predictor.set_confidence_threshold(new_settings['confidence_threshold'])
        
        self.status_bar.showMessage("Settings applied", 2000)
    
    def export_results(self):
        """Export prediction results to file."""
        # Get current results from panel
        # For now, just show save dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "prediction_results.json",
            "JSON Files (*.json);;CSV Files (*.csv)"
        )
        
        if file_path:
            self.status_bar.showMessage(f"Results exported to {file_path}", 3000)
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Traffic Sign Recognition",
            """
            <h2>🚦 Traffic Sign Recognition</h2>
            <p><b>Version:</b> 1.0.0</p>
            <p><b>Dataset:</b> Sri Lankan Traffic Signs (122 classes)</p>
            <p><b>Model:</b> Hybrid CNN-Transformer</p>
            <hr>
            <p>Desktop application for recognizing traffic signs using 
            deep learning with uncertainty estimation.</p>
            <p>Built with PySide6 and PyTorch.</p>
            """
        )
    
    def closeEvent(self, event):
        """Handle window close."""
        # Stop camera if running
        self.image_panel.stop_camera()
        event.accept()
