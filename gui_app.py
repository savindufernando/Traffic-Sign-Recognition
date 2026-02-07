#!/usr/bin/env python
"""
Traffic Sign Recognition GUI Application.

A desktop application for recognizing traffic signs using 
a Hybrid CNN-Transformer model trained on Sri Lankan traffic signs.

Features:
- Drag-and-drop image upload
- Live webcam capture with real-time prediction
- Top-5 predictions with confidence visualization
- Model training interface
- Uncertainty estimation (MC Dropout)

Usage:
    python gui_app.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from src.gui import MainWindow


def main():
    """Launch the Traffic Sign Recognition GUI application."""
    # High DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("Traffic Sign Recognition")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("FYP")
    
    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
