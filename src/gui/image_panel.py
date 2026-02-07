"""
Image Panel Widget for Traffic Sign Recognition GUI.
Handles image display, drag-drop upload, and webcam capture.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable
from PIL import Image

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer, QSize
from PySide6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent


class DropZone(QFrame):
    """
    Drag-and-drop zone for image upload.
    Displays a placeholder when empty, image preview when loaded.
    """
    
    image_dropped = Signal(str)  # Emits file path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Styling
        self.setStyleSheet("""
            DropZone {
                background-color: #1f2b47;
                border: 3px dashed #2d3a5a;
                border-radius: 15px;
            }
            DropZone:hover {
                border-color: #e94560;
                background-color: #16213e;
            }
        """)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        # Placeholder content
        self.icon_label = QLabel("🖼️")
        self.icon_label.setStyleSheet("font-size: 64px; background: transparent;")
        self.icon_label.setAlignment(Qt.AlignCenter)
        
        self.text_label = QLabel("Drag & Drop Image Here\nor click Browse")
        self.text_label.setStyleSheet("""
            font-size: 16px;
            color: #a0a0a0;
            background: transparent;
        """)
        self.text_label.setAlignment(Qt.AlignCenter)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.hide()
        
        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)
        layout.addWidget(self.image_label)
        
        self._current_image_path: Optional[str] = None
        self._current_pil_image: Optional[Image.Image] = None
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter - accept if contains image file."""
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                event.acceptProposedAction()
                self.setStyleSheet("""
                    DropZone {
                        background-color: #16213e;
                        border: 3px solid #e94560;
                        border-radius: 15px;
                    }
                """)
    
    def dragLeaveEvent(self, event):
        """Reset style on drag leave."""
        self.setStyleSheet("""
            DropZone {
                background-color: #1f2b47;
                border: 3px dashed #2d3a5a;
                border-radius: 15px;
            }
            DropZone:hover {
                border-color: #e94560;
                background-color: #16213e;
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop - load and display image."""
        url = event.mimeData().urls()[0]
        file_path = url.toLocalFile()
        self.load_image(file_path)
        self.image_dropped.emit(file_path)
        
        # Reset style
        self.dragLeaveEvent(None)
    
    def load_image(self, file_path: str):
        """Load and display image from file path."""
        self._current_image_path = file_path
        self._current_pil_image = Image.open(file_path).convert('RGB')
        
        # Convert to QPixmap for display
        pixmap = QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(
            self.size() - QSize(40, 40),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # Hide placeholder, show image
        self.icon_label.hide()
        self.text_label.hide()
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.show()
    
    def load_frame(self, frame: np.ndarray):
        """Load and display frame from camera (BGR format)."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._current_pil_image = Image.fromarray(rgb_frame)
        
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        scaled_pixmap = pixmap.scaled(
            self.size() - QSize(40, 40),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.icon_label.hide()
        self.text_label.hide()
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.show()
    
    def get_current_image(self) -> Optional[Image.Image]:
        """Return current PIL image for prediction."""
        return self._current_pil_image
    
    def clear(self):
        """Clear current image and show placeholder."""
        self._current_image_path = None
        self._current_pil_image = None
        self.image_label.hide()
        self.icon_label.show()
        self.text_label.show()
    
    def resizeEvent(self, event):
        """Handle resize - rescale image if present."""
        super().resizeEvent(event)
        if self._current_image_path and self.image_label.isVisible():
            pixmap = QPixmap(self._current_image_path)
            scaled_pixmap = pixmap.scaled(
                self.size() - QSize(40, 40),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)


class ImagePanel(QWidget):
    """
    Main image panel with upload controls and camera integration.
    """
    
    image_loaded = Signal(object)  # Emits PIL Image
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_camera()
    
    def setup_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("📷 Input Image")
        title.setObjectName("title")
        title.setStyleSheet("font-size: 20px; font-weight: bold; background: transparent;")
        layout.addWidget(title)
        
        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.image_dropped.connect(self._on_image_dropped)
        layout.addWidget(self.drop_zone)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        self.browse_btn = QPushButton("📁 Browse")
        self.browse_btn.clicked.connect(self.browse_file)
        
        self.predict_btn = QPushButton("🔍 Predict")
        self.predict_btn.setStyleSheet("background-color: #e94560;")
        self.predict_btn.clicked.connect(self.trigger_prediction)
        
        self.camera_btn = QPushButton("📹 Start Camera")
        self.camera_btn.clicked.connect(self.toggle_camera)
        
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.clicked.connect(self.clear_image)
        
        btn_layout.addWidget(self.browse_btn)
        btn_layout.addWidget(self.predict_btn)
        btn_layout.addWidget(self.camera_btn)
        btn_layout.addWidget(self.clear_btn)
        
        layout.addLayout(btn_layout)
    
    def trigger_prediction(self):
        """Manually trigger prediction on current image."""
        image = self.drop_zone.get_current_image()
        if image:
            self.image_loaded.emit(image)
    
    def setup_camera(self):
        """Initialize camera capture."""
        self.camera: Optional[cv2.VideoCapture] = None
        self.camera_active = False
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self._capture_frame)
    
    def browse_file(self):
        """Open file dialog to select image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Traffic Sign Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:
            self.drop_zone.load_image(file_path)
            self._on_image_dropped(file_path)
    
    def _on_image_dropped(self, file_path: str):
        """Handle image loaded event."""
        image = self.drop_zone.get_current_image()
        if image:
            self.image_loaded.emit(image)
    
    def toggle_camera(self):
        """Start or stop camera capture."""
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Start webcam capture."""
        self.camera = cv2.VideoCapture(0)
        if self.camera.isOpened():
            self.camera_active = True
            self.camera_btn.setText("⏹️ Stop Camera")
            self.camera_btn.setStyleSheet("background-color: #ff4757;")
            self.camera_timer.start(33)  # ~30 FPS
        else:
            self.camera = None
    
    def stop_camera(self):
        """Stop webcam capture."""
        self.camera_timer.stop()
        if self.camera:
            self.camera.release()
            self.camera = None
        self.camera_active = False
        self.camera_btn.setText("📹 Start Camera")
        self.camera_btn.setStyleSheet("")
    
    def _capture_frame(self):
        """Capture and display frame from camera."""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.drop_zone.load_frame(frame)
                image = self.drop_zone.get_current_image()
                if image:
                    self.image_loaded.emit(image)
    
    def clear_image(self):
        """Clear current image."""
        self.stop_camera()
        self.drop_zone.clear()
    
    def get_current_image(self) -> Optional[Image.Image]:
        """Get current image for prediction."""
        return self.drop_zone.get_current_image()
