"""
Results Panel Widget for Traffic Sign Recognition GUI.
Displays prediction results with confidence bars and top-k predictions.
"""

from typing import Dict, List, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QProgressBar, QScrollArea, QSizePolicy, QGroupBox
)
from PySide6.QtCore import Qt, Signal

from .styles import COLORS, get_confidence_color


class ConfidenceBar(QFrame):
    """
    Custom confidence bar with gradient based on confidence level.
    """
    
    def __init__(self, class_name: str, confidence: float, rank: int = 0, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(15)
        
        # Rank badge
        rank_label = QLabel(f"#{rank + 1}")
        rank_label.setFixedWidth(35)
        rank_label.setStyleSheet(f"""
            background-color: {COLORS['primary']};
            border-radius: 5px;
            padding: 3px;
            font-weight: bold;
            font-size: 12px;
        """)
        rank_label.setAlignment(Qt.AlignCenter)
        
        # Class name
        name_label = QLabel(class_name)
        name_label.setFixedWidth(180)
        name_label.setStyleSheet("font-weight: bold; font-size: 13px; background: transparent;")
        name_label.setWordWrap(True)
        
        # Progress bar
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(confidence * 100))
        bar.setTextVisible(False)
        bar.setFixedHeight(18)
        
        color = get_confidence_color(confidence)
        bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['surface_light']};
                border: none;
                border-radius: 9px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 9px;
            }}
        """)
        
        # Confidence percentage
        conf_label = QLabel(f"{confidence:.1%}")
        conf_label.setFixedWidth(55)
        conf_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 14px; background: transparent;")
        conf_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        layout.addWidget(rank_label)
        layout.addWidget(name_label)
        layout.addWidget(bar, stretch=1)
        layout.addWidget(conf_label)
        
        # Frame styling
        self.setStyleSheet(f"""
            ConfidenceBar {{
                background-color: {COLORS['surface']};
                border-radius: 8px;
                margin: 2px 0;
            }}
        """)


class PredictionCard(QFrame):
    """
    Main prediction result card showing the top prediction.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Prediction header
        header = QLabel("🎯 Prediction")
        header.setStyleSheet("font-size: 16px; font-weight: bold; background: transparent;")
        
        # Main prediction
        self.class_label = QLabel("---")
        self.class_label.setStyleSheet(f"""
            font-size: 28px;
            font-weight: bold;
            color: {COLORS['accent']};
            background: transparent;
        """)
        self.class_label.setWordWrap(True)
        self.class_label.setAlignment(Qt.AlignCenter)
        
        # Confidence
        self.confidence_label = QLabel("Confidence: ---%")
        self.confidence_label.setStyleSheet("font-size: 18px; background: transparent;")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        
        # Status indicator
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; background: transparent;")
        
        layout.addWidget(header)
        layout.addWidget(self.class_label)
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.status_label)
        
        self.setStyleSheet(f"""
            PredictionCard {{
                background-color: {COLORS['surface']};
                border: 2px solid {COLORS['border']};
                border-radius: 15px;
            }}
        """)
    
    def update_prediction(self, class_name: str, confidence: float, is_confident: bool):
        """Update the prediction display."""
        self.class_label.setText(class_name)
        
        color = get_confidence_color(confidence)
        self.confidence_label.setText(f"Confidence: {confidence:.1%}")
        self.confidence_label.setStyleSheet(f"font-size: 18px; color: {color}; background: transparent;")
        
        if is_confident:
            self.status_label.setText("✅ High Confidence")
            self.status_label.setStyleSheet(f"font-size: 14px; color: {COLORS['success']}; background: transparent;")
        else:
            self.status_label.setText("⚠️ Low Confidence")
            self.status_label.setStyleSheet(f"font-size: 14px; color: {COLORS['warning']}; background: transparent;")
    
    def clear(self):
        """Clear prediction display."""
        self.class_label.setText("---")
        self.confidence_label.setText("Confidence: ---%")
        self.confidence_label.setStyleSheet("font-size: 18px; background: transparent;")
        self.status_label.setText("")


class ResultsPanel(QWidget):
    """
    Panel displaying prediction results with top-k confidence bars.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("📊 Results")
        title.setStyleSheet("font-size: 20px; font-weight: bold; background: transparent;")
        layout.addWidget(title)
        
        # Main prediction card
        self.prediction_card = PredictionCard()
        layout.addWidget(self.prediction_card)
        
        # Top-K predictions section
        topk_group = QGroupBox("Top 5 Predictions")
        topk_layout = QVBoxLayout(topk_group)
        
        # Scroll area for confidence bars
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        
        self.bars_container = QWidget()
        self.bars_layout = QVBoxLayout(self.bars_container)
        self.bars_layout.setSpacing(8)
        self.bars_layout.setAlignment(Qt.AlignTop)
        
        scroll.setWidget(self.bars_container)
        topk_layout.addWidget(scroll)
        
        layout.addWidget(topk_group, stretch=1)
        
        # Uncertainty section (optional)
        self.uncertainty_group = QGroupBox("Uncertainty Analysis")
        uncertainty_layout = QVBoxLayout(self.uncertainty_group)
        
        self.uncertainty_label = QLabel("MC Dropout: Not enabled")
        self.uncertainty_label.setStyleSheet("color: #a0a0a0; background: transparent;")
        uncertainty_layout.addWidget(self.uncertainty_label)
        
        self.epistemic_label = QLabel("")
        self.epistemic_label.setStyleSheet("background: transparent;")
        uncertainty_layout.addWidget(self.epistemic_label)
        
        self.aleatoric_label = QLabel("")
        self.aleatoric_label.setStyleSheet("background: transparent;")
        uncertainty_layout.addWidget(self.aleatoric_label)
        
        layout.addWidget(self.uncertainty_group)
        self.uncertainty_group.hide()  # Hidden by default
        
        self.confidence_bars: List[ConfidenceBar] = []
    
    def update_results(self, result: Dict):
        """
        Update all results from prediction dictionary.
        
        Expected result format:
        {
            'class_name': str,
            'class_id': int,
            'confidence': float,
            'is_confident': bool,
            'top_k': [{'class_name': str, 'confidence': float}, ...],
            'uncertainty': float (optional)
        }
        """
        # Update main prediction
        self.prediction_card.update_prediction(
            result.get('class_name', 'Unknown'),
            result.get('confidence', 0.0),
            result.get('is_confident', False)
        )
        
        # Clear existing bars
        self.clear_bars()
        
        # Add top-k bars
        top_k = result.get('top_k', [])
        for i, pred in enumerate(top_k[:5]):
            bar = ConfidenceBar(
                pred.get('class_name', f'Class {pred.get("class_id", i)}'),
                pred.get('confidence', 0.0),
                rank=i
            )
            self.bars_layout.addWidget(bar)
            self.confidence_bars.append(bar)
        
        # Update uncertainty if present
        uncertainty = result.get('uncertainty')
        if uncertainty is not None:
            self.uncertainty_group.show()
            self.uncertainty_label.setText(f"Total Uncertainty: {uncertainty:.4f}")
            
            epistemic = result.get('epistemic_uncertainty')
            if epistemic is not None:
                self.epistemic_label.setText(f"Epistemic (Model): {epistemic:.4f}")
            
            aleatoric = result.get('aleatoric_uncertainty')
            if aleatoric is not None:
                self.aleatoric_label.setText(f"Aleatoric (Data): {aleatoric:.4f}")
        else:
            self.uncertainty_group.hide()
    
    def clear_bars(self):
        """Remove all confidence bars."""
        for bar in self.confidence_bars:
            self.bars_layout.removeWidget(bar)
            bar.deleteLater()
        self.confidence_bars.clear()
    
    def clear(self):
        """Clear all results."""
        self.prediction_card.clear()
        self.clear_bars()
        self.uncertainty_group.hide()
