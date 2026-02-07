"""
Modern Dark Theme Styles for Traffic Sign Recognition GUI.
Uses a sleek dark color palette with accent colors.
"""

# Color Palette
COLORS = {
    'background': '#1a1a2e',
    'surface': '#16213e',
    'surface_light': '#1f2b47',
    'primary': '#0f3460',
    'accent': '#e94560',
    'accent_hover': '#ff6b8a',
    'success': '#00d9a5',
    'warning': '#ffc107',
    'error': '#ff4757',
    'text': '#eaeaea',
    'text_secondary': '#a0a0a0',
    'border': '#2d3a5a',
}

# Main Application Stylesheet
DARK_THEME = f"""
QMainWindow {{
    background-color: {COLORS['background']};
}}

QWidget {{
    background-color: {COLORS['background']};
    color: {COLORS['text']};
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    font-size: 13px;
}}

/* Menu Bar */
QMenuBar {{
    background-color: {COLORS['surface']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 5px;
}}

QMenuBar::item {{
    padding: 8px 15px;
    border-radius: 4px;
}}

QMenuBar::item:selected {{
    background-color: {COLORS['primary']};
}}

QMenu {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 5px;
}}

QMenu::item {{
    padding: 10px 25px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: {COLORS['primary']};
}}

/* Buttons */
QPushButton {{
    background-color: {COLORS['primary']};
    color: {COLORS['text']};
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: bold;
    font-size: 14px;
}}

QPushButton:hover {{
    background-color: {COLORS['accent']};
}}

QPushButton:pressed {{
    background-color: {COLORS['accent_hover']};
}}

QPushButton:disabled {{
    background-color: {COLORS['surface_light']};
    color: {COLORS['text_secondary']};
}}

QPushButton#primaryBtn {{
    background-color: {COLORS['accent']};
}}

QPushButton#primaryBtn:hover {{
    background-color: {COLORS['accent_hover']};
}}

QPushButton#successBtn {{
    background-color: {COLORS['success']};
    color: {COLORS['background']};
}}

/* Labels */
QLabel {{
    color: {COLORS['text']};
    background-color: transparent;
}}

QLabel#title {{
    font-size: 24px;
    font-weight: bold;
    color: {COLORS['text']};
}}

QLabel#subtitle {{
    font-size: 14px;
    color: {COLORS['text_secondary']};
}}

/* Group Box */
QGroupBox {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    margin-top: 15px;
    padding: 15px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 15px;
    padding: 0 10px;
    color: {COLORS['accent']};
}}

/* Scroll Area */
QScrollArea {{
    border: none;
    background-color: transparent;
}}

QScrollBar:vertical {{
    background-color: {COLORS['surface']};
    width: 10px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['primary']};
    border-radius: 5px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['accent']};
}}

/* Progress Bar */
QProgressBar {{
    background-color: {COLORS['surface']};
    border: none;
    border-radius: 8px;
    height: 20px;
    text-align: center;
    color: {COLORS['text']};
}}

QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS['accent']}, stop:1 {COLORS['success']});
    border-radius: 8px;
}}

/* Line Edit */
QLineEdit {{
    background-color: {COLORS['surface']};
    border: 2px solid {COLORS['border']};
    border-radius: 8px;
    padding: 10px 15px;
    color: {COLORS['text']};
}}

QLineEdit:focus {{
    border-color: {COLORS['accent']};
}}

/* Combo Box */
QComboBox {{
    background-color: {COLORS['surface']};
    border: 2px solid {COLORS['border']};
    border-radius: 8px;
    padding: 10px 15px;
    color: {COLORS['text']};
}}

QComboBox:hover {{
    border-color: {COLORS['accent']};
}}

QComboBox::drop-down {{
    border: none;
    width: 30px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['primary']};
}}

/* Tab Widget */
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    background-color: {COLORS['surface']};
}}

QTabBar::tab {{
    background-color: {COLORS['surface_light']};
    border: none;
    padding: 12px 25px;
    margin-right: 3px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}}

QTabBar::tab:selected {{
    background-color: {COLORS['primary']};
}}

QTabBar::tab:hover {{
    background-color: {COLORS['accent']};
}}

/* Status Bar */
QStatusBar {{
    background-color: {COLORS['surface']};
    border-top: 1px solid {COLORS['border']};
    padding: 5px;
}}

/* Tool Tip */
QToolTip {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['accent']};
    border-radius: 5px;
    padding: 8px;
}}

/* Spin Box */
QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS['surface']};
    border: 2px solid {COLORS['border']};
    border-radius: 8px;
    padding: 8px;
    color: {COLORS['text']};
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COLORS['accent']};
}}

/* Slider */
QSlider::groove:horizontal {{
    background-color: {COLORS['surface_light']};
    height: 8px;
    border-radius: 4px;
}}

QSlider::handle:horizontal {{
    background-color: {COLORS['accent']};
    width: 20px;
    height: 20px;
    margin: -6px 0;
    border-radius: 10px;
}}

QSlider::handle:horizontal:hover {{
    background-color: {COLORS['accent_hover']};
}}

/* Check Box */
QCheckBox {{
    spacing: 10px;
}}

QCheckBox::indicator {{
    width: 20px;
    height: 20px;
    border-radius: 5px;
    border: 2px solid {COLORS['border']};
    background-color: {COLORS['surface']};
}}

QCheckBox::indicator:checked {{
    background-color: {COLORS['accent']};
    border-color: {COLORS['accent']};
}}
"""

# Confidence bar colors based on level
def get_confidence_color(confidence: float) -> str:
    """Return color based on confidence level."""
    if confidence >= 0.8:
        return COLORS['success']
    elif confidence >= 0.5:
        return COLORS['warning']
    else:
        return COLORS['error']
