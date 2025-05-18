"""
Styles for the Name Matching GUI.

This module provides styling for the Name Matching GUI.
"""

# Main application stylesheet
MAIN_STYLE = """
QMainWindow {
    background-color: #f5f5f5;
}

QTabWidget::pane {
    border: 1px solid #cccccc;
    background-color: #ffffff;
}

QTabBar::tab {
    background-color: #e0e0e0;
    border: 1px solid #cccccc;
    border-bottom-color: #cccccc;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    min-width: 8ex;
    padding: 8px;
}

QTabBar::tab:selected, QTabBar::tab:hover {
    background-color: #ffffff;
}

QTabBar::tab:selected {
    border-color: #cccccc;
    border-bottom-color: #ffffff;
}

QPushButton {
    background-color: #2196f3;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    min-width: 80px;
}

QPushButton:hover {
    background-color: #0d8aee;
}

QPushButton:pressed {
    background-color: #0c7cd5;
}

QPushButton:disabled {
    background-color: #cccccc;
    color: #666666;
}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    border: 1px solid #cccccc;
    border-radius: 4px;
    padding: 4px;
    background-color: white;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #2196f3;
}

QProgressBar {
    border: 1px solid #cccccc;
    border-radius: 4px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #2196f3;
    width: 10px;
}

QTableView {
    border: 1px solid #cccccc;
    gridline-color: #f0f0f0;
    selection-background-color: #e3f2fd;
    selection-color: #000000;
}

QHeaderView::section {
    background-color: #e0e0e0;
    border: 1px solid #cccccc;
    padding: 4px;
}

QStatusBar {
    background-color: #f5f5f5;
    color: #333333;
}

QLabel {
    color: #333333;
}

QGroupBox {
    border: 1px solid #cccccc;
    border-radius: 4px;
    margin-top: 1ex;
    padding-top: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 3px;
    background-color: #f5f5f5;
}
"""

# Dark theme stylesheet (used when dark mode is enabled)
DARK_STYLE = """
/* This will be populated with QDarkStyle */
"""

# Status indicators
CONNECTION_STATUS = {
    "connected": "background-color: #4caf50; border-radius: 8px; color: white; padding: 2px 8px;",
    "disconnected": "background-color: #f44336; border-radius: 8px; color: white; padding: 2px 8px;",
    "connecting": "background-color: #ff9800; border-radius: 8px; color: white; padding: 2px 8px;"
}

# Table styles
TABLE_MATCH_STYLES = {
    "match": "background-color: #e8f5e9;",
    "non_match": "background-color: #ffebee;",
    "manual_review": "background-color: #fff8e1;"
}

# Progress indicator styles
PROGRESS_STYLES = {
    "normal": "QProgressBar { border: 1px solid #cccccc; border-radius: 4px; text-align: center; } "
              "QProgressBar::chunk { background-color: #2196f3; }",
    "success": "QProgressBar { border: 1px solid #cccccc; border-radius: 4px; text-align: center; } "
               "QProgressBar::chunk { background-color: #4caf50; }",
    "error": "QProgressBar { border: 1px solid #cccccc; border-radius: 4px; text-align: center; } "
             "QProgressBar::chunk { background-color: #f44336; }"
}
