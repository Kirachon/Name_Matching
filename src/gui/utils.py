"""
Utility functions for the Name Matching GUI.

This module provides utility functions for the Name Matching GUI.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

from PyQt5.QtWidgets import (
    QMessageBox, QFileDialog, QApplication, QWidget, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QSettings

# Application settings
APP_NAME = "NameMatching"
ORGANIZATION = "NameMatchingOrg"
CONFIG_DIR = os.path.expanduser("~/.name_matching")
CONNECTION_PROFILES_FILE = os.path.join(CONFIG_DIR, "connection_profiles.json")
MATCHING_PRESETS_FILE = os.path.join(CONFIG_DIR, "matching_presets.json")


def ensure_config_dir() -> None:
    """Ensure the configuration directory exists."""
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)


def get_settings() -> QSettings:
    """Get the application settings."""
    return QSettings(ORGANIZATION, APP_NAME)


def save_connection_profiles(profiles: Dict[str, Dict[str, str]]) -> None:
    """
    Save connection profiles to file.
    
    Args:
        profiles: Dictionary of connection profiles
    """
    ensure_config_dir()
    with open(CONNECTION_PROFILES_FILE, 'w') as f:
        json.dump(profiles, f, indent=2)


def load_connection_profiles() -> Dict[str, Dict[str, str]]:
    """
    Load connection profiles from file.
    
    Returns:
        Dictionary of connection profiles
    """
    ensure_config_dir()
    if not os.path.exists(CONNECTION_PROFILES_FILE):
        return {}
    
    try:
        with open(CONNECTION_PROFILES_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_matching_presets(presets: Dict[str, Dict[str, Any]]) -> None:
    """
    Save matching presets to file.
    
    Args:
        presets: Dictionary of matching presets
    """
    ensure_config_dir()
    with open(MATCHING_PRESETS_FILE, 'w') as f:
        json.dump(presets, f, indent=2)


def load_matching_presets() -> Dict[str, Dict[str, Any]]:
    """
    Load matching presets from file.
    
    Returns:
        Dictionary of matching presets
    """
    ensure_config_dir()
    if not os.path.exists(MATCHING_PRESETS_FILE):
        return {
            "Default": {
                "match_threshold": 0.75,
                "non_match_threshold": 0.55,
                "name_weights": {
                    "first_name": 0.4,
                    "middle_name": 0.2,
                    "last_name": 0.3,
                    "full_name_sorted": 0.1
                },
                "additional_field_weights": {
                    "birthdate": 0.3,
                    "geography": 0.3
                }
            },
            "Strict": {
                "match_threshold": 0.85,
                "non_match_threshold": 0.65,
                "name_weights": {
                    "first_name": 0.3,
                    "middle_name": 0.2,
                    "last_name": 0.4,
                    "full_name_sorted": 0.1
                },
                "additional_field_weights": {
                    "birthdate": 0.4,
                    "geography": 0.2
                }
            },
            "Lenient": {
                "match_threshold": 0.65,
                "non_match_threshold": 0.45,
                "name_weights": {
                    "first_name": 0.5,
                    "middle_name": 0.1,
                    "last_name": 0.3,
                    "full_name_sorted": 0.1
                },
                "additional_field_weights": {
                    "birthdate": 0.2,
                    "geography": 0.2
                }
            }
        }
    
    try:
        with open(MATCHING_PRESETS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def show_error(parent: QWidget, title: str, message: str) -> None:
    """
    Show an error message box.
    
    Args:
        parent: Parent widget
        title: Error title
        message: Error message
    """
    QMessageBox.critical(parent, title, message)


def show_info(parent: QWidget, title: str, message: str) -> None:
    """
    Show an information message box.
    
    Args:
        parent: Parent widget
        title: Information title
        message: Information message
    """
    QMessageBox.information(parent, title, message)


def show_question(parent: QWidget, title: str, message: str) -> bool:
    """
    Show a question message box.
    
    Args:
        parent: Parent widget
        title: Question title
        message: Question message
        
    Returns:
        True if the user clicked Yes, False otherwise
    """
    return QMessageBox.question(
        parent, title, message, 
        QMessageBox.Yes | QMessageBox.No, 
        QMessageBox.No
    ) == QMessageBox.Yes


def get_save_file_name(
    parent: QWidget, 
    title: str, 
    directory: str, 
    filter_str: str
) -> str:
    """
    Get a file name for saving.
    
    Args:
        parent: Parent widget
        title: Dialog title
        directory: Initial directory
        filter_str: File filter string
        
    Returns:
        Selected file name or empty string if canceled
    """
    return QFileDialog.getSaveFileName(parent, title, directory, filter_str)[0]


def get_open_file_name(
    parent: QWidget, 
    title: str, 
    directory: str, 
    filter_str: str
) -> str:
    """
    Get a file name for opening.
    
    Args:
        parent: Parent widget
        title: Dialog title
        directory: Initial directory
        filter_str: File filter string
        
    Returns:
        Selected file name or empty string if canceled
    """
    return QFileDialog.getOpenFileName(parent, title, directory, filter_str)[0]


def populate_table_from_dataframe(table: QTableWidget, df, max_rows: int = 100) -> None:
    """
    Populate a QTableWidget from a pandas DataFrame.
    
    Args:
        table: QTableWidget to populate
        df: pandas DataFrame
        max_rows: Maximum number of rows to display
    """
    # Clear the table
    table.clear()
    table.setRowCount(0)
    
    if df is None or df.empty:
        table.setColumnCount(0)
        return
    
    # Set column headers
    table.setColumnCount(len(df.columns))
    table.setHorizontalHeaderLabels(df.columns)
    
    # Limit the number of rows
    display_df = df.head(max_rows) if len(df) > max_rows else df
    
    # Set row count
    table.setRowCount(len(display_df))
    
    # Populate the table
    for row_idx, (_, row) in enumerate(display_df.iterrows()):
        for col_idx, value in enumerate(row):
            item = QTableWidgetItem(str(value) if value is not None else "")
            table.setItem(row_idx, col_idx, item)
    
    # Resize columns to content
    table.resizeColumnsToContents()


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Format a timestamp for display.
    
    Args:
        timestamp: Timestamp to format (default: current time)
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")
