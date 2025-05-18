"""
Database tab for the Name Matching GUI.

This module provides the database connection and table selection tab for the Name Matching GUI.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit, 
    QPushButton, QComboBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QGroupBox, QSplitter, QTabWidget, QMessageBox, QSpinBox, QFileDialog,
    QApplication, QMainWindow, QAction, QMenu, QToolBar, QStatusBar,
    QDialog, QDialogButtonBox, QListWidget, QListWidgetItem, QFrame
)
from PyQt5.QtCore import Qt, QSettings, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QIcon, QFont

from sqlalchemy import Engine

from .utils import (
    load_connection_profiles, save_connection_profiles, show_error, 
    show_info, show_question, populate_table_from_dataframe
)
from .workers import DatabaseConnectionWorker, TableListWorker, TablePreviewWorker
from .styles import CONNECTION_STATUS


class ConnectionProfileDialog(QDialog):
    """Dialog for saving/loading connection profiles."""
    
    def __init__(self, parent=None, profiles=None, current_profile=None, mode="save"):
        """
        Initialize the dialog.
        
        Args:
            parent: Parent widget
            profiles: Dictionary of connection profiles
            current_profile: Current profile name
            mode: Dialog mode ('save' or 'load')
        """
        super().__init__(parent)
        self.profiles = profiles or {}
        self.current_profile = current_profile
        self.mode = mode
        
        self.setWindowTitle("Connection Profiles")
        self.resize(400, 300)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        
        # Profile list
        self.profile_list = QListWidget()
        self.profile_list.itemClicked.connect(self.on_profile_selected)
        
        # Populate profile list
        for profile_name in self.profiles.keys():
            item = QListWidgetItem(profile_name)
            self.profile_list.addItem(item)
            if profile_name == self.current_profile:
                self.profile_list.setCurrentItem(item)
        
        layout.addWidget(self.profile_list)
        
        # Profile name input (for save mode)
        if self.mode == "save":
            form_layout = QFormLayout()
            self.profile_name_edit = QLineEdit()
            if self.current_profile:
                self.profile_name_edit.setText(self.current_profile)
            form_layout.addRow("Profile Name:", self.profile_name_edit)
            layout.addLayout(form_layout)
        
        # Buttons
        button_box = QDialogButtonBox()
        
        if self.mode == "save":
            self.save_button = button_box.addButton("Save", QDialogButtonBox.AcceptRole)
            self.save_button.clicked.connect(self.accept)
        else:
            self.load_button = button_box.addButton("Load", QDialogButtonBox.AcceptRole)
            self.load_button.clicked.connect(self.accept)
            
            self.delete_button = button_box.addButton("Delete", QDialogButtonBox.DestructiveRole)
            self.delete_button.clicked.connect(self.delete_profile)
        
        self.cancel_button = button_box.addButton("Cancel", QDialogButtonBox.RejectRole)
        self.cancel_button.clicked.connect(self.reject)
        
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def on_profile_selected(self, item):
        """Handle profile selection."""
        if self.mode == "save":
            self.profile_name_edit.setText(item.text())
    
    def delete_profile(self):
        """Delete the selected profile."""
        current_item = self.profile_list.currentItem()
        if not current_item:
            show_error(self, "Error", "Please select a profile to delete.")
            return
        
        profile_name = current_item.text()
        if show_question(self, "Confirm Delete", f"Are you sure you want to delete the profile '{profile_name}'?"):
            del self.profiles[profile_name]
            save_connection_profiles(self.profiles)
            self.profile_list.takeItem(self.profile_list.row(current_item))
    
    def get_selected_profile(self):
        """Get the selected profile name."""
        if self.mode == "save":
            return self.profile_name_edit.text().strip()
        else:
            current_item = self.profile_list.currentItem()
            return current_item.text() if current_item else None


class DatabaseTab(QWidget):
    """Tab for database connection and table selection."""
    
    # Signals
    connection_status_changed = pyqtSignal(str, str)  # status, message
    tables_selected = pyqtSignal(str, str, Engine)  # table1, table2, engine
    
    def __init__(self, parent=None):
        """Initialize the tab."""
        super().__init__(parent)
        
        self.engine = None
        self.tables = []
        self.table1_preview = None
        self.table2_preview = None
        
        self.init_ui()
        self.load_settings()
    
    def init_ui(self):
        """Initialize the UI."""
        main_layout = QVBoxLayout()
        
        # Connection group
        connection_group = QGroupBox("Database Connection")
        connection_layout = QVBoxLayout()
        
        # Connection form
        form_layout = QFormLayout()
        
        self.host_edit = QLineEdit()
        self.host_edit.setPlaceholderText("localhost")
        form_layout.addRow("Host:", self.host_edit)
        
        self.port_edit = QLineEdit()
        self.port_edit.setPlaceholderText("3306")
        form_layout.addRow("Port:", self.port_edit)
        
        self.database_edit = QLineEdit()
        self.database_edit.setPlaceholderText("name_matching")
        form_layout.addRow("Database:", self.database_edit)
        
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("root")
        form_layout.addRow("Username:", self.username_edit)
        
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        form_layout.addRow("Password:", self.password_edit)
        
        self.ssl_checkbox = QCheckBox("Use SSL")
        form_layout.addRow("", self.ssl_checkbox)
        
        connection_layout.addLayout(form_layout)
        
        # Connection buttons
        button_layout = QHBoxLayout()
        
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_to_database)
        button_layout.addWidget(self.connect_button)
        
        self.save_profile_button = QPushButton("Save Profile")
        self.save_profile_button.clicked.connect(self.save_connection_profile)
        button_layout.addWidget(self.save_profile_button)
        
        self.load_profile_button = QPushButton("Load Profile")
        self.load_profile_button.clicked.connect(self.load_connection_profile)
        button_layout.addWidget(self.load_profile_button)
        
        connection_layout.addLayout(button_layout)
        
        # Connection status
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet(CONNECTION_STATUS["disconnected"])
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        connection_layout.addLayout(status_layout)
        
        connection_group.setLayout(connection_layout)
        main_layout.addWidget(connection_group)
        
        # Table selection group
        table_group = QGroupBox("Table Selection")
        table_layout = QVBoxLayout()
        
        # Table selection form
        table_form_layout = QFormLayout()
        
        self.table1_combo = QComboBox()
        self.table1_combo.setEnabled(False)
        self.table1_combo.currentIndexChanged.connect(self.on_table1_changed)
        table_form_layout.addRow("Table 1:", self.table1_combo)
        
        self.table2_combo = QComboBox()
        self.table2_combo.setEnabled(False)
        self.table2_combo.currentIndexChanged.connect(self.on_table2_changed)
        table_form_layout.addRow("Table 2:", self.table2_combo)
        
        table_layout.addLayout(table_form_layout)
        
        # Table preview tabs
        self.preview_tabs = QTabWidget()
        
        # Table 1 preview
        self.table1_preview_widget = QTableWidget()
        self.preview_tabs.addTab(self.table1_preview_widget, "Table 1 Preview")
        
        # Table 2 preview
        self.table2_preview_widget = QTableWidget()
        self.preview_tabs.addTab(self.table2_preview_widget, "Table 2 Preview")
        
        table_layout.addWidget(self.preview_tabs)
        
        # Use selected tables button
        self.use_tables_button = QPushButton("Use Selected Tables")
        self.use_tables_button.setEnabled(False)
        self.use_tables_button.clicked.connect(self.use_selected_tables)
        table_layout.addWidget(self.use_tables_button)
        
        table_group.setLayout(table_layout)
        main_layout.addWidget(table_group)
        
        self.setLayout(main_layout)
    
    def load_settings(self):
        """Load settings from QSettings."""
        settings = QSettings()
        
        # Load connection settings
        self.host_edit.setText(settings.value("database/host", "localhost"))
        self.port_edit.setText(settings.value("database/port", "3306"))
        self.database_edit.setText(settings.value("database/name", "name_matching"))
        self.username_edit.setText(settings.value("database/username", "root"))
        self.ssl_checkbox.setChecked(settings.value("database/use_ssl", False, type=bool))
    
    def save_settings(self):
        """Save settings to QSettings."""
        settings = QSettings()
        
        # Save connection settings
        settings.setValue("database/host", self.host_edit.text())
        settings.setValue("database/port", self.port_edit.text())
        settings.setValue("database/name", self.database_edit.text())
        settings.setValue("database/username", self.username_edit.text())
        settings.setValue("database/use_ssl", self.ssl_checkbox.isChecked())
    
    def connect_to_database(self):
        """Connect to the database."""
        # Get connection parameters
        host = self.host_edit.text().strip() or "localhost"
        port = self.port_edit.text().strip() or "3306"
        db_name = self.database_edit.text().strip() or "name_matching"
        user = self.username_edit.text().strip() or "root"
        password = self.password_edit.text()
        use_ssl = self.ssl_checkbox.isChecked()
        
        # Update UI
        self.status_label.setText("Connecting...")
        self.status_label.setStyleSheet(CONNECTION_STATUS["connecting"])
        self.connect_button.setEnabled(False)
        QApplication.processEvents()
        
        # Create worker
        self.connection_worker = DatabaseConnectionWorker(
            db_name, host, port, user, password, use_ssl
        )
        
        # Connect signals
        self.connection_worker.signals.result.connect(self.on_connection_success)
        self.connection_worker.signals.error.connect(self.on_connection_error)
        self.connection_worker.signals.finished.connect(
            lambda: self.connect_button.setEnabled(True)
        )
        
        # Start worker
        self.connection_worker.start()
        
        # Save settings
        self.save_settings()
    
    def on_connection_success(self, engine):
        """Handle successful database connection."""
        self.engine = engine
        self.status_label.setText("Connected")
        self.status_label.setStyleSheet(CONNECTION_STATUS["connected"])
        
        # Emit signal
        self.connection_status_changed.emit("connected", "Connected to database")
        
        # Load tables
        self.load_tables()
    
    def on_connection_error(self, error_message):
        """Handle database connection error."""
        self.status_label.setText("Disconnected")
        self.status_label.setStyleSheet(CONNECTION_STATUS["disconnected"])
        
        # Emit signal
        self.connection_status_changed.emit("disconnected", error_message)
        
        # Show error
        show_error(self, "Connection Error", error_message)
    
    def load_tables(self):
        """Load tables from the database."""
        if not self.engine:
            return
        
        # Create worker
        self.table_worker = TableListWorker(self.engine)
        
        # Connect signals
        self.table_worker.signals.result.connect(self.on_tables_loaded)
        self.table_worker.signals.error.connect(
            lambda msg: show_error(self, "Error", msg)
        )
        
        # Start worker
        self.table_worker.start()
    
    def on_tables_loaded(self, tables):
        """Handle loaded tables."""
        self.tables = tables
        
        # Update UI
        self.table1_combo.clear()
        self.table2_combo.clear()
        
        self.table1_combo.addItems(tables)
        self.table2_combo.addItems(tables)
        
        self.table1_combo.setEnabled(True)
        self.table2_combo.setEnabled(True)
        
        # Load previews if tables are selected
        if self.table1_combo.currentText():
            self.load_table_preview(self.table1_combo.currentText(), 1)
        
        if self.table2_combo.currentText():
            self.load_table_preview(self.table2_combo.currentText(), 2)
        
        # Enable use tables button
        self.update_use_tables_button()
    
    def on_table1_changed(self, index):
        """Handle table 1 selection change."""
        if index >= 0 and self.engine:
            table_name = self.table1_combo.currentText()
            self.load_table_preview(table_name, 1)
            self.update_use_tables_button()
    
    def on_table2_changed(self, index):
        """Handle table 2 selection change."""
        if index >= 0 and self.engine:
            table_name = self.table2_combo.currentText()
            self.load_table_preview(table_name, 2)
            self.update_use_tables_button()
    
    def load_table_preview(self, table_name, table_num):
        """Load preview for a table."""
        if not self.engine or not table_name:
            return
        
        # Create worker
        worker = TablePreviewWorker(self.engine, table_name, limit=100)
        
        # Connect signals
        if table_num == 1:
            worker.signals.result.connect(self.on_table1_preview_loaded)
        else:
            worker.signals.result.connect(self.on_table2_preview_loaded)
        
        worker.signals.error.connect(
            lambda msg: show_error(self, "Error", msg)
        )
        
        # Start worker
        worker.start()
    
    def on_table1_preview_loaded(self, df):
        """Handle loaded table 1 preview."""
        self.table1_preview = df
        populate_table_from_dataframe(self.table1_preview_widget, df)
        self.preview_tabs.setTabText(0, f"Table 1 Preview ({len(df)} rows)")
    
    def on_table2_preview_loaded(self, df):
        """Handle loaded table 2 preview."""
        self.table2_preview = df
        populate_table_from_dataframe(self.table2_preview_widget, df)
        self.preview_tabs.setTabText(1, f"Table 2 Preview ({len(df)} rows)")
    
    def update_use_tables_button(self):
        """Update the state of the use tables button."""
        table1 = self.table1_combo.currentText()
        table2 = self.table2_combo.currentText()
        
        self.use_tables_button.setEnabled(
            self.engine is not None and table1 and table2
        )
    
    def use_selected_tables(self):
        """Use the selected tables for matching."""
        table1 = self.table1_combo.currentText()
        table2 = self.table2_combo.currentText()
        
        if not table1 or not table2:
            show_error(self, "Error", "Please select both tables.")
            return
        
        # Emit signal
        self.tables_selected.emit(table1, table2, self.engine)
    
    def save_connection_profile(self):
        """Save the current connection profile."""
        # Get current connection parameters
        host = self.host_edit.text().strip() or "localhost"
        port = self.port_edit.text().strip() or "3306"
        db_name = self.database_edit.text().strip() or "name_matching"
        user = self.username_edit.text().strip() or "root"
        password = self.password_edit.text()
        use_ssl = self.ssl_checkbox.isChecked()
        
        # Load existing profiles
        profiles = load_connection_profiles()
        
        # Show dialog
        dialog = ConnectionProfileDialog(
            self, profiles, None, "save"
        )
        
        if dialog.exec_() == QDialog.Accepted:
            profile_name = dialog.get_selected_profile()
            
            if not profile_name:
                show_error(self, "Error", "Please enter a profile name.")
                return
            
            # Save profile
            profiles[profile_name] = {
                "host": host,
                "port": port,
                "database": db_name,
                "username": user,
                "password": password,
                "use_ssl": use_ssl
            }
            
            save_connection_profiles(profiles)
            show_info(self, "Success", f"Profile '{profile_name}' saved.")
    
    def load_connection_profile(self):
        """Load a connection profile."""
        # Load existing profiles
        profiles = load_connection_profiles()
        
        if not profiles:
            show_info(self, "No Profiles", "No connection profiles found.")
            return
        
        # Show dialog
        dialog = ConnectionProfileDialog(
            self, profiles, None, "load"
        )
        
        if dialog.exec_() == QDialog.Accepted:
            profile_name = dialog.get_selected_profile()
            
            if not profile_name or profile_name not in profiles:
                show_error(self, "Error", "Please select a profile.")
                return
            
            # Load profile
            profile = profiles[profile_name]
            
            self.host_edit.setText(profile.get("host", "localhost"))
            self.port_edit.setText(profile.get("port", "3306"))
            self.database_edit.setText(profile.get("database", "name_matching"))
            self.username_edit.setText(profile.get("username", "root"))
            self.password_edit.setText(profile.get("password", ""))
            self.ssl_checkbox.setChecked(profile.get("use_ssl", False))
            
            show_info(self, "Success", f"Profile '{profile_name}' loaded.")
