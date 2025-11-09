"""
UI Manager Module
---------------
Handles the PyQt5 graphical user interface for the floor monitoring system.
"""
import sys
import logging
from typing import Dict, List, Optional
from datetime import datetime
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                             QGroupBox, QStatusBar, QToolBar, QAction, QTabWidget,
                             QTextEdit, QProgressBar, QComboBox, QCheckBox, QSpinBox,
                             QDoubleSpinBox, QFormLayout, QDialog, QListWidget, QListWidgetItem, QLineEdit,
                             QHeaderView, QFrame, QScrollArea, QGridLayout, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QColor, QPalette, QLinearGradient, QPainter, QBrush
import time

logger = logging.getLogger(__name__)

class CameraWidget(QWidget):
    """Widget for displaying camera feed."""
    
    def __init__(self, camera_id: int, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.init_ui()
        
    def init_ui(self):
        """Initialize the camera widget UI with modern styling."""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Camera label with gradient background
        self.label = QLabel(f"📹 Camera {self.camera_id}")
        self.label.setAlignment(Qt.AlignCenter)  # type: ignore
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2196F3, stop:1 #1976D2);
                color: white;
                padding: 12px;
                border-radius: 8px;
                font-weight: bold;
            }
        """)
        
        # Video display with border and shadow effect
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)  # type: ignore
        self.video_display.setMinimumSize(640, 480)
        self.video_display.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 3px solid #2196F3;
                border-radius: 10px;
            }
        """)
        
        # Status bar with modern styling
        self.status_label = QLabel("⚫ Status: Disconnected")
        self.status_label.setAlignment(Qt.AlignCenter)  # type: ignore
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                padding: 8px;
                border-radius: 5px;
                font-size: 12px;
            }
        """)
        
        layout.addWidget(self.label)
        layout.addWidget(self.video_display)
        layout.addWidget(self.status_label)
        self.setLayout(layout)
        
        # Set widget background
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 10px;
            }
        """)
    
    def update_frame(self, frame: np.ndarray):
        """Update the video display with a new frame."""
        if frame is None:
            return
            
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage - fix for proper data handling
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            # Convert numpy array to bytes for QImage
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)  # type: ignore
            # Make a copy to avoid issues with memory management
            q_img = q_img.copy()
            
            # Scale to fit the label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(
                self.video_display.size(),
                Qt.KeepAspectRatio,  # type: ignore
                Qt.SmoothTransformation  # type: ignore
            )
            
            self.video_display.setPixmap(scaled_pixmap)
        except Exception as e:
            logger.error(f"Error updating frame for camera {self.camera_id}: {str(e)}")
    
    def set_status(self, status: str, color: str = "black"):
        """Update the status label with modern styling and icons."""
        icon = "⚫"
        bg_color = "#f5f5f5"
        text_color = "black"
        
        if color == "red":
            icon = "🔴"
            bg_color = "#ffebee"
            text_color = "#c62828"
        elif color == "green":
            icon = "🟢"
            bg_color = "#e8f5e9"
            text_color = "#2e7d32"
        elif color == "orange":
            icon = "🟠"
            bg_color = "#fff3e0"
            text_color = "#ef6c00"
        
        self.status_label.setText(f"{icon} Status: {status}")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_color};
                color: {text_color};
                padding: 8px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }}
        """)


class WorkerStatusWidget(QWidget):
    """Widget for displaying worker status information."""
    
    # Signals for worker management
    add_worker_signal = pyqtSignal()
    delete_workers_signal = pyqtSignal(list)  # List of (worker_id, worker_name) tuples
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the worker status widget UI with modern design."""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title with icon
        title = QLabel("👥 Worker Status Dashboard")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.AlignCenter)  # type: ignore
        title.setStyleSheet("""
            QLabel {
                color: #1976D2;
                padding: 10px;
                background-color: #E3F2FD;
                border-radius: 8px;
            }
        """)
        
        # Statistics cards row
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(10)
        
        self.total_workers_card = self._create_stat_card("👤 Total", "0", "#2196F3")
        self.present_workers_card = self._create_stat_card("✅ Present", "0", "#4CAF50")
        self.absent_workers_card = self._create_stat_card("❌ Absent", "0", "#f44336")
        self.exceeded_workers_card = self._create_stat_card("⚠️ Exceeded", "0", "#FF9800")
        
        stats_layout.addWidget(self.total_workers_card)
        stats_layout.addWidget(self.present_workers_card)
        stats_layout.addWidget(self.absent_workers_card)
        stats_layout.addWidget(self.exceeded_workers_card)
        
        # Search and filter bar
        search_layout = QHBoxLayout()
        search_layout.setSpacing(10)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search by name or ID...")
        self.search_input.textChanged.connect(self.filter_workers)
        self.search_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 2px solid #2196F3;
                border-radius: 5px;
                font-size: 13px;
                background-color: white;
            }
            QLineEdit:focus {
                border: 2px solid #1976D2;
            }
        """)
        
        self.status_filter = QComboBox()
        self.status_filter.addItems(["All Status", "Present", "Absent", "Exceeded"])
        self.status_filter.currentTextChanged.connect(self.filter_workers)
        self.status_filter.setStyleSheet("""
            QComboBox {
                padding: 10px;
                border: 2px solid #2196F3;
                border-radius: 5px;
                font-size: 13px;
                background-color: white;
                min-width: 150px;
            }
        """)
        
        search_layout.addWidget(self.search_input, 3)
        search_layout.addWidget(self.status_filter, 1)
        
        # Add Worker button
        add_button_layout = QHBoxLayout()
        self.add_worker_button = QPushButton("➕ Add New Worker")
        self.add_worker_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
            QPushButton:pressed {
                background-color: #2E7D32;
            }
        """)
        self.add_worker_button.clicked.connect(lambda: self.add_worker_signal.emit())
        add_button_layout.addStretch()
        add_button_layout.addWidget(self.add_worker_button)
        
        # Worker table with modern styling
        self.worker_table = QTableWidget()
        self.worker_table.setColumnCount(6)  # Removed Actions column
        self.worker_table.setHorizontalHeaderLabels([
            "🆔 ID", "👤 Name", "🟢 Status", "⏱️ Present", "⏰ Absent", "📅 Last Seen"
        ])
        self.worker_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.worker_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.worker_table.setSelectionMode(QTableWidget.MultiSelection)  # Enable multi-select
        self.worker_table.setAlternatingRowColors(True)
        self.worker_table.verticalHeader().setVisible(False)
        
        # Style the table
        self.worker_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                gridline-color: #f0f0f0;
                font-size: 13px;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QTableWidget::item:selected {
                background-color: #BBDEFB;
                color: black;
            }
            QHeaderView::section {
                background-color: #2196F3;
                color: white;
                padding: 10px;
                border: none;
                font-weight: bold;
                font-size: 13px;
            }
            QTableWidget::item:alternate {
                background-color: #f9f9f9;
            }
        """)
        
        # Configure header
        header = self.worker_table.horizontalHeader()
        if header:
            header.setStretchLastSection(True)
            header.setSectionResizeMode(QHeaderView.Interactive)
        
        # Store original data for filtering
        self.all_workers_data = []
        
        # Bottom action bar with selection info and delete button
        bottom_action_layout = QHBoxLayout()
        bottom_action_layout.setSpacing(10)
        
        self.selection_label = QLabel("📋 Selected: 0 workers")
        self.selection_label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: #666;
                padding: 8px;
                background-color: #f5f5f5;
                border-radius: 5px;
            }
        """)
        
        self.delete_selected_button = QPushButton("🗑️ Delete Selected Workers")
        self.delete_selected_button.setEnabled(False)
        self.delete_selected_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #c62828;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.delete_selected_button.clicked.connect(self.delete_selected_workers)
        
        bottom_action_layout.addWidget(self.selection_label)
        bottom_action_layout.addStretch()
        bottom_action_layout.addWidget(self.delete_selected_button)
        
        # Connect selection changed signal
        self.worker_table.itemSelectionChanged.connect(self.update_selection_label)
        
        layout.addWidget(title)
        layout.addLayout(stats_layout)
        layout.addLayout(search_layout)
        layout.addLayout(add_button_layout)
        layout.addWidget(self.worker_table)
        layout.addLayout(bottom_action_layout)
        self.setLayout(layout)
        
        # Set widget background
        self.setStyleSheet("""
            QWidget {
                background-color: #fafafa;
            }
        """)
    
    def _create_stat_card(self, title: str, value: str, color: str) -> QFrame:
        """Create a statistics card widget."""
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border-left: 5px solid {color};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        
        card_layout = QVBoxLayout()
        card_layout.setSpacing(5)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 12px; color: #666; font-weight: bold;")
        
        value_label = QLabel(value)
        value_label.setStyleSheet(f"font-size: 24px; color: {color}; font-weight: bold;")
        value_label.setAlignment(Qt.AlignCenter)  # type: ignore
        
        card_layout.addWidget(title_label)
        card_layout.addWidget(value_label)
        card.setLayout(card_layout)
        
        # Store value label for updates
        card.value_label = value_label
        
        return card
    
    def filter_workers(self):
        """Filter workers based on search and status filter."""
        search_text = self.search_input.text().lower()
        status_filter = self.status_filter.currentText()
        
        # Filter the stored data
        filtered_workers = []
        for worker in self.all_workers_data:
            # Check search text
            name_match = search_text in worker.get('name', '').lower()
            id_match = search_text in str(worker.get('worker_id', ''))
            
            if not (name_match or id_match or not search_text):
                continue
            
            # Check status filter
            if status_filter != "All Status":
                if worker.get('status', '').lower() != status_filter.lower():
                    continue
            
            filtered_workers.append(worker)
        
        # Update table with filtered data
        self._populate_table(filtered_workers)
    
    def update_workers(self, workers: List[Dict]):
        """Update the worker table with new data."""
        logger.debug(f"update_workers called with {len(workers)} workers")
        if not workers:
            logger.warning("update_workers received empty list - table will be empty")
        
        # Store data for filtering
        self.all_workers_data = workers
        
        # Update statistics cards
        total = len(workers)
        present = sum(1 for w in workers if w.get('status') == 'present')
        absent = sum(1 for w in workers if w.get('status') == 'absent')
        exceeded = sum(1 for w in workers if w.get('status') == 'exceeded')
        
        self.total_workers_card.value_label.setText(str(total))
        self.present_workers_card.value_label.setText(str(present))
        self.absent_workers_card.value_label.setText(str(absent))
        self.exceeded_workers_card.value_label.setText(str(exceeded))
        
        # Apply current filters
        self.filter_workers()
    
    def _populate_table(self, workers: List[Dict]):
        """Populate the table with worker data."""
        self.worker_table.setRowCount(len(workers))
        
        # Track absent workers for special highlighting
        absent_workers = []
        
        for row, worker in enumerate(workers):
            worker_id = worker.get('worker_id', '')
            name = worker.get('name', '')
            status = worker.get('status', 'unknown')
            
            # Worker ID
            item = QTableWidgetItem(str(worker_id))
            item.setTextAlignment(Qt.AlignCenter)  # type: ignore
            self.worker_table.setItem(row, 0, item)
            
            # Name
            item = QTableWidgetItem(name)
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # type: ignore
            self.worker_table.setItem(row, 1, item)
            
            # Status
            item = QTableWidgetItem(status)
            if status == 'present':
                item.setForeground(QColor('green'))
            elif status == 'exceeded':
                item.setForeground(QColor('red'))
            else:
                item.setForeground(QColor('gray'))
            item.setTextAlignment(Qt.AlignCenter)  # type: ignore
            self.worker_table.setItem(row, 2, item)
            
            # Time Present
            time_present = worker.get('time_present', 0)
            if time_present > 0:
                minutes = int(time_present // 60)
                seconds = int(time_present % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
            else:
                time_str = "00:00"
            item = QTableWidgetItem(time_str)
            item.setTextAlignment(Qt.AlignCenter)  # type: ignore
            self.worker_table.setItem(row, 3, item)
            
            # Total Absent Time - real-time formatting (total_absent_time is seconds)
            total_absent_secs = int(worker.get('total_absent_time', 0))
            if total_absent_secs > 0:
                hours = total_absent_secs // 3600
                minutes = (total_absent_secs % 3600) // 60
                seconds = total_absent_secs % 60
                absent_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                absent_str = "00:00:00"
            item = QTableWidgetItem(absent_str)
            item.setTextAlignment(Qt.AlignCenter)  # type: ignore

            # Highlight absent workers with special formatting
            if status == 'absent':
                item.setForeground(QColor('red'))
                item.setBackground(QColor('#ffebee'))  # Light red background
                # Track absent workers for potential additional UI elements
                absent_workers.append({
                    'worker_id': worker_id,
                    'name': name,
                    'absent_time': total_absent_secs
                })
            elif total_absent_secs >= 4 * 3600:  # More than 4 hours total
                item.setForeground(QColor('red'))
                item.setBackground(QColor('#fff3e0'))  # Light orange background

            self.worker_table.setItem(row, 4, item)
            
            # Last Seen
            last_seen = worker.get('last_seen', 0)
            if last_seen > 0:
                dt = datetime.fromtimestamp(last_seen)
                time_str = dt.strftime("%H:%M:%S")
            else:
                time_str = "Never"
            item = QTableWidgetItem(time_str)
            item.setTextAlignment(Qt.AlignCenter)  # type: ignore
            self.worker_table.setItem(row, 5, item)
            
            # Store worker_id in the first column item for later retrieval
            self.worker_table.item(row, 0).setData(Qt.UserRole, worker_id)
            self.worker_table.item(row, 1).setData(Qt.UserRole, name)
        
        # Resize columns to fit content
        self.worker_table.resizeColumnsToContents()
        
        # Log absent workers for monitoring
        if absent_workers:
            logger.info(f"Currently absent workers: {len(absent_workers)} - {[w['name'] for w in absent_workers]}")
    
    def update_selection_label(self):
        """Update the selection label and enable/disable delete button."""
        selected_rows = self.worker_table.selectionModel().selectedRows()
        count = len(selected_rows)
        
        if count == 0:
            self.selection_label.setText("📋 Selected: 0 workers")
            self.delete_selected_button.setEnabled(False)
        elif count == 1:
            self.selection_label.setText("📋 Selected: 1 worker")
            self.delete_selected_button.setEnabled(True)
        else:
            self.selection_label.setText(f"📋 Selected: {count} workers")
            self.delete_selected_button.setEnabled(True)
    
    def delete_selected_workers(self):
        """Emit signal to delete selected workers."""
        selected_rows = self.worker_table.selectionModel().selectedRows()
        
        if not selected_rows:
            return
        
        # Collect worker info from selected rows
        workers_to_delete = []
        for index in selected_rows:
            row = index.row()
            worker_id_item = self.worker_table.item(row, 0)
            worker_name_item = self.worker_table.item(row, 1)
            
            if worker_id_item and worker_name_item:
                worker_id = worker_id_item.data(Qt.UserRole)
                worker_name = worker_name_item.data(Qt.UserRole)
                workers_to_delete.append((worker_id, worker_name))
        
        if workers_to_delete:
            self.delete_workers_signal.emit(workers_to_delete)


class AlertWidget(QWidget):
    """Widget for displaying alerts and notifications."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the alert widget UI with modern design."""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title with icon
        title = QLabel("🔔 Alerts & Notifications")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.AlignCenter)  # type: ignore
        title.setStyleSheet("""
            QLabel {
                color: #1976D2;
                padding: 10px;
                background-color: #E3F2FD;
                border-radius: 8px;
            }
        """)
        
        # Alert count label
        self.alert_count_label = QLabel("📊 Total Alerts: 0")
        self.alert_count_label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: #666;
                padding: 5px;
                background-color: #f5f5f5;
                border-radius: 5px;
            }
        """)
        
        # Alert list with modern styling
        self.alert_list = QListWidget()
        self.alert_list.setStyleSheet("""
            QListWidget {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 5px;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #f0f0f0;
                border-radius: 5px;
                margin: 2px;
            }
            QListWidget::item:selected {
                background-color: #BBDEFB;
                color: black;
            }
            QListWidget::item:hover {
                background-color: #E3F2FD;
            }
        """)
        
        # Control buttons with modern styling
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.clear_button = QPushButton("🗑️ Clear All Alerts")
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #c62828;
            }
        """)
        self.clear_button.clicked.connect(self.clear_alerts)
        
        self.acknowledge_button = QPushButton("✅ Acknowledge Selected")
        self.acknowledge_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
            QPushButton:pressed {
                background-color: #2E7D32;
            }
        """)
        self.acknowledge_button.clicked.connect(self.acknowledge_alert)
        
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.acknowledge_button)
        button_layout.addStretch()
        
        layout.addWidget(title)
        layout.addWidget(self.alert_count_label)
        layout.addWidget(self.alert_list)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Set widget background
        self.setStyleSheet("""
            QWidget {
                background-color: #fafafa;
            }
        """)

    def add_alert_item(self, title: str, message: str, alert_type: str) -> None:
        """Append an alert message to the list with modern styling."""
        try:
            # Add timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Choose icon and color based on type
            if alert_type == "alert":
                icon = "🚨"
                color = QColor('#c62828')
                bg_color = QColor('#ffebee')
            elif alert_type == "warning":
                icon = "⚠️"
                color = QColor('#ef6c00')
                bg_color = QColor('#fff3e0')
            else:
                icon = "ℹ️"
                color = QColor('#1976D2')
                bg_color = QColor('#E3F2FD')
            
            text = f"{icon} [{timestamp}] {title} - {message}"
            item = QListWidgetItem(text)
            item.setForeground(color)  # type: ignore
            item.setBackground(bg_color)  # type: ignore
            
            # Prepend newest on top
            self.alert_list.insertItem(0, item)
            
            # Update alert count
            count = self.alert_list.count()
            self.alert_count_label.setText(f"📊 Total Alerts: {count}")
            
        except Exception:
            # Non-fatal UI update failure should not crash the app
            pass
    
    def clear_alerts(self):
        """Clear all alerts from the list."""
        self.alert_list.clear()
        self.alert_count_label.setText("📊 Total Alerts: 0")
    
    def acknowledge_alert(self):
        """Remove selected alert."""
        current_item = self.alert_list.currentItem()
        if current_item:
            row = self.alert_list.row(current_item)
            self.alert_list.takeItem(row)
            count = self.alert_list.count()
            self.alert_count_label.setText(f"📊 Total Alerts: {count}")


class SettingsDialog(QDialog):
    """Dialog for application settings."""
    
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config.copy()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the settings dialog UI."""
        self.setWindowTitle("Application Settings")
        self.setModal(True)
        self.resize(400, 300)
        
        layout = QVBoxLayout()
        
        # Database settings group
        db_group = QGroupBox("Database Settings")
        db_layout = QFormLayout()
        
        self.db_host = QLineEdit(self.config.get("database", {}).get("host", "localhost"))
        self.db_name = QLineEdit(self.config.get("database", {}).get("name", "floor_monitor"))
        self.db_user = QLineEdit(self.config.get("database", {}).get("user", "postgres"))
        self.db_password = QLineEdit(self.config.get("database", {}).get("password", ""))
        self.db_password.setEchoMode(QLineEdit.Password)
        self.db_port = QSpinBox()
        self.db_port.setRange(1, 65535)
        self.db_port.setValue(self.config.get("database", {}).get("port", 5432))
        
        db_layout.addRow("Host:", self.db_host)
        db_layout.addRow("Database:", self.db_name)
        db_layout.addRow("User:", self.db_user)
        db_layout.addRow("Password:", self.db_password)
        db_layout.addRow("Port:", self.db_port)
        db_group.setLayout(db_layout)
        
        # Threshold settings group
        threshold_group = QGroupBox("Threshold Settings")
        threshold_layout = QFormLayout()
        
        self.warning_minutes = QSpinBox()
        self.warning_minutes.setRange(1, 120)
        self.warning_minutes.setValue(self.config.get("thresholds", {}).get("warning_minutes", 15))
        
        self.alert_minutes = QSpinBox()
        self.alert_minutes.setRange(1, 240)
        self.alert_minutes.setValue(self.config.get("thresholds", {}).get("alert_minutes", 30))
        
        threshold_layout.addRow("Warning (minutes):", self.warning_minutes)
        threshold_layout.addRow("Alert (minutes):", self.alert_minutes)
        threshold_group.setLayout(threshold_layout)
        
        # Notification settings group
        notification_group = QGroupBox("Notification Settings")
        notification_layout = QVBoxLayout()
        
        self.notifications_enabled = QCheckBox("Enable Notifications")
        self.notifications_enabled.setChecked(self.config.get("notifications", {}).get("enabled", True))
        
        self.sound_enabled = QCheckBox("Enable Sound Alerts")
        self.sound_enabled.setChecked(self.config.get("notifications", {}).get("sound", True))
        
        notification_layout.addWidget(self.notifications_enabled)
        notification_layout.addWidget(self.sound_enabled)
        notification_group.setLayout(notification_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        # Connect buttons
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        # Add all widgets to main layout
        layout.addWidget(db_group)
        layout.addWidget(threshold_group)
        layout.addWidget(notification_group)
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def get_config(self) -> dict:
        """Get the updated configuration."""
        return {
            "database": {
                "host": self.db_host.text(),
                "name": self.db_name.text(),
                "user": self.db_user.text(),
                "password": self.db_password.text(),
                "port": self.db_port.value()
            },
            "thresholds": {
                "warning_minutes": self.warning_minutes.value(),
                "alert_minutes": self.alert_minutes.value()
            },
            "notifications": {
                "enabled": self.notifications_enabled.isChecked(),
                "sound": self.sound_enabled.isChecked()
            }
        }


class UIManager(QMainWindow):
    """Main UI manager for the floor monitoring application."""
    
    # Signals for communication with other modules
    start_camera_signal = pyqtSignal()
    stop_camera_signal = pyqtSignal()
    settings_changed_signal = pyqtSignal(dict)
    register_worker_signal = pyqtSignal()  # Signal for worker registration
    # Signal to safely append alerts from any thread
    append_alert_signal = pyqtSignal(str, str, str)
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self.config = config or {}
        self.cameras = {}
        self.init_ui()
        
    def init_ui(self):
        """Initialize the main UI with modern design."""
        self.setWindowTitle("🏢 Floor Monitoring System - Real-time Worker Tracking")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set modern color scheme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                color: #333;
                padding: 12px 24px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-size: 13px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #2196F3;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #BBDEFB;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget for different views
        self.tab_widget = QTabWidget()
        
        # Camera tab
        self.camera_tab = QWidget()
        self.camera_layout = QHBoxLayout(self.camera_tab)
        self.tab_widget.addTab(self.camera_tab, "Cameras")
        
        # Status tab
        self.status_tab = QWidget()
        status_layout = QVBoxLayout(self.status_tab)
        self.worker_status_widget = WorkerStatusWidget()
        status_layout.addWidget(self.worker_status_widget)
        self.tab_widget.addTab(self.status_tab, "Worker Status")
        
        # Alerts tab
        self.alert_tab = QWidget()
        alert_layout = QVBoxLayout(self.alert_tab)
        self.alert_widget = AlertWidget()
        alert_layout.addWidget(self.alert_widget)
        self.tab_widget.addTab(self.alert_tab, "Alerts")

        # Connect cross-thread alert signal to UI slot
        try:
            self.append_alert_signal.connect(self.alert_widget.add_alert_item)
        except Exception:
            pass
        
        # Reports tab
        self.reports_tab = QWidget()
        reports_layout = QVBoxLayout(self.reports_tab)
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        reports_layout.addWidget(self.report_text)
        self.tab_widget.addTab(self.reports_tab, "Reports")
        
        main_layout.addWidget(self.tab_widget)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create status bar with modern styling
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #E3F2FD;
                color: #1976D2;
                font-size: 13px;
                font-weight: bold;
                padding: 5px;
                border-top: 2px solid #2196F3;
            }
        """)
        self.status_bar.showMessage("🟢 Ready - System initialized successfully")
        
        # Set up timer for UI updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(1000)  # Update every second
    
    def create_toolbar(self):
        """Create the application toolbar with modern styling."""
        toolbar = self.addToolBar("Main")
        if toolbar is not None:
            toolbar.setMovable(False)
            toolbar.setIconSize(QSize(32, 32))
            toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            toolbar.setStyleSheet("""
                QToolBar {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #2196F3, stop:1 #1976D2);
                    spacing: 10px;
                    padding: 8px;
                    border-bottom: 3px solid #1565C0;
                }
                QToolButton {
                    background-color: rgba(255, 255, 255, 0.1);
                    color: white;
                    border: 2px solid rgba(255, 255, 255, 0.3);
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-size: 13px;
                    font-weight: bold;
                    margin: 2px;
                }
                QToolButton:hover {
                    background-color: rgba(255, 255, 255, 0.2);
                    border: 2px solid rgba(255, 255, 255, 0.5);
                }
                QToolButton:pressed {
                    background-color: rgba(255, 255, 255, 0.3);
                }
                QToolButton:disabled {
                    background-color: rgba(255, 255, 255, 0.05);
                    color: rgba(255, 255, 255, 0.4);
                    border: 2px solid rgba(255, 255, 255, 0.1);
                }
            """)
            
            # Start action
            self.start_action = QAction(QIcon(), "▶️ Start Monitoring", self)
            self.start_action.setStatusTip("Start camera monitoring")
            self.start_action.triggered.connect(self.start_monitoring)
            toolbar.addAction(self.start_action)
            
            # Stop action
            self.stop_action = QAction(QIcon(), "⏹️ Stop Monitoring", self)
            self.stop_action.setStatusTip("Stop camera monitoring")
            self.stop_action.triggered.connect(self.stop_monitoring)
            self.stop_action.setEnabled(False)  # Initially disabled
            toolbar.addAction(self.stop_action)
            
            # Add separator
            toolbar.addSeparator()
            
            # Register Worker action
            self.register_action = QAction(QIcon(), "👤 Register Worker", self)
            self.register_action.setStatusTip("Register a new worker")
            self.register_action.triggered.connect(self.register_worker)
            toolbar.addAction(self.register_action)
            
            # Add separator
            toolbar.addSeparator()
            
            # Settings action
            self.settings_action = QAction(QIcon(), "⚙️ Settings", self)
            self.settings_action.setStatusTip("Open settings dialog")
            self.settings_action.triggered.connect(self.open_settings)
            toolbar.addAction(self.settings_action)
            
            # Add separator
            toolbar.addSeparator()
            
            # Exit action
            self.exit_action = QAction(QIcon(), "🚪 Exit", self)
            self.exit_action.setStatusTip("Exit application")
            self.exit_action.triggered.connect(self.close)  # type: ignore
            toolbar.addAction(self.exit_action)
    
    def start_monitoring(self):
        """Start camera monitoring."""
        self.start_camera_signal.emit()
        self.status_bar.showMessage("Monitoring started")
        logger.info("UI: Start monitoring requested")
        
        # Update button states
        self.start_action.setEnabled(False)
        self.stop_action.setEnabled(True)
    
    def stop_monitoring(self):
        """Stop camera monitoring."""
        self.stop_camera_signal.emit()
        self.status_bar.showMessage("Monitoring stopped")
        logger.info("UI: Stop monitoring requested")
        
        # Update button states
        self.start_action.setEnabled(True)
        self.stop_action.setEnabled(False)
    
    def register_worker(self):
        """Open worker registration dialog."""
        self.register_worker_signal.emit()
        logger.info("UI: Worker registration requested")
    
    def open_settings(self):
        """Open the settings dialog."""
        dialog = SettingsDialog(self.config, self)
        if dialog.exec_() == QDialog.Accepted:
            new_config = dialog.get_config()
            self.settings_changed_signal.emit(new_config)
            self.status_bar.showMessage("Settings updated")
            logger.info("UI: Settings updated")
    
    def add_camera(self, camera_id: int):
        """Add a camera widget to the UI."""
        if camera_id not in self.cameras:
            camera_widget = CameraWidget(camera_id)
            self.cameras[camera_id] = camera_widget
            self.camera_layout.addWidget(camera_widget)
            logger.info(f"UI: Added camera {camera_id}")
    
    def update_camera_frame(self, camera_id: int, frame: np.ndarray):
        """Update a camera's frame."""
        if camera_id in self.cameras:
            self.cameras[camera_id].update_frame(frame)
    
    def update_camera_status(self, camera_id: int, status: str, color: str = "black"):
        """Update a camera's status."""
        if camera_id in self.cameras:
            self.cameras[camera_id].set_status(status, color)
    
    def update_worker_status(self, workers: List[Dict]):
        """Update the worker status display."""
        self.worker_status_widget.update_workers(workers)
        
        # Update status bar with absent worker information
        absent_count = sum(1 for worker in workers if worker.get('status') == 'absent')
        if absent_count > 0:
            absent_names = [w.get('name', '') for w in workers if w.get('status') == 'absent']
            self.status_bar.showMessage(f"⚠️ {absent_count} worker(s) absent: {', '.join(absent_names[:3])}{'...' if len(absent_names) > 3 else ''}")
        else:
            self.status_bar.showMessage("All workers present")
    
    def update_ui(self):
        """Update UI elements periodically."""
        # This method is called every second by the timer
        # It can be used to update time displays, check for alerts, etc.
        pass

    def add_alert(self, title: str, message: str, alert_type: str = "info") -> None:
        """Surface an alert in the Alerts tab (thread-safe)."""
        try:
            self.append_alert_signal.emit(title, message, alert_type)
        except Exception:
            # Best-effort fallback if emit fails
            if hasattr(self, 'alert_widget') and self.alert_widget:
                try:
                    self.alert_widget.add_alert_item(title, message, alert_type)
                except Exception:
                    pass
    
    def showEvent(self, event):  # type: ignore
        """Handle window show event - refresh worker status when window is first shown."""
        super().showEvent(event)
        # The parent application will handle refreshing worker status
        # This is just a hook in case needed
        logger.debug("UI window shown")
    
    def closeEvent(self, a0):  # type: ignore
        """Handle application close event."""
        self.update_timer.stop()
        super().closeEvent(a0)
        logger.info("UI: Application closed")


def test_ui():
    """Test function for the UIManager class."""
    import sys
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create sample config
    config = {
        "database": {
            "host": "localhost",
            "name": "floor_monitor",
            "user": "postgres",
            "password": "password",
            "port": 5432
        },
        "thresholds": {
            "warning_minutes": 15,
            "alert_minutes": 30
        },
        "notifications": {
            "enabled": True,
            "sound": True
        }
    }
    
    # Create and show UI
    ui = UIManager(config)
    
    # Add a test camera
    ui.add_camera(0)
    ui.update_camera_status(0, "Connected", "green")
    
    # Add test worker data
    test_workers = [
        {
            "worker_id": 1,
            "name": "John Doe",
            "status": "present",
            "time_present": 125.5,
            "last_seen": time.time()
        },
        {
            "worker_id": 2,
            "name": "Jane Smith",
            "status": "exceeded",
            "time_present": 1850.2,
            "last_seen": time.time()
        },
        {
            "worker_id": 3,
            "name": "Bob Johnson",
            "status": "absent",
            "time_present": 0,
            "last_seen": 0
        }
    ]
    
    ui.update_worker_status(test_workers)
    
    ui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    test_ui()