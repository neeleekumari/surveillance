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
                             QDoubleSpinBox, QFormLayout, QDialog, QListWidget, QLineEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QColor
import time

logger = logging.getLogger(__name__)

class CameraWidget(QWidget):
    """Widget for displaying camera feed."""
    
    def __init__(self, camera_id: int, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.init_ui()
        
    def init_ui(self):
        """Initialize the camera widget UI."""
        layout = QVBoxLayout()
        
        # Camera label
        self.label = QLabel(f"Camera {self.camera_id}")
        self.label.setAlignment(Qt.AlignCenter)  # type: ignore
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.label.setFont(font)
        
        # Video display
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)  # type: ignore
        self.video_display.setMinimumSize(640, 480)
        self.video_display.setStyleSheet("background-color: black;")
        
        # Status bar
        self.status_label = QLabel("Status: Disconnected")
        self.status_label.setAlignment(Qt.AlignCenter)  # type: ignore
        
        layout.addWidget(self.label)
        layout.addWidget(self.video_display)
        layout.addWidget(self.status_label)
        self.setLayout(layout)
    
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
        """Update the status label."""
        self.status_label.setText(f"Status: {status}")
        if color == "red":
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        elif color == "green":
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: black;")


class WorkerStatusWidget(QWidget):
    """Widget for displaying worker status information."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the worker status widget UI."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Worker Status")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.AlignCenter)  # type: ignore
        
        # Worker table
        self.worker_table = QTableWidget()
        self.worker_table.setColumnCount(5)
        self.worker_table.setHorizontalHeaderLabels([
            "Worker ID", "Name", "Status", "Time Present", "Last Seen"
        ])
        self.worker_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.worker_table.setSelectionBehavior(QTableWidget.SelectRows)
        header = self.worker_table.horizontalHeader()
        if header:
            header.setStretchLastSection(True)
        
        layout.addWidget(title)
        layout.addWidget(self.worker_table)
        self.setLayout(layout)
    
    def update_workers(self, workers: List[Dict]):
        """Update the worker table with new data."""
        self.worker_table.setRowCount(len(workers))
        
        for row, worker in enumerate(workers):
            # Worker ID
            item = QTableWidgetItem(str(worker.get('worker_id', '')))
            item.setTextAlignment(Qt.AlignCenter)  # type: ignore
            self.worker_table.setItem(row, 0, item)
            
            # Name
            item = QTableWidgetItem(worker.get('name', ''))
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # type: ignore
            self.worker_table.setItem(row, 1, item)
            
            # Status
            status = worker.get('status', 'unknown')
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
            
            # Last Seen
            last_seen = worker.get('last_seen', 0)
            if last_seen > 0:
                dt = datetime.fromtimestamp(last_seen)
                time_str = dt.strftime("%H:%M:%S")
            else:
                time_str = "Never"
            item = QTableWidgetItem(time_str)
            item.setTextAlignment(Qt.AlignCenter)  # type: ignore
            self.worker_table.setItem(row, 4, item)
        
        # Resize columns to fit content
        self.worker_table.resizeColumnsToContents()


class AlertWidget(QWidget):
    """Widget for displaying alerts and notifications."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the alert widget UI."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Alerts & Notifications")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.AlignCenter)  # type: ignore
        
        # Alert list
        self.alert_list = QListWidget()
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.clear_button = QPushButton("Clear Alerts")
        self.acknowledge_button = QPushButton("Acknowledge Selected")
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.acknowledge_button)
        button_layout.addStretch()
        
        layout.addWidget(title)
        layout.addWidget(self.alert_list)
        layout.addLayout(button_layout)
        self.setLayout(layout)


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
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self.config = config or {}
        self.cameras = {}
        self.init_ui()
        
    def init_ui(self):
        """Initialize the main UI."""
        self.setWindowTitle("Floor Monitoring System")
        self.setGeometry(100, 100, 1200, 800)
        
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
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Set up timer for UI updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(1000)  # Update every second
    
    def create_toolbar(self):
        """Create the application toolbar."""
        toolbar = self.addToolBar("Main")
        if toolbar is not None:
            toolbar.setMovable(False)
            
            # Start action
            self.start_action = QAction(QIcon(), "Start", self)
            self.start_action.setStatusTip("Start camera monitoring")
            self.start_action.triggered.connect(self.start_monitoring)
            toolbar.addAction(self.start_action)
            
            # Stop action
            self.stop_action = QAction(QIcon(), "Stop", self)
            self.stop_action.setStatusTip("Stop camera monitoring")
            self.stop_action.triggered.connect(self.stop_monitoring)
            toolbar.addAction(self.stop_action)
            
            # Add separator
            toolbar.addSeparator()
            
            # Register Worker action
            self.register_action = QAction(QIcon(), "Register Worker", self)
            self.register_action.setStatusTip("Register a new worker")
            self.register_action.triggered.connect(self.register_worker)
            toolbar.addAction(self.register_action)
            
            # Add separator
            toolbar.addSeparator()
            
            # Settings action
            self.settings_action = QAction(QIcon(), "Settings", self)
            self.settings_action.setStatusTip("Open settings dialog")
            self.settings_action.triggered.connect(self.open_settings)
            toolbar.addAction(self.settings_action)
            
            # Add separator
            toolbar.addSeparator()
            
            # Exit action
            self.exit_action = QAction(QIcon(), "Exit", self)
            self.exit_action.setStatusTip("Exit application")
            self.exit_action.triggered.connect(self.close)  # type: ignore
            toolbar.addAction(self.exit_action)
    
    def start_monitoring(self):
        """Start camera monitoring."""
        self.start_camera_signal.emit()
        self.status_bar.showMessage("Monitoring started")
        logger.info("UI: Start monitoring requested")
    
    def stop_monitoring(self):
        """Stop camera monitoring."""
        self.stop_camera_signal.emit()
        self.status_bar.showMessage("Monitoring stopped")
        logger.info("UI: Stop monitoring requested")
    
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
    
    def update_ui(self):
        """Update UI elements periodically."""
        # This method is called every second by the timer
        # It can be used to update time displays, check for alerts, etc.
        pass
    
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