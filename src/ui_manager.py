"""
UI Manager Module (Enhanced + Integrated)
----------------------------------------
Manages the main PyQt5 GUI for the real-time 3D Face Recognition
Floor Monitoring System.
"""

import sys
import logging
import time
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
    QTextEdit, QStatusBar, QToolBar, QAction, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QThread
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QColor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------------------------------------------------------
# Helper: Safe frame rendering widget
# ----------------------------------------------------------------------
class CameraWidget(QWidget):
    """Widget for displaying live camera feed."""

    def __init__(self, camera_id: int, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.frame = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        self.label = QLabel(f"Camera {self.camera_id}")
        self.label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.label.setFont(font)

        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumSize(640, 480)
        self.video_display.setStyleSheet("background-color: black; border:2px solid #aaa;")

        self.status_label = QLabel("Status: Disconnected", alignment=Qt.AlignCenter)
        layout.addWidget(self.label)
        layout.addWidget(self.video_display)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

    def update_frame(self, frame: np.ndarray):
        """Safely update camera preview."""
        if frame is None or not isinstance(frame, np.ndarray):
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(
                self.video_display.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_display.setPixmap(scaled)
        except Exception as e:
            logger.debug(f"Frame update error (Cam {self.camera_id}): {e}")

    def set_status(self, status: str, color: str = "black"):
        """Update camera connection status."""
        colors = {"red": "#ff4444", "green": "#4CAF50", "yellow": "#FFC107"}
        css_color = colors.get(color, "#000")
        self.status_label.setStyleSheet(f"color:{css_color}; font-weight:bold;")
        self.status_label.setText(f"Status: {status}")


# ----------------------------------------------------------------------
# Main UI Manager
# ----------------------------------------------------------------------
class UIManager(QMainWindow):
    """Main control panel for 3D monitoring system."""

    # Signal definitions
    start_camera_signal = pyqtSignal()
    stop_camera_signal = pyqtSignal()
    register_worker_signal = pyqtSignal()
    settings_changed_signal = pyqtSignal(dict)
    delete_worker_signal = pyqtSignal(int)
    view_worker_signal = pyqtSignal(int)
    list_all_workers_signal = pyqtSignal()

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self.config = config or {}
        self.cameras: Dict[int, CameraWidget] = {}
        self.monitoring_active = False
        self._init_ui()

    # ------------------------------------------------------------------
    def _init_ui(self):
        self.setWindowTitle("Floor Monitoring - 3D Face Recognition System")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("QMainWindow{background-color:#f5f5f5;} QLabel{font-size:11pt;}")

        # Central layout
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Cameras tab
        self.camera_tab = QWidget()
        self.camera_layout = QHBoxLayout(self.camera_tab)
        self.tabs.addTab(self.camera_tab, "Cameras")

        # Worker tab (simplified)
        self.worker_tab = QWidget()
        self.worker_layout = QVBoxLayout(self.worker_tab)
        
        # Add refresh button for worker list
        worker_header_layout = QHBoxLayout()
        self.refresh_workers_btn = QPushButton("🔄 Refresh Worker List")
        self.refresh_workers_btn.clicked.connect(lambda: self.list_all_workers_signal.emit())
        worker_header_layout.addWidget(self.refresh_workers_btn)
        worker_header_layout.addStretch()
        self.worker_layout.addLayout(worker_header_layout)
        
        # Presence-oriented table: ID, Name, Status, Present (mm:ss), Absent (mm:ss), Last Seen
        self.worker_table = QTableWidget(0, 6)
        self.worker_table.setHorizontalHeaderLabels(["ID", "Name", "Status", "Present", "Absent", "Last Seen"])
        self.worker_table.horizontalHeader().setStretchLastSection(True)
        self.worker_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.worker_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.worker_layout.addWidget(self.worker_table)
        self.tabs.addTab(self.worker_tab, "Workers")

        # Alerts tab
        self.alert_tab = QWidget()
        self.alert_layout = QVBoxLayout(self.alert_tab)
        self.alert_log = QTextEdit(readOnly=True)
        self.alert_layout.addWidget(self.alert_log)
        self.tabs.addTab(self.alert_tab, "Alerts")

        # Toolbar
        self._create_toolbar()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_indicator = QLabel("⚫ Stopped", alignment=Qt.AlignCenter)
        self.status_indicator.setStyleSheet(
            "padding:6px 10px; background-color:#ff6b6b; color:white; border-radius:4px; font-weight:bold;"
        )
        self.status_bar.addPermanentWidget(self.status_indicator)
        self.status_bar.showMessage("Ready")

        # Periodic UI refresh
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self._refresh_ui)
        self.ui_timer.start(1000)

    # ------------------------------------------------------------------
    def _create_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setIconSize(QSize(28, 28))
        tb.setMovable(False)

        self.start_action = QAction("▶ Start", self)
        self.stop_action = QAction("⏹ Stop", self)
        self.register_action = QAction("➕ Register", self)
        self.settings_action = QAction("⚙ Settings", self)
        self.exit_action = QAction("✖ Exit", self)

        tb.addAction(self.start_action)
        tb.addAction(self.stop_action)
        tb.addSeparator()
        tb.addAction(self.register_action)
        tb.addSeparator()
        tb.addAction(self.settings_action)
        tb.addSeparator()
        tb.addAction(self.exit_action)

        # Connect actions
        self.start_action.triggered.connect(self.start_monitoring)
        self.stop_action.triggered.connect(self.stop_monitoring)
        self.register_action.triggered.connect(lambda: self.register_worker_signal.emit())
        self.settings_action.triggered.connect(self._open_settings)
        self.exit_action.triggered.connect(self.close)

        self.stop_action.setEnabled(False)

    # ------------------------------------------------------------------
    def add_camera(self, camera_id: int):
        """Add a camera feed pane."""
        if camera_id not in self.cameras:
            widget = CameraWidget(camera_id)
            self.cameras[camera_id] = widget
            self.camera_layout.addWidget(widget)
            widget.set_status("Connected", "green")

    def update_camera_frame(self, camera_id: int, frame: np.ndarray):
        """Thread-safe UI frame update."""
        if camera_id in self.cameras:
            self.cameras[camera_id].update_frame(frame)

    def update_camera_status(self, camera_id: int, status: str, color: str = "black"):
        """Update camera connection indicator."""
        if camera_id in self.cameras:
            self.cameras[camera_id].set_status(status, color)

    def log_alert(self, text: str, color: str = "red", level: str = "info"):
        """Append an alert to alert log (compatible with AlertManager callback)."""
        ts = datetime.now().strftime("%H:%M:%S")
        self.alert_log.append(f"<span style='color:{color}'><b>[{ts}]</b> {text}</span>")
    
    def update_registered_workers(self, workers: List[Dict]):
        """Display registered workers with default absent status."""
        self.worker_table.setRowCount(len(workers))
        for r, w in enumerate(workers):
            self.worker_table.setItem(r, 0, QTableWidgetItem(str(w.get("worker_id", ""))))
            self.worker_table.setItem(r, 1, QTableWidgetItem(w.get("worker_name", "")))
            status_item = QTableWidgetItem("absent")
            status_item.setForeground(QColor("gray"))
            self.worker_table.setItem(r, 2, status_item)
            self.worker_table.setItem(r, 3, QTableWidgetItem("00:00"))
            self.worker_table.setItem(r, 4, QTableWidgetItem("00:00"))
            self.worker_table.setItem(r, 5, QTableWidgetItem("-"))

    def update_worker_table(self, workers: List[Dict]):
        """Refresh worker presence table."""
        self.worker_table.setRowCount(len(workers))
        for r, w in enumerate(workers):
            self.worker_table.setItem(r, 0, QTableWidgetItem(str(w.get("worker_id", ""))))
            self.worker_table.setItem(r, 1, QTableWidgetItem(w.get("name", "")))
            status = w.get("status", "unknown")
            item = QTableWidgetItem(status)
            if status == "present":
                item.setForeground(QColor("green"))
            elif status == "exceeded":
                item.setForeground(QColor("red"))
            self.worker_table.setItem(r, 2, item)
            secs = int(w.get("time_present", 0))
            mins, sec = divmod(secs, 60)
            self.worker_table.setItem(r, 3, QTableWidgetItem(f"{mins:02d}:{sec:02d}"))
            a_secs = int(w.get("time_absent", 0))
            a_mins, a_sec = divmod(a_secs, 60)
            self.worker_table.setItem(r, 4, QTableWidgetItem(f"{a_mins:02d}:{a_sec:02d}"))
            t = w.get("last_seen", 0)
            t_str = datetime.fromtimestamp(t).strftime("%H:%M:%S") if t else "-"
            self.worker_table.setItem(r, 5, QTableWidgetItem(t_str))

    # ------------------------------------------------------------------
    def start_monitoring(self):
        """Signal backend to start monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.start_camera_signal.emit()
            self.status_bar.showMessage("Monitoring started.")
            self.status_indicator.setText("🟢 Running")
            self.status_indicator.setStyleSheet(
                "padding:6px 10px; background-color:#4CAF50; color:white; border-radius:4px;"
            )
            self.start_action.setEnabled(False)
            self.stop_action.setEnabled(True)

    def stop_monitoring(self):
        """Signal backend to stop monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            self.stop_camera_signal.emit()
            self.status_bar.showMessage("Monitoring stopped.")
            self.status_indicator.setText("⚫ Stopped")
            self.status_indicator.setStyleSheet(
                "padding:6px 10px; background-color:#ff6b6b; color:white; border-radius:4px;"
            )
            self.start_action.setEnabled(True)
            self.stop_action.setEnabled(False)

    def _open_settings(self):
        """Placeholder for settings dialog."""
        QMessageBox.information(self, "Settings", "Settings dialog coming soon...")

    def _refresh_ui(self):
        """Periodic visual updates."""
        if self.monitoring_active:
            active = len(self.cameras)
            self.status_bar.showMessage(f"Running | Active Cameras: {active}")
        else:
            self.status_bar.showMessage("Idle")

    def closeEvent(self, event):
        """Graceful shutdown."""
        try:
            self.ui_timer.stop()
        except Exception:
            pass
        self.stop_camera_signal.emit()
        logger.info("UI closed safely.")
        super().closeEvent(event)