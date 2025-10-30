"""
Worker Registration UI
----------------------
PyQt5 interface for registering workers with face recognition.
"""
import cv2
import numpy as np
import logging
from typing import List, Optional
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QMessageBox, QGroupBox, QFormLayout, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import time

logger = logging.getLogger(__name__)


class WorkerRegistrationDialog(QDialog):
    """Dialog for registering new workers with face capture."""
    
    def __init__(self, camera_manager, face_recognition_system, detector, parent=None):
        super().__init__(parent)
        self.camera_manager = camera_manager
        self.face_system = face_recognition_system
        self.detector = detector
        
        self.captured_faces: List[np.ndarray] = []
        self.target_face_count = 5
        self.current_frame = None
        self.cameras_were_running = False
        
        # Check if cameras are already running
        if self.camera_manager:
            self.cameras_were_running = self.camera_manager.running
        
        self.init_ui()
        
        # Timer for updating camera feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_feed)
        self.timer.start(30)  # 30ms = ~33 FPS
    
    def init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("Register New Worker")
        self.setModal(True)
        self.resize(800, 600)
        
        layout = QVBoxLayout()
        
        # Worker information group
        info_group = QGroupBox("Worker Information")
        info_layout = QFormLayout()
        
        self.worker_id_input = QSpinBox()
        self.worker_id_input.setRange(1, 999999)
        self.worker_id_input.setValue(int(time.time()) % 100000)  # Auto-generate ID
        
        self.worker_name_input = QLineEdit()
        self.worker_name_input.setPlaceholderText("Enter worker name")
        
        self.worker_position_input = QLineEdit()
        self.worker_position_input.setPlaceholderText("Enter position (optional)")
        
        info_layout.addRow("Worker ID:", self.worker_id_input)
        info_layout.addRow("Name:", self.worker_name_input)
        info_layout.addRow("Position:", self.worker_position_input)
        info_group.setLayout(info_layout)
        
        # Camera feed
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black;")
        
        # Status label
        self.status_label = QLabel("Position your face in the frame and click 'Capture Face'")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; padding: 10px;")
        
        # Progress label
        self.progress_label = QLabel(f"Captured: 0/{self.target_face_count} faces")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.capture_button = QPushButton("Capture Face")
        self.capture_button.clicked.connect(self.capture_face)
        
        self.register_button = QPushButton("Register Worker")
        self.register_button.clicked.connect(self.register_worker)
        self.register_button.setEnabled(False)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.register_button)
        button_layout.addWidget(self.cancel_button)
        
        # Add all widgets to main layout
        layout.addWidget(info_group)
        layout.addWidget(self.camera_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_label)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def update_camera_feed(self):
        """Update the camera feed display."""
        try:
            # Check if camera manager exists and has cameras
            if not self.camera_manager or not self.camera_manager.cameras:
                self.status_label.setText("No cameras available. Please configure cameras first.")
                return
            
            # Start cameras if not running
            if not self.camera_manager.running:
                self.camera_manager.start()
                self.status_label.setText("Starting cameras... Please wait.")
                return
            
            camera_id = list(self.camera_manager.cameras.keys())[0]
            frame = self.camera_manager.get_frame(camera_id, timeout=0.1)
            
            if frame is None:
                return
            
            self.current_frame = frame.copy()
            
            # Detect faces in frame
            detections = self.detector.detect(frame)
            
            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = map(int, det.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Face detected",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
            
            # Convert to QPixmap and display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            scaled_pixmap = pixmap.scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.camera_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            logger.error(f"Error updating camera feed: {e}")
    
    def capture_face(self):
        """Capture a face from the current frame."""
        if self.current_frame is None:
            QMessageBox.warning(self, "Error", "No camera feed available")
            return
        
        # Detect faces
        detections = self.detector.detect(self.current_frame)
        
        if not detections:
            QMessageBox.warning(self, "No Face Detected", "Please position your face in the frame")
            return
        
        if len(detections) > 1:
            QMessageBox.warning(
                self,
                "Multiple Faces",
                "Multiple faces detected. Please ensure only one person is in frame"
            )
            return
        
        # Crop face from detection
        from face_recognition_module import crop_face_from_detection
        
        face_img = crop_face_from_detection(self.current_frame, detections[0].bbox)
        
        if face_img is None:
            QMessageBox.warning(self, "Error", "Failed to crop face from frame")
            return
        
        # Add to captured faces
        self.captured_faces.append(face_img)
        
        # Update progress
        self.progress_label.setText(f"Captured: {len(self.captured_faces)}/{self.target_face_count} faces")
        
        # Update status
        if len(self.captured_faces) < self.target_face_count:
            self.status_label.setText(
                f"Good! Capture {self.target_face_count - len(self.captured_faces)} more face(s). "
                "Try different angles."
            )
        else:
            self.status_label.setText("All faces captured! Click 'Register Worker' to complete registration.")
            self.register_button.setEnabled(True)
            self.capture_button.setEnabled(False)
    
    def register_worker(self):
        """Register the worker with captured faces."""
        # Validate inputs
        worker_name = self.worker_name_input.text().strip()
        if not worker_name:
            QMessageBox.warning(self, "Invalid Input", "Please enter a worker name")
            return
        
        if len(self.captured_faces) < 3:
            QMessageBox.warning(
                self,
                "Insufficient Faces",
                "Please capture at least 3 face images"
            )
            return
        
        worker_id = self.worker_id_input.value()
        
        # Register with face recognition system
        try:
            success = self.face_system.register_worker(
                worker_id=worker_id,
                worker_name=worker_name,
                face_images=self.captured_faces
            )
            
            if success:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Worker '{worker_name}' registered successfully with {len(self.captured_faces)} face images!"
                )
                self.accept()
            else:
                QMessageBox.critical(
                    self,
                    "Registration Failed",
                    "Failed to register worker. Please try again."
                )
        except Exception as e:
            logger.error(f"Error registering worker: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred during registration: {str(e)}"
            )
    
    def closeEvent(self, event):
        """Handle dialog close event."""
        self.timer.stop()
        
        # Stop cameras if they weren't running before we opened the dialog
        if self.camera_manager and not self.cameras_were_running and self.camera_manager.running:
            self.camera_manager.stop()
            logger.info("Stopped cameras after registration dialog closed")
        
        super().closeEvent(event)
