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
    QLineEdit, QMessageBox, QGroupBox, QFormLayout, QSpinBox,
    QFileDialog, QListWidget, QListWidgetItem, QTabWidget, QWidget
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import time
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class WorkerRegistrationDialog(QDialog):
    """Dialog for registering new workers with face capture."""
    
    def __init__(self, camera_manager, face_recognition_system, detector, parent=None):
        super().__init__(parent)
        self.camera_manager = camera_manager
        self.face_system = face_recognition_system
        self.detector = detector
        
        self.captured_faces: List[np.ndarray] = []
        self.target_face_count = 10  # Increased from 5 to 10 for better diversity
        self.current_frame = None
        self.cameras_were_running = False
        
        # Check if cameras are already running
        if self.camera_manager:
            self.cameras_were_running = self.camera_manager.running
        
        self.init_ui()
        
        # Timer for updating camera feed (optimized for lower latency)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_feed)
        self.timer.start(16)  # 16ms = ~60 FPS for smoother display
    
    def init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("Worker Registration & Management")
        self.setModal(True)
        self.resize(900, 700)
        
        layout = QVBoxLayout()
        
        # Create tabs for different modes
        tabs = QTabWidget()
        
        # Tab 1: Register from Camera
        camera_tab = QWidget()
        camera_layout = QVBoxLayout()
        self._setup_camera_tab(camera_layout)
        camera_tab.setLayout(camera_layout)
        
        # Tab 2: Register from Photos
        upload_tab = QWidget()
        upload_layout = QVBoxLayout()
        self._setup_upload_tab(upload_layout)
        upload_tab.setLayout(upload_layout)
        
        # Tab 3: Manage Workers
        manage_tab = QWidget()
        manage_layout = QVBoxLayout()
        self._setup_manage_tab(manage_layout)
        manage_tab.setLayout(manage_layout)
        
        tabs.addTab(camera_tab, "📷 Capture from Camera")
        tabs.addTab(upload_tab, "📁 Upload Photos")
        tabs.addTab(manage_tab, "👥 Manage Workers")
        
        layout.addWidget(tabs)
        self.setLayout(layout)
    
    def _setup_camera_tab(self, layout):
        """Setup camera capture tab."""
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
        
        # Camera selector
        from PyQt5.QtWidgets import QComboBox
        self.camera_selector = QComboBox()
        if self.camera_manager and self.camera_manager.cameras:
            for cam_id, cam_config in self.camera_manager.cameras.items():
                self.camera_selector.addItem(f"Camera {cam_id}: {cam_config.name}", cam_id)
        
        info_layout.addRow("Worker ID:", self.worker_id_input)
        info_layout.addRow("Name:", self.worker_name_input)
        info_layout.addRow("Position:", self.worker_position_input)
        info_layout.addRow("Camera:", self.camera_selector)
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
        
        # Add all widgets to layout
        layout.addWidget(info_group)
        layout.addWidget(self.camera_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_label)
        layout.addLayout(button_layout)
    
    def _setup_upload_tab(self, layout):
        """Setup photo upload tab."""
        # Worker information group
        info_group = QGroupBox("Worker Information")
        info_layout = QFormLayout()
        
        self.upload_worker_id_input = QSpinBox()
        self.upload_worker_id_input.setRange(1, 999999)
        self.upload_worker_id_input.setValue(int(time.time()) % 100000)
        
        self.upload_worker_name_input = QLineEdit()
        self.upload_worker_name_input.setPlaceholderText("Enter worker name")
        
        self.upload_worker_position_input = QLineEdit()
        self.upload_worker_position_input.setPlaceholderText("Enter position (optional)")
        
        info_layout.addRow("Worker ID:", self.upload_worker_id_input)
        info_layout.addRow("Name:", self.upload_worker_name_input)
        info_layout.addRow("Position:", self.upload_worker_position_input)
        info_group.setLayout(info_layout)
        
        # Instructions
        instructions = QLabel(
            "📌 Upload 3-5 high-quality photos of the person's face\n"
            "💡 Tips: Different angles, good lighting, clear face visibility\n"
            "⚠️  If worker ID already exists, photos will be REPLACED"
        )
        instructions.setStyleSheet("font-size: 12px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        
        # Upload button
        self.upload_button = QPushButton("📁 Select Photos (3-5 images)")
        self.upload_button.clicked.connect(self.select_photos)
        self.upload_button.setStyleSheet("font-size: 14px; padding: 10px;")
        
        # List of selected photos
        self.photo_list = QListWidget()
        self.photo_list.setMaximumHeight(200)
        
        # Register button
        self.upload_register_button = QPushButton("✓ Register Worker from Photos")
        self.upload_register_button.clicked.connect(self.register_from_photos)
        self.upload_register_button.setEnabled(False)
        self.upload_register_button.setStyleSheet("font-size: 14px; padding: 10px; background-color: #4CAF50; color: white;")
        
        # Add widgets
        layout.addWidget(info_group)
        layout.addWidget(instructions)
        layout.addWidget(self.upload_button)
        layout.addWidget(QLabel("Selected Photos:"))
        layout.addWidget(self.photo_list)
        layout.addWidget(self.upload_register_button)
    
    def _setup_manage_tab(self, layout):
        """Setup worker management tab."""
        # Instructions
        instructions = QLabel(
            "👥 Registered Workers - Select a worker and click 'Delete' to remove\n"
            "🔄 Refresh the list to see current registered workers"
        )
        instructions.setStyleSheet("font-size: 12px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        
        # Worker list
        self.worker_list = QListWidget()
        self.worker_list.setMinimumHeight(400)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        refresh_button = QPushButton("🔄 Refresh List")
        refresh_button.clicked.connect(self.refresh_worker_list)
        refresh_button.setStyleSheet("font-size: 14px; padding: 10px;")
        
        self.delete_button = QPushButton("🗑️ Delete Selected Worker")
        self.delete_button.clicked.connect(self.delete_worker)
        self.delete_button.setEnabled(False)
        self.delete_button.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f44336; color: white;")
        
        button_layout.addWidget(refresh_button)
        button_layout.addWidget(self.delete_button)
        
        # Enable delete button when worker selected
        self.worker_list.itemSelectionChanged.connect(
            lambda: self.delete_button.setEnabled(bool(self.worker_list.selectedItems()))
        )
        
        # Add widgets
        layout.addWidget(instructions)
        layout.addWidget(self.worker_list)
        layout.addLayout(button_layout)
        
        # Auto-refresh on tab show
        self.refresh_worker_list()
    
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
            
            # Get selected camera from dropdown
            camera_id = self.camera_selector.currentData()
            if camera_id is None:
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
        from src.face_recognition_module import crop_face_from_detection
        
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
        
        # Check if worker exists and confirm replacement
        if self._worker_exists(worker_id):
            reply = QMessageBox.question(
                self,
                "Worker Exists",
                f"Worker ID {worker_id} already exists. Replace with new photos?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
            
            # Delete old worker first
            self._delete_worker_by_id(worker_id)
        
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
    
    def select_photos(self):
        """Open file dialog to select photos."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Face Photos (3-5 images)",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        if not file_paths:
            return
        
        if len(file_paths) < 3:
            QMessageBox.warning(
                self,
                "Insufficient Photos",
                "Please select at least 3 photos for better accuracy"
            )
            return
        
        if len(file_paths) > 10:
            QMessageBox.warning(
                self,
                "Too Many Photos",
                "Please select maximum 10 photos. Using first 10 selected."
            )
            file_paths = file_paths[:10]
        
        # Clear previous selection
        self.photo_list.clear()
        
        # Add to list
        for path in file_paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.UserRole, path)  # Store full path
            self.photo_list.addItem(item)
        
        self.upload_register_button.setEnabled(True)
    
    def register_from_photos(self):
        """Register worker from uploaded photos."""
        # Validate inputs
        worker_name = self.upload_worker_name_input.text().strip()
        if not worker_name:
            QMessageBox.warning(self, "Invalid Input", "Please enter a worker name")
            return
        
        if self.photo_list.count() < 3:
            QMessageBox.warning(
                self,
                "Insufficient Photos",
                "Please select at least 3 photos"
            )
            return
        
        worker_id = self.upload_worker_id_input.value()
        
        # Check if worker exists and confirm replacement
        if self._worker_exists(worker_id):
            reply = QMessageBox.question(
                self,
                "Worker Exists",
                f"Worker ID {worker_id} already exists. Replace with new photos?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
            
            # Delete old worker first
            self._delete_worker_by_id(worker_id)
        
        # Load photos
        face_images = []
        failed_photos = []
        
        for i in range(self.photo_list.count()):
            item = self.photo_list.item(i)
            photo_path = item.data(Qt.UserRole)
            
            try:
                # Load image
                img = cv2.imread(photo_path)
                if img is None:
                    failed_photos.append(os.path.basename(photo_path))
                    continue
                
                # Detect face in image
                detections = self.detector.detect(img)
                
                if not detections:
                    failed_photos.append(f"{os.path.basename(photo_path)} (no face detected)")
                    continue
                
                if len(detections) > 1:
                    failed_photos.append(f"{os.path.basename(photo_path)} (multiple faces)")
                    continue
                
                # Crop face
                from src.face_recognition_module import crop_face_from_detection
                face_img = crop_face_from_detection(img, detections[0].bbox)
                
                if face_img is not None:
                    face_images.append(face_img)
                else:
                    failed_photos.append(f"{os.path.basename(photo_path)} (crop failed)")
                    
            except Exception as e:
                logger.error(f"Error processing photo {photo_path}: {e}")
                failed_photos.append(f"{os.path.basename(photo_path)} (error: {str(e)})")
        
        # Show warnings if some photos failed
        if failed_photos:
            QMessageBox.warning(
                self,
                "Some Photos Failed",
                f"Failed to process {len(failed_photos)} photo(s):\n" + "\n".join(failed_photos[:5]) +
                (f"\n... and {len(failed_photos)-5} more" if len(failed_photos) > 5 else "")
            )
        
        if len(face_images) < 3:
            QMessageBox.critical(
                self,
                "Insufficient Valid Photos",
                f"Only {len(face_images)} photos were successfully processed. Need at least 3."
            )
            return
        
        # Register worker
        try:
            success = self.face_system.register_worker(
                worker_id=worker_id,
                worker_name=worker_name,
                face_images=face_images
            )
            
            if success:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Worker '{worker_name}' registered successfully with {len(face_images)} photos!"
                )
                # Clear form
                self.photo_list.clear()
                self.upload_worker_name_input.clear()
                self.upload_register_button.setEnabled(False)
                self.refresh_worker_list()
            else:
                QMessageBox.critical(
                    self,
                    "Registration Failed",
                    "Failed to register worker. Please try again."
                )
        except Exception as e:
            logger.error(f"Error registering worker from photos: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred during registration: {str(e)}"
            )
    
    def refresh_worker_list(self):
        """Refresh the list of registered workers from database."""
        self.worker_list.clear()
        
        try:
            # Load workers from database
            if self.face_system and self.face_system.db_manager:
                workers_data = self.face_system.db_manager.get_all_face_embeddings()
                
                if not workers_data:
                    self.worker_list.addItem("No workers registered yet")
                    return
                
                # Add workers to list
                registered_faces_dir = Path("data/registered_faces")
                
                for worker in workers_data:
                    worker_id = worker['worker_id']
                    worker_name = worker['worker_name']
                    
                    # Count photos from database
                    with self.face_system.db_manager.conn.cursor() as cursor:
                        cursor.execute(
                            "SELECT COUNT(*) FROM face_photos WHERE worker_id = %s",
                            (worker_id,)
                        )
                        photo_count = cursor.fetchone()[0]
                    
                    item_text = f"ID: {worker_id} | Name: {worker_name} | Photos: {photo_count}"
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.UserRole, worker_id)  # Store worker ID
                    self.worker_list.addItem(item)
            else:
                self.worker_list.addItem("No workers registered yet")
                
        except Exception as e:
            logger.error(f"Error refreshing worker list: {e}")
            self.worker_list.addItem(f"Error loading workers: {str(e)}")
    
    def delete_worker(self):
        """Delete selected worker."""
        selected_items = self.worker_list.selectedItems()
        if not selected_items:
            return
        
        item = selected_items[0]
        worker_id = item.data(Qt.UserRole)
        
        if worker_id is None:
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete this worker?\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # Delete worker
        if self._delete_worker_by_id(worker_id):
            QMessageBox.information(self, "Success", "Worker deleted successfully")
            self.refresh_worker_list()
        else:
            QMessageBox.critical(self, "Error", "Failed to delete worker")
    
    def _worker_exists(self, worker_id):
        """Check if worker with given ID exists."""
        try:
            embeddings_file = Path("data/face_embeddings.pkl")
            if not embeddings_file.exists():
                return False
            
            import pickle
            with open(embeddings_file, 'rb') as f:
                workers_data = pickle.load(f)
            
            return any(w['worker_id'] == worker_id for w in workers_data)
        except:
            return False
    
    def _delete_worker_by_id(self, worker_id):
        """Delete worker by ID from database and file system."""
        try:
            # Step 1: Delete from database (this will cascade to embeddings, photos, and activity logs)
            if self.face_system and self.face_system.db_manager:
                success = self.face_system.db_manager.delete_worker(worker_id)
                if not success:
                    logger.warning(f"Worker {worker_id} not found in database")
                    # Continue anyway to clean up local files
            
            # Step 2: Delete from face recognition system's in-memory embeddings
            if self.face_system:
                self.face_system.delete_worker(worker_id)
                logger.info(f"Deleted worker {worker_id} from face recognition system memory")
            
            # Step 3: Delete face images directory
            worker_dir = Path(f"data/registered_faces/worker_{worker_id}")
            if worker_dir.exists():
                import shutil
                shutil.rmtree(worker_dir)
                logger.info(f"Deleted worker {worker_id} face images directory")
            
            # Step 4: Reload embeddings in face system to refresh the cache
            if self.face_system:
                self.face_system.embeddings_db = self.face_system._load_embeddings_db()
                self.face_system._build_embeddings_matrix()
                logger.info(f"Reloaded embeddings after deleting worker {worker_id}")
            
            logger.info(f"✅ Successfully deleted worker ID: {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting worker {worker_id}: {e}")
            return False
    
    def closeEvent(self, event):
        """Handle dialog close event."""
        self.timer.stop()
        
        # Stop cameras if they weren't running before we opened the dialog
        if self.camera_manager and not self.cameras_were_running and self.camera_manager.running:
            self.camera_manager.stop()
            logger.info("Stopped cameras after registration dialog closed")
        
        super().closeEvent(event)
