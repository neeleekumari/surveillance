"""
Floor Monitoring Desktop Application
----------------------------------
This application monitors worker presence using USB cameras and YOLOv8 for person detection.
"""
import sys
import json
import logging
import time
from pathlib import Path
from typing import Optional
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

# Import local modules
from database_module import DatabaseManager
from camera_manager import CameraManager
from detection_module import PersonDetector
from presence_tracker import PresenceTracker
from alert_manager import AlertManager
from ui_manager import UIManager
from config_manager import ConfigManager
from face_recognition_module import FaceRecognitionSystem, crop_face_from_detection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DetectionThread(QThread):
    """Thread for running object detection and face recognition to avoid blocking the UI."""
    detection_complete = pyqtSignal(list, int)  # detections, camera_id
    
    def __init__(self, detector: PersonDetector, camera_manager: CameraManager, face_system: Optional[FaceRecognitionSystem] = None):
        super().__init__()
        self.detector = detector
        self.camera_manager = camera_manager
        self.face_system = face_system
        self.running = True
        
    def run(self):
        """Main detection loop with face recognition."""
        while self.running:
            # Process frames from all cameras
            for camera_id in self.camera_manager.cameras:
                frame = self.camera_manager.get_frame(camera_id, timeout=0.1)
                if frame is not None:
                    detections = self.detector.detect(frame)
                    
                    # Perform face recognition on detected persons
                    if self.face_system and detections:
                        for det in detections:
                            try:
                                # Crop face from detection
                                face_img = crop_face_from_detection(frame, det.bbox)
                                
                                if face_img is not None:
                                    # Recognize worker
                                    result = self.face_system.recognize_worker(face_img)
                                    
                                    if result:
                                        worker_id, worker_name, similarity = result
                                        det.worker_id = worker_id
                                        det.worker_name = worker_name
                                        det.recognition_score = similarity
                            except Exception as e:
                                logger.error(f"Face recognition error: {e}")
                    
                    self.detection_complete.emit(detections, camera_id)
            # Small delay to prevent excessive CPU usage
            self.msleep(10)
    
    def stop(self):
        """Stop the detection thread."""
        self.running = False

class FloorMonitoringApp:
    """Main application class for the Floor Monitoring System."""
    
    def __init__(self):
        """Initialize the application."""
        self.app = QApplication(sys.argv)
        self.db: Optional[DatabaseManager] = None
        self.camera_manager: Optional[CameraManager] = None
        self.detector: Optional[PersonDetector] = None
        self.face_system: Optional[FaceRecognitionSystem] = None
        self.presence_tracker: Optional[PresenceTracker] = None
        self.alert_manager: Optional[AlertManager] = None
        self.ui: Optional[UIManager] = None
        self.detection_thread: Optional[DetectionThread] = None
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        
        # Initialize components
        self._initialize_components()
        
        # Set up application timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1000)  # Update every second
        
    def _load_config(self) -> dict:
        """Load application configuration."""
        return self.config_manager.load_config()
    
    def _initialize_components(self):
        """Initialize all application components."""
        try:
            # Initialize database connection (optional for now)
            try:
                self.db = DatabaseManager("../config/config.json")
                logger.info("Database connection established successfully.")
            except Exception as db_error:
                logger.warning(f"Database connection failed: {str(db_error)}")
                logger.warning("Application will run without database support.")
                self.db = None
            
            # Initialize camera manager
            camera_configs = self.config_manager.get_camera_configs()
            # Disable auto-detect for faster startup when configs are provided
            self.camera_manager = CameraManager(camera_configs, auto_detect=False)
            logger.info("Camera manager initialized.")
            
            # Initialize person detector
            self.detector = PersonDetector(conf_threshold=0.5, device='cpu')
            logger.info("Person detector initialized.")
            
            # Initialize face recognition system
            self.face_system = FaceRecognitionSystem(
                model_name="Facenet512",
                similarity_threshold=0.6,
                distance_metric="cosine"
            )
            logger.info("Face recognition system initialized.")
            
            # Initialize presence tracker
            thresholds = self.config_manager.get_thresholds()
            self.presence_tracker = PresenceTracker(thresholds)
            logger.info("Presence tracker initialized.")
            
            # Initialize alert manager
            notifications = self.config_manager.get_notification_config()
            self.alert_manager = AlertManager(notifications)
            logger.info("Alert manager initialized.")
            
            # Initialize UI
            self.ui = UIManager(self.config)
            self._connect_ui_signals()
            logger.info("UI manager initialized.")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            self.show_error("Initialization Error", f"Failed to initialize components: {str(e)}")
            sys.exit(1)
    
    def _connect_ui_signals(self):
        """Connect UI signals to application functions."""
        if self.ui:
            self.ui.start_camera_signal.connect(self.start_monitoring)
            self.ui.stop_camera_signal.connect(self.stop_monitoring)
            self.ui.settings_changed_signal.connect(self.update_settings)
            self.ui.register_worker_signal.connect(self.open_worker_registration)
    
    def start_monitoring(self):
        """Start camera monitoring and detection."""
        try:
            if self.camera_manager and not self.camera_manager.running:
                # Add cameras to UI
                if self.ui:
                    for camera_id in self.camera_manager.cameras:
                        self.ui.add_camera(camera_id)
                
                # Start cameras
                self.camera_manager.start()
                
                # Update UI status
                if self.ui:
                    for camera_id in self.camera_manager.cameras:
                        self.ui.update_camera_status(camera_id, "Running", "green")
                
                # Start detection thread (only if detector is available)
                if self.detector and self.camera_manager:
                    self.detection_thread = DetectionThread(
                        self.detector,
                        self.camera_manager,
                        self.face_system
                    )
                    self.detection_thread.detection_complete.connect(self.process_detections)
                    self.detection_thread.start()
                
                logger.info("Monitoring started successfully.")
                
        except Exception as e:
            logger.error(f"Failed to start monitoring: {str(e)}")
            self.show_error("Start Error", f"Failed to start monitoring: {str(e)}")
    
    def stop_monitoring(self):
        """Stop camera monitoring and detection."""
        try:
            # Stop detection thread
            if self.detection_thread and self.detection_thread.isRunning():
                self.detection_thread.stop()
                self.detection_thread.wait()
                self.detection_thread = None
            
            # Stop cameras
            if self.camera_manager:
                self.camera_manager.stop()
            
            # Update UI status
            if self.ui and self.camera_manager:
                for camera_id in self.camera_manager.cameras:
                    self.ui.update_camera_status(camera_id, "Stopped", "red")
            
            logger.info("Monitoring stopped successfully.")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
    
    def process_detections(self, detections, camera_id):
        """Process detections from a camera."""
        try:
            # Update UI with frame (even if no detections)
            if self.ui and self.camera_manager:
                frame = self.camera_manager.get_frame(camera_id)
                if frame is not None:
                    # Draw detections on frame if any
                    if detections and self.detector:
                        frame = self.detector.draw_detections(frame, detections)
                    self.ui.update_camera_frame(camera_id, frame)
            
            # Convert detections to worker format for presence tracker
            worker_detections = []
            for det in detections:
                # In a real implementation, you would identify workers here
                # For now, we'll use a simple approach
                worker_detections.append({
                    'worker_id': hash(str(det.bbox)) % 1000,  # Simple ID generation
                    'confidence': det.confidence
                })
            
            # Update presence tracker
            if self.presence_tracker:
                updates = self.presence_tracker.update_detections(worker_detections)
                
                # Update UI with worker status
                if self.ui:
                    worker_statuses = self.presence_tracker.get_all_statuses()
                    self.ui.update_worker_status(worker_statuses)
                
                # Check for alerts
                self.check_for_alerts(updates)
                
        except Exception as e:
            logger.error(f"Error processing detections: {str(e)}")
    
    def check_for_alerts(self, updates):
        """Check for and generate alerts based on worker status updates."""
        if not self.alert_manager:
            return
            
        try:
            for worker_id, status_data in updates.items():
                status = status_data['status']
                time_present = status_data['time_present']
                name = status_data['name']
                
                # Check for exceeded time thresholds
                warning_threshold = self.config_manager.get("thresholds.warning_minutes", 15) * 60
                alert_threshold = self.config_manager.get("thresholds.alert_minutes", 30) * 60
                
                if status == "exceeded" and time_present > alert_threshold:
                    self.alert_manager.add_alert(
                        f"Worker Time Alert: {name}",
                        f"Worker {name} (ID: {worker_id}) has been present for {int(time_present/60)} minutes",
                        "alert",
                        worker_id=worker_id,
                        duration=30
                    )
                elif status == "present" and time_present > warning_threshold:
                    self.alert_manager.add_alert(
                        f"Worker Time Warning: {name}",
                        f"Worker {name} (ID: {worker_id}) has been present for {int(time_present/60)} minutes",
                        "warning",
                        worker_id=worker_id,
                        duration=15
                    )
                    
        except Exception as e:
            logger.error(f"Error checking for alerts: {str(e)}")
    
    def update_settings(self, new_config):
        """Update application settings."""
        try:
            # Update config manager
            self.config_manager.config = new_config
            self.config = new_config
            
            # Update presence tracker thresholds
            if self.presence_tracker:
                thresholds = self.config_manager.get_thresholds()
                self.presence_tracker.warning_threshold = thresholds.get("warning_minutes", 15) * 60
                self.presence_tracker.alert_threshold = thresholds.get("alert_minutes", 30) * 60
            
            # Update alert manager settings
            if self.alert_manager:
                notifications = self.config_manager.get_notification_config()
                self.alert_manager.notifications_enabled = notifications.get("enabled", True)
                self.alert_manager.sound_enabled = notifications.get("sound", True)
            
            logger.info("Settings updated successfully.")
            
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            self.show_error("Settings Error", f"Failed to update settings: {str(e)}")
    
    def open_worker_registration(self):
        """Open the worker registration dialog."""
        try:
            from worker_registration_ui import WorkerRegistrationDialog
            
            dialog = WorkerRegistrationDialog(
                self.camera_manager,
                self.face_system,
                self.detector,
                self.ui
            )
            dialog.exec_()
            
        except Exception as e:
            logger.error(f"Error opening worker registration: {e}")
            self.show_error("Registration Error", f"Failed to open registration dialog: {str(e)}")
    
    def update(self):
        """Main update loop."""
        # This will be called every second
        # Can be used for periodic tasks
        pass
    
    def show_error(self, title: str, message: str):
        """Show an error message dialog."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
    
    def run(self) -> int:
        """Run the application."""
        try:
            logger.info("Starting Floor Monitoring Application")
            if self.ui:
                self.ui.show()
            return self.app.exec_()
        except Exception as e:
            logger.critical(f"Application error: {str(e)}", exc_info=True)
            self.show_error("Application Error", f"An unexpected error occurred: {str(e)}")
            return 1
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Close database connection
        if self.db:
            self.db.close()
        
        # Stop alert manager
        if self.alert_manager:
            self.alert_manager.stop()
        
        # Clean up UI
        if self.ui:
            self.ui.close()


def main():
    """Entry point for the application."""
    app = FloorMonitoringApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()