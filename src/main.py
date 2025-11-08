"""
Floor Monitoring Desktop Application (Stable Version)
-----------------------------------------------------
Modular real-time worker monitoring using YOLOv8, 3D Face Recognition, and PyQt5 UI.
Fully integrated with safety, error resilience, and proper thread management.
"""

import sys
import logging
import time
import numpy as np
from pathlib import Path
from typing import Optional
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer, QThread, pyqtSignal

# Project imports (relative-safe)
sys.path.append(str(Path(__file__).parent))
from src.database_module import DatabaseManager
from src.detection_module import PersonDetector
from src.camera_manager import CameraManager
from src.presence_tracker import PresenceTracker
from src.alert_manager import AlertManager
from src.ui_manager import UIManager
from src.config_manager import ConfigManager

try:
    from src.face_recognition_3d_module import EnhancedFaceRecognitionSystem as FaceRecognitionSystem
    from src.face_recognition_3d_module import crop_face_from_detection
    FACE_RECOGNITION_AVAILABLE = True
    FACE_RECOGNITION_ERROR = None
except Exception as e:
    FaceRecognitionSystem, crop_face_from_detection = None, None
    FACE_RECOGNITION_AVAILABLE = False
    FACE_RECOGNITION_ERROR = str(e)
    print(f"[WARNING] Face recognition module import failed: {e}")
    import traceback
    traceback.print_exc()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MainApp")


# --------------------------------------------------------------------
# Detection Thread
# --------------------------------------------------------------------
class DetectionThread(QThread):
    detection_complete = pyqtSignal(list, int)

    def __init__(self, detector, camera_manager, face_system=None):
        super().__init__()
        self.detector = detector
        self.camera_manager = camera_manager
        self.face_system = face_system
        self.running = True
        self.target_fps = 10  # Balanced speed
        self.interval = 1.0 / self.target_fps

    def run(self):
        """Main detection loop with safety guard."""
        logger.info("Detection thread started.")
        while self.running and not self.isInterruptionRequested():
            start_time = time.time()

            try:
                for cam_id in list(self.camera_manager.cameras.keys()):
                    frame = self.camera_manager.get_frame(cam_id)
                    if frame is None:
                        continue

                    detections = self.detector.detect(frame)

                    # Optional: perform recognition (simplified hook)
                    if self.face_system:
                        recognized_workers_in_frame = {}  # Track {worker_id: (similarity, detection_index)}
                        
                        for idx, det in enumerate(detections):
                            try:
                                face = crop_face_from_detection(frame, det.bbox)
                                if face is not None:
                                    result = self.face_system.recognize_worker(face)
                                    if result:
                                        worker_id, worker_name, similarity, debug_info = result
                                        
                                        # Anti-duplicate logic: One person cannot appear twice in same frame
                                        if worker_id > 0:  # Valid recognized worker (not Unknown)
                                            if worker_id in recognized_workers_in_frame:
                                                # Duplicate detected! Keep only the higher confidence match
                                                prev_similarity, prev_idx = recognized_workers_in_frame[worker_id]
                                                if similarity > prev_similarity:
                                                    # This detection is better, mark previous as Unknown
                                                    detections[prev_idx].worker_id = -1
                                                    detections[prev_idx].worker_name = "Unknown"
                                                    detections[prev_idx].recognition_score = prev_similarity
                                                    # Update tracking with current (better) detection
                                                    recognized_workers_in_frame[worker_id] = (similarity, idx)
                                                    det.worker_id = worker_id
                                                    det.worker_name = worker_name
                                                    det.recognition_score = similarity
                                                    logger.warning(
                                                        f"Duplicate {worker_name} detected in frame - kept best match "
                                                        f"(sim: {similarity:.3f} vs {prev_similarity:.3f})"
                                                    )
                                                else:
                                                    # Previous detection was better, mark this as Unknown
                                                    det.worker_id = -1
                                                    det.worker_name = "Unknown"
                                                    det.recognition_score = similarity
                                                    logger.warning(
                                                        f"Duplicate {worker_name} detected in frame - rejected "
                                                        f"(sim: {similarity:.3f} vs {prev_similarity:.3f})"
                                                    )
                                            else:
                                                # First occurrence of this worker in frame
                                                recognized_workers_in_frame[worker_id] = (similarity, idx)
                                                det.worker_id = worker_id
                                                det.worker_name = worker_name
                                                det.recognition_score = similarity
                                        else:
                                            # Unknown person or liveness failed
                                            det.worker_id = worker_id
                                            det.worker_name = worker_name
                                            det.recognition_score = similarity
                                    else:
                                        # If no embeddings exist yet, explicitly mark as Unknown for clarity
                                        if not getattr(self.face_system, 'embeddings_db', []):
                                            det.worker_id = -1
                                            det.worker_name = "Unknown"
                                            det.recognition_score = 0.0
                            except Exception as rec_err:
                                logger.debug(f"Recognition error: {rec_err}")

                    # Emit results back to main thread
                    self.detection_complete.emit(detections, cam_id)

            except Exception as e:
                logger.error(f"Detection loop error: {e}", exc_info=True)

            # Adaptive frame rate control
            elapsed = time.time() - start_time
            sleep_time = max(0.0, self.interval - elapsed)
            self.msleep(int(sleep_time * 1000))

        logger.info("Detection thread exiting...")

    def stop(self):
        self.running = False
        self.requestInterruption()


# --------------------------------------------------------------------
# Main Application Class
# --------------------------------------------------------------------
class FloorMonitoringApp:
    def __init__(self):
        # Reuse existing QApplication if already created (e.g., by runner), else create a new one
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()

        # Subsystems
        self.db: Optional[DatabaseManager] = None
        self.camera_manager: Optional[CameraManager] = None
        self.detector: Optional[PersonDetector] = None
        self.face_system: Optional[FaceRecognitionSystem] = None
        self.presence_tracker: Optional[PresenceTracker] = None
        self.alert_manager: Optional[AlertManager] = None
        self.ui: Optional[UIManager] = None
        self.detection_thread: Optional[DetectionThread] = None

        # Cross-camera exclusivity map: worker_id -> (camera_id, timestamp)
        # Ensures a worker cannot appear simultaneously in two different cameras.
        self.worker_camera_presence = {}
        self.cross_camera_suppression_window = 1.5  # seconds (tighter setups)

        # Initialize
        self._initialize_components()

        # Periodic UI timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1000)

    # ----------------------------------------------------------------
    def _initialize_components(self):
        try:
            # Ensure DB password present (prompt if missing)
            self._ensure_database_password()
            # Database
            try:
                self.db = DatabaseManager(None)
                logger.info("Database connected.")
            except Exception as e:
                logger.critical(f"FATAL: PostgreSQL connection failed: {e}")
                self.show_error(
                    "Database Connection Failed",
                    "PostgreSQL connection failed. The application requires a running database.\n\n"
                    "Please verify:\n"
                    "1. PostgreSQL service is running\n"
                    "2. Host/port/user/password in config/config.json are correct\n"
                    "3. Network/firewall allows local connection\n\n"
                    f"Error: {e}"
                )
                sys.exit(2)

            # Cameras
            cams = self.config_manager.get_camera_configs()
            self.camera_manager = CameraManager(cams, auto_detect=False)

            # YOLO Detector
            self.detector = PersonDetector(conf_threshold=0.45, iou_threshold=0.4, device="cpu")

            # Face Recognition System (use config settings)
            if not FACE_RECOGNITION_AVAILABLE:
                logger.error(f"Face Recognition NOT available - import failed: {FACE_RECOGNITION_ERROR}")
                self.face_system = None
            elif FaceRecognitionSystem:
                try:
                    fr_cfg = self.config_manager.get_face_recognition_config()
                    logger.info("Initializing Face Recognition System...")
                    self.face_system = FaceRecognitionSystem(
                        model_name=fr_cfg.get("model_name", "ArcFace"),
                        similarity_threshold=float(fr_cfg.get("similarity_threshold", 0.50)),
                        enable_3d=bool(fr_cfg.get("enable_3d", True)),
                        liveness_required=bool(fr_cfg.get("liveness_required", True))
                    )
                    logger.info(
                        f"Face Recognition initialized successfully | "
                        f"model={fr_cfg.get('model_name', 'ArcFace')} "
                        f"| 3D={bool(fr_cfg.get('enable_3d', True))} "
                        f"| liveness={bool(fr_cfg.get('liveness_required', True))} "
                        f"| threshold={float(fr_cfg.get('similarity_threshold', 0.50)):.2f}"
                    )
                except Exception as e:
                    logger.critical(f"Face Recognition initialization failed: {e}", exc_info=True)
                    self.show_error(
                        "Face Recognition Init Failed",
                        "Critical failure initializing face recognition.\n"
                        "Ensure required models and dependencies are installed (DeepFace, mediapipe, sklearn, torch).\n\n"
                        f"Error: {e}"
                    )
                    sys.exit(3)
            else:
                logger.error("FaceRecognitionSystem class is None despite successful import")
                self.face_system = None

            # Presence tracker
            self.presence_tracker = PresenceTracker(self.config_manager.get_thresholds())
            
            # UI (must be created BEFORE AlertManager because AlertManager needs ui_callback)
            self.ui = UIManager(self.config)
            
            # Alert manager (needs UI for callback)
            self.alert_manager = AlertManager(
                self.config_manager.get_notification_config(), 
                ui_callback=self.ui.log_alert
            )
            
            # Connect UI signals
            self._connect_ui_signals()
            
            # Load initial worker list (DB is mandatory now)
            self.refresh_worker_list()

            logger.info("All components initialized successfully.")
        except Exception as e:
            logger.critical(f"Initialization failed: {e}", exc_info=True)
            self.show_error("Initialization Error", str(e))
            sys.exit(1)

    # ----------------------------------------------------------------
    def _connect_ui_signals(self):
        """Connect UI actions to app logic."""
        if not self.ui:
            return
        self.ui.start_camera_signal.connect(self.start_monitoring)
        self.ui.stop_camera_signal.connect(self.stop_monitoring)
        self.ui.register_worker_signal.connect(self.open_worker_registration)
        self.ui.list_all_workers_signal.connect(self.refresh_worker_list)

    def _ensure_database_password(self):
        """Prompt user for DB password if missing and save config."""
        try:
            db_cfg = self.config_manager.get_database_config()
            if not db_cfg.get('password'):
                from PyQt5.QtWidgets import QInputDialog, QLineEdit
                pwd, ok = QInputDialog.getText(None, "PostgreSQL Password", "Enter password for user '" + db_cfg.get('user','postgres') + "':", echo=QLineEdit.Password)
                if not ok or not pwd:
                    raise RuntimeError("Database password is required.")
                self.config_manager.set('database.password', pwd)
                self.config_manager.save_config()
        except Exception as e:
            raise

    # ----------------------------------------------------------------
    def start_monitoring(self):
        """Launch detection + camera streams."""
        try:
            if not self.camera_manager.running:
                for cam_id in self.camera_manager.cameras:
                    self.ui.add_camera(cam_id)
                self.camera_manager.start()

            # Start detection thread
            if not self.detection_thread or not self.detection_thread.isRunning():
                self.detection_thread = DetectionThread(self.detector, self.camera_manager, self.face_system)
                self.detection_thread.detection_complete.connect(self.process_detections)
                self.detection_thread.start()

            logger.info("Monitoring started.")
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            self.show_error("Start Error", str(e))

    def stop_monitoring(self):
        """Stop detection thread and camera feeds."""
        try:
            if self.detection_thread:
                self.detection_thread.stop()
                self.detection_thread.wait()
                self.detection_thread = None
            if self.camera_manager:
                self.camera_manager.stop()
            logger.info("Monitoring stopped.")
        except Exception as e:
            logger.error(f"Stop error: {e}")

    # ----------------------------------------------------------------
    def process_detections(self, detections, camera_id):
        """Main-thread handler for results from DetectionThread."""
        try:
            now = time.time()
            # Enforce cross-camera exclusivity before drawing/UI updates
            for det in detections:
                wid = getattr(det, 'worker_id', None)
                if wid is not None and wid > 0:
                    prev = self.worker_camera_presence.get(wid)
                    if prev:
                        prev_cam, prev_ts = prev
                        # If same worker recognized in different camera within suppression window, demote to Unknown
                        if prev_cam != camera_id and (now - prev_ts) < self.cross_camera_suppression_window:
                            logger.warning(
                                f"Cross-camera duplicate suppressed: worker {wid} appeared in cam {prev_cam} and cam {camera_id} within {now - prev_ts:.2f}s"
                            )
                            det.worker_id = -1
                            det.worker_name = "Unknown"
                            continue
                    # Update presence map with latest camera assignment
                    self.worker_camera_presence[wid] = (camera_id, now)

            frame = self.camera_manager.get_frame(camera_id)
            if frame is not None:
                frame = self.detector.draw_detections(frame, detections)
                self.ui.update_camera_frame(camera_id, frame)

            # Update presence tracker and UI worker status table
            if self.presence_tracker:
                # Extract recognized worker IDs from detections (may be empty)
                recognized = []
                for det in detections:
                    wid = getattr(det, 'worker_id', None)
                    if wid is not None and wid >= 0:
                        recognized.append({'worker_id': wid})
                updates = self.presence_tracker.update_detections(recognized)
                # Transform updates for UI
                ui_rows = []
                for wid, info in updates.items():
                    ui_rows.append({
                        'worker_id': wid,
                        'name': info['name'],
                        'status': info['status'],
                        'time_present': info['session_time'],
                        'time_absent': info.get('absence_duration', 0.0),
                        'last_seen': info['last_seen']
                    })
                if ui_rows:
                    self.ui.update_worker_table(ui_rows)

                # Absence alert logic: alert if worker has been absent for >= 1 minute since last seen
                now = time.time()
                for wid, worker in self.presence_tracker.workers.items():
                    if worker.status == 'absent' and worker.absence_start > 0:
                        absence_duration = now - worker.absence_start
                        if absence_duration >= 60 and (now - worker.last_absence_alert_time) > 60:
                            if self.alert_manager:
                                minutes = int(absence_duration // 60)
                                self.alert_manager.add_alert(
                                    title="Worker Absence",
                                    message=f"{worker.name} not present for {minutes} minute(s)",
                                    alert_type="alert",
                                    worker_id=wid
                                )
                            worker.last_absence_alert_time = now
        except Exception as e:
            logger.debug(f"UI update error: {e}")

    # ----------------------------------------------------------------
    def open_worker_registration(self):
        """Launch 3D registration dialog."""
        try:
            # Check if face system is available
            if self.face_system is None:
                self.show_error(
                    "Face Recognition Not Available",
                    "Face recognition system failed to initialize.\n\n"
                    "Please check:\n"
                    "1. Python environment has all dependencies installed\n"
                    "2. Check logs for initialization errors\n"
                    "3. Try restarting the application"
                )
                return
            
            from src.worker_registration_3d_ui import Enhanced3DRegistrationDialog
            dlg = Enhanced3DRegistrationDialog(self.camera_manager, self.face_system, self.detector, self.ui)
            dlg.exec_()
        except Exception as e:
            logger.error(f"Registration dialog error: {e}", exc_info=True)
            self.show_error("Registration Error", str(e))
    
    def refresh_worker_list(self):
        """Refresh the list of all registered workers in the UI."""
        try:
            if not self.db:
                logger.warning("Database not available for worker list refresh")
                return
            
            # Get all workers from database
            workers = self.db.get_all_workers()
            
            # For each worker, get embedding and photo counts
            worker_list = []
            for worker in workers:
                worker_id = worker['worker_id']
                worker_name = worker.get('name', worker.get('worker_name', 'Unknown'))  # Handle both field names
                
                # Count embeddings
                embeddings = self.db.get_all_face_embeddings()
                embedding_count = sum(1 for e in embeddings if e['worker_id'] == worker_id)
                
                # Count photos
                photos = self.db.get_face_photos(worker_id)
                photo_count = len(photos) if photos else 0
                
                worker_list.append({
                    'worker_id': worker_id,
                    'worker_name': worker_name,
                    'embedding_count': embedding_count,
                    'photo_count': photo_count
                })
                # Seed presence tracker with all registered workers so they appear as absent initially
                if self.presence_tracker:
                    self.presence_tracker.add_worker(worker_id, worker_name)
            
            # Update UI
            self.ui.update_registered_workers(worker_list)
            logger.info(f"Refreshed worker list: {len(worker_list)} workers")
            
        except Exception as e:
            logger.error(f"Failed to refresh worker list: {e}", exc_info=True)

    # ----------------------------------------------------------------
    def update(self):
        """Periodic updates (UI, alerts, etc.)."""
        try:
            if self.presence_tracker:
                statuses = self.presence_tracker.get_all_statuses()
                if statuses and self.ui:
                    ui_rows = []
                    for info in statuses:
                        ui_rows.append({
                            'worker_id': info['worker_id'],
                            'name': info['name'],
                            'status': info['status'],
                            'time_present': info['session_time'],
                            'time_absent': info.get('absence_duration', 0.0),
                            'last_seen': info['last_seen']
                        })
                    self.ui.update_worker_table(ui_rows)

                # Check absence alerts here as well
                now = time.time()
                for wid, worker in self.presence_tracker.workers.items():
                    if worker.status == 'absent' and worker.absence_start > 0:
                        absence_duration = now - worker.absence_start
                        if absence_duration >= 60 and (now - worker.last_absence_alert_time) > 60:
                            if self.alert_manager:
                                minutes = int(absence_duration // 60)
                                self.alert_manager.add_alert(
                                    title="Worker Absence",
                                    message=f"{worker.name} not present for {minutes} minute(s)",
                                    alert_type="alert",
                                    worker_id=wid
                                )
                            worker.last_absence_alert_time = now
        except Exception as e:
            logger.debug(f"Periodic update error: {e}")

    # ----------------------------------------------------------------
    def show_error(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()

    # ----------------------------------------------------------------
    def run(self):
        """Main loop."""
        try:
            logger.info("Application starting...")
            self.ui.show()
            return self.app.exec_()
        finally:
            self.cleanup()

    def cleanup(self):
        """Full cleanup."""
        logger.info("Shutting down gracefully...")
        try:
            self.stop_monitoring()
            if self.alert_manager:
                self.alert_manager.stop()
            if self.db:
                self.db.close()
        except Exception as e:
            logger.warning(f"Cleanup issue: {e}")


# --------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------
def main():
    app = FloorMonitoringApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
