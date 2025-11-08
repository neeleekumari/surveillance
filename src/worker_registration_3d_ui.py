"""
Enhanced Worker Registration UI (Stable)
----------------------------------------
Multi-view 3D face registration dialog:
- Captures 7 angles for 3D depth/mesh features.
- Displays real-time YOLO detections + liveness feedback.
- Safe threading & error recovery for PyQt5.
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Optional
from pathlib import Path
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QMessageBox, QGroupBox, QFormLayout, QSpinBox,
    QProgressBar, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --------------------------------------------------------------------
# Utility safe import
# --------------------------------------------------------------------
def import_from_project(module_name: str, obj_name: Optional[str] = None):
    """
    Dynamically import modules to remain path-safe.
    """
    try:
        import importlib.util
        root = Path(__file__).resolve().parent
        if (root / f"{module_name}.py").exists():
            spec = importlib.util.spec_from_file_location(module_name, root / f"{module_name}.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, obj_name) if obj_name else mod
        return __import__(module_name, fromlist=[obj_name]) if obj_name else __import__(module_name)
    except Exception as e:
        logger.error(f"Import failed: {module_name} -> {e}")
        return None


# --------------------------------------------------------------------
# Registration dialog
# --------------------------------------------------------------------
class Enhanced3DRegistrationDialog(QDialog):
    """3D registration dialog with multi-angle capture and live liveness."""

    def __init__(self, camera_manager, face_recognition_system, detector, parent=None):
        super().__init__(parent)
        self.camera_manager = camera_manager
        self.face_system = face_recognition_system
        self.detector = detector

        self.captured_faces: List[np.ndarray] = []
        self.target_face_count = 7
        self.current_frame: Optional[np.ndarray] = None
        self.cameras_were_running = getattr(self.camera_manager, "running", False)

        self._init_ui()
        self._setup_timers()

    # ---------------- UI -----------------
    def _init_ui(self):
        self.setWindowTitle("3D Face Registration – Multi-View Capture")
        self.resize(1000, 800)
        layout = QVBoxLayout()

        # Worker info group
        info_group = QGroupBox("Worker Information")
        form = QFormLayout()
        self.worker_id_input = QSpinBox()
        self.worker_id_input.setRange(1, 999999)
        self.worker_id_input.setValue(int(time.time()) % 100000)
        self.worker_name_input = QLineEdit(placeholderText="Enter worker name")

        self.camera_selector = QComboBox()
        if self.camera_manager and getattr(self.camera_manager, "cameras", None):
            for cam_id, cam_cfg in self.camera_manager.cameras.items():
                self.camera_selector.addItem(f"Camera {cam_id}: {getattr(cam_cfg, 'name', '')}", cam_id)

        self.enable_3d_checkbox = QCheckBox("Enable 3D Features (Depth + Mesh)")
        self.enable_3d_checkbox.setChecked(True)

        form.addRow("Worker ID:", self.worker_id_input)
        form.addRow("Name:", self.worker_name_input)
        form.addRow("Camera:", self.camera_selector)
        form.addRow("", self.enable_3d_checkbox)
        info_group.setLayout(form)

        # Camera preview
        self.camera_label = QLabel(alignment=Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background-color:black; border:2px solid #4CAF50;")

        # Status / guidance
        self.guidance_label = QLabel("📸 Multi-View Capture Guide", alignment=Qt.AlignCenter)
        self.guidance_label.setStyleSheet("font-size:16px; font-weight:bold; padding:8px; background:#2196F3; color:white;")

        self.status_label = QLabel("Position your face FRONTALLY and click 'Capture'", alignment=Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size:12px; padding:8px;")

        self.progress_bar = QProgressBar(maximum=self.target_face_count)
        self.progress_bar.setFormat("%v / %m views captured")

        self.liveness_label = QLabel("Liveness: Checking...", alignment=Qt.AlignCenter)
        self.liveness_label.setStyleSheet("font-size:14px; padding:5px;")

        # Buttons
        self.capture_button = QPushButton("📷 Capture Angle", clicked=self.capture_face)
        self.register_button = QPushButton("✓ Register Worker", clicked=self.register_worker)
        self.register_button.setEnabled(False)
        self.cancel_button = QPushButton("✗ Cancel", clicked=self.reject)

        for b, color in [(self.capture_button, "#4CAF50"), (self.register_button, "#2196F3")]:
            b.setStyleSheet(f"font-size:14px; padding:8px; background:{color}; color:white; border-radius:4px;")

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.capture_button)
        btn_layout.addWidget(self.register_button)
        btn_layout.addWidget(self.cancel_button)

        layout.addWidget(info_group)
        layout.addWidget(self.guidance_label)
        layout.addWidget(self.camera_label)
        layout.addWidget(self.liveness_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def _setup_timers(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_camera_feed)
        self.timer.start(30)  # ~33 FPS

    # ---------------- Camera Feed -----------------
    def _update_camera_feed(self):
        """Refresh camera preview safely."""
        try:
            if not self.camera_manager or not getattr(self.camera_manager, "cameras", None):
                self.status_label.setText("❌ No cameras available.")
                return

            if not self.camera_manager.running:
                self.camera_manager.start()
                return

            cam_id = self.camera_selector.currentData() or list(self.camera_manager.cameras.keys())[0]
            frame = self.camera_manager.get_frame(cam_id, timeout=0.1)
            if frame is None:
                return

            self.current_frame = frame.copy()
            detections = self.detector.detect(frame)

            for det in detections:
                x1, y1, x2, y2 = map(int, det.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if self.enable_3d_checkbox.isChecked() and hasattr(self.face_system, "system_3d"):
                    crop_func = import_from_project("face_recognition_3d_module", "crop_face_from_detection")
                    if crop_func:
                        face_img = crop_func(self.current_frame, det.bbox)
                        if face_img is not None:
                            self._draw_3d_feedback(frame, face_img, x1, y1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        except Exception as e:
            logger.error(f"Camera update error: {e}")

    def _draw_3d_feedback(self, frame: np.ndarray, face_img: np.ndarray, x1: int, y1: int):
        """Show head angle and liveness feedback overlay."""
        try:
            sys3d = self.face_system.system_3d
            landmarks = sys3d.extract_face_mesh(face_img)
            if landmarks is not None:
                nose, left_eye, right_eye = landmarks[1], landmarks[33], landmarks[263]
                eye_center = (left_eye + right_eye) / 2
                face_vec = nose - eye_center
                angle = np.degrees(np.arctan2(face_vec[0], face_vec[1]))
                cv2.putText(frame, f"Angle: {angle:.1f}°", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            emb = self.face_system.get_face_embedding_2d(face_img)
            if emb is not None:
                f3d = sys3d.extract_3d_features(face_img, emb)
                live = "LIVE" if f3d.is_live else "SPOOF"
                color = "#4CAF50" if f3d.is_live else "#f44336"
                self.liveness_label.setText(f"Liveness: {live} ({f3d.liveness_confidence:.2f})")
                self.liveness_label.setStyleSheet(f"font-size:14px; padding:5px; background:{color}; color:white;")
        except Exception as e:
            logger.debug(f"3D feedback error: {e}")

    # ---------------- Capture / Register -----------------
    def capture_face(self):
        """Capture single face view."""
        if self.current_frame is None:
            QMessageBox.warning(self, "Error", "No camera feed.")
            return

        detections = self.detector.detect(self.current_frame)
        if not detections:
            QMessageBox.warning(self, "No Face", "No face detected.")
            return
        if len(detections) > 1:
            QMessageBox.warning(self, "Multiple Faces", "Ensure only one person is visible.")
            return

        crop_func = import_from_project("face_recognition_3d_module", "crop_face_from_detection")
        if not crop_func:
            QMessageBox.critical(self, "Error", "Face cropper not available.")
            return

        face_img = crop_func(self.current_frame, detections[0].bbox)
        if face_img is None:
            QMessageBox.warning(self, "Error", "Failed to crop face.")
            return

        self.captured_faces.append(face_img)
        self.progress_bar.setValue(len(self.captured_faces))
        self._update_guidance()

        if len(self.captured_faces) >= 3:
            self.register_button.setEnabled(True)
        if len(self.captured_faces) >= self.target_face_count:
            self.capture_button.setEnabled(False)
            self.status_label.setText("✅ All views captured. Click 'Register Worker'.")

    def _update_guidance(self):
        """Dynamic text updates for head pose guidance."""
        step = len(self.captured_faces)
        msgs = [
            "📸 Step 1/7: Face FRONTAL",
            "📸 Step 2/7: Turn SLIGHTLY LEFT (~15°)",
            "📸 Step 3/7: Turn MORE LEFT (~30°)",
            "📸 Step 4/7: Face FRONT again",
            "📸 Step 5/7: Turn SLIGHTLY RIGHT (~15°)",
            "📸 Step 6/7: Turn MORE RIGHT (~30°)",
            "📸 Step 7/7: Final FRONT pose",
        ]
        colors = ["#2196F3", "#FF9800", "#FF9800", "#2196F3", "#9C27B0", "#9C27B0", "#4CAF50"]
        if step < len(msgs):
            self.guidance_label.setText(msgs[step])
            self.guidance_label.setStyleSheet(f"font-size:16px; font-weight:bold; padding:8px; background:{colors[step]}; color:white;")
        else:
            self.guidance_label.setText("✅ All angles captured! Ready to register.")
            self.guidance_label.setStyleSheet("font-size:16px; font-weight:bold; padding:8px; background:#4CAF50; color:white;")

    def register_worker(self):
        """Save captured views and register in DB."""
        name = self.worker_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid", "Enter worker name.")
            return
        if len(self.captured_faces) < 3:
            QMessageBox.warning(self, "Too Few", "Capture at least 3 views.")
            return

        worker_id = self.worker_id_input.value()
        db_mod = import_from_project("database_module", "DatabaseManager")
        if not db_mod:
            QMessageBox.critical(self, "Error", "Database module missing.")
            return

        replace = False
        try:
            db = db_mod()
            existing = {w["worker_id"]: w["name"] for w in db.get_all_workers()}
            if worker_id in existing:
                ans = QMessageBox.question(
                    self,
                    "Worker Exists",
                    f"Worker ID {worker_id} already exists:\nReplace {existing[worker_id]}?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if ans == QMessageBox.No:
                    db.close()
                    return
                db.delete_worker_embeddings(worker_id)
                replace = True
            db.close()
        except Exception as e:
            logger.error(f"DB check failed: {e}")

        msg = QMessageBox(self)
        msg.setWindowTitle("Registering...")
        msg.setText(f"Registering {name} with {len(self.captured_faces)} views...")
        msg.show()
        msg.repaint()

        def guide_cb(m): logger.info(m)

        try:
            ok = self.face_system.register_worker(
                worker_id=worker_id,
                worker_name=name,
                face_images=self.captured_faces,
                guidance_callback=guide_cb,
            )
            msg.close()
            if ok:
                act = "updated" if replace else "registered"
                QMessageBox.information(self, "Success", f"Worker '{name}' {act} successfully.")
                self.accept()
            else:
                QMessageBox.critical(self, "Failure", "Registration failed.")
        except Exception as e:
            msg.close()
            logger.error(f"Registration error: {e}")
            QMessageBox.critical(self, "Error", str(e))

    # ---------------- Cleanup -----------------
    def closeEvent(self, event):
        self.timer.stop()
        if self.camera_manager and not self.cameras_were_running and self.camera_manager.running:
            try:
                self.camera_manager.stop()
                logger.info("Camera stopped after registration.")
            except Exception:
                pass
        super().closeEvent(event)
