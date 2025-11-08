"""
Camera Manager Module (Hardened)
--------------------------------
Handles USB/IP camera initialization, threaded capture, and frame buffering
for real-time 3D/AI processing pipelines.
"""

import cv2
import time
import threading
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from queue import Queue, Empty
import platform

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    camera_id: int
    name: str
    width: int = 1280
    height: int = 720
    fps: int = 30
    roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)


class CameraManager:
    """Manages multiple threaded camera streams with minimal latency."""

    def __init__(self, camera_configs: Optional[List[Dict[str, Any]]] = None, auto_detect: bool = True):
        self.cameras: Dict[int, CameraConfig] = {}
        self.captures: Dict[int, cv2.VideoCapture] = {}
        self.frame_queues: Dict[int, Queue] = {}
        self.threads: Dict[int, threading.Thread] = {}
        self._lock = threading.Lock()
        self.running = False
        # Track reopen backoff per camera
        self._reopen_backoff: Dict[int, float] = {}

        # Initialize
        if camera_configs:
            self._load_from_config(camera_configs, auto_detect)
        else:
            logger.info("No config provided, auto-detecting cameras...")
            self._auto_detect_all()

        logger.info(f"Initialized {len(self.cameras)} camera(s): {list(self.cameras.keys())}")

    # ---------------------- Initialization ----------------------
    def _load_from_config(self, configs: List[Dict[str, Any]], auto_detect: bool):
        configured = set()
        for i, conf in enumerate(configs):
            time.sleep(0.5 * i)  # small stagger to prevent DirectShow lockups
            cam_id = conf.get("id", i)
            if self.add_camera(conf):
                configured.add(cam_id)

        if auto_detect:
            available = self.get_available_cameras()
            new_cams = [cid for cid in available if cid not in configured]
            for cid in new_cams:
                self.add_camera({
                    "id": cid, "name": f"Camera {cid}", "width": 1280, "height": 720, "fps": 30
                })

    def _auto_detect_all(self):
        for cid in self.get_available_cameras():
            self.add_camera({
                "id": cid, "name": f"Camera {cid}", "width": 1280, "height": 720, "fps": 30
            })

    @staticmethod
    def get_available_cameras(max_test: int = 6) -> List[int]:
        """Return list of working camera indices (tries multiple backends on Windows)."""
        available = []
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if platform.system() == "Windows" else [cv2.CAP_ANY]
        for i in range(max_test):
            found = False
            for backend in backends:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ok, _ = cap.read()
                    cap.release()
                    if ok:
                        available.append(i)
                        found = True
                        break
                else:
                    cap.release()
            # continue to next index
        return list(dict.fromkeys(available))

    def _open_capture(self, cam_id: int, conf: Optional[Dict[str, Any]] = None) -> Tuple[Optional[cv2.VideoCapture], Optional[int]]:
        """Try to open a camera with fallback backends and basic property set/verify."""
        backend_candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if platform.system() == "Windows" else [cv2.CAP_ANY]
        for backend in backend_candidates:
            try:
                cap = cv2.VideoCapture(cam_id, backend)
                t0 = time.time()
                while not cap.isOpened() and time.time() - t0 < 2.0:
                    time.sleep(0.05)
                if not cap.isOpened():
                    cap.release()
                    continue
                # Apply requested properties
                if conf:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, conf.get("width", 1280))
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, conf.get("height", 720))
                    cap.set(cv2.CAP_PROP_FPS, conf.get("fps", 30))
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Sanity read
                ok, _ = cap.read()
                if not ok:
                    cap.release()
                    continue
                logger.info(f"Camera {cam_id} opened via backend={backend}")
                return cap, backend
            except Exception:
                try:
                    cap.release()
                except Exception:
                    pass
        logger.warning(f"Failed to open camera {cam_id} with available backends")
        return None, None

    # ---------------------- Camera Management ----------------------
    def add_camera(self, conf: Dict[str, Any]) -> bool:
        """Safely add and initialize one camera."""
        cam_id = conf.get("id")
        cap, backend = self._open_capture(cam_id, conf)
        if not cap:
            logger.warning(f"Camera {cam_id} not accessible.")
            return False

        cfg = CameraConfig(
            camera_id=cam_id,
            name=conf.get("name", f"Camera {cam_id}"),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=int(cap.get(cv2.CAP_PROP_FPS)),
            roi=conf.get("roi")
        )
        self.cameras[cam_id] = cfg
        self.captures[cam_id] = cap
        self.frame_queues[cam_id] = Queue(maxsize=1)
        self._reopen_backoff[cam_id] = 0.5
        logger.info(f"Added camera {cam_id} ({cfg.width}x{cfg.height}@{cfg.fps}fps)")
        return True

    def start(self) -> bool:
        """Start threaded capture for all cameras."""
        if not self.cameras:
            logger.warning("No cameras configured.")
            return False
        self.running = True
        for cam_id in self.cameras:
            t = threading.Thread(target=self._capture_loop, args=(cam_id,), daemon=True)
            self.threads[cam_id] = t
            t.start()
        logger.info(f"Started {len(self.threads)} camera thread(s).")
        return True

    def stop(self):
        """Stop all threads and release resources safely."""
        self.running = False
        logger.info("Stopping camera threads...")
        for t in list(self.threads.values()):
            if t.is_alive():
                t.join(timeout=2.0)

        for cam_id, cap in list(self.captures.items()):
            try:
                if cap.isOpened():
                    cap.release()
                    logger.info(f"Released camera {cam_id}")
            except Exception:
                pass

        self.threads.clear()
        logger.info("All cameras stopped and released.")

    # ---------------------- Capture Loop ----------------------
    def _capture_loop(self, cam_id: int):
        """Continuously capture frames and push latest to queue."""
        cap = self.captures.get(cam_id)
        q = self.frame_queues.get(cam_id)
        if not cap or not q:
            logger.error(f"Camera {cam_id} invalid configuration.")
            return

        logger.info(f"Starting capture loop for camera {cam_id}")
        last_fail = 0
        consecutive_failures = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if time.time() - last_fail > 5:
                    logger.warning(f"Frame read failed from camera {cam_id}. Retrying...")
                    last_fail = time.time()
                # Try to reopen after a number of consecutive failures
                if consecutive_failures >= 20:  # ~2s of failures at 0.1s sleep
                    try:
                        logger.warning(f"Reopening camera {cam_id} after {consecutive_failures} failed reads...")
                        # Release old cap
                        try:
                            if cap and cap.isOpened():
                                cap.release()
                        except Exception:
                            pass
                        # Attempt reopen with backoff
                        conf = dict(id=cam_id,
                                    name=self.cameras[cam_id].name,
                                    width=self.cameras[cam_id].width,
                                    height=self.cameras[cam_id].height,
                                    fps=self.cameras[cam_id].fps)
                        new_cap, _ = self._open_capture(cam_id, conf)
                        if new_cap:
                            self.captures[cam_id] = new_cap
                            cap = new_cap
                            consecutive_failures = 0
                            self._reopen_backoff[cam_id] = 0.5
                            logger.info(f"Camera {cam_id} reopened successfully.")
                        else:
                            # exponential backoff before next reopen attempt
                            backoff = min(5.0, self._reopen_backoff.get(cam_id, 0.5) * 2)
                            self._reopen_backoff[cam_id] = backoff
                            logger.warning(f"Camera {cam_id} reopen failed. Backing off {backoff:.1f}s")
                            time.sleep(backoff)
                            # keep failing until next loop
                    except Exception as e:
                        logger.warning(f"Error during camera {cam_id} reopen: {e}")
                time.sleep(0.1)
                continue

            # Apply ROI
            roi = self.cameras[cam_id].roi
            if roi:
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]

            # Push to queue (keep newest frame only)
            if q.full():
                try:
                    q.get_nowait()
                except Empty:
                    pass
            try:
                q.put_nowait(frame.copy())
            except Exception:
                pass

            # Reset failure counter on success
            consecutive_failures = 0

            # Dynamic sleep to balance CPU usage
            time.sleep(1 / (self.cameras[cam_id].fps * 2))

        logger.info(f"Camera {cam_id} loop terminated.")

    def get_frame(self, cam_id: int, timeout: float = 0.2) -> Optional[np.ndarray]:
        """Retrieve latest frame."""
        if cam_id not in self.frame_queues:
            logger.error(f"Camera {cam_id} not found.")
            return None
        q = self.frame_queues[cam_id]
        try:
            return q.get_nowait()
        except Empty:
            try:
                return q.get(timeout=timeout)
            except Empty:
                return None

    def __del__(self):
        """Destructor: ensure proper cleanup."""
        try:
            self.stop()
        except Exception:
            pass


# ---------------------- Standalone Test ----------------------
def test_camera_manager():
    """Manual test utility."""
    mgr = CameraManager()
    if not mgr.cameras:
        print("❌ No cameras detected.")
        return

    mgr.start()
    print("Press 'q' to quit")
    try:
        while True:
            for cid in mgr.cameras:
                frame = mgr.get_frame(cid)
                if frame is not None:
                    cv2.imshow(f"Camera {cid}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        mgr.stop()
        cv2.destroyAllWindows()