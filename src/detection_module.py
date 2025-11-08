"""
Person Detection Module (Hardened)
----------------------------------
Real-time person detection using YOLOv8 with stable runtime,
automatic device fallback, and safe video loop.
"""

import cv2
import numpy as np
from typing import List, Optional, Any
from dataclasses import dataclass
import logging
import time
from pathlib import Path

# Runtime import of YOLO
try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("ultralytics not installed. Run: pip install ultralytics") from e

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class Detection:
    """Represents a detected person."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    tracker_id: Optional[int] = None
    timestamp: float = time.time()
    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    recognition_score: Optional[float] = None


class PersonDetector:
    """YOLOv8-based person detector with safe runtime checks."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        device: str = None,
    ):
        """
        Initialize detector with automatic device fallback.
        """
        if device is None:
            device = "cuda" if self._cuda_available() else "cpu"
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.model = self._load_model(model_path)
        self.class_names = self.model.names
        self.person_class_id = 0  # COCO class 0 = person

        logger.info(f"YOLOv8 model loaded ({model_path}) on {self.device}")

    @staticmethod
    def _cuda_available() -> bool:
        """Check CUDA availability via PyTorch."""
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _load_model(self, model_path: str) -> Any:
        """Load YOLOv8 model safely."""
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect persons in frame."""
        if frame is None or frame.size == 0:
            return []

        try:
            # Ensure frame dimensions are multiples of 32
            h, w = frame.shape[:2]
            new_w, new_h = int(np.ceil(w / 32) * 32), int(np.ceil(h / 32) * 32)
            if (new_w, new_h) != (w, h):
                frame = cv2.resize(frame, (new_w, new_h))

            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[self.person_class_id],
                verbose=False,
                device=self.device,
            )

            detections = []
            # Handle YOLO result structure (may be a single or list of Results)
            result_list = results if isinstance(results, list) else [results]
            for result in result_list:
                if not hasattr(result, "boxes"):
                    continue
                for box in result.boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf.item())
                    class_id = int(box.cls.item())

                    detections.append(
                        Detection(
                            bbox=bbox,
                            confidence=confidence,
                            class_id=class_id,
                            timestamp=time.time(),
                        )
                    )
            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def draw_detections(
        self, frame: np.ndarray, detections: List[Detection], show_conf: bool = True
    ) -> np.ndarray:
        """Draw bounding boxes and labels."""
        if frame is None:
            return None

        output = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            x1, y1 = max(0, x1), max(30, y1)  # prevent text overflow

            if det.worker_name and det.worker_name != "Unknown":
                color = (0, 255, 255)
                label = f"{det.worker_name}"
                font_scale, thickness = 0.8, 2

            elif det.worker_name == "Unknown":
                color = (0, 0, 255)
                label = "Unknown"
                font_scale, thickness = 0.7, 2

            else:
                color = (0, 255, 0)
                label = ""
                font_scale, thickness = 0.6, 1

            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            # Draw label only when non-empty
            if label:
                # Label background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(output, (x1, y1 - h - 10), (x1 + w + 8, y1), color, -1)

                # Text
                cv2.putText(
                    output,
                    label,
                    (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )
        return output