"""
Person Detection Module
----------------------
Uses YOLOv8 for real-time person detection in video frames.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import time

# Import YOLO from ultralytics
# Use TYPE_CHECKING to separate static analysis import from runtime import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # For static analysis tools that have issues with the direct import
    from ultralytics.models.yolo.model import YOLO
else:
    # Runtime import
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics package not found. Install with: pip install ultralytics")

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Represents a detected person in a frame."""
    bbox: np.ndarray  # [x1, y1, x2, y2] in pixel coordinates
    confidence: float
    class_id: int
    tracker_id: Optional[int] = None
    timestamp: float = 0.0  # Fixed: Initialize with 0.0 instead of None for float type
    worker_id: Optional[int] = None  # Recognized worker ID
    worker_name: Optional[str] = None  # Recognized worker name
    recognition_score: Optional[float] = None  # Face recognition similarity score


class PersonDetector:
    """Handles person detection using YOLOv8."""
    
    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.5, 
                 iou_threshold: float = 0.5, device: str = 'cuda:0'):
        """Initialize the person detector.
        
        Args:
            model_path: Path to YOLOv8 model weights (.pt file)
            conf_threshold: Confidence threshold for detections (0-1)
            iou_threshold: IOU threshold for NMS (0-1)
            device: Device to run inference on ('cuda:0', 'cpu', etc.)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Load YOLOv8 model
        self.model = self._load_model(model_path)
        self.class_names = self.model.names
        self.person_class_id = 0  # COCO person class ID is 0
        
        logger.info(f"Loaded YOLOv8 model (device={device})")
    
    def _load_model(self, model_path: str) -> Any:
        """Load YOLOv8 model from file or download if not found."""
        try:
            model = YOLO(model_path)
            # Test model with a dummy input
            model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect persons in a frame.
        
        Args:
            frame: Input BGR image (numpy array)
            
        Returns:
            List of Detection objects
        """
        if frame is None or frame.size == 0:
            return []
            
        try:
            # Run YOLOv8 inference
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[self.person_class_id],  # Only detect persons
                verbose=False,
                device=self.device
            )
            
            # Process detections
            detections = []
            for result in results:
                for box in result.boxes:
                    # Convert to [x1, y1, x2, y2] format
                    bbox = box.xyxy[0].cpu().numpy()
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    
                    detections.append(Detection(
                        bbox=bbox,
                        confidence=confidence,
                        class_id=class_id,
                        timestamp=time.time()
                    ))
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection], 
                       show_conf: bool = True) -> np.ndarray:
        """Draw detection bounding boxes on the frame.
        
        Args:
            frame: Input BGR image
            detections: List of Detection objects
            show_conf: Whether to show confidence scores
            
        Returns:
            Frame with drawn detections
        """
        if frame is None:
            return None
            
        frame_copy = frame.copy()
        
        # Show labels with color-coded backgrounds based on recognition status
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            
            # Determine status and set colors accordingly
            if det.worker_name and det.worker_name != "Unknown":
                # RECOGNIZED WORKER - Green background
                label = f"✓ {det.worker_name}"
                if show_conf and det.recognition_score:
                    label += f" ({det.recognition_score:.0%})"
                bg_color = (0, 150, 0)  # Green - Recognized and identified
                text_color = (255, 255, 255)  # White text
                font_scale = 0.7
                font_thickness = 2
                
            elif det.worker_name == "Unknown" and det.tracker_id is not None:
                # TRACKED BUT UNKNOWN - Orange background
                label = f"⚠ Unknown Person"
                if det.tracker_id:
                    label += f" [Track {det.tracker_id}]"
                bg_color = (0, 140, 255)  # Orange - Being tracked but not recognized
                text_color = (255, 255, 255)  # White text
                font_scale = 0.6
                font_thickness = 2
                
            elif det.tracker_id is not None:
                # BEING TRACKED - Yellow background
                label = f"👤 Tracking..."
                if det.tracker_id:
                    label += f" [T{det.tracker_id}]"
                bg_color = (0, 200, 255)  # Yellow - Detection in progress
                text_color = (0, 0, 0)  # Black text for better contrast
                font_scale = 0.6
                font_thickness = 2
                
            else:
                # NEW DETECTION - Blue background
                label = "🔍 Detecting..."
                bg_color = (200, 100, 0)  # Blue - New detection, not yet tracked
                text_color = (255, 255, 255)  # White text
                font_scale = 0.6
                font_thickness = 1
            
            # Calculate label size
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Position label at top of detection
            label_x = x1
            label_y = max(30, y1 - 10)  # Keep label visible
            
            # Draw background rectangle with status color
            cv2.rectangle(
                frame_copy, 
                (label_x, label_y - label_h - 8), 
                (label_x + label_w + 10, label_y + 2), 
                bg_color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame_copy,
                label,
                (label_x + 5, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA
            )
        
        return frame_copy


def test_detector():
    """Test function for the PersonDetector class."""
    import cv2
    
    # Initialize detector
    detector = PersonDetector(conf_threshold=0.5, device='cpu')
    
    # Open default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Detect persons
            detections = detector.detect(frame)
            
            # Draw detections
            frame_with_detections = detector.draw_detections(frame, detections)
            
            # Display FPS
            cv2.putText(
                frame_with_detections,
                f"Persons: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            # Show the frame
            cv2.imshow("Person Detection", frame_with_detections)
            
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the test
    test_detector()
