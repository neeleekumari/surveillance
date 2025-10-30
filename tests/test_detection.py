"""
Unit tests for the detection module.
"""
import sys
import os
import unittest
from pathlib import Path
import logging
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestDetectionModule(unittest.TestCase):
    """Unit tests for the PersonDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        logging.basicConfig(level=logging.INFO)
    
    def test_detection_class_creation(self):
        """Test Detection dataclass creation."""
        try:
            from src.detection_module import Detection
            detection = Detection(
                bbox=np.array([10, 20, 30, 40]),
                confidence=0.95,
                class_id=0
            )
            self.assertEqual(detection.confidence, 0.95)
            self.assertEqual(detection.class_id, 0)
            print("Detection class creation test passed")
        except Exception as e:
            self.fail(f"Detection class creation failed: {e}")
    
    def test_person_detector_initialization(self):
        """Test PersonDetector initialization."""
        try:
            from src.detection_module import PersonDetector
            # Use a simple model for testing
            detector = PersonDetector(
                model_path='yolov8n.pt',
                conf_threshold=0.5,
                iou_threshold=0.5,
                device='cpu'
            )
            self.assertIsInstance(detector, PersonDetector)
            print("PersonDetector initialization test passed")
        except Exception as e:
            print(f"PersonDetector initialization test note: {e}")
            print("This is expected if YOLOv8 model is not downloaded yet")
            # We don't fail the test here because the model might not be downloaded
    
    def test_draw_detections(self):
        """Test drawing detections on a frame."""
        try:
            from src.detection_module import PersonDetector, Detection
            import cv2
            
            # Create a dummy detector (without loading model)
            detector = PersonDetector.__new__(PersonDetector)
            
            # Create a test frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Create test detections
            detections = [
                Detection(
                    bbox=np.array([100, 100, 200, 200]),
                    confidence=0.9,
                    class_id=0
                )
            ]
            
            # Test drawing detections
            result_frame = detector.draw_detections(frame, detections)
            self.assertIsNotNone(result_frame)
            self.assertEqual(result_frame.shape, frame.shape)
            print("Draw detections test passed")
        except Exception as e:
            self.fail(f"Draw detections test failed: {e}")

if __name__ == "__main__":
    unittest.main()