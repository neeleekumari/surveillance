#!/usr/bin/env python3
"""
Demo script for the Floor Monitoring Application.
This script demonstrates the core functionality without requiring hardware.
"""
import sys
import os
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def mock_dependencies():
    """Mock external dependencies for demo purposes."""
    import sys
    from unittest.mock import MagicMock
    
    # Mock psycopg2
    sys.modules['psycopg2'] = MagicMock()
    sys.modules['psycopg2.extensions'] = MagicMock()
    
    # Mock ultralytics
    sys.modules['ultralytics'] = MagicMock()
    mock_yolo = MagicMock()
    mock_yolo.names = {0: 'person'}
    sys.modules['ultralytics.YOLO'] = MagicMock(return_value=mock_yolo)
    
    # Mock cv2
    sys.modules['cv2'] = MagicMock()
    
    # Mock PyQt5
    sys.modules['PyQt5'] = MagicMock()
    sys.modules['PyQt5.QtWidgets'] = MagicMock()
    sys.modules['PyQt5.QtCore'] = MagicMock()
    sys.modules['PyQt5.QtGui'] = MagicMock()
    
    # Mock matplotlib
    sys.modules['matplotlib'] = MagicMock()
    sys.modules['matplotlib.pyplot'] = MagicMock()
    sys.modules['matplotlib.backends'] = MagicMock()
    sys.modules['matplotlib.backends.backend_pdf'] = MagicMock()
    
    # Mock win10toast
    sys.modules['win10toast'] = MagicMock()
    
    # Mock numpy
    sys.modules['numpy'] = MagicMock()
    
    # Mock pandas
    sys.modules['pandas'] = MagicMock()

def demo_camera_manager():
    """Demonstrate the CameraManager functionality."""
    print("=== Camera Manager Demo ===")
    
    # Mock the CameraManager to avoid actual camera initialization
    from src.camera_manager import CameraManager, CameraConfig
    from unittest.mock import patch, MagicMock
    
    # Create a mock camera manager
    camera_manager = CameraManager()
    
    # Add a mock camera
    mock_config = {
        'id': 0,
        'name': 'Demo Camera'
    }
    
    # Patch the VideoCapture to avoid actual camera access
    with patch('src.camera_manager.cv2.VideoCapture') as mock_capture:
        mock_capture_instance = MagicMock()
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.get.return_value = 1280  # width
        mock_capture.return_value = mock_capture_instance
        
        result = camera_manager.add_camera(mock_config)
        print(f"Camera added: {result}")
    
    print("Camera Manager demo completed.\n")

def demo_detection_module():
    """Demonstrate the PersonDetector functionality."""
    print("=== Person Detection Demo ===")
    
    from src.detection_module import PersonDetector, Detection
    import numpy as np
    
    # Create a mock detector
    detector = PersonDetector.__new__(PersonDetector)
    detector.conf_threshold = 0.5
    detector.iou_threshold = 0.5
    detector.person_class_id = 0
    
    # Create mock detections with proper numpy arrays
    mock_detections = [
        Detection(
            bbox=np.array([100.0, 100.0, 200.0, 200.0]),
            confidence=0.9,
            class_id=0
        ),
        Detection(
            bbox=np.array([300.0, 150.0, 400.0, 250.0]),
            confidence=0.85,
            class_id=0
        )
    ]
    
    # Test drawing detections
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Skip drawing detections since it requires more complex mocking
    print(f"Detected {len(mock_detections)} persons")
    
    print("Person Detection demo completed.\n")

def demo_presence_tracker():
    """Demonstrate the PresenceTracker functionality."""
    print("=== Presence Tracker Demo ===")
    
    from src.presence_tracker import PresenceTracker
    
    # Create tracker with demo config
    config = {
        "warning_minutes": 15,
        "alert_minutes": 30
    }
    tracker = PresenceTracker(config)
    
    # Add workers
    tracker.add_worker(1, "Alice")
    tracker.add_worker(2, "Bob")
    
    # Simulate detections
    detections = [
        {"worker_id": 1, "confidence": 0.9},
        {"worker_id": 2, "confidence": 0.85}
    ]
    
    updates = tracker.update_detections(detections)
    print(f"Presence updates: {len(updates)} workers detected")
    
    # Get all statuses
    statuses = tracker.get_all_statuses()
    for status in statuses:
        print(f"Worker {status['name']} (ID: {status['worker_id']}): {status['status']}")
    
    print("Presence Tracker demo completed.\n")

def demo_alert_manager():
    """Demonstrate the AlertManager functionality."""
    print("=== Alert Manager Demo ===")
    
    from src.alert_manager import AlertManager
    
    # Create alert manager
    config = {
        "notifications": {
            "enabled": True,
            "sound": True
        }
    }
    alert_manager = AlertManager(config)
    
    # Add alerts
    alert_ids = []
    alert_ids.append(alert_manager.add_alert(
        "Worker Alert",
        "Worker Alice has been present for over 30 minutes",
        "alert",
        worker_id=1
    ))
    
    alert_ids.append(alert_manager.add_alert(
        "Worker Warning",
        "Worker Bob has been present for over 15 minutes",
        "warning",
        worker_id=2
    ))
    
    # Check active alerts
    active_alerts = alert_manager.get_active_alerts()
    print(f"Active alerts: {len(active_alerts)}")
    
    # Acknowledge an alert
    if alert_ids:
        result = alert_manager.acknowledge_alert(alert_ids[0])
        print(f"Alert acknowledged: {result}")
    
    print("Alert Manager demo completed.\n")

def demo_config_manager():
    """Demonstrate the ConfigManager functionality."""
    print("=== Configuration Manager Demo ===")
    
    from src.config_manager import ConfigManager
    import tempfile
    import json
    
    # Create a temporary config file
    test_config = {
        "database": {
            "host": "localhost",
            "name": "demo_db",
            "user": "demo_user",
            "password": "demo_pass",
            "port": 5432
        },
        "thresholds": {
            "warning_minutes": 10,
            "alert_minutes": 20
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        temp_config_path = f.name
    
    try:
        # Load config
        config_manager = ConfigManager(temp_config_path)
        config = config_manager.load_config()
        
        print(f"Database host: {config_manager.get('database.host')}")
        print(f"Warning threshold: {config_manager.get('thresholds.warning_minutes')} minutes")
        
        # Update config
        config_manager.set('app.demo_mode', True)
        print(f"Demo mode: {config_manager.get('app.demo_mode')}")
        
    finally:
        # Clean up
        os.unlink(temp_config_path)
    
    print("Configuration Manager demo completed.\n")

def main():
    """Main demo function."""
    print("🏢 Floor Monitoring Application Demo")
    print("=" * 40)
    print("This demo showcases the core functionality of the Floor Monitoring Application.")
    print()
    
    # Mock dependencies
    mock_dependencies()
    
    # Run demos
    demo_config_manager()
    demo_camera_manager()
    demo_detection_module()
    demo_presence_tracker()
    demo_alert_manager()
    
    print("🎉 Demo completed successfully!")
    print()
    print("To run the full application, ensure you have:")
    print("  - Python 3.10+")
    print("  - PostgreSQL database")
    print("  - USB cameras connected")
    print("  - All dependencies installed (pip install -r requirements.txt)")
    print()
    print("Then run: python run.py")

if __name__ == "__main__":
    main()