"""
Integration tests for the floor monitoring application.
"""
import sys
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestIntegration(unittest.TestCase):
    """Integration tests for the floor monitoring application."""
    
    def test_module_imports(self):
        """Test that all modules can be imported without errors."""
        try:
            # Mock external dependencies
            import sys
            from unittest.mock import MagicMock
            
            # Mock psycopg2
            sys.modules['psycopg2'] = MagicMock()
            sys.modules['psycopg2.extensions'] = MagicMock()
            
            # Mock ultralytics
            sys.modules['ultralytics'] = MagicMock()
            sys.modules['ultralytics.YOLO'] = MagicMock()
            
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
            
            from src.camera_manager import CameraManager
            from src.detection_module import PersonDetector
            from src.presence_tracker import PresenceTracker
            from src.database_module import DatabaseManager
            from src.alert_manager import AlertManager
            from src.ui_manager import UIManager
            from src.report_generator import ReportGenerator
            from src.config_manager import ConfigManager
            print("All modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import modules: {e}")
    
    def test_config_loading(self):
        """Test that configuration can be loaded."""
        try:
            from src.config_manager import ConfigManager
            config_manager = ConfigManager("../config/config.json")
            config = config_manager.load_config()
            self.assertIsInstance(config, dict)
            print("Configuration loaded successfully")
        except Exception as e:
            self.fail(f"Failed to load configuration: {e}")
    
    def test_presence_tracking(self):
        """Test presence tracking functionality."""
        try:
            from src.presence_tracker import PresenceTracker
            import time
            
            # Create tracker
            tracker = PresenceTracker()
            
            # Add a worker
            tracker.add_worker(1, "Test Worker")
            
            # Simulate detection
            detections = [{"worker_id": 1, "confidence": 0.9}]
            updates = tracker.update_detections(detections)
            
            # Check that we got updates
            self.assertIn(1, updates)
            print("Presence tracking working correctly")
            
        except Exception as e:
            self.fail(f"Presence tracking failed: {e}")
    
    def test_alert_system(self):
        """Test alert system functionality."""
        try:
            from src.alert_manager import AlertManager
            
            # Create alert manager
            alert_manager = AlertManager()
            
            # Add an alert
            alert_id = alert_manager.add_alert(
                "Test Alert",
                "This is a test alert",
                "info"
            )
            
            # Check that alert was added
            active_alerts = alert_manager.get_active_alerts()
            self.assertEqual(len(active_alerts), 1)
            print("Alert system working correctly")
            
        except Exception as e:
            self.fail(f"Alert system failed: {e}")

if __name__ == "__main__":
    unittest.main()