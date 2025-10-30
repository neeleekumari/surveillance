"""
Unit tests for the camera manager module.
"""
import sys
import os
import unittest
from pathlib import Path
import logging

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestCameraManager(unittest.TestCase):
    """Unit tests for the CameraManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        logging.basicConfig(level=logging.INFO)
    
    def test_camera_config_creation(self):
        """Test CameraConfig dataclass creation."""
        try:
            from src.camera_manager import CameraConfig
            config = CameraConfig(
                camera_id=0,
                name="Test Camera"
            )
            self.assertEqual(config.camera_id, 0)
            self.assertEqual(config.name, "Test Camera")
            print("CameraConfig creation test passed")
        except Exception as e:
            self.fail(f"CameraConfig creation failed: {e}")
    
    def test_camera_manager_initialization(self):
        """Test CameraManager initialization."""
        try:
            from src.camera_manager import CameraManager
            manager = CameraManager()
            self.assertIsInstance(manager, CameraManager)
            print("CameraManager initialization test passed")
        except Exception as e:
            self.fail(f"CameraManager initialization failed: {e}")
    
    def test_add_camera(self):
        """Test adding a camera configuration."""
        try:
            from src.camera_manager import CameraManager
            manager = CameraManager()
            
            # Test config (using a non-existent camera ID to avoid actually opening a camera)
            test_config = {
                'id': 999,  # Non-existent camera ID
                'name': 'Test Camera'
            }
            
            # This should fail gracefully since camera 999 doesn't exist
            result = manager.add_camera(test_config)
            # We expect this to return False since the camera doesn't exist
            self.assertFalse(result)
            print("Add camera test passed")
        except Exception as e:
            self.fail(f"Add camera test failed: {e}")

if __name__ == "__main__":
    unittest.main()