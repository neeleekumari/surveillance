"""Test camera initialization with config."""
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

from camera_manager import CameraManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test with config
camera_configs = [
    {
        'id': 0,
        'name': 'Camera 0',
        'width': 1280,
        'height': 720,
        'fps': 30,
        'rois': []
    },
    {
        'id': 1,
        'name': 'Camera 1',
        'width': 1280,
        'height': 720,
        'fps': 30,
        'rois': []
    }
]

print("=" * 60)
print("Testing Camera Manager Initialization")
print("=" * 60)

manager = CameraManager(camera_configs, auto_detect=False)

print(f"\nConfigured cameras: {list(manager.cameras.keys())}")
print(f"Number of cameras: {len(manager.cameras)}")

for cam_id, config in manager.cameras.items():
    print(f"  - Camera {cam_id}: {config.name} ({config.width}x{config.height})")

print("\n" + "=" * 60)
