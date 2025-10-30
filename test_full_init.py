"""Test full initialization flow like the main app."""
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

from config_manager import ConfigManager
from camera_manager import CameraManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("=" * 60)
print("Full Initialization Test (like main.py)")
print("=" * 60)

# Load config like main.py does
config_manager = ConfigManager("config/config.json")
camera_configs = config_manager.get_camera_configs()

print(f"\nLoaded {len(camera_configs)} camera config(s)")

# Initialize camera manager like main.py does
camera_manager = CameraManager(camera_configs, auto_detect=False)

print(f"\nCamera manager has {len(camera_manager.cameras)} camera(s)")
print(f"Camera IDs: {list(camera_manager.cameras.keys())}")

for cam_id, config in camera_manager.cameras.items():
    print(f"  - Camera {cam_id}: {config.name}")

print("\n" + "=" * 60)
