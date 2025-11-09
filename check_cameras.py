"""Check configured cameras"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config_manager import ConfigManager

config = ConfigManager()
cameras = config.get_cameras()

print("=" * 70)
print("CONFIGURED CAMERAS")
print("=" * 70)
for cam in cameras:
    print(f"Camera {cam.camera_id}: {cam.name}")
    print(f"  Source: {cam.source}")
    print(f"  FPS: {cam.fps}")
    print(f"  ROI: {cam.roi}")
    print()

print(f"Total cameras: {len(cameras)}")
print("=" * 70)

Path("check_cameras.py").unlink()
