"""Debug script to check config loading."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

from config_manager import ConfigManager

config_manager = ConfigManager("config/config.json")
camera_configs = config_manager.get_camera_configs()

print("=" * 60)
print("Camera Configuration Debug")
print("=" * 60)
print(f"\nNumber of cameras in config: {len(camera_configs)}")
print(f"\nCamera configs:")
for i, config in enumerate(camera_configs):
    print(f"\n  Camera {i}:")
    for key, value in config.items():
        print(f"    {key}: {value}")

print("\n" + "=" * 60)
