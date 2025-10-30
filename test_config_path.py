"""Test config path resolution."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

from config_manager import ConfigManager

# Test with default path (like main.py does)
print("Testing with default path...")
cm1 = ConfigManager()
print(f"Config path: {cm1.config_path}")
print(f"Config path exists: {cm1.config_path.exists()}")
print(f"Number of cameras: {len(cm1.get_camera_configs())}")

# Test with explicit path
print("\nTesting with explicit path...")
cm2 = ConfigManager("config/config.json")
print(f"Config path: {cm2.config_path}")
print(f"Config path exists: {cm2.config_path.exists()}")
print(f"Number of cameras: {len(cm2.get_camera_configs())}")
