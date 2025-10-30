"""
Test script to detect and display all connected cameras.
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from camera_manager import CameraManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("=" * 60)
    print("Camera Detection Test")
    print("=" * 60)
    
    # Detect available cameras
    available = CameraManager.get_available_cameras()
    print(f"\nDetected {len(available)} camera(s): {available}")
    
    if not available:
        print("\nNo cameras detected!")
        return
    
    # Create camera manager (will auto-detect and configure all cameras)
    print("\nInitializing camera manager...")
    manager = CameraManager()
    
    print(f"\nConfigured cameras:")
    for cam_id, config in manager.cameras.items():
        print(f"  - Camera {cam_id}: {config.name} ({config.width}x{config.height} @ {config.fps}fps)")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
