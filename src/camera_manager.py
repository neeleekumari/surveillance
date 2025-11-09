"""
Camera Manager Module
--------------------
Handles USB camera initialization, frame capture, and management.
"""
import cv2
import time
import threading
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from queue import Queue

logger = logging.getLogger(__name__)

@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    camera_id: int
    name: str
    width: int = 1280
    height: int = 720
    fps: int = 30
    roi: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h

class CameraManager:
    """Manages multiple USB camera streams with frame buffering."""
    
    @staticmethod
    def get_available_cameras(max_test: int = 4) -> List[int]:
        """Detect and return a list of available camera indices.
        
        Args:
            max_test: Maximum number of camera indices to test (default: 4)
            
        Returns:
            List of available camera indices
        """
        available = []
        for i in range(max_test):
            # Use DirectShow on Windows with timeout
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1000)  # 1 second timeout
            
            if cap.isOpened():
                # Quick test read to verify camera actually works
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()
            
        return available
    
    def __init__(self, camera_configs: Optional[List[Dict[str, Any]]] = None, auto_detect: bool = True):
        """Initialize the camera manager with a list of camera configurations.
        
        Args:
            camera_configs: List of camera configuration dictionaries
            auto_detect: If True and configs provided, detect and add any additional cameras not in config
        """
        self.cameras: Dict[int, CameraConfig] = {}
        self.captures: Dict[int, cv2.VideoCapture] = {}
        self.frame_queues: Dict[int, Queue] = {}
        self.running = False
        self.threads: List[threading.Thread] = []
        
        # If configs provided, use them directly (fast path)
        if camera_configs:
            logger.info(f"Initializing {len(camera_configs)} camera(s) from config...")
            configured_ids = set()
            for i, config in enumerate(camera_configs):
                logger.info(f"Attempting to add camera {config.get('id', 'unknown')}...")
                # Add small delay between cameras to avoid conflicts
                if i > 0:
                    time.sleep(0.5)
                if self.add_camera(config):
                    configured_ids.add(config['id'])
                    logger.info(f"Successfully added camera {config['id']}")
                else:
                    logger.warning(f"Failed to add camera {config.get('id', 'unknown')}")
            
            # Optionally detect additional cameras not in config
            if auto_detect:
                logger.info("Checking for additional cameras...")
                available_cameras = self.get_available_cameras()
                additional = [cam_id for cam_id in available_cameras if cam_id not in configured_ids]
                
                if additional:
                    logger.info(f"Found {len(additional)} additional camera(s): {additional}")
                    for cam_id in additional:
                        self.add_camera({
                            'id': cam_id,
                            'name': f'Camera {cam_id}',
                            'width': 1280,
                            'height': 720,
                            'fps': 30
                        })
        else:
            # No configs provided - auto-detect all cameras
            logger.info("No camera configs provided, auto-detecting...")
            available_cameras = self.get_available_cameras()
            logger.info(f"Detected {len(available_cameras)} available camera(s): {available_cameras}")
            
            if available_cameras:
                for cam_id in available_cameras:
                    self.add_camera({
                        'id': cam_id,
                        'name': f'Camera {cam_id}',
                        'width': 1280,
                        'height': 720,
                        'fps': 30
                    })
                logger.info(f"Configured {len(self.cameras)} camera(s)")
        
        # Final summary
        logger.info(f"Camera Manager initialized with {len(self.cameras)} camera(s): {list(self.cameras.keys())}")
    
    def add_camera(self, config: Dict[str, Any]) -> bool:
        """Add a camera with the given configuration."""
        cap: Optional[cv2.VideoCapture] = None
        try:
            camera_id = config['id']
            
            # Initialize camera with DirectShow backend and timeout
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)  # 2 second timeout
            
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_id}")
                return False
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get('width', 1280))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get('height', 720))
            cap.set(cv2.CAP_PROP_FPS, config.get('fps', 30))
            
            # Optimize camera buffer for low latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer size for lowest latency
            
            # Create camera config
            camera_config = CameraConfig(
                camera_id=camera_id,
                name=config.get('name', f'Camera {camera_id}'),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=int(cap.get(cv2.CAP_PROP_FPS)),
                roi=config.get('roi')
            )
            
            self.cameras[camera_id] = camera_config
            self.captures[camera_id] = cap
            self.frame_queues[camera_id] = Queue(maxsize=1)  # Keep only latest frame for minimal latency
            
            logger.info(f"Added camera {camera_id}: {camera_config}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding camera {config.get('id', 'unknown')}: {str(e)}")
            if cap is not None and cap.isOpened():
                cap.release()
            return False
    
    def start(self) -> bool:
        """Start all camera capture threads."""
        if not self.cameras:
            logger.warning("No cameras configured")
            return False
        
        # Re-open cameras if they were closed
        for camera_id, camera_config in self.cameras.items():
            if camera_id not in self.captures or not self.captures[camera_id].isOpened():
                logger.info(f"Re-opening camera {camera_id}...")
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)
                
                if not cap.isOpened():
                    logger.error(f"Failed to re-open camera {camera_id}")
                    continue
                
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.height)
                cap.set(cv2.CAP_PROP_FPS, camera_config.fps)
                
                # Optimize camera buffer for low latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                self.captures[camera_id] = cap
                self.frame_queues[camera_id] = Queue(maxsize=1)  # Minimal queue for low latency
                logger.info(f"Successfully re-opened camera {camera_id}")
            
        self.running = True
        
        for camera_id in self.cameras:
            if camera_id in self.captures:
                thread = threading.Thread(
                    target=self._camera_loop,
                    args=(camera_id,),
                    daemon=True
                )
                self.threads.append(thread)
                thread.start()
        
        logger.info(f"Started {len(self.threads)} camera thread(s)")
        return True
    
    def stop(self) -> None:
        """Stop all camera capture threads and release resources."""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        # Release all camera resources but keep camera configs
        for camera_id, cap in list(self.captures.items()):
            if cap.isOpened():
                cap.release()
                logger.info(f"Released camera {camera_id}")
        
        self.threads.clear()
        # Don't clear captures and frame_queues - they'll be recreated on start()
        logger.info("All camera resources released")
    
    def _camera_loop(self, camera_id: int) -> None:
        """Main loop for capturing frames from a camera."""
        cap = self.captures[camera_id]
        frame_queue = self.frame_queues[camera_id]
        
        logger.info(f"Starting capture loop for camera {camera_id}")
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                logger.error(f"Failed to read frame from camera {camera_id}")
                time.sleep(1)  # Prevent tight loop on error
                continue
            
            # Apply ROI if specified
            roi = self.cameras[camera_id].roi
            if roi is not None:
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]
            
            # Add frame to queue, discarding old frames if queue is full (keep only latest)
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()  # Discard old frame
                except:
                    pass
            
            try:
                frame_queue.put_nowait(frame.copy())  # Non-blocking put
            except:
                pass  # Queue full, skip this frame
            
            # Minimal delay to prevent excessive CPU usage
            time.sleep(0.0001)  # Reduced from 0.001 to 0.0001
    
    def get_frame(self, camera_id: int, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get the latest frame from a camera with minimal latency."""
        if camera_id not in self.frame_queues:
            logger.error(f"No such camera: {camera_id}")
            return None
            
        try:
            # Try non-blocking first for lowest latency
            try:
                return self.frame_queues[camera_id].get_nowait()
            except:
                # Fall back to timeout if queue is empty
                return self.frame_queues[camera_id].get(timeout=timeout)
        except:
            return None
    
    def __del__(self):
        """Ensure resources are released when the object is destroyed."""
        self.stop()


def test_camera_manager():
    """Test function for the CameraManager class."""
    import cv2
    import time
    
    # Test configuration
    test_config = [
        {
            'id': 0,
            'name': 'Test Camera',
            'width': 1280,
            'height': 720,
            'fps': 30,
            'roi': None  # (x, y, w, h) or None for full frame
        }
    ]
    
    # Create and start camera manager
    manager = CameraManager(test_config)
    if not manager.cameras:
        print("No cameras available for testing")
        return
    
    try:
        manager.start()
        
        print("Press 'q' to quit")
        while True:
            for camera_id in manager.cameras:
                frame = manager.get_frame(camera_id)
                if frame is not None:
                    # Display the frame
                    cv2.imshow(f"Camera {camera_id}", frame)
            
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        manager.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the test
    test_camera_manager()