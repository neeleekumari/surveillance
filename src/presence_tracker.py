"""
Presence Tracker Module
----------------------
Tracks worker presence over time using detection data.
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Deque
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class WorkerPresence:
    """Tracks presence data for a single worker."""
    worker_id: int
    name: str
    last_seen: float = 0.0
    first_detected: float = 0.0
    total_presence_time: float = 0.0
    status: str = "absent"  # 'present', 'absent', 'exceeded'
    detection_history: Deque[Tuple[float, bool]] = field(default_factory=deque)
    
    def update_presence(self, detected: bool, timestamp: float) -> None:
        """Update presence status based on detection."""
        if detected:
            if self.status == "absent":
                # New detection
                self.first_detected = timestamp
                self.status = "present"
            self.last_seen = timestamp
        else:
            if self.status == "present" and timestamp - self.last_seen > 5.0:  # 5s cooldown
                # Update total presence time
                self.total_presence_time += (self.last_seen - self.first_detected)
                self.status = "absent"
                self.first_detected = 0.0
        
        # Store detection in history (sliding window)
        self.detection_history.append((timestamp, detected))
        if len(self.detection_history) > 100:  # Keep last 100 detections
            self.detection_history.popleft()
    
    def get_current_status(self, current_time: float, warning_threshold: float = 1800, 
                         alert_threshold: float = 3600) -> Tuple[str, float]:
        """Get current status and time since detection."""
        if self.status == "absent":
            return "absent", 0.0
        
        time_present = current_time - self.first_detected
        
        if time_present > alert_threshold:
            self.status = "exceeded"
        elif time_present > warning_threshold:
            self.status = "present"  # Could add 'warning' status
        
        return self.status, time_present


class PresenceTracker:
    """Tracks presence of multiple workers using detection data."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize the presence tracker."""
        self.config = config or {}
        self.workers: Dict[int, WorkerPresence] = {}
        self.detection_zones: Dict[int, Tuple[int, int, int, int]] = {}  # zone_id: (x1, y1, x2, y2)
        self.last_update = time.time()
        
        # Load configuration
        self.warning_threshold = self.config.get("warning_minutes", 30) * 60  # Convert to seconds
        self.alert_threshold = self.config.get("alert_minutes", 60) * 60  # Convert to seconds
        
        logger.info(f"Initialized PresenceTracker with warning={self.warning_threshold}s, "
                   f"alert={self.alert_threshold}s")
    
    def add_worker(self, worker_id: int, name: str) -> None:
        """Add a new worker to track."""
        if worker_id not in self.workers:
            self.workers[worker_id] = WorkerPresence(worker_id=worker_id, name=name)
            logger.info(f"Added worker: {name} (ID: {worker_id})")
    
    def remove_worker(self, worker_id: int) -> None:
        """Remove a worker from tracking."""
        if worker_id in self.workers:
            name = self.workers[worker_id].name
            del self.workers[worker_id]
            logger.info(f"Removed worker: {name} (ID: {worker_id})")
    
    def update_detections(self, detections: List[dict]) -> Dict[int, dict]:
        """Update worker presence based on new detections.
        
        Args:
            detections: List of detection dictionaries with 'worker_id' and 'confidence'
            
        Returns:
            Dictionary of worker_id to status updates
        """
        current_time = time.time()
        updates = {}
        detected_workers = set()
        
        # Process detections
        for det in detections:
            worker_id = det.get('worker_id')
            if worker_id is None:
                continue
                
            if worker_id not in self.workers:
                self.add_worker(worker_id, f"Worker {worker_id}")
            
            # Update presence for detected workers
            self.workers[worker_id].update_presence(True, current_time)
            detected_workers.add(worker_id)
        
        # Update status for all workers
        for worker_id, worker in self.workers.items():
            if worker_id not in detected_workers:
                worker.update_presence(False, current_time)
            
            # Get current status
            status, time_present = worker.get_current_status(
                current_time, 
                self.warning_threshold,
                self.alert_threshold
            )
            
            # Prepare update
            updates[worker_id] = {
                'status': status,
                'time_present': time_present,
                'last_seen': worker.last_seen,
                'name': worker.name
            }
            
            # Log status changes
            if status != worker.status:
                logger.info(f"Worker {worker.name} ({worker_id}): {worker.status} -> {status} "
                          f"({time_present:.1f}s)")
                worker.status = status
        
        self.last_update = current_time
        return updates
    
    def get_worker_status(self, worker_id: int) -> Optional[dict]:
        """Get current status of a worker."""
        if worker_id not in self.workers:
            return None
            
        worker = self.workers[worker_id]
        status, time_present = worker.get_current_status(
            time.time(),
            self.warning_threshold,
            self.alert_threshold
        )
        
        return {
            'worker_id': worker_id,
            'name': worker.name,
            'status': status,
            'time_present': time_present,
            'last_seen': worker.last_seen,
            'total_presence_time': worker.total_presence_time
        }
    
    def get_all_statuses(self) -> List[dict]:
        """Get status for all workers."""
        current_time = time.time()
        statuses = []
        
        for worker_id, worker in self.workers.items():
            status, time_present = worker.get_current_status(
                current_time,
                self.warning_threshold,
                self.alert_threshold
            )
            
            statuses.append({
                'worker_id': worker_id,
                'name': worker.name,
                'status': status,
                'time_present': time_present,
                'last_seen': worker.last_seen,
                'total_presence_time': worker.total_presence_time
            })
        
        return statuses
    
    def reset_worker(self, worker_id: int) -> bool:
        """Reset tracking for a worker."""
        if worker_id in self.workers:
            self.workers[worker_id].total_presence_time = 0.0
            self.workers[worker_id].first_detected = 0.0
            self.workers[worker_id].status = "absent"
            logger.info(f"Reset worker {worker_id}")
            return True
        return False


def test_presence_tracker():
    """Test function for the PresenceTracker class."""
    import random
    import time
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create tracker with 1 minute warning, 2 minute alert
    tracker = PresenceTracker({
        "warning_minutes": 0.05,  # 3 seconds for testing
        "alert_minutes": 0.1      # 6 seconds for testing
    })
    
    # Add some test workers
    test_workers = [
        (1, "Alice"),
        (2, "Bob"),
        (3, "Charlie")
    ]
    
    for worker_id, name in test_workers:
        tracker.add_worker(worker_id, name)
    
    print("Testing presence tracking. Press Ctrl+C to stop.")
    print("Workers will be detected randomly.")
    print("Alert threshold: 6s, Warning threshold: 3s")
    
    try:
        for i in range(30):  # Run for 30 seconds
            # Simulate random detections
            detections = []
            for worker_id, _ in test_workers:
                if random.random() > 0.7:  # 30% chance of detection
                    detections.append({"worker_id": worker_id, "confidence": 0.9})
            
            # Update tracker
            updates = tracker.update_detections(detections)
            
            # Print current status
            print(f"\n--- Update {i+1} ---")
            for worker_id, status in updates.items():
                print(f"{status['name']} ({worker_id}): {status['status']} "
                      f"({status['time_present']:.1f}s)")
            
            time.sleep(1)  # Simulate 1 second between updates
            
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    
    # Print final status
    print("\n--- Final Status ---")
    for status in tracker.get_all_statuses():
        print(f"{status['name']} ({status['worker_id']}): {status['status']} "
              f"(Total: {status['total_presence_time']:.1f}s)")


if __name__ == "__main__":
    test_presence_tracker()
