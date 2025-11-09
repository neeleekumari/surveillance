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
    total_absent_time: float = 0.0  # Total absent time in seconds for current absence session
    absent_start: Optional[float] = None  # Timestamp when current absent session started
    last_db_absent_minutes: int = 0  # Last minute value written to DB for this session
    detection_history: Deque[Tuple[float, bool]] = field(default_factory=deque)
    
    def update_presence(self, detected: bool, timestamp: float, db_manager=None) -> None:
        """Update presence status based on detection.

        New logic for absent-time:
        - Start absent timer only when the worker is absent and the current present session is 0:00.
        - While the worker is present, ensure absent counters are reset to 0 and DB is cleared.
        - While absent and the session has no prior presence (present time 0:00), count absent time from absent_start
          and update DB only when a full minute boundary is crossed (to avoid frequent writes).
        """
        if detected:
            # Worker is present: reset any absent session tracking and update presence timestamps
            if self.status == "absent":
                # Transition absent -> present
                self.first_detected = timestamp
                self.status = "present"
                # Reset absent session tracking
                self.absent_start = None
                self.last_db_absent_minutes = 0
                self.total_absent_time = 0.0
                if db_manager:
                    # Ensure DB shows 0:00 when present
                    try:
                        db_manager.set_worker_absent_time(self.worker_id, 0)
                    except Exception:
                        # Non-fatal: continue even if DB update fails
                        pass
            # Always update last_seen when detected
            self.last_seen = timestamp

        else:
            # Worker not detected
            # If worker was present but now absent, finalize presence and set status to absent
            if self.status == "present" and timestamp - self.last_seen > 5.0:
                # End the presence session
                self.total_presence_time += (self.last_seen - self.first_detected)
                self.status = "absent"
                self.first_detected = 0.0

            # Start absent timer only if there is no current presence session (i.e., present time == 0:00)
            if self.first_detected == 0.0:
                # If absent_start not set, start it now
                if self.absent_start is None:
                    self.absent_start = timestamp

                # Update current absent session duration
                self.total_absent_time = timestamp - self.absent_start

                # Write to DB only when a full minute increments since last write
                if db_manager:
                    elapsed_minutes = int(self.total_absent_time // 60)
                    if elapsed_minutes > self.last_db_absent_minutes:
                        try:
                            # Store the current session length in minutes (overwrite)
                            db_manager.set_worker_absent_time(self.worker_id, elapsed_minutes)
                            self.last_db_absent_minutes = elapsed_minutes
                        except Exception:
                            # Non-fatal; don't block presence tracking
                            pass

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
    
    def get_current_absent_time(self, current_time: float) -> float:
        """Get current total absent time including ongoing session."""
        total_absent = self.total_absent_time
        
        # Add current absent session time if worker is absent
        if self.status == "absent" and self.last_seen > 0:
            # Calculate time since last seen (current absent time)
            current_session_time = current_time - self.last_seen
            total_absent += current_session_time
        
        return total_absent


class PresenceTracker:
    """Tracks presence of multiple workers using detection data."""
    
    def __init__(self, config: Optional[dict] = None, db_manager=None):
        """Initialize the presence tracker."""
        self.config = config or {}
        self.workers: Dict[int, WorkerPresence] = {}
        self.detection_zones: Dict[int, Tuple[int, int, int, int]] = {}  # zone_id: (x1, y1, x2, y2)
        self.last_update = time.time()
        self.db_manager = db_manager
        
        # Load configuration
        self.warning_threshold = self.config.get("warning_minutes", 30) * 60  # Convert to seconds
        self.alert_threshold = self.config.get("alert_minutes", 60) * 60  # Convert to seconds
        
        # Load all workers from database at initialization
        if self.db_manager:
            self.load_all_workers()
        
        logger.info(f"Initialized PresenceTracker with warning={self.warning_threshold}s, "
                   f"alert={self.alert_threshold}s")
    
    def add_worker(self, worker_id: int, name: str) -> None:
        """Add a new worker to track."""
        if worker_id not in self.workers:
            current_time = time.time()
            worker = WorkerPresence(worker_id=worker_id, name=name)
            # Initialize absent_start for workers who have never been detected
            # This ensures absent time starts counting from when they're added
            if worker.last_seen == 0.0 and worker.first_detected == 0.0:
                worker.absent_start = current_time
                worker.status = "absent"
            self.workers[worker_id] = worker
            logger.info(f"Added worker: {name} (ID: {worker_id})")
    
    def remove_worker(self, worker_id: int) -> None:
        """Remove a worker from tracking."""
        if worker_id in self.workers:
            name = self.workers[worker_id].name
            del self.workers[worker_id]
            logger.info(f"Removed worker: {name} (ID: {worker_id})")
    
    def load_all_workers(self) -> None:
        """Load all registered workers from the database (ONLY workers with face embeddings)."""
        if not self.db_manager:
            logger.warning("No database manager available to load workers")
            return
            
        try:
            # CRITICAL FIX: Only load workers who have face embeddings
            # Workers without embeddings cannot be recognized, so don't track them
            embeddings_data = self.db_manager.get_all_face_embeddings()
            
            if not embeddings_data:
                logger.info("No workers with face embeddings found in database")
                return
            
            # Get unique workers from embeddings (workers with face recognition capability)
            workers_with_embeddings = {}
            for emb in embeddings_data:
                worker_id = emb['worker_id']
                worker_name = emb['worker_name']
                if worker_id not in workers_with_embeddings:
                    workers_with_embeddings[worker_id] = worker_name
            
            # Add only workers who have embeddings
            for worker_id, worker_name in workers_with_embeddings.items():
                if worker_id not in self.workers:
                    self.add_worker(worker_id, worker_name)
                    logger.info(f"Loaded worker with embeddings: {worker_name} (ID: {worker_id})")
            
            logger.info(f"Loaded {len(workers_with_embeddings)} worker(s) with face embeddings")
            
        except Exception as e:
            logger.error(f"Failed to load workers from database: {e}")
    
    def update_detections(self, detections: List[dict]) -> Dict[int, dict]:
        """Update worker presence based on new detections.
        
        Args:
            detections: List of detection dictionaries with 'worker_id' and 'confidence'
            
        Returns:
            Dictionary of worker_id to status updates
        """
        current_time = time.time()
        updates = {}
        detected_workers = {d['worker_id'] for d in detections if 'worker_id' in d}
        
        # Process detections
        for det in detections:
            worker_id = det.get('worker_id')
            if worker_id is None:
                continue
            
            # CRITICAL FIX: Don't auto-create workers for unknown detections
            # Only track workers who are already registered (have embeddings)
            # Unknown people should remain unknown, not get random worker IDs
            if worker_id not in self.workers:
                logger.debug(f"Skipping unknown worker ID {worker_id} - not registered")
                continue
            
            # Update presence for detected workers (only registered ones)
            self.workers[worker_id].update_presence(True, current_time, self.db_manager)
            detected_workers.add(worker_id)
        
        # Update status for all workers
        for worker_id, worker in self.workers.items():
            if worker_id not in detected_workers:
                worker.update_presence(False, current_time, self.db_manager)
            else:
                # Ensure we call update_presence for detected workers (already done above during processing)
                pass

            # Get current status
            status, time_present = worker.get_current_status(
                current_time,
                self.warning_threshold,
                self.alert_threshold
            )

            # Calculate current absent time (including ongoing session)
            # This is the REAL-TIME absent time for alert checking
            current_absent_time = worker.total_absent_time
            if status == "absent":
                if worker.absent_start is not None:
                    # Worker has an active absent session - calculate real-time absent time
                    current_absent_time = current_time - worker.absent_start
                elif worker.last_seen > 0:
                    # Worker was detected before but now absent
                    # Calculate time since last seen
                    current_absent_time = current_time - worker.last_seen
                # If both are 0, worker has never been detected - use stored total_absent_time
            
            # Prepare update
            updates[worker_id] = {
                'status': status,
                'time_present': time_present,
                'last_seen': worker.last_seen,
                'name': worker.name,
                # total_absent_time stored in seconds in memory (return seconds for UI)
                'total_absent_time': int(current_absent_time)
            }

            # Log status changes (and sync to database activity_log)
            if status != worker.status:
                logger.info(f"Worker {worker.name} ({worker.worker_id}): {worker.status} -> {status} "
                          f"({time_present:.1f}s)")
                # Write activity event to DB once per transition
                if self.db_manager:
                    try:
                        duration = int(time_present) if status in ("present", "exceeded") else int(current_absent_time)
                        self.db_manager.log_activity(worker.worker_id, status, duration_seconds=duration)
                    except Exception as e:
                        logger.warning(f"Failed to log activity for {worker.name} ({worker.worker_id}): {e}")
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
            'total_presence_time': worker.total_presence_time,
            # Return total_absent_time in seconds for live UI formatting
            'total_absent_time': int(worker.total_absent_time)
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
            
            # Calculate real-time absent time (same logic as in update_detections)
            current_absent_time = worker.total_absent_time
            if status == "absent":
                if worker.absent_start is not None:
                    # Worker has an active absent session - calculate real-time absent time
                    current_absent_time = current_time - worker.absent_start
                elif worker.last_seen > 0:
                    # Worker was detected before but now absent
                    current_absent_time = current_time - worker.last_seen
            
            statuses.append({
                'worker_id': worker_id,
                'name': worker.name,
                'status': status,
                'time_present': time_present,
                'last_seen': worker.last_seen,
                'total_presence_time': worker.total_presence_time,
                # Return total_absent_time in seconds for live UI formatting (real-time calculation)
                'total_absent_time': int(current_absent_time)
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
