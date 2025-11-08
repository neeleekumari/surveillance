"""
Presence Tracker (Stabilized + Real-Time Safe)
----------------------------------------------
Tracks worker presence and activity duration over time.
Handles noisy detections, accumulates total time, and
triggers warning/alert transitions smoothly.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class WorkerPresence:
    """Tracks a worker’s presence session and status."""
    worker_id: int
    name: str
    first_detected: float = 0.0
    last_seen: float = 0.0
    total_presence_time: float = 0.0
    status: str = "absent"  # present / absent / exceeded
    detection_buffer: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    absence_start: float = 0.0
    last_absence_alert_time: float = 0.0

    def update(self, detected: bool, timestamp: float, grace_period: float = 5.0):
        """Update state based on detection at given timestamp."""
        if detected:
            self.detection_buffer.append(timestamp)
            if self.status == "absent":
                # Start a new session
                self.first_detected = timestamp
                self.status = "present"
                logger.debug(f"[{self.name}] detected → present at {timestamp:.1f}")
            self.last_seen = timestamp
        else:
            # Not detected; check for sustained absence
            if self.status == "present" and timestamp - self.last_seen > grace_period:
                session_time = self.last_seen - self.first_detected
                if session_time > 0:
                    self.total_presence_time += session_time
                self.status = "absent"
                self.absence_start = timestamp
                self.first_detected = 0.0
                logger.debug(f"[{self.name}] lost → absent ({self.total_presence_time:.1f}s total)")

    def _evaluate_status(self, current_time: float,
                         warn_th: float, alert_th: float) -> Tuple[str, float]:
        """Evaluate current state vs thresholds."""
        if self.status == "absent":
            return "absent", 0.0

        session_time = current_time - self.first_detected
        total_time = self.total_presence_time + session_time

        # Check thresholds
        if session_time >= alert_th:
            state = "exceeded"
        else:
            state = "present"

        return state, total_time

    def get_status_dict(self, current_time: float,
                        warn_th: float, alert_th: float) -> Dict:
        """Return serializable worker status."""
        state, total_time = self._evaluate_status(current_time, warn_th, alert_th)
        absence_duration = current_time - self.absence_start if self.status == "absent" and self.absence_start > 0 else 0.0
        return dict(
            worker_id=self.worker_id,
            name=self.name,
            status=state,
            session_time=max(0.0, current_time - self.first_detected if self.first_detected else 0.0),
            total_presence_time=total_time,
            last_seen=self.last_seen,
            absence_duration=absence_duration
        )


class PresenceTracker:
    """Tracks and maintains worker presence in real time."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.workers: Dict[int, WorkerPresence] = {}
        self.warning_threshold = self.config.get("warning_minutes", 30) * 60
        self.alert_threshold = self.config.get("alert_minutes", 60) * 60
        self.grace_period = self.config.get("grace_period", 5.0)
        logger.info(
            f"PresenceTracker initialized (warn={self.warning_threshold}s, "
            f"alert={self.alert_threshold}s, grace={self.grace_period}s)"
        )

    # --------------------------------------------------------------
    def add_worker(self, worker_id: int, name: str):
        if worker_id not in self.workers:
            self.workers[worker_id] = WorkerPresence(worker_id, name)
            logger.debug(f"Added worker {name} ({worker_id})")

    def remove_worker(self, worker_id: int):
        if worker_id in self.workers:
            del self.workers[worker_id]
            logger.debug(f"Removed worker {worker_id}")

    # --------------------------------------------------------------
    def update_detections(self, detections: List[dict]) -> Dict[int, dict]:
        """Process a list of YOLO/face recognition detections."""
        now = time.time()
        updates = {}

        # Track which workers were detected this frame
        detected = set(det["worker_id"] for det in detections if "worker_id" in det and det["worker_id"] is not None)

        # Ensure workers exist
        for wid in detected:
            if wid not in self.workers:
                self.add_worker(wid, f"Worker {wid}")

        # Update presence state for all
        for wid, worker in list(self.workers.items()):
            worker.update(wid in detected, now, grace_period=self.grace_period)
            state_dict = worker.get_status_dict(now, self.warning_threshold, self.alert_threshold)
            updates[wid] = state_dict

            # Handle status transitions
            prev_state = worker.status
            if state_dict["status"] != prev_state:
                logger.info(
                    f"[{worker.name}] {prev_state} → {state_dict['status']} "
                    f"({state_dict['session_time']:.1f}s active)"
                )
                worker.status = state_dict["status"]

        return updates

    # --------------------------------------------------------------
    def get_worker_status(self, worker_id: int) -> Optional[Dict]:
        if worker_id not in self.workers:
            return None
        now = time.time()
        worker = self.workers[worker_id]
        return worker.get_status_dict(now, self.warning_threshold, self.alert_threshold)

    def get_all_statuses(self) -> List[Dict]:
        now = time.time()
        return [w.get_status_dict(now, self.warning_threshold, self.alert_threshold)
                for w in self.workers.values()]

    def reset_worker(self, worker_id: int) -> bool:
        if worker_id in self.workers:
            w = self.workers[worker_id]
            w.first_detected = w.last_seen = 0.0
            w.total_presence_time = 0.0
            w.status = "absent"
            logger.info(f"Worker {w.name} ({worker_id}) reset.")
            return True
        return False


# --------------------------------------------------------------
# Test / Debug
# --------------------------------------------------------------
def test_presence_tracker():
    import random

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    tracker = PresenceTracker({"warning_minutes": 0.05, "alert_minutes": 0.1, "grace_period": 2})

    # Simulated workers
    worker_ids = [1, 2, 3]
    for wid in worker_ids:
        tracker.add_worker(wid, f"Worker {wid}")

    print("Running presence tracker test (Ctrl+C to stop)...")
    try:
        for i in range(30):
            detections = []
            for wid in worker_ids:
                if random.random() > 0.6:
                    detections.append({"worker_id": wid})
            updates = tracker.update_detections(detections)
            for wid, info in updates.items():
                print(f"{info['name']}: {info['status']}, total={info['total_presence_time']:.1f}s")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped.")

    print("\nFinal:")
    for s in tracker.get_all_statuses():
        print(s)