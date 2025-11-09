"""
Floor Monitoring Desktop Application
----------------------------------
This application monitors worker presence using USB cameras and YOLOv8 for person detection.
"""
import sys
import json
import logging
import time
from pathlib import Path
from typing import Optional
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
import numpy as np

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

# Import local modules
from src.database_module import DatabaseManager
from src.detection_module import PersonDetector, Detection
from src.camera_manager import CameraManager
from src.presence_tracker import PresenceTracker
from src.alert_manager import AlertManager
from src.ui_manager import UIManager
from src.config_manager import ConfigManager
from src.face_recognition_module import FaceRecognitionSystem, crop_face_from_detection

# Configure logging with UTF-8 encoding for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Fix Windows console encoding for Unicode support
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass  # Ignore if reconfigure not available
logger = logging.getLogger(__name__)

class DetectionThread(QThread):
    """Thread for running object detection and face recognition to avoid blocking the UI."""
    detection_complete = pyqtSignal(list, int)  # detections, camera_id
    
    def __init__(self, detector: PersonDetector, camera_manager: CameraManager, face_system: Optional[FaceRecognitionSystem] = None, db_manager: Optional[DatabaseManager] = None):
        super().__init__()
        self.detector = detector
        self.camera_manager = camera_manager
        self.face_system = face_system
        self.db_manager = db_manager
        self.running = True
        self.frame_count = 0
        self.face_recognition_interval = 3  # Run face recognition every 3 frames (increased frequency)
        
        # Get stability tracker from face recognition system
        self.stability_tracker = face_system.stability_tracker if face_system else None
        
        # PROPER TRACKING SYSTEM WITH CONFIDENCE-BASED LOCKING
        self.next_tracker_id = 1  # Unique ID counter for each detected person
        self.active_tracks = {}  # tracker_id -> {bbox, worker_id, worker_name, last_seen, confidence, locked, recognition_history}
        self.worker_id_assignments = {}  # worker_id -> tracker_id (ensure one worker = one track)
        self.recognition_attempts = {}  # tracker_id -> attempt_count
        
        # TRACKING PARAMETERS - Optimized for MOVING workers
        self.iou_threshold = 0.20  # IoU threshold for matching (lower for moving workers)
        self.max_centroid_distance = 150  # Max pixels between centroids for same person (NEW)
        
        # LOCKING PARAMETERS - Optimized for moving workers
        self.min_confidence_to_lock = 0.65  # 65% average confidence (lowered for moving workers)
        self.confirmations_needed = 3  # 3 consistent recognitions (reduced from 5 for faster locking)
        self.allow_correction_below = 0.55  # Allow corrections if confidence is below 55%
        self.min_face_quality = 35.0  # Minimum face quality (lowered for moving workers - was 40)
        self.recheck_interval = 30  # Recheck locked tracks every 30 frames
        
        # PERSISTENT TRACKING - Track workers even when temporarily out of view
        self.track_memory = {}  # worker_id -> {last_track_id, last_seen, lock_confidence}
        self.track_memory_duration = 15.0  # Remember tracks for 15 seconds (increased for moving workers)
        self.max_track_age = 2.0  # Keep tracks alive for 2 seconds (reduced - create new track faster if person moves)
        self.last_recheck = {}  # tracker_id -> last_recheck_frame
        
        # UNKNOWN PERSON TRACKING - Track unregistered people consistently
        self.unknown_persons = {}  # unknown_id -> {embedding, last_seen, track_id, name}
        self.next_unknown_id = 1  # Counter for Unknown-1, Unknown-2, etc.
        self.unknown_similarity_threshold = 0.75  # Match threshold for same unknown person
        self.unknown_memory_duration = 30.0  # Remember unknown persons for 30 seconds
    
    def match_unknown_person(self, face_embedding, current_time):
        """
        Match unknown person by face embedding and assign consistent ID.
        
        Args:
            face_embedding: 512-D face embedding from DeepFace
            current_time: Current timestamp
            
        Returns:
            (unknown_id, unknown_name) - e.g., (1, "Unknown-1")
        """
        if face_embedding is None:
            # No embedding - assign new unknown ID
            unknown_id = self.next_unknown_id
            self.next_unknown_id += 1
            unknown_name = f"Unknown-{unknown_id}"
            return unknown_id, unknown_name
        
        # Normalize embedding
        face_embedding = face_embedding / (np.linalg.norm(face_embedding) + 1e-8)
        
        # Clean up old unknown persons
        expired_ids = []
        for uid, data in self.unknown_persons.items():
            if current_time - data['last_seen'] > self.unknown_memory_duration:
                expired_ids.append(uid)
        
        for uid in expired_ids:
            logger.info(f"Unknown person {self.unknown_persons[uid]['name']} expired from memory")
            del self.unknown_persons[uid]
        
        # Try to match with existing unknown persons
        best_match_id = None
        best_similarity = 0.0
        
        for uid, data in self.unknown_persons.items():
            stored_emb = data['embedding']
            # Calculate cosine similarity
            similarity = np.dot(face_embedding, stored_emb)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = uid
        
        # If good match found, reuse that unknown ID
        if best_match_id and best_similarity >= self.unknown_similarity_threshold:
            self.unknown_persons[best_match_id]['last_seen'] = current_time
            unknown_name = self.unknown_persons[best_match_id]['name']
            logger.debug(f"Matched to existing {unknown_name} (similarity: {best_similarity:.1%})")
            return best_match_id, unknown_name
        
        # No match - create new unknown person
        unknown_id = self.next_unknown_id
        self.next_unknown_id += 1
        unknown_name = f"Unknown-{unknown_id}"
        
        self.unknown_persons[unknown_id] = {
            'embedding': face_embedding,
            'last_seen': current_time,
            'name': unknown_name
        }
        
        logger.info(f"New unknown person detected: {unknown_name}")
        return unknown_id, unknown_name
    
    def check_face_quality(self, face_img):
        """Check if face image has sufficient quality for recognition (RELAXED for multi-angle)."""
        if face_img is None or face_img.size == 0:
            return False, 0.0
        
        # Convert to grayscale for analysis
        if len(face_img.shape) == 3:
            gray = np.mean(face_img, axis=2).astype(np.uint8)
        else:
            gray = face_img
        
        # Check brightness (avoid too dark or too bright) - RELAXED
        brightness = np.mean(gray)
        if brightness < 25 or brightness > 235:  # Wider range (was 40-220)
            logger.debug(f"Face rejected: brightness {brightness:.1f} out of range [25-235]")
            return False, brightness
        
        # Check contrast (Laplacian variance for sharpness) - RELAXED
        laplacian_var = np.var(gray)
        if laplacian_var < 50:  # Reduced from 80 to accept more angles
            logger.debug(f"Face rejected: too blurry (variance {laplacian_var:.1f} < 50)")
            return False, brightness
        
        # Check size (face should be large enough) - RELAXED
        if face_img.shape[0] < 60 or face_img.shape[1] < 60:  # Reduced from 80x80
            logger.debug(f"Face rejected: too small {face_img.shape} (min 60x60)")
            return False, brightness
        
        # Calculate quality score with relaxed weighting
        quality_score = min(100, (brightness / 2.5) + (laplacian_var / 3))
        is_acceptable = quality_score >= self.min_face_quality
        
        if not is_acceptable:
            logger.debug(f"Face quality score {quality_score:.1f} < {self.min_face_quality}")
        
        return is_acceptable, quality_score
    
    def enhance_face_for_low_light(self, face_img):
        """Enhance face image for better recognition in low light."""
        if face_img is None:
            return None
        
        # Convert to LAB color space
        lab = np.zeros_like(face_img)
        if len(face_img.shape) == 3:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            lab_img = face_img.copy()
            
            # Simple brightness adjustment
            brightness = np.mean(face_img)
            if brightness < 80:  # Dark image
                # Increase brightness
                alpha = 1.5  # Contrast
                beta = 30    # Brightness
                enhanced = np.clip(alpha * face_img + beta, 0, 255).astype(np.uint8)
                return enhanced
            elif brightness > 180:  # Too bright
                # Decrease brightness
                alpha = 0.8
                beta = -20
                enhanced = np.clip(alpha * face_img + beta, 0, 255).astype(np.uint8)
                return enhanced
        
        return face_img
    
    def calculate_centroid(self, bbox):
        """Calculate centroid of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def calculate_centroid_distance(self, bbox1, bbox2):
        """Calculate Euclidean distance between centroids of two bounding boxes."""
        c1 = self.calculate_centroid(bbox1)
        c2 = self.calculate_centroid(bbox2)
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def match_detections_to_tracks(self, detections, current_time):
        """Match new detections to existing tracks using IoU and centroid distance."""
        matched_tracks = set()
        
        for det in detections:
            best_match_id = None
            best_score = 0.0  # Combined score
            
            # Find best matching track using both IoU and centroid distance
            for track_id, track_data in self.active_tracks.items():
                if track_id in matched_tracks:
                    continue  # Already matched
                
                # Calculate IoU
                iou = self.calculate_iou(det.bbox, track_data['bbox'])
                
                # Calculate centroid distance (normalized)
                centroid_dist = self.calculate_centroid_distance(det.bbox, track_data['bbox'])
                
                # If IoU is good, prefer it
                if iou > self.iou_threshold:
                    score = iou
                # Fallback to centroid distance for moving workers
                elif centroid_dist < self.max_centroid_distance:
                    # Convert distance to similarity score (closer = higher score)
                    score = 1.0 - (centroid_dist / self.max_centroid_distance)
                else:
                    score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_match_id = track_id
            
            if best_match_id:
                # Update existing track
                det.tracker_id = best_match_id
                self.active_tracks[best_match_id]['bbox'] = det.bbox
                self.active_tracks[best_match_id]['last_seen'] = current_time
                
                # CRITICAL FIX: DO NOT apply cached identity automatically
                # Identity must be re-verified through face recognition system
                # Only use cached identity if track is LOCKED with high confidence
                track_data = self.active_tracks[best_match_id]
                if track_data.get('locked', False) and track_data.get('confidence', 0.0) >= 0.80:
                    # Track is locked with high confidence - use cached identity
                    det.worker_id = track_data['worker_id']
                    det.worker_name = track_data['worker_name']
                    det.recognition_score = track_data['confidence']
                # Otherwise, identity will be determined by face recognition in next step
                
                matched_tracks.add(best_match_id)
            else:
                # Check track memory before creating new track (for workers who moved out and back)
                reused_worker_id = None
                for worker_id, memory in self.track_memory.items():
                    if current_time - memory['last_seen'] < self.track_memory_duration:
                        # Worker was recently seen - might be same person who moved
                        # Check if worker_id is not currently assigned to another active track
                        if worker_id not in self.worker_id_assignments:
                            # Reuse this worker ID (will be verified by face recognition)
                            reused_worker_id = worker_id
                            logger.info(f"Potential re-appearance of worker {memory['worker_name']} (ID: {worker_id})")
                            break
                
                # Create new track
                det.tracker_id = self.next_tracker_id
                self.active_tracks[self.next_tracker_id] = {
                    'bbox': det.bbox,
                    'worker_id': None,
                    'worker_name': None,
                    'confidence': 0.0,
                    'last_seen': current_time,
                    'locked': False,  # Not locked initially
                    'recognition_history': []  # Track recognition attempts
                }
                self.recognition_attempts[self.next_tracker_id] = 0
                self.next_tracker_id += 1
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track_data in self.active_tracks.items():
            if current_time - track_data['last_seen'] > self.max_track_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            # Save to track memory if worker was locked
            track_data = self.active_tracks[track_id]
            worker_id = track_data['worker_id']
            
            if worker_id and track_data.get('locked', False):
                # Save to memory for potential re-appearance
                self.track_memory[worker_id] = {
                    'last_track_id': track_id,
                    'last_seen': current_time,
                    'lock_confidence': track_data['confidence'],
                    'worker_name': track_data['worker_name']
                }
                logger.info(f"Track {track_id} ({track_data['worker_name']}) saved to memory")
            
            # Free up worker ID assignment
            if worker_id and worker_id in self.worker_id_assignments:
                if self.worker_id_assignments[worker_id] == track_id:
                    del self.worker_id_assignments[worker_id]
            
            # Clean up stability tracker
            if self.stability_tracker:
                self.stability_tracker.reset_track(track_id)
            
            del self.active_tracks[track_id]
            if track_id in self.recognition_attempts:
                del self.recognition_attempts[track_id]
        
            # Clean up old track memories (do not update DB here - presence tracker handles absent-time updates)
            memories_to_remove = []
            for worker_id, memory in self.track_memory.items():
                time_since_seen = current_time - memory['last_seen']
                if time_since_seen > self.track_memory_duration:
                    memories_to_remove.append(worker_id)

            for worker_id in memories_to_remove:
                logger.info(f"Track memory expired for worker {self.track_memory[worker_id]['worker_name']}")
                del self.track_memory[worker_id]
            
            # Clean up inactive tracks from stability tracker
            if self.stability_tracker:
                active_track_ids = list(self.active_tracks.keys())
                self.stability_tracker.cleanup_old_tracks(active_track_ids)
            
    def run(self):
        """Main detection loop with PROPER multi-worker tracking."""
        while self.running:
            # Process frames from all cameras
            for camera_id in self.camera_manager.cameras:
                frame = self.camera_manager.get_frame(camera_id, timeout=0.05)
                if frame is not None:
                    detections = self.detector.detect(frame)
                    current_time = time.time()
                    
                    # STEP 1: Match detections to existing tracks using IoU
                    self.match_detections_to_tracks(detections, current_time)
                    
                    # STEP 2: Perform face recognition on unidentified tracks
                    self.frame_count += 1
                    should_recognize = (self.frame_count % self.face_recognition_interval == 0)
                    
                    if self.face_system and detections and should_recognize:
                        for det in detections:
                            track_data = self.active_tracks.get(det.tracker_id, {})
                            is_locked = track_data.get('locked', False)
                            current_confidence = track_data.get('confidence', 0.0)
                            
                            # Skip if locked with high confidence (unless recheck interval passed)
                            if is_locked and current_confidence >= self.min_confidence_to_lock:
                                # Periodically recheck locked tracks
                                last_check = self.last_recheck.get(det.tracker_id, 0)
                                if self.frame_count - last_check < self.recheck_interval:
                                    continue
                                else:
                                    self.last_recheck[det.tracker_id] = self.frame_count
                                    logger.debug(f"Rechecking locked track {det.tracker_id}")
                            
                            # Allow re-recognition if confidence is low (might be wrong)
                            if det.worker_id is not None and current_confidence >= self.allow_correction_below:
                                continue
                            
                            # Limit recognition attempts per track
                            if det.tracker_id and self.recognition_attempts.get(det.tracker_id, 0) >= 10:
                                continue  # Stop trying after 10 failed attempts
                            
                            try:
                                # Crop face from detection
                                face_img = crop_face_from_detection(frame, det.bbox)
                                
                                if face_img is not None:
                                    # STEP 1: Check face quality
                                    is_good_quality, quality_score = self.check_face_quality(face_img)
                                    
                                    if not is_good_quality:
                                        logger.debug(f"Track {det.tracker_id}: Face quality too low ({quality_score:.1f})")
                                        # Try enhancement for low light
                                        face_img = self.enhance_face_for_low_light(face_img)
                                        is_good_quality, quality_score = self.check_face_quality(face_img)
                                        
                                        if not is_good_quality:
                                            # Still bad quality, skip this frame
                                            if det.tracker_id:
                                                self.recognition_attempts[det.tracker_id] = self.recognition_attempts.get(det.tracker_id, 0) + 1
                                            continue
                                    
                                    logger.debug(f"Track {det.tracker_id}: Face quality OK ({quality_score:.1f})")
                                    
                                    # STEP 2: Recognize worker with quality check enabled
                                    result = self.face_system.recognize_worker(face_img, check_quality=True)
                                    
                                    if result:
                                        worker_id, worker_name, similarity = result
                                        
                                        # IMPROVED: Use stability tracker to VERIFY consistency, not block recognition
                                        # Show recognized name immediately if it passes two-step verification
                                        # But track history to detect and reject inconsistent/flickering recognition
                                        if self.stability_tracker and det.tracker_id:
                                            # Add to history
                                            if det.tracker_id not in self.stability_tracker.track_history:
                                                self.stability_tracker.track_history[det.tracker_id] = []
                                            
                                            history = self.stability_tracker.track_history[det.tracker_id]
                                            history.append((worker_id, worker_name, similarity))
                                            
                                            # Keep last 5 detections
                                            if len(history) > 5:
                                                self.stability_tracker.track_history[det.tracker_id] = history[-5:]
                                            
                                            # Check for RAPID flickering (multiple switches in short time)
                                            # Allow different people over time, but reject rapid alternation
                                            if len(history) >= 3:
                                                recent = history[-3:]  # Last 3 detections
                                                recent_ids = [h[0] for h in recent]
                                                recent_confidences = [h[2] for h in recent]
                                                
                                                # Count ID switches (transitions between different IDs)
                                                switches = sum(1 for i in range(len(recent_ids)-1) if recent_ids[i] != recent_ids[i+1])
                                                avg_confidence = sum(recent_confidences) / len(recent_confidences)
                                                
                                                # Only reject if RAPID alternation (2+ switches) AND low confidence
                                                # This allows: [neel, neel, dutta] ✅ (1 switch, person changed)
                                                # But rejects: [neel, dutta, neel] ❌ (2 switches, flickering)
                                                if switches >= 2 and avg_confidence < 0.85:
                                                    logger.warning(
                                                        f"[UNSTABLE] Track {det.tracker_id} rapid flickering: {[h[1] for h in recent]} "
                                                        f"({switches} switches, avg conf: {avg_confidence:.1%}) - Rejecting"
                                                    )
                                                    det.worker_id = None
                                                    det.worker_name = "Unknown"
                                                    det.recognition_score = 0.0
                                                    continue
                                        
                                        # If we got here, recognition is valid - use it immediately
                                        logger.debug(f"Track {det.tracker_id}: Recognized as {worker_name} ({similarity:.1%})")
                                        
                                        # Check if this worker is in track memory (recently disappeared)
                                        if worker_id in self.track_memory:
                                            memory = self.track_memory[worker_id]
                                            time_since_last = current_time - memory['last_seen']
                                            
                                            if time_since_last < self.track_memory_duration:
                                                # Worker reappeared! Restore locked status
                                                logger.info(f"[RESTORED] {worker_name} reappeared after {time_since_last:.1f}s")
                                                
                                                # Immediately assign and lock
                                                det.worker_id = worker_id
                                                det.worker_name = worker_name
                                                det.recognition_score = similarity
                                                
                                                if det.tracker_id:
                                                    self.active_tracks[det.tracker_id]['worker_id'] = worker_id
                                                    self.active_tracks[det.tracker_id]['worker_name'] = worker_name
                                                    self.active_tracks[det.tracker_id]['confidence'] = max(similarity, memory['lock_confidence'])
                                                    self.active_tracks[det.tracker_id]['locked'] = True  # Restore lock
                                                    self.worker_id_assignments[worker_id] = det.tracker_id
                                                    
                                                    # Add to history with high confidence
                                                    history = [{
                                                        'worker_id': worker_id,
                                                        'name': worker_name,
                                                        'score': memory['lock_confidence'],
                                                        'time': current_time
                                                    }] * self.confirmations_needed  # Fill history
                                                    self.active_tracks[det.tracker_id]['recognition_history'] = history
                                                
                                                # Remove from memory
                                                del self.track_memory[worker_id]
                                                
                                                logger.info(f"[LOCKED-RESTORED] {worker_name} (ID:{worker_id}, Track:{det.tracker_id}, Score:{similarity:.3f})")
                                                continue  # Skip normal processing
                                        
                                        # Add to recognition history
                                        if det.tracker_id:
                                            history = self.active_tracks[det.tracker_id].get('recognition_history', [])
                                            history.append({'worker_id': worker_id, 'name': worker_name, 'score': similarity, 'time': current_time})
                                            # Keep only last 10 recognitions (increased from 5)
                                            self.active_tracks[det.tracker_id]['recognition_history'] = history[-10:]
                                            
                                            # Check consistency - are last N recognitions the same?
                                            recent = history[-self.confirmations_needed:]
                                            if len(recent) >= self.confirmations_needed:
                                                same_worker = all(r['worker_id'] == worker_id for r in recent)
                                                avg_score = sum(r['score'] for r in recent) / len(recent)
                                                min_score_in_history = min(r['score'] for r in recent)
                                                max_score_in_history = max(r['score'] for r in recent)
                                                score_std = np.std([r['score'] for r in recent])
                                                
                                                # DEBUG: Log recognition history to identify shuffling
                                                if not same_worker:
                                                    worker_ids_in_history = [r['worker_id'] for r in recent]
                                                    worker_names_in_history = [r['name'] for r in recent]
                                                    logger.warning(
                                                        f"[!] SHUFFLING DETECTED on Track {det.tracker_id}! "
                                                        f"IDs: {worker_ids_in_history}, Names: {worker_names_in_history}"
                                                    )
                                                
                                                # BALANCED LOCKING: Good scores AND consistency (optimized for moving workers)
                                                # Require: same worker + good average + reasonable minimum + acceptable variance
                                                if (same_worker and 
                                                    avg_score >= self.min_confidence_to_lock and 
                                                    min_score_in_history >= 0.55 and  # ALL scores must be 55%+ (lowered for moving workers)
                                                    score_std < 0.10):  # Slightly relaxed variance (was 0.08)
                                                    should_lock = True
                                                    lock_status = "LOCKED"
                                                    logger.info(f"  -> Lock criteria met: avg={avg_score:.3f}, min={min_score_in_history:.3f}, std={score_std:.3f}")
                                                elif same_worker and avg_score >= 0.65 and min_score_in_history >= 0.50:
                                                    should_lock = False
                                                    lock_status = "CONFIRMED"
                                                elif same_worker and avg_score >= 0.55 and min_score_in_history >= 0.45:
                                                    should_lock = False
                                                    lock_status = "TENTATIVE"
                                                else:
                                                    should_lock = False
                                                    lock_status = "UNCERTAIN"
                                                    if not same_worker:
                                                        lock_status = "SHUFFLING"  # Indicate shuffling issue
                                                    else:
                                                        logger.debug(f"  -> Lock criteria NOT met: avg={avg_score:.3f}, min={min_score_in_history:.3f}, std={score_std:.3f}")
                                            else:
                                                should_lock = False
                                                lock_status = "DETECTING"
                                        
                                        # FIXED: Allow same worker on different tracks if they're far apart
                                        # This handles cases where the same person moves or multiple people are present
                                        if worker_id in self.worker_id_assignments:
                                            existing_track_id = self.worker_id_assignments[worker_id]
                                            
                                            # If existing track is still active
                                            if existing_track_id in self.active_tracks and existing_track_id != det.tracker_id:
                                                existing_locked = self.active_tracks[existing_track_id].get('locked', False)
                                                existing_conf = self.active_tracks[existing_track_id]['confidence']
                                                
                                                # Check if this is the same person moving or a different detection
                                                # If tracks are close (IoU > 0.1), it's likely the same person
                                                # If tracks are far apart, might be misidentification
                                                existing_bbox = self.active_tracks[existing_track_id]['bbox']
                                                current_bbox = det.bbox
                                                track_iou = self.calculate_iou(existing_bbox, current_bbox)
                                                
                                                if track_iou > 0.1:
                                                    # Same person, same location - keep existing track
                                                    logger.debug(f"Same person {worker_name} detected on overlapping tracks {existing_track_id} and {det.tracker_id}")
                                                    continue
                                                
                                                # Tracks are far apart - could be:
                                                # 1. Same person moved (reassign)
                                                # 2. Misidentification (need higher confidence to override)
                                                
                                                if existing_locked and existing_conf >= self.min_confidence_to_lock:
                                                    # Existing track is confident - need much better score to override
                                                    if similarity > existing_conf + 0.15:
                                                        logger.info(f"OVERRIDE: {worker_name} moved from track {existing_track_id} to {det.tracker_id} (higher: {similarity:.3f} vs {existing_conf:.3f})")
                                                        # Unlock old track
                                                        self.active_tracks[existing_track_id]['locked'] = False
                                                        self.active_tracks[existing_track_id]['worker_id'] = None
                                                        self.active_tracks[existing_track_id]['worker_name'] = None
                                                    else:
                                                        # Keep existing track, reject this one
                                                        logger.debug(f"Keeping {worker_name} on track {existing_track_id} (conf: {existing_conf:.3f} vs {similarity:.3f})")
                                                        continue
                                                else:
                                                    # Existing track not locked or low confidence - reassign
                                                    logger.debug(f"Reassigned: {worker_name} from track {existing_track_id} to {det.tracker_id}")
                                                    self.active_tracks[existing_track_id]['worker_id'] = None
                                                    self.active_tracks[existing_track_id]['worker_name'] = None
                                        
                                        # Assign worker to this track
                                        det.worker_id = worker_id
                                        det.worker_name = worker_name
                                        det.recognition_score = similarity
                                        
                                        # Update track data
                                        if det.tracker_id:
                                            self.active_tracks[det.tracker_id]['worker_id'] = worker_id
                                            self.active_tracks[det.tracker_id]['worker_name'] = worker_name
                                            self.active_tracks[det.tracker_id]['confidence'] = similarity
                                            self.active_tracks[det.tracker_id]['locked'] = should_lock
                                            self.worker_id_assignments[worker_id] = det.tracker_id
                                        
                                        # Only log when locked or first detection
                                        if should_lock or lock_status == "DETECTING":
                                            logger.info(f"[{lock_status}] {worker_name} (ID:{worker_id}, Track:{det.tracker_id}, Score:{similarity:.3f})")
                                    else:
                                        # Recognition failed - track as unknown person
                                        if det.tracker_id:
                                            self.recognition_attempts[det.tracker_id] = self.recognition_attempts.get(det.tracker_id, 0) + 1
                                            
                                            # Generate embedding for unknown person tracking
                                            try:
                                                unknown_embedding = self.face_system.get_face_embedding(face_img)
                                                unknown_id, unknown_name = self.match_unknown_person(unknown_embedding, current_time)
                                                
                                                # Assign unknown person ID
                                                det.worker_id = -unknown_id  # Negative ID for unknown persons
                                                det.worker_name = unknown_name
                                                det.recognition_score = 0.0
                                                
                                                # Update track data
                                                self.active_tracks[det.tracker_id]['worker_id'] = -unknown_id
                                                self.active_tracks[det.tracker_id]['worker_name'] = unknown_name
                                                self.active_tracks[det.tracker_id]['confidence'] = 0.0
                                                
                                                logger.debug(f"Track {det.tracker_id} assigned to {unknown_name}")
                                            except Exception as e:
                                                logger.debug(f"Failed to track unknown person: {e}")
                                                # Fallback to generic "Unknown"
                                                det.worker_name = "Unknown"
                                            
                            except Exception as e:
                                logger.error(f"Face recognition error: {e}")
                    
                    self.detection_complete.emit(detections, camera_id)
            # Minimal delay to prevent excessive CPU usage
            self.msleep(5)
    
    def stop(self):
        """Stop the detection thread."""
        self.running = False

class FloorMonitoringApp:
    """Main application class for the Floor Monitoring System."""
    
    def __init__(self):
        """Initialize the application."""
        self.app = QApplication(sys.argv)
        self.db: Optional[DatabaseManager] = None
        self.camera_manager: Optional[CameraManager] = None
        self.detector: Optional[PersonDetector] = None
        self.face_system: Optional[FaceRecognitionSystem] = None
        self.presence_tracker: Optional[PresenceTracker] = None
        self.alert_manager: Optional[AlertManager] = None
        self.ui: Optional[UIManager] = None
        self.detection_thread: Optional[DetectionThread] = None
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        
        # Initialize components
        self._initialize_components()
        
        # Set up application timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1000)  # Update every second
        
        # Track last DB sync time for worker synchronization
        self.last_worker_sync = 0
        self.worker_sync_interval = 5  # Sync with DB every 5 seconds for near real-time updates
        
        # Track workers already alerted for 1-minute absence in their current absence session
        self.absence_alerted_workers = set()
        
    def _load_config(self) -> dict:
        """Load application configuration."""
        return self.config_manager.load_config()
    
    def _initialize_registered_workers(self):
        """Initialize all registered workers in the presence tracker (ONLY workers with embeddings)."""
        try:
            if self.db and self.presence_tracker:
                # CRITICAL FIX: Only load workers who have face embeddings
                # Workers without embeddings cannot be recognized, so don't track them
                embeddings_data = self.db.get_all_face_embeddings()
                
                if not embeddings_data:
                    logger.info("No workers with face embeddings found - none will be tracked")
                    return
                
                # Get unique workers from embeddings
                workers_with_embeddings = {}
                for emb in embeddings_data:
                    worker_id = emb['worker_id']
                    worker_name = emb['worker_name']
                    if worker_id not in workers_with_embeddings:
                        workers_with_embeddings[worker_id] = worker_name
                
                logger.info(f"Found {len(workers_with_embeddings)} workers with face embeddings")
                
                # Get worker status for absent time restoration
                all_worker_status = self.db.get_worker_status()
                worker_status_map = {w['worker_id']: w for w in all_worker_status}
                
                # Add only workers who have embeddings
                for worker_id, worker_name in workers_with_embeddings.items():
                    # Add worker to presence tracker if not already exists
                    if worker_id not in self.presence_tracker.workers:
                        self.presence_tracker.add_worker(worker_id, worker_name)
                        logger.info(f"Added worker with embeddings: {worker_name} (ID: {worker_id})")
                    
                    # Update absent time from database if available
                    if worker_id in worker_status_map:
                        worker_status = worker_status_map[worker_id]
                        if worker_status.get('absent_time', 0) > 0:
                            # DB stores minutes; convert to seconds for in-memory tracker
                            current_time = time.time()
                            worker_obj = self.presence_tracker.workers[worker_id]
                            worker_obj.total_absent_time = worker_status['absent_time'] * 60  # Convert minutes to seconds
                            # Set absent_start so absent time continues to increment
                            if worker_obj.absent_start is None:
                                # Calculate when the absence started based on current absent time
                                worker_obj.absent_start = current_time - worker_obj.total_absent_time
                            logger.info(f"Restored absent time for {worker_name}: {worker_status['absent_time']} minutes")
                
                logger.info(f"Initialized {len(workers_with_embeddings)} worker(s) with face embeddings in presence tracker")
            else:
                logger.warning("Cannot initialize registered workers: database or presence tracker not available")
                
        except Exception as e:
            logger.error(f"Failed to initialize registered workers: {str(e)}")
    
    def _sync_workers_from_db(self):
        """Sync workers from database to presence tracker (ONLY workers with embeddings)."""
        try:
            if self.db and self.presence_tracker:
                # CRITICAL FIX: Only sync workers who have face embeddings
                # Workers without embeddings cannot be recognized, so don't track them
                embeddings_data = self.db.get_all_face_embeddings()
                
                # Get unique workers from embeddings
                workers_with_embeddings = {}
                for emb in embeddings_data:
                    worker_id = emb['worker_id']
                    worker_name = emb['worker_name']
                    if worker_id not in workers_with_embeddings:
                        workers_with_embeddings[worker_id] = worker_name
                
                # Compute DB id set for reconciliation (only workers with embeddings)
                db_ids = set(workers_with_embeddings.keys())
                tracker_ids = set(self.presence_tracker.workers.keys())

                # Add any new workers with embeddings that aren't in the presence tracker
                for worker_id, worker_name in workers_with_embeddings.items():
                    if worker_id not in self.presence_tracker.workers:
                        self.presence_tracker.add_worker(worker_id, worker_name)
                        logger.info(f"Synced new worker with embeddings: {worker_name} (ID: {worker_id})")

                # Remove workers that no longer have embeddings or don't exist in DB
                to_remove = tracker_ids - db_ids
                if to_remove:
                    for wid in list(to_remove):
                        name = self.presence_tracker.workers.get(wid).name if wid in self.presence_tracker.workers else str(wid)
                        self.presence_tracker.remove_worker(wid)
                        logger.info(f"Removed worker (no embeddings or deleted): {name} (ID: {wid})")
                
        except Exception as e:
            logger.error(f"Failed to sync workers from database: {str(e)}")
    
    def _initialize_components(self):
        """Initialize all application components."""
        try:
            # Initialize database connection (optional for now)
            try:
                # Try multiple possible config paths
                config_paths = [
                    "../config/config.json",
                    "config/config.json",
                    str(Path(__file__).parent.parent / "config" / "config.json")
                ]
                db_connected = False
                for config_path in config_paths:
                    try:
                        if Path(config_path).exists():
                            self.db = DatabaseManager(config_path)
                            logger.info(f"Database connection established successfully using {config_path}")
                            db_connected = True
                            break
                    except Exception as e:
                        logger.debug(f"Failed to connect using {config_path}: {e}")
                        continue
                
                if not db_connected:
                    # Try default path
                    self.db = DatabaseManager()
                    logger.info("Database connection established using default path")
            except Exception as db_error:
                logger.warning(f"Database connection failed: {str(db_error)}")
                logger.warning("Application will run without database support.")
                self.db = None
            
            # Initialize camera manager
            camera_configs = self.config_manager.get_camera_configs()
            # Disable auto-detect for faster startup when configs are provided
            self.camera_manager = CameraManager(camera_configs, auto_detect=False)
            logger.info("Camera manager initialized.")
            
            # Initialize person detector with optimized settings for moving workers
            self.detector = PersonDetector(
                conf_threshold=0.45,  # Slightly lower to catch moving workers (was 0.5)
                iou_threshold=0.4,    # Lower NMS threshold for better detection of moving people
                device='cpu'
            )
            logger.info("Person detector initialized.")
            
            # Initialize face recognition system with ArcFace (ONLY supported model)
            # ArcFace: 512-D embeddings, 99.82% accuracy, best for distinguishing similar faces
            self.face_system = FaceRecognitionSystem(
                model_name="ArcFace",  # ONLY supported model - DO NOT CHANGE
                similarity_threshold=0.50,  # 50% threshold - prevents false positives
                distance_metric="cosine"
            )
            logger.info("Face recognition system initialized with ArcFace (512-D embeddings).")
            
            # Initialize presence tracker (pass DB manager so absent-time writes work)
            thresholds = self.config_manager.get_thresholds()
            self.presence_tracker = PresenceTracker(thresholds, db_manager=self.db)
            logger.info("Presence tracker initialized.")
            
            # Initialize alert manager
            notifications = self.config_manager.get_notification_config()
            self.alert_manager = AlertManager(notifications)
            logger.info("Alert manager initialized.")
            
            # Initialize UI
            self.ui = UIManager(self.config)
            self._connect_ui_signals()
            logger.info("UI manager initialized.")
            
            # Initialize all registered workers in presence tracker
            self._initialize_registered_workers()
            
            # CRITICAL: Force update presence tracker for all workers to ensure they're properly initialized
            # This ensures all workers have proper status even if they haven't been detected yet
            if self.presence_tracker:
                worker_count_before = len(self.presence_tracker.workers)
                logger.info(f"Presence tracker has {worker_count_before} workers before update")
                
                if self.presence_tracker.workers:
                    # Update all workers with empty detections to initialize their status
                    self.presence_tracker.update_detections([])
                    logger.info(f"Updated presence tracker with {len(self.presence_tracker.workers)} workers")
                else:
                    logger.warning("Presence tracker has no workers - checking database connection")
                    if self.db:
                        try:
                            # Try to get workers directly from database
                            db_workers = self.db.get_worker_status()
                            logger.info(f"Database has {len(db_workers)} workers")
                            if db_workers:
                                logger.warning("Workers exist in database but not in presence tracker - reloading...")
                                self.presence_tracker.load_all_workers()
                                if self.presence_tracker.workers:
                                    self.presence_tracker.update_detections([])
                        except Exception as e:
                            logger.error(f"Error checking database workers: {e}")
            
            # Update UI with all registered workers immediately (even if not detected yet)
            if self.ui and self.presence_tracker:
                worker_statuses = self.presence_tracker.get_all_statuses()
                logger.info(f"Retrieved {len(worker_statuses)} worker statuses from presence tracker")
                self.ui.update_worker_status(worker_statuses)
                logger.info(f"Initialized UI with {len(worker_statuses)} registered workers from database")
                if worker_statuses:
                    worker_names = [w.get('name', 'Unknown') for w in worker_statuses]
                    logger.info(f"Workers loaded: {', '.join(worker_names)}")
                else:
                    logger.warning("No workers found in presence tracker - UI will show empty worker status table")
                    # Try one more time to load workers
                    if self.db:
                        try:
                            db_workers = self.db.get_worker_status()
                            logger.warning(f"Database reports {len(db_workers)} workers but presence tracker has none")
                            if db_workers:
                                logger.warning("Attempting to reload workers into presence tracker...")
                                self.presence_tracker.load_all_workers()
                                worker_statuses = self.presence_tracker.get_all_statuses()
                                if worker_statuses:
                                    self.ui.update_worker_status(worker_statuses)
                                    logger.info(f"Reloaded and displayed {len(worker_statuses)} workers")
                        except Exception as e:
                            logger.error(f"Error in final worker reload attempt: {e}")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            self.show_error("Initialization Error", f"Failed to initialize components: {str(e)}")
            sys.exit(1)
    
    def _connect_ui_signals(self):
        """Connect UI signals to application functions."""
        if self.ui:
            self.ui.start_camera_signal.connect(self.start_monitoring)
            self.ui.stop_camera_signal.connect(self.stop_monitoring)
            self.ui.settings_changed_signal.connect(self.update_settings)
            self.ui.register_worker_signal.connect(self.open_worker_registration)
            
            # Connect worker management signals from worker status widget
            if hasattr(self.ui, 'worker_status_widget'):
                self.ui.worker_status_widget.add_worker_signal.connect(self.open_worker_registration)
                self.ui.worker_status_widget.delete_workers_signal.connect(self.delete_workers_from_ui)
    
    # (removed duplicate initialize_worker_list; use _initialize_registered_workers instead)

    def start_monitoring(self):
        """Start camera monitoring and detection."""
        try:
            # Check if already running
            if self.camera_manager and self.camera_manager.running:
                logger.warning("Monitoring is already running")
                if self.ui:
                    self.ui.status_bar.showMessage("Monitoring is already running")
                return
            
            if self.camera_manager:
                # Add cameras to UI
                if self.ui:
                    for camera_id in self.camera_manager.cameras:
                        self.ui.add_camera(camera_id)
                
                # Start cameras
                self.camera_manager.start()
                
                # Update UI status
                if self.ui:
                    for camera_id in self.camera_manager.cameras:
                        self.ui.update_camera_status(camera_id, "Running", "green")
                        
                    # Refresh worker status display
                    if self.presence_tracker:
                        worker_statuses = self.presence_tracker.get_all_statuses()
                        self.ui.worker_status_widget.update_workers(worker_statuses)
                    
                    # Update status bar
                    self.ui.status_bar.showMessage("✓ Monitoring started successfully")
                
                # Start detection thread (only if detector is available)
                if self.detector and self.camera_manager:
                    self.detection_thread = DetectionThread(
                        self.detector,
                        self.camera_manager,
                        self.face_system
                    )
                    self.detection_thread.detection_complete.connect(self.process_detections)
                    self.detection_thread.start()
                
                logger.info("Monitoring started successfully.")
            else:
                logger.error("Camera manager not initialized")
                if self.ui:
                    self.ui.status_bar.showMessage("✗ Error: Camera manager not available")
                
        except Exception as e:
            logger.error(f"Failed to start monitoring: {str(e)}")
            if self.ui:
                self.ui.status_bar.showMessage(f"✗ Error starting monitoring: {str(e)}")
            self.show_error("Start Error", f"Failed to start monitoring: {str(e)}")
    
    def stop_monitoring(self):
        """Stop camera monitoring and detection."""
        try:
            # Check if already stopped
            if self.camera_manager and not self.camera_manager.running:
                logger.warning("Monitoring is already stopped")
                if self.ui:
                    self.ui.status_bar.showMessage("Monitoring is already stopped")
                return
            
            # Stop detection thread
            if self.detection_thread and self.detection_thread.isRunning():
                logger.info("Stopping detection thread...")
                self.detection_thread.stop()
                self.detection_thread.wait(2000)  # Wait up to 2 seconds
                self.detection_thread = None
                logger.info("Detection thread stopped")
            
            # Stop cameras
            if self.camera_manager:
                logger.info("Stopping cameras...")
                self.camera_manager.stop()
                logger.info("Cameras stopped")
            
            # Update UI status
            if self.ui:
                if self.camera_manager:
                    for camera_id in self.camera_manager.cameras:
                        self.ui.update_camera_status(camera_id, "Stopped", "red")
                
                # Update status bar
                self.ui.status_bar.showMessage("✓ Monitoring stopped successfully")
            
            logger.info("Monitoring stopped successfully.")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
            if self.ui:
                self.ui.status_bar.showMessage(f"✗ Error stopping monitoring: {str(e)}")
    
    def process_detections(self, detections, camera_id):
        """Process detections from a camera."""
        try:
            # Update UI with frame (even if no detections)
            if self.ui and self.camera_manager:
                frame = self.camera_manager.get_frame(camera_id)
                if frame is not None:
                    # Draw detections on frame if any
                    if detections and self.detector:
                        frame = self.detector.draw_detections(frame, detections)
                    self.ui.update_camera_frame(camera_id, frame)
            
            # Convert detections to worker format for presence tracker
            worker_detections = []
            for det in detections:
                # Use face recognition results if available
                if det.worker_id is not None and det.worker_name is not None:
                    # Recognized worker - use their actual ID
                    worker_detections.append({
                        'worker_id': det.worker_id,
                        'confidence': det.confidence,
                        'name': det.worker_name,
                        'recognition_score': det.recognition_score
                    })
                    logger.debug(f"Tracking recognized worker: {det.worker_name} (ID: {det.worker_id})")
                # Note: Skip unrecognized persons to avoid creating fake worker IDs
            
            # Update presence tracker
            if self.presence_tracker:
                updates = self.presence_tracker.update_detections(worker_detections)
                
                # Update UI with worker status
                if self.ui:
                    worker_statuses = self.presence_tracker.get_all_statuses()
                    self.ui.update_worker_status(worker_statuses)
                
                # Check for alerts
                self.check_for_alerts(updates)
                
        except Exception as e:
            logger.error(f"Error processing detections: {str(e)}")
    
    def check_for_alerts(self, updates):
        """Check for and generate alerts based on worker status updates."""
        if not self.alert_manager:
            logger.debug("Alert manager not available, skipping alert checks")
            return
            
        try:
            for worker_id, status_data in updates.items():
                status = status_data['status']
                time_present = status_data['time_present']
                name = status_data['name']
                total_absent_time = status_data.get('total_absent_time', 0)
                
                # Debug logging for absent workers
                if status == "absent":
                    absent_minutes = total_absent_time / 60.0
                    logger.debug(f"Checking alerts for {name} (ID: {worker_id}): status={status}, absent_time={total_absent_time}s ({absent_minutes:.2f} min), already_alerted={worker_id in self.absence_alerted_workers}")
                
                # Check for exceeded time thresholds
                warning_threshold = self.config_manager.get("thresholds.warning_minutes", 15) * 60
                alert_threshold = self.config_manager.get("thresholds.alert_minutes", 30) * 60
                
                if status == "exceeded" and time_present > alert_threshold:
                    self.alert_manager.add_alert(
                        f"Worker Time Alert: {name}",
                        f"Worker {name} (ID: {worker_id}) has been present for {int(time_present/60)} minutes",
                        "alert",
                        worker_id=worker_id,
                        duration=30
                    )
                    if self.ui:
                        self.ui.add_alert(
                            f"Worker Time Alert: {name}",
                            f"Worker {name} (ID: {worker_id}) has been present for {int(time_present/60)} minutes",
                            "alert"
                        )
                elif status == "present" and time_present > warning_threshold:
                    self.alert_manager.add_alert(
                        f"Worker Time Warning: {name}",
                        f"Worker {name} (ID: {worker_id}) has been present for {int(time_present/60)} minutes",
                        "warning",
                        worker_id=worker_id,
                        duration=15
                    )
                    if self.ui:
                        self.ui.add_alert(
                            f"Worker Time Warning: {name}",
                            f"Worker {name} (ID: {worker_id}) has been present for {int(time_present/60)} minutes",
                            "warning"
                        )
                
                # Check for absent worker alerts
                if status == "absent":
                    # total_absent_time is in seconds
                    absent_minutes = total_absent_time / 60.0
                    absent_hours = total_absent_time / 3600.0
                    
                    # NEW: 1-minute absence alert (plays sound once per absence session)
                    if absent_minutes >= 1.0 and worker_id not in self.absence_alerted_workers:
                        self.alert_manager.add_alert(
                            f"Worker Absent: {name}",
                            f"Worker {name} (ID: {worker_id}) has been absent for {int(absent_minutes)} minute(s)",
                            "alert",
                            worker_id=worker_id,
                            duration=10
                        )
                        if self.ui:
                            self.ui.add_alert(
                                f"Worker Absent: {name}",
                                f"Worker {name} (ID: {worker_id}) has been absent for {int(absent_minutes)} minute(s)",
                                "alert"
                            )
                        self.absence_alerted_workers.add(worker_id)
                        logger.info(f"1-minute absence alert triggered for {name} (ID: {worker_id}) - absent for {absent_minutes:.1f} minutes ({total_absent_time} seconds)")
                    elif absent_hours >= 8:  # 8 hours absent
                        self.alert_manager.add_alert(
                            f"Worker Absent Alert: {name}",
                            f"Worker {name} (ID: {worker_id}) has been absent for {absent_hours:.1f} hours",
                            "alert",
                            worker_id=worker_id,
                            duration=60
                        )
                        if self.ui:
                            self.ui.add_alert(
                                f"Worker Absent Alert: {name}",
                                f"Worker {name} (ID: {worker_id}) has been absent for {absent_hours:.1f} hours",
                                "alert"
                            )
                    elif absent_hours >= 4:  # 4 hours absent
                        self.alert_manager.add_alert(
                            f"Worker Absent Warning: {name}",
                            f"Worker {name} (ID: {worker_id}) has been absent for {absent_hours:.1f} hours",
                            "warning",
                            worker_id=worker_id,
                            duration=30
                        )
                        if self.ui:
                            self.ui.add_alert(
                                f"Worker Absent Warning: {name}",
                                f"Worker {name} (ID: {worker_id}) has been absent for {absent_hours:.1f} hours",
                                "warning"
                            )
                    elif absent_hours >= 1:  # 1 hour absent
                        self.alert_manager.add_alert(
                            f"Worker Absent Notice: {name}",
                            f"Worker {name} (ID: {worker_id}) has been absent for {absent_hours:.1f} hours",
                            "info",
                            worker_id=worker_id,
                            duration=15
                        )
                        if self.ui:
                            self.ui.add_alert(
                                f"Worker Absent Notice: {name}",
                                f"Worker {name} (ID: {worker_id}) has been absent for {absent_hours:.1f} hours",
                                "info"
                            )
                else:
                    # Reset flag when worker is present/exceeded again
                    if worker_id in self.absence_alerted_workers:
                        self.absence_alerted_workers.remove(worker_id)
                        logger.debug(f"Reset 1-minute absence alert flag for {name} (ID: {worker_id}) - worker is now present")
                    
        except Exception as e:
            logger.error(f"Error checking for alerts: {str(e)}")
    
    def update_settings(self, new_config):
        """Update application settings."""
        try:
            # Update config manager
            self.config_manager.config = new_config
            self.config = new_config
            
            # Update presence tracker thresholds
            if self.presence_tracker:
                thresholds = self.config_manager.get_thresholds()
                self.presence_tracker.warning_threshold = thresholds.get("warning_minutes", 15) * 60
                self.presence_tracker.alert_threshold = thresholds.get("alert_minutes", 30) * 60
            
            # Update alert manager settings
            if self.alert_manager:
                notifications = self.config_manager.get_notification_config()
                self.alert_manager.notifications_enabled = notifications.get("enabled", True)
                self.alert_manager.sound_enabled = notifications.get("sound", True)
            
            logger.info("Settings updated successfully.")
            
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            self.show_error("Settings Error", f"Failed to update settings: {str(e)}")
    
    def open_worker_registration(self):
        """Open the worker registration dialog."""
        try:
            from src.worker_registration_ui import WorkerRegistrationDialog
            
            dialog = WorkerRegistrationDialog(
                self.camera_manager,
                self.face_system,
                self.detector,
                self.ui
            )
            result = dialog.exec_()
            
            # Refresh worker list after registration
            if result and self.presence_tracker:
                self._sync_workers_from_db()
                worker_statuses = self.presence_tracker.get_all_statuses()
                if self.ui:
                    self.ui.update_worker_status(worker_statuses)
            
        except Exception as e:
            logger.error(f"Error opening worker registration: {e}")
            self.show_error("Registration Error", f"Failed to open registration dialog: {str(e)}")
    
    def delete_workers_from_ui(self, workers_list: list):
        """Delete multiple workers from the UI with confirmation.
        
        Args:
            workers_list: List of tuples (worker_id, worker_name)
        """
        try:
            from PyQt5.QtWidgets import QMessageBox
            from pathlib import Path
            import shutil
            
            if not workers_list:
                return
            
            count = len(workers_list)
            
            # Build confirmation message
            if count == 1:
                worker_id, worker_name = workers_list[0]
                message = f"Are you sure you want to delete worker '{worker_name}' (ID: {worker_id})?"
            else:
                worker_names = [name for _, name in workers_list[:5]]  # Show first 5
                names_str = ", ".join(worker_names)
                if count > 5:
                    names_str += f", and {count - 5} more"
                message = f"Are you sure you want to delete {count} workers?\n\n{names_str}"
            
            # Confirm deletion
            reply = QMessageBox.question(
                self.ui,
                "Confirm Deletion",
                f"{message}\n\n"
                f"This will remove:\n"
                f"• Worker profile(s)\n"
                f"• Face embeddings\n"
                f"• Registration photos\n"
                f"• Activity history\n\n"
                f"This action cannot be undone!",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
            
            # Delete each worker
            deleted_count = 0
            failed_workers = []
            
            for worker_id, worker_name in workers_list:
                try:
                    # Delete from database
                    if self.db:
                        success = self.db.delete_worker(worker_id)
                        if not success:
                            logger.warning(f"Worker {worker_id} not found in database")
                    
                    # Delete from face recognition system
                    if self.face_system:
                        self.face_system.delete_worker(worker_id)
                    
                    # Delete face images directory
                    worker_dir = Path(f"data/registered_faces/worker_{worker_id}")
                    if worker_dir.exists():
                        shutil.rmtree(worker_dir)
                        logger.info(f"Deleted worker {worker_id} face images directory")
                    
                    # Remove from presence tracker
                    if self.presence_tracker and worker_id in self.presence_tracker.workers:
                        del self.presence_tracker.workers[worker_id]
                    
                    deleted_count += 1
                    logger.info(f"✅ Successfully deleted worker '{worker_name}' (ID: {worker_id})")
                    
                except Exception as e:
                    logger.error(f"❌ Error deleting worker '{worker_name}' (ID: {worker_id}): {e}")
                    failed_workers.append(worker_name)
            
            # Reload embeddings after all deletions
            if self.face_system:
                self.face_system.embeddings_db = self.face_system._load_embeddings_db()
                self.face_system._build_embeddings_matrix()
            
            # Refresh worker list
            self._sync_workers_from_db()
            if self.ui and self.presence_tracker:
                worker_statuses = self.presence_tracker.get_all_statuses()
                self.ui.update_worker_status(worker_statuses)
            
            # Show success/error message
            if self.ui:
                if deleted_count == count:
                    # All successful
                    if count == 1:
                        msg = f"✅ Worker deleted successfully"
                    else:
                        msg = f"✅ {deleted_count} workers deleted successfully"
                    self.ui.status_bar.showMessage(msg, 5000)
                    self.ui.add_alert("Workers Deleted", msg, "info")
                else:
                    # Some failed
                    msg = f"⚠️ Deleted {deleted_count}/{count} workers"
                    if failed_workers:
                        msg += f". Failed: {', '.join(failed_workers[:3])}"
                    self.ui.status_bar.showMessage(msg, 5000)
                    self.ui.add_alert("Partial Deletion", msg, "warning")
            
        except Exception as e:
            logger.error(f"❌ Error deleting workers: {e}")
            self.show_error("Delete Error", f"Failed to delete workers: {str(e)}")
    
    def update(self):
        """Main update loop."""
        # This will be called every second
        current_time = time.time()
        
        # Periodically sync workers from database to pick up newly registered workers
        if current_time - self.last_worker_sync >= self.worker_sync_interval:
            self._sync_workers_from_db()
            self.last_worker_sync = current_time
        
        # Update worker status continuously to ensure all registered workers are displayed
        if self.ui and self.presence_tracker:
            # Update presence tracker for all workers (even without detections)
            # This ensures absent time continues to track for workers not currently detected
            if self.presence_tracker.workers:
                # Update all workers even with empty detections list
                updates = self.presence_tracker.update_detections([])
                
                # Check for alerts (including 1-minute absence alerts)
                if updates and self.alert_manager:
                    self.check_for_alerts(updates)
            
            # Refresh UI with all worker statuses
            worker_statuses = self.presence_tracker.get_all_statuses()
            self.ui.update_worker_status(worker_statuses)
    
    def show_error(self, title: str, message: str):
        """Show an error message dialog."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
    
    def run(self) -> int:
        """Run the application."""
        try:
            logger.info("Starting Floor Monitoring Application")
            if self.ui:
                self.ui.show()
                # Ensure all workers are displayed immediately when window is shown
                if self.presence_tracker:
                    # Force reload workers if none exist
                    if len(self.presence_tracker.workers) == 0 and self.db:
                        logger.warning("No workers in presence tracker on window show - attempting reload")
                        self.presence_tracker.load_all_workers()
                        if self.presence_tracker.workers:
                            self.presence_tracker.update_detections([])
                    
                    worker_statuses = self.presence_tracker.get_all_statuses()
                    self.ui.update_worker_status(worker_statuses)
                    logger.info(f"Refreshed UI with {len(worker_statuses)} workers on window show")
                    if len(worker_statuses) == 0:
                        logger.warning("Worker status table is empty - check database for registered workers")
            return self.app.exec_()
        except Exception as e:
            logger.critical(f"Application error: {str(e)}", exc_info=True)
            self.show_error("Application Error", f"An unexpected error occurred: {str(e)}")
            return 1
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Close database connection
        if self.db:
            self.db.close()
        
        # Stop alert manager
        if self.alert_manager:
            self.alert_manager.stop()
        
        # Clean up UI
        if self.ui:
            self.ui.close()


def main():
    """Entry point for the application."""
    app = FloorMonitoringApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()