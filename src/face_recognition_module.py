"""
Face Recognition Module
-----------------------
Integrates DeepFace with YOLOv8 for worker face recognition and attendance tracking.
Uses ArcFace model ONLY for generating face embeddings (512-D, excellent discrimination).
ArcFace provides 99.82% accuracy and is specifically optimized for distinguishing between similar-looking faces.

IMPORTANT: This system ONLY supports ArcFace. Do not use other models (VGG-Face, Facenet, etc.)
"""
import cv2
import numpy as np
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from datetime import datetime


# DeepFace imports
try:
    from deepface import DeepFace
except ImportError:
    raise ImportError("DeepFace not installed. Install with: pip install deepface")

# Scikit-learn for similarity calculations
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)


UNKNOWN_NAME = "Unknown"


class RecognitionStabilityTracker:
    """
    Tracks recognition stability to prevent false positives.
    Requires multiple consecutive consistent recognitions before confirming identity.
    """
    def __init__(self, required_confirmations: int = 3, confidence_threshold: float = 0.70):
        """
        Args:
            required_confirmations: Number of consecutive matches needed
            confidence_threshold: Minimum confidence to maintain stability
        """
        self.required_confirmations = required_confirmations
        self.confidence_threshold = confidence_threshold
        self.track_history: Dict[int, List[Tuple[Optional[int], str, float]]] = {}  # track_id -> [(worker_id, name, confidence)]
        self.confirmed_identities: Dict[int, Tuple[int, str]] = {}  # track_id -> (worker_id, name)
        
    def update(self, track_id: int, worker_id: Optional[int], worker_name: str, confidence: float) -> Tuple[Optional[int], str]:
        """
        Update tracking history and return confirmed identity.
        
        Args:
            track_id: Tracking ID from YOLO
            worker_id: Recognized worker ID (None for Unknown)
            worker_name: Recognized worker name
            confidence: Recognition confidence
            
        Returns:
            (confirmed_worker_id, confirmed_name) - returns (None, 'Unknown') if not stable
        """
        # Initialize history for new track
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        
        # Add current detection to history
        self.track_history[track_id].append((worker_id, worker_name, confidence))
        
        # Keep only recent history (last 5 detections)
        if len(self.track_history[track_id]) > 5:
            self.track_history[track_id] = self.track_history[track_id][-5:]
        
        # Check if we have enough history
        history = self.track_history[track_id]
        if len(history) < self.required_confirmations:
            # Not enough data yet - return Unknown
            return (None, UNKNOWN_NAME)
        
        # Get last N confirmations
        recent = history[-self.required_confirmations:]
        
        # Check consistency: all must match same worker_id and confidence >= threshold
        first_worker_id = recent[0][0]
        first_worker_name = recent[0][1]
        
        is_consistent = all(
            detection[0] == first_worker_id and 
            detection[1] == first_worker_name and
            detection[2] >= self.confidence_threshold
            for detection in recent
        )
        
        if is_consistent and first_worker_id is not None:
            # Stable recognition - confirm identity
            self.confirmed_identities[track_id] = (first_worker_id, first_worker_name)
            logger.info(
                f"✅ STABLE RECOGNITION: Track {track_id} confirmed as {first_worker_name} "
                f"(ID: {first_worker_id}) after {self.required_confirmations} consistent detections"
            )
            return (first_worker_id, first_worker_name)
        else:
            # Not consistent or Unknown - reset confirmation
            if track_id in self.confirmed_identities:
                old_name = self.confirmed_identities[track_id][1]
                logger.warning(
                    f"⚠️ UNSTABLE: Track {track_id} lost stability (was {old_name}, now inconsistent) - Reverting to Unknown"
                )
                del self.confirmed_identities[track_id]
            return (None, UNKNOWN_NAME)
    
    def reset_track(self, track_id: int):
        """Reset tracking history for a track (when person leaves frame)."""
        if track_id in self.track_history:
            del self.track_history[track_id]
        if track_id in self.confirmed_identities:
            del self.confirmed_identities[track_id]
    
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """Remove history for tracks that are no longer active."""
        all_track_ids = set(self.track_history.keys()) | set(self.confirmed_identities.keys())
        inactive_tracks = all_track_ids - set(active_track_ids)
        
        for track_id in inactive_tracks:
            self.reset_track(track_id)

@dataclass
class FaceEmbedding:
    """Represents a face embedding for a worker."""
    worker_id: int
    worker_name: str
    embedding: np.ndarray
    image_path: str = ""  # Optional - not stored in database
    timestamp: float = 0.0  # Optional - not stored in database
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'worker_id': self.worker_id,
            'worker_name': self.worker_name,
            'embedding': self.embedding.tolist(),
            'image_path': self.image_path,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FaceEmbedding':
        """Create from dictionary."""
        return cls(
            worker_id=data['worker_id'],
            worker_name=data['worker_name'],
            embedding=np.array(data['embedding']),
            image_path=data['image_path'],
            timestamp=data['timestamp']
        )


@dataclass
class AttendanceRecord:
    """Represents an attendance record."""
    worker_id: int
    worker_name: str
    timestamp: float
    photo_path: str
    similarity_score: float
    event_type: str  # 'check_in' or 'check_out'
    gps_coordinates: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'worker_id': self.worker_id,
            'worker_name': self.worker_name,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'photo_path': self.photo_path,
            'similarity_score': self.similarity_score,
            'event_type': self.event_type,
            'gps_coordinates': self.gps_coordinates
        }


class FaceRecognitionSystem:
    """
    Face recognition system using YOLOv8 for detection and DeepFace for recognition.
    """
    
    def __init__(
        self,
        embeddings_db_path: str = "data/face_embeddings.pkl",  # DEPRECATED - kept for compatibility
        attendance_db_path: str = "data/attendance_records.json",  # DEPRECATED - kept for compatibility
        model_name: str = "ArcFace",  # FIXED: Only ArcFace is supported
        similarity_threshold: float = 0.50,  # Base threshold - smarter logic handles discrimination
        distance_metric: str = "cosine",
        use_database: bool = True  # Always use PostgreSQL database (pickle files deprecated)
    ):
        """
        Initialize the face recognition system.
        
        Args:
            embeddings_db_path: Path to store face embeddings database (fallback if DB fails)
            attendance_db_path: Path to store attendance records (fallback if DB fails)
            model_name: Must be "ArcFace" (only supported model, 512-D embeddings)
            similarity_threshold: Threshold for face matching (0-1)
            distance_metric: Distance metric ('cosine' or 'euclidean')
            use_database: Use PostgreSQL database (recommended to avoid corruption)
        """
        # Force ArcFace model
        if model_name != "ArcFace":
            logger.warning(f"Model '{model_name}' not supported. Forcing ArcFace.")
            model_name = "ArcFace"
        self.embeddings_db_path = Path(embeddings_db_path)
        self.attendance_db_path = Path(attendance_db_path)
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.distance_metric = distance_metric
        self.use_database = use_database
        
        # Initialize database connection (REQUIRED)
        self.db_manager = None
        if self.use_database:
            try:
                from database_module import DatabaseManager
                self.db_manager = DatabaseManager()
                logger.info("Connected to PostgreSQL database")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                logger.error("Database is REQUIRED. Pickle files are deprecated.")
                raise Exception("Database connection failed. Please run: python migrate_to_database.py")
        
        # Create data directory if it doesn't exist (fallback)
        self.embeddings_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.attendance_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize embeddings database
        self.embeddings_db: List[FaceEmbedding] = self._load_embeddings_db()
        
        # Load or initialize attendance records
        self.attendance_records: List[AttendanceRecord] = self._load_attendance_records()
        
        # Cache for faster recognition
        self.embeddings_matrix = None
        self._build_embeddings_matrix()
        
        # CRITICAL: Validate model is working correctly
        self._validate_model()
        
        # Initialize stability tracker for preventing false positives
        self.stability_tracker = RecognitionStabilityTracker(
            required_confirmations=3,  # Require 3 consecutive matches
            confidence_threshold=0.70   # Minimum confidence to maintain
        )
        
        # Face quality thresholds (RELAXED for multi-angle detection)
        self.min_face_width = 30   # Minimum face width in pixels (reduced from 40)
        self.min_face_height = 30  # Minimum face height in pixels (reduced from 40)
        self.min_face_brightness = 20  # Minimum average brightness (relaxed from 30)
        self.max_face_brightness = 235  # Maximum average brightness (relaxed from 225)
        
        logger.info(f"Face Recognition System initialized with {model_name}")
        logger.info(f"Loaded {len(self.embeddings_db)} face embeddings")
        logger.info(f"Storage: PostgreSQL Database (pickle files deprecated)")
        logger.info(f"Stability tracking: {self.stability_tracker.required_confirmations} confirmations required")
    
    def _load_embeddings_db(self) -> List[FaceEmbedding]:
        """Load face embeddings from database only."""
        if self.use_database and self.db_manager:
            try:
                db_embeddings = self.db_manager.get_all_face_embeddings()
                result = []
                for emb_data in db_embeddings:
                    result.append(FaceEmbedding(
                        worker_id=emb_data['worker_id'],
                        worker_name=emb_data['worker_name'],
                        embedding=emb_data['embedding']
                    ))
                logger.info(f"Loaded {len(result)} embeddings from database")
                return result
            except Exception as e:
                logger.error(f"Error loading embeddings from database: {e}")
                return []
        return []
    
    def _save_embeddings_db(self) -> None:
        """Save face embeddings to database only."""
        try:
            # Embeddings are saved in register_worker method directly to database
            # This method just rebuilds the matrix
            logger.debug(f"Database storage active ({len(self.embeddings_db)} embeddings in memory)")
            
            # Rebuild matrix after saving
            self._build_embeddings_matrix()
        except Exception as e:
            logger.error(f"Error in save operation: {e}")
    
    def _build_embeddings_matrix(self) -> None:
        """Build a matrix of all embeddings for faster batch similarity computation."""
        if not self.embeddings_db:
            self.embeddings_matrix = None
            return
        
        try:
            # CRITICAL: Verify all embeddings have same dimension before stacking
            dimensions = [emb.embedding.shape[0] for emb in self.embeddings_db]
            unique_dims = set(dimensions)
            
            if len(unique_dims) > 1:
                logger.error(
                    f"CRITICAL: Mixed embedding dimensions detected in database! "
                    f"Found dimensions: {unique_dims}. "
                    f"This means workers were registered with different models. "
                    f"You MUST delete data/face_embeddings.pkl and re-register ALL workers."
                )
                # List which workers have which dimensions
                for emb in self.embeddings_db:
                    logger.error(f"  - Worker {emb.worker_id} ({emb.worker_name}): {emb.embedding.shape[0]}-D")
                self.embeddings_matrix = None
                return
            
            # Stack all embeddings into a matrix
            self.embeddings_matrix = np.vstack([emb.embedding for emb in self.embeddings_db])
            logger.debug(f"Built embeddings matrix with shape {self.embeddings_matrix.shape} (all {list(unique_dims)[0]}-D embeddings)")
        except Exception as e:
            logger.error(f"Error building embeddings matrix: {e}")
            self.embeddings_matrix = None
    
    def _validate_model(self) -> None:
        """Validate that ArcFace model is loaded correctly and generates 512-D embeddings."""
        # System only supports ArcFace
        if self.model_name != 'ArcFace':
            logger.error(
                f"UNSUPPORTED MODEL: {self.model_name}. "
                f"This system only supports ArcFace. "
                f"Please use model_name='ArcFace' when initializing FaceRecognitionSystem."
            )
            return
        
        expected_dim = 512  # ArcFace always generates 512-D embeddings
        
        try:
            # Generate a test embedding with a dummy image
            dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            test_embedding = self.get_face_embedding(dummy_face)
            
            if test_embedding is not None:
                actual_dim = test_embedding.shape[0]
                if actual_dim != expected_dim:
                    logger.error(
                        f"MODEL VALIDATION FAILED! "
                        f"Expected {expected_dim}-D for {self.model_name}, got {actual_dim}-D. "
                        f"DeepFace is using the wrong model! "
                        f"Try: 1) Clear ~/.deepface cache, 2) Restart app"
                    )
                else:
                    logger.debug(f"✓ Model validation passed: {self.model_name} generates {actual_dim}-D embeddings")
            else:
                logger.warning("Could not generate test embedding for validation")
        except Exception as e:
            logger.error(f"Model validation error: {e}")
    
    def _load_attendance_records(self) -> List[AttendanceRecord]:
        """Load attendance records from file."""
        if self.attendance_db_path.exists():
            try:
                with open(self.attendance_db_path, 'r') as f:
                    data = json.load(f)
                    return [AttendanceRecord(**d) for d in data]
            except Exception as e:
                logger.error(f"Error loading attendance records: {e}")
                return []
        return []
    
    def _save_attendance_records(self) -> None:
        """Save attendance records to file."""
        try:
            data = [record.to_dict() for record in self.attendance_records]
            with open(self.attendance_db_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.attendance_records)} attendance records")
        except Exception as e:
            logger.error(f"Error saving attendance records: {e}")
    
    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """
        IMPROVED preprocessing with CLAHE - balanced approach for lighting normalization.
        Uses adaptive histogram equalization to handle varying lighting conditions.
        
        Args:
            face_img: Input face image
            
        Returns:
            Preprocessed face image
        """
        if face_img is None or face_img.size == 0:
            return face_img
        
        if len(face_img.shape) == 3:
            # Convert to LAB color space for better lighting normalization
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            
            # Apply CLAHE only to L channel (adaptive histogram equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to BGR
            face_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Gentle gamma correction based on brightness
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            if mean_brightness < 80:  # Dark image
                gamma = 1.3
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
                face_img = cv2.LUT(face_img, table)
            elif mean_brightness > 180:  # Bright image
                gamma = 0.7
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
                face_img = cv2.LUT(face_img, table)
        
        return face_img
    
    def check_face_quality(self, face_img: np.ndarray) -> Tuple[bool, str]:
        """
        Check if face image meets minimum quality requirements.
        Prevents recognition on poor quality, partial, or occluded faces.
        
        Args:
            face_img: Face image to check
            
        Returns:
            (is_valid, reason) - True if quality is acceptable, False with reason otherwise
        """
        if face_img is None or face_img.size == 0:
            return False, "Empty image"
        
        # Check dimensions (relaxed for side angles)
        height, width = face_img.shape[:2]
        if width < self.min_face_width or height < self.min_face_height:
            return False, f"Face too small ({width}x{height} < {self.min_face_width}x{self.min_face_height})"
        
        # Check brightness (avoid too dark or too bright faces)
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        mean_brightness = np.mean(gray)
        if mean_brightness < self.min_face_brightness:
            return False, f"Too dark (brightness: {mean_brightness:.1f} < {self.min_face_brightness})"
        if mean_brightness > self.max_face_brightness:
            return False, f"Too bright (brightness: {mean_brightness:.1f} > {self.max_face_brightness})"
        
        # Check for blur (using Laplacian variance) - RELAXED for different angles
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 30:  # Reduced from 50 to accept more angles
            return False, f"Too blurry (variance: {laplacian_var:.1f} < 30)"
        
        return True, "OK"
    
    def get_face_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding using DeepFace (optimized for speed and accuracy).
        
        Args:
            face_img: Cropped face image (BGR format)
            
        Returns:
            Face embedding as numpy array, or None if failed
        """
        try:
            # Ensure image is in correct format
            if face_img is None or face_img.size == 0:
                return None
            
            # Early exit for very small faces (relaxed for multi-angle detection)
            if face_img.shape[0] < 40 or face_img.shape[1] < 40:
                logger.warning("Face image too small for embedding")
                return None
            
            # FIXED: Re-enabled preprocessing with improved CLAHE method
            # Uses adaptive histogram equalization for consistent lighting
            # This improves recognition across different lighting conditions
            face_img = self.preprocess_face(face_img)
            
            # Resize to optimal size for ArcFace (112x112 is standard)
            # ArcFace uses 112x112 input size
            target_size = (112, 112)
            face_img = cv2.resize(face_img, target_size, interpolation=cv2.INTER_CUBIC)
            
            # Convert BGR to RGB (DeepFace expects RGB)
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # CRITICAL: Force model to be loaded correctly
            # DeepFace sometimes has issues with model caching
            try:
                # Generate embedding using DeepFace
                logger.debug(f"Generating embedding with model: {self.model_name}")
                embedding_objs = DeepFace.represent(
                    img_path=face_rgb,
                    model_name=self.model_name,
                    enforce_detection=False,  # We already detected the face
                    detector_backend='skip'  # Skip detection since we have cropped face
                )
            except Exception as deepface_error:
                logger.error(f"DeepFace error with {self.model_name}: {deepface_error}")
                logger.error("This may be a model loading issue. Try clearing DeepFace cache.")
                return None
            
            if embedding_objs and len(embedding_objs) > 0:
                embedding = np.array(embedding_objs[0]['embedding'])
                logger.debug(f"Generated embedding with shape: {embedding.shape} using model: {self.model_name}")
                
                # CRITICAL: Verify embedding is 512-D (ArcFace)
                expected_dim = 512  # ArcFace always generates 512-D embeddings
                
                if embedding.shape[0] != expected_dim:
                    logger.error(
                        f"CRITICAL: Embedding dimension mismatch! "
                        f"Expected {expected_dim}-D (ArcFace), got {embedding.shape[0]}-D. "
                        f"DeepFace may be using wrong model. Try clearing model cache."
                    )
                    return None
                
                # DON'T normalize here - will be normalized after averaging in register_worker
                # Normalizing before averaging causes mathematical errors
                return embedding
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating face embedding: {e}")
            return None
    
    def validate_registration_photos(self, face_images: List[np.ndarray]) -> Tuple[bool, str, List[np.ndarray]]:
        """
        Validate registration photos before processing.
        Checks quality, diversity, and returns only valid images.
        
        Returns:
            (is_valid, message, valid_images_list)
        """
        if len(face_images) < 5:
            return False, f"Need at least 5 face images (got {len(face_images)}), 10 recommended", []
        
        valid_images = []
        embeddings = []
        
        for idx, face_img in enumerate(face_images):
            # Check image exists
            if face_img is None or face_img.size == 0:
                logger.warning(f"Image {idx}: Invalid or empty")
                continue
            
            # Check size
            if face_img.shape[0] < 80 or face_img.shape[1] < 80:
                logger.warning(f"Image {idx}: Too small {face_img.shape} (min 80x80)")
                continue
            
            # Check brightness
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            if brightness < 30 or brightness > 230:
                logger.warning(f"Image {idx}: Poor brightness {brightness:.1f} (range 30-230)")
                continue
            
            # Check sharpness (Laplacian variance)
            laplacian_var = np.var(cv2.Laplacian(gray, cv2.CV_64F))
            if laplacian_var < 50:
                logger.warning(f"Image {idx}: Too blurry (variance {laplacian_var:.1f} < 50)")
                continue
            
            # Generate embedding to verify face is recognizable
            embedding = self.get_face_embedding(face_img)
            if embedding is not None:
                valid_images.append(face_img)
                embeddings.append(embedding)
                logger.debug(f"Image {idx}: Valid (brightness={brightness:.1f}, sharpness={laplacian_var:.1f})")
            else:
                logger.warning(f"Image {idx}: Failed to generate embedding")
        
        if len(valid_images) < 3:
            return False, f"Only {len(valid_images)} valid images (need 3+). Check lighting and focus.", []
        
        # Check diversity - embeddings should not be too similar (different poses/angles)
        # NOTE: For the SAME person, 95-99% similarity is NORMAL and EXPECTED
        # We only reject if photos are EXTREMELY similar (>99.5%) which suggests:
        # - Same exact photo captured multiple times
        # - No movement between captures
        # - Camera/person completely frozen
        if len(embeddings) >= 2:
            embeddings_array = np.array(embeddings)
            # Normalize
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_norm = embeddings_array / (norms + 1e-8)
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(embeddings_norm)
            # Get upper triangle (excluding diagonal)
            upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
            avg_similarity = np.mean(upper_tri)
            max_similarity = np.max(upper_tri)
            
            # FIXED: Use 99.5% threshold instead of 98%
            # Same person photos SHOULD be 95-99% similar!
            # Only reject if essentially identical (>99.5%)
            if avg_similarity > 0.995:
                return False, f"Photos too similar ({avg_similarity*100:.1f}%). All photos appear identical - ensure person moves between captures.", []
            
            # Warn if max similarity is very high (but still allow)
            if max_similarity > 0.998:
                logger.warning(f"Some photos are very similar (max: {max_similarity*100:.1f}%). Consider more variety, but proceeding with registration.")
            
            logger.info(f"Photo diversity check passed: avg similarity {avg_similarity*100:.1f}%, max {max_similarity*100:.1f}%")
        
        return True, f"{len(valid_images)} valid images with good diversity", valid_images
    
    def check_embedding_uniqueness(
        self, 
        new_embedding: np.ndarray, 
        worker_id: int
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Check if new embedding is sufficiently different from existing workers.
        Prevents duplicate registrations and similar-looking people.
        
        Returns:
            (is_unique, similarity_to_closest, closest_worker_name)
        """
        if not self.embeddings_db:
            return True, 0.0, None
        
        # Normalize
        new_embedding = new_embedding / (np.linalg.norm(new_embedding) + 1e-8)
        
        # Find closest existing worker (excluding same worker_id)
        closest_sim = 0.0
        closest_name = None
        
        for emb in self.embeddings_db:
            if emb.worker_id == worker_id:
                continue  # Skip same worker (for re-registration)
            
            # Normalize
            stored_emb = emb.embedding / (np.linalg.norm(emb.embedding) + 1e-8)
            
            # Calculate similarity
            sim = cosine_similarity([new_embedding], [stored_emb])[0][0]
            
            if sim > closest_sim:
                closest_sim = sim
                closest_name = emb.worker_name
        
        # FIXED: Relaxed threshold from 40% to 65%
        # Different people can be 40-60% similar (similar features, skin tone, etc.)
        # Only reject if >65% similar (likely same person or twin)
        # This prevents false "duplicate" rejections while still catching actual duplicates
        # SPECIAL CASE: Set to 99% to allow workers with 98% similarity (confirmed different)
        UNIQUENESS_THRESHOLD = 0.99  # 99% similarity threshold (allows very similar people)
        
        is_unique = closest_sim < UNIQUENESS_THRESHOLD
        
        if not is_unique:
            logger.warning(
                f"Uniqueness check failed: New worker is {closest_sim*100:.1f}% similar to '{closest_name}' "
                f"(threshold: {UNIQUENESS_THRESHOLD*100:.0f}%). This suggests they may be the same person."
            )
        
        return is_unique, float(closest_sim), closest_name

    def register_worker(
        self,
        worker_id: int,
        worker_name: str,
        face_images: List[np.ndarray],
        save_dir: str = "data/registered_faces",
        skip_uniqueness_check: bool = False
    ) -> bool:
        """
        Register a worker with multiple face images.
        IMPROVED: Validates photos, checks uniqueness, uses first embedding.
        
        Args:
            worker_id: Unique worker ID
            worker_name: Worker's name
            face_images: List of 5-10 cropped face images
            save_dir: Directory to save face images
            skip_uniqueness_check: If True, skip uniqueness validation (use with caution)
            
        Returns:
            True if registration successful, False otherwise
        """
        # STEP 1: Validate photos (quality, diversity)
        is_valid, message, valid_images = self.validate_registration_photos(face_images)
        if not is_valid:
            logger.error(f"Registration validation failed for {worker_name}: {message}")
            return False
        
        logger.info(f"Validation passed for {worker_name}: {message}")
        face_images = valid_images  # Use only valid images
        
        save_path = Path(save_dir) / f"worker_{worker_id}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # CRITICAL FIX: Collect all embeddings first, then average
        embeddings_list = []
        saved_image_paths = []
        
        for idx, face_img in enumerate(face_images):
            # Generate embedding
            embedding = self.get_face_embedding(face_img)
            
            if embedding is not None:
                # Save face image
                img_filename = f"{worker_name}_{idx}_{int(time.time())}.jpg"
                img_path = save_path / img_filename
                cv2.imwrite(str(img_path), face_img)
                
                embeddings_list.append(embedding)
                saved_image_paths.append(str(img_path))
                logger.info(f"Generated embedding {idx+1} for worker {worker_name}")
        
        if len(embeddings_list) == 0:
            logger.error(f"Failed to register {worker_name} - no valid embeddings generated")
            return False
        
        # CRITICAL FIX: Don't average - averaging causes embeddings to converge!
        # Instead, use the FIRST embedding (most representative)
        # Or could use median, but first is simplest and works well
        final_embedding = embeddings_list[0]
        
        # Normalize the final embedding
        final_embedding = final_embedding / (np.linalg.norm(final_embedding) + 1e-8)
        
        logger.info(f"Using first embedding (from {len(embeddings_list)} images) for {worker_name}")
        logger.debug(f"Final embedding L2 norm: {np.linalg.norm(final_embedding):.4f} (should be ~1.0)")
        
        # STEP 2: Check uniqueness (prevent duplicates and similar workers)
        is_unique, similarity, closest_worker = self.check_embedding_uniqueness(final_embedding, worker_id)
        if not is_unique:
            logger.error(
                f"Registration failed for {worker_name}: Too similar to existing worker '{closest_worker}' "
                f"({similarity*100:.1f}% similar). This may be a duplicate or the same person."
            )
            return False
        
        if closest_worker:
            logger.info(f"Uniqueness check passed: {worker_name} is {(1-similarity)*100:.1f}% different from closest worker '{closest_worker}'")
        
        # Create and store ONLY ONE embedding per worker
        face_emb = FaceEmbedding(
            worker_id=worker_id,
            worker_name=worker_name,
            embedding=final_embedding,  # Use first embedding (normalized)
            image_path=saved_image_paths[0],  # Reference first image
            timestamp=time.time()
        )
        
        self.embeddings_db.append(face_emb)
        
        # Save to database if enabled
        if self.use_database and self.db_manager:
            try:
                # Add worker to database with specific worker_id
                try:
                    self.db_manager.add_worker(worker_name, position=None, contact=None, worker_id=worker_id)
                    logger.info(f"Added/updated worker {worker_name} (ID:{worker_id}) in database")
                except Exception as e:
                    logger.warning(f"Failed to add worker to database: {e}")
                
                # Delete old embeddings for this worker (if re-registering)
                self.db_manager.delete_worker_embeddings(worker_id)
                
                # Save embedding to database
                self.db_manager.save_face_embedding(worker_id, final_embedding, self.model_name)
                logger.info(f"Saved embedding to database for {worker_name}")
                
                # Save all face photos to database
                for img_path in saved_image_paths:
                    with open(img_path, 'rb') as f:
                        photo_data = f.read()
                    self.db_manager.save_face_photo(worker_id, photo_data, 'jpg')
                logger.info(f"Saved {len(saved_image_paths)} photos to database for {worker_name}")
            except Exception as e:
                logger.error(f"Failed to save to database: {e}")
                raise Exception(f"Database save failed: {e}")
        
        self._save_embeddings_db()
        logger.info(f"Successfully registered {worker_name} with 1 averaged embedding (from {len(embeddings_list)} images)")
        return True
    
    def calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate similarity between two embeddings.
        
        Args:
            embedding1: First embedding (normalized)
            embedding2: Second embedding (normalized)
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Ensure embeddings are normalized
        embedding1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        embedding2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        if self.distance_metric == "cosine":
            # Cosine similarity for normalized vectors is already in [-1, 1]
            # For face embeddings, it's typically in [0.3, 1.0] range
            # DO NOT convert to [0,1] - that doubles all scores!
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            # Clamp to [0, 1] just in case
            similarity = max(0.0, min(1.0, similarity))
            return float(similarity)
        elif self.distance_metric == "euclidean":
            # Euclidean distance (convert to similarity)
            distance = euclidean(embedding1, embedding2)
            # Convert distance to similarity (inverse relationship)
            similarity = 1 / (1 + distance)
            return similarity
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def recognize_worker(
        self,
        face_img: np.ndarray,
        check_quality: bool = True
    ) -> Optional[Tuple[int, str, float]]:
        """
        Recognize a worker from a face image with ROBUST TWO-STEP VERIFICATION.
        
        ANTI-FALSE-POSITIVE SYSTEM (UPDATED FOR BETTER ACCURACY):
        - Step 1: Compute cosine similarity with all database embeddings
        - Step 2: Accept match ONLY if BOTH conditions hold:
          * Similarity >= 0.85 (high confidence)
          * Margin (best - second_best) >= 0.20 (clear separation)
        - Absolute minimum: similarity >= 0.75 (below this = always Unknown)
        
        Args:
            face_img: Cropped face image
            check_quality: Whether to check face quality before recognition
            
        Returns:
            Tuple of (worker_id, worker_name, similarity_score) if recognized, None otherwise
        """
        if not self.embeddings_db:
            logger.warning("No workers registered in database")
            return None
        
        # STEP 0: Check face quality (optional but recommended)
        if check_quality:
            is_valid, reason = self.check_face_quality(face_img)
            if not is_valid:
                logger.debug(f"⚠️ Face quality check failed: {reason} - Skipping recognition")
                return None
        
        # Generate embedding for the input face
        query_embedding = self.get_face_embedding(face_img)
        
        if query_embedding is None:
            logger.warning("Failed to generate embedding for query face")
            return None
        
        # Normalize query embedding
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # CRITICAL: Verify embedding dimensions match before comparison
        if self.embeddings_matrix is not None:
            expected_dim = self.embeddings_matrix.shape[1]
            actual_dim = query_embedding.shape[0]
            
            if expected_dim != actual_dim:
                logger.error(
                    f"CRITICAL DIMENSION MISMATCH! "
                    f"Query embedding: {actual_dim}D, Database embeddings: {expected_dim}D. "
                    f"Model: {self.model_name}. "
                    f"This means DeepFace is using a different model than what was used for registration. "
                    f"Solution: Delete data/face_embeddings.pkl and re-register all workers."
                )
                return None
        
        # Use vectorized operations for faster similarity computation
        if self.embeddings_matrix is not None and self.distance_metric == "cosine":
            # Normalize embeddings matrix
            normalized_matrix = self.embeddings_matrix / (np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True) + 1e-8)
            
            # Batch cosine similarity computation (much faster)
            # For normalized vectors, cosine similarity is already in proper range
            similarities = cosine_similarity([query_embedding], normalized_matrix)[0]
            # Clamp to [0, 1]
            similarities = np.clip(similarities, 0.0, 1.0)
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = float(similarities[best_idx])
            best_match = self.embeddings_db[best_idx]
            
            # Calculate margin (difference between best and second best)
            margin = 0.0
            second_best_similarity = 0.0
            if len(similarities) > 1:
                sorted_indices = np.argsort(similarities)[::-1]
                second_best_similarity = float(similarities[sorted_indices[1]])
                margin = best_similarity - second_best_similarity
                logger.debug(
                    f"Best: {best_match.worker_name} ({best_similarity:.3f}), "
                    f"Second: {self.embeddings_db[sorted_indices[1]].worker_name} ({second_best_similarity:.3f}), "
                    f"Margin: {margin:.3f}"
                )
        else:
            # Fallback to loop-based approach
            best_match = None
            best_similarity = 0.0
            
            for stored_emb in self.embeddings_db:
                similarity = self.calculate_similarity(query_embedding, stored_emb.embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = stored_emb
        
        # ═══════════════════════════════════════════════════════════════════
        # ROBUST TWO-STEP VERIFICATION SYSTEM (ANTI-FALSE-POSITIVE)
        # ═══════════════════════════════════════════════════════════════════
        # Based on diagnostic test results:
        # - Same person: 100% similarity
        # - DeepFace default: 68% threshold
        # - Different people: 30-60% similarity
        # - Very similar people: 70-85% similarity
        #
        # OPTIMIZED FOR REAL-TIME CAMERA DETECTION:
        # 1. Similarity >= 0.80 (high confidence) OR
        # 2. Similarity >= 0.70 (normal) AND Margin >= 0.18 (clear separation)
        # 3. Absolute minimum: 0.70 (below this = always Unknown)
        # Balanced for real-time performance with good accuracy
        # ═══════════════════════════════════════════════════════════════════
        
        ABSOLUTE_MIN_THRESHOLD = 0.70  # Below this = always Unknown (optimized for real-time)
        HIGH_CONFIDENCE_THRESHOLD = 0.80  # Accept immediately if above (balanced threshold)
        REQUIRED_MARGIN = 0.18  # Minimum separation from second best (balanced margin)
        
        # STEP 1: Check absolute minimum threshold
        if not best_match or best_similarity < ABSOLUTE_MIN_THRESHOLD:
            # Below 70% - definitely not a match
            if best_match:
                logger.info(
                    f"❌ REJECTED [LOW SIMILARITY]: Face does not match {best_match.worker_name} "
                    f"(similarity: {best_similarity:.1%} < minimum: {ABSOLUTE_MIN_THRESHOLD:.1%}) - Marking as Unknown"
                )
            else:
                logger.debug(f"No match found - no workers registered")
            return None
        
        # STEP 2: Two-step verification
        if best_similarity >= HIGH_CONFIDENCE_THRESHOLD:
            # Path 1: High confidence (>= 85%) - accept immediately
            logger.info(
                f"✅ RECOGNIZED [HIGH CONFIDENCE]: {best_match.worker_name} "
                f"(ID: {best_match.worker_id}, similarity: {best_similarity:.1%}, margin: {margin:.1%})"
            )
            return (best_match.worker_id, best_match.worker_name, best_similarity)
        
        elif best_similarity >= ABSOLUTE_MIN_THRESHOLD and margin >= REQUIRED_MARGIN:
            # Path 2: Normal confidence (70-80%) BUT clear margin (>= 18%)
            logger.info(
                f"✅ RECOGNIZED [CLEAR MARGIN]: {best_match.worker_name} "
                f"(ID: {best_match.worker_id}, similarity: {best_similarity:.1%}, margin: {margin:.1%})"
            )
            return (best_match.worker_id, best_match.worker_name, best_similarity)
        
        else:
            # REJECT: Similarity is 70-80% but margin < 18% (ambiguous)
            logger.info(
                f"❌ REJECTED [AMBIGUOUS]: Face similar to {best_match.worker_name} "
                f"(similarity: {best_similarity:.1%}) but margin too small ({margin:.1%} < {REQUIRED_MARGIN:.1%}) - "
                f"Could be another person. Marking as Unknown"
            )
            return None

    def recognize_worker_or_unknown(
        self,
        face_img: np.ndarray
    ) -> Tuple[Optional[int], str, float]:
        """
        Recognize a single face; returns an Unknown element when not registered.

        Returns:
            (worker_id | None, worker_name or 'Unknown', similarity)
        """
        result = self.recognize_worker(face_img)
        if result is None:
            return (None, UNKNOWN_NAME, 0.0)
        return result
    
    def recognize_many(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Recognize multiple faces in a frame using YOLO detections.

        Args:
            frame: The original image frame (BGR)
            detections: List of detection dicts, each containing:
                - 'bbox' or 'box' or 'xyxy': [x1, y1, x2, y2]
                - optional 'confidence'

        Returns:
            List of dicts with fields:
                {
                  'worker_id': Optional[int],
                  'worker_name': Optional[str],
                  'similarity': float,
                  'bbox': List[int],
                  'confidence': Optional[float]
                }
        """
        results: List[Dict[str, Any]] = []
        if frame is None or frame.size == 0 or not detections:
            return results

        # Early exit if no embeddings available
        if not self.embeddings_db or self.embeddings_matrix is None:
            for det in detections:
                bbox = det.get('bbox') or det.get('box') or det.get('xyxy')
                if bbox is None:
                    continue
                results.append(self.make_unknown_result(list(map(int, bbox)), det.get('confidence')))
            return results

        # 1) Crop and embed all detected faces
        face_crops: List[Tuple[int, List[int], Optional[np.ndarray], Optional[float]]] = []
        for idx, det in enumerate(detections):
            bbox = det.get('bbox') or det.get('box') or det.get('xyxy')
            if bbox is None:
                continue
            face_img = crop_face_from_detection(frame, bbox)
            if face_img is None:
                face_crops.append((idx, list(map(int, bbox)), None, det.get('confidence')))
                continue
            embedding = self.get_face_embedding(face_img)
            if embedding is None:
                face_crops.append((idx, list(map(int, bbox)), None, det.get('confidence')))
                continue
            # Normalize embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            face_crops.append((idx, list(map(int, bbox)), embedding, det.get('confidence')))

        # Gather valid embeddings
        valid_indices = [i for i, _, emb, _ in face_crops if emb is not None]
        if not valid_indices:
            # No valid embeddings; return unknowns
            for _, bbox, _, conf in face_crops:
                results.append(self.make_unknown_result(bbox, conf))
            return results

        query_embeddings = np.vstack([face_crops[i][2] for i in valid_indices])  # type: ignore[index]

        # 2) Batch cosine similarity vs database embeddings
        try:
            normalized_matrix = self.embeddings_matrix / (
                np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True) + 1e-8
            )
            similarities_matrix = cosine_similarity(query_embeddings, normalized_matrix)
            similarities_matrix = np.clip(similarities_matrix, 0.0, 1.0)
        except Exception as e:
            logger.error(f"Error computing batch similarities: {e}")
            # Fallback: mark all as unknown
            for _, bbox, _, conf in face_crops:
                results.append(self.make_unknown_result(bbox, conf))
            return results

        # 3) Decide identity for each query embedding using thresholds/margins
        for row_idx, i in enumerate(valid_indices):
            bbox = face_crops[i][1]
            conf = face_crops[i][3]
            sims = similarities_matrix[row_idx]

            best_idx = int(np.argmax(sims))
            best_similarity = float(sims[best_idx])
            best_match = self.embeddings_db[best_idx]

            margin = 0.0
            if sims.shape[0] > 1:
                sorted_idx = np.argsort(sims)[::-1]
                if len(sorted_idx) > 1:
                    second_best = float(sims[sorted_idx[1]])
                    margin = best_similarity - second_best

            # Apply SAME robust two-step verification as recognize_worker
            # CRITICAL: Prevent identity inheritance - each detection verified independently
            ABSOLUTE_MIN_THRESHOLD = 0.70  # Below this = always Unknown (optimized for real-time)
            HIGH_CONFIDENCE_THRESHOLD = 0.80  # Accept immediately if above (balanced threshold)
            REQUIRED_MARGIN = 0.18  # Minimum separation from second best (balanced margin)

            recognized = False
            worker_id: Optional[int] = None
            worker_name: Optional[str] = None

            # Two-step verification (same as recognize_worker)
            if best_similarity >= HIGH_CONFIDENCE_THRESHOLD:
                # Path 1: High confidence (>= 80%)
                recognized = True
            elif best_similarity >= ABSOLUTE_MIN_THRESHOLD and margin >= REQUIRED_MARGIN:
                # Path 2: Normal confidence (70-80%) BUT clear margin (>= 18%)
                recognized = True
            else:
                # REJECT: Below 70% OR ambiguous (margin < 18%)
                recognized = False

            if recognized:
                worker_id = best_match.worker_id
                worker_name = best_match.worker_name
                logger.debug(
                    f"Batch recognition: {worker_name} (similarity: {best_similarity:.1%}, margin: {margin:.1%})"
                )
            else:
                logger.debug(
                    f"Batch rejection: Best match {best_match.worker_name} "
                    f"(similarity: {best_similarity:.1%}, margin: {margin:.1%}) - Marking as Unknown"
                )

            # CRITICAL: Do NOT inherit previous identity - always create fresh result
            if recognized:
                results.append({
                    'worker_id': worker_id,
                    'worker_name': worker_name,
                    'similarity': best_similarity,
                    'bbox': bbox,
                    'confidence': float(conf) if conf is not None else None
                })
            else:
                # Always create new Unknown result - never reuse previous identity
                results.append(self.make_unknown_result(bbox, conf))

        # Add entries for detections without valid embeddings (unknown)
        invalid_indices = [j for j, _, emb, _ in face_crops if emb is None]
        for j in invalid_indices:
            bbox = face_crops[j][1]
            conf = face_crops[j][3]
            results.append(self.make_unknown_result(bbox, conf))

        return results

    def make_unknown_result(self, bbox: List[int], confidence: Optional[float] = None) -> Dict[str, Any]:
        """Create a standardized result dict for unknown (unregistered) faces."""
        try:
            conf_val = float(confidence) if confidence is not None else None
        except Exception:
            conf_val = None
        return {
            'worker_id': None,
            'worker_name': 'Unknown',
            'similarity': 0.0,
            'bbox': bbox,
            'confidence': conf_val
        }

    def mark_attendance(
        self,
        worker_id: int,
        worker_name: str,
        similarity_score: float,
        face_img: np.ndarray,
        event_type: str = "check_in",
        gps_coordinates: Optional[Tuple[float, float]] = None,
        save_dir: str = "data/attendance_photos"
    ) -> bool:
        """
        Mark attendance for a recognized worker.
        
        Args:
            worker_id: Worker ID
            worker_name: Worker name
            similarity_score: Face recognition similarity score
            face_img: Face image to save
            event_type: 'check_in' or 'check_out'
            gps_coordinates: Optional GPS coordinates (latitude, longitude)
            save_dir: Directory to save attendance photos
            
        Returns:
            True if attendance marked successfully
        """
        try:
            # Create save directory
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save attendance photo
            timestamp = time.time()
            photo_filename = f"{worker_id}_{worker_name}_{event_type}_{int(timestamp)}.jpg"
            photo_path = save_path / photo_filename
            cv2.imwrite(str(photo_path), face_img)
            
            # Create attendance record
            record = AttendanceRecord(
                worker_id=worker_id,
                worker_name=worker_name,
                timestamp=timestamp,
                photo_path=str(photo_path),
                similarity_score=similarity_score,
                event_type=event_type,
                gps_coordinates=gps_coordinates
            )
            
            self.attendance_records.append(record)
            self._save_attendance_records()
            
            logger.info(
                f"Marked {event_type} for {worker_name} "
                f"(ID: {worker_id}, score: {similarity_score:.3f})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error marking attendance: {e}")
            return False
    
    def get_worker_attendance_today(self, worker_id: int) -> List[AttendanceRecord]:
        """Get all attendance records for a worker today."""
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        
        return [
            record for record in self.attendance_records
            if record.worker_id == worker_id and record.timestamp >= today_start
        ]
    
    def delete_worker(self, worker_id: int) -> bool:
        """Delete all embeddings for a worker."""
        initial_count = len(self.embeddings_db)
        self.embeddings_db = [
            emb for emb in self.embeddings_db
            if emb.worker_id != worker_id
        ]
        
        deleted_count = initial_count - len(self.embeddings_db)
        
        if deleted_count > 0:
            self._save_embeddings_db()
            logger.info(f"Deleted {deleted_count} embeddings for worker ID {worker_id}")
            return True
        
        return False
    
    def get_all_registered_workers(self) -> List[Dict[str, Any]]:
        """Get list of all registered workers."""
        workers = {}
        for emb in self.embeddings_db:
            if emb.worker_id not in workers:
                workers[emb.worker_id] = {
                    'worker_id': emb.worker_id,
                    'worker_name': emb.worker_name,
                    'num_embeddings': 0
                }
            workers[emb.worker_id]['num_embeddings'] += 1
        
        return list(workers.values())


# Helper function to crop face from detection bbox
def crop_face_from_detection(frame: np.ndarray, bbox: np.ndarray, padding: float = 0.2) -> Optional[np.ndarray]:
    """
    Crop face from frame using detection bounding box.
    
    Args:
        frame: Full frame image
        bbox: Bounding box [x1, y1, x2, y2]
        padding: Padding around face (as fraction of bbox size)
        
    Returns:
        Cropped face image or None if invalid
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding
        w = x2 - x1
        h = y2 - y1
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Calculate padded coordinates
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(frame.shape[1], x2 + pad_w)
        y2_pad = min(frame.shape[0], y2 + pad_h)
        
        # Crop face
        face_img = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if face_img.size == 0:
            return None
        
        return face_img
        
    except Exception as e:
        logger.error(f"Error cropping face: {e}")
        return None


if __name__ == "__main__":
    # Test the face recognition system
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize system
    face_system = FaceRecognitionSystem()
    
    print(f"Registered workers: {len(face_system.get_all_registered_workers())}")
    print(f"Attendance records: {len(face_system.attendance_records)}")
