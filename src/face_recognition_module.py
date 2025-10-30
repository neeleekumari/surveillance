"""
Face Recognition Module
-----------------------
Integrates DeepFace with YOLOv8 for worker face recognition and attendance tracking.
Uses Facenet512 model for generating face embeddings.
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
import pickle

# DeepFace imports
try:
    from deepface import DeepFace
except ImportError:
    raise ImportError("DeepFace not installed. Install with: pip install deepface")

# Scikit-learn for similarity calculations
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)


@dataclass
class FaceEmbedding:
    """Represents a face embedding for a worker."""
    worker_id: int
    worker_name: str
    embedding: np.ndarray
    image_path: str
    timestamp: float
    
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
        embeddings_db_path: str = "data/face_embeddings.pkl",
        attendance_db_path: str = "data/attendance_records.json",
        model_name: str = "Facenet512",
        similarity_threshold: float = 0.6,
        distance_metric: str = "cosine"
    ):
        """
        Initialize the face recognition system.
        
        Args:
            embeddings_db_path: Path to store face embeddings database
            attendance_db_path: Path to store attendance records
            model_name: DeepFace model name (Facenet512, ArcFace, VGG-Face, etc.)
            similarity_threshold: Threshold for face matching (0-1)
            distance_metric: Distance metric ('cosine' or 'euclidean')
        """
        self.embeddings_db_path = Path(embeddings_db_path)
        self.attendance_db_path = Path(attendance_db_path)
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.distance_metric = distance_metric
        
        # Create data directory if it doesn't exist
        self.embeddings_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.attendance_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize embeddings database
        self.embeddings_db: List[FaceEmbedding] = self._load_embeddings_db()
        
        # Load or initialize attendance records
        self.attendance_records: List[AttendanceRecord] = self._load_attendance_records()
        
        logger.info(f"Face Recognition System initialized with {model_name}")
        logger.info(f"Loaded {len(self.embeddings_db)} face embeddings")
    
    def _load_embeddings_db(self) -> List[FaceEmbedding]:
        """Load face embeddings database from file."""
        if self.embeddings_db_path.exists():
            try:
                with open(self.embeddings_db_path, 'rb') as f:
                    data = pickle.load(f)
                    return [FaceEmbedding.from_dict(d) for d in data]
            except Exception as e:
                logger.error(f"Error loading embeddings database: {e}")
                return []
        return []
    
    def _save_embeddings_db(self) -> None:
        """Save face embeddings database to file."""
        try:
            data = [emb.to_dict() for emb in self.embeddings_db]
            with open(self.embeddings_db_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved {len(self.embeddings_db)} embeddings to database")
        except Exception as e:
            logger.error(f"Error saving embeddings database: {e}")
    
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
    
    def get_face_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding using DeepFace.
        
        Args:
            face_img: Cropped face image (BGR format)
            
        Returns:
            Face embedding as numpy array, or None if failed
        """
        try:
            # Ensure image is in correct format
            if face_img is None or face_img.size == 0:
                return None
            
            # Resize face to minimum size for DeepFace (typically 224x224)
            if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                logger.warning("Face image too small for embedding")
                return None
            
            # Convert BGR to RGB (DeepFace expects RGB)
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Generate embedding using DeepFace
            embedding_objs = DeepFace.represent(
                img_path=face_rgb,
                model_name=self.model_name,
                enforce_detection=False,  # We already detected the face
                detector_backend='skip'  # Skip detection since we have cropped face
            )
            
            if embedding_objs and len(embedding_objs) > 0:
                embedding = np.array(embedding_objs[0]['embedding'])
                return embedding
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating face embedding: {e}")
            return None
    
    def register_worker(
        self,
        worker_id: int,
        worker_name: str,
        face_images: List[np.ndarray],
        save_dir: str = "data/registered_faces"
    ) -> bool:
        """
        Register a worker with multiple face images.
        
        Args:
            worker_id: Unique worker ID
            worker_name: Worker's name
            face_images: List of 3-5 cropped face images
            save_dir: Directory to save face images
            
        Returns:
            True if registration successful, False otherwise
        """
        if len(face_images) < 3:
            logger.error("Need at least 3 face images for registration")
            return False
        
        save_path = Path(save_dir) / f"worker_{worker_id}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        embeddings_added = 0
        
        for idx, face_img in enumerate(face_images):
            # Generate embedding
            embedding = self.get_face_embedding(face_img)
            
            if embedding is not None:
                # Save face image
                img_filename = f"{worker_name}_{idx}_{int(time.time())}.jpg"
                img_path = save_path / img_filename
                cv2.imwrite(str(img_path), face_img)
                
                # Create and store embedding
                face_emb = FaceEmbedding(
                    worker_id=worker_id,
                    worker_name=worker_name,
                    embedding=embedding,
                    image_path=str(img_path),
                    timestamp=time.time()
                )
                
                self.embeddings_db.append(face_emb)
                embeddings_added += 1
                logger.info(f"Added embedding {idx+1} for worker {worker_name}")
        
        if embeddings_added > 0:
            self._save_embeddings_db()
            logger.info(f"Successfully registered {worker_name} with {embeddings_added} embeddings")
            return True
        else:
            logger.error(f"Failed to register {worker_name} - no valid embeddings generated")
            return False
    
    def calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if self.distance_metric == "cosine":
            # Cosine similarity (convert to 0-1 range)
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return (similarity + 1) / 2  # Convert from [-1, 1] to [0, 1]
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
        face_img: np.ndarray
    ) -> Optional[Tuple[int, str, float]]:
        """
        Recognize a worker from a face image.
        
        Args:
            face_img: Cropped face image
            
        Returns:
            Tuple of (worker_id, worker_name, similarity_score) if recognized, None otherwise
        """
        if not self.embeddings_db:
            logger.warning("No workers registered in database")
            return None
        
        # Generate embedding for the input face
        query_embedding = self.get_face_embedding(face_img)
        
        if query_embedding is None:
            logger.warning("Failed to generate embedding for query face")
            return None
        
        # Find best match
        best_match = None
        best_similarity = 0.0
        
        for stored_emb in self.embeddings_db:
            similarity = self.calculate_similarity(query_embedding, stored_emb.embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = stored_emb
        
        # Check if similarity exceeds threshold
        if best_match and best_similarity >= self.similarity_threshold:
            logger.info(
                f"Recognized {best_match.worker_name} "
                f"(ID: {best_match.worker_id}, similarity: {best_similarity:.3f})"
            )
            return (best_match.worker_id, best_match.worker_name, best_similarity)
        else:
            logger.info(f"No match found (best similarity: {best_similarity:.3f})")
            return None
    
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
