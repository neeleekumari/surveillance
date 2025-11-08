"""
Enhanced Face Recognition Module with 3D Features (Stable Version)
------------------------------------------------------------------
Integrates:
1. ArcFace 2D Embeddings (512-D)
2. MiDaS Depth Estimation (optional, robust fallback)
3. MediaPipe 3D Face Mesh (468 landmarks)
4. Multi-view Fusion and Liveness Detection
5. Offline + GPU/CPU Safe Execution
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import time
import torch
import mediapipe as mp
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ======================================================================
# 3D FEATURE EXTRACTION (MiDaS + MediaPipe)
# ======================================================================

@dataclass
class Face3DFeatures:
    """Container for 3D and liveness face features."""
    embedding_2d: np.ndarray
    depth_map: Optional[np.ndarray] = None
    depth_variance: float = 0.0
    depth_mean: float = 0.0
    depth_flatness: float = 1.0
    landmarks_3d: Optional[np.ndarray] = None
    mesh_variance: float = 0.0
    mesh_geometry_hash: Optional[np.ndarray] = None
    view_angle: float = 0.0
    quality_score: float = 0.0
    is_live: bool = False
    liveness_confidence: float = 0.0


class Face3DRecognitionSystem:
    """3D-Enhanced Face Recognition with robust fallbacks."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 depth_model_type: str = "MiDaS_small"):
        self.device = device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable — switching to CPU.")
            self.device = "cpu"

        logger.info(f"Initializing 3D Face Recognition on {self.device}")
        self._init_midas(depth_model_type)
        self._init_mediapipe()

        # Tuned thresholds for realistic conditions
        self.liveness_depth_threshold = 0.07
        self.liveness_flatness_threshold = 0.05
        self.liveness_mesh_variance_threshold = 0.004
        logger.info("3D system ready with adaptive thresholds.")

    def _init_midas(self, model_type: str):
        """Initialize MiDaS safely."""
        try:
            logger.info(f"Loading MiDaS model: {model_type}")
            self.midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            self.midas.to(self.device).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.midas_transform = (
                midas_transforms.dpt_transform
                if model_type in ["DPT_Large", "DPT_Hybrid"]
                else midas_transforms.small_transform
            )
            logger.info("MiDaS depth model initialized.")
        except Exception as e:
            logger.warning(f"MiDaS unavailable: {e}")
            self.midas, self.midas_transform = None, None

    def _init_mediapipe(self):
        """Initialize MediaPipe Face Mesh."""
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe Face Mesh initialized.")
        except Exception as e:
            logger.warning(f"MediaPipe unavailable: {e}")
            self.face_mesh = None

    def estimate_depth(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Generate depth map."""
        if self.midas is None or face_img is None:
            return None
        try:
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            input_batch = self.midas_transform(img_rgb).to(self.device)
            with torch.no_grad():
                prediction = self.midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False
                ).squeeze()
            depth_map = prediction.cpu().numpy()
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            return 1.0 - depth_map
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return None

    def extract_face_mesh(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract 468 3D landmarks."""
        if self.face_mesh is None or face_img is None:
            return None
        try:
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)
            if not results.multi_face_landmarks:
                return None
            h, w = face_img.shape[:2]
            return np.array([[lm.x * w, lm.y * h, lm.z * w]
                             for lm in results.multi_face_landmarks[0].landmark])
        except Exception as e:
            logger.error(f"Mesh extraction failed: {e}")
            return None

    def compute_depth_features(self, depth_map: np.ndarray) -> Dict[str, float]:
        if depth_map is None:
            return dict(depth_mean=0.0, depth_variance=0.0,
                        depth_flatness=1.0, depth_range=0.0)
        c = depth_map[depth_map.shape[0]//4:3*depth_map.shape[0]//4,
                      depth_map.shape[1]//4:3*depth_map.shape[1]//4]
        mean, var = float(np.mean(c)), float(np.var(c))
        flatness = 1.0 - np.exp(-var * 10)
        return dict(depth_mean=mean, depth_variance=var, depth_flatness=flatness)

    def compute_mesh_features(self, landmarks_3d: np.ndarray) -> Dict[str, Any]:
        if landmarks_3d is None:
            return dict(mesh_variance=0.0, mesh_geometry_hash=None, head_pose_angle=0.0)
        z = landmarks_3d[:, 2]
        mesh_var = float(np.var(z))
        try:
            landmarks_flat = landmarks_3d.flatten().reshape(1, -1)
            n_samples = landmarks_flat.shape[0]
            if n_samples < 50:
                geometry_hash = np.mean(landmarks_3d, axis=0)
            else:
                pca = PCA(n_components=min(50, n_samples))
                geometry_hash = pca.fit_transform(landmarks_flat)[0]
        except Exception:
            geometry_hash = np.mean(landmarks_3d, axis=0)
        # Pose angle
        nose, chin, left, right = landmarks_3d[1], landmarks_3d[152], landmarks_3d[33], landmarks_3d[263]
        face_vec = nose - ((left + right) / 2)
        angle = float(np.arctan2(face_vec[0], face_vec[1]) * 180 / np.pi)
        return dict(mesh_variance=mesh_var, mesh_geometry_hash=geometry_hash, head_pose_angle=angle)

    def detect_liveness(self, depth_features, mesh_features):
        scores = []
        scores.append(depth_features.get('depth_variance', 0) > self.liveness_depth_threshold)
        scores.append(depth_features.get('depth_flatness', 0) > self.liveness_flatness_threshold)
        scores.append(mesh_features.get('mesh_variance', 0) > self.liveness_mesh_variance_threshold)
        confidence = np.mean(scores)
        return confidence >= 0.6, float(confidence)

    def extract_3d_features(self, face_img: np.ndarray, embedding_2d: np.ndarray) -> Face3DFeatures:
        depth_map = self.estimate_depth(face_img)
        depth_f = self.compute_depth_features(depth_map)
        mesh = self.extract_face_mesh(face_img)
        mesh_f = self.compute_mesh_features(mesh)
        is_live, conf = self.detect_liveness(depth_f, mesh_f)
        return Face3DFeatures(
            embedding_2d=embedding_2d,
            depth_map=depth_map,
            depth_variance=depth_f['depth_variance'],
            depth_mean=depth_f['depth_mean'],
            depth_flatness=depth_f['depth_flatness'],
            landmarks_3d=mesh,
            mesh_variance=mesh_f['mesh_variance'],
            mesh_geometry_hash=mesh_f['mesh_geometry_hash'],
            view_angle=mesh_f['head_pose_angle'],
            is_live=is_live,
            liveness_confidence=conf,
            quality_score=depth_f['depth_variance'] * 100
        )

    def compare_3d_features(self, f1: Face3DFeatures, f2: Face3DFeatures,
                            emb_sim: float) -> Tuple[float, Dict[str, float]]:
        s = {'embedding_2d': emb_sim}
        s['depth'] = 1.0 - min(1.0, abs(f1.depth_mean - f2.depth_mean) + abs(f1.depth_variance - f2.depth_variance)) \
            if f1.depth_map is not None and f2.depth_map is not None else 0.5
        s['mesh_geometry'] = float(cosine_similarity([f1.mesh_geometry_hash], [f2.mesh_geometry_hash])[0][0]) \
            if f1.mesh_geometry_hash is not None and f2.mesh_geometry_hash is not None else 0.5
        s['head_pose'] = 1.0 - min(1.0, abs(f1.view_angle - f2.view_angle) / 90.0)
        weights = {'embedding_2d': 0.6, 'depth': 0.15, 'mesh_geometry': 0.15, 'head_pose': 0.1}
        return sum(s[k] * weights[k] for k in s), s

    def __del__(self):
        try:
            if hasattr(self, 'face_mesh') and self.face_mesh:
                self.face_mesh.close()
                del self.face_mesh
        except Exception:
            pass


# ======================================================================
# ENHANCED FACE RECOGNITION SYSTEM (2D + 3D)
# ======================================================================

@dataclass
class EnhancedFaceEmbedding:
    """Enhanced face embedding with 3D features."""
    worker_id: int
    worker_name: str
    embedding_2d: np.ndarray
    features_3d_list: List[Face3DFeatures]
    avg_depth_variance: float = 0.0
    avg_mesh_variance: float = 0.0
    view_angles: List[float] = None
    num_views: int = 1
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.features_3d_list:
            self.num_views = len(self.features_3d_list)
            self.avg_depth_variance = np.mean([f.depth_variance for f in self.features_3d_list])
            self.avg_mesh_variance = np.mean([f.mesh_variance for f in self.features_3d_list])
            self.view_angles = [f.view_angle for f in self.features_3d_list]


class EnhancedFaceRecognitionSystem:
    """Enhanced Face Recognition System with 3D Features."""
    
    def __init__(self, model_name: str = "ArcFace", similarity_threshold: float = 0.50,
                 enable_3d: bool = True, liveness_required: bool = True,
                 liveness_fail_patience: int = 3):
        if model_name != "ArcFace":
            logger.warning(f"Model '{model_name}' not supported. Forcing ArcFace.")
            model_name = "ArcFace"
        
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.enable_3d = enable_3d
        self.liveness_required = liveness_required
        
        # Database dependency (graceful fallback if unavailable)
        from database_module import DatabaseManager  # must succeed (main enforces DB)
        self.db_manager = DatabaseManager()
        logger.info("Connected to PostgreSQL database (FaceRecognition)")
        
        self.embeddings_db: List[EnhancedFaceEmbedding] = self._load_embeddings_db()
        self.embeddings_matrix_2d: Optional[np.ndarray] = None
        self._build_embeddings_matrix()
        
        if self.enable_3d:
            try:
                self.system_3d = Face3DRecognitionSystem()
                logger.info("3D face recognition features enabled")
            except Exception as e:
                logger.error(f"Failed to initialize 3D system: {e}")
                logger.warning("Falling back to 2D-only recognition")
                self.enable_3d = False
                self.system_3d = None
        else:
            self.system_3d = None
            logger.info("3D features disabled, using 2D-only recognition")
        
        self._validate_model()
        logger.info(f"Enhanced Face Recognition System initialized (3D: {self.enable_3d})")
        
        # Rate limiting for warning logs
        self._last_no_workers_warning = 0
        self._no_workers_warning_interval = 60  # Log warning once per 60 seconds

        # Recognition state + rate limiting (log only on change or every 60s)
        self._last_recognized_worker_id = None
        self._last_recognition_time = 0
        self._recognition_log_interval = 60  # seconds
        self._last_state = "idle"  # idle | known | unknown
        # Safety margins to reduce false positives
        # Revert margin to stable default to avoid flicker between Known/Unknown
        self._min_similarity_margin = 0.05  # top1 - top2 must exceed this
        self._min_similarity_required = max(0.0, min(1.0, self.similarity_threshold))
        # High-confidence override: if best similarity is extremely high,
        # accept even if margin is slightly below, to prevent false Unknowns
        self._high_confidence_override = 0.92
        # Liveness smoothing
        self._liveness_fail_patience = max(1, liveness_fail_patience)
        self._consecutive_liveness_fails = 0
    
    def _load_embeddings_db(self) -> List[EnhancedFaceEmbedding]:
        if self.db_manager is None:
            return []
        db_embeddings = self.db_manager.get_all_face_embeddings()
        result: List[EnhancedFaceEmbedding] = []
        for emb_data in db_embeddings:
            embedding_2d = emb_data['embedding']
            # Load all 3D feature views for this worker
            features_3d_list: List[Face3DFeatures] = []
            try:
                raw_3d_features = self.db_manager.get_3d_features(emb_data['worker_id'])
                for f in raw_3d_features:
                    features_3d_list.append(Face3DFeatures(
                        embedding_2d=embedding_2d,  # link same 2D embedding (per latest)
                        depth_map=f.get('depth_map'),
                        depth_variance=f.get('depth_variance', 0.0),
                        depth_mean=f.get('depth_mean', 0.0),
                        depth_flatness=f.get('depth_flatness', 0.0),
                        landmarks_3d=f.get('landmarks_3d'),
                        mesh_variance=f.get('mesh_variance', 0.0),
                        mesh_geometry_hash=f.get('mesh_geometry_hash'),
                        view_angle=f.get('view_angle', 0.0),
                        is_live=f.get('liveness_score', 0.0) >= 0.6,
                        liveness_confidence=f.get('liveness_score', 0.0),
                        quality_score=f.get('depth_variance', 0.0) * 100.0
                    ))
            except Exception as e:
                logger.debug(f"3D feature load failed for worker {emb_data['worker_id']}: {e}")
            result.append(EnhancedFaceEmbedding(
                worker_id=emb_data['worker_id'],
                worker_name=emb_data['worker_name'],
                embedding_2d=embedding_2d,
                features_3d_list=features_3d_list
            ))
        logger.info(f"Loaded {len(result)} enhanced embeddings from database (with 3D views)")
        return result
    
    def _build_embeddings_matrix(self) -> None:
        if not self.embeddings_db:
            self.embeddings_matrix_2d = None
            return
        dimensions = [emb.embedding_2d.shape[0] for emb in self.embeddings_db]
        unique_dims = set(dimensions)
        expected_dim = 512
        if len(unique_dims) > 1 or expected_dim not in unique_dims:
            error_msg = f"CRITICAL: Invalid embedding dimensions detected! Found: {unique_dims}, Expected: {expected_dim}-D"
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.embeddings_matrix_2d = np.vstack([emb.embedding_2d for emb in self.embeddings_db])
    
    def _validate_model(self) -> None:
        if self.model_name != 'ArcFace':
            raise ValueError(f"UNSUPPORTED MODEL: {self.model_name}. Only ArcFace is supported.")
        expected_dim = 512
        dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        test_embedding = self.get_face_embedding_2d(dummy_face)
        if test_embedding.shape[0] != expected_dim:
            error_msg = f"MODEL VALIDATION FAILED! Expected {expected_dim}-D, got {test_embedding.shape[0]}-D."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        logger.info(f"Model validation passed: {self.model_name} generates {expected_dim}-D embeddings")
    
    def reload_embeddings(self) -> None:
        logger.info("Reloading embeddings from database...")
        self.embeddings_db = self._load_embeddings_db()
        self._build_embeddings_matrix()
        logger.info(f"Reloaded {len(self.embeddings_db)} embeddings")
    
    def get_face_embedding_2d(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        if face_img is None or face_img.size == 0:
            return None
        target_size = (112, 112)
        face_img = cv2.resize(face_img, target_size, interpolation=cv2.INTER_CUBIC)
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        embedding_objs = DeepFace.represent(img_path=face_rgb, model_name=self.model_name,
                                           enforce_detection=False, detector_backend='skip')
        if embedding_objs and len(embedding_objs) > 0:
            embedding = np.array(embedding_objs[0]['embedding'])
            if embedding.shape[0] != 512:
                logger.error("CRITICAL: Embedding dimension mismatch!")
                return None
            return embedding
        return None
    
    def register_worker(self, worker_id: int, worker_name: str, face_images: List[np.ndarray],
                       guidance_callback=None) -> bool:
        if len(face_images) < 3:
            logger.error("Need at least 3 face images for registration")
            return False
        logger.info(f"Registering {worker_name} with {len(face_images)} views...")
        features_list = []
        for idx, face_img in enumerate(face_images):
            try:
                embedding_2d = self.get_face_embedding_2d(face_img)
                if embedding_2d is None:
                    continue
                if self.enable_3d and self.system_3d:
                    features_3d = self.system_3d.extract_3d_features(face_img, embedding_2d)
                    features_list.append(features_3d)
                else:
                    features_list.append(Face3DFeatures(embedding_2d=embedding_2d))
            except Exception as e:
                logger.error(f"Error extracting features from view {idx}: {e}")
                continue
        
        if not features_list:
            logger.error(f"Failed to register {worker_name} - no valid features extracted")
            return False
        
        embeddings_2d = [f.embedding_2d for f in features_list]
        similarities = np.zeros((len(embeddings_2d), len(embeddings_2d)))
        for i in range(len(embeddings_2d)):
            for j in range(len(embeddings_2d)):
                similarities[i, j] = self.calculate_similarity_2d(embeddings_2d[i], embeddings_2d[j])
        avg_similarities = np.mean(similarities, axis=1)
        median_idx = np.argmax(avg_similarities)
        final_embedding_2d = embeddings_2d[median_idx]
        final_embedding_2d = final_embedding_2d / (np.linalg.norm(final_embedding_2d) + 1e-8)
        
        enhanced_emb = EnhancedFaceEmbedding(
            worker_id=worker_id, worker_name=worker_name,
            embedding_2d=final_embedding_2d,
            features_3d_list=features_list, timestamp=time.time()
        )
        
        self.embeddings_db.append(enhanced_emb)

        if self.db_manager is not None:
            try:
                self.db_manager.add_worker(worker_name, worker_id=worker_id)
                self.db_manager.delete_worker_embeddings(worker_id)
                self.db_manager.save_face_embedding(worker_id, final_embedding_2d, self.model_name)
            except Exception as e:
                logger.warning(f"DB persistence failed (embedding) for {worker_name}: {e}")
            # Photos
            for idx, face_img in enumerate(face_images):
                try:
                    _, buffer = cv2.imencode('.jpg', face_img)
                    photo_bytes = buffer.tobytes()
                    self.db_manager.save_face_photo(worker_id, photo_bytes, 'jpg')
                except Exception as e:
                    logger.warning(f"DB persistence failed (photo {idx+1}) for {worker_name}: {e}")
            # 3D features
            for idx, features_3d in enumerate(features_list):
                try:
                    self.db_manager.save_3d_features(worker_id, features_3d)
                except Exception as e:
                    logger.warning(f"DB persistence failed (3D features {idx+1}) for {worker_name}: {e}")
        else:
            logger.info(f"Registered {worker_name} (memory-only, DB unavailable)")
        
        self._build_embeddings_matrix()
        logger.info(f"Registered {worker_name} with {len(features_list)} views")
        return True
    
    def calculate_similarity_2d(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        embedding1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        embedding2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return float(np.clip(similarity, 0.0, 1.0))
    
    def recognize_worker(self, face_img: np.ndarray, check_liveness: bool = True) -> Optional[Tuple[int, str, float, Dict[str, Any]]]:
        if not self.embeddings_db:
            # Rate-limited warning: only log once per interval
            current_time = time.time()
            if current_time - self._last_no_workers_warning >= self._no_workers_warning_interval:
                logger.warning("No workers registered in database")
                self._last_no_workers_warning = current_time
            return None
        
        query_embedding_2d = self.get_face_embedding_2d(face_img)
        if query_embedding_2d is None:
            logger.warning("Failed to generate embedding for query face")
            return None
        query_embedding_2d = query_embedding_2d / (np.linalg.norm(query_embedding_2d) + 1e-8)
        
        query_features_3d = None
        if self.enable_3d and self.system_3d:
            query_features_3d = self.system_3d.extract_3d_features(face_img, query_embedding_2d)
            if check_liveness and self.liveness_required:
                if not query_features_3d.is_live:
                    self._consecutive_liveness_fails += 1
                else:
                    self._consecutive_liveness_fails = 0
                if self._consecutive_liveness_fails >= self._liveness_fail_patience:
                    if self._should_log_state_change("unknown"):
                        logger.warning(
                            f"Liveness check FAILED ({self._consecutive_liveness_fails}x) Confidence: {query_features_3d.liveness_confidence:.2f} - marking as Unknown"
                        )
                    return (-1, "Unknown", 0.0, {'liveness_failed': True, 'liveness_confidence': query_features_3d.liveness_confidence, 'fails': self._consecutive_liveness_fails})
        
        if self.embeddings_matrix_2d is not None:
            normalized_matrix = self.embeddings_matrix_2d / (np.linalg.norm(self.embeddings_matrix_2d, axis=1, keepdims=True) + 1e-8)
            similarities_2d = cosine_similarity([query_embedding_2d], normalized_matrix)[0]
            similarities_2d = np.clip(similarities_2d, 0.0, 1.0)
            # Margin-based unknown check to reduce false positives
            order = np.argsort(similarities_2d)[::-1]
            best_idx = int(order[0])
            best_similarity_2d = float(similarities_2d[best_idx])
            second_best = float(similarities_2d[int(order[1])]) if len(order) > 1 else 0.0
            margin = best_similarity_2d - second_best
            best_similarity_2d = float(similarities_2d[best_idx])
            best_match = self.embeddings_db[best_idx]
            # Enforce minimum margin and threshold with high-confidence override
            below_threshold = best_similarity_2d < self._min_similarity_required
            below_margin = margin < self._min_similarity_margin
            if below_threshold or (below_margin and best_similarity_2d < self._high_confidence_override):
                if self._should_log_state_change("unknown"):
                    reason = []
                    if below_threshold:
                        reason.append(f"<{self._min_similarity_required:.2f}")
                    if below_margin and best_similarity_2d < self._high_confidence_override:
                        reason.append(f"margin < {self._min_similarity_margin:.3f}")
                    logger.info(
                        f"Unknown person (2D) - top1={best_similarity_2d:.3f}, top2={second_best:.3f}, margin={margin:.3f} ({' and '.join(reason)})"
                    )
                return (-1, "Unknown", best_similarity_2d, {'below_threshold': below_threshold, 'below_margin': below_margin, 'margin': margin})
            
            if self.enable_3d and query_features_3d and best_match.features_3d_list:
                view_similarities = []
                for stored_features_3d in best_match.features_3d_list:
                    combined_sim, component_scores = self.system_3d.compare_3d_features(
                        query_features_3d, stored_features_3d, best_similarity_2d)
                    view_similarities.append((combined_sim, component_scores))
                best_view_idx = np.argmax([s[0] for s in view_similarities])
                final_similarity, component_scores = view_similarities[best_view_idx]
                debug_info = {'component_scores': component_scores, 'best_view_idx': best_view_idx,
                             'num_views_compared': len(view_similarities), 'liveness': query_features_3d.is_live}
                if final_similarity >= self.similarity_threshold:
                    # Log only on state/person change or periodic interval
                    if self._should_log_recognition(best_match.worker_id):
                        logger.info(f"Recognized: {best_match.worker_name} (similarity: {final_similarity:.3f})")
                    return (best_match.worker_id, best_match.worker_name, final_similarity, debug_info)
                else:
                    # Unknown person - log on state change/periodic
                    if self._should_log_state_change("unknown"):
                        logger.info(f"Unknown person - closest: {best_match.worker_name} ({final_similarity:.3f} < {self.similarity_threshold:.3f})")
                    debug_info['below_threshold'] = True
                    return (-1, "Unknown", final_similarity, debug_info)
            else:
                debug_info = {'component_scores': {'embedding_2d': best_similarity_2d}, '3d_available': False}
                if best_similarity_2d >= self.similarity_threshold:
                    if self._should_log_recognition(best_match.worker_id):
                        logger.info(f"Recognized (2D): {best_match.worker_name} (similarity: {best_similarity_2d:.3f})")
                    return (best_match.worker_id, best_match.worker_name, best_similarity_2d, debug_info)
                else:
                    if self._should_log_state_change("unknown"):
                        logger.info(f"Unknown person (2D) - closest: {best_match.worker_name} ({best_similarity_2d:.3f} < {self.similarity_threshold:.3f})")
                    debug_info['below_threshold'] = True
                    return (-1, "Unknown", best_similarity_2d, debug_info)
        return None

    # ----------------------
    # Logging helpers
    # ----------------------
    def _should_log_recognition(self, worker_id: int) -> bool:
        now = time.time()
        if self._last_state != "known" or self._last_recognized_worker_id != worker_id:
            self._last_state = "known"
            self._last_recognized_worker_id = worker_id
            self._last_recognition_time = now
            return True
        if now - self._last_recognition_time >= self._recognition_log_interval:
            self._last_recognition_time = now
            return True
        return False

    def _should_log_state_change(self, new_state: str) -> bool:
        now = time.time()
        if self._last_state != new_state:
            self._last_state = new_state
            self._last_recognized_worker_id = -1 if new_state == "unknown" else self._last_recognized_worker_id
            self._last_recognition_time = now
            return True
        if now - self._last_recognition_time >= self._recognition_log_interval:
            self._last_recognition_time = now
            return True
        return False


# ======================================================================
# SAFE IMAGE CROPPER
# ======================================================================

def crop_face_from_detection(frame: np.ndarray, bbox: np.ndarray, padding: float = 0.2) -> Optional[np.ndarray]:
    """Crop face region with padding."""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        w, h = x2 - x1, y2 - y1
        pad_w, pad_h = int(w * padding), int(h * padding)
        x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
        x2, y2 = min(frame.shape[1], x2 + pad_w), min(frame.shape[0], y2 + pad_h)
        face = frame[y1:y2, x1:x2]
        return face if face.size > 0 else None
    except Exception as e:
        logger.error(f"Error cropping face: {e}")
        return None


# ======================================================================
# QUICK TEST (optional)
# ======================================================================
if __name__ == "__main__":
    logger.info("Testing 3D Face Recognition module initialization...")
    system = Face3DRecognitionSystem()
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    emb = np.random.rand(512)
    features = system.extract_3d_features(dummy, emb)
    logger.info(f"Depth variance: {features.depth_variance:.5f}, Liveness: {features.is_live}, Confidence: {features.liveness_confidence:.2f}")
