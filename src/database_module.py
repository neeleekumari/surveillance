"""
Database module for handling all database operations (hardened, ready-to-run).
- Creates/updates schema idempotently
- Stores face embeddings as float32 with dtype recorded
- Stores 3D features with reconstructable shapes
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import sys

import psycopg2
from psycopg2 import sql

# Third-party numeric
import numpy as np

# Ensure local imports work
sys.path.append(str(Path(__file__).parent))

from config_manager import ConfigManager  # relies on your existing module

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class DatabaseManager:
    def __init__(self, config_path: str = None):
        """
        Initialize database connection using config and ensure schema.
        """
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config" / "config.json")

        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_database_config()
        self.conn: Optional[psycopg2.extensions.connection] = None

        self.connect()
        self._init_database()

    # -------------------------
    # Connection management
    # -------------------------
    def connect(self) -> None:
        """Establish database connection."""
        if self.conn is not None:
            return
        try:
            self.conn = psycopg2.connect(
                host=self.config["host"],
                database=self.config["name"],
                user=self.config["user"],
                password=self.config["password"],
                port=self.config["port"],
            )
            self.conn.autocommit = True
            logger.info("Connected to PostgreSQL.")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}")

    def close(self) -> None:
        """Close the database connection."""
        if self.conn is not None:
            try:
                self.conn.close()
            finally:
                self.conn = None
                logger.info("Database connection closed.")

    def __del__(self):
        self.close()

    # -------------------------
    # Schema management
    # -------------------------
    def _init_database(self) -> None:
        """
        Initialize or migrate database tables in an idempotent way.
        Fixes previous flow where only 'workers' was created unless an exception occurred.  :contentReference[oaicite:4]{index=4}
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")

        try:
            with self.conn.cursor() as cur:
                # workers
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workers (
                        worker_id   SERIAL PRIMARY KEY,
                        name        VARCHAR(100) NOT NULL,
                        position    VARCHAR(100),
                        contact     VARCHAR(100),
                        is_active   BOOLEAN DEFAULT TRUE,
                        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

                # face_embeddings
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS face_embeddings (
                        embedding_id         SERIAL PRIMARY KEY,
                        worker_id            INTEGER REFERENCES workers(worker_id) ON DELETE CASCADE,
                        embedding            BYTEA NOT NULL,
                        embedding_dimension  INTEGER NOT NULL,
                        model_name           VARCHAR(50) NOT NULL DEFAULT 'ArcFace',
                        embedding_dtype      VARCHAR(10) NOT NULL DEFAULT 'float32',
                        created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                # Defensive: add dtype column if older deployments exist
                cur.execute(
                    """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='face_embeddings' AND column_name='embedding_dtype'
                        ) THEN
                            ALTER TABLE face_embeddings ADD COLUMN embedding_dtype VARCHAR(10) NOT NULL DEFAULT 'float32';
                        END IF;
                    END$$;
                    """
                )

                # face_photos
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS face_photos (
                        photo_id     SERIAL PRIMARY KEY,
                        worker_id    INTEGER REFERENCES workers(worker_id) ON DELETE CASCADE,
                        photo_data   BYTEA NOT NULL,
                        photo_format VARCHAR(10) DEFAULT 'jpg',
                        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

                # face_3d_features
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS face_3d_features (
                        feature_id          SERIAL PRIMARY KEY,
                        worker_id           INTEGER REFERENCES workers(worker_id) ON DELETE CASCADE,
                        depth_map           BYTEA,
                        depth_h             INTEGER,
                        depth_w             INTEGER,
                        depth_variance      FLOAT,
                        depth_mean          FLOAT,
                        depth_flatness      FLOAT,
                        landmarks_3d        BYTEA,
                        mesh_variance       FLOAT,
                        mesh_geometry_hash  BYTEA,
                        view_angle          FLOAT,
                        liveness_score      FLOAT,
                        created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                # Defensive: add shape columns for older deployments
                cur.execute(
                    """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='face_3d_features' AND column_name='depth_h'
                        ) THEN
                            ALTER TABLE face_3d_features
                            ADD COLUMN depth_h INTEGER,
                            ADD COLUMN depth_w INTEGER;
                        END IF;
                    END$$;
                    """
                )

            logger.info("Schema ensured / migrated.")
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise RuntimeError(f"Failed to initialize database: {e}")

    # -------------------------
    # Workers
    # -------------------------
    def add_worker(
        self,
        name: str,
        position: Optional[str] = None,
        contact: Optional[str] = None,
        worker_id: Optional[int] = None,
    ) -> int:
        """Add or upsert a worker (optional explicit worker_id)."""
        if self.conn is None:
            raise RuntimeError("Database connection not established")

        try:
            with self.conn.cursor() as cur:
                if worker_id is not None:
                    cur.execute(
                        """
                        INSERT INTO workers (worker_id, name, position, contact)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (worker_id) DO UPDATE SET
                            name = EXCLUDED.name,
                            position = EXCLUDED.position,
                            contact = EXCLUDED.contact
                        RETURNING worker_id
                        """,
                        (worker_id, name, position, contact),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO workers (name, position, contact)
                        VALUES (%s, %s, %s)
                        RETURNING worker_id
                        """,
                        (name, position, contact),
                    )
                new_id = cur.fetchone()[0]
                return new_id
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise RuntimeError(f"Failed to add worker: {e}")

    def delete_worker(self, worker_id: int) -> None:
        """Delete worker and all related data (CASCADE)."""
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM workers WHERE worker_id = %s", (worker_id,))
                logger.info(f"Deleted worker {worker_id}.")
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise RuntimeError(f"Failed to delete worker: {e}")

    def get_all_workers(self) -> List[Dict[str, Any]]:
        """List all workers."""
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT worker_id, name, position, contact, is_active, created_at
                    FROM workers
                    ORDER BY worker_id
                    """
                )
                rows = cur.fetchall()
                return [
                    dict(
                        worker_id=r[0],
                        name=r[1],
                        position=r[2],
                        contact=r[3],
                        is_active=r[4],
                        created_at=r[5],
                    )
                    for r in rows
                ]
        except Exception as e:
            raise RuntimeError(f"Failed to get all workers: {e}")

    # -------------------------
    # Embeddings (2D)
    # -------------------------
    @staticmethod
    def _to_bytes(arr: np.ndarray) -> Tuple[bytes, str, int]:
        """Serialize numpy array to bytes with dtype and dimension."""
        if arr is None:
            raise ValueError("Embedding array is None")
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        arr = arr.astype(np.float32, copy=False)
        return arr.tobytes(), "float32", arr.shape[0]

    @staticmethod
    def _from_bytes(data: bytes, dim: int, dtype: str) -> np.ndarray:
        """Deserialize bytes back to numpy array with dtype and dimension."""
        np_dtype = np.float32 if dtype == "float32" else np.float64
        arr = np.frombuffer(data, dtype=np_dtype)
        if dim and arr.size != dim:
            # If dimension mismatch, trust buffer size
            logger.warning(f"Embedding size {arr.size} != recorded dim {dim}; using buffer size.")
        return arr

    def save_face_embedding(self, worker_id: int, embedding: Any, model_name: str = "ArcFace") -> int:
        """Save face embedding (float32)."""
        if self.conn is None:
            raise RuntimeError("Database connection not established")

        try:
            emb_np = np.asarray(embedding, dtype=np.float32).reshape(-1)
            emb_bytes, dtype_str, dim = self._to_bytes(emb_np)
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO face_embeddings
                        (worker_id, embedding, embedding_dimension, model_name, embedding_dtype)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING embedding_id
                    """,
                    (worker_id, psycopg2.Binary(emb_bytes), dim, model_name, dtype_str),
                )
                emb_id = cur.fetchone()[0]
                logger.info(f"Saved embedding for worker {worker_id} (dim={dim}, dtype={dtype_str}).")
                return emb_id
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise RuntimeError(f"Failed to save face embedding: {e}")

    def get_face_embedding(self, worker_id: int) -> Optional[np.ndarray]:
        """Get most recent face embedding for a worker."""
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT embedding, embedding_dimension, embedding_dtype
                    FROM face_embeddings
                    WHERE worker_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (worker_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                emb_bytes, dim, dtype_str = row
                return self._from_bytes(emb_bytes, dim, dtype_str)
        except Exception as e:
            raise RuntimeError(f"Failed to get face embedding: {e}")

    def get_all_face_embeddings(self) -> List[Dict[str, Any]]:
        """Get latest embeddings for all active workers."""
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT fe.worker_id, w.name, fe.embedding, fe.embedding_dimension, fe.model_name, fe.embedding_dtype
                    FROM face_embeddings fe
                    JOIN workers w ON fe.worker_id = w.worker_id
                    WHERE w.is_active = TRUE
                    AND fe.created_at = (
                        SELECT MAX(created_at) FROM face_embeddings fe2 WHERE fe2.worker_id = fe.worker_id
                    )
                    ORDER BY w.worker_id ASC
                    """
                )
                results = []
                for worker_id, name, emb_bytes, dim, model_name, dtype_str in cur.fetchall():
                    emb = self._from_bytes(emb_bytes, dim, dtype_str)
                    results.append(
                        dict(
                            worker_id=worker_id,
                            worker_name=name,
                            embedding=emb,
                            model_name=model_name,
                            embedding_dtype=dtype_str,
                            embedding_dimension=int(dim),
                        )
                    )
                return results
        except Exception as e:
            raise RuntimeError(f"Failed to get all face embeddings: {e}")

    def delete_worker_embeddings(self, worker_id: int) -> None:
        """Delete all embeddings/photos/3D features for a worker."""
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM face_embeddings WHERE worker_id = %s", (worker_id,))
                cur.execute("DELETE FROM face_photos WHERE worker_id = %s", (worker_id,))
                cur.execute("DELETE FROM face_3d_features WHERE worker_id = %s", (worker_id,))
                logger.info(f"Deleted all biometric data for worker {worker_id}.")
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise RuntimeError(f"Failed to delete worker embeddings: {e}")

    # -------------------------
    # Photos
    # -------------------------
    def save_face_photo(self, worker_id: int, photo_data: bytes, photo_format: str = "jpg") -> int:
        """Save a registration photo."""
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO face_photos (worker_id, photo_data, photo_format)
                    VALUES (%s, %s, %s)
                    RETURNING photo_id
                    """,
                    (worker_id, psycopg2.Binary(photo_data), photo_format),
                )
                pid = cur.fetchone()[0]
                logger.info(f"Saved photo for worker {worker_id} (id={pid}).")
                return pid
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise RuntimeError(f"Failed to save face photo: {e}")

    def get_face_photos(self, worker_id: int) -> List[bytes]:
        """Get all photos for a worker (latest first)."""
        if self.conn is None:
            raise RuntimeError("Database connection not established")
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT photo_data
                    FROM face_photos
                    WHERE worker_id = %s
                    ORDER BY created_at DESC
                    """,
                    (worker_id,),
                )
                return [r[0] for r in cur.fetchall()]
        except Exception as e:
            raise RuntimeError(f"Failed to get face photos: {e}")

    # -------------------------
    # 3D features
    # -------------------------
    def save_3d_features(self, worker_id: int, features_3d: "Face3DFeatures") -> int:
        """
        Save 3D features. Stores depth map with shape so it can be reconstructed.
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")

        try:
            depth_map_bytes, dh, dw = None, None, None
            if getattr(features_3d, "depth_map", None) is not None:
                dm = np.asarray(features_3d.depth_map, dtype=np.float32)
                depth_map_bytes = dm.tobytes()
                dh, dw = dm.shape[:2]

            landmarks_3d_bytes = None
            if getattr(features_3d, "landmarks_3d", None) is not None:
                landmarks_3d_bytes = np.asarray(features_3d.landmarks_3d, dtype=np.float32).reshape(-1, 3).tobytes()

            mesh_hash_bytes = None
            if getattr(features_3d, "mesh_geometry_hash", None) is not None:
                mesh_hash_bytes = np.asarray(features_3d.mesh_geometry_hash, dtype=np.float32).reshape(-1).tobytes()

            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO face_3d_features
                        (worker_id, depth_map, depth_h, depth_w,
                         depth_variance, depth_mean, depth_flatness,
                         landmarks_3d, mesh_variance, mesh_geometry_hash,
                         view_angle, liveness_score)
                    VALUES (%s, %s, %s, %s,
                            %s, %s, %s,
                            %s, %s, %s,
                            %s, %s)
                    RETURNING feature_id
                    """,
                    (
                        worker_id,
                        psycopg2.Binary(depth_map_bytes) if depth_map_bytes else None,
                        dh,
                        dw,
                        float(getattr(features_3d, "depth_variance", 0.0)),
                        float(getattr(features_3d, "depth_mean", 0.0)),
                        float(getattr(features_3d, "depth_flatness", 0.0)),
                        psycopg2.Binary(landmarks_3d_bytes) if landmarks_3d_bytes else None,
                        float(getattr(features_3d, "mesh_variance", 0.0)),
                        psycopg2.Binary(mesh_hash_bytes) if mesh_hash_bytes else None,
                        float(getattr(features_3d, "view_angle", 0.0)),
                        float(getattr(features_3d, "liveness_confidence", 0.0)),
                    ),
                )
                fid = cur.fetchone()[0]
                logger.info(f"Saved 3D features for worker {worker_id} (id={fid}).")
                return fid
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise RuntimeError(f"Failed to save 3D features: {e}")

    def get_3d_features(self, worker_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve all 3D features for a worker with reconstructed arrays where possible.
        """
        if self.conn is None:
            raise RuntimeError("Database connection not established")

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT depth_map, depth_h, depth_w,
                           depth_variance, depth_mean, depth_flatness,
                           landmarks_3d, mesh_variance, mesh_geometry_hash,
                           view_angle, liveness_score
                    FROM face_3d_features
                    WHERE worker_id = %s
                    ORDER BY created_at DESC
                    """,
                    (worker_id,),
                )
                results = []
                for (
                    depth_bytes,
                    dh,
                    dw,
                    depth_var,
                    depth_mean,
                    depth_flat,
                    landmarks_bytes,
                    mesh_var,
                    mesh_hash_bytes,
                    view_angle,
                    liveness,
                ) in cur.fetchall():
                    depth_map = None
                    if depth_bytes is not None and dh and dw:
                        depth_map = np.frombuffer(depth_bytes, dtype=np.float32).reshape(dh, dw)

                    landmarks_3d = None
                    if landmarks_bytes:
                        arr = np.frombuffer(landmarks_bytes, dtype=np.float32)
                        landmarks_3d = arr.reshape(-1, 3)

                    mesh_hash = None
                    if mesh_hash_bytes:
                        mesh_hash = np.frombuffer(mesh_hash_bytes, dtype=np.float32)

                    results.append(
                        dict(
                            depth_map=depth_map,
                            depth_variance=float(depth_var) if depth_var is not None else 0.0,
                            depth_mean=float(depth_mean) if depth_mean is not None else 0.0,
                            depth_flatness=float(depth_flat) if depth_flat is not None else 0.0,
                            landmarks_3d=landmarks_3d,
                            mesh_variance=float(mesh_var) if mesh_var is not None else 0.0,
                            mesh_geometry_hash=mesh_hash,
                            view_angle=float(view_angle) if view_angle is not None else 0.0,
                            liveness_score=float(liveness) if liveness is not None else 0.0,
                        )
                    )
                return results
        except Exception as e:
            raise RuntimeError(f"Failed to get 3D features: {e}")
