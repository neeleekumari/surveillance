"""
Database module for handling all database operations.
"""
import json
import os
import psycopg2
from psycopg2 import sql
from typing import Optional, Dict, Any, List
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

# Import config manager
from config_manager import ConfigManager
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, config_path: str = None):
        """Initialize database connection using config."""
        if config_path is None:
            # Default to config file in project root
            config_path = str(Path(__file__).parent.parent / "config" / "config.json")
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_database_config()
        self.conn: Optional[psycopg2.extensions.connection] = None
        self.connect()
        self._init_database()

    def connect(self) -> None:
        """Establish database connection."""
        if self.conn is None:
            try:
                self.conn = psycopg2.connect(
                    host=self.config['host'],
                    database=self.config['name'],
                    user=self.config['user'],
                    password=self.config['password'],
                    port=self.config['port']
                )
                self.conn.autocommit = True
            except Exception as e:
                raise ConnectionError(f"Failed to connect to database: {str(e)}")

    def _init_database(self) -> None:
        """Initialize database tables if they don't exist."""
        if self.conn is None:
            raise Exception("Database connection not established")
            
        try:
            with self.conn.cursor() as cursor:
                # Create workers table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS workers (
                        worker_id SERIAL PRIMARY KEY,
                        name VARCHAR(100) NOT NULL,
                        position VARCHAR(100),
                        contact VARCHAR(100),
                        is_active BOOLEAN DEFAULT true,
                        absent_time INTEGER DEFAULT 0,  -- Total absent time in minutes
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create activity_log table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS activity_log (
                        log_id SERIAL PRIMARY KEY,
                        worker_id INTEGER REFERENCES workers(worker_id),
                        status VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        duration_seconds INTEGER
                    )
                """)
                
                # Create face_embeddings table to store embeddings
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS face_embeddings (
                        embedding_id SERIAL PRIMARY KEY,
                        worker_id INTEGER REFERENCES workers(worker_id) ON DELETE CASCADE,
                        embedding BYTEA NOT NULL,
                        embedding_dimension INTEGER NOT NULL,
                        model_name VARCHAR(50) NOT NULL DEFAULT 'ArcFace',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create face_photos table to store registration photos
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS face_photos (
                        photo_id SERIAL PRIMARY KEY,
                        worker_id INTEGER REFERENCES workers(worker_id) ON DELETE CASCADE,
                        photo_data BYTEA NOT NULL,
                        photo_format VARCHAR(10) DEFAULT 'jpg',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create attendance_records table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS attendance_records (
                        record_id SERIAL PRIMARY KEY,
                        worker_id INTEGER REFERENCES workers(worker_id),
                        worker_name VARCHAR(100),
                        event_type VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        similarity_score FLOAT,
                        face_snapshot BYTEA,
                        gps_latitude FLOAT,
                        gps_longitude FLOAT
                    )
                """)

                # Migration: if old column name `total_absent_time` exists, rename to `absent_time`
                try:
                    cursor.execute(
                        """
                        SELECT column_name FROM information_schema.columns
                        WHERE table_name = 'workers' AND column_name = 'total_absent_time'
                        """
                    )
                    if cursor.fetchone():
                        cursor.execute("ALTER TABLE workers RENAME COLUMN total_absent_time TO absent_time")
                        logger.info("Migrated workers.total_absent_time -> workers.absent_time")
                except Exception:
                    # Non-fatal migration failure; continue
                    pass
                
                # Migration: Add absent_time column if it doesn't exist
                try:
                    cursor.execute(
                        """
                        SELECT column_name FROM information_schema.columns
                        WHERE table_name = 'workers' AND column_name = 'absent_time'
                        """
                    )
                    if not cursor.fetchone():
                        # Column doesn't exist, add it
                        cursor.execute("ALTER TABLE workers ADD COLUMN absent_time INTEGER DEFAULT 0")
                        logger.info("Added absent_time column to workers table")
                except Exception as e:
                    logger.warning(f"Failed to check/add absent_time column: {e}")
                    # Non-fatal migration failure; continue
                    pass

        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise Exception(f"Failed to initialize database: {str(e)}")

    def get_all_workers(self) -> List[tuple]:
        """Get all registered workers from the database."""
        if self.conn is None:
            return []
            
        with self.conn.cursor() as cursor:
            try:
                cursor.execute("""
                    SELECT worker_id, name, position, contact, is_active, absent_time 
                    FROM workers 
                    WHERE is_active = true 
                    ORDER BY worker_id
                """)
                return cursor.fetchall()
            except Exception as e:
                logger.error(f"Failed to fetch workers: {e}")
                return []

    def add_worker(self, name: str, position: Optional[str] = None, contact: Optional[str] = None, worker_id: Optional[int] = None) -> int:
        """Add a new worker to the database with optional specific ID."""
        if self.conn is None:
            raise Exception("Database connection not established")
            
        try:
            with self.conn.cursor() as cursor:
                if worker_id is not None:
                    # Insert with specific worker_id
                    cursor.execute(
                        """
                        INSERT INTO workers (worker_id, name, position, contact)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (worker_id) DO UPDATE SET name = EXCLUDED.name
                        RETURNING worker_id
                        """,
                        (worker_id, name, position, contact)
                    )
                else:
                    # Auto-generate worker_id
                    cursor.execute(
                        """
                        INSERT INTO workers (name, position, contact)
                        VALUES (%s, %s, %s)
                        RETURNING worker_id
                        """,
                        (name, position, contact)
                    )
                result = cursor.fetchone()
                if result:
                    return result[0]
                else:
                    raise Exception("Failed to add worker")
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise Exception(f"Failed to add worker: {str(e)}")

    def log_activity(self, worker_id: int, status: str, duration_seconds: Optional[int] = None) -> int:
        """Log worker activity."""
        if self.conn is None:
            raise Exception("Database connection not established")
            
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO activity_log (worker_id, status, duration_seconds)
                    VALUES (%s, %s, %s)
                    RETURNING log_id
                    """,
                    (worker_id, status, duration_seconds)
                )
                result = cursor.fetchone()
                if result:
                    return result[0]
                else:
                    raise Exception("Failed to log activity")
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise Exception(f"Failed to log activity: {str(e)}")

    def get_worker_activities(self, worker_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent activities for a worker."""
        if self.conn is None:
            raise Exception("Database connection not established")
            
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT log_id, status, timestamp, duration_seconds
                    FROM activity_log
                    WHERE worker_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (worker_id, limit)
                )
                description = cursor.description
                if description:
                    columns = [desc[0] for desc in description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
                else:
                    return []
        except Exception as e:
            raise Exception(f"Failed to get worker activities: {str(e)}")

    def save_face_embedding(self, worker_id: int, embedding: Any, model_name: str = "ArcFace") -> int:
        """Save face embedding to database."""
        import numpy as np
        
        if self.conn is None:
            raise Exception("Database connection not established")
        
        try:
            # Convert numpy array to bytes
            embedding_bytes = embedding.tobytes()
            embedding_dim = len(embedding)
            
            with self.conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO face_embeddings (worker_id, embedding, embedding_dimension, model_name)
                    VALUES (%s, %s, %s, %s)
                    RETURNING embedding_id
                    """,
                    (worker_id, embedding_bytes, embedding_dim, model_name)
                )
                result = cursor.fetchone()
                if result:
                    return result[0]
                else:
                    raise Exception("Failed to save embedding")
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise Exception(f"Failed to save face embedding: {str(e)}")
    
    def get_face_embedding(self, worker_id: int) -> Optional[Any]:
        """Retrieve face embedding from database."""
        import numpy as np
        
        if self.conn is None:
            raise Exception("Database connection not established")
        
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT embedding, embedding_dimension
                    FROM face_embeddings
                    WHERE worker_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (worker_id,)
                )
                result = cursor.fetchone()
                if result:
                    embedding_bytes, dimension = result
                    # Convert bytes back to numpy array
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
                    return embedding
                return None
        except Exception as e:
            raise Exception(f"Failed to get face embedding: {str(e)}")
    
    def get_all_face_embeddings(self) -> List[Dict[str, Any]]:
        """Get all face embeddings from database."""
        import numpy as np
        
        if self.conn is None:
            raise Exception("Database connection not established")
        
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT fe.worker_id, w.name, fe.embedding, fe.embedding_dimension, fe.model_name
                    FROM face_embeddings fe
                    JOIN workers w ON fe.worker_id = w.worker_id
                    WHERE w.is_active = true
                    ORDER BY fe.created_at DESC
                    """
                )
                results = []
                for row in cursor.fetchall():
                    worker_id, name, embedding_bytes, dimension, model_name = row
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
                    results.append({
                        'worker_id': worker_id,
                        'worker_name': name,
                        'embedding': embedding,
                        'model_name': model_name
                    })
                return results
        except Exception as e:
            raise Exception(f"Failed to get all face embeddings: {str(e)}")
    
    def save_face_photo(self, worker_id: int, photo_data: bytes, photo_format: str = 'jpg') -> int:
        """Save face photo to database."""
        if self.conn is None:
            raise Exception("Database connection not established")
        
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO face_photos (worker_id, photo_data, photo_format)
                    VALUES (%s, %s, %s)
                    RETURNING photo_id
                    """,
                    (worker_id, photo_data, photo_format)
                )
                result = cursor.fetchone()
                if result:
                    return result[0]
                else:
                    raise Exception("Failed to save photo")
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise Exception(f"Failed to save face photo: {str(e)}")
    
    def get_face_photos(self, worker_id: int) -> List[bytes]:
        """Get all face photos for a worker."""
        if self.conn is None:
            raise Exception("Database connection not established")
        
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT photo_data
                    FROM face_photos
                    WHERE worker_id = %s
                    ORDER BY created_at DESC
                    """,
                    (worker_id,)
                )
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            raise Exception(f"Failed to get face photos: {str(e)}")
    
    def save_attendance_record(self, worker_id: int, worker_name: str, event_type: str, 
                               similarity_score: float, face_snapshot: Optional[bytes] = None,
                               gps_latitude: Optional[float] = None, gps_longitude: Optional[float] = None) -> int:
        """Save attendance record to database."""
        if self.conn is None:
            raise Exception("Database connection not established")
        
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO attendance_records 
                    (worker_id, worker_name, event_type, similarity_score, face_snapshot, gps_latitude, gps_longitude)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING record_id
                    """,
                    (worker_id, worker_name, event_type, similarity_score, face_snapshot, gps_latitude, gps_longitude)
                )
                result = cursor.fetchone()
                if result:
                    return result[0]
                else:
                    raise Exception("Failed to save attendance record")
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise Exception(f"Failed to save attendance record: {str(e)}")
    
    def delete_worker_embeddings(self, worker_id: int) -> None:
        """Delete all embeddings and photos for a worker (but keep worker record)."""
        if self.conn is None:
            raise Exception("Database connection not established")
        
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("DELETE FROM face_embeddings WHERE worker_id = %s", (worker_id,))
                cursor.execute("DELETE FROM face_photos WHERE worker_id = %s", (worker_id,))
                logger.info(f"Deleted embeddings and photos for worker {worker_id}")
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise Exception(f"Failed to delete worker embeddings: {str(e)}")
    
    def delete_worker(self, worker_id: int) -> bool:
        """
        Completely delete a worker from the database.
        This removes the worker from workers table and cascades to delete:
        - Face embeddings
        - Face photos
        - Activity logs
        
        Args:
            worker_id: ID of the worker to delete
            
        Returns:
            True if successful, False otherwise
        """
        if self.conn is None:
            raise Exception("Database connection not established")
        
        try:
            with self.conn.cursor() as cursor:
                # Delete worker (CASCADE will automatically delete related records)
                cursor.execute("DELETE FROM workers WHERE worker_id = %s", (worker_id,))
                deleted_count = cursor.rowcount
                
                if deleted_count > 0:
                    logger.info(f"Successfully deleted worker {worker_id} from database")
                    return True
                else:
                    logger.warning(f"Worker {worker_id} not found in database")
                    return False
                    
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            logger.error(f"Failed to delete worker {worker_id}: {str(e)}")
            raise Exception(f"Failed to delete worker: {str(e)}")

    def update_worker_absent_time(self, worker_id: int, minutes: int) -> None:
        """Update the total absent time for a worker."""
        if self.conn is None:
            raise Exception("Database connection not established")
            
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE workers 
                    SET absent_time = absent_time + %s 
                    WHERE worker_id = %s
                    """,
                    (minutes, worker_id)
                )
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise Exception(f"Failed to update absent time: {str(e)}")

    def set_worker_absent_time(self, worker_id: int, minutes: int) -> None:
        """Set the total absent time for a worker (overwrite).

        This is useful when tracking a single absence session (shows current absent duration).
        """
        if self.conn is None:
            raise Exception("Database connection not established")
            
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE workers 
                    SET absent_time = %s 
                    WHERE worker_id = %s
                    """,
                    (minutes, worker_id)
                )
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise Exception(f"Failed to set absent time: {str(e)}")

    def get_worker_status(self, worker_id: int = None) -> List[Dict[str, Any]]:
        """Get worker status including absent time."""
        if self.conn is None:
            raise Exception("Database connection not established")
            
        try:
            # First, ensure absent_time column exists
            self._ensure_absent_time_column()
            
            with self.conn.cursor() as cursor:
                if worker_id is not None:
                    cursor.execute(
                        """
            SELECT w.worker_id, w.name, w.position, w.is_active, w.absent_time,
                               COALESCE(al.status, 'absent') as current_status,
                               COALESCE(al.timestamp, w.created_at) as last_update
                        FROM workers w
                        LEFT JOIN LATERAL (
                            SELECT status, timestamp
                            FROM activity_log
                            WHERE worker_id = w.worker_id
                            ORDER BY timestamp DESC
                            LIMIT 1
                        ) al ON true
                        WHERE w.worker_id = %s
                        """,
                        (worker_id,)
                    )
                else:
                    cursor.execute(
                        """
            SELECT w.worker_id, w.name, w.position, w.is_active, w.absent_time,
                               COALESCE(al.status, 'absent') as current_status,
                               COALESCE(al.timestamp, w.created_at) as last_update
                        FROM workers w
                        LEFT JOIN LATERAL (
                            SELECT status, timestamp
                            FROM activity_log
                            WHERE worker_id = w.worker_id
                            ORDER BY timestamp DESC
                            LIMIT 1
                        ) al ON true
                        """
                    )
                
                results = cursor.fetchall()
                return [
                    {
                        'worker_id': row[0],
                        'name': row[1],
                        'position': row[2],
                        'is_active': row[3],
                        'absent_time': row[4] if len(row) > 4 else 0,  # Total absent time in minutes, default to 0 if missing
                        'current_status': row[5] if len(row) > 5 else 'absent',
                        'last_update': row[6] if len(row) > 6 else None
                    }
                    for row in results
                ]
        except Exception as e:
            raise Exception(f"Failed to get worker status: {str(e)}")
    
    def _ensure_absent_time_column(self) -> None:
        """Ensure absent_time column exists in workers table."""
        if self.conn is None:
            return
        
        try:
            with self.conn.cursor() as cursor:
                # Check if column exists
                cursor.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'workers' AND column_name = 'absent_time'
                    """
                )
                if not cursor.fetchone():
                    # Column doesn't exist, add it
                    cursor.execute("ALTER TABLE workers ADD COLUMN absent_time INTEGER DEFAULT 0")
                    logger.info("Added absent_time column to workers table")
        except Exception as e:
            logger.warning(f"Failed to ensure absent_time column exists: {e}")
            # Non-fatal; continue

    def reset_absent_time(self, worker_id: int) -> None:
        """Reset the total absent time for a worker to zero."""
        if self.conn is None:
            raise Exception("Database connection not established")
            
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE workers 
                    SET absent_time = 0 
                    WHERE worker_id = %s
                    """,
                    (worker_id,)
                )
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise Exception(f"Failed to reset absent time: {str(e)}")

    def close(self) -> None:
        """Close the database connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __del__(self):
        """Ensure connection is closed when the object is destroyed."""
        self.close()


# Example usage
if __name__ == "__main__":
    db: Optional[DatabaseManager] = None
    try:
        db = DatabaseManager()
        
        # Add a test worker
        worker_id = db.add_worker("John Doe", "Operator", "john@example.com")
        print(f"Added worker with ID: {worker_id}")
        
        # Log some activity
        log_id = db.log_activity(worker_id, "present")
        print(f"Logged activity with ID: {log_id}")
        
        # Retrieve activities
        activities = db.get_worker_activities(worker_id)
        print("Worker activities:", activities)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if db is not None:
            db.close()