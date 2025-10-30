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

class DatabaseManager:
    def __init__(self, config_path: str = "../config/config.json"):
        """Initialize database connection using config."""
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

        except Exception as e:
            if self.conn:
                self.conn.rollback()
            raise Exception(f"Failed to initialize database: {str(e)}")

    def add_worker(self, name: str, position: Optional[str] = None, contact: Optional[str] = None) -> int:
        """Add a new worker to the database."""
        if self.conn is None:
            raise Exception("Database connection not established")
            
        try:
            with self.conn.cursor() as cursor:
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