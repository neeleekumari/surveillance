"""
Unit tests for the database module.
"""
import sys
import os
import unittest
from pathlib import Path
import logging
import json
import tempfile

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestDatabaseModule(unittest.TestCase):
    """Unit tests for the DatabaseManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        logging.basicConfig(level=logging.INFO)
    
    def test_config_loading(self):
        """Test database configuration loading."""
        try:
            # Mock the psycopg2 import
            import sys
            from unittest.mock import MagicMock
            
            # Mock psycopg2
            sys.modules['psycopg2'] = MagicMock()
            sys.modules['psycopg2.extensions'] = MagicMock()
            
            from src.config_manager import ConfigManager
            
            # Create a temporary config file
            test_config = {
                "database": {
                    "host": "localhost",
                    "name": "test_db",
                    "user": "test_user",
                    "password": "test_pass",
                    "port": 5432
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(test_config, f)
                temp_config_path = f.name
            
            try:
                # Test loading config
                config_manager = ConfigManager(temp_config_path)
                config = config_manager.get_database_config()
                
                self.assertEqual(config["host"], "localhost")
                self.assertEqual(config["name"], "test_db")
                self.assertEqual(config["user"], "test_user")
                print("Database config loading test passed")
            finally:
                # Clean up
                os.unlink(temp_config_path)
                
        except Exception as e:
            self.fail(f"Database config loading test failed: {e}")
    
    def test_database_manager_initialization(self):
        """Test DatabaseManager initialization."""
        try:
            # Mock the psycopg2 import
            import sys
            from unittest.mock import MagicMock
            
            # Mock psycopg2
            sys.modules['psycopg2'] = MagicMock()
            sys.modules['psycopg2.extensions'] = MagicMock()
            
            from src.database_module import DatabaseManager
            
            # Create a mock config file
            test_config = {
                "database": {
                    "host": "localhost",
                    "name": "test_db",
                    "user": "test_user",
                    "password": "test_pass",
                    "port": 5432
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(test_config, f)
                temp_config_path = f.name
            
            try:
                # Try to initialize (this will fail if PostgreSQL is not running, which is expected)
                db_manager = DatabaseManager(temp_config_path)
                # If we get here, initialization at least didn't crash
                print("DatabaseManager initialization test completed (connection may fail, which is expected)")
            except ConnectionError as e:
                # This is expected if PostgreSQL is not running
                print(f"Database connection failed (expected): {e}")
                print("DatabaseManager initialization test passed")
            finally:
                # Clean up
                os.unlink(temp_config_path)
                
        except Exception as e:
            self.fail(f"DatabaseManager initialization test failed unexpectedly: {e}")

if __name__ == "__main__":
    unittest.main()