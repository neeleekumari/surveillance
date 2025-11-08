"""
Configuration Manager Module
--------------------------
Handles loading, validating, and managing application configuration.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages application configuration loading and validation."""
    
    def __init__(self, config_path: str = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        if config_path is None:
            # Default to config file in project root
            config_path = str(Path(__file__).parent.parent / "config" / "config.json")
        self.config_path = Path(config_path)
        self.default_config = {
            "database": {
                "host": "localhost",
                "name": "floor_monitor",
                "user": "postgres",
                "password": "psql",
                "port": 5432
            },
            "cameras": [
                {
                    "id": 0,
                    "name": "Main Entrance",
                    "rois": []
                }
            ],
            "thresholds": {
                "warning_minutes": 15,
                "alert_minutes": 30
            },
            "notifications": {
                "enabled": True,
                "sound": True
            },
            "app": {
                "version": "1.0.0",
                "debug": True,
                "face_recognition": {
                    "enable_3d": True,
                    "liveness_required": True,
                    "similarity_threshold": 0.50,
                    "model_name": "ArcFace"
                }
            }
        }
        self.config = {}
        
        self.load_config()
        logger.info("ConfigManager initialized")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                # Merge with default config
                self.config = self._merge_config(self.default_config, file_config)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                # Use default config if file doesn't exist
                self.config = self.default_config.copy()
                logger.warning(f"Configuration file not found, using defaults")
                
            # Validate configuration
            self._validate_config()
            
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            # Fall back to default config
            self.config = self.default_config.copy()
            return self.config
    
    def _merge_config(self, default: Dict, override: Dict) -> Dict:
        """Recursively merge two configuration dictionaries.
        
        Args:
            default: Default configuration
            override: Configuration to override defaults
            
        Returns:
            Merged configuration
        """
        merged = default.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_config(merged[key], value)
            else:
                merged[key] = value
                
        return merged
    
    def _validate_config(self) -> None:
        """Validate the configuration values."""
        # Validate database configuration
        db_config = self.config.get("database", {})
        if not isinstance(db_config.get("port", 5432), int):
            raise ValueError("Database port must be an integer")
        
        # Validate thresholds
        thresholds = self.config.get("thresholds", {})
        if not isinstance(thresholds.get("warning_minutes", 15), (int, float)):
            raise ValueError("Warning minutes must be a number")
        if not isinstance(thresholds.get("alert_minutes", 30), (int, float)):
            raise ValueError("Alert minutes must be a number")
        
        # Validate cameras
        cameras = self.config.get("cameras", [])
        if not isinstance(cameras, list):
            raise ValueError("Cameras must be a list")
        
        logger.info("Configuration validation passed")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., "database.host")
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """Set a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
        
        logger.info(f"Configuration updated: {key_path} = {value}")
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save the current configuration to file.
        
        Args:
            config_path: Path to save configuration (defaults to original path)
        """
        save_path = Path(config_path) if config_path else self.config_path
        
        try:
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config = self.default_config.copy()
        logger.info("Configuration reset to defaults")
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration.
        
        Returns:
            Database configuration dictionary
        """
        return self.config.get("database", self.default_config["database"])
    
    def get_camera_configs(self) -> list:
        """Get camera configurations.
        
        Returns:
            List of camera configuration dictionaries
        """
        return self.config.get("cameras", self.default_config["cameras"])
    
    def get_thresholds(self) -> Dict[str, Any]:
        """Get threshold configurations.
        
        Returns:
            Threshold configuration dictionary
        """
        return self.config.get("thresholds", self.default_config["thresholds"])
    
    def get_notification_config(self) -> Dict[str, Any]:
        """Get notification configuration.
        
        Returns:
            Notification configuration dictionary
        """
        return self.config.get("notifications", self.default_config["notifications"])

    def get_face_recognition_config(self) -> Dict[str, Any]:
        """Get face recognition configuration (2D/3D, liveness, thresholds).
        
        Returns:
            Face recognition configuration dictionary
        """
        return self.config.get("app", {}).get("face_recognition", self.default_config["app"]["face_recognition"])


