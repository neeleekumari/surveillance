# Floor Monitoring Application API Documentation

## Overview

This document provides documentation for the Floor Monitoring Application API. The application is built using Python and consists of several modules that work together to provide worker presence monitoring capabilities.

## Module Structure

### 1. Camera Manager (`camera_manager.py`)

Handles USB camera initialization, frame capture, and management.

#### Classes

**CameraConfig**
- Data class for camera configuration
- Fields:
  - `camera_id`: int - Camera identifier
  - `name`: str - Camera name
  - `width`: int - Frame width (default: 1280)
  - `height`: int - Frame height (default: 720)
  - `fps`: int - Frames per second (default: 30)
  - `roi`: Optional[Tuple[int, int, int, int]] - Region of interest (x, y, w, h)

**CameraManager**
- Manages multiple USB camera streams with frame buffering

##### Methods

- `__init__(self, camera_configs: List[Dict] = None)`
  - Initialize the camera manager with a list of camera configurations

- `add_camera(self, config: Dict) -> bool`
  - Add a camera with the given configuration
  - Returns: True if successful, False otherwise

- `start(self) -> bool`
  - Start all camera capture threads
  - Returns: True if successful, False otherwise

- `stop(self) -> None`
  - Stop all camera capture threads and release resources

- `get_frame(self, camera_id: int, timeout: float = 1.0) -> Optional[np.ndarray]`
  - Get the latest frame from a camera
  - Returns: Frame as numpy array or None if unavailable

- `get_available_cameras(self, max_test: int = 4) -> List[int]`
  - Detect and return a list of available camera indices
  - Returns: List of available camera IDs

### 2. Detection Module (`detection_module.py`)

Uses YOLOv8 for real-time person detection in video frames.

#### Classes

**Detection**
- Data class representing a detected person in a frame
- Fields:
  - `bbox`: np.ndarray - Bounding box coordinates [x1, y1, x2, y2]
  - `confidence`: float - Detection confidence (0-1)
  - `class_id`: int - Class identifier
  - `tracker_id`: Optional[int] - Tracker identifier
  - `timestamp`: float - Detection timestamp

**PersonDetector**
- Handles person detection using YOLOv8

##### Methods

- `__init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.5, iou_threshold: float = 0.5, device: str = 'cuda:0')`
  - Initialize the person detector

- `detect(self, frame: np.ndarray) -> List[Detection]`
  - Detect persons in a frame
  - Returns: List of Detection objects

- `draw_detections(self, frame: np.ndarray, detections: List[Detection], show_conf: bool = True) -> np.ndarray`
  - Draw detection bounding boxes on the frame
  - Returns: Frame with drawn detections

### 3. Presence Tracker (`presence_tracker.py`)

Tracks worker presence over time using detection data.

#### Classes

**WorkerPresence**
- Tracks presence data for a single worker
- Fields:
  - `worker_id`: int - Worker identifier
  - `name`: str - Worker name
  - `last_seen`: float - Timestamp of last detection
  - `first_detected`: float - Timestamp of first detection in current session
  - `total_presence_time`: float - Total time present (seconds)
  - `status`: str - Current status ('present', 'absent', 'exceeded')
  - `detection_history`: deque - History of detections

**PresenceTracker**
- Tracks presence of multiple workers using detection data

##### Methods

- `__init__(self, config: dict = None)`
  - Initialize the presence tracker

- `add_worker(self, worker_id: int, name: str) -> None`
  - Add a new worker to track

- `remove_worker(self, worker_id: int) -> None`
  - Remove a worker from tracking

- `update_detections(self, detections: List[dict]) -> Dict[int, dict]`
  - Update worker presence based on new detections
  - Returns: Dictionary of worker_id to status updates

- `get_worker_status(self, worker_id: int) -> Optional[dict]`
  - Get current status of a worker
  - Returns: Worker status dictionary or None

- `get_all_statuses(self) -> List[dict]`
  - Get status for all workers
  - Returns: List of worker status dictionaries

- `reset_worker(self, worker_id: int) -> bool`
  - Reset tracking for a worker
  - Returns: True if successful, False otherwise

### 4. Database Module (`database_module.py`)

Handles all database operations with PostgreSQL.

#### Classes

**DatabaseManager**
- Manages database connections and operations

##### Methods

- `__init__(self, config_path: str = "../config/config.json")`
  - Initialize database connection using config

- `connect(self) -> None`
  - Establish database connection

- `add_worker(self, name: str, position: Optional[str] = None, contact: Optional[str] = None) -> int`
  - Add a new worker to the database
  - Returns: Worker ID

- `log_activity(self, worker_id: int, status: str, duration_seconds: Optional[int] = None) -> int`
  - Log worker activity
  - Returns: Log ID

- `get_worker_activities(self, worker_id: int, limit: int = 100) -> List[Dict[str, Any]]`
  - Get recent activities for a worker
  - Returns: List of activity dictionaries

- `close(self) -> None`
  - Close the database connection

### 5. Alert Manager (`alert_manager.py`)

Handles desktop notifications, sound alerts, and visual indicators.

#### Classes

**Alert**
- Data class representing an alert to be displayed
- Fields:
  - `alert_id`: str - Alert identifier
  - `title`: str - Alert title
  - `message`: str - Alert message
  - `alert_type`: str - Alert type ('warning', 'alert', 'info')
  - `timestamp`: float - Alert timestamp
  - `acknowledged`: bool - Whether alert is acknowledged
  - `worker_id`: Optional[int] - Associated worker ID
  - `duration`: Optional[int] - Duration in seconds

**AlertManager**
- Manages alerts and notifications

##### Methods

- `__init__(self, config: dict = None)`
  - Initialize the alert manager

- `add_alert(self, title: str, message: str, alert_type: str = "info", worker_id: int = None, duration: int = None) -> str`
  - Add a new alert to be displayed
  - Returns: Alert ID

- `acknowledge_alert(self, alert_id: str) -> bool`
  - Acknowledge an alert by ID
  - Returns: True if successful, False otherwise

- `get_active_alerts(self) -> List[Alert]`
  - Get all unacknowledged alerts
  - Returns: List of Alert objects

- `get_alert_history(self, limit: int = 50) -> List[Alert]`
  - Get recent alert history
  - Returns: List of Alert objects

- `clear_alerts(self) -> int`
  - Clear all current alerts
  - Returns: Number of cleared alerts

- `stop(self) -> None`
  - Stop the alert manager

### 6. UI Manager (`ui_manager.py`)

Handles the PyQt5 graphical user interface.

#### Classes

**CameraWidget**
- Widget for displaying camera feed

**WorkerStatusWidget**
- Widget for displaying worker status information

**AlertWidget**
- Widget for displaying alerts and notifications

**SettingsDialog**
- Dialog for application settings

**UIManager**
- Main UI manager for the floor monitoring application

##### Methods

- `__init__(self, config: dict = None)`
  - Initialize the main UI

- `start_monitoring(self)`
  - Start camera monitoring

- `stop_monitoring(self)`
  - Stop camera monitoring

- `open_settings(self)`
  - Open the settings dialog

- `add_camera(self, camera_id: int)`
  - Add a camera widget to the UI

- `update_camera_frame(self, camera_id: int, frame: np.ndarray)`
  - Update a camera's frame

- `update_camera_status(self, camera_id: int, status: str, color: str = "black")`
  - Update a camera's status

- `update_worker_status(self, workers: List[Dict])`
  - Update the worker status display

### 7. Report Generator (`report_generator.py`)

Generates reports and visualizations for worker presence data.

#### Classes

**ReportGenerator**
- Generates reports and visualizations

##### Methods

- `__init__(self, db_manager, config: dict = None)`
  - Initialize the report generator

- `generate_daily_report(self, date: datetime = None) -> Dict`
  - Generate a daily report for worker presence
  - Returns: Report data dictionary

- `generate_weekly_report(self, start_date: datetime = None) -> Dict`
  - Generate a weekly report for worker presence
  - Returns: Report data dictionary

- `export_to_csv(self, report_data: Dict, filename: str = None) -> str`
  - Export report data to CSV format
  - Returns: Path to the generated CSV file

- `export_to_pdf(self, report_data: Dict, filename: str = None) -> str`
  - Export report data to PDF format
  - Returns: Path to the generated PDF file

- `get_worker_statistics(self, worker_id: int, days: int = 30) -> Dict`
  - Get statistics for a specific worker over a period
  - Returns: Worker statistics dictionary

### 8. Configuration Manager (`config_manager.py`)

Handles loading, validating, and managing application configuration.

#### Classes

**ConfigManager**
- Manages application configuration

##### Methods

- `__init__(self, config_path: str = "../config/config.json")`
  - Initialize the configuration manager

- `load_config(self) -> Dict[str, Any]`
  - Load configuration from file
  - Returns: Configuration dictionary

- `get(self, key_path: str, default: Any = None) -> Any`
  - Get a configuration value using dot notation
  - Returns: Configuration value or default

- `set(self, key_path: str, value: Any) -> None`
  - Set a configuration value using dot notation

- `save_config(self, config_path: Optional[str] = None) -> None`
  - Save the current configuration to file

- `reset_to_defaults(self) -> None`
  - Reset configuration to default values

- `get_database_config(self) -> Dict[str, Any]`
  - Get database configuration
  - Returns: Database configuration dictionary

- `get_camera_configs(self) -> list`
  - Get camera configurations
  - Returns: List of camera configuration dictionaries

- `get_thresholds(self) -> Dict[str, Any]`
  - Get threshold configurations
  - Returns: Threshold configuration dictionary

- `get_notification_config(self) -> Dict[str, Any]`
  - Get notification configuration
  - Returns: Notification configuration dictionary

## Main Application (`main.py`)

The main application class that integrates all modules.

### Classes

**DetectionThread**
- Thread for running object detection to avoid blocking the UI

**FloorMonitoringApp**
- Main application class for the Floor Monitoring System

##### Methods

- `__init__(self)`
  - Initialize the application

- `start_monitoring(self)`
  - Start camera monitoring and detection

- `stop_monitoring(self)`
  - Stop camera monitoring and detection

- `process_detections(self, detections, camera_id)`
  - Process detections from a camera

- `check_for_alerts(self, updates)`
  - Check for and generate alerts based on worker status updates

- `update_settings(self, new_config)`
  - Update application settings

- `run(self) -> int`
  - Run the application
  - Returns: Exit code

- `cleanup(self)`
  - Clean up resources

## Configuration (`config/config.json`)

The application is configured through a JSON file with the following structure:

```json
{
    "database": {
        "host": "localhost",
        "name": "floor_monitor",
        "user": "postgres",
        "password": "",
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
        "enabled": true,
        "sound": true
    },
    "app": {
        "version": "1.0.0",
        "debug": true
    }
}
```