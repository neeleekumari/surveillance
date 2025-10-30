# Floor Monitoring Application User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Getting Started](#getting-started)
5. [User Interface](#user-interface)
6. [Configuration](#configuration)
7. [Monitoring Workers](#monitoring-workers)
8. [Receiving Alerts](#receiving-alerts)
9. [Generating Reports](#generating-reports)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)

## Introduction

The Floor Monitoring Application is a comprehensive surveillance system designed to monitor worker presence on factory floors using computer vision and AI-powered object detection. The application uses USB cameras and YOLOv8 object detection to track workers in real-time, generate alerts for extended presence, and provide detailed reporting capabilities.

### Key Features

- Real-time person detection using YOLOv8 AI model
- Multi-camera support for monitoring multiple areas
- Worker presence time tracking
- Configurable alerts and warnings
- Database storage for historical data
- Rich graphical user interface
- Report generation in CSV and PDF formats
- Desktop notifications and sound alerts

## System Requirements

### Hardware Requirements

- **CPU**: Intel i5 or equivalent
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free disk space
- **Graphics**: DirectX 11 compatible GPU (recommended)
- **Cameras**: USB cameras (1 or more)
- **Display**: 1920x1080 resolution recommended

### Software Requirements

- **Operating System**: Windows 10/11, Ubuntu 18.04+, or macOS 10.15+
- **Python**: 3.10 or higher
- **Database**: PostgreSQL 14 or higher
- **Dependencies**: As listed in `requirements.txt`

## Installation

### Prerequisites

Before installing the Floor Monitoring Application, ensure you have:

1. Python 3.10 or higher installed
2. PostgreSQL 14 or higher installed
3. USB cameras connected to your computer

### Installation Steps

1. **Download the Application**

   Download the latest release from the GitHub repository or clone the source code:

   ```bash
   git clone <repository-url>
   cd floor_monitoring_app
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up PostgreSQL Database**

   Create a database and user for the application:

   ```sql
   CREATE USER floor_monitor WITH PASSWORD 'your_password';
   CREATE DATABASE floor_monitor OWNER floor_monitor;
   ```

5. **Configure the Application**

   Edit `config/config.json` with your database settings and preferences.

## Getting Started

### First Run

1. **Configure Cameras**

   Connect your USB cameras to the computer and note their device IDs.

2. **Update Configuration**

   Edit `config/config.json` to match your setup:

   ```json
   {
       "database": {
           "host": "localhost",
           "name": "floor_monitor",
           "user": "floor_monitor",
           "password": "your_password",
           "port": 5432
       },
       "cameras": [
           {
               "id": 0,
               "name": "Main Entrance"
           }
       ],
       "thresholds": {
           "warning_minutes": 15,
           "alert_minutes": 30
       }
   }
   ```

3. **Run the Application**

   ```bash
   python run.py
   ```

## User Interface

The application features a modern graphical interface with multiple tabs for different functions.

### Main Window

The main window consists of:

1. **Toolbar**: Quick access to main functions
2. **Tabbed Interface**: Different views for various functions
3. **Status Bar**: Current application status

### Cameras Tab

This tab displays live video feeds from all configured cameras:

- Real-time video display
- Camera status indicators
- Detection bounding boxes
- Person count overlay

### Worker Status Tab

This tab shows the current status of all detected workers:

- Worker ID and name
- Current status (Present/Absent/Exceeded)
- Time present
- Last seen timestamp

### Alerts Tab

This tab displays all active and historical alerts:

- Alert title and message
- Alert type (Info/Warning/Alert)
- Timestamp
- Acknowledgment options

### Reports Tab

This tab provides access to generated reports:

- Report preview
- Export options
- Report history

## Configuration

### Configuration File

The application is configured through `config/config.json`:

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

### Using the Settings Dialog

1. Click the "Settings" button in the toolbar
2. Modify database settings, thresholds, or notification preferences
3. Click "OK" to save changes

### Camera Configuration

To add or modify cameras:

1. Open the settings dialog
2. Update the "cameras" section in the configuration
3. Restart the application

## Monitoring Workers

### Starting Monitoring

1. Click the "Start" button in the toolbar
2. The application will initialize all cameras
3. Live video feeds will appear in the Cameras tab

### Worker Detection

- Workers are automatically detected in camera feeds
- Each detected person is tracked with a unique ID
- Presence time is calculated from first detection
- Status updates in real-time

### Status Indicators

- **Green**: Worker present and within time limits
- **Yellow**: Worker present, approaching time limit
- **Red**: Worker present, exceeded time limit

## Receiving Alerts

### Alert Types

1. **Warning**: Worker has been present for warning threshold time
2. **Alert**: Worker has been present for alert threshold time
3. **Info**: General information alerts

### Notification Methods

- **Desktop Notifications**: Pop-up notifications on Windows
- **Sound Alerts**: Audible alerts (configurable)
- **Visual Indicators**: Color-coded status in UI

### Managing Alerts

1. **View Alerts**: Switch to the Alerts tab
2. **Acknowledge**: Select an alert and click "Acknowledge Selected"
3. **Clear All**: Click "Clear Alerts" to remove all current alerts

## Generating Reports

### Daily Reports

1. Navigate to the Reports tab
2. Select "Daily Report"
3. Choose the date
4. Click "Generate"

### Weekly Reports

1. Navigate to the Reports tab
2. Select "Weekly Report"
3. Choose the start date
4. Click "Generate"

### Exporting Reports

Reports can be exported in two formats:

1. **CSV**: Comma-separated values for spreadsheet applications
2. **PDF**: Portable Document Format for printing and sharing

To export a report:
1. Generate the report
2. Click the appropriate export button
3. Choose the save location
4. The file will be saved to the selected location

## Troubleshooting

### Common Issues

#### Camera Not Detected

- Check USB connections
- Ensure no other applications are using the camera
- Try a different USB port
- Verify camera compatibility with OpenCV

#### Database Connection Failed

- Verify PostgreSQL is running
- Check database credentials in configuration
- Ensure the database exists
- Test connection with a PostgreSQL client

#### Alerts Not Working

- Check notification settings
- Verify Windows notification permissions
- Test sound settings
- Check alert thresholds in configuration

### Log Files

The application creates log files for troubleshooting:

- `app.log`: Main application log
- Check the log file for detailed error messages

## FAQ

### How accurate is the person detection?

The application uses YOLOv8, which provides high accuracy for person detection. Accuracy may vary based on camera quality, lighting conditions, and camera angle.

### Can I use IP cameras instead of USB cameras?

The current version supports USB cameras. IP camera support can be added by modifying the camera manager module.

### How many cameras can I use simultaneously?

The number of cameras depends on your hardware capabilities. Each camera requires CPU/GPU resources for processing.

### Is my data secure?

The application stores data locally in your PostgreSQL database. No data is transmitted to external servers.

### Can I customize alert thresholds?

Yes, alert thresholds can be configured in the `config.json` file or through the settings dialog.

### How do I add new workers to the system?

Workers are automatically detected and added to the system. For named workers, you can update the database directly or implement a worker registration feature.

### What happens if the application crashes?

The application automatically logs errors and attempts to recover. In case of a crash, restart the application and check the log files for details.

### Can I run the application on multiple computers?

Yes, you can run the application on multiple computers with different camera setups. Each instance will need its own database connection.

## Support

For additional support, please contact the development team at [support@floormonitoring.com](mailto:support@floormonitoring.com).

## Version Information

Current Version: 1.0.0
Release Date: [Release Date]