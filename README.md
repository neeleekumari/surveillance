# 🏢 Floor Monitoring Desktop Application

A comprehensive surveillance system for monitoring worker presence on factory floors using computer vision and AI-powered object detection.

## 📋 Overview

This application uses USB cameras and YOLOv8 object detection to monitor worker presence in real-time. It tracks workers' time on the floor, generates alerts for extended presence, and provides detailed reporting capabilities.

**Status: ✅ COMPLETE** - All planned features have been implemented and tested.

## 🌟 Key Features

- **Real-time Person Detection**: Uses YOLOv8 for accurate person detection in video feeds
- **Multi-Camera Support**: Monitor multiple areas simultaneously
- **Presence Tracking**: Track individual worker presence time
- **Alert System**: Configurable warnings and alerts for extended presence
- **Database Integration**: PostgreSQL backend for data storage
- **Rich GUI**: PyQt5-based interface with live camera feeds
- **Reporting**: Generate daily/weekly reports in CSV/PDF formats
- **Notifications**: Desktop notifications and sound alerts

## 🏗️ Architecture

```
Floor Monitoring App
├── Camera Manager (camera_manager.py)
├── Person Detection (detection_module.py)
├── Presence Tracking (presence_tracker.py)
├── Database (database_module.py)
├── Alert System (alert_manager.py)
├── Configuration (config_manager.py)
├── Reporting (report_generator.py)
└── User Interface (ui_manager.py)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- USB cameras (1 or more)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd floor_monitoring_app
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up PostgreSQL database:
   ```sql
   CREATE DATABASE floor_monitor;
   -- Create tables using the SQL in TODO.md
   ```

5. Configure the application:
   Edit `config/config.json` with your database settings and preferences

### Usage

Run the application:
```bash
python run.py
```

Or try the demo without hardware requirements:
```bash
python demo.py
```

## 📁 Project Structure

```
floor_monitoring_app/
├── src/                    # Source code
│   ├── __init__.py
│   ├── main.py            # Application entry point
│   ├── camera_manager.py  # Camera handling
│   ├── detection_module.py # YOLOv8 person detection
│   ├── presence_tracker.py # Worker presence tracking
│   ├── database_module.py # PostgreSQL interface
│   ├── alert_manager.py   # Notification system
│   ├── ui_manager.py      # PyQt5 GUI
│   ├── report_generator.py # Reporting module
│   └── config_manager.py  # Configuration handling
├── config/                # Configuration files
│   └── config.json
├── assets/                # Images, sounds, etc.
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
├── requirements-dev.txt   # Development dependencies
├── setup.py              # Package setup
├── run.py                # Application runner
├── build.py              # Build script
├── demo.py               # Demo script
├── README.md             # This file
├── TODO.md               # Development roadmap
├── SUMMARY.md            # Project summary
└── verify_project.py     # Project verification
```

## ⚙️ Configuration

The application is configured through `config/config.json`:

```json
{
    "database": {
        "host": "localhost",
        "name": "floor_monitor",
        "user": "postgres",
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
    },
    "notifications": {
        "enabled": true,
        "sound": true
    }
}
```

## 🧪 Testing

Run all tests:
```bash
python -m pytest tests/
```

Run individual module tests:
```bash
python tests/test_camera.py
python tests/test_detection.py
python tests/test_database.py
python tests/test_integration.py
```

## 📊 Reporting

Generate reports through the GUI or programmatically:
- Daily presence reports
- Weekly summaries
- Export to CSV/PDF formats

## 🛠️ Development

### Adding New Features

1. Follow the existing module structure
2. Add unit tests for new functionality
3. Update TODO.md with progress
4. Document new features in README.md

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📚 Documentation

- [TODO.md](TODO.md) - Development roadmap and tasks
- [SUMMARY.md](SUMMARY.md) - Project summary
- [docs/user_manual.md](docs/user_manual.md) - User guide
- [docs/api.md](docs/api.md) - API documentation
- [docs/troubleshooting.md](docs/troubleshooting.md) - Troubleshooting guide
- [docs/deployment.md](docs/deployment.md) - Deployment guide

## 🎉 Project Status

This project is **COMPLETE** with all planned features implemented:

✅ Real-time person detection with YOLOv8  
✅ Multi-camera support  
✅ Worker presence tracking  
✅ Alert system with notifications  
✅ Database integration with PostgreSQL  
✅ Rich PyQt5 GUI  
✅ Reporting capabilities  
✅ Comprehensive testing  
✅ Full documentation  

## 🚀 Future Enhancements

While the core application is complete, several enhancements could be considered:

- Face recognition for worker identification
- Mobile app integration
- Cloud synchronization
- Advanced analytics and trend analysis
- Multi-language support

## 🤝 Support

For issues and feature requests, please create a GitHub issue.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [OpenCV](https://opencv.org/) for computer vision
- [PyQt5](https://pypi.org/project/PyQt5/) for the GUI framework