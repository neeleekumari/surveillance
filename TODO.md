# 🏢 Floor Monitoring Desktop App — Project To-Do List

## 🛠️ 1. Setup Phase
- [x] Create project directory structure
- [x] Set up Python 3.10+ virtual environment
- [x] Create `requirements.txt` with dependencies:
  ```
  opencv-python>=4.7.0
  ultralytics>=8.0.0
  psycopg2-binary>=2.9.5
  PyQt5>=5.15.9
  pandas>=1.5.3
  matplotlib>=3.7.1
  win10toast>=0.9
  pyinstaller>=5.13.0
  python-dotenv>=1.0.0
  ```
- [x] Initialize Git repository with `.gitignore`
- [x] Create basic `README.md` with project overview

## 🗄️ 2. Database Setup (PostgreSQL)
- [x] Install PostgreSQL 14+ and pgAdmin
- [x] Create database `floor_monitor`
- [x] Create tables:
  ```sql
  CREATE TABLE workers (
      worker_id SERIAL PRIMARY KEY,
      name VARCHAR(100) NOT NULL,
      position VARCHAR(100),
      contact VARCHAR(100),
      is_active BOOLEAN DEFAULT true,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE activity_log (
      log_id SERIAL PRIMARY KEY,
      worker_id INTEGER REFERENCES workers(worker_id),
      status VARCHAR(20) NOT NULL, -- 'present', 'absent', 'exceeded'
      timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      duration_seconds INTEGER
  );
  ```
- [x] Create `database_module.py` with CRUD operations
- [x] Add test data and verify database connection

## 📷 3. Camera & Detection Module
- [x] Create `camera_manager.py`:
  - USB camera initialization
  - Frame capture and preprocessing
  - Multi-camera support
- [x] Create `detection_module.py`:
  - Load YOLOv8n model
  - Person detection and tracking
  - ROI definition and management
- [x] Test with multiple USB cameras
- [x] Implement frame rate optimization

## ⏱️ 4. Presence Tracking
- [x] Create `presence_tracker.py`:
  - Worker presence detection
  - Time tracking per worker
  - Threshold monitoring
  - State management (present/absent/exceeded)
- [x] Implement configurable threshold settings
- [x] Add logging for presence events

## 🚨 5. Alert System
- [x] Create `alert_manager.py`:
  - Desktop notifications
  - Sound alerts
  - Visual indicators
- [x] Implement escalation rules
- [x] Add alert history and acknowledgment

## 🖥️ 6. GUI Development (PyQt5)
- [x] Design main window in `ui_manager.py`:
  - Live camera feed display
  - Worker status panel
  - Alert notifications
  - Settings panel
- [x] Implement dark/light theme
- [x] Add real-time status indicators
- [x] Create dashboard with metrics

## 📊 7. Reporting Module
- [x] Create `report_generator.py`:
  - Daily/weekly reports
  - Worker activity summaries
  - Export to CSV/PDF
- [x] Add data visualization with Matplotlib
- [x] Implement report scheduling

## ⚙️ 8. Configuration Management
- [x] Create `config_manager.py` with:
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
      }
  }
  ```
- [x] Add configuration validation
- [x] Implement GUI for settings management

## 🔄 9. Integration
- [x] Create `main.py` as entry point
- [x] Implement module communication
- [x] Add error handling and logging
- [x] Create system tray integration

## 🧪 10. Testing
- [x] Unit tests for all modules
- [x] Integration testing
- [x] Performance testing
- [x] User acceptance testing

## 📦 11. Packaging & Deployment
- [x] Create `setup.py`
- [x] Build executable with PyInstaller:
  ```bash
  pyinstaller --onefile --windowed --icon=assets/icon.ico main.py
  ```
- [x] Create installer (NSIS/Inno Setup)
- [x] Prepare deployment package with documentation

## 📚 12. Documentation
- [x] User manual
- [x] API documentation
- [x] Troubleshooting guide
- [x] Deployment guide

## 🚀 13. Future Enhancements
- [ ] Face recognition
- [ ] Mobile app integration
- [ ] Cloud synchronization
- [ ] Advanced analytics
- [ ] Multi-language support

## 📁 Project Structure
```
floor_monitoring_app/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── camera_manager.py
│   ├── detection_module.py
│   ├── presence_tracker.py
│   ├── database_module.py
│   ├── alert_manager.py
│   ├── ui_manager.py
│   ├── report_generator.py
│   └── config_manager.py
├── config/
│   └── config.json
├── assets/
│   ├── icons/
│   └── models/
├── tests/
│   ├── test_integration.py
│   ├── test_camera.py
│   ├── test_detection.py
│   └── test_database.py
├── docs/
│   ├── api.md
│   ├── troubleshooting.md
│   ├── deployment.md
│   └── user_manual.md
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── run.py
├── build.py
├── .gitignore
├── README.md
└── TODO.md
```

## ✅ Getting Started
1. Clone the repository
2. Set up Python 3.10+ environment
3. Install dependencies: `pip install -r requirements.txt`
4. Configure `config/config.json`
5. Run: `python run.py`

## 🤝 Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request