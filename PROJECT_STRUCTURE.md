# 📁 Project Structure

## Overview
Clean, organized surveillance system with only essential files.

---

## 🗂️ Directory Structure

```
surveillance/
├── 📂 src/                          # Core application modules
│   ├── __init__.py
│   ├── main.py                      # Main application entry point
│   ├── camera_manager.py            # Camera handling and frame capture
│   ├── detection_module.py          # YOLOv8 person detection
│   ├── face_recognition_module.py   # Face recognition with ArcFace
│   ├── database_module.py           # PostgreSQL database interface
│   ├── presence_tracker.py          # Worker presence tracking
│   ├── alert_manager.py             # Alert and notification system
│   ├── ui_manager.py                # Main PyQt5 GUI
│   ├── worker_registration_ui.py    # Worker registration interface
│   ├── report_generator.py          # Report generation (CSV/PDF)
│   └── config_manager.py            # Configuration management
│
├── 📂 config/                       # Configuration files
│   └── config.json                  # App settings (no passwords)
│
├── 📂 assets/                       # Static assets
│   └── (icons, sounds, etc.)
│
├── 📂 training/                     # Model training scripts
│   └── (training utilities)
│
├── 📂 models/                       # Trained models (empty initially)
│
├── 📂 data/                         # Data storage (empty initially)
│
├── 📂 venv/                         # Virtual environment (gitignored)
│
├── 🔧 Utility Scripts
│   ├── run.py                       # Application launcher
│   ├── clear_database.py            # Database cleanup utility
│   ├── sync_database.py             # Database sync utility
│   ├── migrate_to_database.py       # Migration tool
│   ├── fix_embeddings.py            # Fix face embeddings
│   ├── clean_unregistered.py        # Clean unregistered workers
│   ├── check_cameras.py             # Camera detection utility
│   ├── verify_after_registration.py # Verify worker registration
│   └── test_env_migration.py        # Test environment setup
│
├── 📚 Documentation
│   ├── README.md                    # Main project documentation
│   ├── ENV_SETUP.md                 # Environment setup guide
│   ├── README_ENV.md                # Quick env reference
│   ├── PASSWORDS_REFERENCE.md       # Password management guide
│   └── PROJECT_STRUCTURE.md         # This file
│
├── ⚙️ Configuration
│   ├── .env                         # Environment variables (gitignored)
│   ├── .env.example                 # Environment template
│   ├── .gitignore                   # Git ignore rules
│   ├── requirements.txt             # Python dependencies
│   └── requirements-dev.txt         # Development dependencies
│
└── 🤖 Models
    └── yolov8n.pt                   # YOLOv8 model weights
```

---

## 📦 Core Modules

### Application Core
- **`main.py`** - Application entry point with detection thread
- **`ui_manager.py`** - Main GUI with live camera feeds
- **`worker_registration_ui.py`** - Worker registration interface

### Detection & Recognition
- **`detection_module.py`** - YOLOv8 person detection
- **`face_recognition_module.py`** - Face recognition with anti-false-positive system

### Data Management
- **`database_module.py`** - PostgreSQL interface
- **`presence_tracker.py`** - Track worker presence/absence
- **`report_generator.py`** - Generate reports

### System Components
- **`camera_manager.py`** - Multi-camera support
- **`alert_manager.py`** - Notifications and alerts
- **`config_manager.py`** - Configuration with env variables

---

## 🔧 Utility Scripts

### Database Tools
- **`clear_database.py`** - Reset database (keeps structure)
- **`sync_database.py`** - Sync worker data
- **`migrate_to_database.py`** - Migration utilities
- **`fix_embeddings.py`** - Fix face embedding issues

### Maintenance
- **`clean_unregistered.py`** - Remove unregistered workers
- **`check_cameras.py`** - Test camera connections
- **`verify_after_registration.py`** - Verify worker setup
- **`test_env_migration.py`** - Test environment configuration

---

## 📚 Documentation

### User Documentation
- **`README.md`** - Complete project overview
- **`ENV_SETUP.md`** - Detailed environment setup
- **`README_ENV.md`** - Quick environment reference
- **`PASSWORDS_REFERENCE.md`** - Password management

### Developer Documentation
- **`PROJECT_STRUCTURE.md`** - This file
- Code comments in all modules

---

## 🔐 Configuration Files

### Environment Variables (`.env`)
```bash
DB_HOST=localhost
DB_NAME=floor_monitor
DB_USER=postgres
DB_PASSWORD=your_password
DB_PORT=5432
```

### Application Config (`config/config.json`)
```json
{
    "database": { ... },
    "cameras": [ ... ],
    "thresholds": { ... },
    "notifications": { ... }
}
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Copy environment template
cp .env.example .env

# Edit with your passwords
notepad .env
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
python run.py
```

---

## 🧹 Cleanup History

**Date:** 2025-11-09  
**Removed:** 50 unnecessary files
- ✅ 6 empty Python files
- ✅ 12 debug scripts
- ✅ 7 old test scripts
- ✅ 25 redundant documentation files
- ✅ Log files and temp files

**Result:** Clean, maintainable codebase with only essential files.

---

## 📊 File Count Summary

| Category | Count | Description |
|----------|-------|-------------|
| Core Modules | 12 | Essential application code |
| Utility Scripts | 9 | Maintenance and testing tools |
| Documentation | 5 | User and developer guides |
| Configuration | 5 | Settings and environment |
| Assets | 1 | YOLOv8 model |
| **Total** | **32** | **Clean, organized structure** |

---

## 🎯 Best Practices

### File Organization
- ✅ All core code in `src/`
- ✅ Configuration separate from code
- ✅ Utilities in root for easy access
- ✅ Documentation clearly organized

### Security
- ✅ Passwords in `.env` (gitignored)
- ✅ No sensitive data in version control
- ✅ Environment variables for all secrets

### Maintainability
- ✅ Clear module separation
- ✅ Descriptive file names
- ✅ Comprehensive documentation
- ✅ Utility scripts for common tasks

---

## 🔄 Future Additions

When adding new files, follow these guidelines:

### New Modules
- Place in `src/` directory
- Add to imports in `__init__.py`
- Document in this file

### New Utilities
- Place in root directory
- Add description to this file
- Include usage instructions

### New Documentation
- Use descriptive names
- Link from README.md
- Keep concise and relevant

---

**Last Updated:** 2025-11-09  
**Status:** ✅ Clean and Organized
