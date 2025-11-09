# 📦 Dependencies Guide

## Overview
Complete guide to project dependencies and version control configuration.

---

## 📋 Requirements Files

### `requirements.txt` - Production Dependencies

All packages needed to run the surveillance system:

#### Core Computer Vision & AI
- **opencv-python** (>=4.7.0) - Computer vision and image processing
- **ultralytics** (>=8.0.0) - YOLOv8 for person detection
- **deepface** (>=0.0.79) - Face recognition with ArcFace
- **tensorflow** (>=2.13.0) - Deep learning backend
- **scikit-learn** (>=1.3.0) - Machine learning utilities
- **numpy** (>=1.24.0) - Numerical computing
- **scipy** (>=1.10.0) - Scientific computing

#### Database & Configuration
- **psycopg2-binary** (>=2.9.5) - PostgreSQL adapter
- **python-dotenv** (>=1.0.0) - Environment variables

#### GUI & Interface
- **PyQt5** (>=5.15.9) - Desktop GUI framework

#### Reporting & Visualization
- **pandas** (>=1.5.3) - Data manipulation
- **matplotlib** (>=3.7.1) - Data visualization

#### Notifications & Alerts
- **win10toast** (>=0.9) - Windows notifications
- **playsound** (>=1.2.2) - Sound alerts

### `requirements-dev.txt` - Development Dependencies

Additional packages for development:

#### Testing
- **pytest** (>=7.2.0) - Testing framework
- **pytest-cov** (>=4.0.0) - Code coverage

#### Code Quality
- **black** (>=22.12.0) - Code formatter
- **flake8** (>=6.0.0) - Linting
- **isort** (>=5.12.0) - Import sorting
- **mypy** (>=1.0.0) - Type checking

#### Build & Documentation
- **pyinstaller** (>=5.13.0) - Create executables
- **sphinx** (>=6.0.0) - Documentation

---

## 🚀 Installation

### Fresh Installation
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Update Dependencies
```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade opencv-python
```

### Verify Installation
```bash
# List installed packages
pip list

# Check for issues
pip check

# Show package info
pip show deepface
```

---

## 🔒 .gitignore Configuration

### What's Ignored

#### Environment & Secrets
- `.env` - Environment variables with passwords
- `venv/` - Virtual environment
- `.venv/`, `env/`, `ENV/` - Alternative venv names

#### Application Data
- `*.log` - All log files
- `reports/` - Generated reports
- `temp_*.jpg` - Temporary images
- `*.sql.backup` - Database backups

#### AI Models
- `yolov8*.pt` - YOLO models (except yolov8n.pt)
- `.deepface/` - DeepFace model cache
- `*.h5`, `*.pb` - TensorFlow models

#### Build Artifacts
- `build/`, `dist/` - Build directories
- `*.exe`, `*.msi` - Executables
- `__pycache__/` - Python cache

#### IDE & OS
- `.vscode/`, `.idea/` - IDE settings
- `.DS_Store`, `Thumbs.db` - OS files

#### Training Data
- `training/dataset/` - Training datasets
- `training/models/` - Trained models
- `data/workers/*/` - Worker photos

### What's Tracked

#### Core Application
- `src/` - All source code
- `config/config.json` - Configuration (no passwords)
- `run.py` - Application launcher

#### Documentation
- `README.md` - Main docs
- `*.md` - All markdown files

#### Configuration
- `.env.example` - Environment template
- `.gitignore` - This file
- `requirements.txt` - Dependencies

#### Essential Models
- `yolov8n.pt` - Base YOLO model

---

## 📊 Dependency Tree

```
surveillance/
├── Core Vision
│   ├── opencv-python → numpy
│   ├── ultralytics → torch, opencv
│   └── deepface → tensorflow, opencv
│
├── Machine Learning
│   ├── tensorflow → numpy, scipy
│   ├── scikit-learn → numpy, scipy
│   └── numpy (base)
│
├── Database
│   └── psycopg2-binary
│
├── GUI
│   └── PyQt5
│
├── Reporting
│   ├── pandas → numpy
│   └── matplotlib → numpy
│
└── Utilities
    ├── python-dotenv
    ├── win10toast
    └── playsound
```

---

## 🔧 Common Issues

### TensorFlow Installation
```bash
# If TensorFlow fails, try:
pip install tensorflow-cpu>=2.13.0  # CPU-only version
```

### PyQt5 on Linux
```bash
# Install system dependencies first:
sudo apt-get install python3-pyqt5
```

### DeepFace Models
```bash
# Models auto-download on first use
# Cache location: ~/.deepface/weights/
```

### Windows Notifications
```bash
# win10toast only works on Windows 10+
# On other OS, notifications will be skipped
```

---

## 📈 Version Management

### Check Versions
```bash
# Show all versions
pip freeze

# Export current versions
pip freeze > requirements-lock.txt
```

### Update Strategy
1. **Minor updates** - Safe to update regularly
   ```bash
   pip install --upgrade opencv-python
   ```

2. **Major updates** - Test thoroughly
   ```bash
   # Test in separate environment first
   pip install tensorflow>=3.0.0
   ```

3. **Security updates** - Apply immediately
   ```bash
   pip install --upgrade package-name
   ```

---

## 🧪 Testing Dependencies

### Verify All Imports
```python
# test_imports.py
import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import psycopg2
from PyQt5.QtWidgets import QApplication
import pandas as pd
import matplotlib.pyplot as plt

print("✅ All imports successful!")
```

### Check Versions
```python
# check_versions.py
import cv2
import numpy as np
import tensorflow as tf

print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")
print(f"TensorFlow: {tf.__version__}")
```

---

## 🔄 Updating This Guide

When adding new dependencies:

1. **Add to requirements.txt**
   ```bash
   echo "new-package>=1.0.0  # Description" >> requirements.txt
   ```

2. **Update this guide**
   - Add to dependency list
   - Document purpose
   - Note any special installation steps

3. **Test installation**
   ```bash
   pip install -r requirements.txt
   python test_imports.py
   ```

4. **Update .gitignore if needed**
   - Add package-specific cache directories
   - Add generated files

---

## 📚 Additional Resources

### Package Documentation
- [OpenCV](https://docs.opencv.org/)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [DeepFace](https://github.com/serengil/deepface)
- [TensorFlow](https://www.tensorflow.org/api_docs)
- [PyQt5](https://www.riverbankcomputing.com/static/Docs/PyQt5/)

### Troubleshooting
- [pip Documentation](https://pip.pypa.io/)
- [Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [Requirements Files](https://pip.pypa.io/en/stable/reference/requirements-file-format/)

---

**Last Updated:** 2025-11-09  
**Python Version:** 3.10+  
**Status:** ✅ Current
