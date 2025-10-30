# Troubleshooting Guide

This guide provides solutions to common issues you may encounter when using the Floor Monitoring Application.

## Common Issues and Solutions

### 1. Camera Not Detected

**Problem**: The application cannot detect or access your USB camera.

**Solutions**:
1. Check that the camera is properly connected to your computer
2. Ensure no other application is using the camera
3. Verify camera permissions in your operating system settings
4. Try a different USB port
5. Test the camera with another application (e.g., Camera app on Windows)
6. Check if the camera is compatible with OpenCV

**Code-based solution**:
```python
# Test camera access
import cv2
cap = cv2.VideoCapture(0)  # Try different indices (0, 1, 2, etc.)
if not cap.isOpened():
    print("Cannot open camera")
else:
    print("Camera opened successfully")
    cap.release()
```

### 2. Database Connection Failed

**Problem**: The application cannot connect to the PostgreSQL database.

**Solutions**:
1. Verify that PostgreSQL is installed and running
2. Check database credentials in `config/config.json`
3. Ensure the database `floor_monitor` exists
4. Verify that PostgreSQL is accepting connections on the specified port (default 5432)
5. Check firewall settings
6. Test connection with a PostgreSQL client (e.g., pgAdmin)

**Code-based solution**:
```bash
# Test database connection
psql -h localhost -p 5432 -U postgres -d floor_monitor
```

### 3. YOLO Model Not Found

**Problem**: Error loading the YOLOv8 model.

**Solutions**:
1. Ensure you have internet connection (first run downloads the model)
2. Check that `ultralytics` package is installed: `pip install ultralytics`
3. Verify the model file exists or let the application download it automatically
4. Try manually downloading the model:
   ```bash
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   ```

### 4. Missing Dependencies

**Problem**: ImportError or ModuleNotFoundError when running the application.

**Solutions**:
1. Install all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. For development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
3. Check Python version (requires 3.10+)
4. Create a fresh virtual environment and reinstall dependencies

### 5. GUI Not Displaying Properly

**Problem**: The PyQt5 GUI appears blank or has rendering issues.

**Solutions**:
1. Update graphics drivers
2. Install PyQt5 correctly:
   ```bash
   pip install PyQt5
   ```
3. Check display settings and scaling
4. Try running with software rendering:
   ```bash
   QT_QUICK_BACKEND=software python run.py
   ```

### 6. Performance Issues

**Problem**: Application is slow or consuming too much CPU.

**Solutions**:
1. Reduce camera resolution in `config/config.json`
2. Lower the frame rate in camera settings
3. Use CPU instead of GPU for detection (if GPU is causing issues)
4. Reduce the number of cameras being monitored
5. Close other resource-intensive applications

### 7. Alert Notifications Not Working

**Problem**: Desktop notifications or sound alerts are not functioning.

**Solutions**:
1. Check notification settings in `config/config.json`
2. Verify that `win10toast` is installed:
   ```bash
   pip install win10toast
   ```
3. Check Windows notification settings
4. Ensure sound is enabled and not muted
5. Test with a simple notification script:
   ```python
   from win10toast import ToastNotifier
   toaster = ToastNotifier()
   toaster.show_toast("Test", "Notification test")
   ```

### 8. Configuration Errors

**Problem**: Application fails to start due to configuration issues.

**Solutions**:
1. Validate JSON syntax in `config/config.json`
2. Ensure all required fields are present
3. Check data types (e.g., port should be an integer)
4. Use the default configuration as a reference
5. Validate configuration programmatically:
   ```python
   import json
   with open('config/config.json', 'r') as f:
       config = json.load(f)
   print(config)
   ```

### 9. Report Generation Failures

**Problem**: Unable to generate or export reports.

**Solutions**:
1. Check that `pandas` and `matplotlib` are installed:
   ```bash
   pip install pandas matplotlib
   ```
2. Verify write permissions in the reports directory
3. Ensure sufficient disk space
4. Check that the database contains data to report on

### 10. Installation Issues

**Problem**: Errors during installation of dependencies.

**Solutions**:
1. Update pip:
   ```bash
   python -m pip install --upgrade pip
   ```
2. Install Microsoft Visual C++ Build Tools (Windows)
3. Use conda instead of pip for some packages:
   ```bash
   conda install opencv
   ```
4. Install packages one by one to identify problematic ones

## Debugging Steps

### Enable Debug Logging

1. Set `debug: true` in `config/config.json`
2. Check `app.log` file for detailed error messages
3. Run with verbose output:
   ```bash
   python run.py
   ```

### Test Individual Modules

1. Test camera module:
   ```bash
   python src/camera_manager.py
   ```

2. Test detection module:
   ```bash
   python src/detection_module.py
   ```

3. Test database module:
   ```bash
   python src/database_module.py
   ```

### Check System Requirements

Ensure your system meets the minimum requirements:
- Python 3.10 or higher
- At least 4GB RAM
- OpenGL 2.0 compatible graphics
- 2GB available disk space
- USB camera (1 or more)

## Getting Help

If you're still experiencing issues:

1. Check the [GitHub Issues](https://github.com/your-repo/issues) for similar problems
2. Create a new issue with:
   - Detailed description of the problem
   - Error messages
   - System information (OS, Python version, etc.)
   - Steps to reproduce
3. Include relevant log files
4. Provide your configuration (with sensitive data removed)

## Contact Support

For additional support, contact the development team at [support@floormonitoring.com](mailto:support@floormonitoring.com).