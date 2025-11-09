# Environment Configuration Guide

## Overview
This project uses environment variables to manage sensitive configuration like database passwords and API keys.

## Setup Instructions

### 1. Environment File
The `.env` file contains all sensitive configuration. This file is already created with current values from `config/config.json`.

**Important:** The `.env` file is gitignored to prevent committing sensitive data to version control.

### 2. Configuration Variables

#### Database Configuration
- `DB_HOST`: PostgreSQL database host (default: localhost)
- `DB_NAME`: Database name (default: floor_monitor)
- `DB_USER`: Database username (default: postgres)
- `DB_PASSWORD`: Database password (current: 123456)
- `DB_PORT`: Database port (default: 5432)

#### Kaggle API (Optional - for training dataset downloads)
To download training datasets from Kaggle:
1. Go to https://www.kaggle.com/settings/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. Extract username and key from downloaded `kaggle.json`
5. Update `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env`

#### Application Settings
- `APP_VERSION`: Application version
- `APP_DEBUG`: Enable debug mode (true/false)

#### Camera Settings
- `CAMERA_WIDTH`: Camera resolution width (default: 640)
- `CAMERA_HEIGHT`: Camera resolution height (default: 480)
- `CAMERA_FPS`: Camera frames per second (default: 30)

#### Alert Thresholds
- `WARNING_MINUTES`: Minutes before warning alert (default: 15)
- `ALERT_MINUTES`: Minutes before critical alert (default: 30)

#### Notification Settings
- `NOTIFICATIONS_ENABLED`: Enable notifications (true/false)
- `NOTIFICATIONS_SOUND`: Enable sound alerts (true/false)

#### Face Recognition Settings
- `FACE_RECOGNITION_MODEL`: Model to use (default: ArcFace)
- `MIN_FACE_QUALITY`: Minimum face quality score (default: 35.0)
- `RECOGNITION_CONFIDENCE_THRESHOLD`: Minimum confidence for recognition (default: 0.70)
- `STABILITY_CONFIRMATIONS_NEEDED`: Confirmations needed before locking identity (default: 3)

#### Tracking Parameters
- `IOU_THRESHOLD`: Intersection over Union threshold (default: 0.20)
- `MAX_CENTROID_DISTANCE`: Maximum centroid distance for tracking (default: 150)
- `MIN_CONFIDENCE_TO_LOCK`: Minimum confidence to lock identity (default: 0.65)
- `TRACK_MEMORY_DURATION`: How long to remember tracks in seconds (default: 15.0)
- `MAX_TRACK_AGE`: Maximum track age in seconds (default: 2.0)

### 3. Using Environment Variables in Code

To use these environment variables in your Python code, install python-dotenv:

```bash
pip install python-dotenv
```

Then load the variables:

```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access variables
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST', 'localhost')  # with default value
```

### 4. Security Best Practices

1. **Never commit `.env` to version control** - It's already in `.gitignore`
2. **Use strong passwords** - Change the default database password
3. **Rotate credentials regularly** - Update passwords periodically
4. **Use `.env.example` for documentation** - Share structure, not values
5. **Restrict file permissions** - On Unix systems: `chmod 600 .env`

### 5. Deployment

For production deployment:
1. Copy `.env.example` to `.env` on the server
2. Update all placeholder values with production credentials
3. Use strong, unique passwords for production
4. Consider using a secrets management service (AWS Secrets Manager, Azure Key Vault, etc.)

### 6. Current Configuration

The `.env` file has been populated with current values from `config/config.json`:
- Database password: `123456` (⚠️ **Change this for production!**)
- All other settings match your current configuration

### 7. Migration Complete ✅

The system has been updated to use environment variables for sensitive data:
1. ✅ `config_manager.py` now reads passwords from `.env` file
2. ✅ Database password removed from `config.json`
3. ✅ `.env` is the primary source for sensitive credentials
4. ✅ `config.json` contains only non-sensitive settings (cameras, thresholds, etc.)

**How it works:**
- `ConfigManager.get_database_config()` automatically loads password from `.env`
- Environment variables take priority over `config.json` values
- If `.env` is missing, it falls back to `config.json` (with warning)

## Troubleshooting

### .env file not loading
- Ensure the file is named exactly `.env` (not `.env.txt`)
- Check file is in the project root directory
- Verify `python-dotenv` is installed

### Permission denied errors
- On Unix/Linux: `chmod 600 .env`
- On Windows: Right-click → Properties → Security → Edit permissions

### Variables not updating
- Restart your application after changing `.env`
- Some IDEs may cache environment variables

## Support

For issues or questions, refer to the project documentation or contact the development team.
