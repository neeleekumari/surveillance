# Floor Monitoring System - Complete Documentation

## Project Overview

**Purpose:** Real-time worker monitoring system using face recognition and person detection across multiple cameras.

**Key Features:**
- Multi-camera surveillance (up to 3 cameras)
- Face recognition using ArcFace (512-D embeddings)
- Person detection using YOLOv8
- Worker presence tracking
- Attendance logging
- Alert system for worker absence
- PostgreSQL database storage

---

## Technology Stack

### 1. Programming Language
- **Python 3.13**

### 2. Deep Learning Frameworks

#### Computer Vision
- **OpenCV (cv2)** - Camera capture, image processing
- **PyTorch 2.9.0+cpu** - Deep learning backend
- **YOLOv8** - Person detection (via ultralytics)
- **DeepFace** - Face recognition framework
- **TensorFlow/Keras** - Backend for DeepFace models

#### Face Recognition Model
- **ArcFace** - Face embedding model
  - Embedding size: 512 dimensions
  - Accuracy: 99.82%
  - Optimized for distinguishing similar faces
  - Distance metric: Cosine similarity

### 3. Database
- **PostgreSQL** - Primary data storage
- **psycopg2** - PostgreSQL adapter for Python

#### Database Schema
```sql
-- Workers table
CREATE TABLE workers (
    worker_id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    position VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Face embeddings table
CREATE TABLE face_embeddings (
    id SERIAL PRIMARY KEY,
    worker_id INTEGER REFERENCES workers(worker_id),
    embedding BYTEA NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Face photos table
CREATE TABLE face_photos (
    id SERIAL PRIMARY KEY,
    worker_id INTEGER REFERENCES workers(worker_id),
    photo_data BYTEA NOT NULL,
    photo_format VARCHAR(10) DEFAULT 'jpg',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Attendance records table
CREATE TABLE attendance_records (
    id SERIAL PRIMARY KEY,
    worker_id INTEGER REFERENCES workers(worker_id),
    camera_id INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    snapshot BYTEA,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Activity log table
CREATE TABLE activity_log (
    id SERIAL PRIMARY KEY,
    worker_id INTEGER,
    activity_type VARCHAR(50),
    description TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4. GUI Framework
- **PyQt5** - Desktop application UI
  - QtWidgets - UI components
  - QtCore - Core functionality
  - QtGui - Graphics and images

### 5. Scientific Computing
- **NumPy** - Array operations, numerical computing
- **scikit-learn** - Cosine similarity calculations
- **SciPy** - Euclidean distance calculations

### 6. Utilities
- **Pillow (PIL)** - Image processing
- **pathlib** - File path operations
- **json** - Configuration file handling
- **logging** - Application logging
- **datetime** - Timestamp handling
- **dataclasses** - Data structure definitions

### 7. Windows Integration
- **win10toast** - Windows 10 notifications

---

## Project Structure

```
surveillance/
├── config/
│   └── config.json              # System configuration
├── data/
│   └── registered_faces/        # Local photo backups
│       ├── worker_99/           # Photos for worker 99
│       └── worker_100/          # Photos for worker 100
├── src/
│   ├── alert_manager.py         # Alert and notification system
│   ├── camera_manager.py        # Multi-camera management
│   ├── config_manager.py        # Configuration loader
│   ├── database_module.py       # PostgreSQL database interface
│   ├── detection_module.py      # YOLOv8 person detection
│   ├── face_recognition_module.py  # ArcFace face recognition
│   ├── main.py                  # Main application
│   ├── presence_tracker.py      # Worker presence tracking
│   ├── ui_manager.py            # Main UI window
│   └── worker_registration_ui.py   # Worker registration dialog
├── run.py                       # Application entry point
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## Core Modules

### 1. **main.py** - Application Controller
**Purpose:** Main application orchestrator

**Key Components:**
- Initializes all subsystems
- Manages application lifecycle
- Coordinates between modules
- Handles main event loop

**Key Classes:**
- `FloorMonitoringApp` - Main application class

### 2. **camera_manager.py** - Camera Management
**Purpose:** Multi-camera capture and management

**Features:**
- Supports multiple cameras simultaneously
- Thread-based capture for each camera
- Frame buffering and synchronization
- Camera configuration (resolution, FPS, ROI)

**Key Classes:**
- `CameraManager` - Manages all cameras
- `CameraConfig` - Camera configuration dataclass

**Camera Configuration:**
```json
{
    "id": 0,
    "name": "Camera 0",
    "width": 640,
    "height": 480,
    "fps": 30,
    "rois": []
}
```

### 3. **detection_module.py** - Person Detection
**Purpose:** Detect people in camera frames using YOLOv8

**Model:** YOLOv8n (nano) - Fast, lightweight
**Detection Class:** Person (class 0 in COCO dataset)

**Key Classes:**
- `PersonDetector` - YOLOv8 wrapper
- `Detection` - Detection result dataclass

**Configuration:**
- Confidence threshold: 0.45
- IOU threshold: 0.4
- Device: CPU

### 4. **face_recognition_module.py** - Face Recognition
**Purpose:** Face recognition using ArcFace embeddings

**Model:** ArcFace
- Embedding size: 512-D
- Similarity metric: Cosine similarity
- Threshold: 0.50 (50% minimum similarity)

**Key Classes:**
- `FaceRecognitionSystem` - Main recognition system
- `FaceEmbedding` - Embedding dataclass

**Workflow:**
1. Extract face from detection bbox
2. Generate 512-D embedding using ArcFace
3. Normalize embedding (L2 norm = 1.0)
4. Compare with stored embeddings
5. Return best match if similarity > threshold

### 5. **database_module.py** - Database Interface
**Purpose:** PostgreSQL database operations

**Key Classes:**
- `DatabaseManager` - Database connection and operations

**Operations:**
- Worker CRUD operations
- Embedding storage/retrieval
- Photo storage (BYTEA)
- Attendance logging
- Activity tracking

### 6. **presence_tracker.py** - Presence Tracking
**Purpose:** Track worker presence and absence

**Features:**
- Track when workers enter/leave
- Calculate presence duration
- Generate alerts for absence
- Maintain worker state

**Thresholds:**
- Warning: 15 minutes absence
- Alert: 30 minutes absence

### 7. **alert_manager.py** - Alert System
**Purpose:** Manage notifications and alerts

**Features:**
- Windows 10 toast notifications
- Sound alerts
- Alert history
- Configurable alert types

### 8. **ui_manager.py** - Main UI
**Purpose:** Main application window

**Components:**
- Camera feed display (multiple cameras)
- Worker status panel
- Alert notifications
- Settings dialog
- Menu bar

### 9. **worker_registration_ui.py** - Registration Dialog
**Purpose:** Register new workers

**Features:**
- Live camera preview
- Face capture (10 photos)
- Photo upload option
- Worker management (view/delete)
- Camera selector dropdown

**Registration Process:**
1. Select camera
2. Enter worker details (ID, name, position)
3. Capture 10 face photos
4. Generate embedding from first photo
5. Save to database (worker, embedding, photos)

### 10. **config_manager.py** - Configuration
**Purpose:** Load and validate configuration

**Configuration File:** `config/config.json`

**Sections:**
- Database connection
- Camera configurations
- Thresholds (warning/alert times)
- Notifications settings
- App settings

---

## Data Flow

### 1. **Camera Capture**
```
Camera → CameraManager → Frame Buffer → Detection Module
```

### 2. **Person Detection**
```
Frame → YOLOv8 → Bounding Boxes → Face Extraction
```

### 3. **Face Recognition**
```
Face Image → ArcFace → 512-D Embedding → Similarity Comparison → Worker ID
```

### 4. **Database Storage**
```
Worker Data → PostgreSQL → Tables (workers, embeddings, photos, attendance)
```

### 5. **UI Update**
```
Recognition Result → UI Manager → Camera Widget → Display
```

---

## Key Algorithms

### 1. **Face Recognition Pipeline**
```python
1. Detect person (YOLOv8)
2. Extract face region from bbox
3. Resize face to model input size
4. Generate embedding using ArcFace
5. Normalize embedding (L2 norm)
6. Compare with database embeddings (cosine similarity)
7. Return match if similarity > 0.50
```

### 2. **Embedding Comparison**
```python
# Cosine similarity
similarity = dot(emb1_normalized, emb2_normalized)

# Decision
if similarity > 0.50:
    return worker_id
else:
    return "Unknown"
```

### 3. **Track Locking (Stable Recognition)**
```python
# Lock criteria
avg_score > 0.95
min_score > 0.90
std_dev < 0.05
consecutive_frames >= 3
```

---

## Configuration

### Database Configuration
```json
{
    "database": {
        "host": "localhost",
        "name": "floor_monitor",
        "user": "postgres",
        "password": "psql",
        "port": 5432
    }
}
```

### Camera Configuration
```json
{
    "cameras": [
        {
            "id": 0,
            "name": "Camera 0",
            "width": 640,
            "height": 480,
            "fps": 30,
            "rois": []
        },
        {
            "id": 1,
            "name": "Camera 1",
            "width": 640,
            "height": 480,
            "fps": 30,
            "rois": []
        },
        {
            "id": 2,
            "name": "Camera 2",
            "width": 1024,
            "height": 768,
            "fps": 30,
            "rois": []
        }
    ]
}
```

### Threshold Configuration
```json
{
    "thresholds": {
        "warning_minutes": 15,
        "alert_minutes": 30
    }
}
```

---

## Dependencies

### Core Dependencies
```
opencv-python>=4.8.0
numpy>=1.24.0
PyQt5>=5.15.0
psycopg2-binary>=2.9.0
```

### Deep Learning
```
torch>=2.0.0
ultralytics>=8.0.0
deepface>=0.0.79
tensorflow>=2.13.0
tf-keras>=2.13.0
```

### Utilities
```
scikit-learn>=1.3.0
scipy>=1.11.0
Pillow>=10.0.0
win10toast>=0.9
```

---

## Performance Characteristics

### Processing Speed
- Person detection: ~30-50ms per frame (CPU)
- Face recognition: ~100-200ms per face (CPU)
- Database query: ~5-10ms
- Overall FPS: 10-15 FPS per camera

### Memory Usage
- Base application: ~500MB
- Per camera: ~100MB
- Face recognition model: ~200MB
- Total (3 cameras): ~1.2GB

### Storage
- Face embedding: 2KB (512 floats)
- Face photo: 50-100KB (JPEG)
- Per worker: ~1MB (1 embedding + 10 photos)

---

## Security Considerations

### Database
- Password stored in config (should use environment variables)
- No encryption on embeddings
- No user authentication

### Privacy
- Face photos stored in database
- No data anonymization
- No GDPR compliance features

### Recommendations
1. Use environment variables for credentials
2. Encrypt sensitive data
3. Implement user authentication
4. Add audit logging
5. Implement data retention policies

---

## Limitations

### Current Limitations
1. **Only 2 cameras working** (Camera 2 initialization fails)
2. **Embeddings still 94% similar** (averaging bug not fully fixed)
3. **No GPU acceleration** (CPU only)
4. **No real-time optimization** (10-15 FPS)
5. **No multi-worker tracking** (one worker per track)

### Known Issues
1. ID shuffling when embeddings are too similar
2. False positives with low threshold
3. Camera 2 requires different resolution (1024x768)
4. No automatic camera detection

---

## Future Improvements

### High Priority
1. Fix embedding similarity issue (< 10% for different people)
2. Enable all 3 cameras
3. GPU acceleration support
4. Real-time performance optimization

### Medium Priority
1. Multi-face tracking per frame
2. Re-identification across cameras
3. Activity recognition
4. Automated reporting

### Low Priority
1. Web interface
2. Mobile app
3. Cloud storage
4. Advanced analytics

---

## Usage

### Starting the Application
```bash
python run.py
```

### Registering a Worker
1. Click "Register Worker"
2. Select camera from dropdown
3. Enter worker details
4. Capture 10 face photos
5. Click "Register Worker"

### Monitoring
1. Start application
2. Cameras auto-start
3. Workers detected automatically
4. Attendance logged to database

### Checking Database
```sql
-- View workers
SELECT * FROM workers;

-- View embeddings
SELECT worker_id, created_at FROM face_embeddings;

-- View attendance
SELECT w.name, a.timestamp 
FROM attendance_records a 
JOIN workers w ON a.worker_id = w.worker_id 
ORDER BY a.timestamp DESC;
```

---

## Troubleshooting

### Camera Not Detected
```bash
python test_all_cameras.py
```
This will show which cameras are available.

### Embeddings Too Similar
```bash
python verify_after_registration.py
```
This will check similarity between registered workers.

### Database Connection Failed
1. Check PostgreSQL is running
2. Verify credentials in config.json
3. Check database exists: `floor_monitor`

---

## Summary

**This is a comprehensive floor monitoring system that:**
- Uses YOLOv8 for person detection
- Uses ArcFace for face recognition
- Stores data in PostgreSQL
- Supports multiple cameras
- Provides real-time monitoring
- Logs attendance automatically
- Sends alerts for worker absence

**Built with:** Python, PyTorch, OpenCV, PyQt5, PostgreSQL
