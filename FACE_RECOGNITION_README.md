# Face Recognition Integration Guide

## Overview

This application now integrates **YOLOv8** for person detection with **DeepFace (Facenet512)** for face recognition to provide a complete worker attendance tracking system.

## Architecture

### 1. **Detection Pipeline**
```
Camera Feed → YOLOv8 Detection → Face Cropping → DeepFace Embedding → Recognition → Attendance
```

### 2. **Key Components**

#### **face_recognition_module.py**
- `FaceRecognitionSystem`: Main class for face recognition
- `get_face_embedding()`: Generate 512-dimensional embeddings using Facenet512
- `register_worker()`: Register new workers with 3-5 face images
- `recognize_worker()`: Identify workers from face images
- `mark_attendance()`: Record attendance with timestamp and photo
- `crop_face_from_detection()`: Helper to crop faces from YOLO detections

#### **worker_registration_ui.py**
- PyQt5 dialog for registering new workers
- Live camera feed for face capture
- Captures 5 face images from different angles
- Validates single-person detection

#### **detection_module.py** (Enhanced)
- Added `worker_id`, `worker_name`, `recognition_score` to Detection dataclass
- Updated visualization to show recognized worker names in yellow

#### **main.py** (Enhanced)
- Integrated `FaceRecognitionSystem` into main application
- Modified `DetectionThread` to perform real-time face recognition
- Added worker registration dialog integration

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

New dependencies added:
- `deepface>=0.0.79` - Face recognition framework
- `tensorflow>=2.13.0` - Backend for DeepFace
- `scikit-learn>=1.3.0` - For similarity calculations

### 2. Create Data Directories
The application will automatically create:
- `data/face_embeddings.pkl` - Stores worker face embeddings
- `data/attendance_records.json` - Stores attendance logs
- `data/registered_faces/` - Stores registration photos
- `data/attendance_photos/` - Stores attendance photos

## Usage

### 1. **Register a New Worker**

1. Click **"Register Worker"** button in the toolbar
2. Enter worker information:
   - Worker ID (auto-generated or manual)
   - Worker Name (required)
   - Position (optional)
3. Position face in camera frame
4. Click **"Capture Face"** 5 times from different angles:
   - Front view
   - Slight left turn
   - Slight right turn
   - Looking up slightly
   - Looking down slightly
5. Click **"Register Worker"** to complete

**Best Practices:**
- Ensure good lighting
- Only one person in frame
- Capture from multiple angles
- Keep face clearly visible

### 2. **Real-Time Recognition**

Once monitoring is started:
1. Click **"Start"** to begin camera monitoring
2. YOLOv8 detects persons in frame
3. Faces are automatically cropped and recognized
4. Recognized workers show:
   - Yellow bounding box
   - Worker name
   - Recognition score (0-1)

### 3. **Attendance Tracking**

Attendance is automatically marked when:
- Worker is recognized (similarity > 0.6 threshold)
- Photo is saved with timestamp
- Record includes:
  - Worker ID and name
  - Timestamp
  - Photo path
  - Similarity score
  - Event type (check_in/check_out)
  - Optional GPS coordinates

### 4. **View Attendance Records**

Attendance records are stored in `data/attendance_records.json`:
```json
{
  "worker_id": 12345,
  "worker_name": "John Doe",
  "timestamp": 1698765432.123,
  "datetime": "2024-10-30T12:30:32",
  "photo_path": "data/attendance_photos/12345_John_Doe_check_in_1698765432.jpg",
  "similarity_score": 0.85,
  "event_type": "check_in",
  "gps_coordinates": null
}
```

## Configuration

### Face Recognition Parameters

Edit in `main.py` → `_initialize_components()`:

```python
self.face_system = FaceRecognitionSystem(
    model_name="Facenet512",        # Options: Facenet512, ArcFace, VGG-Face
    similarity_threshold=0.6,       # Range: 0.0-1.0 (higher = stricter)
    distance_metric="cosine"        # Options: cosine, euclidean
)
```

### Recommended Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model_name` | `Facenet512` | Best accuracy/speed trade-off |
| `similarity_threshold` | `0.6` | Good balance (adjust based on testing) |
| `distance_metric` | `cosine` | Better for high-dimensional embeddings |

### Threshold Tuning

- **Higher threshold (0.7-0.9)**: Fewer false positives, may miss some matches
- **Lower threshold (0.4-0.6)**: More matches, higher false positive rate
- **Recommended**: Start with 0.6, adjust based on your environment

## API Reference

### FaceRecognitionSystem

```python
# Initialize
face_system = FaceRecognitionSystem(
    embeddings_db_path="data/face_embeddings.pkl",
    attendance_db_path="data/attendance_records.json",
    model_name="Facenet512",
    similarity_threshold=0.6,
    distance_metric="cosine"
)

# Register worker
success = face_system.register_worker(
    worker_id=12345,
    worker_name="John Doe",
    face_images=[img1, img2, img3, img4, img5]
)

# Recognize worker
result = face_system.recognize_worker(face_img)
if result:
    worker_id, worker_name, similarity = result
    print(f"Recognized: {worker_name} (score: {similarity:.2f})")

# Mark attendance
face_system.mark_attendance(
    worker_id=worker_id,
    worker_name=worker_name,
    similarity_score=similarity,
    face_img=face_img,
    event_type="check_in"
)

# Get registered workers
workers = face_system.get_all_registered_workers()

# Delete worker
face_system.delete_worker(worker_id=12345)
```

## Performance Optimization

### Speed Optimizations

1. **GPU Acceleration** (if available):
   ```python
   # In main.py, change detector initialization:
   self.detector = PersonDetector(conf_threshold=0.5, device='cuda:0')
   ```

2. **Skip Frames**: Recognize every Nth frame instead of every frame
   ```python
   # In DetectionThread.run(), add frame counter
   if frame_count % 5 == 0:  # Recognize every 5th frame
       result = self.face_system.recognize_worker(face_img)
   ```

3. **Lower Resolution**: Resize faces before recognition
   ```python
   face_img = cv2.resize(face_img, (160, 160))
   ```

### Accuracy Improvements

1. **More Training Images**: Capture 7-10 images per worker
2. **Better Lighting**: Ensure consistent, good lighting
3. **Quality Check**: Filter blurry or poorly lit images
4. **Regular Updates**: Re-register workers periodically

## Troubleshooting

### Issue: "No match found" for registered workers

**Solutions:**
- Lower similarity threshold (try 0.5)
- Re-register with more diverse angles
- Check lighting conditions
- Ensure face is clearly visible

### Issue: False positives (wrong person recognized)

**Solutions:**
- Increase similarity threshold (try 0.7)
- Add more training images per worker
- Improve image quality during registration

### Issue: Slow performance

**Solutions:**
- Use GPU if available
- Skip frames (recognize every 3-5 frames)
- Reduce camera resolution
- Use lighter model (VGG-Face instead of Facenet512)

### Issue: TensorFlow warnings

**Solution:**
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings
```

## Database Schema

### Face Embeddings (PKL format)
```python
{
    'worker_id': int,
    'worker_name': str,
    'embedding': np.ndarray (512,),  # Facenet512 embedding
    'image_path': str,
    'timestamp': float
}
```

### Attendance Records (JSON format)
```python
{
    'worker_id': int,
    'worker_name': str,
    'timestamp': float,
    'datetime': str (ISO format),
    'photo_path': str,
    'similarity_score': float,
    'event_type': str,  # 'check_in' or 'check_out'
    'gps_coordinates': tuple or null
}
```

## Future Enhancements

1. **Database Integration**: Store embeddings in PostgreSQL
2. **Anti-Spoofing**: Add liveness detection
3. **Multi-Face Tracking**: Track multiple workers simultaneously
4. **Analytics Dashboard**: Attendance reports and statistics
5. **Mobile App**: Remote attendance viewing
6. **Face Mask Detection**: Recognize with/without masks
7. **Age/Gender Estimation**: Additional worker analytics

## Credits

- **YOLOv8**: Ultralytics (https://github.com/ultralytics/ultralytics)
- **DeepFace**: Sefik Ilkin Serengil (https://github.com/serengil/deepface)
- **Facenet512**: Google Research
- **PyQt5**: Riverbank Computing

## License

This integration maintains the same license as the parent project.
