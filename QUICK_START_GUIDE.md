# Quick Start Guide - Face Recognition System

## ✅ Issue Fixed!

The "No camera feed available" error has been resolved. The registration dialog now automatically starts cameras if they're not running.

## How to Use the System

### Step 1: Start the Application
```bash
python run.py
```

### Step 2: Register Workers

#### Option A: Register BEFORE Starting Monitoring (Recommended)
1. Click **"Register Worker"** button in the toolbar
2. Cameras will automatically start for registration
3. Fill in worker details:
   - Worker ID: Auto-generated (or enter manually)
   - Name: Enter worker's full name
   - Position: Optional
4. Position your face in the camera frame
5. Click **"Capture Face"** button 5 times from different angles:
   - ✓ Face forward (straight)
   - ✓ Turn head slightly left
   - ✓ Turn head slightly right  
   - ✓ Tilt head up slightly
   - ✓ Tilt head down slightly
6. Click **"Register Worker"** to save
7. Cameras will automatically stop after registration

#### Option B: Register AFTER Starting Monitoring
1. Click **"Start"** to begin monitoring
2. Click **"Register Worker"** button
3. Follow steps 3-6 above
4. Cameras continue running after registration

### Step 3: Start Monitoring
1. Click **"Start"** button in the toolbar
2. Both cameras will begin capturing
3. System automatically recognizes registered workers
4. Recognized workers show:
   - **Yellow bounding box** (vs green for unknown persons)
   - **Worker name** displayed
   - **Recognition score** (0.00-1.00)

### Step 4: View Results
- **Cameras Tab**: Live camera feeds with recognition
- **Worker Status Tab**: List of detected workers and their status
- **Alerts Tab**: Notifications and warnings
- **Reports Tab**: Attendance logs

## Tips for Best Results

### During Registration:
✅ **Good lighting** - Ensure face is well-lit
✅ **Clear face** - Remove glasses/hats if possible
✅ **Look at camera** - Make eye contact with camera
✅ **Multiple angles** - Capture from different positions
✅ **One person only** - Ensure only one face in frame

### During Recognition:
✅ **Same lighting** - Similar conditions as registration
✅ **Clear view** - Face should be visible to camera
✅ **Patience** - Recognition happens every few frames
✅ **Distance** - Stay within 1-3 meters of camera

## Troubleshooting

### "No camera feed available"
- **Fixed!** Cameras now start automatically
- If still seeing this, check camera connections

### Worker not recognized
- **Lower threshold**: Edit `main.py` line 140, change `0.6` to `0.5`
- **Re-register**: Capture more/better images
- **Check lighting**: Ensure similar lighting as registration

### Multiple false positives
- **Raise threshold**: Change `0.6` to `0.7` or `0.8`
- **Better registration**: Use clearer images

### Slow performance
- **Skip frames**: Recognize every 3-5 frames instead of all
- **Lower resolution**: Reduce camera resolution in config
- **Use GPU**: Change `device='cpu'` to `device='cuda:0'` if available

## File Locations

### Data Storage:
- **Face embeddings**: `data/face_embeddings.pkl`
- **Attendance records**: `data/attendance_records.json`
- **Registration photos**: `data/registered_faces/worker_[ID]/`
- **Attendance photos**: `data/attendance_photos/`

### Configuration:
- **Camera settings**: `config/config.json`
- **Recognition settings**: `src/main.py` (line 137-142)

## Recognition Settings

Current settings (in `src/main.py`):
```python
self.face_system = FaceRecognitionSystem(
    model_name="Facenet512",        # Face recognition model
    similarity_threshold=0.6,       # Match threshold (0.0-1.0)
    distance_metric="cosine"        # Similarity calculation method
)
```

### Adjust Threshold:
- **Stricter** (fewer false positives): `0.7` - `0.9`
- **Default** (balanced): `0.6`
- **Lenient** (more matches): `0.4` - `0.5`

## Support

For issues or questions, check:
1. `FACE_RECOGNITION_README.md` - Detailed documentation
2. `app.log` - Application logs
3. Console output - Real-time error messages

## Next Steps

1. ✅ Register all workers
2. ✅ Test recognition with each worker
3. ✅ Adjust threshold if needed
4. ✅ Start daily monitoring
5. ✅ Review attendance records regularly

---

**System Status**: ✅ Fully Operational with Face Recognition
