# Camera Setup Guide

## Current Configuration

**You now have 4 cameras configured:**
- Camera 0 (ID: 0)
- Camera 1 (ID: 1)
- Camera 2 (ID: 2)
- Camera 3 (ID: 3)

All cameras are configured with:
- Resolution: 1280x720
- FPS: 60
- No ROI restrictions

## How Camera IDs Work

The camera ID corresponds to the **device index** on your system:
- **ID 0** = First webcam (usually built-in laptop camera)
- **ID 1** = Second webcam (first external USB camera)
- **ID 2** = Third webcam (second external USB camera)
- **ID 3** = Fourth webcam (third external USB camera)

## Adding More Cameras

To add more cameras, edit `config/config.json`:

```json
{
    "id": 4,
    "name": "Camera 4",
    "width": 1280,
    "height": 720,
    "fps": 60,
    "rois": []
}
```

**Important:** Make sure you have physical cameras connected for each ID!

## Testing Cameras

**Start the application:**
```bash
python run.py
```

**Check which cameras are working:**
- Look at the UI - it will show which cameras are active
- The logs will show: "Started X camera thread(s)"

**If a camera fails:**
- Check if the physical camera is connected
- Try a different USB port
- Check if another application is using the camera
- Reduce FPS if camera can't handle 60fps

## Camera Settings

### Resolution
```json
"width": 1280,
"height": 720
```
Common resolutions:
- 640x480 (VGA)
- 1280x720 (HD)
- 1920x1080 (Full HD)

### FPS (Frames Per Second)
```json
"fps": 60
```
Common FPS values:
- 30 (standard)
- 60 (smooth)
- 15 (low bandwidth)

**Note:** Not all cameras support 60fps. If a camera fails, try reducing to 30fps.

### ROI (Region of Interest)
```json
"rois": []
```
Leave empty to monitor the entire frame, or add specific regions:
```json
"rois": [
    {"x": 0, "y": 0, "width": 640, "height": 480, "name": "Entrance"}
]
```

## Using Multiple Cameras

### For Monitoring
All cameras will run simultaneously and detect workers in parallel.

### For Registration
Use the camera selector dropdown to choose which camera to register from:
1. Open "Register Worker"
2. Select camera from "Camera:" dropdown
3. Capture 10 faces from that camera
4. Register worker

## Troubleshooting

### Camera Not Detected
```
Error: Failed to open camera X
```
**Solutions:**
1. Check if camera is physically connected
2. Try a different USB port
3. Check if camera works in other apps
4. Reduce FPS to 30
5. Change resolution to 640x480

### Camera Lag
```
Low FPS, delayed frames
```
**Solutions:**
1. Reduce FPS from 60 to 30
2. Reduce resolution to 640x480
3. Close other applications using cameras
4. Use fewer cameras simultaneously

### Wrong Camera
```
Camera 0 shows wrong feed
```
**Solutions:**
1. Cameras may be numbered differently by OS
2. Try swapping camera IDs in config
3. Test each camera individually

## Example: 2 Cameras Setup

**Minimal setup for 2 cameras:**
```json
"cameras": [
    {
        "id": 0,
        "name": "Front Door",
        "width": 1280,
        "height": 720,
        "fps": 30,
        "rois": []
    },
    {
        "id": 1,
        "name": "Back Door",
        "width": 1280,
        "height": 720,
        "fps": 30,
        "rois": []
    }
]
```

## Current Status

✅ **4 cameras configured**
- Camera 0, 1, 2, 3
- All at 1280x720, 60fps
- Ready to use

**Next steps:**
1. Connect physical cameras (if not already)
2. Run `python run.py`
3. Check which cameras are detected
4. Adjust FPS/resolution if needed
