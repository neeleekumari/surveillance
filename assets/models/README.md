# Models Directory

This directory contains AI models used by the Floor Monitoring Application.

## YOLOv8 Model

The application uses the YOLOv8 model for person detection. On first run, the application will automatically download the `yolov8n.pt` model file from the Ultralytics repository.

### Model Files

- `yolov8n.pt` - Nano version of YOLOv8 (smallest, fastest)
- `yolov8s.pt` - Small version of YOLOv8 (balanced)
- `yolov8m.pt` - Medium version of YOLOv8 (more accurate)
- `yolov8l.pt` - Large version of YOLOv8 (most accurate)
- `yolov8x.pt` - Extra large version of YOLOv8 (highest accuracy)

### Model Selection

The default model is `yolov8n.pt` which provides a good balance of speed and accuracy for real-time person detection. You can change the model by modifying the detection module configuration.

### Custom Models

You can use custom trained models by placing them in this directory and updating the configuration in `src/detection_module.py`.