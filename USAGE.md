# 🚗 Smart Driver Monitoring System - Usage Guide

## 🚀 Running the Application

### Start the Application

```bash
python main.py
```

## 📂 Clean Project Structure

### Core Application Files

- `main.py` - Application entry point
- `requirements.txt` - Python dependencies

### Source Code (`src/`)

- `gui.py` - GUI interface with enhanced red alerts
- `logic_optimized.py` - Main orchestrator with ML optimization
- `ml_engine.py` - Pure ML inference engine
- `ml_config.py` - Streamlined configuration (18 parameters)
- `drowsiness_detector.py` - Drowsiness detection logic
- `camera_handler.py` - Camera/video handling with frame resize
- `models.py` - Model loading utilities
- `utils.py` - Utility functions

### Data Directories

- `models/` - Pre-trained model files
  - `classification/` - Classification models (EfficientNet, MobileNet, VGG)
  - `detect/` - Detection models (YOLOv10, YOLOv11)
- `video/` - Test video files
- `config/` - Configuration files
- `logs/` - Application logs
- `alerts/` - Alert recordings
- `sound/` - Alert sound files

### Documentation

- `USAGE.md` - This user guide (you are here)

## ⚙️ System Features

### 🔴 Enhanced Red Alert System

- **Double-thick flashing red borders** around video frame
- **Large bilingual warning text** (Vietnamese + English)
- **GUI background flash** with multi-level alerts
- **Alert sound system** with different patterns for different danger levels
- **Red side panels** for maximum visibility
- **Pulsing effects** and color-coded status indicators

### 🎯 ML Optimizations

- **Streamlined configuration** - Reduced from 50+ to 18 core parameters
- **Automatic video frame resize** to 640x480 for optimal performance
- **Adaptive video timing** - Maintains original playback speed
- **Efficient model loading** - Prevents duplicate loading
- **Real-time inference** with performance monitoring

### 📱 GUI Improvements

- **Simplified model selection** - Dropdown-only interface
- **Clear model display names** - Shows version names instead of file names
- **Real-time detection indicators** with progress bars
- **Alert history tracking** with timestamps
- **Analytics dashboard** with real-time graphs

## 🎮 How to Use

### 1. Start Application

```bash
python main.py
```

### 2. Load Models

- Select **YOLO version** from dropdown (yolov10 or yolov11)
- Select **Classification model** from dropdown
- Click **"Load Models"** button
- Wait for success message: ✅ Models loaded

### 3. Start Monitoring

**Camera Mode:**

- Click **"Start Camera"** for real-time webcam monitoring

**Video Mode:**

- Click **"Upload Video"** to select a video file
- Click **"Start Video"** to begin analysis

### 4. Monitor Alerts

- Watch for **red visual alerts** on video frame
- Listen for **audio alerts** when drowsiness detected
- Check **detection panel** for real-time status
- Review **alert history** for logged events

## 🔧 Configuration

### Model Paths (Auto-detected)

- **YOLO models**: `models/detect/yolov10/best.pt` or `models/detect/yolov11/best.pt`
- **Classification models**: `models/classification/*.pth`

### Performance Settings

Default optimized settings work for most use cases:

- Video frame resize: 640x480
- GUI update rate: 50ms (20 FPS)
- Alert flash rate: 200-500ms depending on severity

## 🚨 Alert Levels

| Level            | Condition           | Visual                  | Audio       |
| ---------------- | ------------------- | ----------------------- | ----------- |
| **Normal**       | No issues           | Green indicators        | None        |
| **Mild Fatigue** | Low drowsiness      | Orange indicators       | None        |
| **Drowsy**       | Moderate drowsiness | Red + slow flash        | Single beep |
| **Danger**       | High drowsiness     | Bright red + fast flash | Double beep |
| **Critical**     | ML alerts active    | Full screen flash       | Triple beep |

## 🔧 Troubleshooting

### Common Issues

**Models not loading:**

- Ensure model files exist in `models/` directory
- Check file permissions
- Verify YOLO version matches folder name

**Video playback too fast/slow:**

- System automatically detects and maintains original FPS
- For custom timing, adjust in `camera_handler.py`

**No alerts showing:**

- Check if models are properly loaded
- Verify video contains faces
- Ensure alert system is enabled

**Performance issues:**

- Video frames are automatically resized to 640x480
- Close other GPU-intensive applications
- Use CPU mode if GPU has issues

### Support

For technical issues, check:

1. Model files are in correct locations
2. Python dependencies are installed: `pip install -r requirements.txt`
3. System has sufficient memory for ML models

## 📊 System Requirements

- **Python**: 3.8+
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for models
- **GPU**: Optional (CUDA support), CPU mode available
- **Camera**: Webcam for real-time monitoring (optional)

## 🎉 Enjoy Safe Driving!

The system is now optimized for maximum alert visibility and performance.
Stay alert, stay safe! 🚗💨
"optimize_for_inference": True
},

# === DETECTION THRESHOLDS ===

"detection": {
"confidence_threshold": 0.5, # General confidence threshold
"eye_closure_time": 2.0, # Seconds before drowsiness alert
"yawn_duration": 1.5, # Seconds of yawn for alert
"alert_cooldown": 3.0 # Seconds between alerts
},

# === PERFORMANCE OPTIMIZATION ===

"performance": {
"max_fps": 30,
"frame_skip": 1, # Process every N frames
"queue_size": 2, # Frame buffer size
"thread_priority": "normal" # normal, high
}

```

### Benefits over Old System

- **Parameter Reduction**: From 50+ to 18 parameters (64% reduction)
- **ML Focus**: Configuration designed around ML workflows
- **Maintainability**: Clean, organized, easier to modify
- **Performance**: Streamlined processing pipeline

## 🖥️ User Interface Guide

### Model Selection
1. From the Configuration tab, select your preferred YOLO version (yolov10/yolov11)
2. Select your preferred classification backbone (efficientnet_b0, mobilenet_v3_small, etc.)
3. Click "🚀 Load Models" to initialize

> **Note**: Auto model selection is disabled as requested.

### Input Selection
- "📷 Start Camera" - Use webcam for real-time monitoring
- "📁 Open Video" - Select a pre-recorded video file
- "⏹️ Stop" - Stop the current monitoring session

### Real-time Detection View
- Eye state indicator shows open/closed state with confidence
- Mouth state indicator shows normal/yawning with confidence
- Drowsiness level indicator shows current alertness level
- Alerts appear when drowsiness is detected

## 🔍 Troubleshooting

### Model Loading Issues
- Ensure model paths are correct in the configuration tab
- Verify model files exist in the expected directories:
  - YOLO models should be in `models/detect/[version]/best.pt`
  - Classification models should be in `models/classification/[backbone].pth`

### Video Playback Issues
- Fixed timing ensures videos play at correct speed
- If video still plays incorrectly, check the video file format

## 💻 Development Notes

### How to Test Configuration Changes

1. Modify parameters in `src/ml_config.py`
2. Run `python test_config.py` to validate
3. Run `python main.py` to test in GUI

### Architecture Benefits

- **Modular**: Each component has single responsibility
- **Testable**: Clear separation allows isolated testing
- **Scalable**: Easy to add new ML models or detection algorithms
- **Maintainable**: Simplified codebase, fewer parameters to manage
```
