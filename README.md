# 🚗 Smart Driver Monitoring System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)


An AI-powered real-time driver drowsiness detection system using computer vision and machine learning. Features enhanced red alert system with multi-modal warnings to prevent accidents caused by driver fatigue.

## 🎯 Key Features

### 🔴 Enhanced Red Alert System

- **Double-thick flashing red borders** (20px) around video frame with 0.33s flash timing
- **Large bilingual warning text** (Vietnamese + English) for maximum visibility
- **GUI background flash** with multi-level color-coded alerts
- **Multi-pattern audio alerts** (1-3 beeps) based on danger severity
- **Red side panels** and pulsing background overlay (60% opacity)
- **Alert level hierarchy**: Normal → Mild → Drowsy → Danger → Critical

### 🎯 ML-Powered Detection

- **Dual model architecture**: YOLO (face detection) + Classification (eye/mouth state)
- **Real-time inference** with optimized ML pipeline
- **Streamlined configuration** - Reduced from 50+ to 18 core parameters
- **Performance monitoring** with FPS and inference time tracking
- **Automatic video frame resize** to 640x480 for optimal performance

### 📱 User-Friendly Interface

- **Simplified model selection** via dropdown interface
- **Real-time detection indicators** with progress bars
- **Alert history tracking** with timestamps
- **Analytics dashboard** with live performance metrics
- **Camera and video file support** for flexible testing

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time monitoring)
- 4GB RAM minimum, 8GB recommended
- 2GB storage for model files

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/driver-monitoring-system.git
cd driver-monitoring-system
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the application**

```bash
python main.py
```

## 🎮 How to Use

### 1. Start Application

Launch the GUI interface:

```bash
python main.py
```

### 2. Load Models

- Select **YOLO version** from dropdown (yolov10 or yolov11)
- Select **Classification model** from dropdown (EfficientNet, MobileNet, VGG16)
- Click **"Load Models"** button
- Wait for success message: ✅ Models loaded

### 3. Start Monitoring

**Real-time Camera Mode:**

- Click **"Start Camera"** for live webcam monitoring

**Video Analysis Mode:**

- Click **"Upload Video"** to select a video file
- Click **"Start Video"** to begin analysis

### 4. Monitor Alerts

- 👀 **Visual alerts**: Red flashing borders and warning text on video frame
- 🔊 **Audio alerts**: System beeps when drowsiness detected
- 📊 **Detection panel**: Real-time eye/mouth state with confidence scores
- 📝 **Alert history**: Timestamped log of all detected events

## 🚨 Alert System

| Alert Level         | Trigger Condition       | Visual Indicator        | Audio Pattern |
| ------------------- | ----------------------- | ----------------------- | ------------- |
| **Normal** 😊       | No drowsiness signs     | Green indicators        | None          |
| **Mild Fatigue** 😐 | Low drowsiness (25-50%) | Orange indicators       | None          |
| **Drowsy** 😴       | Moderate (50-75%)       | Red + slow flash        | Single beep   |
| **Danger** 🚨       | High drowsiness (75%+)  | Bright red + fast flash | Double beep   |
| **Critical** ⚠️     | ML alerts active        | Full screen flash       | Triple beep   |

### Detection Thresholds

- **Eye closure**: 2.0 seconds before drowsiness alert
- **Yawn duration**: 1.5 seconds of continuous yawning
- **Confidence threshold**: 0.5 for ML model predictions
- **Alert cooldown**: 3.0 seconds between consecutive alerts

## 📂 Project Structure

```
driver_system/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── src/                   # Core source code (9 files)
│   ├── gui.py            # GUI interface with red alert system
│   ├── logic_optimized.py   # Main orchestrator with ML optimization
│   ├── ml_engine.py      # Pure ML inference engine
│   ├── ml_config.py      # Streamlined configuration (18 parameters)
│   ├── drowsiness_detector.py  # Drowsiness detection logic
│   ├── camera_handler.py    # Camera/video handling with frame resize
│   ├── models.py         # Model loading utilities
│   ├── utils.py          # Utility functions
│   └── __init__.py       # Package initialization
│
├── models/               # Pre-trained model files
│   ├── classification/   # Eye/mouth state models
│   │   ├── efficientnet_b0.pth
│   │   ├── mobilenet_v2.pth
│   │   ├── mobilenet_v3_small.pth
│   │   └── vgg16.pth
│   └── detect/          # Face detection models
│       ├── yolov10/     # YOLOv10 models
│       └── yolov11/     # YOLOv11 models
│
├── notebook/            # Jupyter notebooks for development
│   ├── emd-yolov10v11-finetune.ipynb    # Model fine-tuning experiments
│   └── train-multitask-split.ipynb      # Multi-task training pipeline
│
├── video/               # Test video files
├── config/              # Configuration files
├── logs/                # Application logs
├── alerts/              # Alert recordings
└── sound/               # Alert sound files
```

## ⚙️ Configuration

The system uses an optimized ML-focused configuration with 18 core parameters:

### Model Configuration

- **Device**: Auto-detection (CUDA/CPU)
- **Input size**: 640x480 (auto-resize)
- **Batch size**: 1 (real-time processing)
- **Optimization**: Inference-optimized models

### Detection Thresholds

- **Confidence threshold**: 0.5
- **Eye closure time**: 2.0 seconds
- **Yawn duration**: 1.5 seconds
- **Alert cooldown**: 3.0 seconds

### Performance Settings

- **Max FPS**: 30
- **Frame processing**: Every frame
- **Queue size**: 2 frames
- **Thread priority**: Normal

## 🔧 Advanced Usage

### Custom Model Integration

1. Place your model files in the appropriate directories:

   - YOLO models: `models/detect/[version]/best.pt`
   - Classification models: `models/classification/[name].pth`

2. Update model configuration in `src/ml_config.py`

3. Restart the application to detect new models

### Performance Optimization

- **GPU acceleration**: Automatic CUDA detection and utilization
- **Frame optimization**: Auto-resize to 640x480 for consistent performance
- **Memory management**: Efficient model loading and frame processing
- **Real-time processing**: Optimized inference pipeline for live video

## 🛠️ Troubleshooting

### Common Issues

**Models not loading:**

- Verify model files exist in `models/` directory
- Check file permissions and paths
- Ensure YOLO version matches folder structure

**Performance issues:**

- Video frames auto-resize to 640x480 for optimization
- Close GPU-intensive applications
- Use CPU mode if GPU memory is insufficient

**No detection results:**

- Ensure proper lighting and face visibility
- Check if models are properly loaded
- Verify camera/video input is working

**Alert system not responding:**

- Check audio system permissions
- Verify alert thresholds in configuration
- Review logs for error messages

## 📊 System Requirements

| Component   | Minimum             | Recommended     |
| ----------- | ------------------- | --------------- |
| **Python**  | 3.8+                | 3.9+            |
| **RAM**     | 4GB                 | 8GB             |
| **Storage** | 2GB                 | 5GB             |
| **GPU**     | None (CPU mode)     | CUDA-compatible |
| **Camera**  | Any webcam          | HD webcam       |
| **OS**      | Windows/Linux/macOS | Windows 10+     |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request


## 🙏 Acknowledgments

- **OpenCV** for computer vision capabilities
- **PyTorch** for deep learning framework
- **YOLO** for real-time object detection
- **Contributors** who helped improve the system

## 📞 Support

For technical support or questions:

1. Check the [Hang Tuan Kiet](#-hangtuankiet)
2. Review [issues](https://github.com/hangtuankiet/driver_monitoring_system/issues)
3. Create a new issue with detailed description

## 🚗 Stay Safe!

This system is designed to enhance driver safety but should not replace responsible driving practices. Always ensure adequate rest before driving and follow traffic safety guidelines.

**Drive safely, arrive alive!** 🛡️✨
