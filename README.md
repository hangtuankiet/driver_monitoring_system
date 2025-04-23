# Smart Driver Monitoring System

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-orange.svg)

The **Smart Driver Monitoring System** is a scientific research project developed at the University of Nha Trang. This system detects driver drowsiness and fatigue in real-time using computer vision and deep learning techniques. It combines YOLOv10 for eye and mouth detection with a custom VGG16 model for state classification to create an effective driver safety monitoring solution.

Developed by **Hàng Tuấn Kiệt** and **Huỳnh Thị Hạnh Nguyên** as part of university research.

## Features

- **Real-time Drowsiness Detection**: Monitors eye closure duration and yawn frequency
- **Multi-model AI Approach**: Uses YOLOv10 for detection and VGG16 for classification
- **Temporal Analysis**: Applies time-based analysis to reduce false positives
- **Audio Alerts**: Provides audible warnings when drowsiness is detected
- **Performance Evaluation**: Includes tools for measuring detection accuracy and latency
- **User-friendly Interface**: Features a clean Tkinter-based GUI with real-time visualization
- **Configurable Parameters**: Allows customization of detection thresholds and alert settings

## System Architecture

The system follows a modular architecture with clear separation of concerns:

![System Architecture](https://i.imgur.com/9FaWJdN.png)

For a detailed architecture diagram and component descriptions, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Requirements

### Hardware
- **Camera**: Standard webcam for real-time monitoring
- **CPU**: Intel i5 or AMD Ryzen 5 (or equivalent) with 4+ cores
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA support recommended (for optimal performance)
- **Storage**: 2GB available disk space for models and application

### Software
- **Python**: Version 3.10 or higher
- **Key Libraries**:
  - OpenCV
  - PyTorch
  - Ultralytics
  - Pygame
  - Tkinter
  - PIL
  - NumPy

## Installation Guide

### Prerequisites
1. Install Python 3.10+
2. Set up a virtual environment (optional but recommended):

```bash
# Using conda (recommended for GPU support)
conda create -n driver_monitor python=3.10
conda activate driver_monitor

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Package Installation

```bash
# Install main dependencies
pip install opencv-python pygame pillow numpy

# Install PyTorch (CPU version)
pip install torch torchvision

# OR for NVIDIA GPU support (adjust cuda version as needed)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# Install YOLOv10
pip install ultralytics
```

### Project Setup

1. Clone/extract the project to your desired location
2. Ensure model files are properly placed:
   - YOLOv10 model at `models/yolov10/yolov10n/best.pt`
   - VGG16 model at `models/vgg16/best_model.pth`
3. Verify sound file exists at `sound/eawr.wav`
4. Configure settings in `json/config.json` as needed

## Usage

### Starting the Application

Run the main application:

```bash
python main.py
```

### Main Features

1. **Real-time Monitoring**:
   - Select "Camera (Real-time)" from the dropdown menu
   - Click "Start" to begin monitoring
   
2. **Video Analysis**:
   - Select "Video" from the dropdown menu
   - Choose a video file from your computer
   - Click "Start" to process the video

3. **Performance Evaluation**:
   - Select "Evaluate Performance" from the dropdown menu
   - Choose a video file and enter ground truth data
   - Click "Start" to evaluate system performance

4. **Alert Review**:
   - Click "View Alerts" to see history of detected events
   - Export alert data to CSV if needed

5. **Settings Adjustment**:
   - Click "Settings" to adjust detection thresholds
   - Configure alert sounds and other parameters

## Performance

### Detection Metrics

The system has been evaluated with the following performance metrics:

- **Accuracy**: Approximately 50.7% on test dataset
- **Sensitivity (Recall)**: ~13.9%
- **Specificity**: ~62.8%
- **Precision**: ~11.0%
- **F1 Score**: ~12.3%
- **Average Latency**: ~0.049 seconds per frame
- **Average FPS**: ~14.0 frames per second

*Note: Performance metrics are based on internal testing and may vary based on hardware, lighting conditions, and subject characteristics.*

### System Limitations

- Performance depends on lighting conditions and camera quality
- May have reduced accuracy with certain eyewear or facial features
- CPU-only mode has significantly lower frame rates
- Detection reliability decreases with increasing distance from camera

## Troubleshooting

### Common Issues

1. **Camera Not Found**:
   - Verify camera is connected
   - Check camera index in `config.json` (default is 0)
   
2. **Models Not Loading**:
   - Ensure model files are in correct locations
   - Check for model file corruption or incomplete downloads

3. **Low Frame Rate**:
   - Enable GPU acceleration if available
   - Reduce video resolution in `config.json`
   - Close other resource-intensive applications

4. **False Detections**:
   - Adjust thresholds in `config.json`
   - Improve lighting conditions
   - Position camera properly facing the driver

## Development

### Project Structure

```
driver_monitoring_system/
├── src/                 # Core source code
│   ├── __init__.py
│   ├── models.py        # Model loading and management
│   ├── logic.py         # Core detection logic
│   ├── gui.py           # User interface
│   ├── utils.py         # Helper functions
│   ├── config.py        # Configuration management
│   └── evaluator.py     # Performance evaluation
├── models/              # Pre-trained AI models
├── json/                # Configuration files
├── logs/                # System logs
├── alerts/              # Alert history
├── sound/               # Alert sound files
├── video/               # Test videos
├── notebook/            # Development notebooks
├── main.py              # Application entry point
└── README.md            # Documentation
```

### Extending the System

To add new features or modify existing ones:

1. **New Detection Metrics**: Extend `evaluator.py`
2. **Additional Models**: Modify `models.py` to support more model types
3. **UI Customization**: Update `gui.py` with additional elements
4. **Configuration Options**: Add new parameters in `config.py` and `json/config.json`

## License

This project is released for academic and research purposes only.
Copyright © 2025 University of Nha Trang

## Acknowledgments

- University of Nha Trang for research support
- Ultralytics for YOLOv10 model architecture
- PyTorch team for the deep learning framework

## Authors

- **Hàng Tuấn Kiệt** - University of Nha Trang
- **Huỳnh Thị Hạnh Nguyên** - University of Nha Trang

*For questions or further information, please contact the authors.*
