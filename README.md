# Smart Driver Monitoring System

The **Smart Driver Monitoring System** is a scientific research project developed as part of a university assignment at the University of Nha Trang. This system aims to improve driver safety by detecting signs of drowsiness or fatigue using advanced computer vision and deep learning techniques. It utilizes the YOLOv10 model for detecting eyes and mouth, and a custom VGG16 model for classifying eye closure and yawn states. The project was collaboratively developed by **Hàng Tuấn Kiệt** and **Huỳnh Thị Hạnh Nguyên**.

## Project Overview
This project was created to explore the application of AI in real-time driver monitoring. It provides a practical solution for detecting drowsiness, issuing alerts, and evaluating system performance, serving as a proof-of-concept for academic research.

## Features
- **Drowsiness Detection**: Tracks eye closure duration and yawn frequency to identify fatigue.
- **Real-time Processing**: Analyzes video feeds from a camera or pre-recorded files.
- **Audio Alerts**: Emits a warning sound when drowsiness is detected.
- **Performance Evaluation**: Assesses system accuracy using ground truth data.
- **GUI**: Offers an intuitive Tkinter-based interface for user interaction.
- **Configurable Settings**: Allows customization of parameters like eye closure threshold and sound volume.

## Project Structure
```
driver_monitoring_system/
├── src/
│   ├── __init__.py
│   ├── models.py          # AI model definitions (YOLO, VGG16)
│   ├── logic.py           # Core logic (DriverMonitor class)
│   ├── gui.py             # GUI implementation
│   ├── utils.py           # Utility functions (preprocessing, logging, etc.)
│   ├── config.py          # Configuration management
│   └── evaluator.py       # Performance evaluation module
├── models/
│   ├── yolov10/
│   │   └── yolov10n/
│   │       └── best.pt    # Pre-trained YOLOv10 weights
│   ├── vgg16/
│   │   ├── eye/
│   │   │   └── eye.pt     # Custom VGG16 weights for eye detection
│   │   └── yawn/
│   │       └── mouth.pt   # Custom VGG16 weights for yawn detection
├── json/
│   └── config.json        # Configuration file
├── logs/
│   └── driver_monitoring.log  # Log file
├── alerts/
│   └── alert_history.json # Alert history storage
├── sound/
│   └── eawr.wav           # Alert sound file
├── video/
│   └── (video files)      # Optional video input files
└── main.py                # Entry point to run the application
```

## Prerequisites
- **Python**: Version 3.8 or higher (tested with 3.10.16).
- **Conda**: Recommended for environment management.
- **GPU (optional)**: NVIDIA GPU with CUDA support for faster processing.

### Required Libraries
- `opencv-python`
- `pygame`
- `torch`
- `torchvision`
- `ultralytics`
- `pillow`

## Installation

### Step 1: Obtain the Project
Since this is a university project, it may not be hosted publicly. Contact **Hàng Tuấn Kiệt** or **Huỳnh Thị Hạnh Nguyên** to request the source code or download it from the provided university repository (if available).

Alternatively, if shared via a compressed file:
1. Extract the ZIP file to a local directory (e.g., `D:\NTU\Can_su\VGG_YOLO_Project\`).

### Step 2: Set Up the Conda Environment
Create and activate a Conda environment:
```bash
conda create -n torch_gpu python=3.10
conda activate torch_gpu
```

### Step 3: Install Dependencies
Install the required libraries:
```bash
pip install opencv-python pygame torch torchvision ultralytics pillow
```

For GPU support:
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```
*(Adjust `cu118` based on your CUDA version.)*

### Step 4: Prepare Pre-trained Models
Ensure the following model files are placed in the correct directories:
- **YOLOv10**: `models/yolov10/yolov10n/best.pt` (download from [Ultralytics](https://github.com/ultralytics/ultralytics) or use your trained model).
- **VGG16**: 
  - `models/vgg16/eye/eye.pt` (custom-trained for eye detection).
  - `models/vgg16/yawn/mouth.pt` (custom-trained for yawn detection).

Contact the authors if these files are not included.

### Step 5: Configure the Application
Edit `json/config_torch.json` to match your setup:
```json
{
    "yolo_model_path": "models/yolov10/yolov10n/best.pt",
    "eye_model_path": "models/vgg16/eye/eye.pt",
    "yawn_model_path": "models/vgg16/yawn/mouth.pt",
    "alert_sound": "sound/eawr.wav",
    "eye_closure_threshold": 2.1,
    "capture_device": 0,
    "video_path": "video/",
    "save_alerts": true,
    "sound_enabled": true,
    "sound_volume": 0.5
}
```

### Step 6: Add Alert Sound
Place a `.wav` file (e.g., `eawr.wav`) in the `sound/` directory. Use an existing file or download one from a free sound library.

## Running the Application
Run the main script:
```bash
conda activate torch_gpu
python main.py
```

- **Camera Mode**: Select "Camera (Real-time)" for webcam input.
- **Video Mode**: Select "Video" and choose a file from `video/`.
- **Evaluation Mode**: Select "Evaluate Performance" and provide ground truth data.

## Usage
- **Start/Stop**: Use "Start" to begin monitoring and "Pause/Resume" to control it.
- **Settings**: Adjust parameters via the "Settings" menu.
- **Alerts**: View and export alert history in the "View Alerts" menu.
- **Evaluation**: Input drowsy time ranges (e.g., "10-15, 20-23") for performance analysis.

## Troubleshooting
- **FileNotFoundError**: Verify all model and sound files are in place.
- **ModuleNotFoundError**: Ensure all dependencies are installed.
- **GPU Issues**: Confirm CUDA compatibility with PyTorch.

## Authors
- **Hàng Tuấn Kiệt** - University of Nha Trang
- **Huỳnh Thị Hạnh Nguyên** - University of Nha Trang

## Acknowledgments
This project was developed as part of a scientific research assignment at the University of Nha Trang. Special thanks to our instructors and peers for their support.
