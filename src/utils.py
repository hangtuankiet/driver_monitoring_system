import cv2
import logging
import pygame
import os
import torchvision.transforms as transforms
from PIL import Image
import torch
from datetime import datetime
from typing import Union
import numpy as np



def setup_logging(log_path: str = "logs/driver_monitoring.log", log_level: int = logging.INFO) -> None:
    """Configure logging to write to a file and display on the console.

    This function sets up a logger that writes to both a file and the console with a specified
    log level. The log file is reset (cleared) each time the program runs. If the log file cannot
    be deleted, it is renamed with a timestamp suffix to avoid conflicts.

    Args:
        log_path (str, optional): Path to the log file. Defaults to "logs/driver_monitoring.log".
        log_level (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG).
            Defaults to logging.INFO.

    Raises:
        OSError: If there are permission issues or other errors when creating directories,
            deleting, or renaming the log file.

    The log format includes the timestamp, log level, and message. Existing handlers are cleared
    to prevent duplicate logging, and new handlers for file and console output are added.
    """
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Reset the log file by deleting it or renaming it if deletion fails
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
            print(f"Reset log file: {log_path}")
        except Exception as e:
            print(f"Error resetting log file {log_path}: {str(e)}")
            # If deletion fails, rename the old log file with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_log_path = f"{log_path}.{timestamp}.bak"
            try:
                os.rename(log_path, new_log_path)
                print(f"Renamed old log file to {new_log_path}")
            except Exception as rename_e:
                print(f"Error renaming log file: {str(rename_e)}")

    # Create and configure the logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Define the log format
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Set up the file handler (overwrite mode)
    try:
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_format)
    except Exception as e:
        print(f"Error setting up file handler: {str(e)}")
        file_handler = logging.NullHandler()

    # Set up the console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Add the new handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def preprocess_image(image: 'np.ndarray') -> Union[torch.Tensor, None]:
    """Preprocess an image for input to a deep learning model.

    This function converts an input image (numpy array) into a tensor suitable for
    a deep learning model, applying resizing, normalization, and device placement.
    The image is expected to be in BGR format (as read by OpenCV).

    Args:
        image (numpy.ndarray): Input image as a numpy array in BGR format.

    Returns:
        torch.Tensor | None: Preprocessed image tensor of shape (1, C, H, W) on the
            appropriate device (CUDA if available, otherwise CPU), or None if preprocessing fails.

    Raises:
        ValueError: If the input image is empty (zero-sized).

    The preprocessing pipeline includes converting the image to a PIL image, resizing to
    224x224, converting to a tensor, normalizing with ImageNet mean and std, and adding a
    batch dimension. Errors during preprocessing are logged, and None is returned.
    """
    try:
        if image.size == 0:
            raise ValueError("Empty image")
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return preprocess(image).unsqueeze(0).to(device)
    except Exception as e:
        logging.error(f"Error preprocessing image: {str(e)}")
        return None


def play_alarm(sound_path: str) -> None:
    """Play an alarm sound using pygame.mixer.

    Args:
        sound_path (str): Path to the sound file to play (e.g., a .wav file).

    Raises:
        pygame.error: If there is an error loading or playing the sound file.

    This function loads and plays the specified sound file using pygame.mixer. Errors
    during playback are logged but not raised, allowing the program to continue running.
    """
    try:
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
    except Exception as e:
        logging.error(f"Error playing alarm: {str(e)}")


def create_storage_directories() -> None:
    """Create storage directories for alerts and logs.

    This function ensures that the 'alerts' and 'logs' directories exist, creating them
    if necessary. These directories are used to store alert history and log files.

    Raises:
        OSError: If there are permission issues or other errors when creating directories.
    """
    os.makedirs('alerts', exist_ok=True)
    os.makedirs('logs', exist_ok=True)