import cv2
import logging
import pygame
import os
import torchvision.transforms as transforms
from PIL import Image
import torch
from datetime import datetime

def setup_logging(log_path="logs/driver_monitoring.log", log_level=logging.INFO):
    """
    Cấu hình logging để ghi vào file và hiển thị trên console.
    File log sẽ được reset (xóa nội dung cũ) mỗi khi chương trình chạy lại.

    Args:
        log_path (str): Đường dẫn đến file log.
        log_level (int): Mức logging (mặc định là INFO).
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Xóa file log cũ nếu tồn tại
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
            print(f"Reset log file: {log_path}")
        except Exception as e:
            print(f"Error resetting log file {log_path}: {str(e)}")
            # Nếu không xóa được, đổi tên file log cũ
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_log_path = f"{log_path}.{timestamp}.bak"
            try:
                os.rename(log_path, new_log_path)
                print(f"Renamed old log file to {new_log_path}")
            except Exception as rename_e:
                print(f"Error renaming log file: {str(rename_e)}")

    # Tạo logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Định dạng log
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Handler cho file (ghi đè file log)
    try:
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_format)
    except Exception as e:
        print(f"Error setting up file handler: {str(e)}")
        file_handler = logging.NullHandler()

    # Handler cho console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)

    # Xóa các handler cũ (nếu có)
    logger.handlers.clear()

    # Thêm các handler mới
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def preprocess_image(image):
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

def play_alarm(sound_path):
    try:
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
    except Exception as e:
        logging.error(f"Error playing alarm: {str(e)}")

def create_storage_directories():
    os.makedirs('alerts', exist_ok=True)
    os.makedirs('logs', exist_ok=True)