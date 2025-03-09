import cv2
import logging
import pygame
import os
import torchvision.transforms as transforms
from PIL import Image
import torch

def setup_logging(log_path="logs/driver_monitoring.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

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