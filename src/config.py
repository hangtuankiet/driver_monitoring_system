import json
import os

class ConfigManager:
    DEFAULT_CONFIG = {
        'yolo_model_path': "models/yolov10/yolov10n/best.pt",
        'eye_model_path': "models/vgg16/eye/eye.pt",
        'yawn_model_path': "models/vgg16/yawn/mouth.pt",
        'alert_sound': "sound/eawr.wav",
        'eye_closure_threshold': 2.1,
        'yawn_threshold': 2.0,
        'yawn_size_threshold': 0.5,
        'capture_device': 0,
        'video_path': "video/",
        'save_alerts': True,
        'sound_enabled': True,
        'sound_volume': 0.5
    }

    def __init__(self, config_path="json/config.json"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.DEFAULT_CONFIG, f, indent=4)
            return self.DEFAULT_CONFIG

    def save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)