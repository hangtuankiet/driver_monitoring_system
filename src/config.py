import json
import os

class ConfigManager:
    DEFAULT_CONFIG = {
        'yolo_model_path': "models/yolov10/yolov10n/best.pt",
        'vgg_model_path': "models/vgg16/best_model.pth",

        'confidence_threshold': 0.6,
      
        'eye_closure_threshold': 2.1,

        'yawn_threshold': 3.0,
        'yawn_size_threshold': 0.45,
        'yawn_confidence_threshold': 0.85,
        'yawn_grace_period': 0.5,

        'capture_device': 0,
        'video_path': "video/",
        'save_alerts': True,
        'sound_enabled': True,
        'sound_volume': 0.5,
        'alert_sound': "sound/eawr.wav",
        
    }

    def __init__(self, config_path="json/config.json"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from file, ensuring all default keys are present."""
        try:
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
        except FileNotFoundError:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.DEFAULT_CONFIG, f, indent=4)
            return self.DEFAULT_CONFIG.copy()

        # Ensure all keys from DEFAULT_CONFIG are present in loaded_config
        updated_config = self.DEFAULT_CONFIG.copy()
        updated_config.update(loaded_config)
        # Save the updated config back to file to ensure consistency
        with open(self.config_path, 'w') as f:
            json.dump(updated_config, f, indent=4)
        return updated_config

    def save_config(self):
        """Save current configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)