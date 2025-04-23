import json
import os

class ConfigManager:
    """Configuration manager for driver monitoring system.
    
    Handles loading, saving, and providing default configuration options.
    Ensures all required config keys are present by merging defaults with user settings.
    """
    
    DEFAULT_CONFIG = {
        # Model paths
        'yolo_model_path': "models/yolov10/yolov10n/best.pt",
        'vgg_model_path': "models/vgg16/best_model.pth",
        
        # Detection thresholds
        'confidence_threshold': 0.6,          # YOLO detection confidence
        
        # Alert thresholds
        'eye_closure_threshold': 2.1,         # Seconds eyes must be closed to trigger alert
        'yawn_threshold': 2.0,                # Seconds of yawning to trigger alert
        'yawn_grace_period': 0.5,             # Grace period after yawn detection
        
        # Yawn detection parameters
        'yawn_confidence_threshold': 0.85,    # VGG confidence for yawn classification
        'yawn_size_threshold': 0.45,          # Mouth aspect ratio threshold for yawn
        
        # I/O settings
        'capture_device': 0,                  # Camera device index
        'video_path': "video/",               # Default path for video files
        'alert_sound': "sound/eawr.wav",      # Alert sound file path
        
        # User preferences
        'save_alerts': True,                  # Whether to save alerts to history
        'sound_enabled': True,                # Enable/disable alert sounds
        'sound_volume': 0.5,                  # Alert sound volume (0.0-1.0)
    }

    def __init__(self, config_path="json/config.json"):
        """Initialize the ConfigManager with a path to the config file.
        
        Args:
            config_path (str): Path to the configuration JSON file
        """
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from file, ensuring all default keys are present.
        
        Returns:
            dict: Complete configuration with all required keys
        """
        try:
            # Try to load existing config
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
                
            # Ensure all required keys exist by merging with defaults
            updated_config = self.DEFAULT_CONFIG.copy()
            updated_config.update(loaded_config)
            
        except FileNotFoundError:
            # Create config directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Use default config
            updated_config = self.DEFAULT_CONFIG.copy()
            
            # Save default config
            with open(self.config_path, 'w') as f:
                json.dump(updated_config, f, indent=4)
                
        return updated_config

    def save_config(self):
        """Save current configuration to file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)