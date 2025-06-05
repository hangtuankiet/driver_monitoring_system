import json
import os

class ConfigManager:
    """Configuration manager for driver monitoring system.
    
    Handles loading, saving, and providing default configuration options.
    Ensures all required config keys are present by merging defaults with user settings.
    """
    
    DEFAULT_CONFIG = {
        # Model paths
        'yolo_model_path': "models/detected/yolov10.pt",
        'classification_model_path': "models/classification/vgg16_model.pth",
        
        # Model selection
        'yolo_version': 'yolov10',            # Available: yolov10, yolov11
        'classification_backbone': 'vgg16',    # Available: vgg16, mobilenet_v2, mobilenet_v3_small, efficientnet_b0
        'num_classes': 4,                     # Number of classification classes
        
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

    def get_available_backbones(self):
        """Get list of available classification backbones.
        
        Returns:
            list: List of available backbone names
        """
        return ['vgg16', 'mobilenet_v2', 'mobilenet_v3_small', 'efficientnet_b0']
    
    def get_model_path_for_backbone(self, backbone_name):
        """Get the model path for a specific backbone.
        
        Args:
            backbone_name (str): Name of the backbone
            
        Returns:
            str: Path to the model file
        """
        model_paths = {
            'vgg16': "models/classification/vgg16_model.pth",
            'mobilenet_v2': "models/classification/mobilenet_v2_model.pth", 
            'mobilenet_v3_small': "models/classification/mobilenet_v3_small_model.pth",
            'efficientnet_b0': "models/classification/efficientnet_b0_model.pth"
        }
        return model_paths.get(backbone_name, self.config['classification_model_path'])

    def update_classification_model(self, backbone_name):
        """Update the classification model configuration.
        
        Args:
            backbone_name (str): Name of the new backbone to use
        """
        if backbone_name in self.get_available_backbones():
            self.config['classification_backbone'] = backbone_name
            self.config['classification_model_path'] = self.get_model_path_for_backbone(backbone_name)
            self.save_config()
    
    def get_available_yolo_versions(self):
        """Get list of available YOLO versions.
        
        Returns:
            list: List of available YOLO version names
        """
        return ['yolov10', 'yolov11']
    
    def get_yolo_model_path_for_version(self, version):
        """Get the model path for a specific YOLO version.
        
        Args:
            version (str): Name of the YOLO version
            
        Returns:
            str: Path to the YOLO model file
        """
        model_paths = {
            'yolov10': "models/detected/yolov10.pt",
            'yolov11': "models/detected/yolov11.pt"
        }
        return model_paths.get(version, self.config['yolo_model_path'])

    def update_yolo_model(self, version):
        """Update the YOLO model configuration.
        
        Args:
            version (str): Name of the new YOLO version to use
        """
        if version in self.get_available_yolo_versions():
            self.config['yolo_version'] = version
            self.config['yolo_model_path'] = self.get_yolo_model_path_for_version(version)
            self.save_config()