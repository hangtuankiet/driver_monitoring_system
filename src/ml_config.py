"""
ML-Optimized Configuration - Streamlined parameters focusing on ML performance
Reduced parameter complexity while maintaining functionality
"""
import json
import os
import logging
import glob

class MLOptimizedConfig:
    """Simplified configuration focusing on ML performance"""
    
    def __init__(self, config_path='config/ml_config.json'):
        self.config_path = config_path
        self.config = self._load_or_create_default()
    
    def _load_or_create_default(self):
        """Load existing config or create optimized default"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load config: {e}, using defaults")
        
        return self._get_ml_optimized_defaults()
    
    def _get_ml_optimized_defaults(self):
        """ML-focused default configuration with minimal parameters"""
        return {
            # === CORE ML PARAMETERS ===
            "ml_config": {
                "detection_model_version": "yolov11",     # Model version name
                "classification_backbone": "efficientnet_b0",  # Backbone name
                "device": "auto",  # auto, cpu, cuda
                "inference_batch_size": 1,
                "model_warmup": True,
                "optimize_for_inference": True
            },
            
            # === DETECTION THRESHOLDS (Simplified) ===
            "detection": {
                "confidence_threshold": 0.5,  # General confidence threshold
                "eye_closure_time": 2.0,      # Seconds before drowsiness alert
                "yawn_duration": 1.5,         # Seconds of yawn for alert
                "alert_cooldown": 3.0          # Seconds between alerts
            },
            
            # === PERFORMANCE OPTIMIZATION ===
            "performance": {
                "max_fps": 30,
                "frame_skip": 1,               # Process every N frames
                "queue_size": 2,               # Frame buffer size
                "thread_priority": "normal"     # normal, high
            },
            
            # === MINIMAL UI PARAMETERS ===
            "ui": {
                "update_interval": 50,         # GUI update interval (ms)
                "display_confidence": True,
                "display_fps": True,
                "log_level": "INFO"
            },
            
            # === REMOVED LEGACY PARAMETERS ===
            # - eye_bias_factor (replaced by confidence_threshold)
            # - yawn_bias_factor (replaced by confidence_threshold) 
            # - min_eye_closure_time (simplified to eye_closure_time)
            # - min_yawn_duration (simplified to yawn_duration)
            # - Multiple redundant timing parameters
            # - Complex state tracking variables
        }
    
    def get_ml_config(self):
        """Get ML-specific configuration"""
        return self.config.get('ml_config', {})
    
    def get_detection_config(self):
        """Get detection thresholds"""
        return self.config.get('detection', {})
    
    def get_performance_config(self):
        """Get performance settings"""
        return self.config.get('performance', {})
    
    def get_ui_config(self):
        """Get UI settings"""
        return self.config.get('ui', {})
    
    def save_config(self):
        """Save current configuration"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logging.info(f"ML config saved to {self.config_path}")
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
    
    def update_parameter(self, section, key, value):
        """Update a specific parameter"""
        if section in self.config:
            self.config[section][key] = value
            self.save_config()
        else:
            logging.error(f"Unknown config section: {section}")
    
    def get_parameter(self, section, key, default=None):
        """Get a specific parameter"""
        return self.config.get(section, {}).get(key, default)
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n=== ML-OPTIMIZED CONFIGURATION SUMMARY ===")
        print(f"Total parameters: {self._count_parameters()}")
        print(f"ML Config: {len(self.config.get('ml_config', {}))}")
        print(f"Detection: {len(self.config.get('detection', {}))}")
        print(f"Performance: {len(self.config.get('performance', {}))}")
        print(f"UI: {len(self.config.get('ui', {}))}")
        print("\nKey ML Settings:")
        ml_config = self.get_ml_config()
        for key, value in ml_config.items():
            print(f"  {key}: {value}")
        print("\nKey Detection Settings:")
        det_config = self.get_detection_config()
        for key, value in det_config.items():
            print(f"  {key}: {value}")
    
    def _count_parameters(self):
        """Count total number of parameters"""
        count = 0
        for section in self.config.values():
            if isinstance(section, dict):
                count += len(section)
        return count

    def get_available_yolo_versions(self):
        """Get list of available YOLO model versions."""
        yolo_dir = "models/detect"
        if not os.path.exists(yolo_dir):
            return []
        
        versions = []
        for item in os.listdir(yolo_dir):
            version_path = os.path.join(yolo_dir, item)
            if os.path.isdir(version_path):
                model_file = os.path.join(version_path, 'best.pt')
                if os.path.exists(model_file):
                    versions.append(item)
        return versions
    
    def get_available_backbones(self):
        """Get list of available classification backbones."""
        cls_dir = "models/classification"
        if not os.path.exists(cls_dir):
            return []
        
        backbones = []
        for file in os.listdir(cls_dir):
            if file.endswith('.pth'):
                backbone = file[:-4]  # Remove .pth extension
                backbones.append(backbone)
        return backbones
    
    def update_setting(self, key, value):
        """Update a configuration setting and save to file."""
        # For backward compatibility with old GUI
        logging.info(f"Legacy setting update: {key}={value}")
        
        # Map old keys to new structure
        if key in ['yolo_model_version', 'yolo_version']:
            self.update_parameter('ml_config', 'detection_model_version', value)
        elif key in ['classification_backbone']:
            self.update_parameter('ml_config', 'classification_backbone', value)
        elif key in ['detection_model']:
            # Extract version from path
            if 'yolov10' in str(value):
                self.update_parameter('ml_config', 'detection_model_version', 'yolov10')
            elif 'yolov11' in str(value):
                self.update_parameter('ml_config', 'detection_model_version', 'yolov11')
        elif key in ['classification_model']:
            # Extract backbone from path
            backbone = str(value).replace('.pth', '').split('/')[-1]
            self.update_parameter('ml_config', 'classification_backbone', backbone)
        elif key == 'confidence_threshold':
            self.update_parameter('detection', 'confidence_threshold', value)
        elif key in ['device']:
            self.update_parameter('ml_config', 'device', value)
        elif key in ['eye_closure_time']:
            self.update_parameter('detection', 'eye_closure_time', value)
        elif key in ['yawn_duration']:
            self.update_parameter('detection', 'yawn_duration', value)
        else:
            logging.warning(f"Unknown legacy setting key: {key}, please use update_parameter() directly")
    
    def get_current_yolo_version(self):
        """Get current YOLO version."""
        return self.get_parameter('ml_config', 'detection_model_version', 'yolov11')
    
    def get_current_classification_backbone(self):
        """Get current classification backbone."""
        return self.get_parameter('ml_config', 'classification_backbone', 'efficientnet_b0')
    
    def get_setting(self, key, default=None):
        """Get setting value for backward compatibility."""
        if key in ['yolo_version', 'yolo_model_version']:
            return self.get_current_yolo_version()
        elif key in ['classification_backbone']:
            return self.get_current_classification_backbone()
        elif key == 'confidence_threshold':
            return self.get_parameter('detection', 'confidence_threshold', default)
        elif key == 'device':
            return self.get_parameter('ml_config', 'device', default)
        elif key == 'eye_closure_time':
            return self.get_parameter('detection', 'eye_closure_time', default)
        elif key == 'yawn_duration':
            return self.get_parameter('detection', 'yawn_duration', default)
        else:
            logging.warning(f"Unknown legacy setting key: {key}")
            return default
    
    def reset_to_defaults(self):
        """Reset configuration to default values."""
        self.config = self._get_ml_optimized_defaults()
        self.save_config()

    def get_model_paths(self):
        """Get actual model file paths from model names"""
        ml_config = self.get_ml_config()
        
        # Build detection model path
        detection_version = ml_config.get('detection_model_version', 'yolov11')
        detection_path = f"models/detect/{detection_version}/best.pt"
        
        # Build classification model path  
        backbone = ml_config.get('classification_backbone', 'efficientnet_b0')
        classification_path = f"models/classification/{backbone}.pth"
        
        return {
            'detection_model': detection_path,
            'classification_model': classification_path
        }
    
    def get_detection_model_path(self):
        """Get detection model path"""
        return self.get_model_paths()['detection_model']
    
    def get_classification_model_path(self):
        """Get classification model path"""
        return self.get_model_paths()['classification_model']
    
    def _auto_select_yolo_version(self):
        """Auto-select best available YOLO version (prefer latest)."""
        available = self.get_available_yolo_versions()
        if not available:
            return 'yolov11'  # fallback
        
        # Prefer yolov11 > yolov10
        if 'yolov11' in available:
            return 'yolov11'
        elif 'yolov10' in available:
            return 'yolov10'
        else:
            return available[0]  # Take first available
    
    def _auto_select_classification_backbone(self):
        """Auto-select best available classification backbone (prefer EfficientNet)."""
        available = self.get_available_backbones()
        if not available:
            return 'efficientnet_b0'  # fallback
        
        # Prefer efficientnet > mobilenet > vgg
        for preferred in ['efficientnet_b0', 'efficientnet_b1', 'mobilenet_v3_small', 'mobilenet_v2', 'vgg16']:
            if preferred in available:
                return preferred
        
        return available[0]  # Take first available
    
    def get_yolo_model_path(self):
        """Get YOLO model path (legacy compatibility)."""
        return self.get_detection_model_path()
    
    def get_cls_model_path_legacy(self):
        """Get classification model path (legacy compatibility)."""
        ml_config = self.get_ml_config()
        backbone = ml_config.get('classification_backbone', 'efficientnet_b0')
        return f"models/classification/{backbone}.pth"


