# src/models.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from ultralytics import YOLO
import warnings

# Suppress specific UserWarnings from torchvision to keep logs clean
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")


def _create_classifier_head(num_features, num_classes):
    """Create a classifier head with the specified number of features and classes."""
    return nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes)
    )

def _create_vgg16_model(num_classes):
    """Create a VGG16 model with custom classifier head."""
    model = models.vgg16(pretrained=True)
    for param in model.features[:28].parameters():
        param.requires_grad = False
    for param in model.features[28:].parameters():
        param.requires_grad = True
    num_features = model.classifier[6].in_features
    model.classifier[6] = _create_classifier_head(num_features, num_classes)
    return model

def _create_mobilenet_v2_model(num_classes):
    """Create a MobileNetV2 model with custom classifier head."""
    model = models.mobilenet_v2(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    num_features = model.classifier[1].in_features
    model.classifier[1] = _create_classifier_head(num_features, num_classes)
    return model

def _create_mobilenet_v3_small_model(num_classes):
    """Create a MobileNetV3 Small model with custom classifier head."""
    model = models.mobilenet_v3_small(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    num_features = model.classifier[3].in_features
    model.classifier[3] = _create_classifier_head(num_features, num_classes)
    return model

def _create_efficientnet_b0_model(num_classes):
    """Create an EfficientNet B0 model with custom classifier head."""
    model = models.efficientnet_b0(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    num_features = model.classifier[1].in_features
    model.classifier[1] = _create_classifier_head(num_features, num_classes)
    return model

def get_model(backbone_name, num_classes=4, class_weights=None):
    """
    Create a model with the specified backbone architecture.
    
    Args:
        backbone_name (str): Name of the backbone ('vgg16', 'mobilenet_v2', 'mobilenet_v3_small', 'efficientnet_b0')
        num_classes (int): Number of output classes (default: 4)
        class_weights (torch.Tensor): Class weights for loss function (optional)
    
    Returns:
        tuple: (model, criterion, optimizer, scheduler)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_creators = {
        'vgg16': _create_vgg16_model,
        'mobilenet_v2': _create_mobilenet_v2_model,
        'mobilenet_v3_small': _create_mobilenet_v3_small_model,
        'efficientnet_b0': _create_efficientnet_b0_model
    }
    
    if backbone_name not in model_creators:
        raise ValueError(f'Unsupported backbone: {backbone_name}')
    
    model = model_creators[backbone_name](num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    return model, criterion, optimizer, scheduler


def load_classification_model(model_path, backbone_name, num_classes=4):
    """
    Load a classification model from the specified path with given backbone.
    
    Args:
        model_path (str): Path to the model weights file.
        backbone_name (str): Name of the backbone architecture.
        num_classes (int): Number of output classes.
    
    Returns:
        torch.nn.Module: The loaded model instance.
    """
    print(f"Loading {backbone_name} model from {model_path}...")
    try:
        # Create model architecture
        model, _, _, _ = get_model(backbone_name, num_classes)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        # Move to device and set to eval mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        
        print(f"{backbone_name} model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading {backbone_name} model: {str(e)}")
        raise


def get_available_yolo_models():
    """Get list of available YOLO models.
    
    Returns:
        dict: Dictionary mapping model names to file paths
    """
    return {
        'yolov10': "models/detected/yolov10.pt",
        'yolov11': "models/detected/yolov11.pt"
    }

def load_yolo_model(model_path: str) -> YOLO:
    """Load a YOLO model from the specified path.

    Args:
        model_path (str): Path to the YOLO model weights file.

    Returns:
        YOLO: The loaded YOLO model instance, ready for inference.

    Raises:
        FileNotFoundError: If the model file at `model_path` does not exist.
        Exception: If there is an error loading the YOLO model.

    This function initializes a YOLO model using the ultralytics library and prints
    status messages to indicate successful loading.
    """
    print("Loading YOLO model...")
    try:
        model = YOLO(model_path)
        print("YOLO model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {str(e)}")
        raise

def load_yolo_model_by_version(version='yolov10'):
    """Load a YOLO model by version name.
    
    Args:
        version (str): YOLO version ('yolov10' or 'yolov11')
        
    Returns:
        YOLO: The loaded YOLO model instance
    """
    available_models = get_available_yolo_models()
    if version not in available_models:
        raise ValueError(f"Unsupported YOLO version: {version}. Available: {list(available_models.keys())}")
    
    model_path = available_models[version]
    return load_yolo_model(model_path)



