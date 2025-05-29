import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO
import warnings

# Suppress torchvision UserWarnings
warnings.filterwarnings("ignore", category=UserWarning,
                        module="torchvision.models._utils")

class MultiTaskNet(nn.Module):
    """
    A multi-task neural network with a shared backbone and two separate heads:
      - Eye head: classifies eye state (Open vs Closed)
      - Mouth head: classifies mouth state (No Yawn vs Yawn)

    Supported backbones:
      * 'vgg16'
      * 'mobilenet_v2'
      * 'mobilenet_v3_small'
      * 'efficientnet_b0'
    """
    def __init__(self, backbone_name: str):
        super().__init__()        # Initialize backbone with pretrained weights
        if backbone_name == 'vgg16':
            base = models.vgg16_bn(pretrained=True)
            feat_dim = 512
        elif backbone_name == 'mobilenet_v2':
            base = models.mobilenet_v2(pretrained=True)
            feat_dim = base.classifier[1].in_features
        elif backbone_name == 'mobilenet_v3_small':
            base = models.mobilenet_v3_small(pretrained=True)
            feat_dim = base.classifier[0].in_features
        elif backbone_name == 'efficientnet_b0':
            base = models.efficientnet_b0(pretrained=True)
            feat_dim = base.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")        # Shared feature extractor
        self.feature = base.features
        # Unfreeze the last few layers to allow fine-tuning
        # For deep networks like VGG, only unfreeze a few layers
        for i, param in enumerate(self.feature.parameters()):
            # Only make the last ~25% of feature layers trainable
            if i >= len(list(self.feature.parameters())) * 0.75:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Eye classification head (2 classes)
        self.eye_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2)
        )
        
        # Mouth classification head (2 classes)
        self.mouth_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the multi-task network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)

        Returns:
            eye_logits (torch.Tensor): (batch_size, 2)
            mouth_logits (torch.Tensor): (batch_size, 2)
        """
        feats = self.feature(x)
        pooled = self.pool(feats)
        eye_logits = self.eye_head(pooled)
        mouth_logits = self.mouth_head(pooled)
        return eye_logits, mouth_logits


def load_yolo_model(model_path: str) -> YOLO:
    """
    Load a YOLO detection model from the specified weights file.
    Args:
        model_path (str): Path to the YOLO .pt file

    Returns:
        YOLO: Initialized YOLO detector in inference mode
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}")


def load_multitask_model(
    model_path: str,
    backbone_name: str,
    device: torch.device = None
) -> MultiTaskNet:
    """
    Load a trained MultiTaskNet from a saved state dict.

    Args:
        model_path (str): Path to the .pth state dictionary
        backbone_name (str): One of supported backbones
        device (torch.device, optional): Device to load the model onto

    Returns:
        MultiTaskNet: Model ready for inference (eval mode)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiTaskNet(backbone_name)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# Example of dynamic model selection based on config:
# config = {'yolo_path': 'yolov10.pt', 'cls_path': 'mt_vgg16.pth', 'backbone': 'vgg16'}
# yolo = load_yolo_model(config['yolo_path'])
# cls_model = load_multitask_model(config['cls_path'], config['backbone'])
