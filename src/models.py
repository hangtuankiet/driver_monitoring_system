# src/models.py
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO
import warnings

# Suppress specific UserWarnings from torchvision to keep logs clean
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")


class CustomVGG16(nn.Module):
    """A custom VGG16 model for classification tasks.

    This class modifies the standard VGG16 architecture by replacing the final
    fully connected layer to support a custom number of output classes. It is
    designed for tasks such as eye and mouth state classification in driver monitoring.

    Attributes:
        features (nn.Module): The convolutional feature extraction layers from VGG16.
        avgpool (nn.Module): The adaptive average pooling layer from VGG16.
        classifier (nn.Sequential): The modified classifier with a custom final layer.
    """

    def __init__(self, num_classes: int = 4) -> None:
        """Initialize the CustomVGG16 model.

        Args:
            num_classes (int, optional): Number of output classes for classification.
                Defaults to 4 (e.g., for eye states: open/closed, and mouth states: yawn/no_yawn).

        The base VGG16 model is loaded without pretrained weights, as weights will be
        loaded from a custom state dictionary later. The final classifier layer is replaced
        to match the specified number of classes.
        """
        super(CustomVGG16, self).__init__()
        base_model = models.vgg16(pretrained=False)  # Pretrained weights are not loaded here
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(
            *list(base_model.classifier.children())[:-1],  # Keep all classifier layers except the last
            nn.Sequential(
                nn.Linear(4096, 256),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(256, num_classes)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CustomVGG16 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes) containing
                the raw scores for each class.

        The input tensor is passed through the feature extraction layers, average pooling,
        flattening, and the modified classifier to produce class scores.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


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


def load_vgg16_model(model_path: str, num_classes: int = 4) -> CustomVGG16:
    """Load a CustomVGG16 model from the specified path.

    Args:
        model_path (str): Path to the VGG16 model weights file.
        num_classes (int, optional): Number of output classes for the model.
            Defaults to 4 (e.g., for eye and mouth state classification).

    Returns:
        CustomVGG16: The loaded CustomVGG16 model instance, moved to the appropriate
            device (CPU or GPU) and set to evaluation mode.

    Raises:
        FileNotFoundError: If the model file at `model_path` does not exist.
        Exception: If there is an error loading the VGG16 model weights.

    This function initializes a CustomVGG16 model, loads its weights from the specified
    path, moves it to the appropriate device (CUDA if available, otherwise CPU), and
    sets it to evaluation mode for inference.
    """
    print(f"Loading VGG16 model from {model_path}...")
    try:
        model = CustomVGG16(num_classes)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        print("VGG16 model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading VGG16 model: {str(e)}")
        raise