import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO

class CustomVGG16(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomVGG16, self).__init__()
        base_model = models.vgg16(pretrained=False)
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_yolo_model(model_path):
    print("Loading YOLO model...")
    model = YOLO(model_path)
    print("YOLO model loaded successfully.")
    return model

def load_vgg16_model(model_path, num_classes=2):
    print(f"Loading VGG16 model from {model_path}...")
    model = CustomVGG16(num_classes)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print("VGG16 model loaded successfully.")
    return model