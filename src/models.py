# src/models.py
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

class CustomVGG16(nn.Module):
    def __init__(self, num_classes=4):  # Số lớp là 4 như trong code training
        super(CustomVGG16, self).__init__()
        base_model = models.vgg16(pretrained=False)  # Không load pretrained ở đây vì sẽ load state_dict sau
        self.features = base_model.features
        self.avgpool = base_model.avgpool  # Thêm avgpool bị thiếu trong code trước
        self.classifier = nn.Sequential(
            *list(base_model.classifier.children())[:-1],  # Lấy tất cả tầng classifier trừ tầng cuối
            nn.Sequential(  # Thay tầng cuối giống code training
                nn.Linear(4096, 256),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(256, num_classes)
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def load_yolo_model(model_path):
    print("Loading YOLO model...")
    model = YOLO(model_path)
    print("YOLO model loaded successfully.")
    return model

def load_vgg16_model(model_path, num_classes=4):
    print(f"Loading VGG16 model from {model_path}...")
    model = CustomVGG16(num_classes)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print("VGG16 model loaded successfully.")
    return model