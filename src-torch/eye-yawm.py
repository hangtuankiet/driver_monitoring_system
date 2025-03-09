import cv2
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

# Đường dẫn tới file weights
yolo_model_path = '../models/yolov10m/train/weights/best.pt'  # YOLO model phát hiện mắt và miệng
vgg16_eye_path = '../models/eye/eye.pt'  # VGG16 model phân loại mắt (Open/Closed)
vgg16_yawn_path = '../models/yawn/vgg16_yawn.pt'  # VGG16 model phân loại miệng (Yawning/Not Yawning)

# Khởi tạo mô hình YOLO
yolo_model = YOLO(yolo_model_path)

# Định nghĩa mô hình VGG16 tùy chỉnh
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

# Khởi tạo và tải mô hình VGG16 cho mắt và miệng
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mô hình cho mắt
vgg16_eye = CustomVGG16(num_classes=2)
vgg16_eye.load_state_dict(torch.load(vgg16_eye_path))
vgg16_eye = vgg16_eye.to(device)
vgg16_eye.eval()

# Mô hình cho miệng
vgg16_yawn = CustomVGG16(num_classes=2)
vgg16_yawn.load_state_dict(torch.load(vgg16_yawn_path))
vgg16_yawn = vgg16_yawn.to(device)
vgg16_yawn.eval()

# Biến đổi ảnh cho VGG16
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Khởi tạo camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
frame_width, frame_height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't capture frame from camera.")
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Reduce image size to speed up processing
    small_frame = cv2.resize(frame, (320, 240))

    # Predict objects on the smaller frame
    results = yolo_model(small_frame, conf=0.5, iou=0.4)

    # Iterate through each result
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = box.cls[0].item()

            # Convert coordinates to the original frame
            x1, y1, x2, y2 = int(x1 * frame_width / 320), int(y1 * frame_height / 240), \
                            int(x2 * frame_width / 320), int(y2 * frame_height / 240)

            # Phát hiện mắt (class_id = 0)
            if class_id == 0:  # Assuming class 0 is eye
                eye_region = frame[y1:y2, x1:x2]
                eye_region = preprocess(eye_region).unsqueeze(0).to(device)

                with torch.no_grad():
                    eye_state = vgg16_eye(eye_region)[0]
                    eye_state = torch.softmax(eye_state, dim=0)

                eye_label = "Closed" if eye_state[1] >= 0.5 else "Open"
                print(f"Eye state probs: Closed: {eye_state[0]:.4f}, Open: {eye_state[1]:.4f}")

                color = (0, 255, 0) if eye_label == "Open" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{eye_label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Phát hiện miệng (class_id = 1)
            elif class_id == 1:  # Assuming class 1 is mouth
                mouth_region = frame[y1:y2, x1:x2]
                mouth_region = preprocess(mouth_region).unsqueeze(0).to(device)

                with torch.no_grad():
                    yawn_state = vgg16_yawn(mouth_region)[0]
                    yawn_state = torch.softmax(yawn_state, dim=0)

                yawn_label = "Yawning" if yawn_state[0] < 0.5 else "Not Yawning"
                print(f"Yawn state probs: Yawning: {yawn_state[0]:.4f}, Not Yawning: {yawn_state[1]:.4f}")

                color = (0, 255, 0) if yawn_label == "Not Yawning" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{yawn_label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display frame from camera
    cv2.imshow('Camera', frame)

    # Stop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()