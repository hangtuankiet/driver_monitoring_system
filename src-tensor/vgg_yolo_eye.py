import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Đường dẫn đến mô hình
yolo_model_path = "../models/yolov10n/train/weight/best.pt"
vgg16_model_path = "../models/eye/model_eye.h5"

# Load mô hình
yolo_model = YOLO(yolo_model_path)
vgg16_model = load_model(vgg16_model_path, compile=False)  # Thêm compile=False để tránh lỗi optimizer

# Khởi tạo sound
video_path = "sound/P1042787_720.mp4"
cap = cv2.VideoCapture(0)

# Thiết lập kích thước khung hình
frame_width, frame_height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'XVID'))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Không đọc được frame. sound có thể đã kết thúc.")
        break

    # Lật khung hình theo chiều ngang (nếu cần)
    frame = cv2.flip(frame, 1)

    # Resize frame nhỏ hơn để YOLO chạy nhanh hơn
    small_frame = cv2.resize(frame, (320, 240))

    # Chạy mô hình YOLO trên khung hình nhỏ
    results = yolo_model(small_frame, conf=0.5, iou=0.4)

    # Xử lý kết quả YOLO
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ bounding box
            class_id = int(box.cls[0].item())  # ID lớp đối tượng

            # Scale lại tọa độ về kích thước khung hình gốc
            x1, y1, x2, y2 = (
                int(x1 * frame_width / 320),
                int(y1 * frame_height / 240),
                int(x2 * frame_width / 320),
                int(y2 * frame_height / 240),
            )

            # Nếu phát hiện mắt
            if class_id == 0:
                eye_region = frame[y1:y2, x1:x2]

                # Kiểm tra vùng mắt có hợp lệ không
                if eye_region.size > 0:
                    eye_region = cv2.resize(eye_region, (224, 224))
                    eye_region = eye_region.astype("float32") / 255.0
                    eye_region = img_to_array(eye_region)
                    eye_region = np.expand_dims(eye_region, axis=0)

                    # Dự đoán trạng thái mắt
                    eye_state = vgg16_model.predict(eye_region, batch_size=1)[0]
                    eye_label = "Open" if eye_state[0] >= 0.5 else "Closed"
                    color = (0, 255, 0) if eye_label == "Open" else (0, 0, 255)

                    # Vẽ bounding box và trạng thái mắt
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{eye_label}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Hiển thị khung hình
    cv2.imshow('Camera', frame)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
