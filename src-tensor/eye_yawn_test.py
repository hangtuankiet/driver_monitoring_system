import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Đường dẫn tới file weights của YOLOv10 và VGG16
yolo_model_path = "../models/yolov10n/train/weight/best.pt"
vgg16_eye_model_path = "../models/eye/model_eye.h5"
vgg16_yawn_model_path = "../models/yawn/model_yawn.h5"

# Khởi tạo mô hình YOLOv10
yolo_model = YOLO(yolo_model_path)

# Khởi tạo mô hình VGG16 cho nhận diện trạng thái mắt và ngáp
vgg16_eye_model = load_model(vgg16_eye_model_path, compile=False)
vgg16_yawn_model = load_model(vgg16_yawn_model_path, compile=False)

# Khởi tạo camera
cap = cv2.VideoCapture(0)  # Số 0 để sử dụng camera mặc định của máy tính

# Chỉnh hướng camera (nếu cần)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Đặt kích thước khung hình nhỏ hơn để tăng tốc độ xử lý
frame_width, frame_height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't capture frame from camera.")
        break

    # Lật frame
    frame = cv2.flip(frame, 1)  # 1 là lật theo chiều ngang, 0 là lật theo chiều dọc

    # Giảm kích thước ảnh để tăng tốc độ xử lý
    small_frame = cv2.resize(frame, (320, 240))

    # Dự đoán đối tượng trên khung hình nhỏ hơn
    results = yolo_model(small_frame, conf=0.5, iou=0.4)  # Sử dụng ngưỡng confidence và IoU cao hơn

    # Lặp qua từng kết quả trong results (danh sách các kết quả)
    for result in results:
        boxes = result.boxes  # Danh sách các bounding box
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ bounding box
            confidence = box.conf[0].item()  # Độ tin cậy của bounding box
            class_id = box.cls[0].item()  # ID của lớp đối tượng

            # Chuyển đổi tọa độ về khung hình gốc
            x1, y1, x2, y2 = (
                int(x1 * frame_width / 320),
                int(y1 * frame_height / 240),
                int(x2 * frame_width / 320),
                int(y2 * frame_height / 240),
            )

            if class_id == 0:  # Giả sử lớp 0 là mắt
                # Cắt vùng mắt từ khung hình
                eye_region = frame[y1:y2, x1:x2]

                # Chuẩn bị dữ liệu đầu vào cho VGG16
                eye_region = cv2.resize(eye_region, (224, 224))
                eye_region = eye_region.astype("float32") / 255.0  # Sử dụng float32 để tăng tốc độ
                eye_region = img_to_array(eye_region)
                eye_region = np.expand_dims(eye_region, axis=0)

                # Dự đoán trạng thái mắt (mở hoặc nhắm)
                eye_state = vgg16_eye_model.predict(eye_region, batch_size=1)[0]

                # Gán nhãn trạng thái mắt
                eye_label = "Open" if eye_state[0] >= 0.5 else "Closed"

                # Đặt màu vẽ dựa trên trạng thái mắt
                color = (0, 255, 0) if eye_label == "Open" else (0, 0, 255)  # Xanh nếu mở, đỏ nếu đóng

                # Vẽ khung hình chữ nhật và nhãn cho mắt
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{eye_label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            elif class_id == 1:  # Giả sử lớp 1 là miệng
                # Cắt vùng miệng từ khung hình
                mouth_region = frame[y1:y2, x1:x2]

                # Chuẩn bị dữ liệu đầu vào cho VGG16
                mouth_region = cv2.resize(mouth_region, (224, 224))
                mouth_region = mouth_region.astype("float32") / 255.0  # Sử dụng float32 để tăng tốc độ
                mouth_region = img_to_array(mouth_region)
                mouth_region = np.expand_dims(mouth_region, axis=0)

                # Dự đoán trạng thái ngáp (ngáp hoặc không ngáp)
                yawn_state = vgg16_yawn_model.predict(mouth_region, batch_size=1)[0]

                # Gán nhãn trạng thái ngáp
                yawn_label = "Not Yawning" if yawn_state[0] >= 0.5 else "Yawning"

                # Đặt màu vẽ dựa trên trạng thái ngáp
                color = (255, 0, 0) if yawn_label == "Yawning" else (255, 255, 0)  # Đỏ nếu ngáp, vàng nếu không

                # Vẽ khung hình chữ nhật và nhãn cho miệng
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{yawn_label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Hiển thị khung hình từ camera
    cv2.imshow('Camera', frame)

    # Dừng khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
