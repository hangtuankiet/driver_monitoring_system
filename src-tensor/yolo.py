import cv2
from ultralytics import YOLO

# Đường dẫn tới file weights của YOLOv10
# model_path = "models/yolov10s/train/weights/best.pt"
model_path = "../models/yolov10n/train/weight/best.pt"

# Khởi tạo mô hình YOLOv10
model = YOLO(model_path)
# model = YOLOv10(load_model(model_path, weights_only=True))

# Khởi tạo camera
cap = cv2.VideoCapture(0)  # Số 0 để sử dụng camera mặc định của máy tính

# Chỉnh hướng camera (nếu cần)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't capture frame from camera.")
        break

    # Lật frame
    frame = cv2.flip(frame, 1)  # 1 là lật theo chiều ngang, 0 là lật theo chiều dọc

    # Dự đoán đối tượng trên khung hình
    results = model(frame)

    # Lặp qua từng kết quả trong results (danh sách các kết quả)
    for result in results:
        boxes = result.boxes  # Danh sách các bounding box
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ bounding box
            confidence = box.conf[0].item()  # Độ tin cậy của bounding box
            class_id = box.cls[0].item()  # ID của lớp đối tượng

            label = f'{model.names[int(class_id)]} {confidence:.2f}'

            # Vẽ khung hình chữ nhật
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Vẽ nhãn
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị khung hình từ camera
    cv2.imshow('Camera', frame)

    # Dừng khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
