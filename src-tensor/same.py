# import tensorflow as tf
# import torch
#
# # Kiểm tra TensorFlow
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     print(f"TensorFlow GPUs available: {gpus}")
# else:
#     print("No GPU detected in TensorFlow.")
#
# # Kiểm tra PyTorch
# print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
# print(f"PyTorch GPU Count: {torch.cuda.device_count()}")
# import tensorflow as tf
#
# print("TensorFlow Version:", tf.__version__)
# gpus = tf.config.list_physical_devices('GPU')
#
# if gpus:
#     print(f"TensorFlow detected {len(gpus)} GPU(s):")
#     for gpu in gpus:
#         print(f"- {gpu}")
# else:
#     print("No GPU detected in TensorFlow.")



import cv2

# Đường dẫn tới video sẵn có
input_video_path = "../video/P1042756_720.mp4"  # Thay bằng đường dẫn video của bạn

# Đường dẫn để lưu video đã cắt
output_video_path = "../video/P1042756_720.avi"

# Mở video đầu vào
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Cannot open input video")
    exit()

# Lấy thông tin video gốc
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Original video size: {original_width}x{original_height}, FPS: {fps}")

# Tạo video writer với kích thước 640x480
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec video (XVID phổ biến cho AVI)
out = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 480))

# Xử lý từng frame
print("Resizing video to 640x480...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Finished processing video")
        break

    # Resize frame về 640x480
    resized_frame = cv2.resize(frame, (640, 480))

    # Ghi frame đã resize vào video đầu ra
    out.write(resized_frame)

    # (Tùy chọn) Hiển thị frame để kiểm tra
    # cv2.imshow('Resized Frame', resized_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video resized and saved as {output_video_path}")