import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Path to YOLOv10 and VGG16 model weights
yolo_model_path = '../models/yolov10m/train/weights/best.pt'  # Change path if needed
vgg16_model_path = '../models/yawn/model_yawn.h5'  # Model for yawn detection
# vgg16_model_path = '../model_yawn.h5'

# Initialize YOLOv10 model
yolo_model = YOLO(yolo_model_path)

# Initialize VGG16 model for yawn detectionplugin
vgg16_model = load_model(vgg16_model_path, compile=False)

# Initialize camera
cap = cv2.VideoCapture(0)  # Use default computer camera

# Set camera direction (if needed)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Set smaller frame size to increase processing speed
frame_width, frame_height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't capture frame from camera.")
        break

    # Flip frame
    frame = cv2.flip(frame, 1)  # 1 is horizontal flip, 0 is vertical flip

    # Reduce image size to speed up processing
    small_frame = cv2.resize(frame, (320, 240))

    # Predict objects on the smaller frame
    results = yolo_model(small_frame, conf=0.5, iou=0.4)  # Use higher confidence and IoU threshold

    # Iterate through each result in results (list of results)
    for result in results:
        boxes = result.boxes  # List of bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            confidence = box.conf[0].item()  # Bounding box confidence
            class_id = box.cls[0].item()  # Object class ID

            # Convert coordinates to the original frame
            x1, y1, x2, y2 = int(x1 * frame_width / 320), int(y1 * frame_height / 240), int(x2 * frame_width / 320), int(y2 * frame_height / 240)

            if class_id == 1:  # Assuming class 1 is mouth
                # Cut mouth region from frame
                mouth_region = frame[y1:y2, x1:x2]

                # Prepare input data for VGG16
                mouth_region = cv2.resize(mouth_region, (224, 224))
                mouth_region = mouth_region.astype("float32") / 255.0  # Use float32 to increase speed
                mouth_region = img_to_array(mouth_region)
                mouth_region = np.expand_dims(mouth_region, axis=0)

                # Predict yawn state (yawn or no yawn)
                yawn_state = vgg16_model.predict(mouth_region, batch_size=1)[0]

                # Assign yawn state label
                yawn_label = "Yawning" if yawn_state[0] < 0.5 else "Not Yawning"

                # Set drawing color based on yawn state
                color = (0, 255, 0) if yawn_label == "Not Yawning" else (0, 0, 255)  # Green if not yawning, red if yawning

                # Draw rectangle and label for mouth
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{yawn_label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display frame from camera
    cv2.imshow('Camera', frame)

    # Stop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
