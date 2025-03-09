import cv2
import numpy as np
import pygame
import threading
import time
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import logging
import json
from datetime import datetime
import os
from system_evaluator import SystemPerformanceEvaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, running on CPU.")

class DriverMonitoringLogic:
    def __init__(self):
        self.setup_logging()
        self.load_config()
        self.initialize_models()
        self.initialize_state_variables()

        self.sound_enabled = True
        self.alert_sound = None
        self.mixer_initialized = False

        self.is_evaluating = False
        self.eval_start_time = None
        self.eval_duration = 300
        self.current_eye_state = "Open"
        self.current_yawn_state = "No Yawn"
        self.eye_closed_time = 0
        self.start_time = None

    def setup_logging(self):
        logging.basicConfig(
            filename='../logs/driver_monitoring.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_config(self):
        try:
            with open('../json/config_torch.json', 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {
                'yolo_model_path': "../models/yolov10m/train/weights/best.pt",
                'eye_model_path': "../models/eye/vgg16_eye.pt",  # Đổi sang .pt
                'yawn_model_path': "../models/yawn/vgg16_yawn.pt",  # Đổi sang .pt
                'alert_sound': "../sound/eawr.wav",
                'eye_closure_threshold': 2.1,
                'capture_device': 0,
                'video_path': "../video/",
                'save_alerts': True,
                'sound_enabled': True,
                'sound_volume': 0.5
            }
            with open('../json/config_torch.json', 'w') as f:
                json.dump(self.config, f, indent=4)

    def initialize_models(self):
        try:
            print("Loading YOLO model...")
            self.yolo_model = YOLO(self.config['yolo_model_path'])
            print("YOLO model loaded successfully.")

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

            # Khởi tạo và tải mô hình VGG16 cho mắt
            print("Loading eye model...")
            self.eye_model = CustomVGG16(num_classes=2)
            self.eye_model.load_state_dict(torch.load(self.config['eye_model_path']))
            self.eye_model = self.eye_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.eye_model.eval()
            print("Eye model loaded successfully.")

            # Khởi tạo và tải mô hình VGG16 cho miệng
            print("Loading yawn model...")
            self.yawn_model = CustomVGG16(num_classes=2)
            self.yawn_model.load_state_dict(torch.load(self.config['yawn_model_path']))
            self.yawn_model = self.yawn_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.yawn_model.eval()
            print("Yawn model loaded successfully.")

            pygame.mixer.init()
            logging.info("Models initialized successfully")
        except FileNotFoundError as e:
            logging.error(f"File not found: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error initializing models: {str(e)}")
            raise

    def initialize_state_variables(self):
        self.cap = None
        self.is_monitoring = False
        self.alert_active = False
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.alert_history = []

    def create_storage_directories(self):
        os.makedirs('../alerts', exist_ok=True)
        os.makedirs('../logs', exist_ok=True)

    def start_monitoring(self):
        try:
            self.cap = cv2.VideoCapture(self.config['capture_device'])
            if not self.cap.isOpened():
                raise Exception("Cannot open camera")
            self.is_monitoring = True
            logging.info("Camera monitoring started")
            return True, None
        except Exception as e:
            logging.error(f"Error starting camera monitoring: {str(e)}")
            return False, str(e)

    def start_monitoring_video(self, video_path):
        try:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise Exception("Cannot open video file")
            self.is_monitoring = True
            logging.info("Video monitoring started")
            return True, None
        except Exception as e:
            logging.error(f"Error starting video monitoring: {str(e)}")
            return False, str(e)

    def stop_monitoring(self):
        self.is_monitoring = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logging.info("Monitoring stopped")

    def update_video(self):
        if not self.is_monitoring:
            return False, None
        try:
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Failed to read frame from camera")
                return False, None

            logging.debug(f"Frame read successfully: {frame.shape}")
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))
            self.frame_count += 1
            if time.time() - self.last_time > 1:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_time = time.time()

            # Phân tích và vẽ bounding box lên frame
            annotated_frame = self.analyze_frame(frame)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            return True, imgtk
        except Exception as e:
            logging.error(f"Error updating video: {str(e)}")
            return False, str(e)

    def preprocess_image(self, image):
        try:
            if image.size == 0:
                raise ValueError("Empty image")
            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            return image_tensor
        except Exception as e:
            logging.error(f"Error preprocessing image: {str(e)}")
            return None

    def analyze_frame(self, frame):
        frame_start = time.time()
        annotated_frame = frame.copy()
        results = self.yolo_model(annotated_frame)
        eye_state, yawn_state = "Open", "No Yawn"
        eyes_detected, mouth_detected = False, False

        logging.debug(f"YOLO results: {len(results)} detections")
        for result in results:
            boxes = result.boxes.data
            logging.debug(f"Number of boxes detected: {len(boxes)}")
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box.tolist()
                logging.debug(f"Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf}, cls={cls}")
                if conf < 0.3:
                    logging.debug(f"Skipping detection with low confidence: {conf}")
                    continue
                label = int(cls)
                obj = annotated_frame[int(y1):int(y2), int(x1):int(x2)]
                if obj.size == 0:
                    logging.warning("Detected object is empty")
                    continue
                try:
                    if label == 0:  # Eyes
                        eyes_detected = True
                        obj_tensor = self.preprocess_image(obj)
                        if obj_tensor is not None:
                            with torch.no_grad():
                                pred = self.eye_model(obj_tensor)[0]
                                pred = torch.softmax(pred, dim=0)
                            eye_state = "Open" if pred[0] > 0.5 else "Closed"
                            logging.info(f"Eye detection: {eye_state}, confidence={pred[0]:.4f}")
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"Eyes: {eye_state}", (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    elif label == 1:  # Mouth
                        mouth_detected = True
                        obj_tensor = self.preprocess_image(obj)
                        if obj_tensor is not None:
                            with torch.no_grad():
                                pred = self.yawn_model(obj_tensor)[0]
                                pred = torch.softmax(pred, dim=0)
                            yawn_state = "No Yawn" if pred[0] > 0.5 else "Yawn"
                            logging.info(f"Yawn detection: {yawn_state}, confidence={pred[0]:.4f}")
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(annotated_frame, f"Mouth: {yawn_state}", (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                except Exception as e:
                    logging.warning(f"Error processing detection: {str(e)}")
                    continue

        self.current_eye_state = eye_state if eyes_detected else self.current_eye_state
        self.current_yawn_state = yawn_state if mouth_detected else self.current_yawn_state
        self.update_state()
        if hasattr(self, 'evaluator') and self.is_evaluating:
            self.evaluator.log_frame(frame_start)

        return annotated_frame

    def update_state(self):
        if self.current_eye_state == "Closed":
            if self.start_time is None:
                self.start_time = time.time()
            self.eye_closed_time = time.time() - self.start_time
        else:
            self.eye_closed_time = 0
            self.start_time = None

        alert_triggered = False
        alert_message = ""
        if self.eye_closed_time > self.config['eye_closure_threshold']:
            alert_message = f"⚠️ Eyes closed too long ({int(self.eye_closed_time)}s)!"
            alert_triggered = True
        elif self.current_yawn_state == "Yawn":
            alert_message = "⚠️ Yawn detected!"
            alert_triggered = True

        if alert_triggered:
            self.trigger_alert(alert_message)
        else:
            self.clear_alert()

        logging.debug(f"State updated: eye={self.current_eye_state}, yawn={self.current_yawn_state}, eye_closed_time={self.eye_closed_time}, alert={alert_triggered}")
        return self.current_eye_state, self.current_yawn_state, self.eye_closed_time, alert_message if self.alert_active else "🚗 Status: Normal", self.alert_active

    def trigger_alert(self, message):
        if not self.alert_active:
            threading.Thread(target=self.play_alarm).start()
            self.save_alert(message)
            if self.is_evaluating:
                timestamp = time.time() - self.eval_start_time
                print(f"Alert triggered at {timestamp:.2f}s: {message}")
        self.alert_active = True

    def clear_alert(self):
        self.alert_active = False

    def play_alarm(self):
        try:
            pygame.mixer.music.load(self.config['alert_sound'])
            pygame.mixer.music.play()
        except Exception as e:
            logging.error(f"Error playing alarm: {str(e)}")

    def save_alert(self, message):
        if not self.config['save_alerts']:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_data = {
            'timestamp': timestamp,
            'message': message,
            'eye_closed_time': self.eye_closed_time
        }
        self.alert_history.append(alert_data)
        try:
            with open('../alerts/alert_history.json', 'w') as f:
                json.dump(self.alert_history, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving alert: {str(e)}")

    def start_evaluation(self, video_path, ground_truth):
        if not self.is_monitoring:
            success, error = self.start_monitoring_video(video_path)
            if not success:
                return False, error
        self.is_evaluating = True
        self.eval_start_time = time.time()
        self.evaluator = SystemPerformanceEvaluator(self)
        self.evaluator.ground_truth = ground_truth
        print("Starting performance evaluation on video...")
        return True, None

    def evaluate_performance(self, video_path, ground_truth):
        try:
            success, error = self.start_evaluation(video_path, ground_truth)
            return success, error
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            self.stop_evaluation()
            return False, str(e)

    def stop_evaluation(self):
        self.is_evaluating = False
        start_time = time.time()
        stats = self.evaluator.finalize_evaluation()
        print(f"Time to compute stats: {time.time() - start_time:.2f}s")
        print("Initial results (without ground truth):")
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")

        final_stats = None
        if hasattr(self.evaluator, 'ground_truth') and self.evaluator.ground_truth:
            final_stats = self.evaluator.finalize_evaluation(self.evaluator.ground_truth)
            print("Results with ground truth from video:")
            for key, value in final_stats.items():
                print(f"{key}: {value:.4f}")

        self.stop_monitoring()
        return stats, final_stats

    def get_fps(self):
        return self.fps

    def get_eye_state(self):
        return self.current_eye_state

    def get_yawn_state(self):
        return self.current_yawn_state

    def get_eye_closed_time(self):
        return self.eye_closed_time

    def get_alert_message(self):
        return f"⚠️ Eyes closed too long ({int(self.eye_closed_time)}s)!" if self.eye_closed_time > self.config['eye_closure_threshold'] else "⚠️ Yawn detected!"

    def is_alert_active(self):
        return self.alert_active

    def get_source_type(self):
        if self.cap is None or not self.is_monitoring:
            return "Unknown"
        return "Camera" if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == 0 else "Video"

# Khởi tạo và chạy (ví dụ)
if __name__ == "__main__":
    monitor = DriverMonitoringLogic()
    monitor.create_storage_directories()
    success, error = monitor.start_monitoring()
    if success:
        while monitor.is_monitoring:
            success, frame = monitor.update_video()
            if not success:
                print(f"Error: {frame}")
                break
            # Ở đây frame là ImageTk.PhotoImage, cần xử lý thêm nếu dùng GUI
            cv2.imshow('Camera', cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                monitor.stop_monitoring()
                break
    else:
        print(f"Failed to start monitoring: {error}")
    cv2.destroyAllWindows()