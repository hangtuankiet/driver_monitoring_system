import cv2
import numpy as np
import pygame
import threading
import time
from PIL import Image, ImageTk
import torch
import logging
import json
from datetime import datetime
import os
from .models import load_yolo_model, load_vgg16_model
from .config import ConfigManager
from .utils import setup_logging, preprocess_image, play_alarm, create_storage_directories
from .evaluator import SystemPerformanceEvaluator

class DriverMonitor:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config
        setup_logging()
        pygame.mixer.init()
        self.initialize_models()
        self.initialize_state_variables()

    def initialize_models(self):
        try:
            self.yolo_model = load_yolo_model(self.config['yolo_model_path'])
            self.eye_model = load_vgg16_model(self.config['eye_model_path'])
            self.yawn_model = load_vgg16_model(self.config['yawn_model_path'])
            logging.info("Models initialized successfully")
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
        self.is_evaluating = False
        self.eval_start_time = None
        self.eval_duration = 300
        self.current_eye_state = "Open"
        self.current_yawn_state = "No Yawn"
        self.eye_closed_time = 0
        self.start_time = None
        self.sound_enabled = self.config['sound_enabled']

    def start_monitoring(self):
        try:
            self.cap = cv2.VideoCapture(self.config['capture_device'])
            if not self.cap.isOpened():
                raise Exception("Cannot open camera")
            self.is_monitoring = True
            logging.info("Camera monitoring started")
            return True, None
        except Exception as e:
            logging.error(f"Error starting camera: {str(e)}")
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
            logging.error(f"Error starting video: {str(e)}")
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
                logging.warning("Failed to read frame")
                return False, None
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))
            self.frame_count += 1
            if time.time() - self.last_time > 1:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_time = time.time()
            annotated_frame = self.analyze_frame(frame)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            return True, imgtk
        except Exception as e:
            logging.error(f"Error updating video: {str(e)}")
            return False, str(e)

    def analyze_frame(self, frame):
        frame_start = time.time()
        annotated_frame = frame.copy()
        results = self.yolo_model(annotated_frame)
        eyes_detected, mouth_detected = False, False

        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                if conf < 0.3:
                    continue
                label = int(cls)
                obj = annotated_frame[int(y1):int(y2), int(x1):int(x2)]
                if obj.size == 0:
                    continue
                obj_tensor = preprocess_image(obj)
                if obj_tensor is None:
                    continue
                with torch.no_grad():
                    if label == 0:  # Eyes
                        eyes_detected = True
                        pred = torch.softmax(self.eye_model(obj_tensor)[0], dim=0)
                        eye_state = "Open" if pred[0] > 0.5 else "Closed"
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"Eyes: {eye_state}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    elif label == 1:  # Mouth
                        mouth_detected = True
                        pred = torch.softmax(self.yawn_model(obj_tensor)[0], dim=0)
                        yawn_state = "No Yawn" if pred[0] > 0.5 else "Yawn"
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, f"Mouth: {yawn_state}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
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
            self.alert_active = False

        return self.current_eye_state, self.current_yawn_state, self.eye_closed_time, alert_message if self.alert_active else "🚗 Status: Normal", self.alert_active

    def trigger_alert(self, message):
        if not self.alert_active and self.sound_enabled:
            threading.Thread(target=play_alarm, args=(self.config['alert_sound'],)).start()
            self.save_alert(message)
            if self.is_evaluating:
                timestamp = time.time() - self.eval_start_time
                print(f"Alert triggered at {timestamp:.2f}s: {message}")
        self.alert_active = True

    def save_alert(self, message):
        if not self.config['save_alerts']:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_data = {'timestamp': timestamp, 'message': message, 'eye_closed_time': self.eye_closed_time}
        self.alert_history.append(alert_data)
        with open('alerts/alert_history.json', 'w') as f:
            json.dump(self.alert_history, f, indent=4)

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
            return self.start_evaluation(video_path, ground_truth)
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            self.stop_evaluation()
            return False, str(e)

    def stop_evaluation(self):
        self.is_evaluating = False
        stats = self.evaluator.finalize_evaluation()
        final_stats = None
        if hasattr(self.evaluator, 'ground_truth') and self.evaluator.ground_truth:
            final_stats = self.evaluator.finalize_evaluation(self.evaluator.ground_truth)
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

    def get_source_type(self):
        if self.cap is None or not self.is_monitoring:
            return "Unknown"
        return "Camera" if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == 0 else "Video"