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

# Constants
EYE_CLASSES = ['Closed', 'Open']
MOUTH_CLASSES = ['no_yawn', 'yawn']
FRAME_SIZE = (640, 640)


class DriverMonitor:
    def __init__(self):
        """
        Initialize the DriverMonitor with configurations, models, and state variables.
        """
        # Initialize configuration and logging
        self.config_manager = ConfigManager() #tải cấu hình từ file config.json
        self.config = self.config_manager.config 
        setup_logging(log_level=logging.INFO) #ghi log để theo dõi hoạt động của hệ thống
        pygame.mixer.init() #khởi tạo phát âm thanh cảnh báo

        # Log device and configuration details
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {self.device}")
        logging.info(f"Loaded configuration: {self.config}")

        # Initialize models and state variables
        self._initialize_models()
        self._initialize_state_variables()

    def _initialize_models(self):
        """
        Load YOLO and VGG16 models for detection and classification.
        """
        try:
            self.yolo_model = load_yolo_model(self.config['yolo_model_path'])
            self.vgg_model = load_vgg16_model(self.config['vgg_model_path'])
            logging.info("Models initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing models: {str(e)}")
            raise

    def _initialize_state_variables(self):
        """
        Initialize state variables for monitoring and tracking.
        """
        self.cap = None
        self.is_monitoring = False  #theo dõi trạng thái giám sát
        self.alert_active = False   #trạng thái cảnh báo (phát âm thanh hay không)
        self.frame_count = 0  #đếm số khung hình đã xử lý
        self.fps = 0  #lưu giá trị fps (số khung hình trên giây) hiện tại
        self.last_time = time.time() #timestamp for fps calculation
        self.alert_history = []
        self.is_evaluating = False
        self.eval_start_time = None
        self.eval_duration = 300  # 5 minutes

        # Eye and yawn state tracking
        self.current_eye_state = "Open"
        self.current_yawn_state = "No Yawn"
        
        self.eye_closed_time = 0   #thời gian nhắm mắt liên tục
        self.eye_closure_start_time = None
        self.last_eye_closure_time = None

        self.yawn_start_time = None
        self.yawn_duration = 0     #thời gian há miệng liên tục
        self.last_yawn_time = None #lưu thời gian ngáp cuối cùng
        self.yawn_preliminary_start_time = None
        self.yawn_preliminary_duration = 0
        self.yawn_min_duration = 0.5

        # Detection timeout for resetting states
        self.last_detection_time = None
        self.detection_timeout = 1.0

        # Alert sound management
        self.last_alert_time = 0  # Thời gian của cảnh báo cuối cùng
        self.alert_cooldown = 5.0  # Thời gian chờ giữa các cảnh báo (giây)

    def start_monitoring(self):
        """
        Start monitoring using the camera.

        Returns:
            tuple: (success: bool, error_message: str or None)
        """
        try:
            logging.info(f"Attempting to open camera with device index: {self.config['capture_device']}")
            self.cap = cv2.VideoCapture(self.config['capture_device'])
            if not self.cap.isOpened():
                raise Exception("Cannot open camera")
            self.is_monitoring = True
            logging.info("Camera monitoring started successfully")
            return True, None
        except Exception as e:
            logging.error(f"Error starting camera: {str(e)}")
            return False, str(e)

    def start_monitoring_video(self, video_path):
        """
        Start monitoring using a video file.

        Args:
            video_path (str): Path to the video file.

        Returns:
            tuple: (success: bool, error_message: str or None)
        """
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
        """
        Stop monitoring and release resources.
        """
        self.is_monitoring = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logging.info("Monitoring stopped")

    #phương thức chính để đọc, xử lí, và hiện thị khung hình video
    def update_video(self):
        """
        Update video frame, process it, and return the processed image for display.

        Returns:
            tuple: (success: bool, imgtk: ImageTk.PhotoImage or error_message: str)
        """
        if not self.is_monitoring:
            logging.warning("Monitoring is not active")
            return False, "Monitoring is not active"

        try:
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Failed to read frame from camera")
                return False, "Failed to read frame"

            # Preprocess frame
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, FRAME_SIZE)

            # Update FPS
            self.frame_count += 1 #đếm số khung hình đã xử lý
            current_time = time.time() #lay thời gian hiện tại
            if current_time - self.last_time >= 1: #nếu đã qua 1 giây từ lần cập nhật fps cuối
                self.fps = self.frame_count #số lượng khung hình đã xử lý trong 1 giây là fps
                self.frame_count = 0 #reset lại để đếm lại từ đầu
                self.last_time = current_time #reset lại = thằng hiện tại để bắt đầu chu trình mới

            # Analyze frame
            start_time = time.time()
            annotated_frame = self.analyze_frame(frame)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)#chuyển màu từ BGR(của OpenCV) sang RGB
            img = Image.fromarray(frame_rgb) #chuyển đổi từ numpy array sang PIL Image
            imgtk = ImageTk.PhotoImage(image=img) #chuyển đổi từ PIL Image sang ImageTk để hiển thị trên GUI
            logging.info(f"Frame processed successfully, time: {time.time() - start_time:.3f}s") #ghi log thời gian xử lý khung hình
            return True, imgtk

        except KeyError as e:
            logging.error(f"Configuration key missing: {str(e)}. Please check config.json.")
            return False, str(e)
        except Exception as e:
            logging.error(f"Error updating video: {str(e)}")
            return False, str(e)

    # phương thức để đặt lại tất cả trạng thái và thời gian khi không phát hiện thấy mặt trong một khoảng thời gian nhất định
    def reset_states(self):
        """
        Reset all states and durations when no face is detected for too long.
        """

        #eye
        self.current_eye_state = "Open"
        self.eye_closed_time = 0
        self.eye_closure_start_time = None
        self.last_eye_closure_time = None

        #yawn
        self.current_yawn_state = "No Yawn"
        self.yawn_start_time = None
        self.yawn_duration = 0
        self.last_yawn_time = None
        self.yawn_preliminary_start_time = None
        self.yawn_preliminary_duration = 0
        logging.info("No face detected for too long, resetting all states and durations")

    # phân tích khung hình để phát hiện trạng thái mắt và miệng
    def analyze_frame(self, frame):
        """
        Analyze the frame to detect eye and mouth states using YOLO and VGG models.

        Args: #đầu vào là khung hình video (mảng numpy)
            frame (numpy.ndarray): Input frame to analyze.

        Returns: #dầu ra là khung hình đã được vẽ bounding box và trạng thái mắt, miệng (mảng numpy)
            numpy.ndarray: Annotated frame with detection results.
        """
        frame_start = time.time() # lấy thời gian hiện tại để bắt đầu đo thời gian xử lí khung hình
        annotated_frame = frame.copy() # tạo bản sao để vẽ lên, tránh làm thay đổi khung hình gốc
        results = self.yolo_model(annotated_frame) # dùng yolov10 đe phát hiện mắt và miệng trong khung hình
        eyes_detected, mouth_detected = False, False # khởi tạo biến để xem có phát hiện được mắt/miệng hay không

        # Extract bounding boxes for eyes and mouth
        # boxes: Trích xuất danh sách các bounding box từ results.
        # Mỗi box có định dạng (x1, y1, x2, y2, conf, cls), trong đó:
        # x1, y1, x2, y2: Tọa độ của bounding box.
        # conf: Độ tin cậy (confidence score) của phát hiện.
        # cls: Class ID (0 cho mắt, 1 cho miệng).
        # box[4]: Độ tin cậy (conf).
        # box[5]: Class ID (cls).
        # eyes: Lọc các bounding box có label == 0 (mắt), lưu tọa độ (x1, y1, x2, y2) và độ tin cậy conf.
        # mouths: Lọc các bounding box có label == 1 (miệng).
        boxes = [(box.tolist(), int(box[5]), box[4]) for result in results for box in result.boxes.data]
        eyes = [(x1, y1, x2, y2, conf) for (x1, y1, x2, y2, conf, cls), label, conf in boxes if label == 0]
        mouths = [(x1, y1, x2, y2, conf) for (x1, y1, x2, y2, conf, cls), label, conf in boxes if label == 1]

        # Sort by confidence and limit to top detections
        eyes = sorted(eyes, key=lambda x: x[4], reverse=True)[:2]
        mouths = sorted(mouths, key=lambda x: x[4], reverse=True)[:1]
        # sorted(eyes, key=lambda x: x[4], reverse=True): Sắp xếp danh sách eyes theo độ tin cậy (x[4]) từ cao đến thấp.
        # [:2]: Chỉ lấy tối đa 2 mắt có độ tin cậy cao nhất (vì một khuôn mặt thường có 2 mắt).
        # Tương tự, lấy 1 miệng có độ tin cậy cao nhất.

        # Process eyes
        for x1, y1, x2, y2, conf in eyes:
            if conf < self.config['confidence_threshold']: #độ tin cậy nhỏ hơn ngưỡng thì bỏ qua
                continue
            obj = annotated_frame[int(y1):int(y2), int(x1):int(x2)] #cắt vùng mắt từ khung hình
            if obj.size == 0: #nếu không có gì trong vùng cắt thì bỏ qua
                continue
            obj_tensor = preprocess_image(obj) #Chuyển vùng mắt thành tensor để đưa vào mô hình VGG16
            if obj_tensor is None:
                continue
            with torch.no_grad():
                pred = torch.softmax(self.vgg_model(obj_tensor)[0], dim=0) #dùng vgg dự đoán trạng thái mắt
            eyes_detected = self.process_eyes(pred, x1, y1, x2, y2, annotated_frame)
            # Xử lý kết quả dự đoán, cập nhật trạng thái mắt (self.current_eye_state), và vẽ thông tin lên annotated_frame.

        # Process mouth
        for x1, y1, x2, y2, conf in mouths:
            if conf < self.config['confidence_threshold']:
                continue
            obj = annotated_frame[int(y1):int(y2), int(x1):int(x2)]
            if obj.size == 0:
                continue
            obj_tensor = preprocess_image(obj)
            if obj_tensor is None:
                continue
            with torch.no_grad():
                pred = torch.softmax(self.vgg_model(obj_tensor)[0], dim=0)
            mouth_detected = self.process_mouth(pred, x1, y1, x2, y2, annotated_frame)

        # Reset states if no face is detected for too long
        current_time = time.time()
        if not eyes_detected and not mouth_detected:
            if self.last_detection_time is None:
                self.last_detection_time = current_time
            elif (current_time - self.last_detection_time) > self.detection_timeout:
                self.reset_states()
        else:
            self.last_detection_time = current_time

        # Log if no detections
        if not eyes_detected:
            logging.debug("No eyes detected, maintaining previous eye state")
        if not mouth_detected:
            logging.debug("No mouth detected, maintaining previous mouth state")

        # Log frame for evaluation if active
        if hasattr(self, 'evaluator') and self.is_evaluating:
            self.evaluator.log_frame(frame_start)

        #trả về khung hình đã được xử lý và vẽ lên
        return annotated_frame

    def process_eyes(self, pred, x1, y1, x2, y2, annotated_frame):
        """
        Process eye prediction from VGG16 and calculate eye closure duration.

        Args:
            pred (torch.Tensor): Prediction probabilities from VGG model.
            x1, y1, x2, y2 (float): Bounding box coordinates.
            annotated_frame (numpy.ndarray): Frame to annotate.

        Returns:
            bool: True if eyes were processed successfully.
        """
        # eye_probs = pred[:2]: Lấy xác suất của hai lớp đầu tiên (Open và Closed) từ đầu ra của VGG16.
        # Giả sử pred = [0.8, 0.2], nghĩa là 80% là Open, 20% là Closed.
        eye_probs = pred[:2]
        eye_probs = eye_probs / torch.sum(eye_probs) #chuẩn hóa xác suất về tổng = 1
        eye_idx = eye_probs.argmax().item() #lấy chỉ số lớp có xác suất cao nhất (0 hoặc 1)
        vgg_eye_state = EYE_CLASSES[eye_idx] #lấy tên lớp tương ứng với chỉ số (Open hoặc Closed)
        logging.info(f"Eyes VGG State: {vgg_eye_state}")

        current_time = time.time() #lấy thời gian hiện tại

        #nếu trạng thái mắt là close
        if vgg_eye_state == "Closed":
            if self.eye_closure_start_time is None: 
                self.eye_closure_start_time = current_time #nếu lần đầu phát hiện nhắm mắt thì lưu thời gian bắt đầu = current_time
            self.eye_closed_time = current_time - self.eye_closure_start_time #tính thời gian nhắm mắt liên tục = thời gian hiện tại - thời gian bắt đầu nhắm mắt

            #nếu thời gian nhắn mắt > ngưỡng thì cập nhật trạng thái mắt và thời gian nhắm mắt
            self.current_eye_state = "Closed" if self.eye_closed_time >= self.config[
                'eye_closure_threshold'] else "Open"
            self.last_eye_closure_time = current_time
            logging.info(f"Eye closure duration: {self.eye_closed_time:.4f}")

        #nếu trạng thái mắt là open
        else:
            if self.last_eye_closure_time and (current_time - self.last_eye_closure_time) > self.config[
                'yawn_grace_period']: #should rename to eye_grace_period
                self.eye_closure_start_time = None
                self.eye_closed_time = 0
                self.current_eye_state = "Open"
                self.last_eye_closure_time = None
                logging.info("Eye state reset due to grace period timeout")
            else:
                logging.debug("Within grace period, maintaining eye state")

        logging.info(f"Final Eye State: {self.current_eye_state}")
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Eyes: {self.current_eye_state}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return True

    def process_mouth(self, pred, x1, y1, x2, y2, annotated_frame):
        """
        Process mouth prediction and detect yawn.

        Args:
            pred (torch.Tensor): Prediction probabilities from VGG model.
            x1, y1, x2, y2 (float): Bounding box coordinates.
            annotated_frame (numpy.ndarray): Frame to annotate.

        Returns:
            bool: True if mouth was processed successfully.
        """
        mouth_probs = pred[2:]
        mouth_probs = mouth_probs / torch.sum(mouth_probs)
        yawn_confidence = mouth_probs[1].item()
        vgg_mouth_state = MOUTH_CLASSES[1] if yawn_confidence > self.config['yawn_confidence_threshold'] else \
        MOUTH_CLASSES[0]
        logging.info(f"Mouth VGG State: {vgg_mouth_state}, Yawn Confidence: {yawn_confidence:.4f}")

        # Calculate mouth aspect ratio
        mouth_width = x2 - x1
        mouth_height = y2 - y1
        mouth_aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        logging.info(f"Mouth aspect ratio: {mouth_aspect_ratio:.4f}")

        yawn_size_threshold_high = 0.7

        # Determine preliminary yawn state
        if mouth_height < 10:
            preliminary_yawn_state = "no_yawn"
            logging.info("Mouth height too small, setting preliminary yawn state to 'no_yawn'")
        else:
            preliminary_yawn_state = "no_yawn"
            if vgg_mouth_state == "yawn" and mouth_aspect_ratio > self.config['yawn_size_threshold']:
                preliminary_yawn_state = "yawn"
                logging.info("Yawn detected based on VGG state and aspect ratio")
            elif mouth_aspect_ratio > yawn_size_threshold_high:
                preliminary_yawn_state = "yawn"
                logging.info("Yawn detected based on high mouth aspect ratio")

        # Update yawn state and duration
        current_time = time.time()
        if preliminary_yawn_state == "yawn":
            if self.yawn_preliminary_start_time is None:
                self.yawn_preliminary_start_time = current_time
                logging.info("Preliminary yawn detection started")
            self.yawn_preliminary_duration = current_time - self.yawn_preliminary_start_time
            logging.info(f"Preliminary yawn duration: {self.yawn_preliminary_duration:.4f}")

            if self.yawn_preliminary_duration >= self.yawn_min_duration:
                if self.yawn_start_time is None:
                    self.yawn_start_time = current_time
                    logging.info("Yawn detection started")
                self.yawn_duration = current_time - self.yawn_start_time
                self.current_yawn_state = "Yawn" if self.yawn_duration >= self.config['yawn_threshold'] else "No Yawn"
                self.last_yawn_time = current_time
                logging.info(f"Yawn duration: {self.yawn_duration:.4f}")
            else:
                self.yawn_duration = 0
        else:
            if self.last_yawn_time and (current_time - self.last_yawn_time) > self.config['yawn_grace_period']:
                self.yawn_start_time = None
                self.yawn_duration = 0
                self.current_yawn_state = "No Yawn"
                self.last_yawn_time = None
                self.yawn_preliminary_start_time = None
                self.yawn_preliminary_duration = 0
                logging.info("Yawn state reset due to grace period timeout")
            else:
                logging.debug("Within grace period, maintaining yawn state")

        logging.info(f"Final Mouth State: {self.current_yawn_state}")
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Mouth: {self.current_yawn_state}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return True

    def update_state(self):
        """
        Update the state of the driver based on eye and yawn detection.

        Returns:
            tuple: (eye_state, yawn_state, eye_closed_time, yawn_duration, alert_message, alert_triggered)
        """
        eye_state = self.current_eye_state
        yawn_state = self.current_yawn_state
        alert_triggered = False
        alert_message = None

        # Check for eye closure alert
        if self.current_eye_state == "Closed" and self.eye_closed_time > self.config['eye_closure_threshold']:
            alert_message = f"⚠️ Eyes closed too long ({int(self.eye_closed_time)}s)!"
            alert_triggered = True
            logging.info("Eye closure alert triggered")
            self.save_alert(alert_message, yawn_duration=0.0)
        # Check for yawn alert
        elif self.current_yawn_state == "Yawn" and self.yawn_duration >= self.config['yawn_threshold']:
            alert_message = "⚠️ Yawn detected!"
            alert_triggered = True
            logging.info("Yawn alert triggered")
            self.save_alert(alert_message, yawn_duration=self.yawn_duration)
        else:
            alert_message = f"🚗 Status: Monitoring ({self.get_source_type()})"

        # Play alert sound if triggered
        if alert_triggered and self.config['sound_enabled']:
            logging.info("Attempting to play alert sound")
            self.play_alert_sound()

        return eye_state, yawn_state, self.eye_closed_time, self.yawn_duration, alert_message, alert_triggered

    def save_alert(self, message, yawn_duration=None):
        """
        Save alert information to a JSON file.

        Args:
            message (str): Alert message to save.
            yawn_duration (float, optional): Duration of the yawn, if applicable.
        """
        if not self.config['save_alerts']:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_data = {
            'timestamp': timestamp,
            'message': message,
            'eye_closed_time': self.eye_closed_time,
            'yawn_duration': yawn_duration if yawn_duration is not None else 0.0
        }
        self.alert_history.append(alert_data)

        try:
            with open('alerts/alert_history.json', 'w') as f:
                json.dump(self.alert_history, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving alert to JSON: {str(e)}")

    def play_alert_sound(self):
        """
        Play the alert sound in a separate thread with cooldown management.
        """
        current_time = time.time()

        # Kiểm tra nếu đã đủ thời gian cooldown kể từ cảnh báo cuối cùng
        if current_time - self.last_alert_time < self.alert_cooldown:
            logging.info("Alert sound skipped due to cooldown period")
            return

        # Kiểm tra nếu không có âm thanh nào đang phát
        if not self.alert_active:
            self.alert_active = True
            self.last_alert_time = current_time
            logging.info("Playing alert sound...")

            # Hàm chạy trong luồng để phát âm thanh và đặt lại trạng thái
            def play_and_reset():
                try:
                    play_alarm(self.config['alert_sound'])
                except Exception as e:
                    logging.error(f"Error playing alert sound: {str(e)}")
                finally:
                    self.alert_active = False
                    logging.info("Alert sound finished, alert_active reset to False")

            # Khởi động luồng để phát âm thanh
            threading.Thread(target=play_and_reset).start()
        else:
            logging.info("Alert sound skipped because another alert is active")

    def start_evaluation(self, video_path, ground_truth):
        """
        Start performance evaluation on a video.

        Args:
            video_path (str): Path to the video file.
            ground_truth (dict): Ground truth data for evaluation.

        Returns:
            tuple: (success: bool, error_message: str or None)
        """
        if not self.is_monitoring:
            success, error = self.start_monitoring_video(video_path)
            if not success:
                return False, error

        self.is_evaluating = True
        self.eval_start_time = time.time()
        self.evaluator = SystemPerformanceEvaluator(self)
        self.evaluator.ground_truth = ground_truth
        logging.info("Starting performance evaluation on video...")
        return True, None

    def evaluate_performance(self, video_path, ground_truth):
        """
        Evaluate system performance on a video.

        Args:
            video_path (str): Path to the video file.
            ground_truth (dict): Ground truth data for evaluation.

        Returns:
            tuple: (success: bool, error_message: str or None)
        """
        try:
            return self.start_evaluation(video_path, ground_truth)
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            self.stop_evaluation()
            return False, str(e)

    def stop_evaluation(self):
        """
        Stop evaluation and return statistics.

        Returns:
            tuple: (stats: dict, final_stats: dict or None)
        """
        self.is_evaluating = False
        stats = None
        final_stats = None

        if hasattr(self, 'evaluator'):
            stats = self.evaluator.finalize_evaluation()
            if self.evaluator.ground_truth:
                final_stats = self.evaluator.finalize_evaluation(self.evaluator.ground_truth)

        self.stop_monitoring()
        return stats, final_stats

    def get_fps(self):
        """
        Get the current FPS.

        Returns:
            int: Frames per second.
        """
        return self.fps

    def get_eye_state(self):
        """
        Get the current eye state.

        Returns:
            str: Current eye state ("Open" or "Closed").
        """
        return self.current_eye_state

    def get_yawn_state(self):
        """
        Get the current yawn state.

        Returns:
            str: Current yawn state ("No Yawn" or "Yawn").
        """
        return self.current_yawn_state

    def get_eye_closed_time(self):
        """
        Get the eye closure duration.

        Returns:
            float: Duration of eye closure in seconds.
        """
        return self.eye_closed_time

    def get_source_type(self):
        """
        Get the source type (Camera or Video).

        Returns:
            str: Source type ("Camera", "Video", or "Unknown").
        """
        if self.cap is None or not self.is_monitoring:
            return "Unknown"
        return "Camera" if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == 0 else "Video"