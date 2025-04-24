import cv2
import numpy as np
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
FRAME_SIZE = (640, 480)


class DriverMonitor:
    """
    Core component that manages driver drowsiness monitoring using computer vision.
    Handles detection and classification of eyes and mouth states to identify fatigue signs.
    """
    def __init__(self):
        """
        Initialize the DriverMonitor with configurations, models, and state variables.
        """
        # Initialize configuration and logging
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config
        setup_logging(log_level=logging.INFO)
        
        # Log device and configuration details
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {self.device}")
        logging.info(f"Loaded configuration: {self.config}")

        # Initialize models and state variables
        self._initialize_models()
        self._initialize_state_variables()
        
        # Initialize object tracking for better persistence
        self.last_eye_boxes = []
        self.last_mouth_boxes = []
        self.detection_persistence = 5
        self.eye_track_count = 0
        self.mouth_track_count = 0
        
        # Video synchronization variables
        self.video_start_time = None
        self.video_fps = 0
        self.frame_position = 0
        self.sync_video = True  # Flag to enable/disable video sync

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
        self.is_monitoring = False
        self.alert_active = False
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.alert_history = []
        self.is_evaluating = False
        self.eval_start_time = None
        self.eval_duration = 300  # 5 minutes

        # Eye and yawn state tracking
        self.current_eye_state = "Open"
        self.current_yawn_state = "No Yawn"
        self.eye_closed_time = 0
        self.eye_closure_start_time = None
        self.last_eye_closure_time = None
        self.yawn_start_time = None
        self.yawn_duration = 0
        self.last_yawn_time = None
        self.yawn_preliminary_start_time = None
        self.yawn_preliminary_duration = 0
        self.yawn_min_duration = 0.5

        # Detection timeout for resetting states
        self.last_detection_time = None
        self.detection_timeout = 1.0

        # Alert sound management
        self.last_alert_time = 0
        self.alert_cooldown = 5.0

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
            # For camera, no need to synchronize
            self.sync_video = False
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
            
            # Get video properties for synchronization
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_start_time = time.time()
            self.frame_position = 0
            self.sync_video = True
            
            logging.info(f"Video opened: {video_path}")
            logging.info(f"Video FPS: {self.video_fps}")
            logging.info(f"Total frames: {int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
            
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

    def update_video(self):
        """
        Update video frame, process it, and return the processed image for display.
        Maintains proper playback speed for video files.

        Returns:
            tuple: (success: bool, imgtk: ImageTk.PhotoImage or error_message: str)
        """
        if not self.is_monitoring:
            logging.warning("Monitoring is not active")
            return False, "Monitoring is not active"

        try:
            # For video files, synchronize playback speed
            if self.sync_video and self.get_source_type() == "Video":
                # Calculate what frame we should be on based on elapsed time
                elapsed_time = time.time() - self.video_start_time
                target_frame = int(elapsed_time * self.video_fps)
                
                # If processing is faster than real-time, wait to maintain correct speed
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current_frame < target_frame:
                    # We're ahead, continue to next frame
                    ret, frame = self.cap.read()
                    if not ret:
                        logging.warning("End of video reached")
                        return False, "End of video"
                elif current_frame > target_frame:
                    # We've fallen behind (processing too slow)
                    # Skip frames to catch up if we're very behind
                    if current_frame - target_frame > 10:
                        logging.warning(f"Video playback running slow. Skipping to catch up. " +
                                      f"Current: {current_frame}, Target: {target_frame}")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                        ret, frame = self.cap.read()
                        if not ret:
                            return False, "Failed to read frame after seeking"
                    else:
                        # We're not too far behind, just read the next frame
                        ret, frame = self.cap.read()
                        if not ret:
                            return False, "Failed to read frame"
                else:
                    # We're exactly where we should be
                    ret, frame = self.cap.read()
                    if not ret:
                        return False, "Failed to read frame"
            else:
                # For camera or when sync is disabled, just read the next frame
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("Failed to read frame from camera/video")
                    return False, "Failed to read frame"

            # Preprocess frame
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, FRAME_SIZE)

            # Update FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_time = current_time

            # Analyze frame
            start_time = time.time()
            annotated_frame = self.analyze_frame(frame)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Add playback position indicator for videos
            if self.get_source_type() == "Video":
                total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                progress = f"Frame: {int(current_frame)}/{int(total_frames)} " + \
                          f"({current_frame/total_frames*100:.1f}%)"
                logging.debug(progress)
            
            processing_time = time.time() - start_time
            logging.info(f"Frame processed successfully, time: {processing_time:.3f}s")
            return True, imgtk

        except KeyError as e:
            logging.error(f"Configuration key missing: {str(e)}. Please check config.json.")
            return False, str(e)
        except Exception as e:
            logging.error(f"Error updating video: {str(e)}")
            return False, str(e)

    def reset_states(self):
        """
        Reset all states and durations when no face is detected for too long.
        """
        self.current_eye_state = "Open"
        self.eye_closed_time = 0
        self.eye_closure_start_time = None
        self.last_eye_closure_time = None
        self.current_yawn_state = "No Yawn"
        self.yawn_start_time = None
        self.yawn_duration = 0
        self.last_yawn_time = None
        self.yawn_preliminary_start_time = None
        self.yawn_preliminary_duration = 0
        logging.info("No face detected for too long, resetting all states and durations")

    def analyze_frame(self, frame):
        """
        Analyze the frame to detect eye and mouth states using YOLO and VGG models.
        
        This is a key function that coordinates object detection with YOLOv10
        and classification with VGG16 to determine driver state.

        Args:
            frame (numpy.ndarray): Input frame to analyze.

        Returns:
            numpy.ndarray: Annotated frame with detection results.
        """
        frame_start = time.time()
        annotated_frame = frame.copy()
        
        # Run YOLO detection with reduced confidence for difficult poses
        detection_confidence = max(0.25, self.config['confidence_threshold'] - 0.2)
        results = self.yolo_model(annotated_frame, conf=detection_confidence)
        
        eyes_detected, mouth_detected = False, False

        # Extract bounding boxes for eyes and mouth
        boxes = [(box.tolist(), int(box[5]), box[4]) for result in results for box in result.boxes.data]
        eyes = [(x1, y1, x2, y2, conf) for (x1, y1, x2, y2, conf, cls), label, conf in boxes if label == 0]
        mouths = [(x1, y1, x2, y2, conf) for (x1, y1, x2, y2, conf, cls), label, conf in boxes if label == 1]

        # Sort by confidence and limit to top detections
        eyes = sorted(eyes, key=lambda x: x[4], reverse=True)[:2]
        mouths = sorted(mouths, key=lambda x: x[4], reverse=True)[:1]
        
        # Handle eye detection with tracking
        if eyes:
            self.last_eye_boxes = eyes
            self.eye_track_count = 0
        elif self.last_eye_boxes and self.eye_track_count < self.detection_persistence:
            logging.info(f"No eye detection, using tracking from previous frame ({self.eye_track_count}/{self.detection_persistence})")
            eyes = self.last_eye_boxes
            self.eye_track_count += 1
            
        # Handle mouth detection with tracking
        if mouths:
            self.last_mouth_boxes = mouths
            self.mouth_track_count = 0
        elif self.last_mouth_boxes and self.mouth_track_count < self.detection_persistence:
            logging.info(f"No mouth detection, using tracking from previous frame ({self.mouth_track_count}/{self.detection_persistence})")
            mouths = self.last_mouth_boxes
            self.mouth_track_count += 1

        # Process eyes
        for x1, y1, x2, y2, conf in eyes:
            obj = annotated_frame[int(y1):int(y2), int(x1):int(x2)]
            if obj.size == 0:
                continue
            obj_tensor = preprocess_image(obj)
            if obj_tensor is None:
                continue
            with torch.no_grad():
                pred = torch.softmax(self.vgg_model(obj_tensor)[0], dim=0)
            eyes_detected = self.process_eyes(pred, x1, y1, x2, y2, annotated_frame)

        # Process mouth
        for x1, y1, x2, y2, conf in mouths:
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

        # Add guidance text when no detections persist
        if not eyes_detected and self.eye_track_count >= self.detection_persistence:
            cv2.putText(annotated_frame, "Eyes not detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if not mouth_detected and self.mouth_track_count >= self.detection_persistence:
            cv2.putText(annotated_frame, "Mouth not detected", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Log if no detections
        if not eyes_detected:
            logging.debug("No eyes detected, maintaining previous eye state")
        if not mouth_detected:
            logging.debug("No mouth detected, maintaining previous mouth state")

        # Log frame for evaluation if active
        if hasattr(self, 'evaluator') and self.is_evaluating:
            self.evaluator.log_frame(frame_start)

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
        eye_probs = pred[:2]
        eye_probs = eye_probs / torch.sum(eye_probs)
        eye_idx = eye_probs.argmax().item()
        vgg_eye_state = EYE_CLASSES[eye_idx]
        logging.info(f"Eyes VGG State: {vgg_eye_state}")

        current_time = time.time()
        if vgg_eye_state == "Closed":
            if self.eye_closure_start_time is None:
                self.eye_closure_start_time = current_time
            self.eye_closed_time = current_time - self.eye_closure_start_time
            self.current_eye_state = "Closed" if self.eye_closed_time >= self.config[
                'eye_closure_threshold'] else "Open"
            self.last_eye_closure_time = current_time
            logging.info(f"Eye closure duration: {self.eye_closed_time:.4f}")
        else:
            if self.last_eye_closure_time and (current_time - self.last_eye_closure_time) > self.config[
                'yawn_grace_period']:
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
        
        Uses a combination of visual features and temporal consistency to accurately
        detect yawn events while minimizing false positives.

        Args:
            pred (torch.Tensor): Prediction probabilities from VGG model.
            x1, y1, x2, y2 (float): Bounding box coordinates.
            annotated_frame (numpy.ndarray): Frame to annotate.

        Returns:
            bool: True if mouth was processed successfully.
        """
        # Extract and normalize mouth probabilities
        mouth_probs = pred[2:]
        mouth_probs = mouth_probs / torch.sum(mouth_probs)
        yawn_confidence = mouth_probs[1].item()
        
        # Get VGG model's classification
        vgg_mouth_state = MOUTH_CLASSES[1] if yawn_confidence > self.config['yawn_confidence_threshold'] else MOUTH_CLASSES[0]
        logging.info(f"Mouth VGG State: {vgg_mouth_state}, Yawn Confidence: {yawn_confidence:.4f}")

        # Calculate mouth measurements
        mouth_width = x2 - x1
        mouth_height = y2 - y1
        mouth_area = mouth_width * mouth_height
        mouth_aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        logging.info(f"Mouth aspect ratio: {mouth_aspect_ratio:.4f}, Area: {mouth_area:.1f}px²")

        # Strict thresholds to reduce false positives in real-world camera usage
        strict_confidence_threshold = 0.92
        strict_aspect_threshold = 0.55
        very_high_aspect_threshold = 0.8
        
        # Temporal consistency check
        if not hasattr(self, 'mouth_state_history'):
            self.mouth_state_history = []
            self.consistency_counter = 0
            
        # Store the last 5 frames of mouth state predictions
        MAX_HISTORY = 5
        current_frame_state = False
        
        # Determine preliminary yawn state with stricter logic
        preliminary_yawn_state = "no_yawn"
        
        # Ignore very small detections (avoid processing noise)
        if mouth_height < 15 or mouth_width < 15 or mouth_area < 400:
            logging.info("Mouth detection too small, ignoring")
        
        # Primary detection: VGG confidence AND aspect ratio must both be high
        elif yawn_confidence > strict_confidence_threshold and mouth_aspect_ratio > strict_aspect_threshold:
            preliminary_yawn_state = "yawn"
            current_frame_state = True
            logging.info(f"Potential yawn detected based on strong VGG confidence and aspect ratio")
        
        # Secondary detection: extremely high aspect ratio (clear physiological yawn indicator)
        elif mouth_aspect_ratio > very_high_aspect_threshold:
            preliminary_yawn_state = "yawn"
            current_frame_state = True
            logging.info("Potential yawn detected based on very high mouth aspect ratio")
            
        # Update history buffer (using sliding window)
        self.mouth_state_history.append(current_frame_state)
        if len(self.mouth_state_history) > MAX_HISTORY:
            self.mouth_state_history.pop(0)
            
        # Calculate consistency score (how many recent frames show yawn)
        consistency_score = sum(self.mouth_state_history) / max(len(self.mouth_state_history), 1)
        
        # Only proceed if we have enough consistent frames showing a yawn
        if consistency_score >= 0.6 and preliminary_yawn_state == "yawn":
            self.consistency_counter += 1
            logging.info(f"Yawn consistency: {consistency_score:.2f}, counter: {self.consistency_counter}")
        else:
            self.consistency_counter = max(0, self.consistency_counter - 1)
            
        # Only consider yawns that demonstrate consistency 
        if self.consistency_counter >= 3:
            logging.info("Consistent yawn pattern detected")
        else:
            preliminary_yawn_state = "no_yawn"
            
        # Update yawn state and duration with temporal filtering
        current_time = time.time()
        self.yawn_min_duration = 0.8
        
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
                self.consistency_counter = 0
                logging.info("Yawn state reset due to grace period timeout")
            else:
                logging.debug("Within grace period, maintaining yawn state")

        # Add mouth state visualization with confidence
        color = (0, 0, 255)
        if self.current_yawn_state == "Yawn":
            color = (0, 165, 255)
            
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        confidence_text = f"{int(yawn_confidence * 100)}%"
        cv2.putText(annotated_frame, f"Mouth: {self.current_yawn_state} {confidence_text}", 
                   (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                   
        # Draw aspect ratio indicator
        ar_text = f"AR: {mouth_aspect_ratio:.2f}"
        cv2.putText(annotated_frame, ar_text, 
                   (int(x1), int(y2) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                   
        return True

    def update_state(self):
        """
        Update the state of the driver based on eye and yawn detection.
        Triggers alerts when drowsiness criteria are met.

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

        if current_time - self.last_alert_time < self.alert_cooldown:
            logging.info("Alert sound skipped due to cooldown period")
            return

        if not self.alert_active:
            self.alert_active = True
            self.last_alert_time = current_time
            logging.info("Playing alert sound...")

            def play_and_reset():
                try:
                    play_alarm(self.config['alert_sound'], volume=self.config.get('sound_volume', 1.0))
                except Exception as e:
                    logging.error(f"Error playing alert sound: {str(e)}")
                finally:
                    self.alert_active = False
                    logging.info("Alert sound finished, alert_active reset to False")

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
        Get the current eye closed time.
        
        Returns:
            float: Time in seconds that eyes have been closed.
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