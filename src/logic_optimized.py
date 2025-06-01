"""
Simplified Driver Monitor - Tập trung vào orchestration
Tối ưu hóa cho ML pipeline với clean architecture
"""
import cv2
import time
import numpy as np
import threading
import logging
from PIL import Image, ImageTk
import winsound
from threading import Thread

from src.ml_engine import MLInferenceEngine
from src.camera_handler import CameraHandler
from src.drowsiness_detector import DrowsinessDetector
from src.ml_config import MLOptimizedConfig

class DriverMonitor:
    """
    Simplified Driver Monitor - chỉ orchestrate các components
    Tập trung vào ML pipeline và performance
    """
    
    def __init__(self, config_manager=None):
        # Configuration - streamlined for ML focus
        self.config = config_manager if config_manager else MLOptimizedConfig()
        
        # Core components - tách biệt concerns
        self.ml_engine = MLInferenceEngine()
        self.camera_handler = CameraHandler()
        
        # Simplified drowsiness detection with ML config
        detection_config = self.config.get_detection_config()
        self.drowsiness_detector = DrowsinessDetector(detection_config)
        
        # Processing state
        self.is_monitoring = False
        self.process_thread = None
        
        # Display cache
        self.display_frame = None
        self.display_image = None
        
        # Performance metrics
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        self.processing_time = 0.0
        
        # Simple alert system
        self.sound_file = "sound/eawr.wav"
        self.last_alert_time = 0
        self.alert_cooldown = 3.0  # 3 giây cooldown
        self.sound_loaded = self._check_sound_file()
        
        logging.info("DriverMonitor initialized with optimized ML pipeline")
    
    def process_frame(self, frame):
        """
        Process a single frame directly (for testing/manual processing)
        Returns annotated frame with alerts
        """
        if self.ml_engine.yolo_model is None or self.ml_engine.cls_model is None:
            # If models not loaded, return frame with error message
            cv2.putText(frame, "Models not loaded", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        # ML processing
        start_time = time.time()
        ml_results = self.ml_engine.process_frame(frame)
        self.processing_time = time.time() - start_time
        
        # Process ML results and add alerts
        annotated_frame = self._process_ml_results(frame, ml_results)
        
        # Update FPS calculation
        self._update_fps()
        
        return annotated_frame
    
    def load_models(self, yolo_path, cls_path, backbone='vgg16'):
        """Load ML models"""
        return self.ml_engine.load_models(yolo_path, cls_path, backbone)
    
    def start_camera(self, device_id=0):
        """Start camera monitoring"""
        try:
            self.camera_handler.start_camera(device_id)
            self._start_processing()
            logging.info(f"Camera monitoring started on device {device_id}")
        except Exception as e:
            logging.error(f"Failed to start camera: {e}")
            raise
    
    def start_video(self, video_path):
        """Start video monitoring"""
        try:
            self.camera_handler.start_video(video_path)
            self._start_processing()
            logging.info(f"Video monitoring started: {video_path}")
        except Exception as e:
            logging.error(f"Failed to start video: {e}")
            raise
    
    def _start_processing(self):
        """Start processing thread"""
        self.is_monitoring = True
        self.process_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.process_thread.start()
    
    def _processing_loop(self):
        """Main processing loop - optimized for ML performance"""
        # Check if models are loaded
        if self.ml_engine.yolo_model is None or self.ml_engine.cls_model is None:
            logging.error("Models not loaded, cannot start processing")
            self.is_monitoring = False
            return
            
        while self.is_monitoring:
            # Get frame from camera
            frame = self.camera_handler.get_frame(timeout=1.0)
            if frame is None:
                continue
            
            # ML processing
            start_time = time.time()
            ml_results = self.ml_engine.process_frame(frame)
            self.processing_time = time.time() - start_time
            
            # Process ML results
            annotated_frame = self._process_ml_results(frame, ml_results)
            
            # Update display cache
            self.display_frame = annotated_frame
            self.display_image = None  # Reset cache
            
            # Update FPS
            self._update_fps()
    
    def _process_ml_results(self, frame, ml_results):
        """Process ML results and generate alerts"""
        if ml_results['error']:
            cv2.putText(frame, f"ML Error: {ml_results['error']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        # Process eye detections
        for eye_result in ml_results['eye_states']:
            self.drowsiness_detector.process_eye_detection(
                eye_result['state'], eye_result['confidence']
            )
            self._draw_detection(frame, eye_result, 'eye')
        
        # Process yawn detections  
        for yawn_result in ml_results['yawn_states']:
            self.drowsiness_detector.process_yawn_detection(
                yawn_result['state'], yawn_result['confidence']
            )
            self._draw_detection(frame, yawn_result, 'mouth')
          # Check for drowsiness alerts
        self.drowsiness_detector.check_drowsiness_alerts()
        
        # Get alert status
        status = self.drowsiness_detector.get_status()
        has_alerts = len(status.get('alerts', [])) > 0
        
        # Trigger sound alert if needed
        if has_alerts:
            self._trigger_simple_alert()
            # Add red border to frame
            frame = self._add_red_border(frame)
        
        # Add minimal status overlay (no warning text)
        self._add_status_overlay(frame)
        
        return frame
    
    def _draw_detection(self, frame, detection_result, detection_type):
        """Draw detection results on frame"""
        x1, y1, x2, y2 = detection_result['bbox']
        state = detection_result['state']
        confidence = detection_result['confidence']
        
        # Color based on state
        if detection_type == 'eye':
            color = (0, 255, 0) if state == "Open" else (0, 0, 255)
        else:  # mouth
            color = (0, 255, 0) if state == "no_yawn" else (255, 0, 0)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)        
        # Draw label
        label = f"{detection_type}: {state} ({confidence:.2f})"
        cv2.putText(frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def _add_status_overlay(self, frame):
        """Clean video overlay - only simple status info, no warning text"""
        status = self.drowsiness_detector.get_status()
        
        # Simple transparent background for status info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, frame.shape[0] - 120), (300, frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Minimal status text at bottom
        y_pos = frame.shape[0] - 90
        
        # Eye state indicator
        eye_color = (0, 255, 0) if status['eye_state'] == "Open" else (0, 255, 255)
        cv2.putText(frame, f"Eye: {status['eye_state']}", (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 1)
        
        # Mouth state indicator
        y_pos += 20
        mouth_color = (0, 255, 0) if status['yawn_state'] == "Normal" else (0, 255, 255)
        cv2.putText(frame, f"Mouth: {status['yawn_state']}", (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, mouth_color, 1)
        
        # FPS info
        y_pos += 20
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
    
    def stop(self):
        """Stop monitoring"""
        self.is_monitoring = False
        
        # Stop processing thread
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        
        # Stop camera
        self.camera_handler.stop()
        
        logging.info("Driver monitoring stopped")
    
    def get_display_image(self):
        """Get processed image for GUI display"""
        if self.display_frame is not None and self.display_image is None:
            # Convert to PIL/tkinter format only when needed
            rgb_frame = cv2.cvtColor(self.display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            self.display_image = ImageTk.PhotoImage(pil_image)
        
        return self.display_image
    
    def get_fps(self):
        """Get current FPS"""
        return self.fps
    
    def get_ml_metrics(self):
        """Get ML performance metrics"""
        return {
            'inference_time': self.processing_time,
            'fps': self.fps,
            'model_info': self.ml_engine.get_model_info()
        }
    
    def get_detection_status(self):
        """Get current detection status for GUI"""
        return self.drowsiness_detector.get_status()
    
    # Properties for GUI compatibility
    @property
    def current_eye_state(self):
        return self.drowsiness_detector.current_eye_state
    
    @property
    def current_yawn_state(self):
        return self.drowsiness_detector.current_yawn_state
    
    @property
    def last_eye_confidence(self):
        return getattr(self, '_last_eye_confidence', 0.0)
    
    @property
    def last_yawn_confidence(self):
        return getattr(self, '_last_yawn_confidence', 0.0)
    
    @property
    def eye_closed_time(self):
        return self.drowsiness_detector.get_eye_closure_time()
    
    @property
    def current_alerts(self):
        return self.drowsiness_detector.current_alerts
    
    def _check_sound_file(self):
        """Kiểm tra file âm thanh có tồn tại không"""
        import os
        exists = os.path.exists(self.sound_file)
        if exists:
            logging.info(f"✅ Sound file found: {self.sound_file}")
        else:
            logging.warning(f"❌ Sound file not found: {self.sound_file}")
        return exists
    
    def _trigger_simple_alert(self):
        """Kích hoạt cảnh báo đơn giản - chỉ âm thanh"""
        current_time = time.time()
        
        # Kiểm tra cooldown
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False
        
        # Phát âm thanh trong thread riêng để không block
        def play_sound():
            try:
                if self.sound_loaded:
                    winsound.PlaySound(self.sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
                    logging.info("🔊 Alert sound played")
                else:
                    # Fallback: system beep
                    winsound.Beep(1000, 500)  # 1000Hz, 500ms
                    logging.info("🔊 System beep played")
            except Exception as e:
                logging.error(f"Sound play error: {e}")
        
        Thread(target=play_sound, daemon=True).start()
        self.last_alert_time = current_time
        return True
    
    def _add_red_border(self, frame, thickness=15):
        """Thêm viền đỏ nháy cho frame khi có alert"""
        # Flash effect - nháy 2 lần/giây
        flash_state = int(time.time() * 4) % 2
        
        if flash_state == 0:
            color = (0, 0, 255)  # Đỏ sáng BGR
        else:
            color = (0, 50, 200)  # Đỏ tối BGR
        
        # Vẽ viền đỏ
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w-1, h-1), color, thickness)
        cv2.rectangle(frame, (thickness//2, thickness//2), 
                     (w-thickness//2-1, h-thickness//2-1), color, thickness//2)
        
        return frame
