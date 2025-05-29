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
        
        logging.info("DriverMonitor initialized with optimized ML pipeline")
    
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
        
        # Add status overlay
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
        """Add status information overlay with enhanced RED alerts"""
        status = self.drowsiness_detector.get_status()
        
        # Check if there are active alerts for red warning overlay
        has_alerts = len(status['alerts']) > 0
        
        # ENHANCED RED WARNING OVERLAY when alerts are active
        if has_alerts:
            # DOUBLE THICK RED BORDER - flash effect
            border_thickness = 20  # Increased from 10
            flash_time = int(time.time() * 3) % 2  # Flash every 0.33 seconds
            border_color = (0, 0, 255) if flash_time == 0 else (0, 50, 255)  # Bright to dark red flash
            
            # Outer red border
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), 
                         border_color, border_thickness)
            # Inner red border for extra thickness
            cv2.rectangle(frame, (border_thickness//2, border_thickness//2), 
                         (frame.shape[1]-border_thickness//2-1, frame.shape[0]-border_thickness//2-1), 
                         (0, 0, 255), border_thickness//2)
            
            # BRIGHT RED OVERLAY at top - increased coverage
            red_overlay = frame.copy()
            cv2.rectangle(red_overlay, (0, 0), (frame.shape[1], 120), (0, 0, 255), -1)
            cv2.addWeighted(red_overlay, 0.6, frame, 0.4, 0, frame)  # More intense red
            
            # LARGE FLASHING WARNING TEXT at top center
            warning_text = "🚨 NGUY HIỂM - TÀI XẾ BỊ NGỦ GẬT 🚨"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 4)[0]
            text_x = max(5, (frame.shape[1] - text_size[0]) // 2)
            
            # White outline for visibility
            cv2.putText(frame, warning_text, (text_x, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 6)
            # Red text on top
            cv2.putText(frame, warning_text, (text_x, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # SECOND LINE WARNING
            warning_text2 = "⚠️ DROWSINESS DETECTED - WAKE UP! ⚠️"
            text_size2 = cv2.getTextSize(warning_text2, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)[0]
            text_x2 = max(5, (frame.shape[1] - text_size2[0]) // 2)
            cv2.putText(frame, warning_text2, (text_x2, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 5)
            cv2.putText(frame, warning_text2, (text_x2, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # RED SIDE PANELS for extra visibility
            cv2.rectangle(frame, (0, 120), (100, frame.shape[0]), (0, 0, 255), -1)
            cv2.rectangle(frame, (frame.shape[1]-100, 120), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
        
        # Status background with RED tint when alerts active
        bg_color = (50, 0, 0) if has_alerts else (0, 0, 0)  # Dark red when alerts
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 140), (380, 260), bg_color, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Status text with ENHANCED color coding
        y_pos = 160
        eye_color = (0, 0, 255) if status['eye_state'] == "Closed" else (0, 255, 0)
        eye_thickness = 3 if status['eye_state'] == "Closed" else 2  # Thicker when closed
        cv2.putText(frame, f"👁️ Eye: {status['eye_state']}", (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, eye_thickness)
        
        y_pos += 30
        mouth_color = (0, 0, 255) if status['yawn_state'] == "Yawning" else (0, 255, 0)
        mouth_thickness = 3 if status['yawn_state'] == "Yawning" else 2  # Thicker when yawning
        cv2.putText(frame, f"👄 Mouth: {status['yawn_state']}", (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouth_color, mouth_thickness)
        
        y_pos += 30
        cv2.putText(frame, f"📊 FPS: {self.fps:.1f}", (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_pos += 25
        cv2.putText(frame, f"⚡ ML Time: {self.processing_time*1000:.1f}ms", (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ENHANCED BRIGHT RED alert display with pulsing effect
        if status['alerts']:
            for i, alert in enumerate(status['alerts'][:3]):  # Max 3 alerts
                y_pos = 300 + i * 40  # Moved down to avoid overlap
                
                # PULSING RED background with flash effect
                pulse_intensity = 0.3 + 0.4 * abs(np.sin(time.time() * 4))  # Pulse 4x per second
                alert_bg = frame.copy()
                cv2.rectangle(alert_bg, (5, y_pos-30), (frame.shape[1]-5, y_pos+15), (0, 0, 255), -1)
                cv2.addWeighted(alert_bg, pulse_intensity, frame, 1-pulse_intensity, 0, frame)
                
                # THICK WHITE border around alert
                cv2.rectangle(frame, (5, y_pos-30), (frame.shape[1]-5, y_pos+15), (255, 255, 255), 3)
                
                # Alert text with THICK white outline and red fill
                alert_text = f"🚨🚨 {alert.upper()} 🚨🚨"
                cv2.putText(frame, alert_text, (15, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 5)  # Thick white outline
                cv2.putText(frame, alert_text, (15, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)      # Red text
    
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
