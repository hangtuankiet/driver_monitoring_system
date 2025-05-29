"""
ML Inference Engine - Tối ưu cho Machine Learning
Tách biệt hoàn toàn logic ML khỏi camera/GUI handling
"""
import torch
import numpy as np
import cv2
import time
import logging
from src.models import load_yolo_model, load_multitask_model
from src.utils import preprocess_image

class MLInferenceEngine:
    """Pure ML inference engine - chỉ tập trung vào ML operations"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = None
        self.cls_model = None
        
        # ML metrics
        self.inference_time = 0.0
        self.last_inference_start = 0.0
        
        logging.info(f"MLInferenceEngine initialized on device: {self.device}")
    
    def load_models(self, yolo_path, cls_path, backbone='vgg16'):
        """Load ML models"""
        try:
            logging.info(f"Loading YOLO model: {yolo_path}")
            self.yolo_model = load_yolo_model(yolo_path)
            
            logging.info(f"Loading classification model: {cls_path}")
            self.cls_model = load_multitask_model(cls_path, backbone_name=backbone, device=self.device)
            
            # Warm up models
            self._warmup_models()
            
            logging.info("ML models loaded and warmed up successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to load ML models: {e}")
            return False
    
    def _warmup_models(self):
        """Warm up models với dummy input"""
        if self.yolo_model and self.cls_model:
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Warmup YOLO
            _ = self.yolo_model(dummy_frame)
            # Warmup classification model với dummy eye/mouth regions
            dummy_region = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            _ = self.classify_eye_state(dummy_region)
            _ = self.classify_yawn_state(dummy_region)
            logging.info("Models warmed up")
    
    def detect_objects(self, frame):
        """YOLO object detection - returns detections"""
        if not self.yolo_model:
            return []
        
        self.last_inference_start = time.time()
        
        try:
            results = self.yolo_model(frame)
            detections = []
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())
                        
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': float(confidence),
                            'class_id': cls_id,
                            'class_name': 'eye' if cls_id == 0 else 'mouth'
                        })
            
            self.inference_time = time.time() - self.last_inference_start
            return detections
            
        except Exception as e:
            logging.error(f"YOLO detection error: {e}")
            return []
    
    def classify_eye_state(self, eye_region):
        """Classify eye state (Open/Closed)"""
        if not self.cls_model or eye_region.size == 0:
            return "Unknown", 0.0
        
        try:
            processed_region = preprocess_image(eye_region)
            if processed_region.dim() == 3:
                processed_region = processed_region.unsqueeze(0)
            
            processed_region = processed_region.to(self.device)
            
            with torch.no_grad():
                eye_output, _ = self.cls_model(processed_region)
                eye_probs = torch.softmax(eye_output, dim=1)
                eye_confidence, eye_pred = torch.max(eye_probs, 1)
                
                eye_state = "Open" if eye_pred.item() == 0 else "Closed"
                confidence = eye_confidence.item()
                
                return eye_state, confidence
                
        except Exception as e:
            logging.error(f"Eye classification error: {e}")
            return "Unknown", 0.0
    
    def classify_yawn_state(self, mouth_region):
        """Classify yawn state (no_yawn/yawn)"""
        if not self.cls_model or mouth_region.size == 0:
            return "Unknown", 0.0
        
        try:
            processed_region = preprocess_image(mouth_region)
            if processed_region.dim() == 3:
                processed_region = processed_region.unsqueeze(0)
            
            processed_region = processed_region.to(self.device)
            
            with torch.no_grad():
                _, yawn_output = self.cls_model(processed_region)
                yawn_probs = torch.softmax(yawn_output, dim=1)
                yawn_confidence, yawn_pred = torch.max(yawn_probs, 1)
                
                yawn_state = "no_yawn" if yawn_pred.item() == 0 else "yawn"
                confidence = yawn_confidence.item()
                
                return yawn_state, confidence
                
        except Exception as e:
            logging.error(f"Yawn classification error: {e}")
            return "Unknown", 0.0
    
    def process_frame(self, frame):
        """Main ML pipeline - process một frame và return results"""
        if not self.yolo_model or not self.cls_model:
            return {
                'detections': [],
                'eye_states': [],
                'yawn_states': [],
                'inference_time': 0.0,
                'error': 'Models not loaded'
            }
        
        start_time = time.time()
        
        # YOLO detection
        detections = self.detect_objects(frame)
        
        # Classification cho từng detection
        eye_states = []
        yawn_states = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            region = frame[y1:y2, x1:x2]
            
            if detection['class_name'] == 'eye':
                eye_state, eye_conf = self.classify_eye_state(region)
                eye_states.append({
                    'state': eye_state,
                    'confidence': eye_conf,
                    'bbox': detection['bbox']
                })
            
            elif detection['class_name'] == 'mouth':
                yawn_state, yawn_conf = self.classify_yawn_state(region)
                yawn_states.append({
                    'state': yawn_state,
                    'confidence': yawn_conf,
                    'bbox': detection['bbox']
                })
        
        total_time = time.time() - start_time
        
        return {
            'detections': detections,
            'eye_states': eye_states,
            'yawn_states': yawn_states,
            'inference_time': total_time,
            'error': None
        }
    
    def get_model_info(self):
        """Get information about loaded models"""
        return {
            'yolo_loaded': self.yolo_model is not None,
            'cls_loaded': self.cls_model is not None,
            'device': str(self.device),
            'last_inference_time': self.inference_time
        }
