"""
Drowsiness Detection Logic - ML-Optimized Version
Simplified configuration focusing on ML performance
"""
import time
import logging
from src.utils import play_alarm

class DrowsinessDetector:
    """ML-optimized drowsiness detection with simplified parameters"""
    
    def __init__(self, config):
        # Simplified ML-focused configuration
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.eye_closure_time = config.get('eye_closure_time', 2.0)
        self.yawn_duration = config.get('yawn_duration', 1.5)
        self.alert_cooldown = config.get('alert_cooldown', 3.0)
        
        # State tracking
        self.eye_closure_start = None
        self.yawn_start_time = None
        self.last_alert_time = 0
        
        # Current states
        self.current_eye_state = "Unknown"
        self.current_yawn_state = "Unknown"
        self.current_alerts = []
        
        logging.info("DrowsinessDetector initialized with ML-optimized config")
        logging.info("  Confidence threshold: %.2f", self.confidence_threshold)
        logging.info("  Eye closure time: %.1fs", self.eye_closure_time)
        logging.info("  Yawn duration: %.1fs", self.yawn_duration)
    
    def process_eye_detection(self, eye_state, confidence):
        """Simplified eye detection processing"""
        # Direct threshold-based decision using ML config
        final_state = "Closed" if (eye_state == "Closed" and confidence > self.confidence_threshold) else "Open"
        
        current_time = time.time()
        
        # Track eye closure duration
        if final_state == "Closed":
            if self.eye_closure_start is None:
                self.eye_closure_start = current_time
        else:
            self.eye_closure_start = None
        
        self.current_eye_state = final_state
        return final_state, confidence
    
    def process_yawn_detection(self, yawn_state, confidence):
        """Simplified yawn detection processing"""
        # Direct threshold-based decision using ML config
        final_state = "yawn" if (yawn_state == "yawn" and confidence > self.confidence_threshold) else "no_yawn"
        
        current_time = time.time()
        
        # Track yawn duration
        if final_state == "yawn":
            if self.yawn_start_time is None:
                self.yawn_start_time = current_time
        else:
            self.yawn_start_time = None
        
        self.current_yawn_state = final_state
        return final_state, confidence
    
    def check_drowsiness_alerts(self):
        """Check for drowsiness conditions and generate alerts"""
        current_time = time.time()
        alerts = []
        
        # Check cooldown
        if (current_time - self.last_alert_time) < self.alert_cooldown:
            return alerts
        
        # Check prolonged eye closure
        if self.eye_closure_start is not None:
            closure_duration = current_time - self.eye_closure_start
            if closure_duration >= self.eye_closure_time:
                alerts.append("Eyes closed for %.1fs - DROWSINESS ALERT!" % closure_duration)
                self.last_alert_time = current_time
        
        # Check prolonged yawning
        if self.yawn_start_time is not None:
            yawn_duration = current_time - self.yawn_start_time
            if yawn_duration >= self.yawn_duration:
                alerts.append("Yawning for %.1fs - FATIGUE ALERT!" % yawn_duration)
                self.last_alert_time = current_time
        
        # Play alarm if alerts detected
        if alerts:
            try:
                play_alarm("sound/eawr.wav")
            except Exception as e:
                logging.warning("Failed to play alarm: %s", str(e))
        
        self.current_alerts = alerts
        return alerts
    
    def get_eye_closure_time(self):
        """Get current eye closure duration"""
        if self.eye_closure_start is None:
            return 0.0
        return time.time() - self.eye_closure_start
    
    def get_yawn_duration(self):
        """Get current yawn duration"""
        if self.yawn_start_time is None:
            return 0.0
        return time.time() - self.yawn_start_time
    
    def get_status(self):
        """Get current detection status"""
        return {
            'eye_state': self.current_eye_state,
            'yawn_state': self.current_yawn_state,
            'eye_closure_time': self.get_eye_closure_time(),
            'yawn_duration': self.get_yawn_duration(),
            'alerts': self.current_alerts.copy()
        }
