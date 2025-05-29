"""
Camera/Video Handler - Tách biệt camera logic khỏi ML
"""
import cv2
import threading
import queue
import time
import logging

FRAME_SIZE = (640, 480)

class CameraHandler:
    """Chuyên xử lý camera/video input"""
    
    def __init__(self):
        self.cap = None
        self.is_active = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.read_thread = None
        self.consecutive_failures = 0
        self.max_failures = 10
        self.video_fps = 30  # Default FPS
        self.is_video_mode = False
        
    def start_camera(self, device_id=0):
        """Start camera with optimized settings"""
        logging.info(f"Starting camera {device_id}...")
        
        # Try DirectShow first (Windows optimization)
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logging.warning("DirectShow failed, trying default backend...")
            self.cap = cv2.VideoCapture(device_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {device_id}")
        
        # Configure camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.is_video_mode = False  # Camera mode
        
        # Camera warm-up
        for _ in range(3):
            ret, _ = self.cap.read()
            if ret:
                break
            time.sleep(0.1)
        
        self.is_active = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        
        logging.info("Camera started successfully")
    
    def start_video(self, video_path):
        """Start video file with correct FPS handling"""
        logging.info(f"Starting video: {video_path}")
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Get video properties for playback timing
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        video_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        # Validate FPS
        if self.video_fps <= 0 or self.video_fps > 120:
            logging.warning(f"Invalid FPS detected ({self.video_fps}), using default 30 FPS")
            self.video_fps = 30  # fallback to 30 FPS
        
        # Log video info for debugging
        video_info = {
            "path": video_path,
            "fps": self.video_fps,
            "original_resolution": f"{int(video_width)}x{int(video_height)}",
            "processed_resolution": f"{FRAME_SIZE[0]}x{FRAME_SIZE[1]}",
            "frames": int(total_frames),
            "duration": f"{total_frames/self.video_fps:.2f} seconds"
        }
        logging.info(f"Video info: {video_info}")
        
        # Check if resize will be needed
        if int(video_width) != FRAME_SIZE[0] or int(video_height) != FRAME_SIZE[1]:
            logging.info(f"Video will be resized from {int(video_width)}x{int(video_height)} to {FRAME_SIZE[0]}x{FRAME_SIZE[1]} for optimal performance")
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.is_video_mode = True  # Video mode
        
        self.is_active = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        
        logging.info("Video started successfully")
    
    def _read_loop(self):
        """Thread function to read frames with adaptive timing control"""
        self.consecutive_failures = 0
        
        # Calculate frame delay for video synchronization
        frame_delay = 1.0 / self.video_fps if self.is_video_mode else 0
        last_frame_time = time.time()
        frame_count = 0
        
        # For adaptive timing
        processing_times = []
        max_processing_times = 10  # Keep track of last N processing times
        
        logging.info(f"Starting frame reading loop (target delay={frame_delay:.3f}s per frame)")
        
        while self.is_active and self.cap and self.cap.isOpened():
            # Get frame start time for FPS control
            loop_start_time = time.time()
            
            # Read frame from camera/video
            ret, frame = self.cap.read()
            
            # If frame read is successful
            if ret:
                # Resize frame to standard size for optimal performance
                if frame.shape[1] != FRAME_SIZE[0] or frame.shape[0] != FRAME_SIZE[1]:
                    frame = cv2.resize(frame, FRAME_SIZE)
                
                # Put frame in queue for processing
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                    frame_count += 1
                    
                    # Log FPS every 100 frames
                    if frame_count % 100 == 0:
                        actual_fps = 100 / (time.time() - last_frame_time)
                        last_frame_time = time.time()
                        logging.info(f"Camera reading at {actual_fps:.1f} FPS")
                        
                self.consecutive_failures = 0
            else:
                # Handle failure or end of video
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_failures:
                    logging.warning("Too many consecutive frame read failures, stopping")
                    break
                
                if self.is_video_mode:
                    logging.info("End of video reached")
                    break
            
            # Adaptive timing control for video playback
            if self.is_video_mode and ret:
                # Calculate how long processing took
                processing_time = time.time() - loop_start_time
                
                # Keep track of processing times for adaptive timing
                processing_times.append(processing_time)
                if len(processing_times) > max_processing_times:
                    processing_times.pop(0)
                
                # Calculate average processing time
                avg_processing_time = sum(processing_times) / len(processing_times)
                
                # Adjust sleep time based on processing overhead
                adjusted_delay = max(0, frame_delay - avg_processing_time)
                
                if adjusted_delay > 0:
                    time.sleep(adjusted_delay)
        
        # Clean up when loop exits
        self.is_active = False
        logging.info("Frame reading loop stopped")
    
    def get_frame(self, timeout=1.0):
        """Get latest frame"""
        if not self.is_active:
            return None
        
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_video_info(self):
        """Get video information"""
        if not self.cap or not self.is_video_mode:
            return None
        
        return {
            'fps': self.video_fps,
            'frame_count': self.cap.get(cv2.CAP_PROP_FRAME_COUNT),
            'duration': self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.video_fps if self.video_fps > 0 else 0,
            'current_frame': self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        }
    
    def stop(self):
        """Stop camera/video"""
        self.is_active = False
        
        if self.read_thread:
            self.read_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logging.info("Camera/video stopped")
    
    def is_running(self):
        """Check if camera/video is running"""
        return self.is_active and self.cap and self.cap.isOpened()
