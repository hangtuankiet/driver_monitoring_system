import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import time
import logging
import os
import threading
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.logic_optimized import DriverMonitor
from src.ml_config import MLOptimizedConfig

class DriverMonitoringGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Smart Driver Monitoring System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        # Configuration manager
        self.config_manager = MLOptimizedConfig()
        
        # DriverMonitor instance (logic)
        self.logic = DriverMonitor()
        self.loaded = False

        # Available models
        self.available_yolo_versions = []
        self.available_backbones = []
        self.refresh_available_models()

        # Create dummy entry variables for backward compatibility
        self.yolo_entry = None
        self.cls_entry = None

        # Real-time monitoring variables
        self.eye_state_history = []
        self.mouth_state_history = []
        self.max_history = 50
        self.is_monitoring = False
        
        # Current detection states
        self.current_eye_confidence = 0.0
        self.current_mouth_confidence = 0.0
        self.current_eye_state = "Unknown"
        self.current_mouth_state = "Unknown"
        
        # RED ALERT SYSTEM - Enhanced for visibility
        self.alert_active = False
        self.alert_flash_state = False
        self.original_bg_color = '#f0f0f0'
        self.last_alert_time = 0
        self.alert_sound_enabled = True
        self.current_drowsiness_level = 0.0

        # Alert state for GUI
        self.alert_active = False
        self.alert_flash_time = 0
        
        # GUI alert colors
        self.normal_bg = '#f0f0f0'
        self.alert_bg = '#ff4444'

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_display()
        self.update_detection_indicators()

    def create_widgets(self):
        # Create main container with notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main monitoring tab
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="🎥 Monitoring")
        
        # Configuration tab
        self.config_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.config_tab, text="⚙️ Configuration")
        
        # Analytics tab
        self.analytics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analytics_tab, text="📊 Analytics")
        
        self.create_main_tab()
        self.create_config_tab()
        self.create_analytics_tab()

    def create_main_tab(self):
        # Main container
        main_container = ttk.Frame(self.main_tab)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for video and controls
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
          # Video display with dynamic border for alerts
        self.video_frame = ttk.LabelFrame(left_panel, text="📹 Live Feed", padding=10)
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a frame inside video_frame that can change color for alerts
        self.video_container = tk.Frame(self.video_frame, bg='#f0f0f0', relief='solid', bd=2)
        self.video_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.video_label = ttk.Label(self.video_container, text="🎬 No video feed\nClick 'Start Camera' or 'Open Video' to begin", 
                                    anchor="center", font=('Arial', 12))
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Control buttons with modern styling
        control_frame = ttk.LabelFrame(left_panel, text="🎮 Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)
        
        # Style for buttons
        style = ttk.Style()
        style.configure('Action.TButton', font=('Arial', 10, 'bold'))
        
        self.btn_start_cam = ttk.Button(btn_frame, text="📷 Start Camera", 
                                       command=self.start_camera, state=tk.DISABLED,
                                       style='Action.TButton')
        self.btn_start_cam.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        self.btn_start_vid = ttk.Button(btn_frame, text="📁 Open Video", 
                                       command=self.open_video, state=tk.DISABLED,
                                       style='Action.TButton')
        self.btn_start_vid.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        self.btn_stop = ttk.Button(btn_frame, text="⏹️ Stop", 
                                  command=self.stop, state=tk.DISABLED,
                                  style='Action.TButton')
        self.btn_stop.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # Right panel for detection indicators
        right_panel = ttk.Frame(main_container, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        right_panel.pack_propagate(False)
        
        self.create_detection_panel(right_panel)
        self.create_status_panel(left_panel)

    def create_detection_panel(self, parent):
        # Detection indicators panel
        detection_frame = ttk.LabelFrame(parent, text="👁️ Real-time Detection", padding=10)
        detection_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Eye state indicator
        eye_frame = ttk.LabelFrame(detection_frame, text="👁️ Eye State", padding=10)
        eye_frame.pack(fill=tk.X, pady=5)
        
        # Eye state progress bar
        ttk.Label(eye_frame, text="Eye Openness:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.eye_progress = ttk.Progressbar(eye_frame, mode='determinate', length=300)
        self.eye_progress.pack(fill=tk.X, pady=2)
        
        self.eye_state_label = ttk.Label(eye_frame, text="👁️ Open (100%)", 
                                        font=('Arial', 12, 'bold'), foreground='green')
        self.eye_state_label.pack(pady=2)
        
        # Eye closure time
        self.eye_time_label = ttk.Label(eye_frame, text="Closure time: 0.0s", font=('Arial', 9))
        self.eye_time_label.pack()
        
        # Mouth state indicator
        mouth_frame = ttk.LabelFrame(detection_frame, text="👄 Mouth State", padding=10)
        mouth_frame.pack(fill=tk.X, pady=5)
        
        # Mouth state progress bar
        ttk.Label(mouth_frame, text="Mouth Opening:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.mouth_progress = ttk.Progressbar(mouth_frame, mode='determinate', length=300)
        self.mouth_progress.pack(fill=tk.X, pady=2)
        
        self.mouth_state_label = ttk.Label(mouth_frame, text="👄 Closed (0%)", 
                                          font=('Arial', 12, 'bold'), foreground='blue')
        self.mouth_state_label.pack(pady=2)
        
        # Yawn duration
        self.mouth_time_label = ttk.Label(mouth_frame, text="Yawn duration: 0.0s", font=('Arial', 9))
        self.mouth_time_label.pack()
        
        # Overall drowsiness indicator
        drowsy_frame = ttk.LabelFrame(detection_frame, text="⚠️ Drowsiness Level", padding=10)
        drowsy_frame.pack(fill=tk.X, pady=5)
        
        # Drowsiness level gauge
        ttk.Label(drowsy_frame, text="Alertness Level:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.drowsiness_progress = ttk.Progressbar(drowsy_frame, mode='determinate', length=300)
        self.drowsiness_progress.pack(fill=tk.X, pady=2)
        
        self.drowsiness_label = ttk.Label(drowsy_frame, text="😊 Alert", 
                                         font=('Arial', 14, 'bold'), foreground='green')
        self.drowsiness_label.pack(pady=2)
        
        # Alert history
        history_frame = ttk.LabelFrame(detection_frame, text="📝 Recent Alerts", padding=5)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create scrollable text widget for alerts
        text_frame = ttk.Frame(history_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.alert_text = tk.Text(text_frame, height=8, width=40, font=('Consolas', 9))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.alert_text.yview)
        self.alert_text.configure(yscrollcommand=scrollbar.set)
        
        self.alert_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Clear alerts button
        ttk.Button(history_frame, text="🗑️ Clear History", 
                  command=self.clear_alert_history).pack(pady=2)

    def create_status_panel(self, parent):
        # Status panel with modern design
        status_frame = ttk.LabelFrame(parent, text="📊 System Status", padding=10)
        status_frame.pack(fill=tk.X, pady=5)
        
        # Create grid for status info
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill=tk.X)
        
        # Model status
        ttk.Label(status_grid, text="Model Status:", font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=5)
        self.model_status_label = ttk.Label(status_grid, text="❌ Not loaded", foreground="red", font=('Arial', 9))
        self.model_status_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # FPS
        ttk.Label(status_grid, text="FPS:", font=('Arial', 9, 'bold')).grid(row=0, column=2, sticky=tk.W, padx=5)
        self.fps_label = ttk.Label(status_grid, text="--", font=('Arial', 9))
        self.fps_label.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Processing time
        ttk.Label(status_grid, text="Processing:", font=('Arial', 9, 'bold')).grid(row=1, column=0, sticky=tk.W, padx=5)
        self.processing_label = ttk.Label(status_grid, text="-- ms", font=('Arial', 9))
        self.processing_label.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Detection count
        ttk.Label(status_grid, text="Detections:", font=('Arial', 9, 'bold')).grid(row=1, column=2, sticky=tk.W, padx=5)
        self.detection_label = ttk.Label(status_grid, text="--", font=('Arial', 9))
        self.detection_label.grid(row=1, column=3, sticky=tk.W, padx=5)
        
        # Status
        ttk.Label(status_grid, text="Status:", font=('Arial', 9, 'bold')).grid(row=2, column=0, sticky=tk.W, padx=5)
        self.status_label = ttk.Label(status_grid, text="Ready", font=('Arial', 9))
        self.status_label.grid(row=2, column=1, sticky=tk.W, padx=5)

    def create_config_tab(self):
        """Create the configuration tab with model settings."""
        frame = ttk.Frame(self.config_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Model selection
        model_frame = ttk.LabelFrame(frame, text="🤖 Model Configuration", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        # Auto/Manual selection mode
        mode_frame = ttk.Frame(model_frame)
        mode_frame.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        self.auto_mode = tk.BooleanVar(value=False)  # Disable auto by default
        ttk.Checkbutton(mode_frame, text="🔧 Auto Model Selection (Disabled)", 
                       variable=self.auto_mode, state='disabled',
                       command=self.toggle_model_mode).pack(side=tk.LEFT)
        
        ttk.Button(mode_frame, text="🔄 Refresh Models", 
                  command=self.refresh_available_models).pack(side=tk.RIGHT, padx=5)
        
        # YOLO Model Selection
        ttk.Label(model_frame, text="YOLO Version:", font=('Arial', 10, 'bold')).grid(row=1, column=0, padx=5, sticky=tk.W, pady=2)
        # Initial setup for model selection
        self.yolo_var = tk.StringVar(value='yolov11')  # Set default YOLO version
        self.yolo_combo = ttk.Combobox(model_frame, textvariable=self.yolo_var, 
                                      values=self.available_yolo_versions,
                                      state='readonly', width=25)
        self.yolo_combo.grid(row=1, column=1, padx=5, sticky=tk.W)
        self.yolo_combo.bind('<<ComboboxSelected>>', self.on_yolo_changed)
        
        # Classification Model Selection
        ttk.Label(model_frame, text="Classification Backbone:", font=('Arial', 10, 'bold')).grid(row=2, column=0, padx=5, sticky=tk.W, pady=2)
        self.backbone_var = tk.StringVar(value='efficientnet_b0')  # Set default backbone
        self.backbone_combo = ttk.Combobox(model_frame, textvariable=self.backbone_var,
                                          values=self.available_backbones,
                                          state='readonly', width=25)
        self.backbone_combo.grid(row=2, column=1, padx=5, sticky=tk.W)
        self.backbone_combo.bind('<<ComboboxSelected>>', self.on_backbone_changed)

        # Model load button and status
        load_frame = ttk.Frame(model_frame)
        load_frame.grid(row=3, column=0, columnspan=3, pady=15, sticky=tk.EW)
        
        ttk.Button(load_frame, text="🚀 Load Models", command=self.load_models,
                  style='Action.TButton').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(load_frame, text="🔄 Reset to Auto", command=self.reset_to_auto).pack(side=tk.RIGHT, padx=5)

        # Advanced Configuration Panel (collapsible)
        self.show_advanced = tk.BooleanVar(value=False)
        advanced_toggle = ttk.Checkbutton(frame, text="⚙️ Show Advanced Configuration", 
                                         variable=self.show_advanced, 
                                         command=self.toggle_advanced_panel)
        advanced_toggle.pack(pady=5)
        
        self.advanced_frame = ttk.LabelFrame(frame, text="🎛️ Advanced Configuration", padding=10)
        
        # Detection Thresholds
        thresh_frame = ttk.Frame(self.advanced_frame)
        thresh_frame.grid(row=0, column=0, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(thresh_frame, text="Detection Confidence:", font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=5)
        self.confidence_var = tk.DoubleVar(value=self.config_manager.config.get('confidence_threshold', 0.5))
        confidence_scale = ttk.Scale(thresh_frame, from_=0.1, to=1.0, variable=self.confidence_var, 
                                   orient=tk.HORIZONTAL, length=200, command=self.update_confidence)
        confidence_scale.grid(row=0, column=1, padx=5)
        self.confidence_label = ttk.Label(thresh_frame, text=f"{self.confidence_var.get():.2f}")
        self.confidence_label.grid(row=0, column=2, padx=5)
        
        ttk.Label(thresh_frame, text="Eye Closure Threshold (s):", font=('Arial', 9, 'bold')).grid(row=1, column=0, sticky=tk.W, padx=5)
        self.eye_threshold_var = tk.DoubleVar(value=self.config_manager.config.get('eye_closure_threshold', 0.8))
        eye_scale = ttk.Scale(thresh_frame, from_=0.2, to=3.0, variable=self.eye_threshold_var,
                            orient=tk.HORIZONTAL, length=200, command=self.update_eye_threshold)
        eye_scale.grid(row=1, column=1, padx=5)
        self.eye_threshold_label = ttk.Label(thresh_frame, text=f"{self.eye_threshold_var.get():.1f}s")
        self.eye_threshold_label.grid(row=1, column=2, padx=5)
        
        ttk.Label(thresh_frame, text="Yawn Threshold (s):", font=('Arial', 9, 'bold')).grid(row=2, column=0, sticky=tk.W, padx=5)
        self.yawn_threshold_var = tk.DoubleVar(value=self.config_manager.config.get('yawn_threshold', 1.5))
        yawn_scale = ttk.Scale(thresh_frame, from_=0.5, to=5.0, variable=self.yawn_threshold_var,
                             orient=tk.HORIZONTAL, length=200, command=self.update_yawn_threshold)
        yawn_scale.grid(row=2, column=1, padx=5)
        self.yawn_threshold_label = ttk.Label(thresh_frame, text=f"{self.yawn_threshold_var.get():.1f}s")
        self.yawn_threshold_label.grid(row=2, column=2, padx=5)
        
        # Alert System Mode
        alert_frame = ttk.Frame(self.advanced_frame)
        alert_frame.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        ttk.Label(alert_frame, text="Alert System Mode:", font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=5)
        self.alert_mode_var = tk.StringVar(value=self.config_manager.config.get('alert_system_mode', 'adaptive'))
        alert_mode_combo = ttk.Combobox(alert_frame, textvariable=self.alert_mode_var,
                                       values=['strict', 'adaptive', 'relaxed'],
                                       state='readonly', width=15)
        alert_mode_combo.grid(row=0, column=1, padx=5)
        alert_mode_combo.bind('<<ComboboxSelected>>', self.update_alert_mode)
        
        # Sound Settings
        ttk.Label(alert_frame, text="Sound Volume:", font=('Arial', 9, 'bold')).grid(row=1, column=0, sticky=tk.W, padx=5)
        self.volume_var = tk.DoubleVar(value=self.config_manager.config.get('sound_volume', 0.7))
        volume_scale = ttk.Scale(alert_frame, from_=0.0, to=1.0, variable=self.volume_var,
                               orient=tk.HORIZONTAL, length=150, command=self.update_volume)
        volume_scale.grid(row=1, column=1, padx=5)
        
        self.sound_enabled_var = tk.BooleanVar(value=self.config_manager.config.get('sound_enabled', True))
        ttk.Checkbutton(alert_frame, text="🔊 Enable Sound", variable=self.sound_enabled_var,
                       command=self.update_sound_enabled).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5)
        
        # Control buttons for advanced settings
        control_frame = ttk.Frame(self.advanced_frame)
        control_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(control_frame, text="💾 Save Settings", 
                  command=self.save_current_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🔄 Load Defaults", 
                  command=self.load_default_settings).pack(side=tk.LEFT, padx=5)

    def create_analytics_tab(self):
        """Create analytics tab with performance graphs."""
        analytics_frame = ttk.Frame(self.analytics_tab)
        analytics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure for real-time plotting
        self.fig = Figure(figsize=(12, 8), dpi=80)
        self.fig.patch.set_facecolor('white')
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(221)  # Eye state
        self.ax2 = self.fig.add_subplot(222)  # Mouth state
        self.ax3 = self.fig.add_subplot(223)  # Drowsiness level
        self.ax4 = self.fig.add_subplot(224)  # FPS
        
        # Configure subplots
        self.ax1.set_title('Eye State Over Time')
        self.ax1.set_ylabel('Eye Openness (%)')
        self.ax1.set_ylim(0, 100)
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('Mouth State Over Time')
        self.ax2.set_ylabel('Mouth Opening (%)')
        self.ax2.set_ylim(0, 100)
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_title('Drowsiness Level')
        self.ax3.set_ylabel('Drowsiness (%)')
        self.ax3.set_ylim(0, 100)
        self.ax3.grid(True, alpha=0.3)
        
        self.ax4.set_title('Performance (FPS)')
        self.ax4.set_ylabel('FPS')
        self.ax4.set_xlabel('Time (s)')
        self.ax4.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        
        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, analytics_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control frame for analytics
        analytics_control = ttk.Frame(analytics_frame)
        analytics_control.pack(fill=tk.X, pady=5)
        
        ttk.Button(analytics_control, text="🔄 Clear Graphs", 
                  command=self.clear_analytics).pack(side=tk.LEFT, padx=5)
        ttk.Button(analytics_control, text="💾 Export Data", 
                  command=self.export_analytics).pack(side=tk.LEFT, padx=5)
        
        # Initialize data lists for plotting
        self.time_data = []
        self.eye_data = []
        self.mouth_data = []
        self.drowsiness_data = []
        self.fps_data = []

    def refresh_available_models(self):
        """Refresh the list of available models."""
        try:
            self.available_yolo_versions = self.config_manager.get_available_yolo_versions()
            self.available_backbones = self.config_manager.get_available_backbones()
            
            # Update combobox values if they exist
            if hasattr(self, 'yolo_combo'):
                self.yolo_combo['values'] = self.available_yolo_versions
            if hasattr(self, 'backbone_combo'):
                self.backbone_combo['values'] = self.available_backbones
                
            print(f"Found YOLO versions: {self.available_yolo_versions}")
            print(f"Found classification backbones: {self.available_backbones}")
        except Exception as e:
            print(f"Error refreshing models: {e}")
    
    def toggle_model_mode(self):
        """Toggle between auto and manual model selection."""
        # Since we removed manual path entries, just ensure dropdowns are enabled
        self.yolo_combo.config(state='readonly')  
        self.backbone_combo.config(state='readonly')
    
    def on_yolo_changed(self, event=None):
        """Handle YOLO model selection change."""
        selected = self.yolo_var.get()
        if selected != 'auto':
            # Update config
            self.config_manager.update_setting('yolo_version', selected)
            print(f"Selected YOLO version: {selected}")
    
    def on_backbone_changed(self, event=None):
        """Handle classification backbone selection change."""
        selected = self.backbone_var.get()
        if selected != 'auto':
            # Update config
            self.config_manager.update_setting('classification_backbone', selected)
            print(f"Selected classification backbone: {selected}")
    
    def refresh_auto_selection(self):
        """Refresh auto model selection based on current priority."""
        try:
            # Force refresh of auto selection
            if self.yolo_var.get() == 'auto':
                selected_yolo = self.config_manager._auto_select_yolo_version()
                print(f"Auto-selected YOLO: {selected_yolo}")
                
            if self.backbone_var.get() == 'auto':
                selected_backbone = self.config_manager._auto_select_classification_backbone()
                print(f"Auto-selected backbone: {selected_backbone}")
        except Exception as e:
            print(f"Error in auto selection: {e}")

    def get_model_paths(self):
        """Get the current model paths based on dropdown selections."""
        try:
            # Get selected versions from comboboxes
            selected_yolo = self.yolo_var.get()
            selected_backbone = self.backbone_var.get()
            
            # Build paths based on selections
            yolo_path = f"models/detect/{selected_yolo}/best.pt" 
            cls_path = f"models/classification/{selected_backbone}.pth"
            
            logging.info(f"Using model paths: YOLO={yolo_path}, CLS={cls_path}, Backbone={selected_backbone}")
            return yolo_path, cls_path, selected_backbone
            
        except Exception as e:
            logging.error(f"Error getting model paths: {e}")
            # Return default paths as fallback
            return None, None, None

    def browse_yolo(self):
        path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("YOLO Weights", "*.pt"), ("All Files", "*.*")],
            initialdir="models/detect/"
        )
        if path:
            self.yolo_entry.delete(0, tk.END)
            self.yolo_entry.insert(0, path)

    def browse_cls(self):
        path = filedialog.askopenfilename(
            title="Select Classification Model", 
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")],
            initialdir="models/classification/"
        )
        if path:
            self.cls_entry.delete(0, tk.END)
            self.cls_entry.insert(0, path)

    def load_models(self):
        """Load models with improved error handling and status updates."""
        try:
            # Check if models are already loaded
            if self.loaded:
                logging.info("Models already loaded, skipping redundant load")
                messagebox.showinfo("Info", "Models already loaded")
                return
                
            self.model_status_label.config(text="Loading models...", foreground="orange")
            self.root.update()
            
            yolo_path, cls_path, backbone = self.get_model_paths()
            
            # Validate paths are not empty
            if not yolo_path or not cls_path:
                raise ValueError("Please specify both YOLO and Classification model paths")
            
            # Validate paths exist
            if not os.path.exists(yolo_path):
                raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
            if not os.path.exists(cls_path):
                raise FileNotFoundError(f"Classification model not found: {cls_path}")
            
            # Update the configuration manager
            self.config_manager.update_parameter('ml_config', 'detection_model_version', self.yolo_var.get())
            self.config_manager.update_parameter('ml_config', 'classification_backbone', self.backbone_var.get())
            
            # Load models with logging
            logging.info(f"Loading models - YOLO: {yolo_path}, Classification: {cls_path}, Backbone: {backbone}")
            success = self.logic.load_models(yolo_path, cls_path, backbone)
            
            if not success:
                raise RuntimeError("Failed to load models. Check console for errors.")
                
            self.loaded = True
            
            # Update UI
            self.btn_start_cam.config(state=tk.NORMAL)
            self.btn_start_vid.config(state=tk.NORMAL)
            selected_yolo = self.yolo_var.get()
            selected_backbone = self.backbone_var.get()
            self.model_status_label.config(text=f"✅ Models loaded: {selected_yolo} + {selected_backbone}", foreground="green")
            
            # Show success message with model info
            selected_yolo = self.yolo_var.get()
            selected_backbone = self.backbone_var.get()
            model_info = f"YOLO Version: {selected_yolo}\nClassification Model: {selected_backbone}"
            messagebox.showinfo("Success", f"Models loaded successfully!\n\n{model_info}")
            
        except Exception as e:
            self.model_status_label.config(text=f"Error: {str(e)}", foreground="red")
            messagebox.showerror("Error Loading Models", str(e))
    def toggle_advanced_panel(self):
        """Toggle the advanced configuration panel."""
        if self.show_advanced.get():
            self.advanced_frame.pack(fill=tk.X, pady=5)
            self.root.geometry("1100x900")  # Expand window
        else:
            self.advanced_frame.pack_forget()
            self.root.geometry("1100x800")  # Shrink window
    
    def update_confidence(self, value=None):
        """Update confidence threshold."""
        val = self.confidence_var.get()
        self.confidence_label.config(text=f"{val:.2f}")
        self.config_manager.update_setting('confidence_threshold', val)
    
    def update_eye_threshold(self, value=None):
        """Update eye closure threshold."""
        val = self.eye_threshold_var.get()
        self.eye_threshold_label.config(text=f"{val:.1f}s")
        self.config_manager.update_setting('eye_closure_threshold', val)
    
    def update_yawn_threshold(self, value=None):
        """Update yawn threshold."""
        val = self.yawn_threshold_var.get()
        self.yawn_threshold_label.config(text=f"{val:.1f}s")
        self.config_manager.update_setting('yawn_threshold', val)
    
    def update_alert_mode(self, event=None):
        """Update alert system mode."""
        mode = self.alert_mode_var.get()
        self.config_manager.update_setting('alert_system_mode', mode)
    
    def update_volume(self, value=None):
        """Update sound volume."""
        vol = self.volume_var.get()
        self.config_manager.update_setting('sound_volume', vol)
    
    def update_sound_enabled(self):
        """Update sound enabled setting."""
        enabled = self.sound_enabled_var.get()
        self.config_manager.update_setting('sound_enabled', enabled)
    
    def reset_to_auto(self):
        """Reset all settings to auto/optimized values."""
        if messagebox.askyesno("Reset to Auto", "Reset all settings to auto-optimized values?"):
            # Reset model selection to auto
            self.auto_mode.set(True)
            self.yolo_var.set('auto')
            self.backbone_var.set('auto')
            
            # Reset config to auto values
            self.config_manager.update_setting('auto_model_selection', True)
            self.config_manager.update_setting('auto_optimize_thresholds', True)
            self.config_manager.update_setting('yolo_version', 'auto')
            self.config_manager.update_setting('classification_backbone', 'auto')
            
            # Update sliders
            self.confidence_var.set(0.5)
            self.eye_threshold_var.set(0.8)
            self.yawn_threshold_var.set(1.5)
            self.alert_mode_var.set('adaptive')
            self.volume_var.set(0.7)
            self.sound_enabled_var.set(True)
            
            self.toggle_model_mode()
            messagebox.showinfo("Success", "Settings reset to auto-optimized values!")
    
    def save_current_settings(self):
        """Save current settings to config file."""
        try:
            self.config_manager.save_config()
            messagebox.showinfo("Success", "Settings saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def load_default_settings(self):
        """Load default settings."""
        if messagebox.askyesno("Load Defaults", "Load default settings? This will overwrite current settings."):
            try:
                self.config_manager.reset_to_defaults()
                self.refresh_ui_from_config()
                messagebox.showinfo("Success", "Default settings loaded!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load defaults: {e}")
    
    def refresh_ui_from_config(self):
        """Refresh UI elements from current config."""
        config = self.config_manager.config
        
        # Update sliders and variables
        self.confidence_var.set(config.get('confidence_threshold', 0.5))
        self.eye_threshold_var.set(config.get('eye_closure_threshold', 0.8))
        self.yawn_threshold_var.set(config.get('yawn_threshold', 1.5))
        self.alert_mode_var.set(config.get('alert_system_mode', 'adaptive'))
        self.volume_var.set(config.get('sound_volume', 0.7))
        self.sound_enabled_var.set(config.get('sound_enabled', True))
        # Update labels
        self.update_confidence()
        self.update_eye_threshold()
        self.update_yawn_threshold()

    def start_camera(self):
        """Start camera with improved status updates and non-blocking operation."""
        if not self.loaded:
            # Prompt user to load models first
            if messagebox.askyesno("Models Not Loaded", "Models have not been loaded yet. Do you want to load them now?"):
                self.load_models()
                if not self.loaded:  # If loading failed, abort
                    return
            else:
                return
        
        # Disable buttons immediately to prevent multiple starts
        self.btn_start_cam.config(state=tk.DISABLED)
        self.btn_start_vid.config(state=tk.DISABLED)
        self.status_label.config(text="Starting camera...", foreground="orange")
        
        # Start camera in a separate thread to avoid blocking GUI
        def start_camera_thread():
            try:
                self.logic.start_camera()
                self.is_monitoring = True
                
                # Update UI from main thread
                self.root.after(0, lambda: self._on_camera_started())
                
            except Exception as e:
                # Update UI with error from main thread
                self.root.after(0, lambda: self._on_camera_error(str(e)))
        
        threading.Thread(target=start_camera_thread, daemon=True).start()
    
    def _on_camera_started(self):
        """Called when camera starts successfully (main thread)"""
        self.btn_stop.config(state=tk.NORMAL)
        
        # Update window title with model info
        yolo_version = self.config_manager.config.get('yolo_version', 'unknown')
        backbone = self.config_manager.config.get('classification_backbone', 'unknown')
        self.root.title(f"Driver Monitoring - Camera (YOLO: {yolo_version}, Backbone: {backbone})")
        
        self.status_label.config(text="Camera started successfully", foreground="green")
        self.add_alert_to_history("INFO", "Camera monitoring started")
    
    def _on_camera_error(self, error_msg):
        """Called when camera fails to start (main thread)"""
        self.btn_start_cam.config(state=tk.NORMAL)
        self.btn_start_vid.config(state=tk.NORMAL)
        self.status_label.config(text="Camera start failed", foreground="red")
        messagebox.showerror("Error", f"Failed to start camera: {error_msg}")

    def open_video(self):
        """Open video file with improved status updates and non-blocking operation."""
        if not self.loaded:
            # Prompt user to load models first
            if messagebox.askyesno("Models Not Loaded", "Models have not been loaded yet. Do you want to load them now?"):
                self.load_models()
                if not self.loaded:  # If loading failed, abort
                    return
            else:
                return
            
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")],
            initialdir="video/"
        )
        if path:
            # Disable buttons immediately
            self.btn_start_cam.config(state=tk.DISABLED)
            self.btn_start_vid.config(state=tk.DISABLED)
            self.status_label.config(text="Starting video...", foreground="orange")
            
            # Start video in separate thread
            def start_video_thread():
                try:
                    self.logic.start_video(path)
                    self.is_monitoring = True
                    
                    # Update UI from main thread
                    self.root.after(0, lambda: self._on_video_started(path))
                    
                except Exception as e:
                    # Update UI with error from main thread
                    self.root.after(0, lambda: self._on_video_error(str(e)))
            
            threading.Thread(target=start_video_thread, daemon=True).start()
    
    def _on_video_started(self, video_path):
        """Called when video starts successfully (main thread)"""
        self.btn_stop.config(state=tk.NORMAL)
        
        # Update window title with model and video info
        yolo_version = self.config_manager.config.get('yolo_version', 'unknown')
        backbone = self.config_manager.config.get('classification_backbone', 'unknown')
        video_name = os.path.basename(video_path)
        self.root.title(f"Driver Monitoring - {video_name} (YOLO: {yolo_version}, Backbone: {backbone})")
        
        self.status_label.config(text="Video started successfully", foreground="green")
        self.add_alert_to_history("INFO", f"Video monitoring started: {video_name}")
    
    def _on_video_error(self, error_msg):
        """Called when video fails to start (main thread)"""
        self.btn_start_cam.config(state=tk.NORMAL)
        self.btn_start_vid.config(state=tk.NORMAL)
        self.status_label.config(text="Video start failed", foreground="red")
        messagebox.showerror("Error", f"Failed to open video: {error_msg}")

    def stop(self):
        """Stop monitoring with status reset and non-blocking operation."""
        # Update UI immediately
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_start_cam.config(state=tk.DISABLED)  # Disable temporarily
        self.btn_start_vid.config(state=tk.DISABLED)
        self.status_label.config(text="Stopping...", foreground="orange")
        self.is_monitoring = False
        
        # Stop in separate thread to avoid blocking GUI
        def stop_thread():
            try:
                self.logic.stop()
                # Update UI from main thread
                self.root.after(0, self._on_stop_success)
            except Exception as e:
                # Handle error from main thread
                self.root.after(0, lambda: self._on_stop_error(str(e)))
        
        threading.Thread(target=stop_thread, daemon=True).start()
    
    def _on_stop_success(self):
        """Called when stop is successful (main thread)"""
        self.btn_start_cam.config(state=tk.NORMAL if self.loaded else tk.DISABLED)
        self.btn_start_vid.config(state=tk.NORMAL if self.loaded else tk.DISABLED)
        self.status_label.config(text="Stopped", foreground="black")
        self.root.title("Driver Monitoring System")
        
        # Reset video display
        self.video_label.config(image='', text="🎬 Ready to monitor\nClick 'Start Camera' or 'Open Video' to begin")
        
    def _on_stop_error(self, error_msg):
        """Called when stop fails (main thread)"""
        self.btn_start_cam.config(state=tk.NORMAL if self.loaded else tk.DISABLED)
        self.btn_start_vid.config(state=tk.NORMAL if self.loaded else tk.DISABLED)
        self.status_label.config(text="Stop failed", foreground="red")
        logging.error(f"Stop error: {error_msg}")

    def update_display(self):
        """Update display with processed image and enhanced status information."""
        try:
            if hasattr(self.logic, 'is_monitoring') and self.logic.is_monitoring:
                # Use a timeout to prevent blocking
                try:
                    display_image = self.logic.get_display_image()
                    if display_image:
                        self.video_label.config(image=display_image, text="")
                        self.video_label.image = display_image
                except Exception as e:
                    logging.warning(f"Display image update failed: {e}")
                    
                # Update status information safely
                try:
                    fps = self.logic.get_fps()
                    self.fps_label.config(text=f"{fps:.1f}")
                except Exception as e:
                    self.fps_label.config(text="--")
                
                # Update processing time (if available)
                if hasattr(self.logic, 'processing_time'):
                    try:
                        proc_time = getattr(self.logic, 'processing_time', 0) * 1000
                        self.processing_label.config(text=f"{proc_time:.1f} ms")
                    except:
                        pass
                
                # Update detection count (if available)
                if hasattr(self.logic, 'detection_count'):
                    try:
                        det_count = getattr(self.logic, 'detection_count', 0)
                        self.detection_label.config(text=f"{det_count}")
                    except:
                        pass
                          # Check for alerts and update border
                has_alert = False
                if hasattr(self.logic, 'drowsiness_detector'):
                    try:
                        status = self.logic.drowsiness_detector.get_status()
                        has_alert = len(status.get('alerts', [])) > 0
                        self.update_alert_border(has_alert)
                          # Add alerts to history
                        if has_alert and status['alerts']:
                            for alert in status['alerts']:
                                if alert != getattr(self, 'last_alert', None):
                                    self.add_alert_to_history("ALERT", alert)
                                    self.last_alert = alert
                    except:
                        pass
                else:
                    self.update_alert_border(False)
                
                # Update status safely
                try:
                    if has_alert:
                        self.status_label.config(text="🚨 DROWSINESS ALERT 🚨", foreground="red", font=('Arial', 10, 'bold'))
                    else:
                        self.status_label.config(text="✅ Monitoring Active", foreground="green", font=('Arial', 10, 'normal'))
                except:
                    self.status_label.config(text="Monitoring active", foreground="green")
            else:
                # Reset displays when not monitoring
                self.update_alert_border(False)  # Reset border to normal
                try:
                    if hasattr(self, 'fps_label'):
                        self.fps_label.config(text="--")
                    if hasattr(self, 'detection_label'):
                        self.detection_label.config(text="--")
                    if hasattr(self, 'status_label'):
                        self.status_label.config(text="Ready", foreground="black", font=('Arial', 10, 'normal'))
                except:
                    pass
        except Exception as e:
            logging.error(f"Update display error: {e}")
        
        # Schedule next update with longer interval to prevent GUI freeze
        self.root.after(50, self.update_display)  # Reduced to 20 FPS for GUI

    def on_close(self):
        self.stop()
        self.root.destroy()

    def update_detection_indicators(self):
        """Update real-time detection indicators."""
        try:
            if not self.is_monitoring:
                self.root.after(300, self.update_detection_indicators)  # Longer interval when not monitoring
                return
                
            # Get current detection states from logic with safer attribute access
            eye_state = getattr(self.logic, 'current_eye_state', 'Unknown')
            eye_confidence = getattr(self.logic, 'last_eye_confidence', 0.0)
            eye_closed_time = getattr(self.logic, 'eye_closed_time', 0.0)
            
            # Update eye indicators
            if eye_state == 'Open':
                self.eye_progress['value'] = eye_confidence * 100
                self.eye_state_label.config(text=f"👁️ Open ({eye_confidence*100:.0f}%)", 
                                          foreground='green')
            else:
                # For closed eyes, show closure confidence
                closure_confidence = 1.0 - eye_confidence if eye_confidence > 0 else 0.5
                self.eye_progress['value'] = closure_confidence * 100
                self.eye_state_label.config(text=f"😴 Closed ({closure_confidence*100:.0f}%)", 
                                          foreground='red')
            
            self.eye_time_label.config(text=f"Closure time: {eye_closed_time:.1f}s")
            
            # Get mouth/yawn states
            yawn_state = getattr(self.logic, 'current_yawn_state', 'No Yawn')
            yawn_confidence = getattr(self.logic, 'last_yawn_confidence', 0.0)
            yawn_duration = getattr(self.logic, 'yawn_duration', 0.0)
            
            # Update mouth indicators
            if yawn_state == 'yawn':
                self.mouth_progress['value'] = yawn_confidence * 100
                self.mouth_state_label.config(text=f"🥱 Yawning ({yawn_confidence*100:.0f}%)", 
                                            foreground='orange')
            else:
                # For no yawn, show normal confidence
                normal_confidence = 1.0 - yawn_confidence if yawn_confidence > 0 else 0.8
                self.mouth_progress['value'] = normal_confidence * 100
                self.mouth_state_label.config(text=f"👄 Normal ({normal_confidence*100:.0f}%)", 
                                            foreground='blue')
            
            self.mouth_time_label.config(text=f"Yawn duration: {yawn_duration:.1f}s")
            
            # Calculate overall drowsiness level
            drowsiness_level = 0
            eye_factor = min(eye_closed_time / 2.0, 1.0) if eye_closed_time > 0 else 0
            yawn_factor = min(yawn_duration / 3.0, 1.0) if yawn_duration > 0 else 0
            drowsiness_level = max(eye_factor, yawn_factor) * 100
            
            self.drowsiness_progress['value'] = drowsiness_level
            
            # Update drowsiness status with ENHANCED RED ALERTS
            if drowsiness_level < 25:
                self.drowsiness_label.config(text="😊 Alert", foreground='green')
                self.stop_red_alert()  # Stop any active alerts
            elif drowsiness_level < 75:
                self.drowsiness_label.config(text="😴 Drowsy", foreground='red')
                self.start_red_alert("DROWSY")  # Start moderate alert
            else:
                self.drowsiness_label.config(text="🚨 NGUY HIỂM! 🚨", foreground='darkred')
                self.start_red_alert("DANGER")  # Start critical alert
            
            # Check for current alerts from detection system
            current_alerts = getattr(self.logic, 'current_alerts', [])
            if current_alerts:
                self.start_red_alert("CRITICAL")
                # Add alerts to history
                for alert in current_alerts:
                    self.add_alert_to_history(f"🚨 {alert}")
            
            # Update analytics data (less frequently to improve performance)
            if hasattr(self, 'last_analytics_update'):
                if time.time() - self.last_analytics_update >= 0.5:  # Update every 500ms
                    self.update_analytics_data()
                    self.last_analytics_update = time.time()
            else:
                self.last_analytics_update = time.time()
                self.update_analytics_data()
            
        except Exception as e:
            logging.warning(f"Error updating indicators: {e}")
        
        # Schedule next update with reasonable interval
        self.root.after(200, self.update_detection_indicators)
    
    def update_analytics_data(self):
        """Update analytics data less frequently to improve performance"""
        try:
            current_time = time.time()
            if not hasattr(self, 'start_time'):
                self.start_time = current_time
            
            relative_time = current_time - self.start_time
            
            # Get data from logic
            eye_confidence = getattr(self.logic, 'last_eye_confidence', 0) * 100
            yawn_confidence = getattr(self.logic, 'last_yawn_confidence', 0) * 100
            current_fps = getattr(self.logic, 'fps', 0)
            
            # Calculate drowsiness level
            eye_closed_time = getattr(self.logic, 'eye_closed_time', 0.0)
            yawn_duration = getattr(self.logic, 'yawn_duration', 0.0)
            drowsiness_level = max(
                min(eye_closed_time / 2.0, 1.0),
                min(yawn_duration / 3.0, 1.0)
            ) * 100
            
            # Append data
            self.time_data.append(relative_time)
            self.eye_data.append(eye_confidence)
            self.mouth_data.append(yawn_confidence)
            self.drowsiness_data.append(drowsiness_level)
            self.fps_data.append(current_fps)
            
            # Keep only last 50 data points (reduced from 100)
            max_points = 50
            if len(self.time_data) > max_points:
                self.time_data = self.time_data[-max_points:]
                self.eye_data = self.eye_data[-max_points:]
                self.mouth_data = self.mouth_data[-max_points:]
                self.drowsiness_data = self.drowsiness_data[-max_points:]
                self.fps_data = self.fps_data[-max_points:]
            
            # Update plots only if we have enough data
            if len(self.time_data) > 2:
                self.update_analytics_plots()
                
        except Exception as e:
            print(f"Error updating analytics data: {e}")

    def update_analytics_plots(self):
        """Update the analytics plots with latest data - optimized for performance."""
        if len(self.time_data) < 2:
            return
            
        try:
            # Only update if plots are visible (Analytics tab is selected)
            current_tab = self.notebook.tab(self.notebook.select(), "text")
            if current_tab != "Analytics":
                return
            
            # Limit data points for smoother plotting
            display_points = min(len(self.time_data), 30)
            time_subset = self.time_data[-display_points:]
            eye_subset = self.eye_data[-display_points:]
            mouth_subset = self.mouth_data[-display_points:]
            drowsiness_subset = self.drowsiness_data[-display_points:]
            fps_subset = self.fps_data[-display_points:]
            
            # Clear and update plots efficiently
            self.ax1.clear()
            self.ax1.plot(time_subset, eye_subset, 'b-', linewidth=2, alpha=0.8)
            self.ax1.set_title('👁️ Eye State Over Time', fontsize=10)
            self.ax1.set_ylabel('Eye Openness (%)', fontsize=9)
            self.ax1.set_ylim(0, 100)
            self.ax1.grid(True, alpha=0.3)
            self.ax1.tick_params(labelsize=8)
            
            self.ax2.clear()
            self.ax2.plot(time_subset, mouth_subset, 'g-', linewidth=2, alpha=0.8)
            self.ax2.set_title('👄 Mouth State Over Time', fontsize=10)
            self.ax2.set_ylabel('Mouth Opening (%)', fontsize=9)
            self.ax2.set_ylim(0, 100)
            self.ax2.grid(True, alpha=0.3)
            self.ax2.tick_params(labelsize=8)
            
            self.ax3.clear()
            self.ax3.plot(time_subset, drowsiness_subset, 'r-', linewidth=2, alpha=0.8)
            self.ax3.set_title('⚠️ Drowsiness Level', fontsize=10)
            self.ax3.set_ylabel('Drowsiness (%)', fontsize=9)
            self.ax3.set_ylim(0, 100)
            self.ax3.grid(True, alpha=0.3)
            self.ax3.axhline(y=75, color='r', linestyle='--', alpha=0.5)
            self.ax3.tick_params(labelsize=8)
            
            self.ax4.clear()
            self.ax4.plot(time_subset, fps_subset, 'm-', linewidth=2, alpha=0.8)
            self.ax4.set_title('Performance (FPS)', fontsize=10)
            self.ax4.set_ylabel('FPS', fontsize=9)
            self.ax4.set_xlabel('Time (s)', fontsize=9)
            self.ax4.grid(True, alpha=0.3)
            self.ax4.tick_params(labelsize=8)
            
            # Use tight_layout with reduced padding for better performance
            self.fig.tight_layout(pad=1.0)
            
            # Draw canvas efficiently
            self.canvas.draw_idle()  # Use draw_idle instead of draw for better performance
            
        except Exception as e:
            print(f"Error updating plots: {e}")

    def clear_analytics(self):
        """Clear all analytics data and plots."""
        self.time_data.clear()
        self.eye_data.clear()
        self.mouth_data.clear()
        self.drowsiness_data.clear()
        self.fps_data.clear()
        
        # Clear plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        self.canvas.draw()
        
    def export_analytics(self):
        """Export analytics data to CSV file."""
        try:
            import csv
            from datetime import datetime
            
            filename = f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialvalue=filename
            )
            
            if filepath:
                with open(filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Time', 'Eye_Openness', 'Mouth_Opening', 'Drowsiness_Level', 'FPS'])
                    
                    for i in range(len(self.time_data)):
                        writer.writerow([
                            self.time_data[i],
                            self.eye_data[i] if i < len(self.eye_data) else 0,
                            self.mouth_data[i] if i < len(self.mouth_data) else 0,
                            self.drowsiness_data[i] if i < len(self.drowsiness_data) else 0,
                            self.fps_data[i] if i < len(self.fps_data) else 0
                        ])
                
                messagebox.showinfo("Export Complete", f"Analytics data exported to:\n{filepath}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{e}")

    def clear_alert_history(self):
        """Clear the alert history display."""
        self.alert_text.delete(1.0, tk.END)

    def add_alert_to_history(self, alert_type, message):
        """Add an alert to the history display."""
        timestamp = time.strftime("%H:%M:%S")
        alert_entry = f"[{timestamp}] {alert_type}: {message}\n"
        
        self.alert_text.insert(tk.END, alert_entry)
        self.alert_text.see(tk.END)  # Scroll to bottom
        
        # Keep only last 50 alerts
        lines = self.alert_text.get(1.0, tk.END).split('\n')
        if len(lines) > 50:
            self.alert_text.delete(1.0, f"{len(lines)-50}.0")
    
    def start_red_alert(self, alert_level="NORMAL"):
        """Start RED ALERT system with flashing GUI and sound"""
        if not self.alert_active:
            self.alert_active = True
            self.alert_level = alert_level
            self.last_alert_time = time.time()
            self.flash_red_alert()
            
            # Play alert sound if enabled
            if self.alert_sound_enabled:
                self.play_alert_sound(alert_level)
    
    def stop_red_alert(self):
        """Stop RED ALERT system"""
        if self.alert_active:
            self.alert_active = False
            self.alert_flash_state = False
            # Restore original background color
            self.root.configure(bg=self.original_bg_color)
    
    def flash_red_alert(self):
        """Flash the entire GUI red when in alert mode"""
        if not self.alert_active:
            return
            
        # Toggle flash state
        self.alert_flash_state = not self.alert_flash_state
        
        # Set background color based on flash state and alert level
        if self.alert_flash_state:
            if self.alert_level == "DANGER" or self.alert_level == "CRITICAL":
                bg_color = '#ff0000'  # Bright red for danger
            else:
                bg_color = '#ff6666'  # Light red for drowsy
        else:
            bg_color = self.original_bg_color
        
        # Apply background color
        self.root.configure(bg=bg_color)
        
        # Schedule next flash
        flash_interval = 200 if self.alert_level in ["DANGER", "CRITICAL"] else 500  # Faster flash for critical
        self.root.after(flash_interval, self.flash_red_alert)
    
    def play_alert_sound(self, alert_level):
        """Play alert sound - simplified version using system beep"""
        try:
            import winsound
            # Different beep patterns for different alert levels
            if alert_level == "CRITICAL":
                # Rapid triple beep for critical
                for _ in range(3):
                    winsound.Beep(1000, 200)
                    time.sleep(0.1)
            elif alert_level == "DANGER":
                # Double beep for danger
                winsound.Beep(800, 400)
                time.sleep(0.2)
                winsound.Beep(800, 400)
            else:
                # Single beep for drowsy
                winsound.Beep(600, 500)
        except ImportError:
            # Fallback for non-Windows systems
            print('\a')  # Terminal bell
        except Exception as e:
            logging.warning(f"Could not play alert sound: {e}")
    
    def add_alert_to_history(self, alert_message):
        """Add alert to history display with timestamp"""
        try:
            timestamp = time.strftime("%H:%M:%S")
            full_message = f"[{timestamp}] {alert_message}\n"
            
            # Insert at the beginning (most recent first)
            self.alert_text.insert(tk.END, full_message)
            
            # Auto-scroll to bottom
            self.alert_text.see(tk.END)
            
            # Configure text color for alerts
            if "🚨" in alert_message:
                # Make alert text red
                last_line = self.alert_text.index(tk.END + "-1c linestart")
                self.alert_text.tag_add("alert", last_line, tk.END)
                self.alert_text.tag_config("alert", foreground="red", font=("Consolas", 9, "bold"))
            
        except Exception as e:
            logging.warning(f"Error adding alert to history: {e}")
    
    def clear_alert_history(self):
        """Clear alert history display"""
        try:
            self.alert_text.delete(1.0, tk.END)
        except Exception as e:
            logging.warning(f"Error clearing alert history: {e}")
    
    def update_alert_border(self, has_alert):
        """Update video border color based on alert status"""
        if has_alert:
            # Flash red border
            flash_state = int(time.time() * 4) % 2  # Flash 2 times per second
            if flash_state == 0:
                self.video_container.config(bg='#ff0000', bd=8, relief='solid')  # Bright red
            else:
                self.video_container.config(bg='#cc0000', bd=8, relief='solid')  # Dark red
        else:
            # Normal gray border
            self.video_container.config(bg='#f0f0f0', bd=2, relief='solid')


