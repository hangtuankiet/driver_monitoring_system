import tkinter as tk
from tkinter import Label, Frame, ttk, messagebox, filedialog, simpledialog
import threading
from .logic import DriverMonitor
from datetime import datetime
import json
import time


class DriverMonitoringGUI:
    """A GUI application for the Smart Driver Monitoring System."""
    
    def __init__(self, root: tk.Tk) -> None:
        """Initialize the GUI with root window and setup components."""
        self.root = root
        # Initialize the logic without loading models initially
        self.logic = DriverMonitor()
        self.is_paused = False
        self.is_monitoring = False
        self.setup_gui()

    def setup_gui(self) -> None:
        """Configure window and initialize all GUI components."""
        # Basic window setup
        self.root.title("🚗 Smart Driver Monitoring System")
        self.root.geometry("1024x768")
        self.root.configure(bg="#ECEFF1")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initialize GUI components
        self._setup_styles()
        self._create_menu()
        self._create_main_frames()
        self._create_video_display()
        self._create_status_indicators()
        self._create_control_buttons()

    def _setup_styles(self) -> None:
        """Configure ttk styles for consistent UI appearance."""
        style = ttk.Style()
        
        # Button styles
        style.configure("TButton", font=("Arial", 12, "bold"), padding=10)
        style.configure("Green.TButton", background="#4CAF50", foreground="#504B38")
        style.map("Green.TButton", background=[("active", "#388E3C")])
        style.configure("Purple.TButton", background="#9C27B0", foreground="#504B38")
        style.map("Purple.TButton", background=[("active", "#7B1FA2")])
        style.configure("Gray.TButton", background="#757575", foreground="#504B38")
        style.map("Gray.TButton", background=[("active", "#616161")])
        style.configure("Blue.TButton", background="#2196F3", foreground="#504B38")
        style.map("Blue.TButton", background=[("active", "#1976D2")])
        style.configure("Red.TButton", background="#F44336", foreground="#504B38")
        style.map("Red.TButton", background=[("active", "#D32F2F")])
        
        # Progress bar styles
        style.configure("green.Horizontal.TProgressbar", troughcolor="#ECEFF1", background="#4CAF50")
        style.configure("red.Horizontal.TProgressbar", troughcolor="#ECEFF1", background="#F44336")

    def _create_menu(self) -> None:
        """Create the application menu bar."""
        menubar = tk.Menu(self.root, bg="#455A64", fg="#504B38")
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg="#FFFFFF", fg="#504B38")
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Settings", command=self.show_settings)
        file_menu.add_command(label="View Alerts", command=self.show_alerts)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0, bg="#FFFFFF", fg="#504B38")
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo(
            "About", "Smart Driver Monitoring System\nVersion 1.0\n\nDetects driver drowsiness using computer vision."))

    def _create_main_frames(self) -> None:
        """Create main layout panels."""
        self.left_panel = Frame(self.root, bg="#ECEFF1")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.right_panel = Frame(self.root, bg="#ECEFF1")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

    def _create_video_display(self) -> None:
        """Create the video feed display area."""
        self.frame_video = Frame(self.left_panel, bg="black", bd=3, relief="ridge")
        self.frame_video.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.lbl_video = Label(self.frame_video, bg="black")
        self.lbl_video.pack()

    def _create_status_indicators(self) -> None:
        """Create status displays and progress indicators."""
        status_frame = Frame(self.right_panel, bg="#ECEFF1")
        status_frame.pack(pady=10, fill=tk.X)

        # FPS counter
        self.lbl_fps = Label(status_frame, text="FPS: 0", 
                            font=("Arial", 12), fg="#263238", bg="#ECEFF1")
        self.lbl_fps.pack(pady=5)
        
        # Eye closure progress bar
        self.eye_progress = ttk.Progressbar(status_frame, length=200, 
                                          mode='determinate', 
                                          style="green.Horizontal.TProgressbar")
        self.eye_progress.pack(pady=5)
        
        # Yawn progress bar
        self.yawn_progress = ttk.Progressbar(status_frame, length=200, 
                                           mode='determinate',
                                           style="red.Horizontal.TProgressbar")
        self.yawn_progress.pack(pady=5)
        
        # Status indicators
        self.lbl_status = Label(status_frame, text="🚗 Status: Idle", 
                              font=("Arial", 14, "bold"), fg="#263238", bg="#ECEFF1")
        self.lbl_status.pack(pady=10)
        
        self.lbl_eye = Label(status_frame, text="👀 Eyes: --", 
                           font=("Arial", 14), fg="#263238", bg="#ECEFF1")
        self.lbl_eye.pack()
        
        self.lbl_yawn = Label(status_frame, text="🗣️ Yawn: --", 
                            font=("Arial", 14), fg="#263238", bg="#ECEFF1")
        self.lbl_yawn.pack()

        # Model configuration display
        self._create_model_info_display()

    def _create_model_info_display(self) -> None:
        """Create a panel to display current model configuration."""
        model_info_frame = Frame(self.right_panel, bg="#ECEFF1", relief="ridge", bd=2)
        model_info_frame.pack(pady=10, fill=tk.X, padx=5)
        
        # Title
        title_label = Label(model_info_frame, text="📊 Current Configuration", 
                           font=("Arial", 12, "bold"), fg="#263238", bg="#ECEFF1")
        title_label.pack(pady=5)
        
        # YOLO model info
        self.lbl_yolo_model = Label(model_info_frame, text="🎯 YOLO: Not loaded", 
                                   font=("Arial", 10), fg="#37474F", bg="#ECEFF1")
        self.lbl_yolo_model.pack(pady=2)
        
        # Classification model info
        self.lbl_classification_model = Label(model_info_frame, text="🧠 Classifier: Not loaded", 
                                             font=("Arial", 10), fg="#37474F", bg="#ECEFF1")
        self.lbl_classification_model.pack(pady=2)
        
        # Source type info
        self.lbl_source_type = Label(model_info_frame, text="📹 Source: None", 
                                    font=("Arial", 10), fg="#37474F", bg="#ECEFF1")
        self.lbl_source_type.pack(pady=2)
        
        # Models status
        self.lbl_models_status = Label(model_info_frame, text="⚠️ Models not loaded", 
                                      font=("Arial", 10, "bold"), fg="#F57C00", bg="#ECEFF1")
        self.lbl_models_status.pack(pady=2)
        
        # Update display initially
        self._update_model_info_display()

    def _create_control_buttons(self) -> None:
        """Create control buttons for the application."""
        btn_frame = Frame(self.right_panel, bg="#ECEFF1")
        btn_frame.pack(pady=20)

        # Main control buttons
        self.btn_monitor = ttk.Button(btn_frame, text="▶ Start", 
                                    command=self.toggle_monitoring,
                                    style="Green.TButton")
        self.btn_monitor.pack(pady=5)

        self.btn_evaluate = ttk.Button(btn_frame, text="📊 Evaluate Performance", 
                                      command=self.toggle_evaluation,
                                      style="Purple.TButton")
        self.btn_evaluate.pack(pady=5)

        # Toolbar with secondary buttons
        toolbar_frame = Frame(self.right_panel, bg="#ECEFF1")
        toolbar_frame.pack(pady=5)

        ttk.Button(toolbar_frame, text="⚙ Settings", 
                  command=self.show_settings, 
                  style="Gray.TButton").pack(side=tk.LEFT, padx=5)
                  
        ttk.Button(toolbar_frame, text="📊 View Alerts", 
                  command=self.show_alerts, 
                  style="Blue.TButton").pack(side=tk.LEFT, padx=5)

    def _update_model_info_display(self) -> None:
        """Update the model information display with current configuration."""
        try:
            # Get current configuration
            yolo_version = self.logic.get_current_yolo_version()
            backbone = self.logic.get_current_backbone()
            
            # Update YOLO model display
            self.lbl_yolo_model.config(text=f"🎯 YOLO: {yolo_version}")
            
            # Update classification model display
            self.lbl_classification_model.config(text=f"🧠 Classifier: {backbone}")
            
            # Check if models are loaded
            models_loaded = (hasattr(self.logic, 'yolo_model') and self.logic.yolo_model is not None and
                           hasattr(self.logic, 'classification_model') and self.logic.classification_model is not None)
            
            if models_loaded:
                self.lbl_models_status.config(text="✅ Models loaded", fg="#2E7D32")
            else:
                self.lbl_models_status.config(text="⚠️ Models not loaded", fg="#F57C00")
            
            # Update source type
            if self.is_monitoring:
                source_type = self.logic.get_source_type() if hasattr(self.logic, 'get_source_type') else "Unknown"
                self.lbl_source_type.config(text=f"📹 Source: {source_type}")
            else:
                self.lbl_source_type.config(text="📹 Source: None")
                
        except Exception as e:
            print(f"Error updating model info display: {e}")
            self.lbl_models_status.config(text="❌ Config error", fg="#D32F2F")

    def toggle_monitoring(self) -> None:
        """Toggle between start, pause, and resume monitoring."""
        if not self.is_monitoring:
            self.show_start_options()
        elif self.is_paused:
            self.resume_monitoring()
        else:
            self.pause_monitoring()

    def show_start_options(self) -> None:
        """Show dialog to choose settings and monitoring source."""
        options_window = tk.Toplevel(self.root)
        options_window.title("Start Monitoring - Configuration")
        options_window.geometry("500x550")
        options_window.configure(bg="#ECEFF1")
        options_window.resizable(False, False)
        options_window.transient(self.root)
        options_window.grab_set()
        
        # Center the window
        options_window.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Title
        title_label = tk.Label(options_window, text="Configure Monitoring Settings", 
                              font=("Arial", 16, "bold"), fg="#263238", bg="#ECEFF1")
        title_label.pack(pady=20)
        
        # Model Settings Section
        model_frame = tk.LabelFrame(options_window, text="Model Settings", 
                                   font=("Arial", 12, "bold"), fg="#263238", bg="#ECEFF1")
        model_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(model_frame, text="YOLO Version:", fg="#263238", bg="#ECEFF1").pack(pady=5)
        yolo_var = tk.StringVar(value=self.logic.get_current_yolo_version())
        yolo_combo = ttk.Combobox(model_frame, textvariable=yolo_var, 
                                 values=self.logic.get_available_yolo_versions(),
                                 state="readonly", width=27)
        yolo_combo.pack(pady=5)
        
        tk.Label(model_frame, text="Classification Backbone:", fg="#263238", bg="#ECEFF1").pack(pady=5)
        backbone_var = tk.StringVar(value=self.logic.get_current_backbone())
        backbone_combo = ttk.Combobox(model_frame, textvariable=backbone_var, 
                                     values=self.logic.get_available_backbones(),
                                     state="readonly", width=27)
        backbone_combo.pack(pady=5)
        
        # Source Settings Section
        source_frame = tk.LabelFrame(options_window, text="Monitoring Source", 
                                    font=("Arial", 12, "bold"), fg="#263238", bg="#ECEFF1")
        source_frame.pack(pady=10, padx=20, fill="x")
        
        source_var = tk.StringVar(value="camera")
        tk.Radiobutton(source_frame, text="Camera", variable=source_var, value="camera",
                      fg="#263238", bg="#ECEFF1").pack(pady=5)
        tk.Radiobutton(source_frame, text="Video File", variable=source_var, value="video",
                      fg="#263238", bg="#ECEFF1").pack(pady=5)
        
        # Camera ID setting (only for camera)
        camera_frame = tk.Frame(source_frame, bg="#ECEFF1")
        camera_frame.pack(pady=5, fill="x")
        tk.Label(camera_frame, text="Camera ID:", fg="#263238", bg="#ECEFF1").pack(side="left")
        camera_id_var = tk.StringVar(value=str(self.logic.config['capture_device']))
        camera_entry = ttk.Entry(camera_frame, textvariable=camera_id_var, width=10)
        camera_entry.pack(side="left", padx=5)
        
        # Video path setting (only for video)
        video_frame = tk.Frame(source_frame, bg="#ECEFF1")
        video_frame.pack(pady=5, fill="x")
        tk.Label(video_frame, text="Video Path:", fg="#263238", bg="#ECEFF1").pack(side="left")
        video_path_var = tk.StringVar()
        video_entry = ttk.Entry(video_frame, textvariable=video_path_var, width=25)
        video_entry.pack(side="left", padx=5)
        ttk.Button(video_frame, text="Browse", 
                  command=lambda: self._browse_video_file(video_path_var)).pack(side="left", padx=5)
        
        # Buttons
        button_frame = tk.Frame(options_window, bg="#ECEFF1")
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Start Monitoring", 
                  command=lambda: self._start_with_settings(options_window, yolo_var, backbone_var, 
                                                          source_var, camera_id_var, video_path_var),
                  style="Green.TButton").pack(side="left", padx=10)
        
        ttk.Button(button_frame, text="Cancel", 
                  command=options_window.destroy,
                  style="Gray.TButton").pack(side="left", padx=10)

    def _browse_video_file(self, video_path_var):
        """Browse for video file."""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*")]
        )
        if file_path:
            video_path_var.set(file_path)
    
    def _start_with_settings(self, window, yolo_var, backbone_var, source_var, camera_id_var, video_path_var):
        """Start monitoring with selected settings."""
        try:
            # Apply model settings first
            new_yolo = yolo_var.get()
            current_yolo = self.logic.get_current_yolo_version()
            if new_yolo != current_yolo:
                success, error = self.logic.change_yolo_model(new_yolo)
                if not success:
                    messagebox.showerror("Error", f"Failed to change YOLO model: {error}")
                    return
            new_backbone = backbone_var.get()
            current_backbone = self.logic.get_current_backbone()
            if new_backbone != current_backbone:
                success, error = self.logic.change_classification_model(new_backbone)
                if not success:
                    messagebox.showerror("Error", f"Failed to change classification model: {error}")
                    return
            
            # Update model info display after changes
            self._update_model_info_display()
            
            # Update camera ID if needed
            if source_var.get() == "camera":
                self.logic.config['capture_device'] = int(camera_id_var.get())
            
            # Close settings window
            window.destroy()
            
            # Start monitoring based on source
            if source_var.get() == "camera":
                self.start_monitoring()
            else:
                video_path = video_path_var.get()
                if not video_path:
                    messagebox.showerror("Error", "Please select a video file")
                    return
                self.start_monitoring_video(video_path)
                
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid camera ID: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start monitoring: {str(e)}")

    def start_monitoring(self) -> None:
        """Start real-time monitoring using camera."""
        success, error = self.logic.start_monitoring()
        if success:
            self._on_monitoring_started("Camera")
        else:
            messagebox.showerror("Error", error)
            
    def start_monitoring_video(self, video_path):
        """Start monitoring using a video file."""
        success, error = self.logic.start_monitoring_video(video_path)
        if success:
            self._on_monitoring_started("Video")
        else:
            messagebox.showerror("Error", error)

    def _on_monitoring_started(self, source_type) -> None:
        """Update UI when monitoring starts with specified source."""
        self.is_monitoring = True
        self.btn_monitor.config(text="⏸ Pause")
        self.btn_evaluate.config(state="disabled")
        self.lbl_status.config(text=f"🚗 Status: Monitoring ({source_type})")
        
        # Update model info display to show models are loaded and source type
        self._update_model_info_display()
        
        self.update_video_thread()
            
    def pause_monitoring(self) -> None:
        """Pause active monitoring."""
        self.is_paused = True
        self.btn_monitor.config(text="▶ Resume")
        self.lbl_status.config(text="🚗 Status: Paused")
        
    def resume_monitoring(self) -> None:
        """Resume paused monitoring."""
        self.is_paused = False
        self.btn_monitor.config(text="⏸ Pause")
        self.lbl_status.config(text=f"🚗 Status: Monitoring ({self.logic.get_source_type()})")
        self.update_video_thread()

    def toggle_evaluation(self) -> None:
        """Toggle performance evaluation mode."""
        if not self.logic.is_evaluating:
            self.start_evaluation()
        else:
            self.stop_evaluation()
            
    def start_evaluation(self) -> None:
        """Start performance evaluation on a video file."""
        # Select video file for evaluation
        video_path = filedialog.askopenfilename(
            title="Select Video for Evaluation",
            initialdir=self.logic.config['video_path'],
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if not video_path:
            return

        # Get ground truth data
        ground_truth_input = simpledialog.askstring(
            "Enter Ground Truth",
            "Enter drowsy time ranges (e.g., '10-15, 20-23'):"
        )
        
        # Parse ground truth input
        ground_truth = []
        if ground_truth_input:
            try:
                intervals = [tuple(map(float, interval.split("-"))) 
                           for interval in ground_truth_input.split(", ")]
                ground_truth = [{"start": start, "end": end} for start, end in intervals]
            except ValueError:
                messagebox.showerror("Error", "Invalid ground truth format!")
                return

        # Start evaluation
        success, error = self.logic.evaluate_performance(video_path, ground_truth)
        if success:
            self.is_monitoring = True
            self.btn_evaluate.config(text="⏹ Stop Evaluation")
            self.btn_monitor.config(state="disabled")
            self.lbl_status.config(text="🚗 Status: Evaluating (Video)")
            
            # Add evaluation info label
            self.eval_info_label = Label(self.frame_video, text="", font=("Arial", 12, "bold"),
                                       fg="yellow", bg="black", anchor="nw")
            self.eval_info_label.place(x=10, y=10)
            
            self.update_video_thread()
        else:
            messagebox.showerror("Error", error)

    def stop_evaluation(self) -> None:
        """Stop performance evaluation and show results."""
        stats, final_stats = self.logic.stop_evaluation()
        
        # Reset UI state
        self.is_paused = False
        self.is_monitoring = False
        self.btn_monitor.config(state="normal", text="▶ Start")
        self.btn_evaluate.config(text="📊 Evaluate Performance")
        self.lbl_status.config(text="🚗 Status: Idle")
        
        # Hide evaluation info label
        if hasattr(self, 'eval_info_label') and self.eval_info_label:
            self.eval_info_label.place_forget()
            
        # Show results
        if final_stats is not None:
            messagebox.showinfo("Success", "Performance evaluation completed. Results saved!")

    def update_video_thread(self) -> None:
        """Start a background thread to update the video feed and UI elements."""
        def update():
            while self.logic.is_monitoring and not self.is_paused:
                try:
                    # Get next frame
                    success, result = self.logic.update_video()
                    if not success:
                        self._handle_video_error()
                        break

                    # Update video display
                    imgtk = result
                    self._safe_update(lambda: self.lbl_video.config(image=imgtk))
                    self.lbl_video.image = imgtk  # Keep reference to prevent garbage collection

                    # Update states and UI
                    eye_state, yawn_state, eye_closed_time, yawn_duration, status_text, alert_triggered = self.logic.update_state()
                    self._update_ui_elements(eye_state, yawn_state, eye_closed_time, yawn_duration, status_text, alert_triggered)
                    
                    # Handle evaluation if active
                    if self.logic.is_evaluating:
                        self._update_evaluation_status()
                    
                    # Brief sleep to reduce CPU usage
                    time.sleep(0.01)
                except Exception as e:
                    print(f"Error in video thread: {str(e)}")
                    break

        # Start the update thread
        threading.Thread(target=update, daemon=True).start()
        
    def _handle_video_error(self):
        """Handle video processing errors."""
        if self.logic.is_evaluating:
            self.stop_evaluation()
        else:
            self.logic.stop_monitoring()
            self.is_monitoring = False
            self._safe_update(lambda: self.btn_monitor.config(state="normal", text="▶ Start"))
            self._safe_update(lambda: self.btn_evaluate.config(state="normal"))
            self._safe_update(lambda: self.lbl_status.config(text="🚗 Status: Idle"))
        
    def _safe_update(self, update_func):
        """Thread-safe UI updates."""
        self.root.after(0, update_func)
        
    def _update_ui_elements(self, eye_state, yawn_state, eye_closed_time, yawn_duration, status_text, alert_triggered):
        """Update all UI elements with new state information."""
        # Update FPS
        self._safe_update(lambda: self.lbl_fps.config(text=f"FPS: {self.logic.get_fps()}"))
        
        # Update eye progress
        eye_progress_value = min(eye_closed_time, self.logic.config['eye_closure_threshold']) / \
                          self.logic.config['eye_closure_threshold'] * 100
        self._safe_update(lambda: self.eye_progress.config(value=eye_progress_value))
        
        # Update yawn progress
        yawn_progress_value = 0
        if yawn_duration > 0:
            yawn_progress_value = min(yawn_duration, self.logic.config['yawn_threshold']) / \
                               self.logic.config['yawn_threshold'] * 100
        self._safe_update(lambda: self.yawn_progress.config(value=yawn_progress_value))
        
        # Update status texts and colors
        self._safe_update(lambda: self.lbl_status.config(text=status_text,
                                                    fg="red" if alert_triggered else "#263238"))
        self._safe_update(lambda: self.lbl_eye.config(text=f"👀 Eyes: {eye_state}"))
        self._safe_update(lambda: self.lbl_yawn.config(text=f"🗣️ Yawn: {yawn_state}"))
        self._safe_update(lambda: self.frame_video.config(bg="red" if alert_triggered else "black"))
        
    def _update_evaluation_status(self):
        """Update evaluation status information."""
        if hasattr(self, 'eval_info_label') and self.eval_info_label:
            elapsed = time.time() - self.logic.eval_start_time
            remaining = max(0, self.logic.eval_duration - elapsed)
            self._safe_update(lambda: self.eval_info_label.config(
                text=f"⏱ Evaluation: {int(remaining)}s | FPS: {self.logic.get_fps()}"
            ))
            
            if elapsed >= self.logic.eval_duration:
                self.stop_evaluation()

    def show_settings(self) -> None:
        """Display settings configuration window."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("450x650")
        settings_window.configure(bg="#ECEFF1")
        settings_window.resizable(False, False)
        settings_window.transient(self.root)  # Make window modal
          # Model Settings
        self._create_settings_section(settings_window, "Model Settings", 10)
        
        tk.Label(settings_window, text="YOLO Version:", fg="#263238", bg="#ECEFF1").pack()
        yolo_var = tk.StringVar(value=self.logic.get_current_yolo_version())
        yolo_combo = ttk.Combobox(settings_window, textvariable=yolo_var, 
                                 values=self.logic.get_available_yolo_versions(),
                                 state="readonly", width=27)
        yolo_combo.pack(pady=5)
        
        tk.Label(settings_window, text="Classification Backbone:", fg="#263238", bg="#ECEFF1").pack()
        backbone_var = tk.StringVar(value=self.logic.get_current_backbone())
        backbone_combo = ttk.Combobox(settings_window, textvariable=backbone_var, 
                                     values=self.logic.get_available_backbones(),
                                     state="readonly", width=27)
        backbone_combo.pack(pady=5)
        
        # Camera/Video Settings
        self._create_settings_section(settings_window, "Camera/Video Settings", 10)
        
        tk.Label(settings_window, text="Camera ID:", fg="#263238", bg="#ECEFF1").pack()
        camera_id = ttk.Entry(settings_window, width=30)
        camera_id.insert(0, str(self.logic.config['capture_device']))
        camera_id.pack(pady=5)

        tk.Label(settings_window, text="Default Video Path:", fg="#263238", bg="#ECEFF1").pack()
        video_path = ttk.Entry(settings_window, width=30)
        video_path.insert(0, self.logic.config['video_path'])
        video_path.pack(pady=5)

        # Alert Settings
        self._create_settings_section(settings_window, "Alert Settings", 20)
        
        tk.Label(settings_window, text="Eye Closure Threshold (seconds):", fg="#263238", bg="#ECEFF1").pack()
        eye_threshold = ttk.Entry(settings_window, width=30)
        eye_threshold.insert(0, str(self.logic.config['eye_closure_threshold']))
        eye_threshold.pack(pady=5)

        tk.Label(settings_window, text="Yawn Duration Threshold (seconds):", fg="#263238", bg="#ECEFF1").pack()
        yawn_threshold = ttk.Entry(settings_window, width=30)
        yawn_threshold.insert(0, str(self.logic.config['yawn_threshold']))
        yawn_threshold.pack(pady=5)

        tk.Label(settings_window, text="Yawn Size Threshold (aspect ratio):", fg="#263238", bg="#ECEFF1").pack()
        yawn_size_threshold = ttk.Entry(settings_window, width=30)
        yawn_size_threshold.insert(0, str(self.logic.config['yawn_size_threshold']))
        yawn_size_threshold.pack(pady=5)

        # Audio Settings
        self._create_settings_section(settings_window, "Audio Settings", 20)
        
        sound_enabled_var = tk.BooleanVar(value=self.logic.config['sound_enabled'])
        tk.Checkbutton(settings_window, text="Enable Alert Sound", variable=sound_enabled_var, 
                      fg="#263238", bg="#ECEFF1").pack(pady=5)
                      
        tk.Label(settings_window, text="Volume:", fg="#263238", bg="#ECEFF1").pack()
        volume_scale = ttk.Scale(settings_window, from_=0, to=1, orient="horizontal", 
                               value=self.logic.config['sound_volume'])
        volume_scale.pack(pady=5)
        
        # Misc Settings
        self._create_settings_section(settings_window, "Other Settings", 20)
        
        save_alerts_var = tk.BooleanVar(value=self.logic.config['save_alerts'])
        tk.Checkbutton(settings_window, text="Save Alert History", 
                      variable=save_alerts_var, fg="#263238", bg="#ECEFF1").pack(pady=10)        # Save button
        ttk.Button(settings_window, text="Save Settings", 
                  command=lambda: self._save_settings(
                      settings_window, yolo_var, backbone_var, camera_id, video_path, eye_threshold,
                      yawn_threshold, yawn_size_threshold, sound_enabled_var,
                      volume_scale, save_alerts_var
                  ), 
                  style="Green.TButton").pack(pady=20)    

    def _create_settings_section(self, parent, title, pady=10):
        """Create a titled section in settings dialog."""
        tk.Label(parent, text=title, font=("Arial", 12, "bold"), 
               fg="#263238", bg="#ECEFF1").pack(pady=pady)    
    def _save_settings(self, window, yolo_var, backbone_var, camera_id, video_path, eye_threshold,
                     yawn_threshold, yawn_size_threshold, sound_enabled_var,
                     volume_scale, save_alerts_var):
        """Save all settings from the settings window."""
        try:
            # Models don't need to be loaded here - we'll just update the config
            # When user starts monitoring, models will be loaded with the selected options
            
            # Update YOLO model config if changed
            new_yolo = yolo_var.get()
            current_yolo = self.logic.get_current_yolo_version()
            if new_yolo != current_yolo:
                self.logic.change_yolo_model(new_yolo)
            
            # Update classification model config if changed
            new_backbone = backbone_var.get()
            current_backbone = self.logic.get_current_backbone()
            if new_backbone != current_backbone:
                self.logic.change_classification_model(new_backbone)
            
            # Update other settings
            self.logic.config['capture_device'] = int(camera_id.get())
            self.logic.config['video_path'] = video_path.get()
            self.logic.config['eye_closure_threshold'] = float(eye_threshold.get())
            self.logic.config['yawn_threshold'] = float(yawn_threshold.get())
            self.logic.config['yawn_size_threshold'] = float(yawn_size_threshold.get())
            self.logic.config['sound_enabled'] = sound_enabled_var.get()
            self.logic.config['sound_volume'] = volume_scale.get()
            self.logic.config['save_alerts'] = save_alerts_var.get()
            
            # Save to file and close window
            self.logic.config_manager.save_config()
            messagebox.showinfo("Success", "Settings saved successfully!")
            window.destroy()
        except ValueError:
            messagebox.showerror("Error", "Invalid value entered. Please check your inputs.")

    def show_alerts(self) -> None:
        """Display alert history in a table view."""
        alerts_window = tk.Toplevel(self.root)
        alerts_window.title("Alert History")
        alerts_window.geometry("900x400")
        alerts_window.configure(bg="#ECEFF1")

        # Create treeview for alert data
        columns = ('Time', 'Alert', 'Eye Closure Time', 'Yawn Duration')
        tree = ttk.Treeview(alerts_window, columns=columns, show='headings')

        # Configure columns
        tree.column('Time', width=150)
        tree.column('Alert', width=300)
        tree.column('Eye Closure Time', width=120)
        tree.column('Yawn Duration', width=120)

        for col in columns:
            tree.heading(col, text=col)

        # Configure tags for row styling
        tree.tag_configure('eye_alert', background='#C8E6C9')
        tree.tag_configure('yawn_alert', background='#FFCDD2')
        
        # Load and display alert data
        self._load_alert_data(tree)
        tree.pack(fill='both', expand=True)

        # Add control buttons
        button_frame = tk.Frame(alerts_window, bg="#ECEFF1")
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Export to CSV", 
                 command=lambda: self._export_alerts(), 
                 style="Blue.TButton").pack(side=tk.LEFT, padx=5)
                 
        ttk.Button(button_frame, text="Clear History", 
                 command=lambda: self._clear_alert_history(tree), 
                 style="Red.TButton").pack(side=tk.LEFT, padx=5)

    def _load_alert_data(self, tree):
        """Load alert history data into the treeview."""
        try:
            with open('alerts/alert_history.json', 'r') as f:
                alerts = json.load(f)
                for alert in alerts:
                    # Format times
                    eye_time = f"{alert['eye_closed_time']:.2f}s" if alert['eye_closed_time'] > 0 else "--"
                    yawn_duration = float(alert.get('yawn_duration', 0))
                    yawn_time = f"{yawn_duration:.2f}s" if yawn_duration > 0 else "--"

                    # Style based on alert type
                    message = alert['message']
                    if "Eyes closed" in message:
                        tag = 'eye_alert'
                        message = f"👀 {message}"
                    else:
                        tag = 'yawn_alert'
                        message = f"🗣️ {message}"

                    # Insert alert data
                    tree.insert('', 'end', values=(
                        alert['timestamp'], 
                        message, 
                        eye_time, 
                        yawn_time
                    ), tags=(tag,))
        except FileNotFoundError:
            tree.insert('', 'end', values=('No data', '', '', ''))

    def _clear_alert_history(self, tree):
        """Clear all alert history data."""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear the alert history?"):
            try:
                # Clear file and update view
                with open('alerts/alert_history.json', 'w') as f:
                    json.dump([], f)
                    
                for item in tree.get_children():
                    tree.delete(item)
                tree.insert('', 'end', values=('No data', '', '', ''))
                
                # Clear runtime history
                self.logic.alert_history = []
                messagebox.showinfo("Success", "Alert history cleared!")
            except Exception as e:
                messagebox.showerror("Error", f"Error clearing history: {str(e)}")

    def _export_alerts(self):
        """Export alert history to CSV file."""
        try:
            # Load alert data
            with open('alerts/alert_history.json', 'r') as f:
                alerts = json.load(f)
                
            if not alerts:
                messagebox.showinfo("Info", "No alert data to export.")
                return
                
            # Generate export filename with timestamp
            export_file = f"alerts/export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Write data to CSV
            with open(export_file, 'w', encoding='utf-8') as f:
                f.write("Time,Alert,Eye Closure Time,Yawn Duration\n")
                for alert in alerts:
                    eye_time = f"{alert['eye_closed_time']:.2f}s" if alert['eye_closed_time'] > 0 else "--"
                    yawn_duration = float(alert.get('yawn_duration', 0))
                    yawn_time = f"{yawn_duration:.2f}s" if yawn_duration > 0 else "--"
                    f.write(f"{alert['timestamp']},{alert['message']},{eye_time},{yawn_time}\n")
                    
            messagebox.showinfo("Success", f"Data exported to {export_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting data: {str(e)}")

    def on_closing(self) -> None:
        """Handle application close event."""
        if messagebox.askokcancel("Exit", "Do you want to exit?"):
            self.logic.stop_monitoring()
            self.root.quit()