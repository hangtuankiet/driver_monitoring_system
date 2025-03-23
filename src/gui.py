import tkinter as tk
from tkinter import Label, Frame, ttk, messagebox, filedialog
import threading
from .logic import DriverMonitor
from datetime import datetime
import json
import time


class DriverMonitoringGUI:
    """A GUI application for the Smart Driver Monitoring System.

    This class provides a graphical interface for monitoring driver behavior using
    the `DriverMonitor` logic backend. It displays real-time video feed, status indicators,
    and alerts for eye closure and yawning, with options to start/stop monitoring,
    evaluate performance, view alerts, and adjust settings.

    Attributes:
        root (tk.Tk): The main Tkinter window.
        logic (DriverMonitor): The backend logic for driver monitoring.
        is_paused (bool): Indicates if monitoring is paused.
        is_monitoring (bool): Indicates if monitoring is active.
    """

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the DriverMonitoringGUI with the main window and logic backend.

        Args:
            root (tk.Tk): The Tkinter root window for the GUI.

        Sets up the GUI components, including the menu, video display, status indicators,
        and control buttons. Also initializes the `DriverMonitor` instance for backend logic.
        """
        self.root = root
        self.logic = DriverMonitor()
        self.is_paused = False
        self.is_monitoring = False
        self.setup_gui()

    def setup_gui(self) -> None:
        """Set up the main GUI layout and components.

        Configures the window title, size, and background color, and initializes
        the menu, main frames, video display, status indicators, and control buttons.
        Also sets up a window close handler.
        """
        self.root.title("🚗 Smart Driver Monitoring System")
        self.root.geometry("1024x768")
        self.root.configure(bg="#ECEFF1")

        self.create_menu()
        self.create_main_frames()
        self.create_video_display()
        self.create_status_indicators()
        self.create_control_buttons()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_menu(self) -> None:
        """Create the top menu bar with File menu options.

        Adds a File menu with options for settings, viewing alerts, and exiting the application.
        The menu is styled with a custom background and foreground color.
        """
        menubar = tk.Menu(self.root, bg="#455A64", fg="#504B38")
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0, bg="#FFFFFF", fg="#504B38")
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Settings", command=self.show_settings)
        file_menu.add_command(label="View Alerts", command=self.show_alerts)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

    def create_main_frames(self) -> None:
        """Create the main left and right panels for the GUI layout.

        The left panel holds the video display, while the right panel contains
        status indicators and control buttons.
        """
        self.left_panel = Frame(self.root, bg="#ECEFF1")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.right_panel = Frame(self.root, bg="#ECEFF1")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

    def create_video_display(self) -> None:
        """Create the video display area in the left panel.

        Sets up a frame with a black background to display the video feed,
        and adds a label to hold the video frames.
        """
        self.frame_video = Frame(self.left_panel, bg="black", bd=3, relief="ridge")
        self.frame_video.pack(pady=10, fill=tk.BOTH, expand=True)
        self.lbl_video = Label(self.frame_video, bg="black")
        self.lbl_video.pack()

    def create_status_indicators(self) -> None:
        """Create status indicators in the right panel.

        Adds labels and progress bars to display FPS, eye closure duration,
        yawn duration, and overall status. The indicators are styled for clarity.
        """
        status_frame = Frame(self.right_panel, bg="#ECEFF1")
        status_frame.pack(pady=10, fill=tk.X)

        self.lbl_fps = Label(status_frame, text="FPS: 0", font=("Arial", 12), fg="#263238", bg="#ECEFF1")
        self.lbl_fps.pack(pady=5)
        self.eye_progress = ttk.Progressbar(status_frame, length=200, mode='determinate',
                                            style="green.Horizontal.TProgressbar")
        self.eye_progress.pack(pady=5)
        self.yawn_progress = ttk.Progressbar(status_frame, length=200, mode='determinate',
                                             style="red.Horizontal.TProgressbar")
        self.yawn_progress.pack(pady=5)
        self.lbl_status = Label(status_frame, text="🚗 Status: Idle", font=("Arial", 14, "bold"), fg="#263238",
                                bg="#ECEFF1")
        self.lbl_status.pack(pady=10)
        self.lbl_eye = Label(status_frame, text="👀 Eyes: --", font=("Arial", 14), fg="#263238", bg="#ECEFF1")
        self.lbl_eye.pack()
        self.lbl_yawn = Label(status_frame, text="🗣️ Yawn: --", font=("Arial", 14), fg="#263238", bg="#ECEFF1")
        self.lbl_yawn.pack()

    def create_control_buttons(self) -> None:
        """Create control buttons and toolbar in the right panel.

        Adds buttons for starting/pausing monitoring, evaluating performance,
        and accessing settings and alerts. Also configures button styles for a consistent look.
        """
        btn_frame = Frame(self.right_panel, bg="#ECEFF1")
        btn_frame.pack(pady=20)

        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12, "bold"), padding=10)

        self.btn_monitor = ttk.Button(btn_frame, text="▶ Start", command=self.toggle_monitoring,
                                      style="Green.TButton")
        self.btn_monitor.pack(pady=5)

        self.btn_evaluate = ttk.Button(btn_frame, text="📊 Evaluate Performance", command=self.toggle_evaluation,
                                       style="Purple.TButton")
        self.btn_evaluate.pack(pady=5)

        toolbar_frame = Frame(self.right_panel, bg="#ECEFF1")
        toolbar_frame.pack(pady=5)

        ttk.Button(toolbar_frame, text="⚙ Settings", command=self.show_settings, style="Gray.TButton").pack(side=tk.LEFT,
                                                                                                           padx=5)
        ttk.Button(toolbar_frame, text="📊 View Alerts", command=self.show_alerts, style="Blue.TButton").pack(
            side=tk.LEFT, padx=5)

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
        style.configure("green.Horizontal.TProgressbar", troughcolor="#ECEFF1", background="#4CAF50")
        style.configure("red.Horizontal.TProgressbar", troughcolor="#ECEFF1", background="#F44336")

    def toggle_monitoring(self) -> None:
        """Toggle the monitoring state between start, pause, and resume.

        If monitoring is not active, shows input source options. If paused, resumes monitoring.
        If active, pauses monitoring and updates the UI accordingly.
        """
        if not self.is_monitoring:
            self.show_start_options()
        elif self.is_paused:
            self.is_paused = False
            self.btn_monitor.config(text="⏸ Pause")
            self.lbl_status.config(text=f"🚗 Status: Monitoring ({self.logic.get_source_type()})")
            self.update_video_thread()
        else:
            self.is_paused = True
            self.btn_monitor.config(text="▶ Resume")
            self.lbl_status.config(text="🚗 Status: Paused")

    def show_start_options(self) -> None:
        """Display a dialog to select the input source for monitoring.

        Opens a new window with options to start monitoring using a camera or a video file.
        The dialog is modal to ensure user interaction before proceeding.
        """
        top = tk.Toplevel(self.root)
        top.title("Select Input Source")
        top.geometry("300x150")
        top.configure(bg="#ECEFF1")
        top.grab_set()

        Label(top, text="Select Input Source:", font=("Arial", 12), fg="#263238", bg="#ECEFF1").pack(pady=10)

        ttk.Button(top, text="Camera (Real-time)", command=lambda: [self.start_monitoring(), top.destroy()],
                   style="Green.TButton").pack(pady=5)
        ttk.Button(top, text="Video", command=lambda: [self.start_video_selection(top), top.destroy()],
                   style="Blue.TButton").pack(pady=5)

    def start_video_selection(self, top: tk.Toplevel = None) -> None:
        """Open a file dialog to select a video file for monitoring.

        Args:
            top (tk.Toplevel, optional): The parent dialog window to close after selection.

        Initiates monitoring with the selected video file if successful, otherwise shows an error.
        """
        video_path = filedialog.askopenfilename(
            title="Select Video",
            initialdir=self.logic.config['video_path'],
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if video_path:
            success, error = self.logic.start_monitoring_video(video_path)
            if success:
                self.is_monitoring = True
                self.btn_monitor.config(text="⏸ Pause")
                self.btn_evaluate.config(state="disabled")
                self.lbl_status.config(text="🚗 Status: Monitoring (Video)")
                self.update_video_thread()
            else:
                messagebox.showerror("Error", error)

    def start_monitoring(self) -> None:
        """Start real-time monitoring using the camera.

        Initiates camera-based monitoring if successful, otherwise displays an error message.
        Updates the UI to reflect the monitoring state.
        """
        success, error = self.logic.start_monitoring()
        if success:
            self.is_monitoring = True
            self.btn_monitor.config(text="⏸ Pause")
            self.btn_evaluate.config(state="disabled")
            self.lbl_status.config(text="🚗 Status: Monitoring (Camera)")
            self.update_video_thread()
        else:
            messagebox.showerror("Error", error)

    def toggle_evaluation(self) -> None:
        """Toggle performance evaluation mode.

        If not evaluating, prompts for a video file and ground truth data to start evaluation.
        If already evaluating, stops the evaluation and displays results.
        """
        if not self.logic.is_evaluating:
            video_path = filedialog.askopenfilename(
                title="Select Video for Evaluation",
                initialdir=self.logic.config['video_path'],
                filetypes=[("Video files", "*.mp4 *.avi *.mov")]
            )
            if not video_path:
                return

            ground_truth_input = tk.simpledialog.askstring("Enter Ground Truth",
                                                           "Enter drowsy time ranges (e.g., '10-15, 20-23'):")
            if ground_truth_input:
                try:
                    intervals = [tuple(map(float, interval.split("-"))) for interval in ground_truth_input.split(", ")]
                    ground_truth = [{"start": start, "end": end} for start, end in intervals]
                except ValueError:
                    messagebox.showerror("Error", "Invalid ground truth format!")
                    return
            else:
                ground_truth = []

            success, error = self.logic.evaluate_performance(video_path, ground_truth)
            if success:
                self.is_monitoring = True
                self.btn_evaluate.config(text="⏹ Stop Evaluation")
                self.btn_monitor.config(state="disabled")
                self.lbl_status.config(text="🚗 Status: Evaluating (Video)")
                self.eval_info_label = Label(self.frame_video, text="", font=("Arial", 12, "bold"),
                                             fg="yellow", bg="black", anchor="nw")
                self.eval_info_label.place(x=10, y=10)
                self.update_video_thread()
            else:
                messagebox.showerror("Error", error)
        else:
            self.stop_evaluation()

    def stop_evaluation(self) -> None:
        """Stop the performance evaluation and display results.

        Stops the evaluation process, resets the UI, and shows a message with the results
        if available.
        """
        stats, final_stats = self.logic.stop_evaluation()
        self.is_paused = False
        self.is_monitoring = False
        self.btn_monitor.config(state="normal", text="▶ Start")
        self.btn_evaluate.config(state="normal", text="📊 Evaluate Performance")
        self.lbl_status.config(text="🚗 Status: Idle")
        if hasattr(self, 'eval_info_label') and self.eval_info_label:
            self.eval_info_label.place_forget()
        if final_stats is not None:
            messagebox.showinfo("Success", "Performance evaluation completed. Results saved!")

    def update_video_thread(self) -> None:
        """Start a thread to continuously update the video feed and UI.

        Runs a background thread to process video frames, update the video display,
        and refresh status indicators. The thread stops if monitoring is paused or
        an error occurs.
        """
        def update():
            while self.logic.is_monitoring and not self.is_paused:
                success, result = self.logic.update_video()
                if not success:
                    if self.logic.is_evaluating:
                        self.stop_evaluation()
                    else:
                        self.logic.stop_monitoring()
                        self.is_monitoring = False
                        self.btn_monitor.config(state="normal", text="▶ Start")
                        self.btn_evaluate.config(state="normal")
                        self.lbl_status.config(text="🚗 Status: Idle")
                    break

                imgtk = result
                self.root.after(0, lambda: self.lbl_video.config(image=imgtk))
                self.lbl_video.image = imgtk

                eye_state, yawn_state, eye_closed_time, yawn_duration, status_text, alert_triggered = self.logic.update_state()

                self.root.after(0, lambda: self.lbl_fps.config(text=f"FPS: {self.logic.get_fps()}"))

                eye_progress_value = min(eye_closed_time, self.logic.config['eye_closure_threshold']) / \
                                     self.logic.config['eye_closure_threshold'] * 100
                self.root.after(0, lambda: self.eye_progress.config(value=eye_progress_value))

                if yawn_duration > 0:
                    yawn_progress_value = min(yawn_duration, self.logic.config['yawn_threshold']) / self.logic.config[
                        'yawn_threshold'] * 100
                else:
                    yawn_progress_value = 0
                self.root.after(0, lambda: self.yawn_progress.config(value=yawn_progress_value))

                self.root.after(0, lambda: self.lbl_status.config(text=status_text,
                                                                  fg="red" if alert_triggered else "#263238"))
                self.root.after(0, lambda: self.lbl_eye.config(text=f"👀 Eyes: {eye_state}"))
                self.root.after(0, lambda: self.lbl_yawn.config(text=f"🗣️ Yawn: {yawn_state}"))
                self.root.after(0, lambda: self.frame_video.config(bg="red" if alert_triggered else "black"))

                if self.logic.is_evaluating and hasattr(self, 'eval_info_label') and self.eval_info_label:
                    elapsed = time.time() - self.logic.eval_start_time
                    remaining = max(0, self.logic.eval_duration - elapsed)
                    self.root.after(0, lambda: self.eval_info_label.config(
                        text=f"⏱ Evaluation: {int(remaining)}s | FPS: {self.logic.get_fps()}"))
                    if elapsed >= self.logic.eval_duration:
                        self.stop_evaluation()
                        break

                time.sleep(0.01)

        threading.Thread(target=update, daemon=True).start()

    def show_settings(self) -> None:
        """Display a settings window to configure monitoring parameters.

        Opens a modal window allowing the user to adjust camera settings, alert thresholds,
        audio settings, and alert history saving options. Changes are saved to the configuration.
        """
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x600")
        settings_window.configure(bg="#ECEFF1")

        tk.Label(settings_window, text="Camera/Video Settings", font=("Arial", 12, "bold"), fg="#263238",
                 bg="#ECEFF1").pack(pady=10)
        tk.Label(settings_window, text="Camera ID:", fg="#263238", bg="#ECEFF1").pack()
        camera_id = ttk.Entry(settings_window)
        camera_id.insert(0, str(self.logic.config['capture_device']))
        camera_id.pack()

        tk.Label(settings_window, text="Default Video Path:", fg="#263238", bg="#ECEFF1").pack()
        video_path = ttk.Entry(settings_window)
        video_path.insert(0, self.logic.config['video_path'])
        video_path.pack()

        tk.Label(settings_window, text="Alert Settings", font=("Arial", 12, "bold"), fg="#263238", bg="#ECEFF1").pack(
            pady=10)
        tk.Label(settings_window, text="Eye Closure Threshold (seconds):", fg="#263238", bg="#ECEFF1").pack()
        eye_threshold = ttk.Entry(settings_window)
        eye_threshold.insert(0, str(self.logic.config['eye_closure_threshold']))
        eye_threshold.pack()

        tk.Label(settings_window, text="Yawn Duration Threshold (seconds):", fg="#263238", bg="#ECEFF1").pack()
        yawn_threshold = ttk.Entry(settings_window)
        yawn_threshold.insert(0, str(self.logic.config['yawn_threshold']))
        yawn_threshold.pack()

        tk.Label(settings_window, text="Yawn Size Threshold (aspect ratio):", fg="#263238", bg="#ECEFF1").pack()
        yawn_size_threshold = ttk.Entry(settings_window)
        yawn_size_threshold.insert(0, str(self.logic.config['yawn_size_threshold']))
        yawn_size_threshold.pack()

        tk.Label(settings_window, text="Audio Settings", font=("Arial", 12, "bold"), fg="#263238", bg="#ECEFF1").pack(
            pady=10)
        sound_enabled_var = tk.BooleanVar(value=self.logic.config['sound_enabled'])
        tk.Checkbutton(settings_window, text="Enable Alert Sound", variable=sound_enabled_var, fg="#263238",
                       bg="#ECEFF1").pack(pady=5)
        tk.Label(settings_window, text="Volume:", fg="#263238", bg="#ECEFF1").pack()
        volume_scale = ttk.Scale(settings_window, from_=0, to=1, orient="horizontal",
                                 value=self.logic.config['sound_volume'])
        volume_scale.pack(pady=5)

        save_alerts_var = tk.BooleanVar(value=self.logic.config['save_alerts'])
        tk.Checkbutton(settings_window, text="Save Alert History", variable=save_alerts_var, fg="#263238",
                       bg="#ECEFF1").pack(pady=10)

        def save_settings():
            try:
                self.logic.config['capture_device'] = int(camera_id.get())
                self.logic.config['video_path'] = video_path.get()
                self.logic.config['eye_closure_threshold'] = float(eye_threshold.get())
                self.logic.config['yawn_threshold'] = float(yawn_threshold.get())
                self.logic.config['yawn_size_threshold'] = float(yawn_size_threshold.get())
                self.logic.config['sound_enabled'] = sound_enabled_var.get()
                self.logic.config['sound_volume'] = volume_scale.get()
                self.logic.config['save_alerts'] = save_alerts_var.get()
                self.logic.config_manager.save_config()
                messagebox.showinfo("Success", "Settings saved successfully!")
                settings_window.destroy()
            except ValueError as e:
                messagebox.showerror("Error", "Invalid value!")

        ttk.Button(settings_window, text="Save Settings", command=save_settings, style="Green.TButton").pack(pady=20)

    def show_alerts(self) -> None:
        """Display a window showing the history of alerts.

        Opens a new window with a table listing past alerts, including timestamps,
        alert messages, eye closure times, and yawn durations. Provides options to
        export alerts to CSV or clear the history.
        """
        alerts_window = tk.Toplevel(self.root)
        alerts_window.title("Alert History")
        alerts_window.geometry("900x400")
        alerts_window.configure(bg="#ECEFF1")

        columns = ('Time', 'Alert', 'Eye Closure Time', 'Yawn Duration')
        tree = ttk.Treeview(alerts_window, columns=columns, show='headings')

        tree.column('Time', width=150)
        tree.column('Alert', width=300)
        tree.column('Eye Closure Time', width=120)
        tree.column('Yawn Duration', width=120)

        for col in columns:
            tree.heading(col, text=col)

        tree.tag_configure('eye_alert', background='#C8E6C9')
        tree.tag_configure('yawn_alert', background='#FFCDD2')

        try:
            with open('alerts/alert_history.json', 'r') as f:
                alerts = json.load(f)
                for alert in alerts:
                    eye_time = f"{alert['eye_closed_time']:.2f}s" if alert['eye_closed_time'] > 0 else "--"
                    yawn_duration = float(alert.get('yawn_duration', 0))
                    yawn_time = f"{yawn_duration:.2f}s" if yawn_duration > 0 else "--"

                    message = alert['message']
                    if "Eyes closed" in message:
                        tag = 'eye_alert'
                        message = f"👀 {message}"
                    else:
                        tag = 'yawn_alert'
                        message = f"🗣️ {message}"

                    tree.insert('', 'end', values=(alert['timestamp'], message, eye_time, yawn_time), tags=(tag,))
        except FileNotFoundError:
            tree.insert('', 'end', values=('No data', '', '', ''))
        tree.pack(fill='both', expand=True)

        def clear_history():
            if messagebox.askyesno("Confirm", "Are you sure you want to clear the alert history?"):
                try:
                    with open('alerts/alert_history.json', 'w') as f:
                        json.dump([], f)
                    for item in tree.get_children():
                        tree.delete(item)
                    tree.insert('', 'end', values=('No data', '', '', ''))
                    messagebox.showinfo("Success", "Alert history cleared!")
                except Exception as e:
                    messagebox.showerror("Error", f"Error clearing history: {str(e)}")

        def export_alerts():
            try:
                with open('alerts/alert_history.json', 'r') as f:
                    alerts = json.load(f)
                export_file = f"alerts/export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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

        button_frame = tk.Frame(alerts_window, bg="#ECEFF1")
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Export to CSV", command=export_alerts, style="Blue.TButton").pack(side=tk.LEFT,
                                                                                                         padx=5)
        ttk.Button(button_frame, text="Clear History", command=clear_history, style="Red.TButton").pack(side=tk.LEFT,
                                                                                                        padx=5)

    def on_closing(self) -> None:
        """Handle the window close event.

        Prompts the user to confirm exiting the application. If confirmed, stops
        monitoring and closes the application.
        """
        if messagebox.askokcancel("Exit", "Do you want to exit?"):
            self.logic.stop_monitoring()
            self.root.quit()