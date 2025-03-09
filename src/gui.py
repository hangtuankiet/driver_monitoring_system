import tkinter as tk
from tkinter import Label, Frame, ttk, messagebox, filedialog
import threading
from .logic import DriverMonitor
from datetime import datetime
import json
import time

class DriverMonitoringGUI:
    def __init__(self, root):
        self.root = root
        self.logic = DriverMonitor()
        self.is_paused = False
        self.is_monitoring = False
        self.setup_gui()

    def setup_gui(self):
        self.root.title("🚗 Smart Driver Monitoring System")
        self.root.geometry("1024x768")
        self.root.configure(bg="#ECEFF1")

        self.create_menu()
        self.create_main_frames()
        self.create_video_display()
        self.create_status_indicators()
        self.create_control_buttons()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_menu(self):
        menubar = tk.Menu(self.root, bg="#455A64", fg="#504B38")
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0, bg="#FFFFFF", fg="#504B38")
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Settings", command=self.show_settings)
        file_menu.add_command(label="View Alerts", command=self.show_alerts)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

    def create_main_frames(self):
        self.left_panel = Frame(self.root, bg="#ECEFF1")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.right_panel = Frame(self.root, bg="#ECEFF1")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

    def create_video_display(self):
        self.frame_video = Frame(self.left_panel, bg="black", bd=3, relief="ridge")
        self.frame_video.pack(pady=10, fill=tk.BOTH, expand=True)
        self.lbl_video = Label(self.frame_video, bg="black")
        self.lbl_video.pack()

    def create_status_indicators(self):
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

    def create_control_buttons(self):
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
        style.configure("green.Horizontal.TProgressbar", troughcolor="#ECEFF1", background="#4CAF50")
        style.configure("red.Horizontal.TProgressbar", troughcolor="#ECEFF1", background="#F44336")

    def toggle_monitoring(self):
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

    def show_start_options(self):
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

    def start_video_selection(self, top=None):
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

    def start_monitoring(self):
        success, error = self.logic.start_monitoring()
        if success:
            self.is_monitoring = True
            self.btn_monitor.config(text="⏸ Pause")
            self.btn_evaluate.config(state="disabled")
            self.lbl_status.config(text="🚗 Status: Monitoring (Camera)")
            self.update_video_thread()
        else:
            messagebox.showerror("Error", error)

    def toggle_evaluation(self):
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

    def stop_evaluation(self):
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

    def update_video_thread(self):
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

                eye_state, yawn_state, eye_closed_time, status_text, alert_triggered = self.logic.update_state()

                self.root.after(0, lambda: self.lbl_fps.config(text=f"FPS: {self.logic.get_fps()}"))
                self.root.after(0, lambda: self.eye_progress.config(value=min(eye_closed_time, self.logic.config['eye_closure_threshold']) * 33))
                self.root.after(0, lambda: self.yawn_progress.config(value=100 if yawn_state == "Yawn" else 0))
                self.root.after(0, lambda: self.lbl_status.config(text=status_text, fg="red" if alert_triggered else "#263238"))
                self.root.after(0, lambda: self.lbl_eye.config(text=f"👀 Eyes: {eye_state}"))
                self.root.after(0, lambda: self.lbl_yawn.config(text=f"🗣️ Yawn: {yawn_state}"))
                self.root.after(0, lambda: self.frame_video.config(bg="red" if alert_triggered else "black"))

                if self.logic.is_evaluating and hasattr(self, 'eval_info_label') and self.eval_info_label:
                    elapsed = time.time() - self.logic.eval_start_time
                    remaining = max(0, self.logic.eval_duration - elapsed)
                    self.root.after(0, lambda: self.eval_info_label.config(text=f"⏱ Evaluation: {int(remaining)}s | FPS: {self.logic.get_fps()}"))
                    if elapsed >= self.logic.eval_duration:
                        self.stop_evaluation()
                        break

                time.sleep(0.01)

        threading.Thread(target=update, daemon=True).start()

    def show_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x600")  # Tăng chiều cao để chứa thêm trường
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
                self.logic.sound_enabled = self.logic.config['sound_enabled']
                self.logic.yawn_threshold = self.logic.config['yawn_threshold']
                self.logic.yawn_size_threshold = self.logic.config['yawn_size_threshold']
                messagebox.showinfo("Success", "Settings saved successfully!")
                settings_window.destroy()
            except ValueError as e:
                messagebox.showerror("Error", "Invalid value!")

        ttk.Button(settings_window, text="Save Settings", command=save_settings, style="Green.TButton").pack(pady=20)
    def show_alerts(self):
        alerts_window = tk.Toplevel(self.root)
        alerts_window.title("Alert History")
        alerts_window.geometry("600x400")
        alerts_window.configure(bg="#ECEFF1")

        columns = ('Time', 'Alert', 'Eye Closure Time')
        tree = ttk.Treeview(alerts_window, columns=columns, show='headings')
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        try:
            with open('alerts/alert_history.json', 'r') as f:
                alerts = json.load(f)
                for alert in alerts:
                    tree.insert('', 'end', values=(alert['timestamp'], alert['message'], f"{alert['eye_closed_time']:.1f}s"))
        except FileNotFoundError:
            tree.insert('', 'end', values=('No data', '', ''))
        tree.pack(fill='both', expand=True)

        def export_alerts():
            try:
                with open('alerts/alert_history.json', 'r') as f:
                    alerts = json.load(f)
                export_file = f"alerts/export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                with open(export_file, 'w', encoding='utf-8') as f:
                    f.write("Time,Alert,Eye Closure Time\n")
                    for alert in alerts:
                        f.write(f"{alert['timestamp']},{alert['message']},{alert['eye_closed_time']:.1f}s\n")
                messagebox.showinfo("Success", f"Data exported to {export_file}")
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting data: {str(e)}")

        ttk.Button(alerts_window, text="Export to CSV", command=export_alerts, style="Blue.TButton").pack(pady=10)

    def on_closing(self):
        if messagebox.askokcancel("Exit", "Do you want to exit?"):
            self.logic.stop_monitoring()
            self.root.quit()