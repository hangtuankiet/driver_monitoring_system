import tkinter as tk
from tkinter import Label, Button, Frame, ttk, messagebox, filedialog
import cv2
import numpy as np
import pygame
import threading
import time
from PIL import Image, ImageTk
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import logging
import json
from datetime import datetime
import os
from system_evaluator import SystemPerformanceEvaluator


class DriverMonitoringSystem:
    def __init__(self):
        self.setup_logging()
        self.load_config()
        self.initialize_models()
        self.initialize_state_variables()
        self.create_storage_directories()
        self.setup_gui()

        self.sound_enabled = True
        self.alert_sound = None
        self.mixer_initialized = False

        self.is_evaluating = False
        self.eval_start_time = None
        self.eval_duration = 300
        self.eval_info_label = None

    def setup_logging(self):
        logging.basicConfig(
            filename='../logs/driver_monitoring.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_config(self):
        try:
            with open('../json/config.json', 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {
                'yolo_model_path': "../models/yolov10m/train/weights/best.pt",
                'eye_model_path': "../models/eye/model_eye.h5",
                'yawn_model_path': "../models/yawn/model_yawn.h5",
                'alert_sound': "../sound/eawr.wav",
                'eye_closure_threshold': 2.1,
                'capture_device': 0,
                'video_path': "../video/",
                'save_alerts': True,
                'sound_enabled': True,
                'sound_volume': 0.5
            }
            with open('../json/config.json', 'w') as f:
                json.dump(self.config, f, indent=4)

    def initialize_models(self):
        try:
            print("Loading YOLO model...")
            self.yolo_model = YOLO(self.config['yolo_model_path'])
            print("YOLO model loaded successfully.")
            print("Loading eye model...")
            self.eye_model = load_model(self.config['eye_model_path'], compile=False)
            print("Eye model loaded successfully.")
            print("Loading yawn model...")
            self.yawn_model = load_model(self.config['yawn_model_path'], compile=False)
            print("Yawn model loaded successfully.")
            pygame.mixer.init()
            logging.info("Models initialized successfully")
        except FileNotFoundError as e:
            logging.error(f"File not found: {str(e)}")
            messagebox.showerror("Error", f"Model file not found: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error initializing models: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize models: {str(e)}")
            raise

    def start_evaluation(self):
        video_path = filedialog.askopenfilename(
            title="Chọn video để đánh giá",
            initialdir=self.config['video_path'],
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if not video_path:
            return

        ground_truth_input = tk.simpledialog.askstring("Nhập Ground Truth",
                                                       "Nhập khoảng thời gian buồn ngủ (e.g., '10-15, 20-23'):")
        if ground_truth_input:
            try:
                intervals = [tuple(map(float, interval.split("-"))) for interval in ground_truth_input.split(", ")]
                ground_truth = [{"start": start, "end": end} for start, end in intervals]
            except ValueError:
                messagebox.showerror("Lỗi", "Định dạng ground truth không hợp lệ!")
                return
        else:
            ground_truth = []

        if not self.is_monitoring:
            self.start_monitoring_video(video_path)
        self.is_evaluating = True
        self.eval_start_time = time.time()
        self.evaluator = SystemPerformanceEvaluator(self)
        self.evaluator.ground_truth = ground_truth
        if not self.eval_info_label:
            self.eval_info_label = Label(self.frame_video, text="", font=("Arial", 12, "bold"),
                                         fg="yellow", bg="black", anchor="nw")
            self.eval_info_label.place(x=10, y=10)
        self.eval_info_label.config(text=f"⏱ Đánh giá: {self.eval_duration}s | FPS: {self.fps}")
        self.btn_stop_eval.config(state="normal")
        print("Starting performance evaluation on video...")

    def stop_evaluation(self):
        self.is_evaluating = False
        self.root.update()
        start_time = time.time()
        stats = self.evaluator.finalize_evaluation()
        print(f"Time to compute stats: {time.time() - start_time:.2f}s")
        print("Initial results (without ground truth):")
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")

        if hasattr(self.evaluator, 'ground_truth') and self.evaluator.ground_truth:
            final_stats = self.evaluator.finalize_evaluation(self.evaluator.ground_truth)
            print("Results with ground truth from video:")
            for key, value in final_stats.items():
                print(f"{key}: {value:.4f}")

        messagebox.showinfo("Hoàn tất", "Đánh giá hiệu năng đã hoàn tất. Kết quả đã được lưu.")
        if self.eval_info_label:
            self.eval_info_label.place_forget()
        self.btn_stop_eval.config(state="disabled")
        self.stop_monitoring()

    def evaluate_performance(self):
        try:
            self.start_evaluation()
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            self.stop_evaluation()

    def initialize_state_variables(self):
        self.cap = None
        self.is_monitoring = False
        self.eye_closed_time = 0
        self.start_time = None
        self.alert_active = False
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.alert_history = []

    def create_storage_directories(self):
        os.makedirs('../alerts', exist_ok=True)
        os.makedirs('../logs', exist_ok=True)

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("🚗 Hệ thống cảnh báo lái xe thông minh")
        self.root.geometry("1024x768")
        self.root.configure(bg="#ECEFF1")  # Xám nhạt sáng

        self.create_menu()
        self.create_main_frames()
        self.create_video_display()
        self.create_status_indicators()
        self.create_control_buttons()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_menu(self):
        menubar = tk.Menu(self.root, bg="#455A64", fg="#504B38")  # Xanh xám đậm, chữ vàng nhạt
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0, bg="#455A64", fg="#504B38")
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
        self.lbl_status = Label(status_frame, text="🚗 Trạng thái: Chờ", font=("Arial", 14, "bold"), fg="#263238",
                                bg="#ECEFF1")
        self.lbl_status.pack(pady=10)
        self.lbl_eye = Label(status_frame, text="👀 Mắt: --", font=("Arial", 14), fg="#263238", bg="#ECEFF1")
        self.lbl_eye.pack()
        self.lbl_yawn = Label(status_frame, text="🗣️ Ngáp: --", font=("Arial", 14), fg="#263238", bg="#ECEFF1")
        self.lbl_yawn.pack()

    def create_control_buttons(self):
        btn_frame = Frame(self.right_panel, bg="#ECEFF1")
        btn_frame.pack(pady=20)

        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12, "bold"), padding=10)

        self.btn_start = ttk.Button(btn_frame, text="▶ Bắt đầu", command=self.show_start_options, style="Green.TButton")
        self.btn_start.pack(pady=5)
        ttk.Button(btn_frame, text="⚙ Cài đặt", command=self.show_settings, style="Gray.TButton").pack(pady=5)
        ttk.Button(btn_frame, text="📊 Xem lịch sử", command=self.show_alerts, style="Blue.TButton").pack(pady=5)
        ttk.Button(btn_frame, text="📊 Đánh giá hiệu năng", command=self.evaluate_performance,
                   style="Purple.TButton").pack(pady=5)
        self.btn_stop_eval = ttk.Button(btn_frame, text="⏹ Dừng đánh giá", command=self.stop_evaluation,
                                        style="Red.TButton", state="disabled")
        self.btn_stop_eval.pack(pady=5)

        # Style cho các nút với màu chữ vàng nhạt
        style.configure("Green.TButton", background="#4CAF50", foreground="#504B38")
        style.map("Green.TButton", background=[("active", "#388E3C")])
        style.configure("Gray.TButton", background="#757575", foreground="#504B38")
        style.map("Gray.TButton", background=[("active", "#616161")])
        style.configure("Blue.TButton", background="#2196F3", foreground="#504B38")
        style.map("Blue.TButton", background=[("active", "#1976D2")])
        style.configure("Purple.TButton", background="#9C27B0", foreground="#504B38")
        style.map("Purple.TButton", background=[("active", "#7B1FA2")])
        style.configure("Red.TButton", background="#F44336", foreground="#504B38")
        style.map("Red.TButton", background=[("active", "#D32F2F")])
        style.configure("green.Horizontal.TProgressbar", troughcolor="#ECEFF1", background="#4CAF50")
        style.configure("red.Horizontal.TProgressbar", troughcolor="#ECEFF1", background="#F44336")

    def show_start_options(self):
        top = tk.Toplevel(self.root)
        top.title("Chọn nguồn đầu vào")
        top.geometry("300x150")
        top.configure(bg="#ECEFF1")
        top.grab_set()

        Label(top, text="Chọn nguồn đầu vào:", font=("Arial", 12), fg="#263238", bg="#ECEFF1").pack(pady=10)

        ttk.Button(top, text="Camera (Real-time)", command=lambda: [self.start_monitoring(), top.destroy()],
                   style="Green.TButton").pack(pady=5)
        ttk.Button(top, text="Video", command=lambda: [self.start_video_selection(top), top.destroy()],
                   style="Blue.TButton").pack(pady=5)

    def start_video_selection(self, top=None):
        video_path = filedialog.askopenfilename(
            title="Chọn video",
            initialdir=self.config['video_path'],
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if video_path:
            self.start_monitoring_video(video_path)

    def start_monitoring(self):
        try:
            self.cap = cv2.VideoCapture(self.config['capture_device'])
            if not self.cap.isOpened():
                raise Exception("Cannot open camera")
            self.is_monitoring = True
            self.btn_start.config(state="disabled")
            self.btn_stop_eval.config(state="normal")
            self.lbl_status.config(text="🚗 Trạng thái: Đang giám sát (Camera)")
            self.update_video()
            logging.info("Camera monitoring started")
        except Exception as e:
            logging.error(f"Error starting camera monitoring: {str(e)}")
            messagebox.showerror("Error", f"Failed to start camera monitoring: {str(e)}")

    def start_monitoring_video(self, video_path):
        try:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise Exception("Cannot open video file")
            self.is_monitoring = True
            self.btn_start.config(state="disabled")
            self.btn_stop_eval.config(state="normal")
            self.lbl_status.config(text="🚗 Trạng thái: Đang giám sát (Video)")
            self.update_video()
            logging.info("Video monitoring started")
        except Exception as e:
            logging.error(f"Error starting video monitoring: {str(e)}")
            messagebox.showerror("Error", f"Failed to start video monitoring: {str(e)}")

    def stop_monitoring(self):
        self.is_monitoring = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_start.config(state="normal")
        self.btn_stop_eval.config(state="disabled")
        self.lbl_status.config(text="🚗 Trạng thái: Dừng")
        if self.eval_info_label:
            self.eval_info_label.place_forget()
        logging.info("Monitoring stopped")

    def update_video(self):
        if not self.is_monitoring:
            return
        try:
            ret, frame = self.cap.read()
            if not ret:
                if self.is_evaluating:
                    self.stop_evaluation()
                else:
                    self.stop_monitoring()
                return

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))
            self.frame_count += 1
            if time.time() - self.last_time > 1:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_time = time.time()
                self.lbl_fps.config(text=f"FPS: {self.fps}")

            if self.frame_count % 2 == 0:
                self.analyze_frame(frame)
            self.display_frame(frame)

            if self.is_evaluating and self.eval_info_label:
                elapsed = time.time() - self.eval_start_time
                remaining = max(0, self.eval_duration - elapsed)
                self.eval_info_label.config(text=f"⏱ Đánh giá: {int(remaining)}s | FPS: {self.fps}")
                if elapsed >= self.eval_duration:
                    self.stop_evaluation()

            self.root.after(10, self.update_video)
        except Exception as e:
            logging.error(f"Error in video update: {str(e)}")
            self.stop_monitoring()
            messagebox.showerror("Error", f"Video update error: {str(e)}")

    def analyze_frame(self, frame):
        frame_start = time.time()
        results = self.yolo_model(frame)
        eye_state, yawn_state = "Open", "No Yawn"
        eyes_detected, mouth_detected = False, False

        for result in results:
            boxes = result.boxes.data
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box.tolist()
                if conf < 0.5:
                    continue
                label = int(cls)
                obj = frame[int(y1):int(y2), int(x1):int(x2)]
                if obj.size == 0:
                    continue
                try:
                    if label == 0:
                        eyes_detected = True
                        pred = self.eye_model.predict(self.preprocess_image(obj), verbose=0)
                        eye_state = "Open" if pred[0][0] > 0.5 else "Closed"
                    elif label == 1:
                        mouth_detected = True
                        pred = self.yawn_model.predict(self.preprocess_image(obj), verbose=0)
                        yawn_state = "No Yawn" if pred[0][0] > 0.5 else "Yawn"
                except Exception as e:
                    logging.warning(f"Error processing detection: {str(e)}")
                    continue

        self.update_state(eye_state, yawn_state)
        if hasattr(self, 'evaluator') and self.is_evaluating:
            self.evaluator.log_frame(frame_start)

    def update_state(self, eye_state, yawn_state):
        if eye_state == "Closed":
            if self.start_time is None:
                self.start_time = time.time()
            self.eye_closed_time = time.time() - self.start_time
        else:
            self.eye_closed_time = 0
            self.start_time = None

        self.eye_progress['value'] = min(self.eye_closed_time, self.config['eye_closure_threshold']) * 33
        self.yawn_progress['value'] = 100 if yawn_state == "Yawn" else 0

        alert_triggered = False
        alert_message = ""
        if self.eye_closed_time > self.config['eye_closure_threshold']:
            alert_message = f"⚠️ Nhắm mắt quá lâu ({int(self.eye_closed_time)}s)!"
            alert_triggered = True
        elif yawn_state == "Yawn":
            alert_message = "⚠️ Phát hiện ngáp!"
            alert_triggered = True

        if alert_triggered:
            self.trigger_alert(alert_message)
        else:
            self.clear_alert()

        self.lbl_eye.config(text=f"👀 Mắt: {eye_state}")
        self.lbl_yawn.config(text=f"🗣️ Ngáp: {yawn_state}")

    def trigger_alert(self, message):
        if not self.alert_active:
            threading.Thread(target=self.play_alarm).start()
            self.save_alert(message)
            if self.is_evaluating:
                timestamp = time.time() - self.eval_start_time
                print(f"Alert triggered at {timestamp:.2f}s: {message}")
        self.alert_active = True
        self.lbl_status.config(text=message, fg="red")
        self.frame_video.config(bg="red")

    def clear_alert(self):
        self.alert_active = False
        self.lbl_status.config(text="🚗 Trạng thái: Bình thường", fg="#263238")
        self.frame_video.config(bg="black")

    def play_alarm(self):
        try:
            pygame.mixer.music.load(self.config['alert_sound'])
            pygame.mixer.music.play()
        except Exception as e:
            logging.error(f"Error playing alarm: {str(e)}")

    def save_alert(self, message):
        if not self.config['save_alerts']:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_data = {
            'timestamp': timestamp,
            'message': message,
            'eye_closed_time': self.eye_closed_time
        }
        self.alert_history.append(alert_data)
        try:
            with open('../alerts/alert_history.json', 'w') as f:
                json.dump(self.alert_history, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving alert: {str(e)}")

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (800, 600))
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lbl_video.imgtk = imgtk
        self.lbl_video.config(image=imgtk)

    @staticmethod
    def preprocess_image(image):
        try:
            if image.size == 0:
                raise ValueError("Empty image")
            image = cv2.resize(image, (224, 224))
            image = image.astype("float32") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            return image
        except Exception as e:
            logging.error(f"Error in image preprocessing: {str(e)}")
            return None

    def show_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Cài đặt")
        settings_window.geometry("400x500")
        settings_window.configure(bg="#ECEFF1")

        tk.Label(settings_window, text="Cài đặt Camera/Video", font=("Arial", 12, "bold"), fg="#263238",
                 bg="#ECEFF1").pack(pady=10)
        tk.Label(settings_window, text="Camera ID:", fg="#263238", bg="#ECEFF1").pack()
        camera_id = ttk.Entry(settings_window)
        camera_id.insert(0, str(self.config['capture_device']))
        camera_id.pack()

        tk.Label(settings_window, text="Đường dẫn video mặc định:", fg="#263238", bg="#ECEFF1").pack()
        video_path = ttk.Entry(settings_window)
        video_path.insert(0, self.config['video_path'])
        video_path.pack()

        tk.Label(settings_window, text="Cài đặt Cảnh báo", font=("Arial", 12, "bold"), fg="#263238", bg="#ECEFF1").pack(
            pady=10)
        tk.Label(settings_window, text="Ngưỡng thời gian nhắm mắt (giây):", fg="#263238", bg="#ECEFF1").pack()
        eye_threshold = ttk.Entry(settings_window)
        eye_threshold.insert(0, str(self.config['eye_closure_threshold']))
        eye_threshold.pack()

        tk.Label(settings_window, text="Cài đặt Âm thanh", font=("Arial", 12, "bold"), fg="#263238", bg="#ECEFF1").pack(
            pady=10)
        sound_enabled_var = tk.BooleanVar(value=self.config['sound_enabled'])
        tk.Checkbutton(settings_window, text="Bật âm thanh cảnh báo",
                       variable=sound_enabled_var, fg="#263238", bg="#ECEFF1").pack(pady=5)
        tk.Label(settings_window, text="Âm lượng:", fg="#263238", bg="#ECEFF1").pack()
        volume_scale = ttk.Scale(settings_window, from_=0, to=1, orient="horizontal",
                                 value=self.config['sound_volume'])
        volume_scale.pack(pady=5)

        save_alerts_var = tk.BooleanVar(value=self.config['save_alerts'])
        tk.Checkbutton(settings_window, text="Lưu lịch sử cảnh báo",
                       variable=save_alerts_var, fg="#263238", bg="#ECEFF1").pack(pady=10)

        def save_settings():
            try:
                self.config['capture_device'] = int(camera_id.get())
                self.config['video_path'] = video_path.get()
                self.config['eye_closure_threshold'] = float(eye_threshold.get())
                self.config['sound_enabled'] = sound_enabled_var.get()
                self.config['sound_volume'] = volume_scale.get()
                self.config['save_alerts'] = save_alerts_var.get()
                with open('../json/config.json', 'w') as f:
                    json.dump(self.config, f, indent=4)
                self.sound_enabled = self.config['sound_enabled']
                messagebox.showinfo("Thành công", "Đã lưu cài đặt")
                settings_window.destroy()
            except ValueError as e:
                messagebox.showerror("Lỗi", "Giá trị không hợp lệ")

        ttk.Button(settings_window, text="Lưu cài đặt", command=save_settings, style="Green.TButton").pack(pady=20)

    def show_alerts(self):
        alerts_window = tk.Toplevel(self.root)
        alerts_window.title("Lịch sử Cảnh báo")
        alerts_window.geometry("600x400")
        alerts_window.configure(bg="#ECEFF1")

        columns = ('Thời gian', 'Cảnh báo', 'Thời gian nhắm mắt')
        tree = ttk.Treeview(alerts_window, columns=columns, show='headings')
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        try:
            with open('../alerts/alert_history.json', 'r') as f:
                alerts = json.load(f)
                for alert in alerts:
                    tree.insert('', 'end', values=(
                        alert['timestamp'],
                        alert['message'],
                        f"{alert['eye_closed_time']:.1f}s"
                    ))
        except FileNotFoundError:
            tree.insert('', 'end', values=('Không có dữ liệu', '', ''))
        tree.pack(fill='both', expand=True)

        def export_alerts():
            try:
                with open('../alerts/alert_history.json', 'r') as f:
                    alerts = json.load(f)
                export_file = f"../alerts/export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                with open(export_file, 'w', encoding='utf-8') as f:
                    f.write("Thời gian,Cảnh báo,Thời gian nhắm mắt\n")
                    for alert in alerts:
                        f.write(f"{alert['timestamp']},{alert['message']},{alert['eye_closed_time']:.1f}s\n")
                messagebox.showinfo("Thành công", f"Đã xuất dữ liệu sang {export_file}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi xuất dữ liệu: {str(e)}")

        ttk.Button(alerts_window, text="Xuất CSV", command=export_alerts, style="Blue.TButton").pack(pady=10)

    def on_closing(self):
        if messagebox.askokcancel("Thoát", "Bạn có muốn thoát không?"):
            self.stop_monitoring()
            self.root.quit()


def main():
    try:
        app = DriverMonitoringSystem()
        app.root.mainloop()
    except Exception as e:
        logging.critical(f"Critical error: {str(e)}")
        messagebox.showerror("Error", f"Critical error: {str(e)}")


if __name__ == "__main__":
    main()