import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import json
from datetime import datetime
import numpy as np
import cv2


class SystemPerformanceEvaluator:
    def __init__(self, monitoring_system):
        self.system = monitoring_system
        self.stats = {
            "total_frames": 0,
            "correct_detections": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "latency": [],
            "fps_history": [],
            "alerts_triggered": 0
        }
        self.ground_truth_log = []  # Lưu trữ ground truth từ người quan sát

    def log_frame(self, frame_start):
        """Ghi lại thông tin mỗi frame trong quá trình đánh giá"""
        latency = time.time() - frame_start
        self.stats["latency"].append(latency)
        self.stats["total_frames"] += 1
        self.stats["fps_history"].append(self.system.fps)

        predicted_state = {
            "eye_state": self.system.lbl_eye.cget("text").split(": ")[1],
            "yawn_state": self.system.lbl_yawn.cget("text").split(": ")[1],
            "alert_active": self.system.alert_active,
            "timestamp": time.time() - self.system.eval_start_time
        }
        self.ground_truth_log.append(predicted_state)
        if predicted_state["alert_active"]:
            self.stats["alerts_triggered"] += 1

    def run_realtime_evaluation(self, duration=300):
        """
        Đánh giá hệ thống trên camera thật trong thời gian thực.
        - duration: thời gian chạy (giây), mặc định 5 phút.
        """
        print("Starting real-time evaluation... Press 'q' to stop early.")
        self.system.start_monitoring()  # Bật camera

        start_time = time.time()
        while self.system.is_monitoring and (time.time() - start_time < duration):
            # Đo latency mỗi frame
            frame_start = time.time()
            if self.system.cap and self.system.cap.isOpened():
                ret, frame = self.system.cap.read()
                if ret:
                    self.system.analyze_frame(frame)
                    latency = time.time() - frame_start
                    self.stats["latency"].append(latency)
                    self.stats["total_frames"] += 1
                    self.stats["fps_history"].append(self.system.fps)

                    # Lấy trạng thái hiện tại
                    predicted_state = {
                        "eye_state": self.system.lbl_eye.cget("text").split(": ")[1],
                        "yawn_state": self.system.lbl_yawn.cget("text").split(": ")[1],
                        "alert_active": self.system.alert_active,
                        "timestamp": time.time() - start_time
                    }

                    # Ghi lại dự đoán (sẽ so sánh với ground truth sau)
                    self.compare_with_input(predicted_state)

            # Thoát sớm nếu nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.system.stop_monitoring()
        return self.finalize_evaluation()

    def compare_with_input(self, predicted_state):
        """Ghi lại trạng thái dự đoán để so sánh sau"""
        self.ground_truth_log.append(predicted_state)
        if predicted_state["alert_active"]:
            self.stats["alerts_triggered"] += 1

    def finalize_evaluation(self, manual_ground_truth=None):
        """
        Hoàn tất đánh giá bằng cách so sánh với ground truth thủ công.
        - manual_ground_truth: danh sách các khoảng thời gian buồn ngủ thực tế [{'start': 10.0, 'end': 15.0}, ...]
        """
        if manual_ground_truth:
            print("Ground truth provided:", manual_ground_truth)
            for truth in manual_ground_truth:
                for pred in self.ground_truth_log:
                    if truth["start"] <= pred["timestamp"] <= truth["end"]:
                        if pred["alert_active"]:
                            self.stats["correct_detections"] += 1
                        else:
                            self.stats["false_negatives"] += 1
                    elif pred["alert_active"] and not any(
                            t["start"] <= pred["timestamp"] <= t["end"] for t in manual_ground_truth):
                        self.stats["false_positives"] += 1

        total_alert_conditions = self.stats["correct_detections"] + self.stats["false_negatives"]
        total_non_alert = self.stats["total_frames"] - total_alert_conditions
        metrics = {
            "accuracy": (
                        self.stats["correct_detections"] / total_alert_conditions) if total_alert_conditions > 0 else 0,
            "sensitivity": (
                        self.stats["correct_detections"] / total_alert_conditions) if total_alert_conditions > 0 else 0,
            "specificity": (total_non_alert - self.stats[
                "false_positives"]) / total_non_alert if total_non_alert > 0 else 1.0,
            "avg_latency": np.mean(self.stats["latency"]) if self.stats["latency"] else 0,
            "avg_fps": np.mean(self.stats["fps_history"]) if self.stats["fps_history"] else 0,
            "alerts_triggered": self.stats["alerts_triggered"]
        }

        # Lưu kết quả
        os.makedirs("../evaluation", exist_ok=True)
        with open(f"../evaluation/realtime_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(metrics, f, indent=4)

        return metrics