# system_evaluator.py
import os
import time
import json
from datetime import datetime
import numpy as np


class SystemPerformanceEvaluator:
    """Evaluates the performance of a driver monitoring system.

    This class tracks and analyzes the performance of a driver monitoring system by logging
    frame-by-frame statistics, comparing predictions against ground truth (if provided), and
    calculating performance metrics such as accuracy, sensitivity, specificity, latency, and FPS.
    Results are saved to a JSON file for further analysis.

    Attributes:
        system (object): The driver monitoring system instance to evaluate.
        stats (dict): Dictionary containing performance statistics, including:
            - total_frames (int): Total number of frames processed.
            - correct_detections (int): Number of correct alert detections.
            - false_positives (int): Number of incorrect alert detections.
            - false_negatives (int): Number of missed alert conditions.
            - latency (list): List of frame processing latencies (in seconds).
            - fps_history (list): List of FPS values recorded during evaluation.
            - alerts_triggered (int): Total number of alerts triggered.
        ground_truth_log (list): List of predicted states with timestamps for comparison.
    """
    
    def __init__(self, monitoring_system: object) -> None:
        """Initialize the SystemPerformanceEvaluator with a monitoring system.

        Args:
            monitoring_system (object): The driver monitoring system instance to evaluate.
                Must provide methods like `get_eye_state()`, `get_yawn_state()`, `get_fps()`,
                and attributes like `alert_active` and `eval_start_time`.

        Initializes the evaluator with empty statistics and a log for ground truth comparison.
        """
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
        self.ground_truth_log = []

    def log_frame(self, frame_start: float) -> None:
        """Log performance statistics for a single frame.

        Args:
            frame_start (float): The timestamp (in seconds) when frame processing started.

        Records frame processing latency, FPS, and the system's predicted state (eye state,
        yawn state, and alert status) at the current timestamp. Updates the total frame count
        and alerts triggered.
        """
        latency = time.time() - frame_start
        self.stats["latency"].append(latency)
        self.stats["total_frames"] += 1
        self.stats["fps_history"].append(self.system.get_fps())

        predicted_state = {
            "eye_state": self.system.get_eye_state(),
            "yawn_state": self.system.get_yawn_state(),
            "alert_active": self.system.alert_active,
            "timestamp": time.time() - self.system.eval_start_time
        }
        self.ground_truth_log.append(predicted_state)
        if predicted_state["alert_active"]:
            self.stats["alerts_triggered"] += 1

    def compare_with_input(self, predicted_state: dict) -> None:
        """Log a predicted state for comparison with ground truth.

        Args:
            predicted_state (dict): A dictionary containing the predicted state with keys:
                - eye_state (str): The predicted eye state (e.g., "Open", "Closed").
                - yawn_state (str): The predicted yawn state (e.g., "Yawn", "No Yawn").
                - alert_active (bool): Whether an alert was triggered.
                - timestamp (float): The timestamp of the prediction (in seconds).

        Adds the predicted state to the ground truth log and increments the alerts triggered
        counter if an alert was active.
        """
        self.ground_truth_log.append(predicted_state)
        if predicted_state["alert_active"]:
            self.stats["alerts_triggered"] += 1
    from typing import Union

    def finalize_evaluation(self, manual_ground_truth: Union[list, None] = None) -> dict:
        """Finalize the evaluation and compute performance metrics.

        Args:
            manual_ground_truth (list, optional): A list of ground truth intervals, where each
                interval is a dictionary with "start" and "end" keys (in seconds) indicating
                periods when an alert should be triggered. Defaults to None.

        Returns:
            dict: A dictionary containing the computed performance metrics:
                - accuracy (float): Proportion of correct alert detections among all alert conditions.
                - sensitivity (float): Proportion of true positives among all actual alert conditions.
                - specificity (float): Proportion of true negatives among all non-alert conditions.
                - avg_latency (float): Average frame processing latency (in seconds).
                - avg_fps (float): Average FPS during evaluation.
                - alerts_triggered (int): Total number of alerts triggered.

        Raises:
            OSError: If there are permission issues or other errors when creating the evaluation
                directory or writing the results file.

        If manual ground truth is provided, compares predictions against it to calculate
        correct detections, false positives, and false negatives. Saves the metrics to a JSON
        file in the "evaluation" directory with a timestamped filename.
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

        os.makedirs("evaluation", exist_ok=True)
        with open(f"evaluation/realtime_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(metrics, f, indent=4)

        return metrics