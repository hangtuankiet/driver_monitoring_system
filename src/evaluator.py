# evaluator.py
import os
import time
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc


class SystemPerformanceEvaluator:
    """
    Evaluates the performance of a driver monitoring system.
    
    Provides comprehensive metrics and visualization capabilities to assess
    system accuracy, detection speed, and stability.
    """

    def __init__(self, monitoring_system: object) -> None:
        """
        Initialize the SystemPerformanceEvaluator with a monitoring system.
        
        Args:
            monitoring_system: The driver monitoring system to evaluate
        """
        self.system = monitoring_system
        self.stats = {
            "total_frames": 0,
            "correct_detections": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_negatives": 0,
            "latency": [],
            "fps_history": [],
            "alerts_triggered": 0,
            # Separate metrics for eyes and mouth
            "eye_alerts": {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "true_negatives": 0,
            },
            "yawn_alerts": {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "true_negatives": 0,
            },
            # Alert timing and stability metrics
            "time_to_alert": [],
            "alert_durations": [],
            "alert_flickers": 0,
        }
        self.ground_truth = None
        self.ground_truth_log = []
        self.current_alert_start = None
        self.last_alert_state = False
        self.alert_state_changes = 0
        
        # Store alert confidence values for ROC analysis
        self.eye_confidences = []
        self.yawn_confidences = []
        self.ground_truth_eyes = []
        self.ground_truth_yawn = []

    def log_frame(self, frame_start: float) -> None:
        """
        Log performance statistics for a single frame.
        
        Records latency, FPS, driver state, and alert conditions for performance analysis.
        
        Args:
            frame_start: Timestamp when frame processing began
        """
        latency = time.time() - frame_start
        self.stats["latency"].append(latency)
        self.stats["total_frames"] += 1
        self.stats["fps_history"].append(self.system.get_fps())
        
        current_time = time.time() - self.system.eval_start_time
        eye_state = self.system.get_eye_state()
        yawn_state = self.system.get_yawn_state()
        
        # Determine alert state based on eye and mouth conditions
        eye_closed_time = self.system.get_eye_closed_time() if hasattr(self.system, 'get_eye_closed_time') else 0
        yawn_duration = self.system.yawn_duration if hasattr(self.system, 'yawn_duration') else 0
        
        # Define alert conditions based on thresholds
        eye_alert = eye_state == "Closed" and eye_closed_time > self.system.config['eye_closure_threshold'] * 0.7
        yawn_alert = yawn_state == "Yawn" and yawn_duration > self.system.config['yawn_threshold'] * 0.7
        
        # Combined drowsiness indicator
        is_drowsy = eye_alert or yawn_alert
        
        # Store normalized confidence values for ROC analysis
        if hasattr(self.system, 'eye_closed_time'):
            eye_conf = min(eye_closed_time / self.system.config['eye_closure_threshold'], 1.0)
            self.eye_confidences.append(eye_conf)
            
        if hasattr(self.system, 'yawn_duration'):
            yawn_conf = min(yawn_duration / self.system.config['yawn_threshold'], 1.0)
            self.yawn_confidences.append(yawn_conf)
        
        # Track alert state changes for stability analysis
        if is_drowsy != self.last_alert_state:
            self.alert_state_changes += 1
            
            # If alert was active and now inactive, record duration
            if self.last_alert_state and self.current_alert_start is not None:
                alert_duration = current_time - self.current_alert_start
                self.stats["alert_durations"].append(alert_duration)
                
                # Alert flicker detection (<1 second alert duration)
                if alert_duration < 1.0:
                    self.stats["alert_flickers"] += 1
            
            # If new alert starting
            if is_drowsy:
                self.current_alert_start = current_time
                self.stats["alerts_triggered"] += 1
                
        self.last_alert_state = is_drowsy

        predicted_state = {
            "eye_state": eye_state,
            "yawn_state": yawn_state,
            "alert_active": is_drowsy,
            "timestamp": current_time
        }
        self.ground_truth_log.append(predicted_state)

    def compare_with_ground_truth(self, manual_ground_truth: list) -> None:
        """
        Compare predictions with ground truth data and update metrics.
        
        Evaluates system performance by comparing recorded alerts against
        known drowsiness periods from ground truth data.
        
        Args:
            manual_ground_truth: List of dictionaries with start/end times of known drowsiness periods
        """
        self.ground_truth = manual_ground_truth

        for pred in self.ground_truth_log:
            timestamp = pred["timestamp"]
            is_in_ground_truth = any(
                truth["start"] <= timestamp <= truth["end"] for truth in manual_ground_truth
            )
            
            # Overall metrics
            if pred["alert_active"] and is_in_ground_truth:
                self.stats["correct_detections"] += 1
            elif pred["alert_active"] and not is_in_ground_truth:
                self.stats["false_positives"] += 1
            elif not pred["alert_active"] and is_in_ground_truth:
                self.stats["false_negatives"] += 1
            else:  # not alert_active and not in ground truth
                self.stats["true_negatives"] += 1
                
            # Record ground truth states for ROC/PR curves
            self.ground_truth_eyes.append(1 if is_in_ground_truth else 0)
            self.ground_truth_yawn.append(1 if is_in_ground_truth else 0)
                
            # Track time to first alert for each ground truth interval
            for truth in manual_ground_truth:
                if truth["start"] <= timestamp <= truth["end"] and pred["alert_active"]:
                    if not hasattr(truth, "first_detected"):
                        truth["first_detected"] = timestamp
                        self.stats["time_to_alert"].append(timestamp - truth["start"])

    def compute_metrics(self) -> dict:
        """
        Compute comprehensive performance metrics from collected statistics.
        
        Calculates various evaluation metrics including accuracy, sensitivity,
        specificity, precision, F1 score, and specialized alert metrics.
        
        Returns:
            Dictionary containing all computed performance metrics
        """
        tp = self.stats["correct_detections"]
        fp = self.stats["false_positives"]
        fn = self.stats["false_negatives"]
        tn = self.stats["true_negatives"]
        
        # Basic metrics
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # Advanced metrics
        avg_time_to_alert = np.mean(self.stats["time_to_alert"]) if self.stats["time_to_alert"] else 0
        alert_stability = 1.0 - (self.stats["alert_flickers"] / self.stats["alerts_triggered"]) if self.stats["alerts_triggered"] > 0 else 1.0
        
        return {
            "accuracy": accuracy,
            "sensitivity": sensitivity,  # Same as recall
            "specificity": specificity,
            "precision": precision,
            "f1_score": f1_score,
            "avg_latency": np.mean(self.stats["latency"]) if self.stats["latency"] else 0,
            "avg_fps": np.mean(self.stats["fps_history"]) if self.stats["fps_history"] else 0,
            "alerts_triggered": self.stats["alerts_triggered"],
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "avg_time_to_alert": avg_time_to_alert,
            "alert_stability": alert_stability,
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn
            }
        }

    def generate_roc_curve(self, output_dir="evaluation"):
        """
        Generate ROC curves for eye and yawn detection.
        
        Creates and saves receiver operating characteristic curves to visualize
        the performance of the detection system at various threshold settings.
        
        Args:
            output_dir: Directory to save the generated ROC curve plots
        """
        if not (self.eye_confidences and self.ground_truth_eyes):
            return  # Not enough data
            
        # Create ROC curves
        plt.figure(figsize=(12, 5))
        
        # Eye ROC
        if self.eye_confidences and self.ground_truth_eyes:
            plt.subplot(1, 2, 1)
            fpr, tpr, _ = roc_curve(self.ground_truth_eyes, self.eye_confidences)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Eye Detection (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Eye Detection')
            plt.legend(loc='lower right')
        
        # Yawn ROC
        if self.yawn_confidences and self.ground_truth_yawn:
            plt.subplot(1, 2, 2)
            fpr, tpr, _ = roc_curve(self.ground_truth_yawn, self.yawn_confidences)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='orange', label=f'Yawn Detection (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Yawn Detection')
            plt.legend(loc='lower right')
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/roc_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()

    def visualize_confusion_matrix(self, output_dir="evaluation"):
        """
        Visualize confusion matrix of detection results.
        
        Creates and saves a heatmap visualization of the confusion matrix
        to show true/false positive and negative rates.
        
        Args:
            output_dir: Directory to save the generated confusion matrix plot
        """
        plt.figure(figsize=(6, 5))
        cm = np.array([
            [self.stats["true_negatives"], self.stats["false_positives"]],
            [self.stats["false_negatives"], self.stats["correct_detections"]]
        ])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Alert', 'Alert'],
                   yticklabels=['No Alert', 'Alert'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()

    def visualize_metrics_over_time(self, output_dir="evaluation"):
        """
        Visualize system performance metrics over time.
        
        Creates and saves plots showing how FPS, latency, alert durations,
        and response times changed during evaluation.
        
        Args:
            output_dir: Directory to save the generated performance plots
        """
        if not self.stats["fps_history"]:
            return  # No data to visualize
            
        plt.figure(figsize=(12, 8))
        
        # Plot FPS over time
        plt.subplot(2, 2, 1)
        plt.plot(self.stats["fps_history"], label='FPS')
        plt.xlabel('Frame')
        plt.ylabel('FPS')
        plt.title('Performance - FPS Over Time')
        plt.grid(True)
        
        # Plot latency over time
        plt.subplot(2, 2, 2)
        plt.plot(self.stats["latency"], label='Latency', color='orange')
        plt.xlabel('Frame')
        plt.ylabel('Latency (s)')
        plt.title('Performance - Processing Latency')
        plt.grid(True)
        
        # Plot alert durations
        if self.stats["alert_durations"]:
            plt.subplot(2, 2, 3)
            plt.hist(self.stats["alert_durations"], bins=10, color='green')
            plt.xlabel('Duration (s)')
            plt.ylabel('Count')
            plt.title('Alert Durations')
        
        # Plot time to alert
        if self.stats["time_to_alert"]:
            plt.subplot(2, 2, 4)
            plt.hist(self.stats["time_to_alert"], bins=10, color='red')
            plt.xlabel('Time (s)')
            plt.ylabel('Count')
            plt.title('Time to Alert')
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()

    def finalize_evaluation(self, manual_ground_truth: list | None = None) -> dict:
        """
        Finalize evaluation with metrics and visualizations.
        
        Completes the evaluation process by comparing against ground truth (if provided),
        computing all metrics, generating visualizations, and saving results to files.
        
        Args:
            manual_ground_truth: Optional list of ground truth data for comparison
            
        Returns:
            Dictionary containing the computed performance metrics
        """
        if manual_ground_truth:
            self.compare_with_ground_truth(manual_ground_truth)
        
        metrics = self.compute_metrics()
        
        # Generate visualizations
        if manual_ground_truth:
            self.visualize_confusion_matrix()
            self.generate_roc_curve()
        self.visualize_metrics_over_time()
        
        # Save detailed results
        os.makedirs("evaluation", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save metrics to JSON
        with open(f"evaluation/metrics_{timestamp}.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        # Save full evaluation data for later analysis
        full_data = {
            "metrics": metrics,
            "raw_stats": self.stats,
            "ground_truth_log": self.ground_truth_log,
            "ground_truth": manual_ground_truth if manual_ground_truth else []
        }
        
        with open(f"evaluation/full_evaluation_{timestamp}.json", "w") as f:
            json.dump(full_data, f, indent=4)
            
        print(f"Evaluation completed successfully.")
        print(f"Accuracy: {metrics['accuracy']:.4f}, F1-score: {metrics['f1_score']:.4f}")
        print(f"Avg FPS: {metrics['avg_fps']:.2f}, Avg latency: {metrics['avg_latency']*1000:.2f} ms")
        
        return metrics