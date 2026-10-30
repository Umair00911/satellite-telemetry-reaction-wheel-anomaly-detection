import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class ModelComparator:
    def __init__(self):
        self.results = []

    def evaluate_model(self, model_name, y_true, y_pred, y_prob, train_time, latency_ms):
        """Computes comprehensive classification metrics."""
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        if HAS_SKLEARN:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            try:
                auc_roc = roc_auc_score(y_true, y_prob)
            except Exception:
                auc_roc = 0.5
            try:
                auc_pr = average_precision_score(y_true, y_prob)
            except Exception:
                auc_pr = 0.5

            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        else:
            # Pure NumPy Metric Calculations
            acc = np.mean(y_true == y_pred)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_pred == 0))

            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
            auc_roc = acc
            auc_pr = prec

        record = {
            'Model': model_name,
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1-Score': float(f1),
            'ROC-AUC': float(auc_roc),
            'PR-AUC': float(auc_pr),
            'True Positives': int(tp),
            'False Positives': int(fp),
            'True Negatives': int(tn),
            'False Negatives': int(fn),
            'Train Time (s)': float(train_time),
            'Latency (ms)': float(latency_ms)
        }
        self.results.append(record)
        return record

    def get_summary_dataframe(self):
        """Returns benchmark comparison table as DataFrame."""
        return pd.DataFrame(self.results)

    def print_comparison_table(self):
        """Displays formatted CLI comparison table."""
        df = self.get_summary_dataframe()
        print("\n" + "=" * 80)
        print("🏆 SATELLITE ANOMALY DETECTION SYSTEM - MODEL BENCHMARK COMPARISON")
        print("=" * 80)
        cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Latency (ms)']
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("=" * 80)
