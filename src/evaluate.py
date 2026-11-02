import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels).astype(int)
    predictions = (np.asarray(scores) > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(labels, scores)) if labels.any() else float("nan"),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n": int(len(labels)),
    }


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-np.asarray(scores))[:k]
    return float(np.asarray(labels)[order].mean()) if len(order) else float("nan")
