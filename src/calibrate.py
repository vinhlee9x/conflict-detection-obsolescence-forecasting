import numpy as np
import pandas as pd

from src.conflict import fuse, normalize_similarity
from src.evaluate import binary_metrics


def grid_search(s_norm: np.ndarray, p: np.ndarray, labels: np.ndarray, lambdas, taus) -> pd.DataFrame:
    labels = np.asarray(labels).astype(int)
    rows = []
    for lam in lambdas:
        score = fuse(s_norm, p, lam)
        predicted = score[None, :] > np.asarray(taus)[:, None]
        tp = (predicted & (labels == 1)).sum(axis=1)
        fp = (predicted & (labels == 0)).sum(axis=1)
        fn = ((~predicted) & (labels == 1)).sum(axis=1)
        precision = np.divide(tp, tp + fp, out=np.zeros(len(taus)), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros(len(taus)), where=(tp + fn) > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(taus)), where=(precision + recall) > 0)
        rows.append(pd.DataFrame({"lambda": lam, "tau": taus, "precision": precision, "recall": recall, "f1": f1}))
    return pd.concat(rows, ignore_index=True)


def select_best(grid: pd.DataFrame) -> pd.Series:
    """Highest F1; ties go to higher precision (fewer false alarms for reviewers), then higher tau."""
    return grid.sort_values(["f1", "precision", "tau"], ascending=False).iloc[0]


def calibrate(dev: pd.DataFrame, lambdas, taus) -> tuple[dict, pd.DataFrame]:
    """Choose s bounds, lambda and tau on dev pairs only. Also return the
    best tau for the s-only (lambda=1) and p-only (lambda=0) ablations."""
    s_min, s_max = float(dev["s"].min()), float(dev["s"].max())
    s_norm = normalize_similarity(dev["s"], s_min, s_max)
    grid = grid_search(s_norm, dev["p"].to_numpy(), dev["label"].to_numpy(), lambdas, taus)
    best = select_best(grid)
    ablation = {
        name: float(select_best(grid[np.isclose(grid["lambda"], lam)])["tau"])
        for name, lam in (("s_only", 1.0), ("p_only", 0.0))
    }
    config = {
        "FUSION_LAMBDA": float(best["lambda"]),
        "CONFLICT_THRESHOLD_TAU": float(best["tau"]),
        "s_min": s_min,
        "s_max": s_max,
        "dev_f1": float(best["f1"]),
        "ablation_tau": ablation,
    }
    return config, grid
