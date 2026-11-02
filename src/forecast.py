"""Layer 3: TabPFN teacher (offline) distilled into an XGBoost student.

The teacher produces out-of-fold probabilities on the training snapshots; the
student learns alpha * y + (1 - alpha) * p_teacher, with alpha chosen on the
validation snapshots. Test snapshots are touched once, at the end.
"""

import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor

from src import settings
from src.evaluate import precision_at_k
from src.forecast_features import build_forecast_table, encode, snapshot_dates, temporal_split
from src.io_utils import read_json, read_parquet, write_json, write_npz, write_parquet
from src.paths import ARTIFACTS, DATA_PROCESSED
from src.similarity import DocumentVectors

TABPFN_V2_URL = "https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/main/tabpfn-v2-classifier.ckpt"
TABPFN_V2_PATH = ARTIFACTS / "models" / "tabpfn-v2-classifier.ckpt"
STUDENT_PARAMS = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9}


def ensure_tabpfn_weights(path: Path = TABPFN_V2_PATH) -> Path:
    """TabPFN v2 weights are public on Hugging Face; newer releases need a Prior Labs token."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TABPFN_V2_URL, path)
    return path


class TabPFNTeacher:
    name = "tabpfn"

    def __init__(self, max_context: int, seed: int, device: str | None = None):
        self.max_context = max_context
        self.seed = seed
        self.device = device

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNTeacher":
        import torch
        from tabpfn import TabPFNClassifier

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        X, y = _stratified_cap(X, y, self.max_context, self.seed)
        self.model = TabPFNClassifier(device=device, model_path=str(ensure_tabpfn_weights()), random_state=self.seed)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class GradientBoostingTeacher:
    """Fallback teacher when TabPFN cannot be loaded: a large boosted ensemble."""

    name = "gbm"

    def __init__(self, seed: int, **_):
        self.seed = seed

    def fit(self, X, y):
        self.model = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.03, random_state=self.seed).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


def make_teacher(kind: str, max_context: int, seed: int):
    return TabPFNTeacher(max_context, seed) if kind == "tabpfn" else GradientBoostingTeacher(seed)


def _stratified_cap(X, y, limit, seed):
    if len(y) <= limit:
        return X, y
    rng = np.random.default_rng(seed)
    keep = []
    for value in np.unique(y):
        idx = np.flatnonzero(y == value)
        share = max(1, int(round(limit * len(idx) / len(y))))
        keep.append(rng.choice(idx, size=min(share, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]


def out_of_fold_teacher(kind: str, X: np.ndarray, y: np.ndarray, max_context: int, seed: int, n_folds: int = 5) -> np.ndarray:
    oof = np.zeros(len(y))
    for fold, (fit_idx, held_idx) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y)):
        teacher = make_teacher(kind, max_context, seed + fold).fit(X[fit_idx], y[fit_idx])
        oof[held_idx] = teacher.predict_proba(X[held_idx])
    return oof


def train_student(X, target, X_valid, y_valid, seed) -> XGBRegressor:
    """Logistic objective on targets in [0, 1]; XGBClassifier would reject soft labels."""
    model = XGBRegressor(objective="reg:logistic", eval_metric="logloss", random_state=seed, **STUDENT_PARAMS)
    model.fit(X, target, eval_set=[(X, target), (X_valid, y_valid)], verbose=False)
    return model


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision_at_10": precision_at_k(y, p, 10),
        "precision_at_20": precision_at_k(y, p, 20),
        "positives": int(y.sum()),
        "n": int(len(y)),
    }


def time_per_row(predict, X, repeats: int = 3) -> float:
    best = np.inf
    for _ in range(repeats):
        started = time.perf_counter()
        predict(X)
        best = min(best, time.perf_counter() - started)
    return float(best / len(X) * 1000.0)


def run_forecast(dataset: str = "kb", teacher: str = "tabpfn", max_context: int | None = None) -> dict:
    out = ARTIFACTS / dataset
    seed = settings.SEED
    if max_context is None:
        max_context = settings.TABPFN_MAX_CONTEXT
    docs = read_parquet(DATA_PROCESSED / f"{dataset}_docs.parquet")
    edges = read_parquet(DATA_PROCESSED / f"{dataset}_edges.parquet")
    events = read_parquet(DATA_PROCESSED / f"{dataset}_conflict_events.parquet")
    vectors = DocumentVectors.from_npz(dict(np.load(out / "embeddings.npz")))
    config = read_json(out / "config.json")

    snapshots = snapshot_dates(docs, settings.FORECAST_HORIZON_DAYS, settings.FORECAST_SNAPSHOT_FREQ)
    table = build_forecast_table(docs, edges, events, vectors, config, snapshots, settings.FORECAST_HORIZON_DAYS)
    table["split"] = temporal_split(table, settings.FORECAST_VALID_SNAPSHOTS, settings.FORECAST_TEST_SNAPSHOTS)
    write_parquet(table, DATA_PROCESSED / f"{dataset}_forecast_table.parquet")

    features = encode(table)
    feature_names = list(features.columns)
    X_all = features.to_numpy(dtype=np.float32)
    y_all = table["label"].to_numpy()
    part = {name: (table["split"] == name).to_numpy() for name in ("train", "valid", "test")}
    X, y = X_all[part["train"]], y_all[part["train"]]
    X_valid, y_valid = X_all[part["valid"]], y_all[part["valid"]]
    X_test, y_test = X_all[part["test"]], y_all[part["test"]]

    started = time.perf_counter()
    p_teacher_oof = out_of_fold_teacher(teacher, X, y, max_context, seed)
    teacher_model = make_teacher(teacher, max_context, seed).fit(X, y)
    teacher_seconds = time.perf_counter() - started

    alpha_rows, students = [], {}
    for alpha in settings.FORECAST_ALPHAS:
        target = alpha * y + (1 - alpha) * p_teacher_oof
        model = train_student(X, target, X_valid, y_valid, seed)
        students[alpha] = model
        alpha_rows.append({"alpha": alpha, **ranking_metrics(y_valid, model.predict(X_valid))})
    alpha_table = pd.DataFrame(alpha_rows)
    distilled = alpha_table[alpha_table["alpha"] < 1.0].sort_values(["pr_auc", "roc_auc"], ascending=False)
    best_alpha = float(distilled.iloc[0]["alpha"])
    hard_label_model, student = students[1.0], students[best_alpha]

    p_teacher_test = teacher_model.predict_proba(X_test)
    models = {
        "xgb_hard_labels": hard_label_model.predict(X_test),
        "xgb_distilled": student.predict(X_test),
        teacher: p_teacher_test,
    }
    latency = {
        "xgb_hard_labels": time_per_row(hard_label_model.predict, X_test),
        "xgb_distilled": time_per_row(student.predict, X_test),
        teacher: time_per_row(teacher_model.predict_proba, X_test, repeats=1),
    }
    test_rows = [{"model": name, **ranking_metrics(y_test, p), "ms_per_row": latency[name]} for name, p in models.items()]

    joblib.dump({"model": student, "features": feature_names, "alpha": best_alpha}, out / "xgb.joblib")
    test_frame = table[part["test"]].assign(**{f"p_{name}": p for name, p in models.items()})
    write_parquet(test_frame, out / "forecast_test_predictions.parquet")

    latest = table[table["snapshot"] == table["snapshot"].max()].copy()
    X_latest = X_all[(table["snapshot"] == table["snapshot"].max()).to_numpy()]
    latest["p_obsolete"] = student.predict(X_latest)
    write_parquet(latest.sort_values("p_obsolete", ascending=False), out / "forecast_scores.parquet")
    _write_shap(student, X_latest, feature_names, latest["doc_id"].to_numpy(), out)

    results = {
        "dataset": dataset,
        "teacher": teacher,
        "teacher_max_context": max_context,
        "teacher_seconds": teacher_seconds,
        "horizon_days": settings.FORECAST_HORIZON_DAYS,
        "rows": {name: int(mask.sum()) for name, mask in part.items()},
        "positives": {name: int(y_all[mask].sum()) for name, mask in part.items()},
        "snapshots": {name: [str(table.loc[mask, "snapshot"].min().date()), str(table.loc[mask, "snapshot"].max().date())] for name, mask in part.items()},
        "best_alpha": best_alpha,
        "alpha_selection": alpha_table.to_dict(orient="records"),
        "test": test_rows,
        "learning_curve": {
            "train_logloss": student.evals_result()["validation_0"]["logloss"],
            "valid_logloss": student.evals_result()["validation_1"]["logloss"],
        },
        "feature_importance": dict(zip(feature_names, map(float, student.feature_importances_))),
    }
    write_json(results, out / "layer3_results.json")
    return {k: results[k] for k in ("rows", "positives", "best_alpha", "test")}


def _write_shap(model, X, feature_names, doc_ids, out) -> None:
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    write_npz(
        out / "shap_values.npz",
        values=np.asarray(values, dtype=np.float32),
        base_value=np.array(float(np.ravel(explainer.expected_value)[0])),
        features=X.astype(np.float32),
        feature_names=np.array(feature_names),
        doc_ids=np.asarray(doc_ids).astype(str),
    )
