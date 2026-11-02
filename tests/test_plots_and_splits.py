import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from src import plots
from src.splits import assign_group_splits, rebalance_negatives, top_up_random_negatives


def _scored(n=60, seed=0):
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    variants = np.where(label == 1, "conflict", "random")
    s = 0.6 + 0.3 * label + rng.normal(0, 0.05, n)
    p = 0.3 * label * rng.random(n)
    frame = pd.DataFrame({
        "tenant": rng.choice(["tech", "policy"], n), "split": "test", "label": label, "variant_type": variants,
        "s": s, "p": p, "s_norm": np.clip(s, 0, 1),
    })
    for method in ("s_only", "p_only", "fusion", "global"):
        frame[f"score_{method}"] = frame["s_norm"] if method != "p_only" else frame["p"]
        frame[f"tau_{method}"] = 0.75 if method != "p_only" else 0.1
        frame[f"pred_{method}"] = (frame[f"score_{method}"] > frame[f"tau_{method}"]).astype(int)
    return frame


def test_layer2_figures_render():
    scored = _scored()
    grid = pd.DataFrame([
        {"tenant": t, "lambda": l, "tau": tau, "f1": l * tau, "precision": 0.5, "recall": 0.5}
        for t in ("tech", "policy") for l in (0.0, 0.5, 1.0) for tau in (0.3, 0.6, 0.9)
    ])
    config = {"tenants": {t: {"FUSION_LAMBDA": 0.5, "CONFLICT_THRESHOLD_TAU": 0.6} for t in ("tech", "policy")}}
    for fig in (plots.tenant_heatmaps(grid, config), plots.lambda_profile(grid), plots.pr_curves(scored),
                plots.signal_scatter(scored), plots.score_distribution(scored, "tech")):
        assert fig.axes
    metrics = plots.threshold_metrics(scored, lam=1.0, tau=0.75)
    assert metrics["tp"] + metrics["fn"] == scored["label"].sum()


def test_layer3_figures_render():
    rng = np.random.default_rng(1)
    predictions = pd.DataFrame({"label": rng.integers(0, 2, 50), "p_xgb_distilled": rng.random(50)})
    results = {"best_alpha": 0.5, "alpha_selection": [{"alpha": a, "pr_auc": 0.3, "roc_auc": 0.7} for a in (0, 0.5, 1)],
               "learning_curve": {"train_logloss": [0.6, 0.5], "valid_logloss": [0.65, 0.6]}}
    shap_data = {"values": rng.normal(size=(3, 4)), "feature_names": np.array(list("abcd")), "features": rng.random((3, 4)),
                 "doc_ids": np.array(["x", "y", "z"]), "base_value": np.array(-2.0)}
    for fig in (plots.forecast_curves(predictions, ["xgb_distilled"]), plots.alpha_selection(results),
                plots.learning_curve(results), plots.shap_summary(shap_data), plots.shap_waterfall(shap_data, "y")):
        assert fig.axes


def test_random_negatives_are_refilled_inside_each_split():
    groups = [f"g{i}" for i in range(30)]
    docs = pd.DataFrame({"doc_id": groups, "issued_at": pd.date_range("2022-01-01", periods=30), "topic": ["a", "b"] * 15})
    pairs = pd.DataFrame({
        "doc_a": groups, "doc_b": [f"{g}-v2" for g in groups], "tenant": "t", "group_a": groups, "group_b": groups,
        "label": 1, "variant_type": "conflict",
    })
    split = assign_group_splits(pairs, test_fraction=0.3, seed=0)
    filled = rebalance_negatives(top_up_random_negatives(split, docs, 3, seed=0), 3, seed=0)
    for _, frame in filled.groupby("split"):
        assert (frame["label"] == 0).sum() == 3 * frame["label"].sum()
        assert set(frame["group_a"]) | set(frame["group_b"]) <= set(split.loc[split["split"] == frame["split"].iloc[0], ["group_a", "group_b"]].stack())
