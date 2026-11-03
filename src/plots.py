import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

COLORS = {"s_only": "#4C78A8", "p_only": "#F58518", "fusion": "#54A24B", "global": "#B279A2", "default": "#999999"}
MODEL_COLORS = {"xgb_hard_labels": "#4C78A8", "xgb_distilled": "#54A24B", "tabpfn": "#E45756", "gbm": "#E45756"}
METHOD_LABELS = {"s_only": "Chỉ s", "p_only": "Chỉ p", "fusion": "Fusion (theo tenant)", "global": "Fusion (τ chung)", "default": "Mặc định (λ 0,5, τ 0,6)"}
MODEL_LABELS = {"xgb_hard_labels": "XGBoost nhãn cứng", "xgb_distilled": "XGBoost + soft-label", "tabpfn": "TabPFN (teacher)", "gbm": "GBM (teacher)"}
VARIANT_LABELS = {
    "label_link": "Link Duplicate/Supersede",
    "hard_negative": "Hard negative (s cao, không link)",
    "conflict": "Mâu thuẫn (có cạnh)",
    "hidden_conflict": "Mâu thuẫn ẩn",
    "paraphrase": "Diễn đạt lại",
    "lexical_overlap": "Trùng từ vựng",
    "same_template": "Cùng mẫu, khác đối tượng",
    "random": "Ngẫu nhiên",
}
TENANT_LABELS = {"tech": "KB kỹ thuật", "policy": "KB chính sách", "all": "Gộp hai tenant"}


def dataset_overview(pairs: pd.DataFrame) -> plt.Figure:
    counts = pairs.groupby(["variant_type", "tenant"]).size().unstack(fill_value=0)
    counts.index = [VARIANT_LABELS.get(v, v) for v in counts.index]
    counts.columns = [TENANT_LABELS.get(c, c) for c in counts.columns]
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.barh(ax=ax)
    ax.set_xlabel("Số cặp")
    ax.set_title("Cặp có nhãn theo loại biến thể")
    fig.tight_layout()
    return fig


def vinli_cosine(sample: pd.DataFrame, summary: dict | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = {"entailment": "Kéo theo", "neutral": "Trung lập", "contradiction": "Mâu thuẫn"}
    bins = np.linspace(sample["cosine"].min(), sample["cosine"].max(), 40)
    for label, name in labels.items():
        ax.hist(sample.loc[sample["label"] == label, "cosine"], bins=bins, alpha=0.5, label=name, density=True)
    title = "ViNLI: cosine BGE-M3 giữa hai câu"
    if summary:
        title += f"\nAUC kéo theo vs mâu thuẫn = {summary['auc_entailment_vs_contradiction']:.2f}"
    ax.set_title(title)
    ax.set_xlabel("cosine")
    ax.set_ylabel("Mật độ")
    ax.legend()
    fig.tight_layout()
    return fig


def grid_heatmap(grid: pd.DataFrame, tenant: str, ax=None, best: dict | None = None):
    from src import settings

    frame = grid[grid["tenant"] == tenant]
    # Display every evaluated (λ, τ) point. Coarse-step filtering is no longer
    # needed after the search itself was reduced to a 9×9 grid.
    table = frame.pivot(index="tau", columns="lambda", values="f1").sort_index()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 7))
    lambdas, taus = table.columns.to_numpy(), table.index.to_numpy()
    mesh = ax.pcolormesh(_edges(lambdas), _edges(taus), table.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1)
    for i, tau in enumerate(taus):
        for j, lam in enumerate(lambdas):
            value = table.iloc[i, j]
            ax.text(lam, tau, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if value > 0.55 else "black")
    ax.scatter([settings.DEFAULT_LAMBDA], [settings.DEFAULT_TAU], marker="X", s=180, c="#F58518", edgecolors="black",
               zorder=3, label=f"Mặc định λ = {settings.DEFAULT_LAMBDA}, τ = {settings.DEFAULT_TAU}")
    if best:
        ax.scatter([best["FUSION_LAMBDA"]], [best["CONFLICT_THRESHOLD_TAU"]], marker="*", s=320, c="red", edgecolors="white",
                   zorder=3, label=f"Tối ưu λ = {best['FUSION_LAMBDA']:.2f}, τ = {best['CONFLICT_THRESHOLD_TAU']:.2f}")
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{v:.1f}" for v in lambdas], fontsize=7)
    ax.set_yticks(taus)
    ax.set_yticklabels([f"{v:.2f}" for v in taus], fontsize=7)
    ax.set_xlabel("λ (0 = chỉ cấu trúc p, 1 = chỉ ngữ nghĩa s)")
    ax.set_ylabel("τ (ngưỡng cảnh báo)")
    ax.set_title(f"F1 trên dev – {TENANT_LABELS.get(tenant, tenant)}")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    plt.colorbar(mesh, ax=ax, label="F1")
    return ax.figure if own else ax


def tenant_heatmaps(grid: pd.DataFrame, config: dict) -> plt.Figure:
    tenants = sorted(config["tenants"])
    fig, axes = plt.subplots(1, len(tenants), figsize=(8 * len(tenants), 7))
    for ax, tenant in zip(np.atleast_1d(axes), tenants):
        grid_heatmap(grid, tenant, ax=ax, best=config["tenants"][tenant])
    fig.suptitle("Grid search λ×τ trên dev (lưới 9×9)")
    fig.tight_layout()
    return fig


def threshold_curves(grid: pd.DataFrame, config: dict) -> plt.Figure:
    from src import settings

    tenants = sorted(config["tenants"])
    fig, axes = plt.subplots(1, len(tenants), figsize=(7 * len(tenants), 4.8), sharey=True)
    for ax, tenant in zip(np.atleast_1d(axes), tenants):
        cfg = config["tenants"][tenant]
        frame = grid[grid["tenant"] == tenant]
        tuned = frame[np.isclose(frame["lambda"], cfg["FUSION_LAMBDA"])].sort_values("tau")
        default = frame[np.isclose(frame["lambda"], settings.DEFAULT_LAMBDA)].sort_values("tau")
        ax.plot(tuned["tau"], tuned["precision"], color="#4C78A8", label="Precision")
        ax.plot(tuned["tau"], tuned["recall"], color="#E45756", label="Recall")
        ax.plot(tuned["tau"], tuned["f1"], color="#54A24B", linestyle="--", linewidth=2, label="F1")
        ax.plot(default["tau"], default["f1"], color="#999999", linestyle=":", label=f"F1 tại λ = {settings.DEFAULT_LAMBDA}")
        tau_best = cfg["CONFLICT_THRESHOLD_TAU"]
        at_best = tuned[np.isclose(tuned["tau"], tau_best)].iloc[0]
        ax.axvline(tau_best, color="red", linestyle=":", label=f"τ tối ưu = {tau_best:.2f}")
        ax.axvline(settings.DEFAULT_TAU, color="#F58518", linestyle=":", label=f"τ mặc định = {settings.DEFAULT_TAU}")
        ax.scatter([tau_best], [at_best["f1"]], color="red", zorder=3)
        ax.annotate(f"P = {at_best['precision']:.2f}\nR = {at_best['recall']:.2f}\nF1 = {at_best['f1']:.2f}",
                    (tau_best, at_best["f1"]), xytext=(10, -45), textcoords="offset points", fontsize=8,
                    bbox={"boxstyle": "round", "fc": "white", "ec": "red"})
        ax.set_title(f"{TENANT_LABELS.get(tenant, tenant)}: P/R/F1 theo τ tại λ = {cfg['FUSION_LAMBDA']:.2f}")
        ax.set_xlabel("τ (ngưỡng cảnh báo)")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    np.atleast_1d(axes)[0].set_ylabel("Giá trị trên dev")
    fig.tight_layout()
    return fig


def _edges(centers: np.ndarray) -> np.ndarray:
    half = np.diff(centers) / 2
    return np.concatenate([[centers[0] - half[0]], centers[:-1] + half, [centers[-1] + half[-1]]])


def lambda_profile(grid: pd.DataFrame) -> plt.Figure:
    best = grid.groupby(["tenant", "lambda"])["f1"].max().unstack(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    for tenant in best.columns:
        ax.plot(best.index, best[tenant], marker="o", label=TENANT_LABELS.get(tenant, tenant))
    ax.set_xlabel("λ (0 = chỉ p, 1 = chỉ s)")
    ax.set_ylabel("F1 tốt nhất trên dev (τ tối ưu)")
    ax.set_title("Ảnh hưởng của λ")
    ax.legend()
    fig.tight_layout()
    return fig


def pr_curves(scored: pd.DataFrame, split: str = "test") -> plt.Figure:
    frame = scored[scored["split"] == split]
    tenants = ["all"] + sorted(frame["tenant"].unique())
    fig, axes = plt.subplots(1, len(tenants), figsize=(5 * len(tenants), 4))
    for ax, tenant in zip(axes, tenants):
        part = frame if tenant == "all" else frame[frame["tenant"] == tenant]
        for method in ("s_only", "p_only", "fusion"):
            precision, recall, _ = precision_recall_curve(part["label"], part[f"score_{method}"])
            ax.plot(recall, precision, label=METHOD_LABELS[method], color=COLORS[method])
            hit = part[f"pred_{method}"].astype(bool)
            tp = (hit & (part["label"] == 1)).sum()
            if hit.any():
                ax.scatter([tp / part["label"].sum()], [tp / hit.sum()], color=COLORS[method], edgecolors="black", zorder=3)
        ax.set_title(f"Precision–Recall trên {split} – {TENANT_LABELS.get(tenant, tenant)}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def score_distribution(scored: pd.DataFrame, tenant: str, split: str = "test") -> plt.Figure:
    frame = scored[(scored["split"] == split) & (scored["tenant"] == tenant)]
    fig, ax = plt.subplots(figsize=(8, 4))
    order = [v for v in VARIANT_LABELS if v in set(frame["variant_type"])]
    data = [frame.loc[frame["variant_type"] == v, "score_fusion"] for v in order]
    ax.boxplot(data, orientation="horizontal", tick_labels=[VARIANT_LABELS[v] for v in order])
    ax.axvline(frame["tau_fusion"].iloc[0], color="red", linestyle="--", label=f"τ = {frame['tau_fusion'].iloc[0]:.2f}")
    ax.set_xlabel("score = λ·s + (1−λ)·p")
    ax.set_title(f"Phân phối score theo loại cặp – {TENANT_LABELS.get(tenant, tenant)} ({split})")
    ax.legend()
    fig.tight_layout()
    return fig


def signal_scatter(scored: pd.DataFrame, split: str = "test") -> plt.Figure:
    frame = scored[scored["split"] == split]
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, part in frame.groupby("variant_type"):
        marker = "o" if part["label"].iloc[0] == 1 else "x"
        ax.scatter(part["s"], part["p"], label=VARIANT_LABELS.get(variant, variant), marker=marker, alpha=0.7)
    ax.set_xlabel("s (cosine BGE-M3)")
    ax.set_ylabel("p (lan truyền đa bước)")
    ax.set_title("Hai tín hiệu của Layer 2 (o = mâu thuẫn, x = không)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def threshold_metrics(scored: pd.DataFrame, lam: float, tau: float, tenant: str | None = None, split: str = "test") -> dict:
    from src.conflict import fuse
    from src.evaluate import binary_metrics

    frame = scored[scored["split"] == split]
    if tenant:
        frame = frame[frame["tenant"] == tenant]
    return binary_metrics(frame["label"].to_numpy(), fuse(frame["s_norm"], frame["p"], lam), tau)


def forecast_curves(predictions: pd.DataFrame, models: list[str]) -> plt.Figure:
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(11, 4))
    y = predictions["label"]
    for model in models:
        p = predictions[f"p_{model}"]
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        roc_ax.plot(fpr, tpr, label=MODEL_LABELS.get(model, model), color=MODEL_COLORS.get(model))
        pr_ax.plot(recall, precision, label=MODEL_LABELS.get(model, model), color=MODEL_COLORS.get(model))
    roc_ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
    roc_ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC trên test (Layer 3)")
    pr_ax.axhline(y.mean(), color="grey", linestyle=":", label="Tỉ lệ dương")
    pr_ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall trên test (Layer 3)")
    roc_ax.legend()
    pr_ax.legend()
    fig.tight_layout()
    return fig


def learning_curve(results: dict) -> plt.Figure:
    curve = results["learning_curve"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["train_logloss"], label="train (target soft-label)")
    ax.plot(curve["valid_logloss"], label="validation (nhãn thật)")
    ax.set(xlabel="Số cây", ylabel="Log loss", title=f"Học của XGBoost student (α = {results['best_alpha']})")
    ax.legend()
    fig.tight_layout()
    return fig


def alpha_selection(results: dict) -> plt.Figure:
    table = pd.DataFrame(results["alpha_selection"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table["alpha"], table["pr_auc"], marker="o", label="PR-AUC")
    ax.plot(table["alpha"], table["roc_auc"], marker="s", label="ROC-AUC")
    ax.axvline(results["best_alpha"], color="red", linestyle="--", label="α được chọn")
    ax.set(xlabel="α (1 = chỉ nhãn thật, 0 = chỉ soft-label)", ylabel="Điểm trên validation", title="Chọn α cho distillation")
    ax.legend()
    fig.tight_layout()
    return fig


def shap_summary(shap_data: dict, max_features: int = 12) -> plt.Figure:
    values = shap_data["values"]
    names = shap_data["feature_names"]
    importance = np.abs(values).mean(axis=0)
    order = np.argsort(importance)[::-1][:max_features][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(order) + 1))
    ax.barh([names[i] for i in order], importance[order], color="#54A24B")
    ax.set(xlabel="mean |SHAP|", title="Đặc trưng ảnh hưởng nhất tới dự báo lỗi thời")
    fig.tight_layout()
    return fig


def shap_waterfall(shap_data: dict, doc_id: str, max_features: int = 8) -> plt.Figure:
    row = int(np.flatnonzero(shap_data["doc_ids"] == doc_id)[0])
    values = shap_data["values"][row]
    names = shap_data["feature_names"]
    features = shap_data["features"][row]
    order = np.argsort(np.abs(values))[::-1][:max_features][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(order) + 1.2))
    colors = ["#E45756" if values[i] > 0 else "#4C78A8" for i in order]
    ax.barh([f"{names[i]} = {features[i]:g}" for i in order], values[order], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    base = float(shap_data["base_value"])
    logit = base + values.sum()
    ax.set(xlabel="Đóng góp SHAP (log-odds)", title=f"{doc_id}: xác suất lỗi thời ≈ {1 / (1 + np.exp(-logit)):.2f}")
    fig.tight_layout()
    return fig


def save(fig: plt.Figure, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
