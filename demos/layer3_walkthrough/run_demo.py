"""Layer 3 walkthrough demo: TabPFN/GBM teacher -> XGBoost student obsolescence forecasting.

This script does NOT re-run `python -m src.pipeline layer3`, because that
command overwrites the tracked results in `artifacts/<dataset>/`. Instead it
calls the exact same production functions from `src.forecast` and
`src.forecast_features` directly, step by step, and writes its own output
only under `demos/layer3_walkthrough/output/` so the repo's committed
baseline results are never touched.

Run from the repo root:

    python -m demos.layer3_walkthrough.run_demo
    python -m demos.layer3_walkthrough.run_demo --teacher tabpfn   # slower, downloads weights
    python -m demos.layer3_walkthrough.run_demo --rebuild-table    # recompute features instead of reusing the cached parquet

See demos/layer3_walkthrough/README.md for context.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import settings
from src.evaluate import precision_at_k
from src.forecast import (
    STUDENT_PARAMS,
    make_teacher,
    out_of_fold_teacher,
    ranking_metrics,
    time_per_row,
    train_student,
)
from src.forecast_features import build_forecast_table, encode, snapshot_dates, temporal_split
from src.io_utils import read_json, read_parquet
from src.paths import ARTIFACTS, DATA_PROCESSED
from src.similarity import DocumentVectors

OUT_DIR = Path(__file__).resolve().parent / "output"
SEP = "=" * 78


def banner(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def load_dataset(dataset: str):
    out = ARTIFACTS / dataset
    docs = read_parquet(DATA_PROCESSED / f"{dataset}_docs.parquet")
    edges = read_parquet(DATA_PROCESSED / f"{dataset}_edges.parquet")
    events = read_parquet(DATA_PROCESSED / f"{dataset}_conflict_events.parquet")
    vectors = DocumentVectors.from_npz(dict(np.load(out / "embeddings.npz")))
    config = read_json(out / "config.json")
    return docs, edges, events, vectors, config, out


def get_forecast_table(dataset: str, docs, edges, events, vectors, config, rebuild: bool) -> pd.DataFrame:
    cached = DATA_PROCESSED / f"{dataset}_forecast_table.parquet"
    if cached.exists() and not rebuild:
        print(f"Dùng lại bảng feature đã cache tại {cached.relative_to(Path.cwd())} (bỏ qua, truyền --rebuild-table để tính lại từ đầu).")
        return read_parquet(cached)
    print("Đang tính lại bảng feature point-in-time từ đầu (build_forecast_table)...")
    snapshots = snapshot_dates(docs, settings.FORECAST_HORIZON_DAYS, settings.FORECAST_SNAPSHOT_FREQ)
    table = build_forecast_table(docs, edges, events, vectors, config, snapshots, settings.FORECAST_HORIZON_DAYS)
    table["split"] = temporal_split(table, settings.FORECAST_VALID_SNAPSHOTS, settings.FORECAST_TEST_SNAPSHOTS)
    return table


def main():
    parser = argparse.ArgumentParser(description="Layer 3 walkthrough demo")
    parser.add_argument("--dataset", default="kb", choices=["kb"], help="Dataset đã có layer3 ground-truth đầy đủ (jira/vinli chưa có nhãn superseded_at)")
    parser.add_argument("--teacher", default="gbm", choices=["gbm", "tabpfn"], help="gbm = HistGradientBoostingClassifier (nhanh, không cần tải weight); tabpfn = model thật nhưng chậm hơn và cần tải checkpoint ~lần đầu")
    parser.add_argument("--rebuild-table", action="store_true", help="Tính lại bảng feature từ đầu thay vì dùng file parquet đã cache")
    parser.add_argument("--max-context", type=int, default=None, help="Override TABPFN_MAX_CONTEXT (chỉ ảnh hưởng khi --teacher tabpfn)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = settings.SEED
    max_context = args.max_context or settings.TABPFN_MAX_CONTEXT

    # ------------------------------------------------------------------
    banner("BƯỚC 1/6 — Load dataset đã xử lý (Layer 1/2 artifacts)")
    docs, edges, events, vectors, config, out = load_dataset(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"  docs: {len(docs)} rows ({docs['variant_type'].value_counts().to_dict()})")
    print(f"  edges: {len(edges)} rows, loại: {edges['type'].value_counts().to_dict()}")
    print(f"  conflict_events (đã confirm): {len(events)} rows")
    print(f"  tenants: {list(config['tenants'].keys())}")

    # ------------------------------------------------------------------
    banner("BƯỚC 2/6 — Bảng feature point-in-time (snapshot hàng tháng, không nhìn trước tương lai)")
    table = get_forecast_table(args.dataset, docs, edges, events, vectors, config, args.rebuild_table)
    print(f"Tổng số (document, snapshot) rows: {len(table)}")
    print(f"Khoảng thời gian snapshot: {table['snapshot'].min().date()} -> {table['snapshot'].max().date()}")
    print(f"Tỉ lệ label=1 (sẽ bị supersede trong {settings.FORECAST_HORIZON_DAYS} ngày tới): {table['label'].mean():.4f}")
    print("\nSplit theo thời gian (train/valid/test đều tách theo snapshot, không shuffle ngẫu nhiên):")
    print(table["split"].value_counts())
    print("\nVí dụ 1 dòng feature thật (document, snapshot):")
    sample = table.sample(1, random_state=seed).iloc[0]
    for col in ["doc_id", "snapshot", "tenant", "department", "topic"] + __import__("src.forecast_features", fromlist=["x"]).NUMERIC_FEATURES + ["label"]:
        print(f"  {col:28s} = {sample[col]}")

    # ------------------------------------------------------------------
    banner("BƯỚC 3/6 — Encode feature + chia train/valid/test")
    features = encode(table)
    feature_names = list(features.columns)
    X_all = features.to_numpy(dtype=np.float32)
    y_all = table["label"].to_numpy()
    part = {name: (table["split"] == name).to_numpy() for name in ("train", "valid", "test")}
    X, y = X_all[part["train"]], y_all[part["train"]]
    X_valid, y_valid = X_all[part["valid"]], y_all[part["valid"]]
    X_test, y_test = X_all[part["test"]], y_all[part["test"]]
    print(f"Số feature sau one-hot encode: {len(feature_names)} ({feature_names})")
    print(f"train = {len(y)} rows ({int(y.sum())} positive)")
    print(f"valid = {len(y_valid)} rows ({int(y_valid.sum())} positive)")
    print(f"test  = {len(y_test)} rows ({int(y_test.sum())} positive)")

    # ------------------------------------------------------------------
    banner(f"BƯỚC 4/6 — Teacher ({args.teacher}) train Out-of-Fold trên tập train")
    print("OOF = train 5-fold, mỗi fold teacher chỉ thấy 4 fold kia rồi predict lên fold còn lại.")
    print("=> p_teacher_oof không bị leak nhãn của chính dòng đó, dùng an toàn làm soft-label cho student.")
    started = time.perf_counter()
    p_teacher_oof = out_of_fold_teacher(args.teacher, X, y, max_context, seed)
    teacher_model = make_teacher(args.teacher, max_context, seed).fit(X, y)
    teacher_seconds = time.perf_counter() - started
    print(f"Thời gian train teacher (OOF 5 fold + fit cuối trên toàn bộ train): {teacher_seconds:.2f}s")

    # ------------------------------------------------------------------
    banner("BƯỚC 5/6 — Grid search alpha cho student (XGBoost distillation)")
    print("target = alpha * y_thật + (1 - alpha) * p_teacher_oof ; alpha càng nhỏ -> học theo teacher càng nhiều")
    print(f"alpha grid: {settings.FORECAST_ALPHAS} (alpha=1.0 tương đương KHÔNG distill, chỉ học hard label)")
    alpha_rows, students = [], {}
    for alpha in settings.FORECAST_ALPHAS:
        target = alpha * y + (1 - alpha) * p_teacher_oof
        model = train_student(X, target, X_valid, y_valid, seed)
        students[alpha] = model
        alpha_rows.append({"alpha": alpha, **ranking_metrics(y_valid, model.predict(X_valid))})
    alpha_table = pd.DataFrame(alpha_rows)
    print("\nKết quả trên tập valid (chọn alpha tốt nhất theo pr_auc rồi roc_auc, chỉ trong các alpha < 1.0):")
    print(alpha_table.to_string(index=False))
    distilled = alpha_table[alpha_table["alpha"] < 1.0].sort_values(["pr_auc", "roc_auc"], ascending=False)
    best_alpha = float(distilled.iloc[0]["alpha"])
    hard_label_model, student = students[1.0], students[best_alpha]
    print(f"\n=> best_alpha (distilled) = {best_alpha}")

    # ------------------------------------------------------------------
    banner("BƯỚC 6/6 — Đánh giá trên test set (chỉ chạm 1 lần, cuối cùng)")
    p_teacher_test = teacher_model.predict_proba(X_test)
    models = {
        "xgb_hard_labels (alpha=1.0, không distill)": hard_label_model.predict(X_test),
        f"xgb_distilled (alpha={best_alpha})": student.predict(X_test),
        f"teacher ({args.teacher}) only": p_teacher_test,
    }
    rows = []
    for name, p in models.items():
        m = ranking_metrics(y_test, p)
        rows.append({"model": name, **m})
    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))
    print(
        "\nGhi chú (khớp artifacts/kb/layer3_results.json đã có sẵn trong repo):\n"
        "  - roc_auc đo khả năng xếp hạng toàn bộ tập test.\n"
        "  - pr_auc + precision@10/20 mới là thứ người review thật sự nhìn, vì base rate thấp.\n"
        "  - Distillation KHÔNG phải lúc nào cũng thắng cả hai metric cùng lúc trên dữ liệu nhỏ -\n"
        "    đây là điểm phải trình bày trung thực khi báo cáo, không chỉ show số đẹp nhất."
    )

    # ------------------------------------------------------------------
    banner("SHAP — feature nào đẩy điểm 'sắp lỗi thời' lên cao nhất")
    import shap

    explainer = shap.TreeExplainer(student)
    shap_values = explainer.shap_values(X_test)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    print(importance.head(8).to_string())

    chart_path = OUT_DIR / "layer3_demo_summary.png"
    _save_chart(alpha_table, result_df, importance, chart_path)
    print(f"\nĐã lưu chart tổng hợp: {chart_path}")

    print(f"\nXong. Toàn bộ output demo nằm ở: {OUT_DIR} (không đụng tới artifacts/{args.dataset}/ đã commit).")


def _save_chart(alpha_table: pd.DataFrame, result_df: pd.DataFrame, importance: pd.Series, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(alpha_table["alpha"], alpha_table["roc_auc"], marker="o", label="roc_auc")
    ax.plot(alpha_table["alpha"], alpha_table["pr_auc"], marker="o", label="pr_auc")
    ax.set_xlabel("alpha (distillation weight)")
    ax.set_title("Valid: chọn alpha cho student")
    ax.legend()

    ax = axes[1]
    labels = [r.split(" (")[0] for r in result_df["model"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, result_df["roc_auc"], width=0.4, label="roc_auc")
    ax.bar(x + 0.2, result_df["pr_auc"], width=0.4, label="pr_auc")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_title("Test: hard-label vs distilled vs teacher")
    ax.legend()

    ax = axes[2]
    top = importance.head(8).iloc[::-1]
    ax.barh(top.index, top.values)
    ax.set_title("SHAP mean |value| (student)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
