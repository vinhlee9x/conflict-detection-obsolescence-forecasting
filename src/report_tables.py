"""Write docs/results.md from the artifacts, so every number in the report traces to a file."""

import pandas as pd

from src.io_utils import read_json, read_parquet
from src.paths import ARTIFACTS, DATA_PROCESSED, ROOT
from src.plots import MODEL_LABELS, TENANT_LABELS, VARIANT_LABELS


DOC_LABELS = {
    "root": "Tài liệu gốc",
    "conflict": "Bản sửa có mâu thuẫn",
    "paraphrase": "Bản sửa diễn đạt lại",
    "hidden_conflict": "Tài liệu mâu thuẫn ẩn",
    "lexical_overlap": "Tài liệu trùng từ vựng",
}


def _fmt(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _markdown(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def dataset_section() -> str:
    docs = read_parquet(DATA_PROCESSED / "kb_docs.parquet")
    pairs = read_parquet(ARTIFACTS / "kb" / "scored_pairs.parquet")
    by_doc = docs.groupby(["tenant", "variant_type"]).size().unstack(0, fill_value=0)
    by_doc.index = [DOC_LABELS[v] for v in by_doc.index]
    by_pair = pairs.groupby(["variant_type", "split"]).size().unstack(fill_value=0)
    by_pair.insert(0, "nhãn", pairs.groupby("variant_type")["label"].first())
    by_pair.index = [VARIANT_LABELS.get(v, v) for v in by_pair.index]
    return "\n\n".join([
        "## Dataset (KB tổng hợp)",
        f"{len(docs)} tài liệu, {len(pairs)} cặp có nhãn sau khi chia tập ({int(pairs['label'].sum())} cặp mâu thuẫn).",
        "Tài liệu theo loại và tenant:",
        _markdown(by_doc.rename(columns=TENANT_LABELS).reset_index(names="loại")),
        "Cặp theo loại và tập (dev dùng chọn λ, τ; test chỉ dùng để báo cáo):",
        _markdown(by_pair.reset_index(names="loại cặp")),
    ])


def vinli_section() -> str:
    path = ARTIFACTS / "vinli" / "vinli_summary.json"
    if not path.exists():
        return ""
    s = read_json(path)
    means = ", ".join(f"{k} {v:.3f}" for k, v in s["mean_cosine"].items())
    return "\n\n".join([
        "## ViNLI: cosine BGE-M3 có nhận ra mâu thuẫn không?",
        f"{s['n']} cặp câu tập {s['split']}. Cosine trung bình: {means}.",
        f"AUC kéo theo vs mâu thuẫn = {s['auc_entailment_vs_contradiction']:.3f}; "
        f"AUC mâu thuẫn vs trung lập = {s['auc_contradiction_vs_neutral']:.3f}.",
    ])


def layer2_section(dataset: str = "kb") -> str:
    path = ARTIFACTS / dataset / "layer2_results.json"
    if not path.exists():
        return ""
    r = read_json(path)
    table = pd.DataFrame(r["test"])
    table["scope"] = table["scope"].map(lambda t: TENANT_LABELS.get(t, t))
    table["F1 95% CI"] = table["f1_ci95"].map(lambda ci: f"[{ci[0]:.2f}, {ci[1]:.2f}]")
    columns = {"scope": "phạm vi", "method_label": "phương pháp", "lambda": "λ", "tau": "τ", "precision": "P", "recall": "R", "f1": "F1", "F1 95% CI": "F1 95% CI", "pr_auc": "PR-AUC"}
    table = table[list(columns)].rename(columns=columns).astype(object).fillna("")
    variants = pd.DataFrame(r["test_by_variant"])
    variants["variant_type"] = variants["variant_type"].map(lambda v: VARIANT_LABELS.get(v, v))
    variants = variants[["variant_type", "label", "n", "mean_s", "mean_p", "flagged_s_only", "flagged_p_only", "flagged_fusion"]]
    variants.columns = ["loại cặp", "nhãn", "n", "s TB", "p TB", "bị gắn cờ (chỉ s)", "bị gắn cờ (chỉ p)", "bị gắn cờ (fusion)"]
    configs = "; ".join(
        f"{TENANT_LABELS.get(t, t)}: λ = {c['FUSION_LAMBDA']:.2f}, τ = {c['CONFLICT_THRESHOLD_TAU']:.2f} (F1 dev {c['dev_f1']:.3f})"
        for t, c in r["config"]["tenants"].items()
    )
    return "\n\n".join([
        "## Layer 2: phát hiện mâu thuẫn trên tập test" if dataset == "kb" else
        "## Layer 2 trên Jira (dữ liệu thật, nhãn proxy Duplicate/Supersede), tập test",
        f"Tham số chọn trên dev: {configs}. K = {r['config']['k_hops']}, γ = {r['config']['gamma']}.",
        _markdown(table),
        "Tỉ lệ cặp bị gắn cờ theo từng loại (với cặp nhãn 1 đây là recall, với nhãn 0 là tỉ lệ báo nhầm):",
        _markdown(variants),
    ])


def layer3_section() -> str:
    r = read_json(ARTIFACTS / "kb" / "layer3_results.json")
    table = pd.DataFrame(r["test"])
    table["model"] = table["model"].map(lambda m: MODEL_LABELS.get(m, m))
    table = table[["model", "roc_auc", "pr_auc", "precision_at_10", "precision_at_20", "ms_per_row"]]
    table.columns = ["mô hình", "ROC-AUC", "PR-AUC", "P@10", "P@20", "ms / tài liệu"]
    return "\n\n".join([
        "## Layer 3: dự báo lỗi thời trên các mốc test",
        f"Teacher: {r['teacher']} (context tối đa {r['teacher_max_context']} dòng, {r['teacher_seconds']:.0f} giây). "
        f"Horizon {r['horizon_days']} ngày. Số dòng train/valid/test: {r['rows']['train']}/{r['rows']['valid']}/{r['rows']['test']}, "
        f"số dương: {r['positives']['train']}/{r['positives']['valid']}/{r['positives']['test']}. "
        f"Mốc test: {r['snapshots']['test'][0]} đến {r['snapshots']['test'][1]}. α chọn trên validation: {r['best_alpha']}.",
        _markdown(table),
    ])


def main() -> None:
    parts = [
        "# Kết quả thí nghiệm",
        "File này được sinh tự động bởi `python -m src.report_tables` từ các file trong `artifacts/`. Hình nằm trong `docs/figures/` (notebook 04).",
        dataset_section(),
        vinli_section(),
        layer2_section(),
        layer2_section("jira"),
        layer3_section(),
    ]
    path = ROOT / "docs" / "results.md"
    path.write_text("\n\n".join(p for p in parts if p) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
