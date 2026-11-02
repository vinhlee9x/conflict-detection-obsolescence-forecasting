"""Write the Colab notebooks from cell sources kept here, so they stay reviewable as code."""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"
REPO_URL = "https://github.com/vinhlee9x/conflict-detection-obsolescence-forecasting.git"

SETUP = f'''import os, sys, subprocess
from importlib import metadata
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.exists("/content/repo"):
        subprocess.run(["git", "clone", "{REPO_URL}", "/content/repo"], check=True)
    os.chdir("/content/repo")
    subprocess.run(["git", "pull", "-q", "--ff-only"], check=False)
    core = ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow")
    def installed_versions():
        found = {{}}
        for name in core:
            try:
                found[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                found[name] = None
        return found
    before = installed_versions()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
    if installed_versions() != before:
        print("Đã cài phiên bản mới của numpy/pandas/scipy. Runtime tự khởi động lại, đợi vài giây rồi chạy lại notebook từ đầu.")
        os.kill(os.getpid(), 9)
else:
    while not os.path.exists("src") and os.getcwd() != "/":
        os.chdir("..")
sys.path.insert(0, os.getcwd())
print("Thư mục làm việc:", os.getcwd())'''

DRIVE_SYNC = '''# Tuỳ chọn: giữ data/ và artifacts/ trên Google Drive giữa các session Colab
USE_DRIVE = True
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    target = "/content/drive/MyDrive/conflict-detection"
    for folder in ("data/processed", "artifacts"):
        os.makedirs(f"{target}/{folder}", exist_ok=True)
        subprocess.run(f"cp -rn {folder}/. {target}/{folder}/ && rm -rf {folder} && ln -s {target}/{folder} {folder}", shell=True, check=True)'''


def notebook(cells):
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["accelerator"] = "GPU"
    nb.cells = [new_markdown_cell(c[3:]) if c.startswith("md:") else new_code_cell(c) for c in cells]
    return nb


BUILD_DATASET = [
    "md:# 01. Dựng dataset\n\nSinh KB tổng hợp tiếng Việt (2 tenant), xuất mẫu kiểm tra nhãn, tải ViNLI. Phần Jira là tuỳ chọn.",
    SETUP,
    DRIVE_SYNC,
    "md:## KB tổng hợp",
    '''from src import pipeline
from src.io_utils import read_parquet
from src.paths import DATA_PROCESSED
pipeline.generate_kb()
docs = read_parquet(DATA_PROCESSED / "kb_docs.parquet")
pairs = read_parquet(DATA_PROCESSED / "kb_pairs.parquet")
edges = read_parquet(DATA_PROCESSED / "kb_edges.parquet")
print(len(docs), "tài liệu,", len(pairs), "cặp,", int(pairs.label.sum()), "cặp mâu thuẫn")
pairs.groupby(["tenant", "variant_type", "label"]).size().unstack("tenant")''',
    '''example = pairs[pairs.variant_type == "conflict"].iloc[0]
text = docs.set_index("doc_id")["text"]
print(text[example.doc_a]); print("-" * 60); print(text[example.doc_b])''',
    "md:## Mẫu 100 cặp để nhóm gán nhãn lại\n\nMỗi người thêm một cột `annot_<tên>` (1 = mâu thuẫn, 0 = không), rồi chạy ô tính kappa.",
    '''from src.synth.qa_sample import export_blind_sample
sample = export_blind_sample(docs, pairs, n=100, seed=42)
sample.to_csv(DATA_PROCESSED / "kb_qa_sample.csv", index=False)
sample.head()''',
    '''from pathlib import Path
import pandas as pd
from src.synth.qa_sample import kappa_report, disagreements
annotated_path = DATA_PROCESSED / "kb_qa_sample_annotated.csv"
if annotated_path.exists():
    annotated = pd.read_csv(annotated_path)
    display(kappa_report(annotated, pairs))
    display(disagreements(annotated, pairs))
else:
    print("Chưa có file gán nhãn:", annotated_path)''',
    "md:## ViNLI",
    '''from src.datasets.vinli import load_vinli
vinli_test = load_vinli("test")
vinli_test.label.value_counts()''',
    """md:## Jira (tuỳ chọn)

Dữ liệu thật từ Public Jira Dataset (Montgomery và cộng sự, MSR 2022; Zenodo doi 10.5281/zenodo.5882881). Đặt `RUN_JIRA = True` rồi chạy các ô bên dưới. Notebook tự làm các bước sau:

1. Cài MongoDB 8.0 và `mongorestore` qua apt (Colab chạy Ubuntu, có quyền root).
2. Tải file zip khoảng 5,8 GB, tự tải tiếp nếu bị ngắt, rồi kiểm tra MD5.
3. Chỉ restore các collection trong `JIRA_COLLECTIONS`. Archive được đọc thẳng từ zip, không giải nén. Restore toàn bộ dump cần khoảng 60 GB, vượt dung lượng đĩa Colab.
4. Thống kê số issue và số link nhãn (Duplicate, Supersede, Replace) theo project. Nếu `JIRA_PROJECTS = None` thì chọn `JIRA_PROJECT_COUNT` project có nhiều link nhãn nhất trong khoảng `JIRA_PROJECT_SIZE` issue. Mỗi project là một tenant, nên Layer 2 chọn λ và τ riêng cho từng project.

Mặc định là collection `Spring` với 3 project ROO, SEC, DATAMONGO (khoảng 9.700 issue). Đổi `JIRA_COLLECTIONS` nếu muốn instance khác, ví dụ `Hyperledger`, `MongoDB`, `Sakai`.

Lần đầu mất khoảng 30–40 phút, chủ yếu là tải và đọc hết archive. Nên bật `USE_DRIVE` ở đầu notebook để giữ lại `data/processed/jira_*.parquet`. Session sau chỉ cần đọc parquet, không phải tải lại dump.""",
    '''from pathlib import Path
RUN_JIRA = True
JIRA_COLLECTIONS = ["Spring"]
JIRA_COLLECTION = "Spring"
JIRA_PROJECTS = None
JIRA_PROJECT_COUNT = 3
JIRA_PROJECT_SIZE = (500, 5000)
JIRA_WORKDIR = Path("/content/jira") if IN_COLAB else Path("data/raw/jira")
JIRA_REUSE_PARQUET = True''',
    '''jira_ready = JIRA_REUSE_PARQUET and (DATA_PROCESSED / "jira_docs.parquet").exists()
if RUN_JIRA and not jira_ready:
    from src.datasets import jira_setup
    jira_setup.install_mongodb()
    zip_path = jira_setup.download_dump(JIRA_WORKDIR)
    mongo_uri = jira_setup.start_mongod(JIRA_WORKDIR / "mongo-data")
    print(jira_setup.restore_collections(zip_path, JIRA_COLLECTIONS, uri=mongo_uri))
    overview = jira_setup.project_overview(JIRA_COLLECTION, uri=mongo_uri)
    display(overview.head(20))
    jira_projects = JIRA_PROJECTS or jira_setup.suggest_projects(
        overview, count=JIRA_PROJECT_COUNT, min_issues=JIRA_PROJECT_SIZE[0], max_issues=JIRA_PROJECT_SIZE[1])
    print("Project được chọn:", jira_projects)''',
    '''if RUN_JIRA:
    from src.datasets.jira_link_stats import link_type_counts, resolution_counts, project_summary
    from src.io_utils import write_parquet
    if jira_ready:
        jira_docs = read_parquet(DATA_PROCESSED / "jira_docs.parquet")
        jira_edges = read_parquet(DATA_PROCESSED / "jira_edges.parquet")
    else:
        from src.datasets.jira_export import export_project
        jira_docs, jira_edges = export_project(mongo_uri, jira_setup.DATABASE, JIRA_COLLECTION, project_key=jira_projects)
        write_parquet(jira_docs, DATA_PROCESSED / "jira_docs.parquet")
        write_parquet(jira_edges, DATA_PROCESSED / "jira_edges.parquet")
    display(project_summary(jira_docs, jira_edges))
    display(link_type_counts(jira_edges).head(20))
    display(resolution_counts(jira_docs))''',
    '''if RUN_JIRA:
    from src.datasets.jira_pairs import build_pairs
    pipeline.embed("jira", backend="bge-m3")
    neighbors = read_parquet(pipeline.artifact_dir("jira") / "neighbors.parquet")
    jira_pairs = build_pairs(jira_docs, jira_edges, neighbors, seed=42)
    write_parquet(jira_pairs, DATA_PROCESSED / "jira_pairs.parquet")
    display(jira_pairs.groupby(["variant_type", "label"]).size())''',
]

CALIBRATE = [
    "md:# 02. Layer 1 và Layer 2\n\nEmbed BGE-M3 (nên bật GPU), tính s và p, grid search λ×τ trên dev theo từng tenant, báo cáo trên test.",
    SETUP,
    DRIVE_SYNC,
    '''from src import pipeline
import pandas as pd
summary = pipeline.embed("kb", backend="bge-m3")
summary''',
    "md:## ViNLI: cosine có tách được mâu thuẫn không?",
    '''pipeline.vinli(backend="bge-m3", per_label=1000)''',
    "md:## Calibration và ablation",
    '''results = pipeline.layer2("kb")
pd.DataFrame(results["test"])[["scope", "method_label", "lambda", "tau", "precision", "recall", "f1", "f1_ci95", "pr_auc"]]''',
    '''pd.DataFrame(results["test_by_variant"])''',
    '''{tenant: {k: cfg[k] for k in ("FUSION_LAMBDA", "CONFLICT_THRESHOLD_TAU", "dev_f1")} for tenant, cfg in results["config"]["tenants"].items()}''',
    "md:## Luồng end-to-end: tài liệu mới → candidate → xác nhận\n\nLấy 3 tài liệu ban hành gần nhất làm ví dụ; người dùng xác nhận candidate đầu tiên.",
    '''from src.conflict import confirm_conflicts, detect_candidates
from src.graph import KnowledgeGraph
from src.io_utils import read_parquet
from src.paths import DATA_PROCESSED
docs = read_parquet(DATA_PROCESSED / "kb_docs.parquet")
edges = read_parquet(DATA_PROCESSED / "kb_edges.parquet")
new_ids = docs.sort_values("issued_at").doc_id.tail(3).tolist()
graph = KnowledgeGraph(docs.doc_id.tolist(), edges)
candidates = detect_candidates(new_ids, docs, pipeline.load_vectors("kb"), graph, results["config"], top_k=15)
candidates''',
    '''confirm_conflicts(graph, candidates.head(1))
graph.edges[graph.edges.type == "CONFLICTS_WITH"]''',
    '''import os
if os.path.exists("data/processed/jira_pairs.parquet"):
    jira = pipeline.layer2("jira")
    display(pd.DataFrame(jira["test"])[["scope", "method_label", "precision", "recall", "f1", "pr_auc"]])''',
]

TRAIN_LAYER3 = [
    "md:# 03. Layer 3: dự báo lỗi thời\n\nTabPFN v2 (teacher, cần GPU cho nhanh) sinh soft-label out-of-fold, XGBoost (student) học lại. Chọn α trên validation, báo cáo trên test theo thời gian.",
    SETUP,
    DRIVE_SYNC,
    '''from src import pipeline, settings
import pandas as pd
settings.TABPFN_MAX_CONTEXT = 3000
summary = pipeline.layer3("kb", teacher="tabpfn")
summary["rows"], summary["positives"], summary["best_alpha"]''',
    '''pd.DataFrame(summary["test"])''',
]

RESULTS = [
    "md:# 04. Kết quả và biểu đồ\n\nChỉ đọc file trong `data/processed/` và `artifacts/`, chạy trên CPU trong vài giây. Hình được lưu vào `docs/figures/`.",
    SETUP,
    '''%matplotlib inline
import numpy as np, pandas as pd
from src import plots
from src.io_utils import read_json, read_npz, read_parquet
from src.paths import ARTIFACTS, DATA_PROCESSED, FIGURES
kb = ARTIFACTS / "kb"
pairs = read_parquet(DATA_PROCESSED / "kb_pairs.parquet")
scored = read_parquet(kb / "scored_pairs.parquet")
grid = pd.read_csv(kb / "grid_results.csv")
config = read_json(kb / "config.json")
layer2 = read_json(kb / "layer2_results.json")
layer3 = read_json(kb / "layer3_results.json")
predictions = read_parquet(kb / "forecast_test_predictions.parquet")
shap_data = read_npz(kb / "shap_values.npz")
def show(fig, name):
    plots.save(fig, FIGURES / f"{name}.png")
    display(fig)''',
    "md:## 1. Dataset",
    '''show(plots.dataset_overview(pairs), "01_dataset_overview")
pd.DataFrame(layer2["counts"])''',
    "md:## 2. Động lực: cosine trên ViNLI",
    '''vinli = read_parquet(ARTIFACTS / "vinli" / "vinli_cosine.parquet")
vinli_summary = read_json(ARTIFACTS / "vinli" / "vinli_summary.json")
show(plots.vinli_cosine(vinli, vinli_summary), "02_vinli_cosine")
vinli_summary''',
    "md:## 3. Layer 2: grid search λ×τ và ablation",
    '''show(plots.tenant_heatmaps(grid, config), "03_grid_heatmaps")''',
    '''show(plots.lambda_profile(grid), "04_lambda_profile")''',
    '''table = pd.DataFrame(layer2["test"])
table[["scope", "method_label", "lambda", "tau", "precision", "recall", "f1", "f1_ci95", "pr_auc"]].round(3)''',
    '''show(plots.pr_curves(scored), "05_pr_curves")''',
    '''show(plots.signal_scatter(scored), "06_signal_scatter")''',
    '''for tenant in sorted(scored.tenant.unique()):
    show(plots.score_distribution(scored, tenant), f"07_score_distribution_{tenant}")
pd.DataFrame(layer2["test_by_variant"]).round(2)''',
    "md:## 4. Layer 3: dự báo lỗi thời",
    '''pd.DataFrame(layer3["test"]).round(3)''',
    '''models = [row["model"] for row in layer3["test"]]
show(plots.forecast_curves(predictions, models), "08_forecast_curves")''',
    '''show(plots.alpha_selection(layer3), "09_alpha_selection")
show(plots.learning_curve(layer3), "10_learning_curve")''',
    '''show(plots.shap_summary(shap_data), "11_shap_summary")
top = read_parquet(kb / "forecast_scores.parquet").head(5)
display(top[["doc_id", "tenant", "topic", "age_days", "n_open_candidates", "n_confirmed_conflicts", "p_obsolete"]])
show(plots.shap_waterfall(shap_data, top.doc_id.iloc[0]), "12_shap_waterfall")''',
    "md:## 5. Tuỳ chọn: tương tác\n\nChọn một cặp để xem s, p, score; kéo λ, τ để thấy P/R/F1 trên test thay đổi.",
    '''import ipywidgets as widgets
docs = read_parquet(DATA_PROCESSED / "kb_docs.parquet").set_index("doc_id")
test = scored[scored.split == "test"].reset_index(drop=True)
options = [(f"{r.doc_a} ↔ {r.doc_b} ({plots.VARIANT_LABELS[r.variant_type]})", i) for i, r in test.iterrows()]
def inspect_pair(i):
    r = test.loc[i]
    verdict = "CÓ thể mâu thuẫn" if r.pred_fusion else "không vượt ngưỡng"
    print(f"s = {r.s:.3f}  p = {r.p:.3f}  score = {r.score_fusion:.3f}  τ = {r.tau_fusion:.2f}  →  {verdict}  (nhãn thật: {r.label})")
    print(docs.loc[r.doc_a, "text"]); print("-" * 60); print(docs.loc[r.doc_b, "text"])
widgets.interact(inspect_pair, i=widgets.Dropdown(options=options, description="Cặp"))''',
    '''def tune(tenant, lam, tau):
    m = plots.threshold_metrics(scored, lam, tau, tenant=None if tenant == "all" else tenant)
    print(f"P = {m['precision']:.3f}  R = {m['recall']:.3f}  F1 = {m['f1']:.3f}   TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
widgets.interact(tune, tenant=["all", "tech", "policy"],
                 lam=widgets.FloatSlider(value=config["global"]["FUSION_LAMBDA"], min=0, max=1, step=0.05, description="λ"),
                 tau=widgets.FloatSlider(value=config["global"]["CONFLICT_THRESHOLD_TAU"], min=0.3, max=0.95, step=0.01, description="τ"))''',
]


def main() -> None:
    NOTEBOOKS.mkdir(exist_ok=True)
    for name, cells in {
        "01_build_dataset": BUILD_DATASET,
        "02_calibrate_layer2": CALIBRATE,
        "03_train_layer3": TRAIN_LAYER3,
        "04_results": RESULTS,
    }.items():
        nbformat.write(notebook(cells), NOTEBOOKS / f"{name}.ipynb")


if __name__ == "__main__":
    main()
