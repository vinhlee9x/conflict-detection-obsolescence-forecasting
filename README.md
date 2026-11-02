# Phát hiện mâu thuẫn và dự báo lỗi thời trong kho tri thức doanh nghiệp

Project cuối kỳ môn Học máy tiên tiến và ứng dụng. Ba tầng:

1. **Layer 1**: chuẩn hoá văn bản tiếng Việt (NFC, ngày, số), chia theo điều khoản, embedding BGE-M3, đồ thị quan hệ (REFERENCES, SAME_SERIES, SAME_DEPARTMENT).
2. **Layer 2**: điểm không cần huấn luyện `score = λ·s + (1−λ)·p`, với `s` là cosine lớn nhất giữa các chunk và `p = Σ γ^k M^k[A, B] / Σ γ^k` là lan truyền đa bước trên đồ thị. λ, τ chọn bằng grid search trên dev cho từng tenant. Cặp có `score > τ` là candidate; người xác nhận thì thành cạnh `CONFLICTS_WITH`.
3. **Layer 3**: dự báo tài liệu lỗi thời trong 90 ngày tới. TabPFN v2 làm teacher (offline, soft-label out-of-fold), XGBoost làm student.

Demo chỉ dùng numpy (`.npz`) và NetworkX/`scipy.sparse`; PostgreSQL + pgvector là hướng triển khai thực tế, không nằm trong demo.

## Chạy trên Colab

Mở các notebook trong `notebooks/` bằng Colab (Runtime → GPU), chạy theo thứ tự:

| Notebook | Việc | Thời gian ước tính |
|---|---|---|
| `01_build_dataset.ipynb` | Sinh KB tổng hợp, xuất mẫu 100 cặp để nhóm gán nhãn, tải ViNLI, (tuỳ chọn) Jira | 1 phút |
| `02_calibrate_layer2.ipynb` | Embed BGE-M3, phân tích ViNLI, grid search λ×τ, bảng ablation | 3–5 phút trên GPU |
| `03_train_layer3.ipynb` | Bảng đặc trưng theo thời gian, TabPFN → XGBoost, SHAP | vài phút trên GPU |
| `04_results.ipynb` | Toàn bộ biểu đồ và bảng, lưu vào `docs/figures/` | vài giây, CPU |

Mỗi notebook tự `git clone` repo và cài `requirements-colab.txt`. Đặt `USE_DRIVE = True` để giữ `data/processed/` và `artifacts/` trên Google Drive giữa các session.

Lúc báo cáo chỉ cần mở `04_results.ipynb`: các file kết quả đã có sẵn trong repo, nên không phải chạy lại embedding hay TabPFN.

## Chạy local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-colab.txt        # chỉ cần requirements.txt nếu chỉ xem kết quả
pytest

python -m src.pipeline kb-generate
python -m src.pipeline embed --dataset kb --backend bge-m3   # khoảng 6 phút trên CPU 4 nhân
python -m src.pipeline vinli --backend bge-m3
python -m src.pipeline layer2 --dataset kb
python -m src.pipeline layer3 --dataset kb --teacher tabpfn  # trên CPU cần TABPFN_ALLOW_CPU_LARGE_DATASET=1 và khá chậm
python -m src.report_tables                                  # sinh docs/results.md
```

`--backend tfidf` là baseline từ vựng chạy không cần tải model. `--teacher gbm` là teacher dự phòng nếu không tải được TabPFN.

## Dữ liệu

- **KB tổng hợp tiếng Việt** (`src/synth/`): 2 tenant (KB kỹ thuật, KB chính sách), 400 tài liệu gốc sinh từ template điều khoản có số liệu. Biến thể có nhãn: bản sửa mâu thuẫn, bản diễn đạt lại, mâu thuẫn ẩn (không có cạnh), tài liệu trùng từ vựng khác chủ đề. Bản mâu thuẫn và bản diễn đạt lại đều nối về bản gốc, nên cạnh đồ thị không làm lộ nhãn. Nhãn đúng theo cách sinh, kiểm tra lại bằng 100 cặp gán tay (Cohen's kappa).
- **ViNLI**: dùng bản mirror công khai trên Hugging Face (`presencesw/vinli_4_label`) để kiểm tra cosine BGE-M3 có phân biệt mâu thuẫn hay không.
- **Public Jira Dataset** (tuỳ chọn): đặt `RUN_JIRA = True` trong notebook 01. Notebook tự cài MongoDB, tải dump từ Zenodo (5,8 GB, có kiểm tra MD5), chỉ restore collection đã chọn (mặc định `Spring`), chọn 3 project có nhiều link Duplicate/Supersede nhất (ROO, SEC, DATAMONGO, khoảng 9.700 issue và 513 cặp dương), mỗi project là một tenant, rồi xuất `data/processed/jira_*.parquet`. Lần đầu mất khoảng 30–40 phút. Code nằm ở `src/datasets/jira_setup.py` và `jira_export.py`.

Kết quả mới nhất: [docs/results.md](docs/results.md).

## Cấu trúc

```text
src/synth/          sinh KB tổng hợp, mẫu kiểm tra kappa
src/datasets/       Jira, ViNLI
src/ingest.py       chuẩn hoá, chia chunk, embedding
src/similarity.py   s(A, B), láng giềng gần nhất
src/graph.py        đồ thị và p(A, B)
src/conflict.py     fusion, ngưỡng, sinh candidate
src/calibrate.py    grid search λ×τ
src/splits.py       chia dev/test theo nhóm tài liệu gốc
src/forecast*.py    Layer 3
src/plots.py        biểu đồ cho notebook 04
src/pipeline.py     lệnh chạy từng bước
scripts/build_notebooks.py  sinh lại các notebook
```
