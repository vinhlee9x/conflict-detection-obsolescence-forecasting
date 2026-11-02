---
phase: 1
title: "Khởi tạo repo và môi trường"
status: pending
priority: P1
effort: 4h
depends_on: []
---

# Phase 01: Khởi tạo repo và môi trường

## Mục tiêu

Có khung repo chạy được trên cả Colab và local, với cấu hình và seed cố định để mọi phase sau tái tạo được kết quả.

## File tạo mới

```text
requirements.txt
requirements-colab.txt        # thêm FlagEmbedding, tabpfn, torch (Colab có sẵn torch)
pyproject.toml                # cấu hình pytest, ruff
.gitignore                    # data/raw/jira/, *.bson, .env, __pycache__
README.md                     # cách chạy Colab và local
src/__init__.py
src/paths.py                  # đường dẫn data/, artifacts/ dùng chung
src/settings.py               # SEED, K_HOPS, GAMMA, TOP_K, lưới λ và τ
src/io_utils.py               # đọc/ghi parquet, npz, json
notebooks/00_colab_setup.ipynb
tests/test_smoke.py
data/raw/.gitkeep
data/processed/.gitkeep
artifacts/.gitkeep
```

## Các bước

1. Tạo cấu trúc thư mục như sơ đồ trong tài liệu kế hoạch.
2. Ghim phiên bản trong `requirements.txt`: `numpy`, `pandas`, `pyarrow`, `scipy`, `networkx`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `plotly`, `ipywidgets`, `joblib`, `pytest`. Python 3.11.
3. `requirements-colab.txt` thêm `FlagEmbedding` (hoặc `sentence-transformers`) và `tabpfn`. Không cài hai gói này trên local trừ khi demo end-to-end với tài liệu mới.
4. `src/settings.py` gom mọi hằng số: `SEED = 42`, `K_HOPS = 3`, `GAMMA = 0.5`, `TOP_K_NEIGHBORS = 15`, `LAMBDA_GRID = 0.0..1.0 bước 0.05`, `TAU_GRID = 0.30..0.95 bước 0.01`.
5. `notebooks/00_colab_setup.ipynb`: mount Google Drive, `git clone`, `pip install -r requirements-colab.txt`, symlink `data/` và `artifacts/` sang Drive để giữ kết quả giữa các session.
6. Quy ước lưu trữ: KB tổng hợp, `config.json`, `grid_results.csv`, `xgb.joblib` đủ nhỏ để commit. Embedding lưu `float16` trong `.npz`; nếu vượt 50 MB thì để trên Drive và README ghi link tải. Dump Jira không bao giờ commit.

## Kiểm tra

- `pip install -r requirements.txt && pytest` chạy xanh trên local.
- Notebook 00 chạy hết trên Colab runtime mới, in ra phiên bản các thư viện.

## Tiêu chí hoàn thành

- Một thành viên khác clone repo về và chạy `pytest` thành công chỉ bằng hướng dẫn trong README.
