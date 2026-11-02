---
phase: 7
title: "Notebook kết quả và biểu đồ"
status: pending
priority: P1
effort: 5h
depends_on: [5, 6]
---

# Phase 07: Notebook kết quả và biểu đồ

## Mục tiêu

Trình bày kết quả bằng một notebook vẽ biểu đồ benchmark và thông số huấn luyện, chạy được trên Colab hoặc local chỉ từ file trong `data/processed/` và `artifacts/`. Không làm web app riêng: người nghe là thầy bộ môn, cần thấy thuật toán, dataset và số liệu, không cần giao diện sản phẩm.

## File tạo mới

```text
src/plots.py                     # hàm vẽ dùng chung cho notebook và báo cáo
notebooks/04_results.ipynb       # chạy "Run all" ra toàn bộ hình và bảng
docs/figures/*.png               # hình xuất cho slide và báo cáo
tests/test_plots.py
```

## Các biểu đồ

| Nhóm | Biểu đồ | Dữ liệu |
|---|---|---|
| Dataset | Số tài liệu, số cặp, tỉ lệ nhãn theo tenant và nguồn; bảng kappa | `kb_docs`, `kb_pairs`, `kb_qa_sample.csv` |
| Động lực | Histogram cosine BGE-M3 trên ViNLI theo ba nhãn | `embeddings_vinli.npz`, `vinli.parquet` |
| Layer 2 | Heatmap F1 theo λ×τ trên dev, đánh dấu ô tốt nhất | `grid_results.csv` |
| Layer 2 | Hai heatmap `tech` và `policy` cạnh nhau, nêu τ tối ưu khác nhau | `grid_results.csv` |
| Layer 2 | Đường Precision–Recall của chỉ s, chỉ p, fusion trên test; bảng ablation | `scored_pairs`, `layer2_results.json` |
| Layer 2 | Phân phối score của cặp dương và âm, vạch τ | `scored_pairs` |
| Layer 3 | Bảng ROC-AUC, PR-AUC, Precision@k của ba model; cột thời gian suy luận | `layer3_results.json` |
| Layer 3 | Đường ROC và PR của ba model trên test | `forecast_scores` |
| Layer 3 | Learning curve XGBoost (logloss train và validation theo số cây); kết quả theo α | `xgb.joblib` (evals_result), `layer3_results.json` |
| Layer 3 | SHAP summary và waterfall cho 1–2 tài liệu tiêu biểu | `shap_values.npz` |

## Tuỳ chọn tương tác (chỉ thêm nếu còn thời gian)

Hai ô `ipywidgets` ngay trong notebook, không cần web app:

- Chọn một cặp tài liệu, in s, p, score và kết luận (ý tưởng "spam filter visualizer" trong đề tài).
- Slider λ, τ, tính lại P/R/F1 và confusion matrix trên test (ý tưởng "threshold slider").

## Các bước

1. Để Phase 06 lưu `evals_result` của XGBoost và kết quả từng α vào `layer3_results.json`, để có learning curve.
2. Viết `src/plots.py` dùng `matplotlib` (hoặc `plotly`), nhãn tiếng Việt, màu thống nhất giữa các hình.
3. `notebooks/04_results.ipynb` gọi các hàm này và lưu PNG vào `docs/figures/`.

## Kiểm tra

- `tests/test_plots.py`: mỗi hàm vẽ chạy được trên dữ liệu giả nhỏ và trả về figure không lỗi.
- "Run all" notebook 04 trên runtime mới tạo đủ hình trong `docs/figures/`.

## Tiêu chí hoàn thành

- Có đủ hình trong bảng trên, số liệu trên hình khớp với `layer2_results.json` và `layer3_results.json`.
