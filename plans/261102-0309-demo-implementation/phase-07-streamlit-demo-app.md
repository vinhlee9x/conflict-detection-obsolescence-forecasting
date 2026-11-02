---
phase: 7
title: "App Streamlit"
status: pending
priority: P1
effort: 10h
depends_on: [5, 6]
---

# Phase 07: App Streamlit

## Mục tiêu

App chạy local, chỉ đọc file trong `data/processed/` và `artifacts/`, mở trong dưới 10 giây khi không có mạng. Mỗi tab ứng với một ý tưởng demo trong đề tài.

## File tạo mới

```text
app/streamlit_app.py           # điều hướng tab, load dữ liệu có cache
app/data_loader.py             # st.cache_data cho parquet, npz, json, joblib
app/tabs/pair_inspector.py     # tab 1
app/tabs/grid_heatmap.py       # tab 2
app/tabs/threshold_slider.py   # tab 3
app/tabs/tenant_compare.py     # tab 4
app/tabs/end_to_end_graph.py   # tab 5
app/tabs/obsolescence.py       # tab 6
tests/test_app_smoke.py
```

## Các tab theo thứ tự ưu tiên

| Ưu tiên | Tab | Nội dung | Dữ liệu |
|---|---|---|---|
| 1 | Cặp tài liệu | Chọn 2 tài liệu (có sẵn cặp mẫu SOP-047 và SOP-047-v2), hiện s, p, score, τ của tenant và kết luận; tô sáng cặp chunk có cosine cao nhất | `scored_pairs`, `embeddings_kb.npz`, `config.json` |
| 1 | Grid search | Heatmap F1 theo λ×τ trên dev, đánh dấu ô tốt nhất, đường Precision–Recall, bảng ablation trên test | `grid_results.csv`, `layer2_results.json` |
| 2 | Thanh trượt ngưỡng | Slider λ, τ, tính lại P/R/F1 và confusion matrix trên test ngay lập tức | `scored_pairs` |
| 2 | Dự báo lỗi thời | Bảng xếp hạng tài liệu theo xác suất, chọn một tài liệu xem SHAP waterfall, bảng so sánh ba model | `forecast_scores`, `shap_values.npz`, `layer3_results.json` |
| 3 | Theo tenant | Hai heatmap cạnh nhau cho `tech` và `policy`, nêu τ tối ưu khác nhau | `grid_results.csv` |
| 3 | Đồ thị end-to-end | Chọn 3–4 tài liệu "mới" có sẵn, hiện candidate có score > τ, nút "Xác nhận" thêm cạnh CONFLICTS_WITH và vẽ lại đồ thị con bằng pyvis | `kb_edges`, `conflict.detect_candidates` |

## Các bước

1. `data_loader.py` load một lần với `st.cache_data` hoặc `st.cache_resource`. Báo lỗi rõ ràng nếu thiếu file artifact, kèm tên notebook cần chạy.
2. Làm tab ưu tiên 1 trước, sau đó ưu tiên 2, rồi 3. Tab 3 (slider) tính lại metric bằng numpy trên `scored_pairs`, không gọi lại embedding.
3. Tab 5 dùng các tài liệu đã embed sẵn để không cần BGE-M3 trên local. Chỉ khi nhóm muốn dán văn bản mới tuỳ ý mới cần cài BGE-M3 trên local (chạy CPU được với vài tài liệu); để đây là tuỳ chọn.
4. Đồng nhất màu và nhãn tiếng Việt trên mọi biểu đồ.

## Kiểm tra

- `tests/test_app_smoke.py` dùng `streamlit.testing.v1.AppTest` mở app, chuyển qua từng tab, không có exception.
- Tắt mạng, chạy `streamlit run app/streamlit_app.py`, đo thời gian mở.

## Tiêu chí hoàn thành

- Sáu tab chạy được, hoặc tối thiểu bốn tab ưu tiên 1 và 2 nếu thiếu thời gian.
- Số liệu hiển thị trong app khớp với `layer2_results.json` và `layer3_results.json`.
