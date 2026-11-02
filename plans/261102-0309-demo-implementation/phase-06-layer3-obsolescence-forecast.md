---
phase: 6
title: "Layer 3: dự báo lỗi thời"
status: pending
priority: P1
effort: 12h
depends_on: [5]
---

# Phase 06: Layer 3, dự báo lỗi thời

## Mục tiêu

Dự báo xác suất một tài liệu trở nên lỗi thời trong H tháng tới. TabPFN làm teacher offline, XGBoost làm student chạy nhanh lúc demo. Chứng minh bằng số rằng distill có ích (độ chính xác gần teacher, nhanh hơn nhiều).

## File tạo mới

```text
src/forecast_features.py        # bảng (doc, mốc t) với đặc trưng chỉ dùng thông tin trước t
src/forecast.py                 # teacher, student, đánh giá, SHAP
notebooks/03_train_layer3.ipynb
data/processed/forecast_table.parquet
artifacts/xgb.joblib
artifacts/forecast_scores.parquet   # xác suất lỗi thời của từng tài liệu ở mốc mới nhất
artifacts/shap_values.npz
artifacts/layer3_results.json
tests/test_forecast_features.py
```

## Thiết kế bài toán

- Mốc snapshot t theo tháng trên trục thời gian của KB.
- Một dòng = (tài liệu, t), chỉ cho tài liệu đã ban hành và chưa lỗi thời tại t.
- Nhãn: `1` nếu `superseded_at` rơi vào (t, t + H], với H = 3 tháng (thử thêm 6 tháng nếu dương quá ít).
- Đặc trưng tại t: số ngày từ lần cập nhật cuối, tuổi tài liệu, số cạnh CONFLICTS_WITH đã xác nhận trước t, số candidate Layer 2 chưa xác nhận (dùng `config.json` từ phase 05), bậc vào/ra, số tài liệu mới hơn cùng series, số phiên bản, phòng ban, tenant.
- Chia theo thời gian: train trên các mốc cũ, validation trên mốc kế tiếp, test trên các mốc mới nhất. Không xáo trộn ngẫu nhiên.

## Các bước

1. Đầu phase, trên Colab: cài `tabpfn`, chạy thử một ví dụ nhỏ để chắc tải được weight (có thể phải chấp nhận license hoặc đặt token Hugging Face). Ghi lại phiên bản chạy được vào `requirements-colab.txt`. Kiểm tra giới hạn số dòng và số đặc trưng của phiên bản đó.
2. `forecast_features.py` dựng `forecast_table`. Mọi đặc trưng tính bằng hàm nhận `as_of = t` và chỉ lọc dữ liệu có thời điểm nhỏ hơn hoặc bằng t.
3. Teacher: TabPFN trên tập train. Sinh soft-label bằng cross-fitting 5 fold trên train (mỗi dòng nhận xác suất từ model không thấy nó), và dự đoán thêm cho các dòng chưa có nhãn người xác nhận nếu có.
4. Student: XGBoost `binary:logistic` với target là `α·y + (1−α)·p_teacher`, α chọn trên validation (thử 0; 0,3; 0,5; 0,7; 1). Tuning nhẹ `max_depth`, `n_estimators` trên validation.
5. Baseline: XGBoost chỉ học nhãn cứng (α = 1).
6. Đánh giá trên test, xuất `layer3_results.json`: ROC-AUC, PR-AUC, Precision@k (k = 10, 20) cho ba model (XGBoost nhãn cứng, XGBoost distill, TabPFN), và thời gian suy luận trung bình mỗi tài liệu trên CPU.
7. SHAP: `TreeExplainer` cho XGBoost, lưu giá trị cho các tài liệu ở mốc mới nhất để notebook kết quả vẽ waterfall.
8. Lưu `xgb.joblib` và `forecast_scores.parquet`.

## Kiểm tra

- `tests/test_forecast_features.py`: dựng KB nhỏ có sự kiện sau t, kiểm tra đặc trưng tại t không đổi khi xoá các sự kiện đó (chống rò rỉ thời gian); nhãn đúng với khoảng (t, t + H].
- Mốc nhỏ nhất của test lớn hơn mốc lớn nhất của train.

## Tiêu chí hoàn thành

- Có đủ ba dòng trong bảng kết quả và số liệu thời gian suy luận.
- XGBoost load lại từ `xgb.joblib` trên local cho cùng xác suất với trên Colab (sai khác dưới 1e-6).

## Rủi ro

- Ít dòng dương: tăng H, hoặc tăng tỉ lệ tài liệu bị thay thế ở phase 02.
- TabPFN không tải được: dùng một mô hình mạnh hơn làm teacher thay thế (ví dụ ensemble gradient boosting lớn), ghi rõ trong báo cáo; khung teacher–student giữ nguyên.
