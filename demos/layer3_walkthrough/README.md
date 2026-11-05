# Layer 3 Demo — TabPFN teacher → XGBoost student (obsolescence forecasting)

Demo này KHÔNG viết lại logic mới. Nó gọi trực tiếp các hàm production thật
trong `src/forecast.py` và `src/forecast_features.py`, chạy từng bước và in
ra console, để dùng khi báo cáo/demo phần Layer 3 của luận văn.

Khác với `python -m src.pipeline layer3 --dataset kb`, script này **không**
ghi đè lên `artifacts/kb/layer3_results.json` (kết quả đã commit, dùng teacher
`tabpfn` thật, baseline của luận văn) — mọi output của demo nằm riêng trong
`demos/layer3_walkthrough/output/`.

## Chạy

```bash
pip install -r requirements.txt   # nếu chưa có xgboost/shap/...
python -m demos.layer3_walkthrough.run_demo
```

Mặc định dùng `--teacher gbm` (HistGradientBoostingClassifier) — chạy trong
vài giây, không cần tải gì cả. Đây là teacher dự phòng mà README chính của
repo cũng ghi rõ ("`--teacher gbm` là teacher dự phòng nếu không tải được
TabPFN").

Muốn dùng đúng teacher thật (TabPFN) như kết quả đã commit trong
`artifacts/kb/layer3_results.json`:

```bash
python -m demos.layer3_walkthrough.run_demo --teacher tabpfn
```

Lần đầu chạy với `tabpfn` sẽ tự tải checkpoint (~vài trăm MB) từ Hugging Face
về `artifacts/models/tabpfn-v2-classifier.ckpt`, và chậm hơn `gbm` khá nhiều
(dataset `kb` ở test set: `teacher_seconds` ~45s theo kết quả đã commit).

Các tuỳ chọn khác:

```bash
python -m demos.layer3_walkthrough.run_demo --rebuild-table   # tính lại bảng feature từ đầu (chậm hơn) thay vì dùng parquet đã cache
python -m demos.layer3_walkthrough.run_demo --max-context 1000 # override TABPFN_MAX_CONTEXT, chỉ có ý nghĩa với --teacher tabpfn
```

## Dataset dùng cho demo: `kb`

Repo có 3 dataset (`kb`, `vinli`, `jira`), nhưng chỉ `kb` có nhãn
`superseded_at` thật (lấy từ metadata lúc generate dữ liệu synthetic, không
phải heuristic/proxy) nên chỉ `kb` có file `layer3_results.json` sẵn. `vinli`
chỉ dùng để validate semantic ở Layer 1/2, `jira` mới chạy tới Layer 2. Vì
vậy demo này cố định dataset = `kb`.

## 6 bước script in ra

1. **Load dataset đã xử lý** — docs/edges/conflict_events/embeddings/config của
   Layer 1-2, đếm số document theo loại, loại edge, số tenant.
2. **Bảng feature point-in-time** (`build_forecast_table`) — mỗi dòng là
   (document, snapshot hàng tháng); feature chỉ được tính từ thông tin có tới
   thời điểm `t`, nhãn `label` chỉ nhìn `(t, t+90 ngày]` — không leak tương
   lai. In luôn 1 dòng feature mẫu để minh hoạ.
3. **Encode + split train/valid/test theo thời gian** (không shuffle ngẫu
   nhiên — test luôn là các snapshot gần nhất).
4. **Teacher train Out-of-Fold** (`out_of_fold_teacher`) — 5-fold
   StratifiedKFold, mỗi fold teacher chỉ thấy 4 fold kia rồi predict lên fold
   còn lại, nên `p_teacher_oof` không leak nhãn của chính dòng đó.
5. **Grid search alpha cho student** — `target = alpha*y + (1-alpha)*p_teacher_oof`,
   train XGBoost cho từng alpha trong `settings.FORECAST_ALPHAS`, chọn alpha
   tốt nhất trên **valid** theo `pr_auc` rồi `roc_auc` (chỉ trong các alpha <
   1.0, vì alpha=1.0 là baseline "không distill").
6. **Đánh giá trên test** (chạm 1 lần duy nhất) — so sánh 3 model:
   `xgb_hard_labels` (alpha=1.0), `xgb_distilled` (best alpha), và teacher
   đơn lẻ. Kèm SHAP feature importance của student và 1 chart PNG tổng hợp
   3 panel (alpha selection, so sánh test, SHAP).

## Điểm cần nói thẳng khi báo cáo (không chỉ show số đẹp)

Kết quả đã commit (`artifacts/kb/layer3_results.json`, teacher=`tabpfn`) cho
thấy trên test set, `xgb_hard_labels` (không distill) **thắng**
`xgb_distilled` (alpha=0.7) về ROC AUC (0.7835 vs 0.7779), nhưng
`xgb_distilled` thắng về PR AUC (0.2663 vs 0.2472) và precision@10/20 (0.8/0.5
vs 0.7/0.4). Chạy demo này với teacher `gbm` cũng cho một bức tranh tương tự
— không model nào thắng tuyệt đối trên cả 2 metric.

Đây đúng là điều literature review đã cảnh báo (distillation không phải lúc
nào cũng có lợi trên dữ liệu nhỏ/ít chiều — xem phần related work về TabPFN
teacher/XGBoost student trong báo cáo nghiên cứu Layer 3), và vì base rate
(~5-6% positive) rất thấp, **PR AUC + precision@k mới là metric quyết định
trong thực tế**, không phải ROC AUC. Khi demo/báo cáo nên trình bày cả hai
mặt, không chỉ chọn metric nào đẹp hơn.

## File liên quan trong repo chính (không sửa gì ở đây)

- `src/forecast.py` — teacher (`TabPFNTeacher`/`GradientBoostingTeacher`),
  OOF, train student, `run_forecast()`.
- `src/forecast_features.py` — point-in-time feature table, label, split.
- `src/settings.py` — toàn bộ hằng số (`FORECAST_ALPHAS`,
  `FORECAST_HORIZON_DAYS`, `TABPFN_MAX_CONTEXT`, ...).
- `artifacts/kb/layer3_results.json` — kết quả Layer 3 thật đã commit, dùng
  làm baseline để đối chiếu khi demo.

## Ghi chú kỹ thuật còn mở (chưa sửa trong nhánh này)

`src/forecast.py` hiện lưu model student bằng `joblib.dump(...)`
(`xgb.joblib`). Tài liệu chính thức của XGBoost khuyến nghị dùng
`model.save_model("xgb.json")` (hoặc `.ubj`) cho việc lưu lâu dài/tương thích
giữa phiên bản, thay vì pickle/joblib. Demo này không thay đổi điều đó — chỉ
ghi chú lại để cân nhắc khi làm tiếp, không nằm trong phạm vi yêu cầu lần này
(chỉ làm demo).
