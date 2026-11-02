---
phase: 5
title: "Layer 2: lan truyền đồ thị, fusion, calibration"
status: completed
priority: P1
effort: 12h
depends_on: [4]
---

# Phase 05: Layer 2, lan truyền đồ thị, fusion và calibration

## Mục tiêu

Cài đặt hàm điểm không cần train `score = λ·s + (1−λ)·p`, chọn λ và τ theo từng tenant trên dev, và ra bảng số liệu trên test. Đây là phần thầy sẽ hỏi kỹ nhất.

## File tạo mới

```text
src/splits.py                    # chia dev/test theo nhóm root_id
src/graph.py                     # ma trận kề thưa, p(A, B) đa bước
src/conflict.py                  # score, so ngưỡng τ, sinh candidate
src/calibrate.py                 # grid search λ×τ, chọn theo F1 trên dev
src/evaluate.py                  # P, R, F1, PR-AUC, confusion matrix
notebooks/02_calibrate_layer2.ipynb
artifacts/config.json            # FUSION_LAMBDA, CONFLICT_THRESHOLD_TAU theo tenant
artifacts/grid_results.csv       # tenant, λ, τ, precision, recall, f1 trên dev
artifacts/scored_pairs.parquet   # s, p, score, label, split cho từng cặp
artifacts/layer2_results.json    # bảng ablation trên test
tests/test_splits.py
tests/test_graph.py
tests/test_conflict.py
```

## Thuật toán

- Đồ thị vô hướng từ `kb_edges` (và `jira_edges`), trọng số theo loại cạnh (cấu hình trong `settings.py`).
- **Bỏ cạnh nhãn**: với Jira, loại mọi cạnh `is_label_edge`. Với KB, không có cạnh nào mã hoá nhãn trực tiếp (phase 02 đã thiết kế như vậy), nhưng vẫn loại cạnh CONFLICTS_WITH nếu có.
- Ma trận chuyển `M = D⁻¹A` (chuẩn hoá theo hàng) để giá trị mỗi bước nằm trong [0, 1].
- `p(A, B) = Σ_{k=1..K} γ^k · M^k[A, B] / Σ_{k=1..K} γ^k`, với K = 3, γ = 0,5. Chia cho tổng γ^k để p nằm trong [0, 1], cùng thang với s. Lấy đối xứng `max(p(A,B), p(B,A))`.
- Tính bằng phép nhân ma trận thưa của `scipy.sparse`, chỉ cho các cặp trong tập candidate (top-k láng giềng cộng các đỉnh trong bán kính K bước), không tính O(n²).
- s cần đưa về [0, 1] trước khi fusion: dùng min-max ước lượng trên dev rồi áp nguyên cho test.

## Các bước

1. `splits.py`: chia 60/20/20 theo `root_id` bằng `GroupShuffleSplit` với seed cố định. Layer 2 không train nên gộp phần train vào dev (dev 80%, test 20%). Ghi `split` vào `scored_pairs`.
2. Tính s, p, score cho mọi cặp có nhãn.
3. `calibrate.py`: grid λ (0..1, bước 0,05) × τ (0,30..0,95, bước 0,01) trên dev, riêng từng tenant và một lần cho cả hai tenant gộp. Chọn theo F1; hoà thì ưu tiên precision cao hơn (ít báo nhầm hơn cho người xác nhận).
4. Ghi `config.json` dạng `{"tech": {"FUSION_LAMBDA": ..., "CONFLICT_THRESHOLD_TAU": ...}, "policy": {...}, "global": {...}}`.
5. Đánh giá trên test, xuất `layer2_results.json` với các dòng:
   - Chỉ s (λ = 1, τ chọn lại trên dev), chỉ p (λ = 0), fusion.
   - τ chung (global) so với τ theo tenant.
   - Kết quả riêng trên nhóm "mâu thuẫn ẩn" để nói thẳng điểm yếu của p.
   - Cùng bảng cho Jira nếu có.
6. Phân tích ViNLI: histogram cosine theo ba nhãn, kèm AUC của s khi phân biệt contradiction với entailment. Đưa hình vào báo cáo làm động lực cho p.
7. Hàm `detect_candidates(new_doc_ids, tenant)` trong `conflict.py`: trả danh sách cặp có score > τ để minh hoạ luồng end-to-end trong notebook; khi xác nhận thì thêm cạnh CONFLICTS_WITH vào đồ thị trong bộ nhớ.

## Kiểm tra

- `tests/test_splits.py`: không có `root_id` nào xuất hiện ở cả dev và test.
- `tests/test_graph.py`: đồ thị đường A–B–C, p(A, C) > 0 chỉ khi K ≥ 2; p nằm trong [0, 1]; cạnh nhãn bị loại không ảnh hưởng tới p.
- `tests/test_conflict.py`: λ = 1 thì score bằng s đã chuẩn hoá; λ = 0 thì score bằng p.
- Kiểm tra thủ công: λ, τ in trong báo cáo khớp với `config.json`; số liệu test không được dùng ở bất kỳ bước chọn nào.

## Tiêu chí hoàn thành

- Có `config.json`, `grid_results.csv`, `layer2_results.json`.
- Fusion có F1 test không thấp hơn dòng tốt nhất trong hai dòng chỉ s và chỉ p. Nếu thấp hơn thì báo cáo trung thực và phân tích nguyên nhân, không chỉnh dữ liệu cho đẹp.
- τ tối ưu của hai tenant khác nhau (minh hoạ ý tưởng per-tenant calibration). Nếu bằng nhau, ghi nhận và giải thích.
