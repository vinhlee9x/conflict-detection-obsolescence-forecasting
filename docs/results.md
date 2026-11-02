# Kết quả thí nghiệm

File này được sinh tự động bởi `python -m src.report_tables` từ các file trong `artifacts/`. Hình nằm trong `docs/figures/` (notebook 04).

## Dataset (KB tổng hợp)

972 tài liệu, 1144 cặp có nhãn sau khi chia tập (286 cặp mâu thuẫn).

Tài liệu theo loại và tenant:

| loại | KB chính sách | KB kỹ thuật |
|---|---|---|
| Bản sửa có mâu thuẫn | 81 | 78 |
| Tài liệu mâu thuẫn ẩn | 61 | 66 |
| Tài liệu trùng từ vựng | 57 | 64 |
| Bản sửa diễn đạt lại | 83 | 82 |
| Tài liệu gốc | 200 | 200 |

Cặp theo loại và tập (dev dùng chọn λ, τ; test chỉ dùng để báo cáo):

| loại cặp | nhãn | dev | test |
|---|---|---|---|
| Mâu thuẫn (có cạnh) | 1 | 129 | 30 |
| Mâu thuẫn ẩn | 1 | 100 | 27 |
| Trùng từ vựng | 0 | 95 | 26 |
| Diễn đạt lại | 0 | 130 | 35 |
| Ngẫu nhiên | 0 | 326 | 85 |
| Cùng mẫu, khác đối tượng | 0 | 136 | 25 |

## ViNLI: cosine BGE-M3 có nhận ra mâu thuẫn không?

2264 cặp câu tập test. Cosine trung bình: contradiction 0.686, entailment 0.774, neutral 0.689.

AUC kéo theo vs mâu thuẫn = 0.723; AUC mâu thuẫn vs trung lập = 0.496.

## Layer 2: phát hiện mâu thuẫn trên tập test

Tham số chọn trên dev: KB chính sách: λ = 0.95, τ = 0.72 (F1 dev 0.712); KB kỹ thuật: λ = 1.00, τ = 0.75 (F1 dev 0.669). K = 3, γ = 0.5.

| phạm vi | phương pháp | λ | τ | P | R | F1 | F1 95% CI | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| Gộp hai tenant | Chỉ s (λ = 1) |  |  | 0.543 | 0.772 | 0.638 | [0.55, 0.73] | 0.638 |
| Gộp hai tenant | Chỉ p (λ = 0) |  |  | 0.480 | 0.421 | 0.449 | [0.33, 0.57] | 0.417 |
| Gộp hai tenant | Fusion, λ và τ theo tenant |  |  | 0.530 | 0.772 | 0.629 | [0.53, 0.72] | 0.620 |
| Gộp hai tenant | Fusion, λ và τ chung |  |  | 0.524 | 0.772 | 0.624 | [0.52, 0.72] | 0.631 |
| KB chính sách | Chỉ s (λ = 1) | 1.000 | 0.770 | 0.535 | 0.742 | 0.622 | [0.47, 0.75] | 0.652 |
| KB chính sách | Chỉ p (λ = 0) | 0.000 | 0.300 | 0.462 | 0.387 | 0.421 | [0.26, 0.57] | 0.427 |
| KB chính sách | Fusion, λ và τ theo tenant | 0.950 | 0.720 | 0.511 | 0.742 | 0.605 | [0.47, 0.72] | 0.635 |
| KB chính sách | Fusion, λ và τ chung | 0.950 | 0.720 | 0.511 | 0.742 | 0.605 | [0.46, 0.72] | 0.635 |
| KB kỹ thuật | Chỉ s (λ = 1) | 1.000 | 0.750 | 0.553 | 0.808 | 0.656 | [0.52, 0.78] | 0.631 |
| KB kỹ thuật | Chỉ p (λ = 0) | 0.000 | 0.300 | 0.500 | 0.462 | 0.480 | [0.29, 0.65] | 0.425 |
| KB kỹ thuật | Fusion, λ và τ theo tenant | 1.000 | 0.750 | 0.553 | 0.808 | 0.656 | [0.51, 0.78] | 0.631 |
| KB kỹ thuật | Fusion, λ và τ chung | 0.950 | 0.720 | 0.538 | 0.808 | 0.646 | [0.49, 0.77] | 0.630 |

Tỉ lệ cặp bị gắn cờ theo từng loại (với cặp nhãn 1 đây là recall, với nhãn 0 là tỉ lệ báo nhầm):

| loại cặp | nhãn | n | s TB | p TB | bị gắn cờ (chỉ s) | bị gắn cờ (chỉ p) | bị gắn cờ (fusion) |
|---|---|---|---|---|---|---|---|
| Mâu thuẫn (có cạnh) | 1 | 30 | 0.968 | 0.347 | 1.000 | 0.800 | 1.000 |
| Mâu thuẫn ẩn | 1 | 27 | 0.884 | 0.000 | 0.519 | 0.000 | 0.519 |
| Trùng từ vựng | 0 | 26 | 0.784 | 0.000 | 0.000 | 0.000 | 0.077 |
| Diễn đạt lại | 0 | 35 | 0.955 | 0.332 | 1.000 | 0.743 | 1.000 |
| Ngẫu nhiên | 0 | 85 | 0.665 | 0.001 | 0.000 | 0.000 | 0.000 |
| Cùng mẫu, khác đối tượng | 0 | 25 | 0.837 | 0.023 | 0.080 | 0.000 | 0.080 |

## Layer 3: dự báo lỗi thời trên các mốc test

Teacher: tabpfn (context tối đa 3000 dòng, 368 giây). Horizon 90 ngày. Số dòng train/valid/test: 5623/838/1046, số dương: 357/45/52. Mốc test: 2024-06-01 đến 2024-09-01. α chọn trên validation: 0.7.

| mô hình | ROC-AUC | PR-AUC | P@10 | P@20 | ms / tài liệu |
|---|---|---|---|---|---|
| XGBoost nhãn cứng | 0.793 | 0.291 | 0.900 | 0.500 | 0.001 |
| XGBoost + soft-label | 0.803 | 0.312 | 0.900 | 0.600 | 0.001 |
| TabPFN (teacher) | 0.764 | 0.262 | 0.700 | 0.500 | 57.807 |
