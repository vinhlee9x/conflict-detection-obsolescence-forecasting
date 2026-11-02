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

Tham số chọn trên dev: KB chính sách: λ = 1.00, τ = 0.77 (F1 dev 0.708); KB kỹ thuật: λ = 1.00, τ = 0.68 (F1 dev 0.665). K = 3, γ = 0.5.

| phạm vi | phương pháp | λ | τ | P | R | F1 | F1 95% CI | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| Gộp hai tenant | Chỉ s (λ = 1) |  |  | 0.547 | 0.825 | 0.657 | [0.57, 0.74] | 0.636 |
| Gộp hai tenant | Chỉ p (λ = 0) |  |  | 0.469 | 0.526 | 0.496 | [0.38, 0.60] | 0.417 |
| Gộp hai tenant | Fusion, λ và τ theo tenant |  |  | 0.547 | 0.825 | 0.657 | [0.56, 0.74] | 0.636 |
| Gộp hai tenant | Fusion, λ và τ chung |  |  | 0.529 | 0.789 | 0.634 | [0.53, 0.73] | 0.634 |
| Gộp hai tenant | Mặc định chưa tinh chỉnh (λ = 0.5, τ = 0.6) |  |  | 0.518 | 0.509 | 0.513 | [0.39, 0.62] | 0.576 |
| KB chính sách | Chỉ s (λ = 1) | 1.000 | 0.770 | 0.535 | 0.742 | 0.622 | [0.49, 0.74] | 0.650 |
| KB chính sách | Chỉ p (λ = 0) | 0.000 | 0.230 | 0.471 | 0.516 | 0.492 | [0.33, 0.64] | 0.427 |
| KB chính sách | Fusion, λ và τ theo tenant | 1.000 | 0.770 | 0.535 | 0.742 | 0.622 | [0.48, 0.74] | 0.650 |
| KB chính sách | Fusion, λ và τ chung | 1.000 | 0.750 | 0.511 | 0.742 | 0.605 | [0.47, 0.73] | 0.650 |
| KB chính sách | Mặc định chưa tinh chỉnh (λ = 0.5, τ = 0.6) | 0.500 | 0.600 | 0.516 | 0.516 | 0.516 | [0.35, 0.66] | 0.579 |
| KB kỹ thuật | Chỉ s (λ = 1) | 1.000 | 0.680 | 0.558 | 0.923 | 0.696 | [0.55, 0.81] | 0.630 |
| KB kỹ thuật | Chỉ p (λ = 0) | 0.000 | 0.200 | 0.467 | 0.538 | 0.500 | [0.34, 0.65] | 0.425 |
| KB kỹ thuật | Fusion, λ và τ theo tenant | 1.000 | 0.680 | 0.558 | 0.923 | 0.696 | [0.56, 0.81] | 0.630 |
| KB kỹ thuật | Fusion, λ và τ chung | 1.000 | 0.750 | 0.550 | 0.846 | 0.667 | [0.52, 0.78] | 0.630 |
| KB kỹ thuật | Mặc định chưa tinh chỉnh (λ = 0.5, τ = 0.6) | 0.500 | 0.600 | 0.520 | 0.500 | 0.510 | [0.32, 0.67] | 0.616 |

Tỉ lệ cặp bị gắn cờ theo từng loại (với cặp nhãn 1 đây là recall, với nhãn 0 là tỉ lệ báo nhầm):

| loại cặp | nhãn | n | s TB | p TB | bị gắn cờ (chỉ s) | bị gắn cờ (chỉ p) | bị gắn cờ (fusion) |
|---|---|---|---|---|---|---|---|
| Mâu thuẫn (có cạnh) | 1 | 30 | 0.968 | 0.347 | 1.000 | 1.000 | 1.000 |
| Mâu thuẫn ẩn | 1 | 27 | 0.883 | 0.000 | 0.630 | 0.000 | 0.630 |
| Trùng từ vựng | 0 | 26 | 0.784 | 0.000 | 0.000 | 0.000 | 0.000 |
| Diễn đạt lại | 0 | 35 | 0.955 | 0.332 | 1.000 | 0.971 | 1.000 |
| Ngẫu nhiên | 0 | 85 | 0.666 | 0.001 | 0.000 | 0.000 | 0.000 |
| Cùng mẫu, khác đối tượng | 0 | 25 | 0.837 | 0.023 | 0.160 | 0.000 | 0.160 |

## Layer 2 trên Jira (dữ liệu thật, nhãn proxy Duplicate/Supersede), tập test

Tham số chọn trên dev: DATAMONGO: λ = 0.95, τ = 0.50 (F1 dev 0.579); ROO: λ = 0.45, τ = 0.22 (F1 dev 0.555); SEC: λ = 0.90, τ = 0.44 (F1 dev 0.585). K = 3, γ = 0.5.

| phạm vi | phương pháp | λ | τ | P | R | F1 | F1 95% CI | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| Gộp hai tenant | Chỉ s (λ = 1) |  |  | 0.634 | 0.549 | 0.588 | [0.49, 0.67] | 0.702 |
| Gộp hai tenant | Chỉ p (λ = 0) |  |  | 0.000 | 0.000 | 0.000 | [0.00, 0.00] | 0.259 |
| Gộp hai tenant | Fusion, λ và τ theo tenant |  |  | 0.644 | 0.573 | 0.606 | [0.51, 0.70] | 0.649 |
| Gộp hai tenant | Fusion, λ và τ chung |  |  | 0.662 | 0.524 | 0.585 | [0.48, 0.68] | 0.701 |
| Gộp hai tenant | Mặc định chưa tinh chỉnh (λ = 0.5, τ = 0.6) |  |  | 0.000 | 0.000 | 0.000 | [0.00, 0.00] | 0.704 |
| DATAMONGO | Chỉ s (λ = 1) | 1.000 | 0.530 | 0.667 | 0.300 | 0.414 | [0.17, 0.61] | 0.693 |
| DATAMONGO | Chỉ p (λ = 0) | 0.000 | 0.370 | 0.000 | 0.000 | 0.000 | [0.00, 0.00] | 0.250 |
| DATAMONGO | Fusion, λ và τ theo tenant | 0.950 | 0.500 | 0.700 | 0.350 | 0.467 | [0.22, 0.69] | 0.693 |
| DATAMONGO | Fusion, λ và τ chung | 0.550 | 0.290 | 0.700 | 0.350 | 0.467 | [0.20, 0.65] | 0.693 |
| DATAMONGO | Mặc định chưa tinh chỉnh (λ = 0.5, τ = 0.6) | 0.500 | 0.600 | 0.000 | 0.000 | 0.000 | [0.00, 0.00] | 0.693 |
| ROO | Chỉ s (λ = 1) | 1.000 | 0.490 | 0.531 | 0.548 | 0.540 | [0.38, 0.68] | 0.578 |
| ROO | Chỉ p (λ = 0) | 0.000 | 0.140 | 0.000 | 0.000 | 0.000 | [0.00, 0.00] | 0.274 |
| ROO | Fusion, λ và τ theo tenant | 0.450 | 0.220 | 0.545 | 0.581 | 0.562 | [0.40, 0.70] | 0.584 |
| ROO | Fusion, λ và τ chung | 0.550 | 0.290 | 0.577 | 0.484 | 0.526 | [0.36, 0.68] | 0.584 |
| ROO | Mặc định chưa tinh chỉnh (λ = 0.5, τ = 0.6) | 0.500 | 0.600 | 0.000 | 0.000 | 0.000 | [0.00, 0.00] | 0.584 |
| SEC | Chỉ s (λ = 1) | 1.000 | 0.490 | 0.733 | 0.710 | 0.721 | [0.57, 0.83] | 0.836 |
| SEC | Chỉ p (λ = 0) | 0.000 | 0.320 | 0.000 | 0.000 | 0.000 | [0.00, 0.00] | 0.250 |
| SEC | Fusion, λ và τ theo tenant | 0.900 | 0.440 | 0.733 | 0.710 | 0.721 | [0.59, 0.84] | 0.836 |
| SEC | Fusion, λ và τ chung | 0.550 | 0.290 | 0.724 | 0.677 | 0.700 | [0.55, 0.82] | 0.836 |
| SEC | Mặc định chưa tinh chỉnh (λ = 0.5, τ = 0.6) | 0.500 | 0.600 | 0.000 | 0.000 | 0.000 | [0.00, 0.00] | 0.836 |

Tỉ lệ cặp bị gắn cờ theo từng loại (với cặp nhãn 1 đây là recall, với nhãn 0 là tỉ lệ báo nhầm):

| loại cặp | nhãn | n | s TB | p TB | bị gắn cờ (chỉ s) | bị gắn cờ (chỉ p) | bị gắn cờ (fusion) |
|---|---|---|---|---|---|---|---|
| Hard negative (s cao, không link) | 0 | 35 | 0.663 | 0.000 | 0.657 | 0.000 | 0.657 |
| Link Duplicate/Supersede | 1 | 82 | 0.678 | 0.001 | 0.549 | 0.000 | 0.573 |
| Ngẫu nhiên | 0 | 211 | 0.473 | 0.000 | 0.014 | 0.000 | 0.014 |

## Layer 3: dự báo lỗi thời trên các mốc test

Teacher: tabpfn (context tối đa 3000 dòng, 45 giây). Horizon 90 ngày. Số dòng train/valid/test: 5623/838/1046, số dương: 357/45/52. Mốc test: 2024-06-01 đến 2024-09-01. α chọn trên validation: 0.7.

| mô hình | ROC-AUC | PR-AUC | P@10 | P@20 | ms / tài liệu |
|---|---|---|---|---|---|
| XGBoost nhãn cứng | 0.783 | 0.247 | 0.700 | 0.400 | 0.005 |
| XGBoost + soft-label | 0.778 | 0.266 | 0.800 | 0.500 | 0.005 |
| TabPFN (teacher) | 0.788 | 0.309 | 1.000 | 0.500 | 4.711 |
