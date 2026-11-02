---
phase: 8
title: "Tổng hợp kết quả và chuẩn bị báo cáo"
status: pending
priority: P2
effort: 4h
depends_on: [5, 6, 7]
---

# Phase 08: Tổng hợp kết quả và chuẩn bị báo cáo

## Mục tiêu

Gom số liệu vào báo cáo theo đúng thứ thầy quan tâm (thuật toán, dataset, kết quả) và chắc chắn buổi demo không gặp sự cố.

## File tạo mới

```text
src/report_tables.py         # đọc *_results.json, xuất bảng Markdown/LaTeX và hình
docs/results.md              # bảng và hình cuối cùng, kèm cách tái tạo
```

## Các bước

1. `report_tables.py` xuất:
   - Thống kê dataset: số tài liệu, số cặp, tỉ lệ nhãn, theo tenant và theo nguồn; kappa của nhóm.
   - Bảng ablation Layer 2 trên test, λ và τ đã chọn trên dev.
   - Hình ViNLI: phân phối cosine theo nhãn.
   - Bảng Layer 3 ba model và thời gian suy luận.
2. Viết phần "Hạn chế" thẳng thắn: Jira dùng nhãn proxy, KB là dữ liệu tổng hợp, p yếu với mâu thuẫn ẩn.
3. Phần kiến trúc trong báo cáo nhắc PostgreSQL + pgvector như hướng triển khai thực tế, nói rõ demo dùng numpy và NetworkX.
4. Đưa hình từ `docs/figures/` vào slide theo thứ tự: dataset, động lực (ViNLI), Layer 2, Layer 3.
5. Diễn tập 5–7 phút; mở sẵn notebook 04 đã chạy để trả lời nếu thầy hỏi chi tiết.

## Tiêu chí hoàn thành

- Mọi con số trong slide và báo cáo truy được về một file trong `artifacts/`.
- Slide có đủ hình và bảng, mỗi hình truy được về notebook 04.
