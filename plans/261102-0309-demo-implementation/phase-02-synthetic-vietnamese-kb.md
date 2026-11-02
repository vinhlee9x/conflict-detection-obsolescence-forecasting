---
phase: 2
title: "Sinh KB tổng hợp tiếng Việt"
status: pending
priority: P1
effort: 12h
depends_on: [1]
---

# Phase 02: Sinh KB tổng hợp tiếng Việt

## Mục tiêu

Tạo nguồn dữ liệu chính cho demo: 2 tenant, tổng 400–600 tài liệu, có nhãn mâu thuẫn đúng ngay từ lúc sinh, có đồ thị và trục thời gian.

## File tạo mới

```text
src/synth/templates.py        # template SOP, chính sách, hướng dẫn kỹ thuật theo tenant
src/synth/generate_roots.py   # gọi LLM sinh tài liệu gốc, lưu raw JSON kèm prompt và seed
src/synth/mutate.py           # sinh biến thể có nhãn bằng luật
src/synth/build_kb.py         # ghép thành docs, pairs, edges, timeline
src/synth/qa_sample.py        # lấy 100 cặp cho nhóm gán lại, tính Cohen's kappa
data/raw/synth/roots_*.json
data/processed/kb_docs.parquet
data/processed/kb_pairs.parquet
data/processed/kb_edges.parquet
data/processed/kb_qa_sample.csv
tests/test_synth.py
```

## Thiết kế dữ liệu

`kb_docs`: `doc_id`, `tenant` (`tech` hoặc `policy`), `root_id`, `series_id`, `version`, `title`, `text`, `department`, `issued_at`, `superseded_at` (rỗng nếu còn hiệu lực).

`kb_pairs`: `doc_a`, `doc_b`, `tenant`, `root_id`, `label` (1 mâu thuẫn, 0 không), `variant_type`.

`kb_edges`: `src`, `dst`, `type` (`REFERENCES`, `SAME_SERIES`, `SAME_DEPARTMENT`).

## Các bước

1. Viết template cho mỗi tenant với các trường cố định: mã tài liệu (`SOP-047`, `POL-012`), ngày ban hành, phòng ban, 4–8 điều khoản, mỗi điều khoản có ít nhất một con số hoặc điều kiện cụ thể. Tenant `tech` dùng từ vựng đa dạng hơn, tenant `policy` lặp nhiều cụm như "quy trình", "phê duyệt", "đơn hàng" để τ tối ưu của hai tenant khác nhau một cách tự nhiên.
2. `generate_roots.py`: sinh khoảng 150 tài liệu gốc mỗi tenant. Đầu ra LLM ở dạng JSON có cấu trúc (danh sách điều khoản, mỗi điều khoản có trường số liệu riêng) để bước mutate sửa đúng chỗ. Lưu prompt, model, seed vào raw JSON. Chỉ chạy một lần, sau đó chỉ đọc lại file raw.
3. `mutate.py` sinh biến thể từ mỗi tài liệu gốc:
   - **Mâu thuẫn** (nhãn 1): đổi con số, điều kiện, phạm vi hoặc người phê duyệt của 1–2 điều khoản (ví dụ 7 ngày thành 3 ngày). Thao tác bằng luật trên trường có cấu trúc nên nhãn chắc chắn đúng. Có thể nhờ LLM viết lại câu cho tự nhiên, sau đó kiểm tra con số mới vẫn còn trong văn bản.
   - **Diễn đạt lại** (nhãn 0): cùng nội dung, khác cách viết, do LLM viết lại; kiểm tra mọi con số gốc vẫn còn.
   - **Trùng từ vựng, khác chủ đề** (nhãn 0, hard negative): ghép tiêu đề và cụm từ quen thuộc với nội dung khác.
   - **Mâu thuẫn ẩn** (nhãn 1): như mâu thuẫn nhưng không có cạnh nào nối với bản gốc.
4. `build_kb.py` dựng cạnh và trục thời gian:
   - REFERENCES bằng regex mã tài liệu trong văn bản (`SOP-\d+`, `POL-\d+`).
   - Chống rò rỉ nhãn qua cạnh: cả bản **mâu thuẫn** lẫn bản **diễn đạt lại** đều là phiên bản mới cùng `series_id`, đều có SAME_SERIES và REFERENCES về bản gốc. Như vậy cạnh chỉ cho biết "có liên quan", không cho biết "có mâu thuẫn". Bản mâu thuẫn ẩn và hard negative không có cạnh trực tiếp.
   - Rải `issued_at` trên 2–3 năm. Tài liệu gốc bị một bản mâu thuẫn mới hơn thay thế thì `superseded_at = issued_at` của bản mới. Một phần bản mâu thuẫn chưa được phát hiện để Layer 3 có việc dự báo.
5. Cặp âm thêm: cặp ngẫu nhiên cùng tenant. Giữ tỉ lệ khoảng 1 dương : 3 âm.
6. `qa_sample.py`: lấy ngẫu nhiên phân tầng 100 cặp, xuất CSV ẩn nhãn. Mỗi thành viên gán độc lập, script tính kappa giữa từng người với nhãn sinh và giữa các người với nhau. Cặp bị gán khác nhãn sinh thì xem lại, sửa hoặc bỏ.

## Kiểm tra

- `tests/test_synth.py`: mỗi cặp nhãn 1 thực sự có ít nhất một con số hoặc điều kiện khác nhau; mỗi cặp diễn đạt lại giữ nguyên tập con số; tỉ lệ cặp có cạnh trực tiếp giữa nhãn 1 và nhãn 0 không chênh quá 15 điểm phần trăm (trừ nhóm mâu thuẫn ẩn).
- Chạy `build_kb.py` hai lần cùng seed cho ra file giống hệt nhau.

## Tiêu chí hoàn thành

- 400–600 tài liệu, 2 tenant, tỉ lệ nhãn khoảng 1:3.
- Kappa giữa nhóm và nhãn sinh từ 0,8 trở lên (nếu thấp hơn thì sửa luật mutate trước khi sang phase 04).

## Rủi ro

- LLM viết lại làm mất con số. Bước kiểm tra tự động ở mục 3 loại các bản này.
- Chi phí API: 300 tài liệu gốc cộng khoảng 300 bản viết lại là vài trăm lời gọi, nên chạy batch và cache.
