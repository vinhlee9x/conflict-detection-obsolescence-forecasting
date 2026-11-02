---
phase: 4
title: "Layer 1: tiền xử lý và embedding"
status: completed
priority: P1
effort: 6h
depends_on: [2, 3]
---

# Phase 04: Layer 1, tiền xử lý và embedding

## Mục tiêu

Biến mọi tài liệu thành các chunk đã chuẩn hoá và vector BGE-M3, rồi tính được s của một cặp tài liệu.

## File tạo mới

```text
src/ingest.py                 # normalize, chunk, embed, ocr (tuỳ chọn)
src/similarity.py             # s(A, B) = max cosine giữa các chunk
artifacts/embeddings_kb.npz   # chunk_vectors (float16), chunk_doc_ids, chunk_texts index
artifacts/embeddings_jira.npz
artifacts/embeddings_vinli.npz
artifacts/neighbors.parquet   # top-k láng giềng theo s cho mỗi tài liệu
tests/test_ingest.py
tests/test_similarity.py
```

## Các bước

1. `normalize`: Unicode NFC, bỏ header/footer lặp, gộp khoảng trắng, chuẩn hoá định dạng ngày và số (để "7 ngày" và "07 ngày" giống nhau).
2. `chunk`: tách theo điều khoản hoặc mục; nếu không có cấu trúc thì cắt khoảng 300–500 token có chồng lấn nhỏ.
3. `embed`: BGE-M3 dense vector (1024 chiều), chuẩn hoá L2, batch trên GPU T4. Lưu `float16` để giảm dung lượng.
4. `similarity.py`: `s(A, B) = max_{i ∈ A, j ∈ B} cos(a_i, b_j)`. Lấy max thay vì trung bình vì mâu thuẫn thường nằm ở một điều khoản.
5. Tạo `neighbors.parquet`: với mỗi tài liệu lấy top-k (k = `TOP_K_NEIGHBORS`) theo s trong cùng tenant. Danh sách này là tập candidate cho Layer 2 và là nguồn hard negative cho Jira (quay lại bước 4 của phase 03).
6. OCR (tuỳ chọn): một file PDF scan mẫu chạy qua Tesseract gói `vie` để chứng minh bước này có trong pipeline; không đưa vào thí nghiệm định lượng.

## Kiểm tra

- `tests/test_ingest.py`: chuỗi tổ hợp dấu (NFD) và dựng sẵn (NFC) cho cùng kết quả; chunk không rỗng và không vượt độ dài.
- `tests/test_similarity.py`: dùng vector nhỏ dựng tay, kiểm tra s đối xứng, nằm trong [-1, 1], và bằng 1 khi hai tài liệu chung một chunk.

## Tiêu chí hoàn thành

- Có file embedding cho KB, ViNLI (và Jira nếu phase 03 xong), số chunk khớp với số chunk trong parquet.
- Thời gian embed toàn bộ KB trên T4 được ghi lại (dùng cho báo cáo).
