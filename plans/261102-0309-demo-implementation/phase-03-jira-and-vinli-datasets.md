---
phase: 3
title: "Dataset Jira và ViNLI"
status: in-progress
priority: P2
effort: 10h
depends_on: [1]
---

# Phase 03: Dataset Jira và ViNLI

## Mục tiêu

Bổ sung dữ liệu thật để số liệu không chỉ dựa vào KB tổng hợp: Jira cho đồ thị thật và nhãn proxy, ViNLI cho câu hỏi "embedding có tách được mâu thuẫn khỏi giống chủ đề không".

## File tạo mới

```text
src/datasets/jira_export.py      # đọc MongoDB đã restore, xuất parquet cho 1–2 project
src/datasets/jira_link_stats.py  # thống kê link type và resolution
src/datasets/jira_pairs.py       # tạo cặp dương, hard negative, cặp ngẫu nhiên
src/datasets/vinli.py            # tải và chuẩn hoá ViNLI
notebooks/01_build_dataset.ipynb # phần Jira và ViNLI (phần KB gọi src/synth)
data/processed/jira_docs.parquet
data/processed/jira_edges.parquet
data/processed/jira_pairs.parquet
data/processed/vinli.parquet
tests/test_jira_pairs.py
```

## Các bước

### Jira (timebox 6h)

1. Trên Colab, tải bản dump từ [Zenodo](https://zenodo.org/doi/10.5281/zenodo.5882881), cài MongoDB, `mongorestore` chỉ các collection cần thiết. Đã tự động hoá trong `src/datasets/jira_setup.py` và notebook 01 (`RUN_JIRA = True`): bản mới nhất là record 15719919, một file zip 5,8 GB chứa `mongodump-JiraReposAnon.archive` (gzip, database `JiraReposAnon`). Restore toàn bộ cần khoảng 60 GB nên archive được stream từ zip vào `mongorestore --nsInclude` chỉ với collection đã chọn.
2. Chọn 1–2 project có vài nghìn issue và nhiều link (ưu tiên instance nhỏ). Xuất `jira_docs` (`summary + description`, `created`, `updated`, `status`, `resolution`, `resolutiondate`) và `jira_edges` (issue link với `type`).
3. Chạy `jira_link_stats.py` và ghi kết quả vào notebook. Quyết định nhãn dựa trên số liệu thật:
   - Nếu có link kiểu "Supersedes" hoặc "Replaces", hoặc resolution kiểu "Obsolete" hay "Won't Fix" kèm link, dùng làm nhãn sát nghĩa hơn.
   - Nếu không, dùng "Duplicate" làm nhãn proxy và ghi rõ trong báo cáo rằng đây là trùng lặp, không phải mâu thuẫn.
4. `jira_pairs.py`: dương là cặp có link nhãn; âm gồm hard negative (cùng project, s cao, không có link) và cặp ngẫu nhiên. Hard negative cần embedding nên bước này chạy lại sau phase 04.
5. Cạnh nhãn (Duplicate hoặc Supersedes) được đánh dấu `is_label_edge = True` trong `jira_edges` để phase 05 loại ra khi tính p.
6. Layer 3 trên Jira (tuỳ chọn): issue được coi là "lỗi thời" khi bị resolve là Duplicate hoặc Obsolete; mốc thời gian lấy từ `resolutiondate`.

Nếu quá 6h mà chưa xuất được parquet, dừng phần Jira, ghi lý do vào báo cáo và đi tiếp với KB tổng hợp cộng ViNLI.

### ViNLI

1. Tải [ViNLI](https://aclanthology.org/2022.coling-1.339) (theo link trong bài báo), chuẩn hoá NFC, giữ `premise`, `hypothesis`, `label`.
2. Lấy mẫu phân tầng khoảng 3.000 cặp đủ ba nhãn (entailment, neutral, contradiction).
3. Mục đích dùng ở phase 05: vẽ phân phối cosine BGE-M3 theo từng nhãn. Nếu cặp contradiction có cosine cao gần bằng entailment, đó là bằng chứng rằng chỉ dùng s là không đủ, và là lý do cần thêm tín hiệu đồ thị p.

WikiContradict là tuỳ chọn, chỉ thêm nếu còn thời gian.

## Kiểm tra

- `tests/test_jira_pairs.py`: không có cặp trùng, không có cặp tự nối, mọi cạnh nhãn đều được đánh dấu, tỉ lệ dương/âm đúng cấu hình.

## Tiêu chí hoàn thành

- Có `jira_pairs.parquet` với ít nhất 300 cặp dương, hoặc có ghi chú lý do bỏ Jira.
- Có `vinli.parquet` với ba nhãn cân bằng.

## Ghi chú triển khai Jira (2026-11-02)

- Đã chạy thật trên máy cloud: tải dump khoảng 20 phút, restore collection `Spring` và `Hyperledger` khoảng 2 phút, không lỗi. Dump được tạo từ MongoDB 7.0.14, restore vào 8.0 có cảnh báo khác phiên bản nhưng 0 document lỗi.
- `Hyperledger` không có project nào trong khoảng 500–5.000 issue đạt 300 cặp dương. `Spring` với ROO, SEC, DATAMONGO cho 9.727 issue và 513 cạnh nhãn (Duplicate 166 trong ROO, còn lại là Supersede và Duplicate ở hai project kia), nên chọn làm mặc định.
- Nhóm chia dev/test của Jira là thành phần liên thông theo link. Chia theo từng issue thì cả hai đầu một cặp Duplicate chỉ cùng rơi vào test khoảng 4%, test chỉ còn 16 cặp dương. Sau khi sửa, test có 100 dương và 300 âm.
- Chạy thử với baseline TF-IDF trên CPU: F1 test của s khoảng 0,62, p bằng 0 vì cạnh nhãn bị loại và link còn lại hiếm khi nối hai issue trùng nhau. Số liệu chính thức cần chạy BGE-M3 trên Colab.
