import unicodedata

import numpy as np
import pandas as pd

from src.ingest import TfidfEmbedder, build_chunks, chunk, normalize
from src.similarity import DocumentVectors, top_k_neighbors


def test_normalize_unifies_unicode_dates_and_numbers():
    composed = "Đổi trả trong 07 ngày, từ 1/3/2024"
    decomposed = unicodedata.normalize("NFD", composed)
    assert normalize(decomposed) == normalize(composed) == "Đổi trả trong 7 ngày, từ 01/03/2024"


def test_chunk_respects_clause_boundaries_and_size():
    text = "Tiêu đề\n" + "\n".join(f"Điều {i}. " + "nội dung " * 20 for i in range(1, 9))
    pieces = chunk(text, max_chars=400)
    assert all(0 < len(p) <= 400 for p in pieces)
    assert "\n".join(pieces).split() == text.split()
    assert all(p.startswith("Điều") for p in pieces[1:])


def test_similarity_is_max_over_chunk_pairs():
    vectors = np.array([[1, 0], [0, 1], [0, 1], [1 / np.sqrt(2), 1 / np.sqrt(2)]], dtype=np.float32)
    docs = DocumentVectors(vectors, np.array(["a", "a", "b", "c"]))
    assert np.isclose(docs.similarity("a", "b"), 1.0)
    assert np.isclose(docs.similarity("a", "b"), docs.similarity("b", "a"))
    assert -1.0 <= docs.similarity("b", "c") <= 1.0
    matrix = docs.similarity_matrix(["a", "b", "c"])
    assert np.isclose(matrix[0, 1], docs.similarity("a", "b"))
    assert np.isclose(matrix[1, 2], docs.similarity("b", "c"))


def test_neighbors_stay_inside_tenant():
    texts = ["sao lưu dữ liệu mỗi ngày", "sao lưu dữ liệu mỗi tuần", "nghỉ phép năm", "nghỉ phép ốm"]
    docs = pd.DataFrame({"doc_id": list("abcd"), "text": texts, "tenant": ["x", "x", "y", "y"]})
    chunks = build_chunks(docs)
    vectors = DocumentVectors(TfidfEmbedder(dim=3).encode(chunks["text"].tolist()), chunks["doc_id"].to_numpy())
    neighbors = top_k_neighbors(vectors, docs.set_index("doc_id")["tenant"], k=2)
    tenant = docs.set_index("doc_id")["tenant"]
    assert (neighbors["doc_id"].map(tenant) == neighbors["neighbor_id"].map(tenant)).all()
