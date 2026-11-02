import numpy as np
import pandas as pd


class DocumentVectors:
    """Chunk vectors grouped by document, for s(A, B) = max cosine over chunk pairs."""

    def __init__(self, vectors: np.ndarray, doc_ids: np.ndarray):
        self.vectors = vectors.astype(np.float32)
        self.doc_ids = np.asarray(doc_ids).astype(str)
        order = np.argsort(self.doc_ids, kind="stable")
        self.vectors = self.vectors[order]
        self.doc_ids = self.doc_ids[order]
        unique, starts = np.unique(self.doc_ids, return_index=True)
        ends = np.append(starts[1:], len(self.doc_ids))
        self.spans = {doc: (s, e) for doc, s, e in zip(unique, starts, ends)}

    @classmethod
    def from_npz(cls, data: dict) -> "DocumentVectors":
        return cls(data["vectors"], data["doc_ids"])

    def documents(self) -> list[str]:
        return list(self.spans)

    def chunks(self, doc_id: str) -> np.ndarray:
        start, end = self.spans[doc_id]
        return self.vectors[start:end]

    def similarity(self, doc_a: str, doc_b: str) -> float:
        return float(np.max(self.chunks(doc_a) @ self.chunks(doc_b).T))

    def best_chunk_pair(self, doc_a: str, doc_b: str) -> tuple[int, int, float]:
        sims = self.chunks(doc_a) @ self.chunks(doc_b).T
        i, j = np.unravel_index(int(np.argmax(sims)), sims.shape)
        return int(i), int(j), float(sims[i, j])

    def pair_similarities(self, doc_a: list[str], doc_b: list[str]) -> np.ndarray:
        return np.array([self.similarity(a, b) for a, b in zip(doc_a, doc_b)], dtype=np.float32)

    def similarity_matrix(self, docs: list[str]) -> np.ndarray:
        """Exact s for every pair in docs: max over chunk pairs, via a chunk-level product."""
        spans = [self.spans[d] for d in docs]
        chunk_rows = np.concatenate([np.arange(s, e) for s, e in spans])
        starts = np.cumsum([0] + [e - s for s, e in spans[:-1]])
        chunk_sims = self.vectors[chunk_rows] @ self.vectors[chunk_rows].T
        by_row = np.maximum.reduceat(chunk_sims, starts, axis=0)
        return np.maximum.reduceat(by_row, starts, axis=1)

    def document_matrix(self) -> tuple[list[str], np.ndarray]:
        """Mean-pooled, re-normalized vector per document, used only to shortlist neighbors."""
        docs = self.documents()
        pooled = np.stack([self.chunks(d).mean(axis=0) for d in docs])
        pooled /= np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12
        return docs, pooled


def top_k_neighbors(vectors: DocumentVectors, groups: pd.Series, k: int) -> pd.DataFrame:
    """Top-k neighbors per document by exact s, restricted to the same group (tenant)."""
    rows = []
    docs, pooled = vectors.document_matrix()
    group_of = groups.to_dict()
    for group in sorted(set(group_of[d] for d in docs if d in group_of)):
        members = [i for i, d in enumerate(docs) if group_of.get(d) == group]
        shortlist = min(len(members) - 1, k * 3)
        sims = pooled[members] @ pooled[members].T
        np.fill_diagonal(sims, -np.inf)
        for row, i in enumerate(members):
            candidates = np.argsort(-sims[row])[:shortlist]
            exact = [(docs[members[c]], vectors.similarity(docs[i], docs[members[c]])) for c in candidates]
            exact.sort(key=lambda item: -item[1])
            rows.extend({"doc_id": docs[i], "neighbor_id": n, "s": s, "rank": r} for r, (n, s) in enumerate(exact[:k], 1))
    return pd.DataFrame(rows)
