import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.settings import EDGE_WEIGHTS, GAMMA, K_HOPS, LABEL_EDGE_TYPES


class KnowledgeGraph:
    """Undirected weighted graph over documents with multi-hop propagation

    p(A, B) = sum_{k=1..K} gamma^k * M^k[A, B] / sum_{k=1..K} gamma^k,  M = D^-1 W

    Rows of M^k sum to 1, so p lies in [0, 1] and shares a scale with s.
    Label edges (confirmed conflicts, Jira Duplicate links) are excluded so the
    feature never sees the answer.
    """

    def __init__(
        self,
        doc_ids: list[str],
        edges: pd.DataFrame,
        edge_weights: dict[str, float] = EDGE_WEIGHTS,
        exclude_types: set[str] = LABEL_EDGE_TYPES,
        default_weight: float = 1.0,
    ):
        self.doc_ids = list(doc_ids)
        self.index = {d: i for i, d in enumerate(self.doc_ids)}
        self.edges = edges.reset_index(drop=True)
        self.exclude_types = set(exclude_types)
        self.edge_weights = edge_weights
        self.default_weight = default_weight
        self._rebuild()

    def propagation_edges(self) -> pd.DataFrame:
        usable = self.edges[~self.edges["type"].isin(self.exclude_types)]
        if "is_label_edge" in usable:
            usable = usable[~usable["is_label_edge"].fillna(False).astype(bool)]
        return usable[usable["src"].isin(self.index) & usable["dst"].isin(self.index)]

    def _rebuild(self) -> None:
        n = len(self.doc_ids)
        usable = self.propagation_edges()
        rows = usable["src"].map(self.index).to_numpy()
        cols = usable["dst"].map(self.index).to_numpy()
        weights = usable["type"].map(lambda t: self.edge_weights.get(t, self.default_weight)).to_numpy(dtype=float)
        adjacency = sp.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
        adjacency = adjacency + adjacency.T
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        inverse = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
        self.transition = (sp.diags(inverse) @ adjacency).tocsr()

    def add_edges(self, new_edges: pd.DataFrame) -> None:
        self.edges = pd.concat([self.edges, new_edges], ignore_index=True)
        self._rebuild()

    def add_documents(self, doc_ids: list[str]) -> None:
        for doc_id in doc_ids:
            if doc_id not in self.index:
                self.index[doc_id] = len(self.doc_ids)
                self.doc_ids.append(doc_id)
        self._rebuild()

    def propagation_from(self, sources: list[str], k_hops: int = K_HOPS, gamma: float = GAMMA) -> np.ndarray:
        """Dense (len(sources), n) matrix of normalized k-hop propagation scores."""
        source_index = np.array([self.index[s] for s in sources])
        current = sp.csr_matrix(
            (np.ones(len(source_index)), (np.arange(len(source_index)), source_index)),
            shape=(len(source_index), len(self.doc_ids)),
        )
        total = np.zeros(current.shape)
        for k in range(1, k_hops + 1):
            current = current @ self.transition
            total += (gamma ** k) * current.toarray()
        return total / sum(gamma ** k for k in range(1, k_hops + 1))

    def pair_scores(self, doc_a: list[str], doc_b: list[str], k_hops: int = K_HOPS, gamma: float = GAMMA) -> np.ndarray:
        """Symmetric p: max of the two propagation directions."""
        nodes = sorted(set(doc_a) | set(doc_b))
        position = {d: i for i, d in enumerate(nodes)}
        scores = self.propagation_from(nodes, k_hops, gamma)
        forward = np.array([scores[position[a], self.index[b]] for a, b in zip(doc_a, doc_b)])
        backward = np.array([scores[position[b], self.index[a]] for a, b in zip(doc_a, doc_b)])
        return np.maximum(forward, backward).astype(np.float32)

    def within_hops(self, doc_id: str, k_hops: int = K_HOPS) -> set[str]:
        reached = self.propagation_from([doc_id], k_hops, gamma=1.0)[0]
        return {self.doc_ids[i] for i in np.flatnonzero(reached > 0) if self.doc_ids[i] != doc_id}
