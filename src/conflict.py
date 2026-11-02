import numpy as np
import pandas as pd

from src.graph import KnowledgeGraph
from src.similarity import DocumentVectors


def normalize_similarity(s: np.ndarray, s_min: float, s_max: float) -> np.ndarray:
    """Min-max scale s to [0, 1] with bounds estimated on dev only."""
    return np.clip((np.asarray(s) - s_min) / max(s_max - s_min, 1e-9), 0.0, 1.0)


def fuse(s_norm: np.ndarray, p: np.ndarray, lam: float) -> np.ndarray:
    return lam * np.asarray(s_norm) + (1.0 - lam) * np.asarray(p)


def tenant_config(config: dict, tenant: str) -> dict:
    return config["tenants"].get(tenant, config["global"])


def score_pairs(pairs: pd.DataFrame, config: dict, mode: str = "tenant") -> np.ndarray:
    scores = np.zeros(len(pairs))
    for tenant, index in pairs.groupby("tenant").groups.items():
        cfg = config["global"] if mode == "global" else tenant_config(config, tenant)
        rows = pairs.loc[index]
        s_norm = normalize_similarity(rows["s"], cfg["s_min"], cfg["s_max"])
        scores[pairs.index.get_indexer(index)] = fuse(s_norm, rows["p"], cfg["FUSION_LAMBDA"])
    return scores


def detect_candidates(
    new_doc_ids: list[str],
    docs: pd.DataFrame,
    vectors: DocumentVectors,
    graph: KnowledgeGraph,
    config: dict,
    top_k: int,
) -> pd.DataFrame:
    """Candidate conflicts for newly added documents: top-k neighbors by s plus
    every document within K hops, scored with the tenant's lambda and tau."""
    tenant_of = docs.set_index("doc_id")["tenant"].to_dict()
    known = [d for d in vectors.documents() if d in tenant_of]
    rows = []
    for doc_id in new_doc_ids:
        tenant = tenant_of[doc_id]
        same_tenant = [d for d in known if tenant_of[d] == tenant and d != doc_id]
        sims = {d: vectors.similarity(doc_id, d) for d in same_tenant}
        shortlist = set(sorted(sims, key=sims.get, reverse=True)[:top_k]) | (graph.within_hops(doc_id) & set(same_tenant))
        others = sorted(shortlist)
        if not others:
            continue
        cfg = tenant_config(config, tenant)
        s = np.array([sims[d] for d in others])
        p = graph.pair_scores([doc_id] * len(others), others)
        score = fuse(normalize_similarity(s, cfg["s_min"], cfg["s_max"]), p, cfg["FUSION_LAMBDA"])
        for other, s_i, p_i, score_i in zip(others, s, p, score):
            if score_i > cfg["CONFLICT_THRESHOLD_TAU"]:
                rows.append({"doc_id": doc_id, "candidate_id": other, "s": s_i, "p": p_i, "score": score_i})
    columns = ["doc_id", "candidate_id", "s", "p", "score"]
    return pd.DataFrame(rows, columns=columns).sort_values("score", ascending=False).reset_index(drop=True)


def confirm_conflicts(graph: KnowledgeGraph, confirmed: pd.DataFrame) -> None:
    """A human-confirmed candidate becomes a CONFLICTS_WITH edge. It is stored for
    Layer 3 features but never used by p (it is a label edge)."""
    graph.add_edges(pd.DataFrame({"src": confirmed["doc_id"], "dst": confirmed["candidate_id"], "type": "CONFLICTS_WITH"}))
