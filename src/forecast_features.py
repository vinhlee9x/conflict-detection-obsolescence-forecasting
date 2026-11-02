"""Point-in-time feature table for obsolescence forecasting.

One row per (root document, monthly snapshot t) while the document is still in
force at t. Every feature is computed from documents issued, edges created and
conflicts confirmed on or before t; the label looks only at (t, t + horizon].
"""

import numpy as np
import pandas as pd

from src.conflict import fuse, normalize_similarity
from src.graph import KnowledgeGraph
from src.similarity import DocumentVectors

NUMERIC_FEATURES = [
    "age_days",
    "n_confirmed_conflicts",
    "n_open_candidates",
    "max_candidate_score",
    "n_cited_by",
    "n_cites",
    "n_recent_department_docs",
]
CATEGORICAL_FEATURES = ["tenant", "department", "topic"]
RECENT_WINDOW_DAYS = 90


def snapshot_dates(docs: pd.DataFrame, horizon_days: int, freq: str, warmup_months: int = 3) -> list[pd.Timestamp]:
    start = docs["issued_at"].min().normalize() + pd.DateOffset(months=warmup_months)
    end = docs["issued_at"].max() - pd.Timedelta(days=horizon_days)
    return list(pd.date_range(start, end, freq=freq))


def build_forecast_table(
    docs: pd.DataFrame,
    edges: pd.DataFrame,
    conflict_events: pd.DataFrame,
    vectors: DocumentVectors,
    config: dict,
    snapshots: list[pd.Timestamp],
    horizon_days: int,
) -> pd.DataFrame:
    issued = docs.set_index("doc_id")["issued_at"]
    references = edges[edges["type"] == "REFERENCES"].assign(created_at=lambda e: e["src"].map(issued))
    tables = []
    for tenant, tenant_docs in docs.groupby("tenant"):
        tenant_ids = tenant_docs["doc_id"].tolist()
        similarity = pd.DataFrame(vectors.similarity_matrix(tenant_ids), index=tenant_ids, columns=tenant_ids)
        cfg = config["tenants"][tenant]
        for t in snapshots:
            rows = _snapshot_rows(tenant_docs, edges, references, conflict_events, similarity, cfg, t, horizon_days)
            if rows is not None:
                tables.append(rows)
    return pd.concat(tables, ignore_index=True)


def _snapshot_rows(docs, edges, references, events, similarity, cfg, t, horizon_days):
    known = docs[docs["issued_at"] <= t]
    active = known[(known["variant_type"] == "root") & (known["superseded_at"].isna() | (known["superseded_at"] > t))]
    if active.empty:
        return None

    known_ids = known["doc_id"].tolist()
    active_ids = active["doc_id"].tolist()
    graph = KnowledgeGraph(known_ids, edges)
    propagation = graph.propagation_from(known_ids)
    symmetric = np.maximum(propagation, propagation.T)
    active_rows = [graph.index[d] for d in active_ids]
    p = symmetric[active_rows]
    s = similarity.loc[active_ids, known_ids].to_numpy()
    score = fuse(normalize_similarity(s, cfg["s_min"], cfg["s_max"]), p, cfg["FUSION_LAMBDA"])
    self_mask = np.equal.outer(np.array(active_ids), np.array(known_ids))
    score[self_mask] = -np.inf

    confirmed = events[events["confirmed_at"] <= t]
    confirmed_pairs = set(zip(confirmed["doc_a"], confirmed["doc_b"])) | set(zip(confirmed["doc_b"], confirmed["doc_a"]))
    confirmed_mask = np.array([[(a, b) in confirmed_pairs for b in known_ids] for a in active_ids])
    candidate_mask = (score > cfg["CONFLICT_THRESHOLD_TAU"]) & ~confirmed_mask

    refs = references[references["created_at"] <= t]
    cited_by = refs["dst"].value_counts()
    cites = refs["src"].value_counts()
    recent = known[known["issued_at"] > t - pd.Timedelta(days=RECENT_WINDOW_DAYS)]
    recent_by_department = recent["department"].value_counts()
    horizon_end = t + pd.Timedelta(days=horizon_days)

    frame = pd.DataFrame({
        "doc_id": active_ids,
        "snapshot": t,
        "tenant": active["tenant"].to_numpy(),
        "department": active["department"].to_numpy(),
        "topic": active["topic"].to_numpy(),
        "age_days": (t - active["issued_at"]).dt.days.to_numpy(),
        "n_confirmed_conflicts": confirmed_mask.sum(axis=1),
        "n_open_candidates": candidate_mask.sum(axis=1),
        "max_candidate_score": np.where(np.isfinite(score).any(axis=1), np.max(np.where(np.isfinite(score), score, 0.0), axis=1), 0.0),
        "n_cited_by": [int(cited_by.get(d, 0)) for d in active_ids],
        "n_cites": [int(cites.get(d, 0)) for d in active_ids],
        "n_recent_department_docs": [int(recent_by_department.get(dep, 0)) for dep in active["department"]],
    })
    superseded = active["superseded_at"].to_numpy()
    frame["label"] = [int(pd.notna(x) and t < x <= horizon_end) for x in superseded]
    return frame


def temporal_split(table: pd.DataFrame, n_valid: int, n_test: int) -> pd.Series:
    snapshots = sorted(table["snapshot"].unique())
    test = set(snapshots[-n_test:])
    valid = set(snapshots[-(n_test + n_valid):-n_test])
    return table["snapshot"].map(lambda s: "test" if s in test else "valid" if s in valid else "train")


def encode(table: pd.DataFrame) -> pd.DataFrame:
    features = table[NUMERIC_FEATURES].astype(float)
    dummies = pd.get_dummies(table[CATEGORICAL_FEATURES], dtype=float)
    return pd.concat([features, dummies], axis=1)
