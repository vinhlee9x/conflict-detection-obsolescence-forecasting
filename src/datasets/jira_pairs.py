"""Labeled pairs from Jira links: positives are label links (Duplicate or a
supersede-like type), negatives are high-similarity unlinked neighbors (hard)
plus random pairs from the same project."""

import numpy as np
import pandas as pd


def build_pairs(
    docs: pd.DataFrame,
    edges: pd.DataFrame,
    neighbors: pd.DataFrame,
    negatives_per_positive: int = 3,
    hard_share: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenant_of = docs.set_index("doc_id")["tenant"].to_dict()
    issued = docs.set_index("doc_id")["issued_at"].to_dict()
    linked = {frozenset((a, b)) for a, b in zip(edges["src"], edges["dst"])}

    positives = edges[edges["is_label_edge"]]
    rows = [_pair(a, b, 1, "label_link", tenant_of, issued) for a, b in zip(positives["src"], positives["dst"])]
    rows = list({frozenset((r["doc_a"], r["doc_b"])): r for r in rows}.values())
    taken = {frozenset((r["doc_a"], r["doc_b"])) for r in rows}
    n_positive = len(rows)

    n_negative = negatives_per_positive * n_positive
    n_hard = int(round(n_negative * hard_share))
    hard = neighbors.sample(frac=1.0, random_state=seed)
    for a, b in zip(hard["doc_id"], hard["neighbor_id"]):
        if len(rows) - n_positive >= n_hard:
            break
        key = frozenset((a, b))
        if key in taken or key in linked:
            continue
        taken.add(key)
        rows.append(_pair(a, b, 0, "hard_negative", tenant_of, issued))

    ids = docs["doc_id"].to_numpy()
    target = len(rows) + (n_negative - n_hard)
    attempts = 0
    while len(rows) < target and attempts < n_negative * 50:
        attempts += 1
        a, b = rng.choice(ids, size=2, replace=False)
        key = frozenset((a, b))
        if key in taken or key in linked or tenant_of[a] != tenant_of[b]:
            continue
        taken.add(key)
        rows.append(_pair(a, b, 0, "random", tenant_of, issued))
    return pd.DataFrame(rows)


def _pair(a, b, label, variant_type, tenant_of, issued) -> dict:
    first, second = (a, b) if issued[a] <= issued[b] else (b, a)
    return {
        "doc_a": first,
        "doc_b": second,
        "tenant": tenant_of[first],
        "group_a": first,
        "group_b": second,
        "label": label,
        "variant_type": variant_type,
    }
