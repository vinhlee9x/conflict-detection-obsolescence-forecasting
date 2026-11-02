import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def assign_group_splits(pairs: pd.DataFrame, test_fraction: float, seed: int) -> pd.DataFrame:
    """Split by root-document group within each tenant, so the same source
    document never appears in both dev and test. Pairs whose two groups land in
    different splits are dropped."""
    split_of: dict[str, str] = {}
    for tenant, frame in pairs.groupby("tenant"):
        groups = np.array(sorted(set(frame["group_a"]) | set(frame["group_b"])))
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        dev_idx, test_idx = next(splitter.split(groups, groups=groups))
        split_of.update({g: "dev" for g in groups[dev_idx]})
        split_of.update({g: "test" for g in groups[test_idx]})

    result = pairs.assign(split_a=pairs["group_a"].map(split_of), split_b=pairs["group_b"].map(split_of))
    result = result[result["split_a"] == result["split_b"]].rename(columns={"split_a": "split"})
    return result.drop(columns="split_b").reset_index(drop=True)


def top_up_random_negatives(pairs: pd.DataFrame, docs: pd.DataFrame, negatives_per_positive: int, seed: int) -> pd.DataFrame:
    """Random pairs mostly straddle splits and get dropped, so refill each
    tenant/split with random root-to-root pairs drawn inside that split."""
    rng = np.random.default_rng(seed)
    meta = docs.set_index("doc_id")
    topic = meta["topic"] if "topic" in meta else pd.Series("", index=meta.index)
    taken = {frozenset(k) for k in zip(pairs["doc_a"], pairs["doc_b"])}
    extra = []
    for (tenant, split), frame in pairs.groupby(["tenant", "split"]):
        missing = negatives_per_positive * int(frame["label"].sum()) - int((frame["label"] == 0).sum())
        groups = sorted((set(frame["group_a"]) | set(frame["group_b"])) & set(meta.index))
        attempts = 0
        while missing > 0 and len(groups) > 1 and attempts < 100 * negatives_per_positive * len(frame):
            attempts += 1
            a, b = rng.choice(groups, size=2, replace=False)
            if frozenset((a, b)) in taken:
                continue
            taken.add(frozenset((a, b)))
            first, second = (a, b) if meta.loc[a, "issued_at"] <= meta.loc[b, "issued_at"] else (b, a)
            extra.append({
                "doc_a": first, "doc_b": second, "tenant": tenant, "group_a": first, "group_b": second,
                "label": 0, "variant_type": "same_template" if topic[a] == topic[b] and topic[a] else "random", "split": split,
            })
            missing -= 1
    return pd.concat([pairs, pd.DataFrame(extra, columns=pairs.columns)], ignore_index=True) if extra else pairs


def rebalance_negatives(pairs: pd.DataFrame, negatives_per_positive: int, seed: int) -> pd.DataFrame:
    """Trim random negatives (never hard ones) so each tenant/split keeps the target ratio."""
    parts = []
    for _, frame in pairs.groupby(["tenant", "split"]):
        n_target = negatives_per_positive * int(frame["label"].sum())
        negatives = frame[frame["label"] == 0]
        excess = len(negatives) - n_target
        removable = negatives[negatives["variant_type"].isin(["random", "same_template"])]
        if excess > 0:
            frame = frame.drop(removable.sample(n=min(excess, len(removable)), random_state=seed).index)
        parts.append(frame)
    return pd.concat(parts).reset_index(drop=True)
