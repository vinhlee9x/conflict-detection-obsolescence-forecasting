"""Human check of the generated labels: export a stratified blind sample, then
compute Cohen's kappa once each member has filled a column `annot_<name>`
with 1 (conflict) or 0 (no conflict)."""

from itertools import combinations

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def export_blind_sample(docs: pd.DataFrame, pairs: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    per_variant = max(1, n // pairs["variant_type"].nunique())
    sample = (
        pairs.groupby("variant_type", group_keys=False)
        .apply(lambda g: g.sample(n=min(per_variant, len(g)), random_state=seed))
        .sample(frac=1.0, random_state=seed)
        .head(n)
    )
    text = docs.set_index("doc_id")["text"]
    return pd.DataFrame({
        "pair_id": range(1, len(sample) + 1),
        "doc_a": sample["doc_a"].to_numpy(),
        "doc_b": sample["doc_b"].to_numpy(),
        "text_a": sample["doc_a"].map(text).to_numpy(),
        "text_b": sample["doc_b"].map(text).to_numpy(),
    })


def kappa_report(annotated: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    labels = pairs.set_index(["doc_a", "doc_b"])["label"]
    generated = annotated.set_index(["doc_a", "doc_b"]).index.map(labels)
    annotators = [c for c in annotated.columns if c.startswith("annot_")]
    rows = [{"a": "generated", "b": name, "kappa": cohen_kappa_score(generated, annotated[name])} for name in annotators]
    rows += [{"a": x, "b": y, "kappa": cohen_kappa_score(annotated[x], annotated[y])} for x, y in combinations(annotators, 2)]
    return pd.DataFrame(rows)


def disagreements(annotated: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    labels = pairs.set_index(["doc_a", "doc_b"])["label"]
    annotated = annotated.assign(generated=annotated.set_index(["doc_a", "doc_b"]).index.map(labels).to_numpy())
    annotators = [c for c in annotated.columns if c.startswith("annot_")]
    return annotated[(annotated[annotators].ne(annotated["generated"], axis=0)).any(axis=1)]
