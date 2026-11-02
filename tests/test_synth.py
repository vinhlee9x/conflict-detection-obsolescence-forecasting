import json

import pandas as pd
import pytest

from src.synth.generate_kb import generate_kb


@pytest.fixture(scope="module")
def kb():
    return generate_kb(n_roots_per_tenant=60, seed=7)


def _values(docs: pd.DataFrame) -> dict[str, dict[str, dict]]:
    return {d: {c["key"]: c["values"] for c in json.loads(raw)} for d, raw in zip(docs["doc_id"], docs["clauses"])}


def test_conflict_pairs_change_a_shared_clause_value(kb):
    values = _values(kb["docs"])
    for pair in kb["pairs"][kb["pairs"]["label"] == 1].itertuples():
        a, b = values[pair.doc_a], values[pair.doc_b]
        shared = set(a) & set(b)
        assert any(a[k] != b[k] for k in shared), pair


def test_paraphrase_keeps_every_shared_value(kb):
    values = _values(kb["docs"])
    for pair in kb["pairs"][kb["pairs"]["variant_type"] == "paraphrase"].itertuples():
        a, b = values[pair.doc_a], values[pair.doc_b]
        assert all(a[k] == b[k] for k in set(a) & set(b))


def test_changed_values_appear_in_text(kb):
    docs = kb["docs"].set_index("doc_id")
    for pair in kb["pairs"][kb["pairs"]["variant_type"] == "conflict"].itertuples():
        for doc_id in (pair.doc_a, pair.doc_b):
            text = docs.loc[doc_id, "text"]
            for clause in json.loads(docs.loc[doc_id, "clauses"]):
                for value in clause["values"].values():
                    assert str(value).lower() in text.lower()


def test_edges_do_not_encode_the_label(kb):
    edges = kb["edges"]
    linked = {frozenset(e) for e in zip(edges["src"], edges["dst"]) if True}
    pairs = kb["pairs"]

    def direct_link_rate(variant):
        rows = pairs[pairs["variant_type"] == variant]
        return sum(frozenset((a, b)) in linked for a, b in zip(rows["doc_a"], rows["doc_b"])) / len(rows)

    assert abs(direct_link_rate("conflict") - direct_link_rate("paraphrase")) <= 0.15
    hidden = pairs[pairs["variant_type"] == "hidden_conflict"]
    series_edges = edges[edges["type"].isin(["REFERENCES", "SAME_SERIES"])]
    series_linked = {frozenset(e) for e in zip(series_edges["src"], series_edges["dst"])}
    assert not any(frozenset((a, b)) in series_linked for a, b in zip(hidden["doc_a"], hidden["doc_b"]))


def test_obsolete_documents_are_superseded_by_a_later_conflict(kb):
    docs = kb["docs"].set_index("doc_id")
    superseded = docs[docs["superseded_at"].notna()]
    assert (superseded["variant_type"] == "root").all()
    revisions = docs[docs["variant_type"] == "conflict"]
    for revision_id, revision in revisions.iterrows():
        assert docs.loc[revision["revision_of"], "superseded_at"] == revision["issued_at"]


def test_label_ratio_close_to_one_to_three(kb):
    pairs = kb["pairs"]
    ratio = (pairs["label"] == 0).sum() / pairs["label"].sum()
    assert 3.0 <= ratio <= 5.0


def test_generation_is_deterministic():
    first = generate_kb(n_roots_per_tenant=20, seed=3)
    second = generate_kb(n_roots_per_tenant=20, seed=3)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
