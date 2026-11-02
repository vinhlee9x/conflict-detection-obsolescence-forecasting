import pandas as pd

from src.datasets.jira_export import export_issues
from src.datasets.jira_pairs import build_pairs


def _issue(key, created, links=(), resolution=None):
    return {
        "key": key,
        "fields": {
            "summary": f"Issue {key}",
            "description": "mô tả",
            "created": created,
            "updated": created,
            "status": {"name": "Closed"},
            "resolution": {"name": resolution} if resolution else None,
            "issuelinks": list(links),
        },
    }


def _link(kind, outward=None, inward=None):
    link = {"type": {"name": kind}}
    if outward:
        link["outwardIssue"] = {"key": outward}
    if inward:
        link["inwardIssue"] = {"key": inward}
    return link


def _fixture():
    issues = [
        _issue("P-1", "2020-01-01T00:00:00.000+0000", [_link("Duplicate", outward="P-2"), _link("Relates", outward="P-3")]),
        _issue("P-2", "2020-02-01T00:00:00.000+0000", [_link("Duplicate", inward="P-1")], resolution="Duplicate"),
        _issue("P-3", "2020-03-01T00:00:00.000+0000", [_link("Blocks", outward="X-9")]),
    ] + [_issue(f"P-{i}", f"2020-04-{i:02d}T00:00:00.000+0000") for i in range(4, 20)]
    return export_issues(issues, tenant="P")


def test_export_keeps_internal_links_and_flags_label_edges():
    docs, edges = _fixture()
    assert len(docs) == 19 and docs["issued_at"].notna().all()
    assert set(edges["dst"]) <= set(docs["doc_id"])
    assert edges.loc[edges["type"] == "Duplicate", "is_label_edge"].all()
    assert not edges.loc[edges["type"] == "Relates", "is_label_edge"].any()
    assert len(edges[edges["type"] == "Duplicate"]) == 1


def test_pairs_have_no_duplicates_self_loops_or_linked_negatives():
    docs, edges = _fixture()
    neighbors = pd.DataFrame({"doc_id": ["P-1", "P-3", "P-4"], "neighbor_id": ["P-3", "P-5", "P-6"], "s": [0.9, 0.8, 0.7], "rank": [1, 1, 1]})
    pairs = build_pairs(docs, edges, neighbors, negatives_per_positive=3, seed=0)
    keys = [frozenset(k) for k in zip(pairs["doc_a"], pairs["doc_b"])]
    assert len(keys) == len(set(keys))
    assert (pairs["doc_a"] != pairs["doc_b"]).all()
    linked = {frozenset(k) for k in zip(edges["src"], edges["dst"])}
    negatives = pairs[pairs["label"] == 0]
    assert not any(frozenset(k) in linked for k in zip(negatives["doc_a"], negatives["doc_b"]))
    assert pairs["label"].sum() == 1 and len(negatives) == 3
