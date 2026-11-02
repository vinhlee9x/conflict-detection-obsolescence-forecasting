import numpy as np
import pandas as pd

from src.calibrate import calibrate, grid_search, select_best
from src.conflict import fuse, normalize_similarity
from src.graph import KnowledgeGraph
from src.splits import assign_group_splits


def _path_graph(extra=None):
    edges = pd.DataFrame({"src": ["A", "B"], "dst": ["B", "C"], "type": ["REFERENCES", "REFERENCES"]})
    if extra is not None:
        edges = pd.concat([edges, extra], ignore_index=True)
    return KnowledgeGraph(["A", "B", "C", "D"], edges)


def test_multi_hop_reaches_two_hops_only_when_k_allows():
    graph = _path_graph()
    assert graph.pair_scores(["A"], ["C"], k_hops=1)[0] == 0.0
    assert graph.pair_scores(["A"], ["C"], k_hops=2)[0] > 0.0
    assert graph.pair_scores(["A"], ["D"], k_hops=3)[0] == 0.0


def test_propagation_is_bounded_and_symmetric():
    graph = _path_graph()
    scores = graph.pair_scores(["A", "B", "C"], ["B", "C", "A"])
    assert np.all((scores >= 0) & (scores <= 1))
    assert np.allclose(graph.pair_scores(["A"], ["C"]), graph.pair_scores(["C"], ["A"]))


def test_label_edges_never_reach_p():
    with_label = _path_graph(pd.DataFrame({"src": ["A"], "dst": ["D"], "type": ["CONFLICTS_WITH"]}))
    jira_style = _path_graph(pd.DataFrame({"src": ["A"], "dst": ["D"], "type": ["Duplicate"], "is_label_edge": [True]}))
    plain = _path_graph()
    for graph in (with_label, jira_style):
        assert graph.pair_scores(["A"], ["D"])[0] == 0.0
        assert np.allclose(graph.pair_scores(["A"], ["C"]), plain.pair_scores(["A"], ["C"]))


def test_fusion_endpoints():
    s_norm, p = np.array([0.2, 0.9]), np.array([0.7, 0.1])
    assert np.allclose(fuse(s_norm, p, 1.0), s_norm)
    assert np.allclose(fuse(s_norm, p, 0.0), p)
    assert np.allclose(normalize_similarity(np.array([0.5, 0.75, 1.2]), 0.5, 1.0), [0.0, 0.5, 1.0])


def test_grid_search_recovers_separating_threshold():
    s_norm = np.array([0.9, 0.8, 0.85, 0.2, 0.3, 0.75])
    p = np.zeros(6)
    labels = np.array([1, 1, 1, 0, 0, 0])
    best = select_best(grid_search(s_norm, p, labels, [1.0], np.round(np.arange(0.3, 0.95, 0.01), 2)))
    assert best["f1"] == 1.0 and 0.75 <= best["tau"] < 0.8


def test_calibrate_prefers_fusion_when_both_signals_matter():
    rng = np.random.default_rng(0)
    n = 400
    labels = rng.integers(0, 2, n)
    s = 0.5 + 0.2 * labels + rng.normal(0, 0.08, n)
    p = 0.3 * labels + rng.normal(0, 0.08, n)
    config, _ = calibrate(pd.DataFrame({"s": s, "p": p, "label": labels}), np.round(np.arange(0, 1.01, 0.05), 2), np.round(np.arange(0.1, 0.95, 0.01), 2))
    assert 0.0 < config["FUSION_LAMBDA"] < 1.0


def test_group_split_never_shares_a_root():
    pairs = pd.DataFrame({
        "tenant": ["t"] * 40,
        "group_a": [f"g{i % 10}" for i in range(40)],
        "group_b": [f"g{(i * 3) % 10}" for i in range(40)],
        "label": [i % 2 for i in range(40)],
    })
    split = assign_group_splits(pairs, test_fraction=0.3, seed=1)
    dev_groups = set(split.loc[split["split"] == "dev", ["group_a", "group_b"]].stack())
    test_groups = set(split.loc[split["split"] == "test", ["group_a", "group_b"]].stack())
    assert dev_groups and test_groups and not dev_groups & test_groups
