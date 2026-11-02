import numpy as np
import pandas as pd

from src.forecast_features import build_forecast_table, temporal_split
from src.similarity import DocumentVectors

CONFIG = {"tenants": {"t": {"FUSION_LAMBDA": 0.5, "CONFLICT_THRESHOLD_TAU": 0.5, "s_min": 0.0, "s_max": 1.0}}}


def _doc(doc_id, issued, variant="root", superseded=None, department="D"):
    return {
        "doc_id": doc_id, "tenant": "t", "department": department, "topic": "x", "subject": doc_id,
        "variant_type": variant, "issued_at": pd.Timestamp(issued),
        "superseded_at": pd.Timestamp(superseded) if superseded else pd.NaT,
    }


def _world(include_future: bool):
    docs = [
        _doc("A", "2022-01-01", superseded="2022-06-15"),
        _doc("B", "2022-01-10"),
        _doc("H", "2022-03-01", variant="hidden_conflict", department="E"),
    ]
    edges = [("B", "A", "REFERENCES")]
    events = [("A", "H", "2022-03-20")]
    if include_future:
        docs.append(_doc("A2", "2022-06-15", variant="conflict"))
        edges += [("A2", "A", "REFERENCES"), ("A2", "A", "SAME_SERIES")]
        events.append(("A", "A2", "2022-06-15"))
    docs = pd.DataFrame(docs)
    edges = pd.DataFrame(edges, columns=["src", "dst", "type"])
    events = pd.DataFrame(events, columns=["doc_a", "doc_b", "confirmed_at"]).assign(confirmed_at=lambda e: pd.to_datetime(e["confirmed_at"]))
    ids = docs["doc_id"].to_numpy()
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(len(ids), 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return docs, edges, events, DocumentVectors(vectors[:3] if not include_future else vectors, ids)


def test_features_ignore_everything_after_the_snapshot():
    snapshot = [pd.Timestamp("2022-04-01")]
    past = build_forecast_table(*_world(False), CONFIG, snapshot, horizon_days=90)
    full = build_forecast_table(*_world(True), CONFIG, snapshot, horizon_days=90)
    features = [c for c in past.columns if c != "label"]
    pd.testing.assert_frame_equal(past[features], full[features])


def test_label_covers_only_the_horizon_window():
    docs, edges, events, vectors = _world(True)
    table = build_forecast_table(docs, edges, events, vectors, CONFIG, [pd.Timestamp("2022-02-01"), pd.Timestamp("2022-04-01")], horizon_days=90)
    label = table.set_index(["doc_id", "snapshot"])["label"]
    assert label[("A", pd.Timestamp("2022-02-01"))] == 0
    assert label[("A", pd.Timestamp("2022-04-01"))] == 1
    assert set(table["doc_id"]) <= {"A", "B"}


def test_confirmed_conflicts_count_only_after_confirmation():
    docs, edges, events, vectors = _world(False)
    table = build_forecast_table(docs, edges, events, vectors, CONFIG, [pd.Timestamp("2022-03-10"), pd.Timestamp("2022-04-01")], horizon_days=90)
    a = table[table["doc_id"] == "A"].set_index("snapshot")["n_confirmed_conflicts"]
    assert a[pd.Timestamp("2022-03-10")] == 0 and a[pd.Timestamp("2022-04-01")] == 1


def test_temporal_split_orders_snapshots():
    table = pd.DataFrame({"snapshot": pd.date_range("2022-01-01", periods=10, freq="MS").repeat(2)})
    split = temporal_split(table, n_valid=2, n_test=3)
    assert table.loc[split == "train", "snapshot"].max() < table.loc[split == "valid", "snapshot"].min()
    assert table.loc[split == "valid", "snapshot"].max() < table.loc[split == "test", "snapshot"].min()
