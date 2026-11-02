"""Command-line entry points shared by the Colab notebooks and local runs.

    python -m src.pipeline kb-generate
    python -m src.pipeline embed --dataset kb --backend bge-m3
    python -m src.pipeline layer2 --dataset kb
    python -m src.pipeline vinli --backend bge-m3
    python -m src.pipeline layer3 --dataset kb
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import settings
from src.calibrate import calibrate
from src.conflict import fuse, normalize_similarity
from src.evaluate import binary_metrics
from src.graph import KnowledgeGraph
from src.ingest import build_chunks, embed_chunks, get_embedder
from src.io_utils import read_npz, read_parquet, write_json, write_npz, write_parquet
from src.paths import ARTIFACTS, DATA_PROCESSED
from src.similarity import DocumentVectors, top_k_neighbors
from src.splits import assign_group_splits, rebalance_negatives, top_up_random_negatives


def artifact_dir(dataset: str):
    path = ARTIFACTS / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_corpus(dataset: str) -> dict[str, pd.DataFrame]:
    return {
        "docs": read_parquet(DATA_PROCESSED / f"{dataset}_docs.parquet"),
        "pairs": read_parquet(DATA_PROCESSED / f"{dataset}_pairs.parquet"),
        "edges": read_parquet(DATA_PROCESSED / f"{dataset}_edges.parquet"),
    }


def load_vectors(dataset: str) -> DocumentVectors:
    return DocumentVectors.from_npz(read_npz(artifact_dir(dataset) / "embeddings.npz"))


def generate_kb() -> None:
    from src.synth.generate_kb import generate_kb as generate

    for name, frame in generate(seed=settings.SEED).items():
        write_parquet(frame, DATA_PROCESSED / f"kb_{name}.parquet")


def embed(dataset: str, backend: str) -> dict:
    docs = read_parquet(DATA_PROCESSED / f"{dataset}_docs.parquet")
    chunks = build_chunks(docs)
    result = embed_chunks(chunks, get_embedder(backend))
    out = artifact_dir(dataset)
    write_npz(out / "embeddings.npz", **result)
    write_parquet(chunks, out / "chunks.parquet")
    vectors = DocumentVectors(result["vectors"], result["doc_ids"])
    neighbors = top_k_neighbors(vectors, docs.set_index("doc_id")["tenant"], settings.TOP_K_NEIGHBORS)
    write_parquet(neighbors, out / "neighbors.parquet")
    summary = {"backend": backend, "documents": len(docs), "chunks": len(chunks), "seconds": float(result["seconds"])}
    write_json(summary, out / "embedding_summary.json")
    return summary


def score_layer2_inputs(dataset: str) -> pd.DataFrame:
    corpus = load_corpus(dataset)
    vectors = load_vectors(dataset)
    pairs = assign_group_splits(corpus["pairs"], settings.TEST_FRACTION, settings.SEED)
    pairs = top_up_random_negatives(pairs, corpus["docs"], settings.NEGATIVES_PER_POSITIVE, settings.SEED)
    pairs = rebalance_negatives(pairs, settings.NEGATIVES_PER_POSITIVE, settings.SEED)
    pairs["s"] = vectors.pair_similarities(pairs["doc_a"].tolist(), pairs["doc_b"].tolist())
    graph = KnowledgeGraph(corpus["docs"]["doc_id"].tolist(), corpus["edges"])
    pairs["p"] = graph.pair_scores(pairs["doc_a"].tolist(), pairs["doc_b"].tolist())
    return pairs


def layer2(dataset: str, n_bootstrap: int = 1000) -> dict:
    pairs = score_layer2_inputs(dataset)
    dev = pairs[pairs["split"] == "dev"]

    configs, grids = {}, []
    for tenant, frame in dev.groupby("tenant"):
        configs[tenant], grid = calibrate(frame, settings.LAMBDA_GRID, settings.TAU_GRID)
        grids.append(grid.assign(tenant=tenant))
    global_config, global_grid = calibrate(dev, settings.LAMBDA_GRID, settings.TAU_GRID)
    grids.append(global_grid.assign(tenant="all"))
    config = {"tenants": configs, "global": global_config, "k_hops": settings.K_HOPS, "gamma": settings.GAMMA}

    pairs = _attach_scores(pairs, config)
    test = pairs[pairs["split"] == "test"]
    results = {
        "dataset": dataset,
        "config": config,
        "counts": _counts(pairs),
        "test": _ablation_table(test, config, n_bootstrap),
        "test_by_variant": _variant_table(test),
    }

    out = artifact_dir(dataset)
    write_json(config, out / "config.json")
    pd.concat(grids, ignore_index=True).to_csv(out / "grid_results.csv", index=False)
    write_parquet(pairs, out / "scored_pairs.parquet")
    write_json(results, out / "layer2_results.json")
    return results


def _attach_scores(pairs: pd.DataFrame, config: dict) -> pd.DataFrame:
    pairs = pairs.copy()
    for column in ("s_norm", "score_s_only", "score_p_only", "score_fusion", "score_global", "tau_s_only", "tau_p_only", "tau_fusion", "tau_global"):
        pairs[column] = np.nan
    g = config["global"]
    for tenant, index in pairs.groupby("tenant").groups.items():
        cfg = config["tenants"][tenant]
        rows = pairs.loc[index]
        s_norm = normalize_similarity(rows["s"], cfg["s_min"], cfg["s_max"])
        pairs.loc[index, "s_norm"] = s_norm
        pairs.loc[index, "score_s_only"] = s_norm
        pairs.loc[index, "score_p_only"] = rows["p"]
        pairs.loc[index, "score_fusion"] = fuse(s_norm, rows["p"], cfg["FUSION_LAMBDA"])
        pairs.loc[index, "score_global"] = fuse(normalize_similarity(rows["s"], g["s_min"], g["s_max"]), rows["p"], g["FUSION_LAMBDA"])
        pairs.loc[index, "tau_s_only"] = cfg["ablation_tau"]["s_only"]
        pairs.loc[index, "tau_p_only"] = cfg["ablation_tau"]["p_only"]
        pairs.loc[index, "tau_fusion"] = cfg["CONFLICT_THRESHOLD_TAU"]
        pairs.loc[index, "tau_global"] = g["CONFLICT_THRESHOLD_TAU"]
    for method in ("s_only", "p_only", "fusion", "global"):
        pairs[f"pred_{method}"] = (pairs[f"score_{method}"] > pairs[f"tau_{method}"]).astype(int)
    return pairs


METHODS = {
    "s_only": "Chỉ s (λ = 1)",
    "p_only": "Chỉ p (λ = 0)",
    "fusion": "Fusion, λ và τ theo tenant",
    "global": "Fusion, λ và τ chung",
}


def _ablation_table(test: pd.DataFrame, config: dict, n_bootstrap: int) -> list[dict]:
    rows = []
    rng = np.random.default_rng(settings.SEED)
    for scope, frame in [("all", test)] + list(test.groupby("tenant")):
        for method, label in METHODS.items():
            metrics = _metrics_with_thresholds(frame, method)
            metrics["f1_ci95"] = _bootstrap_f1(frame, method, n_bootstrap, rng)
            if scope != "all":
                cfg = config["global"] if method == "global" else config["tenants"][scope]
                metrics["lambda"] = 1.0 if method == "s_only" else 0.0 if method == "p_only" else cfg["FUSION_LAMBDA"]
                metrics["tau"] = float(frame[f"tau_{method}"].iloc[0])
            rows.append({"scope": scope, "method": method, "method_label": label, **metrics})
    return rows


def _metrics_with_thresholds(frame: pd.DataFrame, method: str) -> dict:
    labels = frame["label"].to_numpy()
    margin = frame[f"score_{method}"].to_numpy() - frame[f"tau_{method}"].to_numpy()
    return binary_metrics(labels, margin, threshold=0.0) | {
        "pr_auc": binary_metrics(labels, frame[f"score_{method}"].to_numpy(), 0.5)["pr_auc"]
    }


def _bootstrap_f1(frame: pd.DataFrame, method: str, n: int, rng) -> list[float]:
    labels = frame["label"].to_numpy()
    predictions = frame[f"pred_{method}"].to_numpy()
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(frame), len(frame))
        y, yhat = labels[idx], predictions[idx]
        tp = int(((y == 1) & (yhat == 1)).sum())
        denominator = 2 * tp + int(((y == 0) & (yhat == 1)).sum()) + int(((y == 1) & (yhat == 0)).sum())
        scores.append(2 * tp / denominator if denominator else 0.0)
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def _variant_table(test: pd.DataFrame) -> list[dict]:
    rows = []
    for variant, frame in test.groupby("variant_type"):
        row = {"variant_type": variant, "label": int(frame["label"].iloc[0]), "n": int(len(frame))}
        for method in METHODS:
            row[f"flagged_{method}"] = float(frame[f"pred_{method}"].mean())
        row["mean_s"] = float(frame["s"].mean())
        row["mean_p"] = float(frame["p"].mean())
        rows.append(row)
    return rows


def _counts(pairs: pd.DataFrame) -> list[dict]:
    grouped = pairs.groupby(["tenant", "split"])["label"].agg(n="size", positives="sum").reset_index()
    return grouped.to_dict(orient="records")


def vinli(backend: str, per_label: int = 1000, split: str = "test") -> dict:
    from src.datasets.vinli import load_vinli, stratified_sample

    sample = stratified_sample(load_vinli(split), per_label, settings.SEED)
    write_parquet(sample, DATA_PROCESSED / "vinli.parquet")
    embedder = get_embedder(backend)
    started = time.perf_counter()
    vectors = embedder.encode(sample["premise"].tolist() + sample["hypothesis"].tolist())
    seconds = time.perf_counter() - started
    premise, hypothesis = vectors[: len(sample)], vectors[len(sample):]
    sample["cosine"] = np.sum(premise * hypothesis, axis=1)

    out = artifact_dir("vinli")
    write_parquet(sample, out / "vinli_cosine.parquet")
    summary = {
        "backend": backend,
        "split": split,
        "n": int(len(sample)),
        "seconds": seconds,
        "mean_cosine": sample.groupby("label")["cosine"].mean().to_dict(),
        "auc_entailment_vs_contradiction": _auc(sample, positive="entailment", negative="contradiction"),
        "auc_contradiction_vs_neutral": _auc(sample, positive="contradiction", negative="neutral"),
    }
    write_json(summary, out / "vinli_summary.json")
    return summary


def _auc(sample: pd.DataFrame, positive: str, negative: str) -> float:
    frame = sample[sample["label"].isin([positive, negative])]
    return float(roc_auc_score(frame["label"] == positive, frame["cosine"]))


def layer3(dataset: str, teacher: str = "tabpfn") -> dict:
    from src.forecast import run_forecast

    return run_forecast(dataset, teacher=teacher)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["kb-generate", "embed", "layer2", "vinli", "layer3"])
    parser.add_argument("--dataset", default="kb")
    parser.add_argument("--backend", default="bge-m3", choices=["bge-m3", "tfidf"])
    parser.add_argument("--teacher", default="tabpfn", choices=["tabpfn", "gbm"])
    args = parser.parse_args()

    if args.command == "kb-generate":
        generate_kb()
    elif args.command == "embed":
        print(embed(args.dataset, args.backend))
    elif args.command == "layer2":
        results = layer2(args.dataset)
        print(pd.DataFrame(results["test"])[["scope", "method", "precision", "recall", "f1", "pr_auc"]].to_string(index=False))
    elif args.command == "vinli":
        print(vinli(args.backend))
    elif args.command == "layer3":
        print(layer3(args.dataset, args.teacher))


if __name__ == "__main__":
    main()
