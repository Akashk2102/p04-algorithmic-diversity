from __future__ import annotations
import argparse, json, time
from pathlib import Path
import yaml
import pandas as pd
from src.data import load_jsonl, normalize_record, group_split
from src.features import FeatureStore, jaccard

METHODS = ("random", "semantic", "mmr", "algorithm_aware")


def retrieve_all(train, query, features, k, lam, seed):
    import random

    pool = list(range(len(train)))
    rng = random.Random(seed)

    selections = {}
    timings = {}

    # Random
    start = time.perf_counter()
    selections["random"] = rng.sample(pool, min(k, len(pool)))
    timings["random"] = (time.perf_counter() - start) * 1000

    # Semantic
    start = time.perf_counter()
    rel = features.semantic_scores(query["input"])
    selections["semantic"] = sorted(
        pool, key=lambda i: rel[i], reverse=True
    )[:k]
    timings["semantic"] = (time.perf_counter() - start) * 1000

    qtags = features.query_tags(query["input"])

    def greedy(use_tags: bool, use_sum_redundancy: bool):
        remaining = pool.copy()
        selected = []

        while remaining and len(selected) < k:

            def score(i):
                relevance = rel[i]

                if use_tags:
                    relevance = (
                        0.75 * rel[i]
                        + 0.25 * jaccard(features.tags[i], qtags)
                    )

                if use_sum_redundancy:
                    redundancy = sum(
                        features.algorithm_similarity(i, j)
                        for j in selected
                    )
                else:
                    redundancy = max(
                        (
                            features.algorithm_similarity(i, j)
                            for j in selected
                        ),
                        default=0.0
                    )

                return relevance - lam * redundancy

            best = max(remaining, key=score)
            selected.append(best)
            remaining.remove(best)

        return selected

    # MMR (Maximal Marginal Relevance: uses max pairwise redundancy)
    start = time.perf_counter()
    selections["mmr"] = greedy(use_tags=False, use_sum_redundancy=False)
    timings["mmr"] = (time.perf_counter() - start) * 1000

    # Algorithm-aware (Proposed methodology: uses tag relevance + pairwise sum redundancy)
    start = time.perf_counter()
    selections["algorithm_aware"] = greedy(use_tags=True, use_sum_redundancy=True)
    timings["algorithm_aware"] = (time.perf_counter() - start) * 1000

    return selections, timings


def metrics(train, query, features, selected):
    tags = [features.tags[i] for i in selected]
    union = set().union(*tags) if tags else set()
    sims = [jaccard(tags[a], tags[b]) for a in range(len(tags)) for b in range(a + 1, len(tags))]
    return {
        "algorithm_coverage": len(union),
        "query_algorithm_coverage": len(union & features.query_tags(query["input"])),
        "mean_pair_algorithm_similarity": sum(sims) / len(sims) if sims else 0.0,
        "prompt_tokens": sum(len((train[i]["input"] + " " + train[i]["solution"]).split()) for i in selected),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    raw = load_jsonl(cfg["data"]["path"], cfg["data"].get("limit"))
    rows = [normalize_record(r, i) for i, r in enumerate(raw)]

    if len(rows) < 8:
        raise SystemExit("Need at least 8 records.")

    train, dev, test = group_split(
        rows,
        cfg["evaluation"]["test_size"],
        cfg["evaluation"]["dev_size"],
        cfg["retrieval"]["seed"]
    )

    features = FeatureStore(train)
    records = []

    for qi, q in enumerate(test):
        selections, timings = retrieve_all(
            train,
            q,
            features,
            cfg["retrieval"]["k"],
            cfg["retrieval"]["lambda_diversity"],
            cfg["retrieval"]["seed"] + qi
        )

        for method, selected in selections.items():
            records.append({
                "query_id": q["id"],
                "problem_id": q["problem_id"],
                "method": method,
                "k": len(selected),
                "retrieval_ms": timings[method],
                **metrics(train, q, features, selected)
            })

    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(out / "baseline_results.csv", index=False)

    summary = df.groupby("method", as_index=False).mean(numeric_only=True)
    summary.to_csv(out / "baseline_table.csv", index=False)

    # Required ablation: proposed method without algorithm-tag relevance.
    ab = []

    for qi, q in enumerate(test):
        rel = features.semantic_scores(q["input"])
        pool = list(range(len(train)))
        selected = []

        while pool and len(selected) < cfg["retrieval"]["k"]:
            best = max(
                pool,
                key=lambda i: rel[i]
                - cfg["retrieval"]["lambda_diversity"]
                * sum(
                    features.algorithm_similarity(i, j)
                    for j in selected
                )
            )

            selected.append(best)
            pool.remove(best)

        result = metrics(train, q, features, selected)

        ab.append({
            "query_id": q["id"],
            "variant": "diversity_only",
            "algorithm_coverage": result["algorithm_coverage"],
            "query_algorithm_coverage": result["query_algorithm_coverage"],
            "prompt_tokens": result["prompt_tokens"]
        })

    pd.DataFrame(ab).to_csv(
        out / "ablation_table.csv",
        index=False
    )

    # EDA and leakage evidence.
    eda = pd.DataFrame(rows)

    eda_report = {
        "records": len(rows),
        "unique_problem_ids": int(eda.problem_id.nunique()),
        "duplicate_problem_ids": int(eda.problem_id.duplicated().sum()),
        "sources": eda.source.value_counts(dropna=False).to_dict(),
        "difficulties": eda.difficulty.value_counts(dropna=False).to_dict(),
        "avg_input_tokens": float(
            eda.input.str.split().str.len().mean()
        ),
        "avg_solution_tokens": float(
            eda.solution.str.split().str.len().mean()
        ),
        "split_problem_id_overlap": {
            a_b: len(
                set(r["problem_id"] for r in a)
                & set(r["problem_id"] for r in b)
            )
            for a_b, a, b in [
                ("train_test", train, test),
                ("train_dev", train, dev),
                ("dev_test", dev, test)
            ]
        }
    }

    (out / "eda_and_leakage.json").write_text(
        json.dumps(eda_report, indent=2),
        encoding="utf-8"
    )

    print(summary.to_string(index=False))
    print(f"Results: {out}")

if __name__ == "__main__": main()
