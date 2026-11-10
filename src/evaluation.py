from __future__ import annotations
import time
import numpy as np
import pandas as pd
from .features import jaccard


def token_count(text: str) -> int:
    return len(text.split())


def set_metrics(rows, features, query_idx, selected):
    qtags = features.query_tags(rows[query_idx]["input"])
    tags = [features.tags[i] for i in selected]
    union = set().union(*tags) if tags else set()
    pair_sims = []
    for a in range(len(selected)):
        for b in range(a + 1, len(selected)):
            pair_sims.append(jaccard(tags[a], tags[b]))
    return {
        "algorithm_coverage": len(union),
        "query_algorithm_coverage": len(union & qtags),
        "mean_pair_algorithm_similarity": float(np.mean(pair_sims)) if pair_sims else 0.0,
        "prompt_tokens": sum(token_count(rows[i]["input"] + "\n" + rows[i]["solution"]) for i in selected),
    }


def evaluate_retrieval(rows, retriever, query_indices):
    records = []
    for qi in query_indices:
        all_methods = retriever.retrieve_all(qi)
        for method, selected in all_methods.items():
            start = time.perf_counter()
            # metrics are intentionally computed after retrieval; timing above is not retrieval timing.
            metrics = set_metrics(rows, retriever.features, qi, selected)
            elapsed_ms = (time.perf_counter() - start) * 1000
            records.append({
                "query_id": rows[qi]["id"],
                "problem_id": rows[qi]["problem_id"],
                "method": method,
                "k": len(selected),
                "retrieval_metric_ms": elapsed_ms,
                **metrics,
            })
    return pd.DataFrame(records)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("method", as_index=False).agg(
        algorithm_coverage=("algorithm_coverage", "mean"),
        query_algorithm_coverage=("query_algorithm_coverage", "mean"),
        mean_pair_algorithm_similarity=("mean_pair_algorithm_similarity", "mean"),
        prompt_tokens=("prompt_tokens", "mean"),
        retrieval_metric_ms=("retrieval_metric_ms", "mean"),
    )
