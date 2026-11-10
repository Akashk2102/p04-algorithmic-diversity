from __future__ import annotations
import random
import numpy as np
from .features import FeatureStore, jaccard


class Retriever:
    def __init__(self, rows: list[dict], k: int = 5, lambda_diversity: float = 0.65, seed: int = 42):
        self.rows = rows
        self.k = k
        self.lam = lambda_diversity
        self.rng = random.Random(seed)
        self.features = FeatureStore(rows)

    def _eligible(self, query_idx: int):
        # Retrieval candidates should not contain the target problem itself.
        target_pid = self.rows[query_idx]["problem_id"]
        return [i for i, r in enumerate(self.rows) if r["problem_id"] != target_pid]

    def random(self, query_idx: int):
        pool = self._eligible(query_idx)
        return self.rng.sample(pool, min(self.k, len(pool)))

    def semantic(self, query_idx: int):
        pool = self._eligible(query_idx)
        scores = self.features.semantic_scores(self.rows[query_idx]["input"])
        return sorted(pool, key=lambda i: scores[i], reverse=True)[: self.k]

    def mmr(self, query_idx: int):
        pool = self._eligible(query_idx)
        rel = self.features.semantic_scores(self.rows[query_idx]["input"])
        selected = []
        while pool and len(selected) < self.k:
            if not selected:
                best = max(pool, key=lambda i: rel[i])
            else:
                best = max(
                    pool,
                    key=lambda i: rel[i] - self.lam * max(
                        self.features.algorithm_similarity(i, j) for j in selected
                    ),
                )
            selected.append(best)
            pool.remove(best)
        return selected

    def algorithm_aware(self, query_idx: int):
        """Greedy approximation of the handbook objective.

        S* = argmax [sum relevance - lambda * sum pairwise algorithm similarity].
        Relevance combines semantic similarity and overlap with the query's algorithm tags.
        """
        pool = self._eligible(query_idx)
        query = self.rows[query_idx]["input"]
        rel = self.features.semantic_scores(query)
        qtags = self.features.query_tags(query)
        selected = []
        while pool and len(selected) < self.k:
            def score(i):
                tag_rel = jaccard(self.features.tags[i], qtags)
                relevance = 0.75 * rel[i] + 0.25 * tag_rel
                redundancy = sum(self.features.algorithm_similarity(i, j) for j in selected)
                return relevance - self.lam * redundancy
            best = max(pool, key=score)
            selected.append(best)
            pool.remove(best)
        return selected

    def retrieve_all(self, query_idx: int):
        return {
            "random": self.random(query_idx),
            "semantic": self.semantic(query_idx),
            "mmr": self.mmr(query_idx),
            "algorithm_aware": self.algorithm_aware(query_idx),
        }
