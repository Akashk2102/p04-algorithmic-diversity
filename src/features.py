from __future__ import annotations
import re
from collections import Counter
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

ALGORITHM_TAGS = {
    "hash_map": [r"\b(dict|defaultdict|Counter)\b", r"\.get\("],
    "two_pointer": [r"two.?pointer", r"left\s*=", r"right\s*="],
    "sliding_window": [r"sliding.?window", r"window", r"left\s*=", r"right\s*="],
    "binary_search": [r"binary.?search", r"bisect", r"\bmid\b", r"\b(lo|hi)\b"],
    "sorting": [r"\.sort\(", r"sorted\(", r"sort\("],
    "heap": [r"heapq", r"heappush", r"heappop", r"priority.?queue"],
    "stack": [r"stack", r"\.append\(", r"\.pop\("],
    "queue_bfs": [r"deque", r"bfs", r"queue"],
    "dfs": [r"\bdfs\b", r"recurs", r"visited"],
    "dynamic_programming": [r"dynamic.?program", r"\bdp\b", r"memo", r"lru_cache"],
    "greedy": [r"greedy", r"sort.*then", r"choose.*maximum", r"choose.*minimum"],
    "graph": [r"graph", r"adj", r"edges", r"vertices"],
    "union_find": [r"union.?find", r"disjoint", r"parent\["],
    "prefix_sum": [r"prefix.?sum", r"cumulative", r"accum"],
    "math": [r"gcd", r"lcm", r"sqrt", r"math\.", r"modulo", r"%"],
    "string": [r"string", r"\.split\(", r"\.find\(", r"\.replace\("],
}


def algorithm_tags(problem: str, code: str) -> set[str]:
    text = (problem + "\n" + code).lower()
    tags = set()
    for tag, patterns in ALGORITHM_TAGS.items():
        if any(re.search(p, text) for p in patterns):
            tags.add(tag)
    # Add lightweight structural tags to avoid empty representations.
    if "for " in code or re.search(r"\bwhile\b", code):
        tags.add("iteration")
    if re.search(r"\bdef\b|\bfunction\b", code):
        tags.add("function")
    if re.search(r"\bif\b", code):
        tags.add("branching")
    if re.search(r"\breturn\b", code):
        tags.add("return")
    return tags


def tag_vector(tags: set[str], vocabulary: list[str]) -> np.ndarray:
    return np.array([1.0 if t in tags else 0.0 for t in vocabulary], dtype=float)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


class FeatureStore:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        corpus = [r["input"] + "\n" + r["solution"] for r in rows]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=30000, ngram_range=(1, 2))
        self.X = normalize(self.vectorizer.fit_transform(corpus))
        self.tags = [algorithm_tags(r["input"], r["solution"]) for r in rows]
        vocab = sorted(set().union(*self.tags)) if self.tags else []
        self.tag_vocab = vocab
        self.T = normalize(np.vstack([tag_vector(t, vocab) for t in self.tags])) if vocab else np.zeros((len(rows), 0))

    def query_vector(self, problem: str):
        return normalize(self.vectorizer.transform([problem]))

    def semantic_scores(self, problem: str) -> np.ndarray:
        return cosine_similarity(self.query_vector(problem), self.X)[0]

    def query_tags(self, problem: str) -> set[str]:
        return algorithm_tags(problem, "")

    def algorithm_similarity(self, i: int, j: int) -> float:
        return jaccard(self.tags[i], self.tags[j])
