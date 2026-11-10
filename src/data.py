from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_record(row: dict, idx: int) -> dict:
    # OpenCodeReasoning has id/input/solution in the released OCR-1 representation.
    # The local demo also supplies problem_id explicitly.
    problem_id = row.get("problem_id") or row.get("question_id") or row.get("id")
    return {
        "id": str(row.get("id", idx)),
        "problem_id": str(problem_id),
        "input": str(row.get("input", row.get("question", ""))),
        "solution": str(row.get("solution", "")),
        "source": row.get("source", "unknown"),
        "difficulty": row.get("difficulty", "unknown"),
    }


def group_split(rows: list[dict], test_size: float, dev_size: float, seed: int = 42):
    groups = np.array([r["problem_id"] for r in rows])
    idx = np.arange(len(rows))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_dev_idx, test_idx = next(gss.split(idx, groups=groups))
    train_dev = [rows[i] for i in train_dev_idx]
    test = [rows[i] for i in test_idx]
    groups2 = np.array([r["problem_id"] for r in train_dev])
    idx2 = np.arange(len(train_dev))
    # dev_size is interpreted as a fraction of the original data.
    relative_dev = dev_size / max(1e-9, (1 - test_size))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_dev, random_state=seed)
    train_idx, dev_idx = next(gss2.split(idx2, groups=groups2))
    train = [train_dev[i] for i in train_idx]
    dev = [train_dev[i] for i in dev_idx]
    return train, dev, test


def save_jsonl(rows: Iterable[dict], path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
