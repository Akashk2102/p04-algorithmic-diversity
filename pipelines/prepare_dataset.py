from __future__ import annotations
import argparse
from pathlib import Path
from datasets import load_dataset
from src.data import normalize_record, save_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nvidia/OpenCodeReasoning")
    ap.add_argument("--split", default="split_0")
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--output", default="data/opencode_sample.jsonl")
    args = ap.parse_args()
    ds = load_dataset(args.dataset, "split_0", split=args.split)
    rows = [normalize_record(dict(r), i) for i, r in enumerate(ds.select(range(min(args.limit, len(ds)))))]
    save_jsonl(rows, args.output)
    print(f"Saved {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()
