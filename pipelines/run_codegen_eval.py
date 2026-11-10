from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import yaml
import pandas as pd
from src.data import load_jsonl, normalize_record, group_split
from src.features import FeatureStore, jaccard
from src.generation import build_prompt, generate
from src.executor import evaluate_code


def load_cases(path: str) -> dict[str, list[dict]]:
    out = {}
    for row in load_jsonl(path):
        out[str(row["problem_id"])] = row.get("cases", [])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    rows = [normalize_record(r, i) for i, r in enumerate(load_jsonl(cfg["data"]["path"], cfg["data"].get("limit")))]
    train, _, test = group_split(rows, cfg["evaluation"]["test_size"], cfg["evaluation"]["dev_size"], cfg["retrieval"]["seed"])
    features = FeatureStore(train)
    cases = load_cases(cfg["evaluation"]["cases_path"])

    # Filter test problems that have execution cases before limiting
    eligible_test = [q for q in test if q["problem_id"] in cases]
    limit = cfg["evaluation"].get("max_queries")
    eval_queries = eligible_test[:limit] if limit else eligible_test

    if not eval_queries:
        print(f"No test problems found that have matching execution cases in {cfg['evaluation']['cases_path']}.")
        print("Tip: increase data.limit or check data/execution_cases.jsonl.")
        return

    print(f"Loaded {len(rows)} records ({len(train)} train, {len(test)} test).")
    print(f"Evaluating {len(eval_queries)} queries with {len(cases)} available execution test suites.\n")

    results = []
    pool = list(range(len(train)))

    for qi, q in enumerate(eval_queries):
        rel = features.semantic_scores(q["input"])
        qtags = features.query_tags(q["input"])

        def choose(method):
            if method == "random":
                return random.Random(cfg["retrieval"]["seed"] + qi).sample(pool, min(cfg["retrieval"]["k"], len(pool)))
            if method == "semantic":
                return sorted(pool, key=lambda i: rel[i], reverse=True)[: cfg["retrieval"]["k"]]
            selected = []
            remaining = pool.copy()
            while remaining and len(selected) < cfg["retrieval"]["k"]:
                def score(i):
                    relevance = rel[i]
                    if method == "algorithm_aware":
                        relevance = 0.75 * rel[i] + 0.25 * jaccard(features.tags[i], qtags)
                        redundancy = sum(features.algorithm_similarity(i, j) for j in selected)
                    else:  # mmr
                        redundancy = max((features.algorithm_similarity(i, j) for j in selected), default=0.0)
                    return relevance - cfg["retrieval"]["lambda_diversity"] * redundancy
                best = max(remaining, key=score)
                selected.append(best)
                remaining.remove(best)
            return selected

        print(f"[{qi+1}/{len(eval_queries)}] Problem ID: {q['problem_id'][:12]}...")

        for method in ("random", "semantic", "mmr", "algorithm_aware"):
            selected = choose(method)
            demos = [train[i] for i in selected]
            prompt = build_prompt(q, demos)
            start = time.perf_counter()
            try:
                code = generate(prompt, cfg["generation"])
                generation_status = "ok"
            except Exception as exc:
                code = ""
                generation_status = f"generation_error:{type(exc).__name__}"
                print(f"  [{method}] Generation error: {type(exc).__name__}: {exc}")

            generation_ms = (time.perf_counter() - start) * 1000

            if code:
                execution = evaluate_code(code, cases[q["problem_id"]], float(cfg["evaluation"].get("timeout_seconds", 3)))
                print(f"  [{method}] {execution['status']} ({execution.get('passed',0)}/{execution.get('total',0)} passed) in {generation_ms:.0f}ms")
            else:
                execution = {"status": "generation_error", "passed": 0, "total": len(cases[q["problem_id"]]), "pass_rate": 0.0}

            results.append({
                "query_id": q["id"],
                "problem_id": q["problem_id"],
                "method": method,
                "generation_ms": generation_ms,
                "generation_status": generation_status,
                "execution_status": execution["status"],
                "passed": execution.get("passed", 0),
                "total": execution.get("total", 0),
                "execution_accuracy": execution.get("pass_rate", 0.0),
                "generated_code": code,
            })

    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "codegen_results.jsonl").write_text("\n".join(json.dumps(r) for r in results), encoding="utf-8")

    df = pd.DataFrame(results)
    if not df.empty:
        df.drop(columns=["generated_code"]).to_csv(out / "codegen_results.csv", index=False)
        summary = df.groupby("method", as_index=False).agg(
            execution_accuracy=("execution_accuracy", "mean"),
            generation_ms=("generation_ms", "mean"),
        )
        summary.to_csv(out / "execution_accuracy_table.csv", index=False)
        taxonomy = pd.crosstab(df["method"], df["execution_status"])
        taxonomy.to_csv(out / "error_taxonomy.csv")
        print("\nSummary Results:")
        print(summary.to_string(index=False))
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
