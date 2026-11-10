# P04 — Algorithmic Diversity Retrieval for Code Generation

This is the **minimal implementation of P04** from the LLM Research Project Handbook.

**Dataset:** NVIDIA OpenCodeReasoning.

## What the handbook asks for

The project compares four retrieval strategies at the same prompt budget:
1. Random retrieval
2. Semantic nearest-neighbour retrieval
3. Maximal Marginal Relevance (MMR)
4. Proposed algorithm-aware retrieval

The proposed method selects examples that are relevant to the target while reducing algorithmic redundancy. The handbook asks for evaluation using execution accuracy, algorithm coverage, prompt tokens, and failure types, plus dataset EDA, leakage controls, baseline/ablation tables, efficiency results, error taxonomy, reproducible code, and limitations.

## Included

- Dataset preparation for NVIDIA OpenCodeReasoning
- Problem-ID grouped train/dev/test split for leakage control
- The four required retrieval methods
- Algorithmic tags and diversity scoring
- Baseline table
- Required ablation table
- EDA + leakage report
- Retrieval efficiency measurement
- Reproducible tests

## Run

```bash
pip install -r requirements.txt
python -m pipelines.prepare_dataset --limit 10000
python -m pipelines.run_experiment --config configs/experiment.yaml
```

For a local check without the NVIDIA dataset:

```bash
python -m pipelines.run_experiment --config configs/demo.yaml
pytest -q
```

### End-to-End Execution Accuracy & Failure Analysis

1. Extract execution cases from the dataset:
   ```bash
   python -m pipelines.create_execution_cases
   ```

2. Run code generation evaluation:
   - **Via Local Ollama** (recommended model for 4GB GPUs like RTX 2050 is `1.5b` or `3b` to avoid CPU offload timeouts):
     ```bash
     ollama run qwen2.5-coder:1.5b
     python -m pipelines.run_codegen_eval --config configs/codegen.yaml
     ```
   - **Via Gemini API** (uses your `GEMINI_API_KEY`):
     ```bash
     python -m pipelines.run_codegen_eval --config configs/codegen_gemini.yaml
     ```

Results are saved to `results/codegen/` (`execution_accuracy_table.csv`, `error_taxonomy.csv`, `codegen_results.csv`).

## Scope note

The handbook does **not** prescribe a particular code-generation model or programming language. Therefore no unrelated UI, web app, database, deployment layer, or extra AI feature is included. End-to-end **execution accuracy and failure-type analysis require selecting a code generator and an executable benchmark setup**; those choices are not specified in the handbook and are intentionally left as the experiment's model/evaluation configuration rather than invented as project requirements.

## Limitations

1. **Rule-based Algorithmic Representations**: Algorithmic diversity is computed over 16 core algorithmic families (dynamic programming, two-pointer, BFS/DFS, binary search, heaps, greedy, prefix sums, etc.) and structural syntactic markers. While interpretable and fast ($O(1)$ tag evaluation), rule-based tagging cannot detect latent or composite algorithmic strategies that deep neural code representations or program dependency graphs might identify.
2. **Dataset Domain Specificity**: NVIDIA OpenCodeReasoning is centered on competitive programming problems in Python 3. Industrial software engineering workloads (e.g., microservices, database transactions, GUI frameworks) exhibit different redundancy patterns not captured by standard algorithmic problem corpora.
3. **Execution Sandbox Constraints**: The execution evaluator tests standard competitive programming I/O with a 3.0-second timeout per test case. Interactive problems requiring bidirectional inter-process communication or problems with multiple valid outputs (e.g., any valid topological order) require problem-specific test harnesses.
4. **Model Scale & Sensitivity**: Smaller locally-hosted models (such as 1.5B parameters) produce lower absolute pass rates (~10%) due to raw reasoning capacity constraints, though relative retrieval gains remain observable. Larger LLMs or commercial APIs provide higher absolute execution pass rates.
