from __future__ import annotations

import json
import re
from pathlib import Path


DATA_PATH = Path("data/opencode_sample.jsonl")
OUTPUT_PATH = Path("data/execution_cases.jsonl")


def clean_block(text: str) -> str:
    """Clean extracted sample input/output while preserving line structure."""
    lines = [line.rstrip() for line in text.splitlines()]

    # Remove empty lines at the beginning/end
    while lines and not lines[0].strip():
        lines.pop(0)

    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines).strip()


def extract_examples(problem_text: str):
    """
    Extract Input/Output examples from a problem statement.

    Looks for sections such as:
        Example
        Input:
        ...
        Output:
        ...

    Also handles examples where Input/Output appear without
    a separate Example heading.
    """

    pattern = re.compile(
        r"(?:Example(?:\s+\d+)?\s*:?\s*)?"
        r"Input\s*:\s*(.*?)"
        r"Output\s*:\s*(.*?)(?="
        r"\n\s*(?:Explanation|Constraints|Input|Output|Example)"
        r"\s*:?\s*|\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    matches = pattern.findall(problem_text)

    cases = []

    for input_text, output_text in matches:
        input_text = clean_block(input_text)
        output_text = clean_block(output_text)

        if not input_text or not output_text:
            continue

        # Ignore obvious placeholder/example labels
        if input_text.lower() in {"...", "your input"}:
            continue

        cases.append({
            "input": input_text,
            "output": output_text,
        })

    return cases


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    records = 0
    extracted = 0
    skipped = 0

    with DATA_PATH.open("r", encoding="utf-8") as infile, \
         OUTPUT_PATH.open("w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            records += 1

            problem_id = str(record.get("problem_id", ""))
            problem_text = record.get("input", "")

            if not problem_id or not problem_text:
                skipped += 1
                continue

            cases = extract_examples(problem_text)

            if not cases:
                skipped += 1
                continue

            outfile.write(
                json.dumps(
                    {
                        "problem_id": problem_id,
                        "cases": cases,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            extracted += 1

    print(f"Records scanned: {records}")
    print(f"Problems with extracted examples: {extracted}")
    print(f"Problems skipped: {skipped}")
    print(f"Execution cases saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()