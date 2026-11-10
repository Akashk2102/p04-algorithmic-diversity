from __future__ import annotations
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _run_once(code: str, stdin_text: str, timeout_s: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="p04_exec_") as td:
        script = Path(td) / "solution.py"
        script.write_text(code, encoding="utf-8")
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                input=stdin_text,
                text=True,
                capture_output=True,
                timeout=timeout_s,
                cwd=td,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return {"status": "time_limit_exceeded", "stdout": "", "stderr": "", "runtime_ms": (time.perf_counter()-start)*1000}
        elapsed = (time.perf_counter() - start) * 1000
        if proc.returncode != 0:
            status = "syntax_or_format_error" if "SyntaxError" in proc.stderr else "runtime_error"
            return {"status": status, "stdout": proc.stdout, "stderr": proc.stderr, "runtime_ms": elapsed}
        return {"status": "ok", "stdout": proc.stdout, "stderr": proc.stderr, "runtime_ms": elapsed}


def _normalize_output(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def evaluate_code(code: str, cases: list[dict], timeout_s: float = 3.0) -> dict:
    if not cases:
        return {"status": "no_test_cases", "passed": 0, "total": 0, "pass_rate": None}
    passed = 0
    statuses = []
    for case in cases:
        result = _run_once(code, str(case.get("input", "")), timeout_s)
        if result["status"] != "ok":
            statuses.append(result["status"])
            continue
        actual = _normalize_output(result["stdout"])
        expected = _normalize_output(str(case.get("output", "")))
        if actual == expected:
            passed += 1
            statuses.append("passed")
        else:
            statuses.append("wrong_answer")
    if passed == len(cases):
        status = "passed"
    elif any(s == "time_limit_exceeded" for s in statuses):
        status = "time_limit_exceeded"
    elif any(s == "runtime_error" for s in statuses):
        status = "runtime_error"
    elif any(s == "syntax_or_format_error" for s in statuses):
        status = "syntax_or_format_error"
    else:
        status = "wrong_answer"
    return {"status": status, "passed": passed, "total": len(cases), "pass_rate": passed / len(cases), "case_statuses": statuses}
