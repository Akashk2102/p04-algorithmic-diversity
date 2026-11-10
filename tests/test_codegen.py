from src.generation import build_prompt, _extract_code
from src.executor import evaluate_code


def test_prompt_contains_demo_and_target():
    p = build_prompt({"input": "target"}, [{"input": "demo", "solution": "print(1)"}])
    assert "demo" in p and "target" in p and "print(1)" in p


def test_executor_passes_case():
    result = evaluate_code("print(input())", [{"input": "hello", "output": "hello"}])
    assert result["status"] == "passed"
    assert result["pass_rate"] == 1.0


def test_extract_code_fences():
    assert _extract_code("```python\nprint(1)\n```") == "print(1)"
    assert _extract_code("```python3\nprint(2)\n```") == "print(2)"
    assert _extract_code("```py\nprint(3)\n```") == "print(3)"
    assert _extract_code("```\nprint(4)\n```") == "print(4)"
    assert _extract_code("print(5)") == "print(5)"
