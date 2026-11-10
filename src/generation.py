from __future__ import annotations
import json
import os
import re
import urllib.request
from typing import Iterable


def build_prompt(query: dict, demonstrations: Iterable[dict]) -> str:
    blocks = [
        "You are a competitive programming code generator.",
        "Solve the target problem in Python 3. Return only executable Python code.",
        "Use the demonstrations as examples, but solve the target problem yourself.",
    ]
    for n, d in enumerate(demonstrations, 1):
        blocks.append(f"\n--- Demonstration {n} ---\nProblem:\n{d['input']}\nSolution:\n{d['solution']}")
    blocks.append(f"\n--- Target Problem ---\n{query['input']}\n\nReturn only Python code.")
    return "\n".join(blocks)


def _extract_code(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:python|python3|py)?\s*\r?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            lines = part.strip().splitlines()
            if lines and lines[0].strip().lower() in {"python", "py", "python3"}:
                return "\n".join(lines[1:]).strip()
        for part in parts:
            if any(kw in part for kw in ("def ", "import ", "from ")):
                return part.strip()
    return text


def generate_openai_compatible(prompt: str, model: str, base_url: str, api_key: str,
                               temperature: float = 0.0, max_tokens: int = 2048,
                               timeout: float = 120.0) -> str:
    """Call an OpenAI-compatible /chat/completions endpoint.

    Works with OpenAI-compatible local (e.g. Ollama, vLLM) or hosted servers (e.g. Gemini, OpenAI, Groq).
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return _extract_code(body["choices"][0]["message"]["content"])


def generate(prompt: str, cfg: dict) -> str:
    provider = cfg.get("provider", "openai_compatible")
    timeout = float(cfg.get("timeout_seconds", 120.0))

    if provider == "gemini":
        base_url = cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai")
        api_key_env = cfg.get("api_key_env", "GEMINI_API_KEY")
        api_key = os.getenv(api_key_env, "")
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} is not set or empty.")
        model = cfg.get("model", "gemini-1.5-flash")
        return generate_openai_compatible(
            prompt, model, base_url, api_key,
            temperature=float(cfg.get("temperature", 0.0)),
            max_tokens=int(cfg.get("max_tokens", 2048)),
            timeout=timeout,
        )

    if provider == "openai_compatible":
        api_key_env = cfg.get("api_key_env", "LLM_API_KEY")
        api_key = os.getenv(api_key_env, "") if api_key_env else ""
        base_url = cfg.get("base_url", os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
        model = cfg.get("model", os.getenv("LLM_MODEL", "qwen2.5-coder:1.5b"))
        return generate_openai_compatible(
            prompt, model, base_url, api_key,
            temperature=float(cfg.get("temperature", 0.0)),
            max_tokens=int(cfg.get("max_tokens", 2048)),
            timeout=timeout,
        )

    raise ValueError(f"Unsupported generation provider: {provider}")
