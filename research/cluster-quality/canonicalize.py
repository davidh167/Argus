"""Local-LLM canonicalization: ask a small Ollama model to expand
acronyms / normalize each phrase to its standard full-form name, cached to
data/canon_<model>.json. Local only -- no API cost, no DB access."""

import json
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).parent
PROMPT = (
    "Give the standard, fully spelled-out name of this professional skill or technology, "
    "expanding acronyms and abbreviations, in lowercase, with no extra words. "
    "Keep it as specific as the input (do not generalize). "
    'Reply as JSON: {"name": "..."}.\n\nSkill: '
)


def canon(phrase: str, model: str) -> str:
    r = httpx.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "user", "content": PROMPT + phrase}],
        },
        timeout=120,
    )
    r.raise_for_status()
    try:
        return json.loads(r.json()["message"]["content"])["name"].strip().lower() or phrase.lower()
    except (KeyError, ValueError, AttributeError):
        return phrase.lower()


def canon_all(phrases: list[str], model: str) -> dict[str, str]:
    path = HERE / "data" / f"canon_{model.replace(':', '_')}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    for p in phrases:
        if p not in cache:
            cache[p] = canon(p, model)
    path.write_text(json.dumps(cache, indent=1))
    return cache


if __name__ == "__main__":
    rows = json.loads((HERE / "data" / "phrases.json").read_text())
    ph = sorted({s for r in rows if r["interest"] >= 1.0 for s in r["skills"]})
    out = canon_all(ph, sys.argv[1] if len(sys.argv) > 1 else "llama3.2:3b")
    for k in [
        "NLP",
        "LLMs",
        "Go",
        "Golang",
        "RAG pipelines",
        "ASR models",
        "Postgres",
        "NLU",
        "Kubernetes",
        "C++",
    ]:
        print(k, "->", out[k])
