"""
config.py
---------
Central configuration for the Vectorless RAG pipeline.

Initialises the single LLM client that every module imports from here:
  - llm_client → OpenAI SDK pointed at the LOCAL Ollama endpoint

No cloud clients needed — PDF parsing is fully local via PyMuPDF.

Design decisions:
  * We use api_key="ollama" as a dummy value for the OpenAI constructor;
    Ollama ignores the key entirely but the SDK requires it to be non-empty.
  * Temperature is NOT set here — callers choose it per-task
    (0 for tree search / node selection, 0.2 for answer generation).
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# ── Load .env file ────────────────────────────────────────────────────────────
load_dotenv()

# ── Ollama / Mistral ──────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434/v1"
)

# Model tag must match what you pulled: `ollama pull mistral`
OLLAMA_MODEL: str = "mistral"

# Single shared OpenAI SDK client pointing to Ollama's OpenAI-compatible API.
# api_key is mandatory in the SDK constructor but Ollama ignores its value.
llm_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",           # dummy — Ollama doesn't authenticate
)
