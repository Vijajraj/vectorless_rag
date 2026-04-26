"""
tree_search.py
--------------
Gives the LLM a "Table of Contents" view of the PageIndex tree and asks it
to pick which node IDs are relevant to the user's question.

This is the key insight of vectorless RAG:
  - Instead of embedding every chunk and doing kNN, we show the LLM the
    *structure* of the document (headings, section titles, node summaries).
  - The LLM acts like a reader skimming a book's TOC to decide which chapters
    to open before reading in detail.
  - Temperature=0 → deterministic, structured JSON output.

Functions:
  flatten_tree_toc(tree) → list[dict]   Build a flat TOC from the nested tree.
  select_relevant_nodes(toc, question)  Ask Mistral which node IDs matter.
"""

import json
import re
from typing import Any
from src.config import llm_client, OLLAMA_MODEL


def flatten_tree_toc(tree: Any, max_nodes: int = 80) -> list[dict]:
    """
    Walk the PageIndex tree depth-first and produce a flat list of nodes,
    each containing just enough info for the LLM to make a routing decision.

    We cap at *max_nodes* to stay within the context window — if a document
    has hundreds of tiny paragraphs we keep the first 80.

    Args:
        tree:      PageIndex tree object (root node with nested children).
        max_nodes: Maximum TOC entries to send to the LLM (default 80).

    Returns:
        List of dicts: [{"id": "...", "depth": 0, "title": "...", "summary": "..."}, ...]
    """
    toc: list[dict] = []

    def _walk(node: Any, depth: int = 0) -> None:
        if len(toc) >= max_nodes:
            return

        # PageIndex tree nodes use 'node_id' as the identifier key.
        # We also check 'id' as a fallback for robustness.
        if isinstance(node, dict):
            node_id = node.get("node_id") or node.get("id") or ""
            title   = node.get("title") or ""
            summary = node.get("summary") or ""
            children = node.get("children") or []
        else:
            node_id  = getattr(node, "node_id", "") or getattr(node, "id", "") or ""
            title    = getattr(node, "title", "") or ""
            summary  = getattr(node, "summary", "") or ""
            children = getattr(node, "children", []) or []

        if node_id:
            toc.append({
                "id":      node_id,
                "depth":   depth,
                "title":   str(title).strip(),
                "summary": str(summary).strip(),
            })

        for child in children:
            _walk(child, depth + 1)

    _walk(tree)
    return toc


# ─── Prompt templates ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a precise document navigation assistant.
Given a document's Table of Contents (TOC) and a user question,
you must return ONLY a JSON array of the most relevant node IDs.

Rules:
- Return ONLY valid JSON: a JSON array of strings, e.g. ["id1", "id2"]
- Include at most 5 node IDs, fewer if fewer are relevant.
- Do NOT include any explanation, markdown, or extra text.
- If nothing is relevant return an empty array: []
""".strip()

_USER_TEMPLATE = """
Document Table of Contents (JSON):
{toc_json}

User Question:
{question}

Which node IDs should I retrieve to answer this question?
Return ONLY a JSON array of node IDs.
""".strip()


def _strip_markdown_fences(text: str) -> str:
    """
    Mistral sometimes wraps its JSON response in ```json ... ``` fences.
    This strips those fences so json.loads() doesn't choke.
    """
    # Remove ```json ... ``` or ``` ... ``` blocks
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
    return text.strip()


def select_relevant_nodes(toc: list[dict], question: str) -> list[str]:
    """
    Ask Mistral (via Ollama) to scan the TOC and return the node IDs that
    are most relevant to answering *question*.

    Uses temperature=0 for deterministic, structured output — we want the LLM
    to behave like a lookup function here, not a creative writer.

    Args:
        toc:      Flat TOC list from flatten_tree_toc().
        question: The user's natural-language question.

    Returns:
        List of node ID strings selected by the LLM (may be empty).
    """
    # Build a compact JSON representation of the TOC for the prompt.
    # We include depth as indentation-style prefix to help the LLM understand
    # the hierarchy without sending full content.
    toc_compact = [
        {
            "id":      entry["id"],
            "level":   entry["depth"],
            "title":   entry["title"],
            "summary": entry["summary"][:120] if entry["summary"] else "",
        }
        for entry in toc
    ]
    toc_json = json.dumps(toc_compact, indent=2)

    user_msg = _USER_TEMPLATE.format(toc_json=toc_json, question=question)

    print("[TreeSearch] Asking LLM to select relevant nodes …")

    response = llm_client.chat.completions.create(
        model=OLLAMA_MODEL,
        temperature=0,           # deterministic — we need parseable JSON
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )

    raw = response.choices[0].message.content or "[]"
    cleaned = _strip_markdown_fences(raw)

    try:
        node_ids: list[str] = json.loads(cleaned)
        if not isinstance(node_ids, list):
            raise ValueError("LLM returned non-list JSON")
        # Filter: only keep IDs that actually exist in our TOC
        valid_ids = {entry["id"] for entry in toc}
        node_ids = [nid for nid in node_ids if nid in valid_ids]
        print(f"[TreeSearch] Selected node IDs: {node_ids}")
        return node_ids
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[TreeSearch] Warning: Could not parse LLM response → {e}")
        print(f"[TreeSearch] Raw response was: {raw!r}")
        return []
