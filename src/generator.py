"""
generator.py
------------
Local answer generation using Mistral 7B via Ollama.

Receives the retrieved document nodes from retriever.py and synthesises a
grounded, cited answer. Uses temperature=0.2 for slight creativity while
still staying factual and close to the source material.

Grounding strategy:
  - Each node is injected as a labelled [SOURCE: title] block so Mistral
    can see exactly where each piece of information came from.
  - The system prompt instructs the model to cite sources by title,
    flag uncertainty, and not hallucinate beyond the provided context.

Functions:
  generate_answer(question, nodes, conversation_history) → str
"""

from src.config import llm_client, OLLAMA_MODEL
from src.retriever import RetrievedNode


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a precise, citation-grounded Q&A assistant.
You will be given excerpts from a document and a user question.

Rules:
- Answer ONLY from the provided source excerpts. Do not rely on outside knowledge.
- Cite the source section title in your answer using the format: [SOURCE: <title>].
- If the provided sources don't contain enough information to answer, say so clearly.
- Be concise but complete. Use bullet points for multi-part answers.
- Never fabricate facts, page numbers, or quotes that are not in the sources.
""".strip()


def _build_context_block(nodes: list[RetrievedNode]) -> str:
    """
    Format retrieved nodes into a labelled context block for the LLM prompt.
    Each node gets a clear SOURCE header so Mistral can reference it.
    """
    if not nodes:
        return "(No relevant document sections were found.)"

    parts: list[str] = []
    for node in nodes:
        header = f"[SOURCE: {node.title}]" if node.title else "[SOURCE: (untitled)]"
        # Truncate very long nodes to ~3000 chars to avoid context overflow
        content = node.content[:3000]
        if len(node.content) > 3000:
            content += "\n… [content truncated]"
        parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(parts)


def generate_answer(
    question: str,
    nodes: list[RetrievedNode],
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Generate a grounded answer using Mistral 7B (local via Ollama).

    Args:
        question:             The user's current question.
        nodes:                Retrieved document nodes from retriever.py.
        conversation_history: Optional list of previous {"role", "content"} turns
                              for multi-turn chat context (without system message).

    Returns:
        The model's answer as a plain string.
    """
    context_block = _build_context_block(nodes)

    user_message = (
        f"Document excerpts:\n\n{context_block}\n\n"
        f"Question: {question}"
    )

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    print("[Generator] Asking Mistral to synthesise an answer …")

    response = llm_client.chat.completions.create(
        model=OLLAMA_MODEL,
        temperature=0.2,
        messages=messages,
    )

    answer: str = response.choices[0].message.content or "(No response generated.)"
    return answer.strip()


def generate_answer_stream(
    question: str,
    nodes: list[RetrievedNode],
    conversation_history: list[dict] | None = None,
):
    """
    Streaming variant of generate_answer.

    Yields text chunks (strings) as Mistral produces them so the UI can
    display tokens word-by-word instead of waiting for the full response.
    This makes the perceived latency much lower even on slow hardware.

    Args: same as generate_answer.
    Yields: str chunks of the answer.
    """
    context_block = _build_context_block(nodes)
    user_message = (
        f"Document excerpts:\n\n{context_block}\n\n"
        f"Question: {question}"
    )

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    print("[Generator] Streaming answer from Mistral …")

    stream = llm_client.chat.completions.create(
        model=OLLAMA_MODEL,
        temperature=0.2,
        messages=messages,
        stream=True,        # ← key change: get chunks as they arrive
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
