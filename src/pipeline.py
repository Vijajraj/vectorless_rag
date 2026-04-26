"""
pipeline.py
-----------
Orchestrates the full Vectorless RAG pipeline end-to-end:

  1. Receive the locally-parsed tree (from pdf_parser.parse_pdf).
  2. Flatten the tree into a TOC.
  3. LLM scans the TOC and selects relevant node IDs (tree_search).
  4. Walk the tree and retrieve those nodes' full content (retriever).
  5. Local Mistral generates a cited answer from the retrieved context (generator).

The pipeline maintains a conversation_history list so that multi-turn
questions build on one another within the same session.

Usage:
    from src.pdf_parser import parse_pdf
    from src.pipeline import RAGPipeline

    tree = parse_pdf("document.pdf")
    pipeline = RAGPipeline(tree)
    answer = pipeline.query("What are the key findings?")
    answer2 = pipeline.query("Can you elaborate on the third finding?")
"""

from src.tree_search import flatten_tree_toc, select_relevant_nodes
from src.retriever  import retrieve_nodes, RetrievedNode
from src.generator  import generate_answer, generate_answer_stream
from typing import Any


class RAGPipeline:
    """
    Stateful RAG pipeline that holds the document tree and conversation history.

    Keeping the tree and TOC in memory avoids re-parsing the PDF on every
    question within the same session.
    """

    def __init__(self, tree: Any) -> None:
        """
        Args:
            tree: Local tree dict returned by pdf_parser.parse_pdf().
        """
        self.tree = tree
        # Pre-compute the TOC once per session — done during __init__.
        self.toc = flatten_tree_toc(tree)
        print(f"[Pipeline] TOC built: {len(self.toc)} nodes indexed.")

        # Conversation history for multi-turn chat (excludes the system message,
        # which generator.py injects itself).
        self.conversation_history: list[dict] = []

    def query(self, question: str) -> str:
        """
        Run a full RAG cycle for the given question and return the answer.

        Steps:
          1. Tree Search  — ask the LLM which node IDs are relevant
          2. Retrieval    — extract those nodes' full text from the tree
          3. Generation   — ask Mistral to produce a grounded cited answer

        The answer and the question are appended to conversation_history so
        follow-up questions have context.

        Args:
            question: The user's natural-language question.

        Returns:
            The generated answer string.
        """
        print(f"\n[Pipeline] Question: {question!r}")

        # ── Step 1: Tree search ─────────────────────────────────────────────
        node_ids: list[str] = select_relevant_nodes(self.toc, question)

        if not node_ids:
            print("[Pipeline] No relevant nodes found — answering from general context.")

        # ── Step 2: Retrieve node content ───────────────────────────────────
        nodes: list[RetrievedNode] = retrieve_nodes(self.tree, node_ids)

        # ── Step 3: Generate answer ─────────────────────────────────────────
        answer: str = generate_answer(
            question=question,
            nodes=nodes,
            conversation_history=self.conversation_history if self.conversation_history else None,
        )

        # ── Update conversation history (keep last 10 turns to avoid overflow) ─
        self.conversation_history.append({"role": "user",      "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        if len(self.conversation_history) > 20:      # 10 turns × 2 messages
            self.conversation_history = self.conversation_history[-20:]

        return answer

    def query_stream(self, question: str):
        """
        Streaming version of query().
        Yields chunks of the answer as Mistral produces them.
        Updates conversation history at the end.
        """
        print(f"\n[Pipeline] Question (streaming): {question!r}")
        
        node_ids: list[str] = select_relevant_nodes(self.toc, question)
        if not node_ids:
            print("[Pipeline] No relevant nodes found — answering from general context.")
            
        nodes: list[RetrievedNode] = retrieve_nodes(self.tree, node_ids)
        
        # We need to capture the full answer as it streams out to save in history
        full_answer = []
        
        stream_gen = generate_answer_stream(
            question=question,
            nodes=nodes,
            conversation_history=self.conversation_history if self.conversation_history else None,
        )
        
        for chunk in stream_gen:
            full_answer.append(chunk)
            yield chunk
            
        # Update conversation history
        answer_text = "".join(full_answer)
        self.conversation_history.append({"role": "user",      "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer_text})
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def reset_history(self) -> None:
        """Clear conversation history to start a fresh topic."""
        self.conversation_history.clear()
        print("[Pipeline] Conversation history cleared.")
