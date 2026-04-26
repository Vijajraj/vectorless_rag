"""
main.py
-------
CLI entry point for the Vectorless RAG system (local PyMuPDF edition).

Usage:
    # Parse a PDF and start chatting
    python main.py --pdf path/to/document.pdf

The session enters an interactive loop. Type 'quit', 'exit', or Ctrl+C to stop.
Type 'reset' to clear conversation history within the same document session.

How it works:
  1. PyMuPDF parses the PDF locally → builds a hierarchical tree (TOC-based)
  2. User asks a question → Mistral scans the tree TOC and picks relevant nodes
  3. Text from those nodes is retrieved
  4. Mistral generates a cited answer grounded in the retrieved text

No cloud API, no embeddings, no vector DB — fully local.
"""

import argparse
import sys

from src.pdf_parser import parse_pdf
from src.pipeline   import RAGPipeline


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectorless RAG: local PyMuPDF tree search + Mistral 7B via Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --pdf research_paper.pdf
  python main.py --pdf "C:\\Users\\you\\Downloads\\AI.pdf"
        """,
    )

    parser.add_argument(
        "--pdf",
        metavar="PATH",
        required=True,
        help="Path to a PDF file to parse and index locally.",
    )

    return parser.parse_args()


# ── Interactive chat loop ─────────────────────────────────────────────────────

def chat_loop(pipeline: RAGPipeline) -> None:
    """
    Run an interactive Q&A session over the loaded document.
    The pipeline maintains conversation history between turns.
    """
    print("\n" + "="*60)
    print("  Vectorless RAG — Interactive Chat  (fully local)")
    print("  Commands: 'reset' (clear history), 'quit' / 'exit' to stop")
    print("="*60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Session ended.]")
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("[Session ended.]")
            break

        if question.lower() == "reset":
            pipeline.reset_history()
            print("Assistant: Conversation history cleared. Ask a new question.\n")
            continue

        answer = pipeline.query(question)
        print(f"\nAssistant:\n{answer}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Parse the PDF locally with PyMuPDF
    try:
        tree = parse_pdf(args.pdf)
    except FileNotFoundError:
        print(f"\n[Error] PDF not found: {args.pdf!r}", file=sys.stderr)
        print("  Check the path and try again.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] Failed to parse PDF: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialise the RAG pipeline with the locally-built tree
    pipeline = RAGPipeline(tree)

    # Run the interactive chat session
    chat_loop(pipeline)


if __name__ == "__main__":
    main()
