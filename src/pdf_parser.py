"""
pdf_parser.py
-------------
Builds a hierarchical tree from a PDF using pypdf — fully local, no API.

Performance fix (v2):
  - Reads ALL page texts in a SINGLE pass at the start (O(pages) once).
  - Tree construction is pure dict-building from the pre-read list — zero extra disk IO.
  - Each node's content is capped at MAX_PAGES_PER_NODE (default 10) to prevent
    a single huge chapter from dominating the context window.

Tree node schema:
    {
        "node_id":  "node_3",
        "title":    "Section title",
        "summary":  "First 250 chars of section text",
        "content":  "Full text of this section (capped at MAX_PAGES_PER_NODE)",
        "children": [...]
    }

Public API:
    parse_pdf(pdf_path) -> dict    Returns the root tree node.
"""

from pathlib import Path
from pypdf import PdfReader

MAX_PAGES_PER_NODE = 10     # cap content per node — keeps context window sane


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_summary(text: str, max_chars: int = 250) -> str:
    """Return the first *max_chars* of *text* as a quick summary for the TOC LLM."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"


def _get_content(page_texts: list[str], p_start: int, p_end: int, total: int) -> str:
    """
    Slice *page_texts* from p_start to p_end (inclusive, 0-indexed),
    capped at MAX_PAGES_PER_NODE. No disk read — all text is pre-loaded.
    """
    p_start = max(0, p_start)
    p_end   = min(p_end, p_start + MAX_PAGES_PER_NODE - 1, total - 1)
    return "\n".join(page_texts[p_start : p_end + 1]).strip()


def _build_tree_fallback(
    page_texts: list[str],
    total_pages: int,
    pages_per_chunk: int = 3,
) -> dict:
    """
    Fallback for PDFs without a built-in TOC.
    Splits into fixed page-window nodes. All text from pre-loaded list.
    """
    children: list[dict] = []
    ctr = 1

    for start in range(0, total_pages, pages_per_chunk):
        end = min(start + pages_per_chunk - 1, total_pages - 1)
        content = _get_content(page_texts, start, end, total_pages)
        children.append({
            "node_id":  f"node_{ctr}",
            "title":    f"Pages {start + 1}–{end + 1}",
            "summary":  _make_summary(content),
            "content":  content,
            "children": [],
        })
        ctr += 1

    root_content = page_texts[0][:500] if page_texts else ""
    return {
        "node_id":  "node_0",
        "title":    "Document Root",
        "summary":  _make_summary(root_content),
        "content":  root_content,
        "children": children,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str) -> dict:
    """
    Parse a PDF into a hierarchical tree of section nodes.

    Performance: reads every page exactly ONCE using pypdf.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Root tree node dict with nested children.
    """
    pdf_path = str(Path(pdf_path).resolve())
    print(f"[PDFParser] Opening '{pdf_path}' …", flush=True)

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"[PDFParser] {total_pages} pages — reading all text in one pass …", flush=True)

    # ── Single-pass text extraction ───────────────────────────────────────────
    page_texts: list[str] = [page.extract_text() or "" for page in reader.pages]

    # ── Build tree ────────────────────────────────────────────────────────────
    print(f"[PDFParser] Using page-window fallback (3 pages/node).", flush=True)
    tree = _build_tree_fallback(page_texts, total_pages)

    def _count(node: dict) -> int:
        return 1 + sum(_count(c) for c in node.get("children", []))

    print(f"[PDFParser] Done. {_count(tree)} nodes total.", flush=True)
    return tree
