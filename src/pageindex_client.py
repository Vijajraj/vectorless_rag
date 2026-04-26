"""
pageindex_client.py
-------------------
Thin wrapper around the PageIndex SDK for two operations:

  1. upload_pdf(path)    → doc_id  (uploads & starts tree indexing)
  2. wait_for_index(doc_id) → tree  (polls until processing is complete)
  3. get_tree(doc_id)    → tree    (fetch tree of an already-indexed document)

PageIndex builds a hierarchical tree from the document that we traverse
instead of doing nearest-neighbour vector search — hence "vectorless RAG".

SDK API (pageindex package):
  pi_client.submit_document(path)  → {"doc_id": "...", ...}
  pi_client.get_tree(doc_id)       → {"status": "completed"|"processing"|...,
                                       "result": <tree dict>}

Each node in the tree has: node_id, title, page_index, summary, children[].
"""

import time
import sys
from src.config import pi_client


def upload_pdf(pdf_path: str) -> str:
    """
    Upload a PDF to PageIndex and return the assigned document ID.

    PageIndex ingests the file, parses its logical structure (sections,
    headings, paragraphs) into a tree, and returns a doc_id you can reuse
    across sessions — no need to re-upload the same file.

    Args:
        pdf_path: Absolute or relative path to the PDF on disk.

    Returns:
        doc_id: Opaque string that identifies the indexed document.
    """
    print(f"[PageIndex] Uploading '{pdf_path}' …")
    response = pi_client.submit_document(pdf_path)
    doc_id: str = response["doc_id"]
    print(f"[PageIndex] Uploaded successfully. doc_id = {doc_id}")
    return doc_id


def wait_for_index(doc_id: str, poll_interval: int = 5, timeout: int = 300) -> dict:
    """
    Poll PageIndex until the tree index for *doc_id* is ready.

    PageIndex processes documents asynchronously. This function blocks
    until the status becomes 'completed', then returns the full tree dict.

    Args:
        doc_id:        Document ID returned by upload_pdf().
        poll_interval: Seconds between status checks (default 5).
        timeout:       Maximum seconds to wait before giving up (default 300).

    Returns:
        tree: The raw tree dict returned by PageIndex (root with nested children).

    Raises:
        TimeoutError: If processing takes longer than *timeout* seconds.
        RuntimeError: If PageIndex reports a failed status.
    """
    print(f"[PageIndex] Waiting for tree index to build (doc_id={doc_id}) …")
    elapsed = 0

    while elapsed < timeout:
        result = pi_client.get_tree(doc_id)
        status = result.get("status", "unknown")

        if status == "completed":
            print(f"\n[PageIndex] Index ready after ~{elapsed}s.")
            return result.get("result", result)    # unwrap if nested under "result"

        elif status in ("failed", "error"):
            raise RuntimeError(
                f"[PageIndex] Document processing failed for doc_id={doc_id}. "
                f"Status: {status}"
            )

        # Still processing — wait and retry
        sys.stdout.write(f"\r[PageIndex] Status: {status} … ({elapsed}s elapsed)  ")
        sys.stdout.flush()
        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(
        f"[PageIndex] Timed out after {timeout}s waiting for doc_id={doc_id}."
    )


def get_tree(doc_id: str) -> dict:
    """
    Retrieve the tree for a previously indexed document without polling.
    Use this when you already know the document is ready (e.g., reuse via --doc-id).

    Args:
        doc_id: Document ID of a ready document.

    Returns:
        tree: The raw tree dict (root node with nested children).

    Raises:
        RuntimeError: If the document is not ready or processing failed.
    """
    print(f"[PageIndex] Fetching existing tree for doc_id={doc_id} …")
    result = pi_client.get_tree(doc_id)
    status = result.get("status", "unknown")

    if status == "completed":
        print("[PageIndex] Tree fetched.")
        return result.get("result", result)
    elif status in ("failed", "error"):
        raise RuntimeError(f"[PageIndex] Document {doc_id} has status '{status}'.")
    else:
        # Still processing — give the user a hint
        raise RuntimeError(
            f"[PageIndex] Document {doc_id} is still '{status}'. "
            f"Wait for it to finish or use upload_pdf() + wait_for_index()."
        )
