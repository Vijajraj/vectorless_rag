"""
retriever.py
------------
Given a list of node IDs chosen by the tree search step, this module
walks the PageIndex tree and extracts the full text content of those nodes.

Walk strategy:
  - We do a DFS over the tree, collecting any node whose ID is in our target set.
  - If a selected node has children, we also gather their text to capture the
    full section — not just the heading node — giving the generator richer context.

Functions:
  retrieve_nodes(tree, node_ids)  → list[RetrievedNode]
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedNode:
    """Represents a single retrieved piece of document content."""
    node_id: str
    title:   str
    content: str                       # Full text of the node + its children
    depth:   int = 0
    child_titles: list[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _attr(node: Any, *keys: str, default: str = "") -> str:
    """Safely get an attribute from a PageIndex node (object or dict)."""
    for key in keys:
        val = (
            node.get(key)
            if isinstance(node, dict)
            else getattr(node, key, None)
        )
        if val:
            return str(val).strip()
    return default


def _get_id(node: Any) -> str:
    """Get node ID — PageIndex uses 'node_id'; fall back to 'id'."""
    if isinstance(node, dict):
        return node.get("node_id") or node.get("id") or ""
    return getattr(node, "node_id", "") or getattr(node, "id", "") or ""


def _get_children(node: Any) -> list:
    """Return children list regardless of whether node is an object or dict."""
    if isinstance(node, dict):
        return node.get("children") or []
    return getattr(node, "children", None) or []


def _collect_text(node: Any) -> str:
    """
    Recursively collect all text content from a node and its descendants.
    PageIndex stores content in a 'text' or 'content' field on leaf nodes.
    """
    own_text = _attr(node, "text", "content")
    parts = [own_text] if own_text else []
    for child in _get_children(node):
        child_text = _collect_text(child)
        if child_text:
            parts.append(child_text)
    return "\n".join(parts)


# ── Main function ──────────────────────────────────────────────────────────────

def retrieve_nodes(tree: Any, node_ids: list[str]) -> list[RetrievedNode]:
    """
    Walk the PageIndex tree and extract full content for each node in *node_ids*.

    We collect the node's own text PLUS all descendant text, so a section header
    node returns the complete section body — mirroring how a human would "open"
    that chapter.

    Args:
        tree:     PageIndex tree object (root node, may be nested).
        node_ids: IDs of nodes to retrieve (from tree_search.select_relevant_nodes).

    Returns:
        Ordered list of RetrievedNode objects, one per matched ID.
        Order matches the original node_ids list.
    """
    if not node_ids:
        return []

    target_set = set(node_ids)
    found: dict[str, RetrievedNode] = {}   # id → RetrievedNode

    def _walk(node: Any, depth: int = 0) -> None:
        node_id = _get_id(node)
        if not node_id:
            for child in _get_children(node):
                _walk(child, depth)
            return

        if node_id in target_set:
            title    = _attr(node, "title", "heading")
            content  = _collect_text(node)
            children = _get_children(node)
            child_titles = [
                _attr(c, "title", "heading")
                for c in children
                if _attr(c, "title", "heading")
            ]
            found[node_id] = RetrievedNode(
                node_id=node_id,
                title=title,
                content=content,
                depth=depth,
                child_titles=child_titles,
            )
            # Don't recurse further — we already gathered all descendant text.
            return

        # Not a target: keep searching children
        for child in _get_children(node):
            _walk(child, depth + 1)

    _walk(tree)

    # Return in the same order as node_ids so the generator sees them ranked
    result: list[RetrievedNode] = []
    for nid in node_ids:
        if nid in found:
            result.append(found[nid])
        else:
            print(f"[Retriever] Warning: node_id '{nid}' not found in tree.")

    print(f"[Retriever] Retrieved {len(result)} node(s).")
    return result
