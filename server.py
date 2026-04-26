"""
server.py
---------
FastAPI web server that wraps the Vectorless RAG pipeline.

Endpoints:
  POST /api/upload    Upload a PDF → parse locally → init the pipeline
  POST /api/query     Send a question → get a cited answer from Mistral
  POST /api/reset     Clear conversation history
  GET  /api/status    Return info about the currently loaded document
  GET  /              Serve the frontend (frontend/index.html)

State model:
  This is a single-user local app. The pipeline (tree + chat history)
  is stored in a global dict. Uploading a new PDF replaces the old pipeline.

Run with:
  python server.py
  # OR
  uvicorn server:app --reload --port 8000
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from src.pdf_parser import parse_pdf
from src.pipeline import RAGPipeline

# ── Global state (single-user local app) ──────────────────────────────────────
state: dict = {
    "pipeline":    None,   # RAGPipeline instance
    "doc_name":    None,   # Uploaded filename
    "node_count":  0,
    "page_count":  0,
    "has_toc":     False,
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Vectorless RAG API", version="1.0.0")

# Allow all origins for local dev (frontend is served from the same server anyway)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Helper ────────────────────────────────────────────────────────────────────

def _count_nodes(node: dict) -> int:
    return 1 + sum(_count_nodes(c) for c in node.get("children", []))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    """Serve the chat frontend."""
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index))


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF upload, parse it locally with PyMuPDF, and initialise
    the RAG pipeline. Replaces any previously loaded document.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the upload to a temporary file so PyMuPDF can open it
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Parse PDF in a thread (CPU-bound) so FastAPI stays responsive
        tree = await run_in_threadpool(parse_pdf, tmp_path)
    finally:
        os.unlink(tmp_path)   # clean up temp file

    # Initialise the RAG pipeline with the new tree
    pipeline = RAGPipeline(tree)

    # Count nodes for display
    node_count = _count_nodes(tree)
    has_toc = bool(tree.get("children"))

    state["pipeline"]   = pipeline
    state["doc_name"]   = file.filename
    state["node_count"] = node_count
    state["has_toc"]    = has_toc

    return {
        "success":    True,
        "doc_name":   file.filename,
        "node_count": node_count,
        "has_toc":    has_toc,
    }


class QueryRequest(BaseModel):
    question: str


@app.post("/api/query")
async def query(req: QueryRequest):
    """
    Run a RAG query against the currently loaded document.
    The pipeline maintains conversation history automatically.
    """
    if state["pipeline"] is None:
        raise HTTPException(status_code=400, detail="No document loaded. Upload a PDF first.")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Run in thread — LLM calls are blocking I/O + CPU
    answer = await run_in_threadpool(state["pipeline"].query, req.question)

    return {"answer": answer}


@app.post("/api/query_stream")
async def query_stream(req: QueryRequest):
    """
    Run a RAG query against the currently loaded document, but stream
    the response back token-by-token.
    """
    if state["pipeline"] is None:
        raise HTTPException(status_code=400, detail="No document loaded. Upload a PDF first.")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # We return a StreamingResponse that reads from the generator
    return StreamingResponse(
        state["pipeline"].query_stream(req.question),
        media_type="text/plain"
    )


@app.post("/api/reset")
async def reset_history():
    """Clear the conversation history for the current document."""
    if state["pipeline"] is None:
        raise HTTPException(status_code=400, detail="No document loaded.")

    state["pipeline"].reset_history()
    return {"success": True, "message": "Conversation history cleared."}


@app.get("/api/status")
async def status():
    """Return info about the currently loaded document."""
    return {
        "loaded":      state["pipeline"] is not None,
        "doc_name":    state["doc_name"],
        "node_count":  state["node_count"],
        "has_toc":     state["has_toc"],
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n🚀  Vectorless RAG  →  http://localhost:8000\n")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
