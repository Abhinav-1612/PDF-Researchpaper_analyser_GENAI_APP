"""
FastAPI Server for the Advanced Agentic RAG Platform.

Endpoints:
  GET  /health   — health check
  GET  /models   — list available LLM models
  POST /upload   — ingest PDF documents
  POST /query    — single-shot Q&A (JSON response)
  POST /stream   — streaming Q&A (Server-Sent Events)
"""
import json
import logging
from typing import List, AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import settings, configure_environment
from app.ingestion.ocr import init_paddle_ocr
from app.ingestion.pdf_loader import process_pdf_bytes_structured, process_pdf_bytes
from app.ingestion.chunking import create_parent_child_chunks, chunk_documents
from app.retrieval.vector_store import build_hybrid_retriever, build_vectorstore_retriever
from app.generation.answer_generator import create_rag_chain

configure_environment()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Intelligence API",
    description="Agentic Document Intelligence — Hybrid RAG + Streaming",
    version="2.0.0",
)

# ── CORS — allow the React dev server and any local origin ────────────────── #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state (ephemeral ChromaDB per session) ─────────────────────────── #
global_retriever = None
global_rag_chain  = None
global_doc_names: list = []   # filenames of all currently-indexed PDFs
_ocr_engine = None            # cached PaddleOCR instance


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = init_paddle_ocr()
    return _ocr_engine


# ── Pydantic schemas ──────────────────────────────────────────────────────── #

class QueryRequest(BaseModel):
    query: str
    model: str = settings.DEFAULT_LLM_MODEL
    chat_history: List[dict] = []


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


# ── Helpers ───────────────────────────────────────────────────────────────── #

def _format_sources(context_docs) -> List[dict]:
    return [
        {
            "content": doc.page_content,
            "extraction_method": doc.metadata.get("extraction_method", "Unknown"),
            "page": doc.metadata.get("page", "?"),
            "section": doc.metadata.get("section", ""),
            "document_name": doc.metadata.get("document_name", ""),
            "content_type": doc.metadata.get("content_type", "text"),
            "dense_score": doc.metadata.get("dense_score"),
            "bm25_score": doc.metadata.get("bm25_score"),
            "rrf_score": doc.metadata.get("rrf_score"),
            "rerank_score": doc.metadata.get("rerank_score"),
        }
        for doc in context_docs
    ]


def _build_chat_history(raw: List[dict]):
    """Convert plain dicts to LangChain message objects."""
    from langchain_core.messages import HumanMessage, AIMessage
    history = []
    for m in raw:
        if m.get("role") == "user":
            history.append(HumanMessage(content=m["content"]))
        elif m.get("role") == "assistant":
            history.append(AIMessage(content=m["content"]))
    return history


# ── Endpoints ─────────────────────────────────────────────────────────────── #

@app.get("/health")
def health_check():
    """Health check — also reports whether an index has been built."""
    return {"status": "ok", "index_ready": global_rag_chain is not None}


@app.get("/models")
def list_models():
    """Return the list of available LLM models."""
    return {
        "models": [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
            "meta-llama/llama-3.3-70b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "deepseek/deepseek-chat:free"
        ],
        "default": "qwen/qwen3.6-27b",
    }


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF documents, parse and index them.
    Returns chunk statistics on success.
    """
    global global_retriever, global_rag_chain, global_doc_names

    ocr_engine = _get_ocr_engine()
    all_parent_docs, all_child_docs, fallback_chunks = [], [], []
    use_parent_child = True
    uploaded_names   = []

    logger.info(f"Processing {len(files)} uploaded file(s) via API")

    for file in files:
        content  = await file.read()
        doc_name = file.filename or "document.pdf"
        uploaded_names.append(doc_name)

        doc_id, sections = process_pdf_bytes_structured(
            file_bytes=content,
            document_name=doc_name,
            ocr_engine=ocr_engine,
        )

        if sections:
            parents, children = create_parent_child_chunks(sections, doc_name, doc_id)
            all_parent_docs.extend(parents)
            all_child_docs.extend(children)
        else:
            use_parent_child = False
            page_chunks = process_pdf_bytes(content, doc_name, ocr_engine)
            fallback_chunks.extend(chunk_documents(page_chunks))

    if use_parent_child and all_child_docs:
        global_retriever = build_hybrid_retriever(
            parent_docs=all_parent_docs,
            child_docs=all_child_docs,
            enable_reranking=True,
        )
    else:
        combined = fallback_chunks + all_child_docs
        global_retriever = build_vectorstore_retriever(combined or [])

    global_rag_chain = create_rag_chain(global_retriever)
    global_doc_names = uploaded_names   # ← remember what was uploaded

    return {
        "message": "Documents successfully indexed.",
        "parent_chunks": len(all_parent_docs),
        "child_chunks":  len(all_child_docs),
        "fallback_chunks": len(fallback_chunks),
        "documents": uploaded_names,
    }


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Single-shot Q&A — returns full answer + sources as JSON."""
    if not global_rag_chain:
        raise HTTPException(status_code=400, detail="No documents indexed. Please POST to /upload first.")

    # Rebuild chain with the model the frontend requested
    from app.generation.answer_generator import create_rag_chain as _build
    chain = _build(global_retriever, model_name=request.model)

    try:
        response = chain.invoke({
            "input":        request.query,
            "chat_history": _build_chat_history(request.chat_history),
            "doc_names":    global_doc_names,
        })
        return QueryResponse(
            answer=response["answer"],
            sources=_format_sources(response.get("context", [])),
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream")
async def stream_rag(request: QueryRequest):
    """
    Streaming Q&A via Server-Sent Events (SSE).

    Event types emitted:
      data: {"type": "status",  "content": "Retrieving..."}
      data: {"type": "token",   "content": "word "}
      data: {"type": "sources", "content": [...]}
      data: {"type": "timing",  "content": {retrieval_ms, stream_ms, total_ms}}
      data: {"type": "done"}
      data: {"type": "error",   "content": "message"}
    """
    if not global_rag_chain:
        async def _err():
            yield _sse({"type": "error", "content": "No documents indexed. Please upload first."})
        return StreamingResponse(_err(), media_type="text/event-stream")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            import asyncio, time as _time

            # ── Step 1: Run the full agentic pipeline (retrieval + generation) ── #
            yield _sse({"type": "status", "content": "🔍 Retrieving document fragments..."})

            from app.generation.answer_generator import create_rag_chain as _build
            chain = _build(global_retriever, model_name=request.model)

            yield _sse({"type": "status", "content": "⚙️ Running agentic pipeline..."})

            t_start = _time.perf_counter()
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: chain.invoke({
                    "input":        request.query,
                    "chat_history": _build_chat_history(request.chat_history),
                    "doc_names":    global_doc_names,
                })
            )
            retrieval_ms = round((_time.perf_counter() - t_start) * 1000)

            sources     = _format_sources(response.get("context", []))
            full_answer = response.get("answer", "")

            # ── Step 2: Strip <think> block and stream only the clean answer ── #
            yield _sse({"type": "status", "content": "✍️ Streaming response..."})

            # Extract <think> block if present and send it as a separate event.
            # IMPORTANT: require a proper </think> closing tag — using $ as fallback
            # caused the regex to consume the entire answer as "thinking" content.
            import re as _re
            think_match = _re.search(r"<think>([\s\S]*?)</think>", full_answer, _re.IGNORECASE)
            if think_match:
                yield _sse({"type": "thinking", "content": think_match.group(1).strip()})
                # Remove the think block (and any leading whitespace) from the visible answer
                full_answer = _re.sub(r"<think>[\s\S]*?</think>", "", full_answer, flags=_re.IGNORECASE).strip()
            elif full_answer.strip().startswith("<think>"):
                # Model started thinking but never closed the tag — treat entire content as thinking
                # and leave a fallback message so the answer is not empty
                think_content = _re.sub(r"^<think>\s*", "", full_answer.strip(), flags=_re.IGNORECASE)
                yield _sse({"type": "thinking", "content": think_content.strip()})
                full_answer = "I've completed my analysis. Please see the chain of thought above."

            t_stream = _time.perf_counter()
            words = full_answer.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                yield _sse({"type": "token", "content": token})
                await asyncio.sleep(0.015)  # ~65 words/sec typewriter pace
            stream_ms = round((_time.perf_counter() - t_stream) * 1000)

            # ── Step 3: Send timing, sources, then done ───────────────────────── #
            yield _sse({"type": "timing", "content": {
                "retrieval_ms": retrieval_ms,
                "stream_ms":    stream_ms,
                "total_ms":     retrieval_ms + stream_ms,
            }})
            yield _sse({"type": "sources", "content": sources})
            yield _sse({"type": "done"})

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield _sse({"type": "error", "content": str(e)})


    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"
