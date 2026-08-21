"""
Core configuration for the Advanced RAG Platform.
All settings are loaded from environment variables / .env file.
"""
import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Central configuration for the entire RAG platform.
    Override any value via environment variables or the .env file.
    """

    # --- API Keys ---
    GROQ_API_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://localhost:3000"

    # --- LLM ---
    DEFAULT_LLM_MODEL: str = "qwen/qwen3.6-27b"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_RETRIES: int = 2
    AVAILABLE_LLM_MODELS: List[str] = [
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-20b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
    ]

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DEVICE: str = "cpu"

    # --- Vector Store ---
    VECTOR_STORE_TYPE: str = "pinecone"
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "pdf-analyzer"
    PINECONE_ENVIRONMENT: str = "us-east-1"

    # --- Retrieval ---
    RETRIEVAL_TOP_K: int = 5           # final chunks sent to LLM
    RETRIEVAL_FETCH_K: int = 15        # candidates before reranking (drastically cuts CPU time)
    BM25_TOP_K: int = 10               # BM25 candidates (Phase 3+)
    RERANKER_TOP_K: int = 5            # after cross-encoder (Phase 3+)
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Query Intelligence ---
    ENABLE_HYDE: bool = False          # HyDE is off by default (latency cost)
    MULTI_QUERY_COUNT: int = 3         # number of query reformulations

    # --- OCR ---
    TESSERACT_CMD: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    OCR_CONFIDENCE_THRESHOLD: float = 0.80
    OCR_MIN_TEXT_LENGTH: int = 50      # chars below which OCR is triggered
    OCR_DPI_SCALE: float = 2.0         # PyMuPDF render scale for OCR pages

    # --- Chunking ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 400
    PARENT_CHUNK_SIZE: int = 3000      # parent chunks (Phase 2+)
    PARENT_CHUNK_OVERLAP: int = 200

    # --- Agentic RAG ---
    MAX_AGENT_RETRIES: int = 2         # max corrective RAG retries

    # --- Observability ---
    ENABLE_TRACING: bool = False       # toggle Langfuse tracing
    TRACING_PROVIDER: str = "langfuse" # "langfuse" | "langsmith"

    model_config = {"env_file": ".env", "extra": "ignore", "env_file_encoding": "utf-8"}


# Singleton settings instance — import this everywhere
settings = Settings()


def configure_environment() -> None:
    """
    Apply environment-level settings that must be set before imports.
    Call this once at application startup.
    """
    # Suppress HuggingFace tokenizer parallelism warning
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Disable PaddlePaddle MKL-DNN to prevent PIR attribute errors on CPU
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_use_onednn", "0")
    
    # CRITICAL: Prevent silent crashes on Windows when PyTorch and Paddle both load OpenMP
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Set API keys from settings
    if settings.GROQ_API_KEY:
        os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
    if settings.PINECONE_API_KEY:
        os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
