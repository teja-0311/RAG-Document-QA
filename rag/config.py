"""Central configuration. Values come from environment variables / a `.env` file."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Models
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "8"))

# Embedding in small batches avoids free-tier rate limits
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "10"))
EMBED_SLEEP_SECONDS = float(os.getenv("EMBED_SLEEP_SECONDS", "10"))

# Where the Chroma vector database is stored on disk
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")


def get_google_api_key() -> str:
    """Return the Google API key or fail with a helpful message."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key
