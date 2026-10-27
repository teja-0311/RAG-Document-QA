"""Step 3 - Embed + store: turn chunks into vectors and keep them in ChromaDB."""

import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from rag import config


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.get_google_api_key(),
    )


def load_vector_store(embeddings=None) -> Chroma:
    """Open the persisted Chroma collection (created by `ingest.py`)."""
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def build_vector_store(
    chunks: list[Document],
    embeddings=None,
    batch_size: int = config.EMBED_BATCH_SIZE,
    sleep_seconds: float = config.EMBED_SLEEP_SECONDS,
) -> Chroma:
    """Embed chunks in batches (pausing between batches to respect rate limits)."""
    store = load_vector_store(embeddings)

    total = len(chunks)
    for start in range(0, total, batch_size):
        store.add_documents(chunks[start : start + batch_size])
        done = min(start + batch_size, total)
        print(f"Embedded {done}/{total} chunks")
        if done < total and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return store
