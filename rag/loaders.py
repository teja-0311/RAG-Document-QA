"""Step 1 - Load: turn files and URLs into LangChain `Document` objects."""

from pathlib import Path
from typing import Iterator

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

TEXT_SUFFIXES = {".txt", ".md"}


def get_loader(source: str) -> BaseLoader:
    """Pick the right loader for a file path or URL."""
    if source.startswith(("http://", "https://")):
        return WebBaseLoader(source)

    suffix = Path(source).suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(source)
    if suffix == ".csv":
        return CSVLoader(file_path=source)
    if suffix in TEXT_SUFFIXES:
        return TextLoader(source, encoding="utf-8")

    raise ValueError(f"Unsupported source type: {source!r}")


def load_documents(source: str) -> list[Document]:
    """Eager loading: read everything into memory at once."""
    return get_loader(source).load()


def lazy_load_documents(source: str) -> Iterator[Document]:
    """Lazy loading: yield one Document at a time (good for big files)."""
    yield from get_loader(source).lazy_load()


def load_many(sources: list[str]) -> list[Document]:
    """Load several files/URLs into a single list of Documents."""
    documents: list[Document] = []
    for source in sources:
        docs = load_documents(source)
        print(f"Loaded {len(docs):>4} document(s) from {source}")
        documents.extend(docs)
    return documents
