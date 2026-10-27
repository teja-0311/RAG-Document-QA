"""Step 2 - Split: cut documents into overlapping chunks that fit an embedding model."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config


def split_documents(
    documents: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """Split with paragraph -> line -> word -> character fallbacks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)
