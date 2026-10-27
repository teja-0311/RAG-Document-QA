"""Build the vector database from one or more files / URLs.

Usage:
    python ingest.py data/report.pdf
    python ingest.py data/notes.txt https://en.wikipedia.org/wiki/Mahatma_Gandhi
"""

import argparse

from rag.loaders import load_many
from rag.splitter import split_documents
from rag.vectorstore import build_vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument("sources", nargs="+", help="File paths (.pdf/.txt/.md/.csv) or URLs")
    args = parser.parse_args()

    documents = load_many(args.sources)
    chunks = split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks")

    build_vector_store(chunks)
    print("Done. Now run: python ask.py")


if __name__ == "__main__":
    main()
