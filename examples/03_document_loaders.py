"""Example 03 - Document loaders (text, PDF, CSV, web) and eager vs lazy loading.

Usage:
    python examples/03_document_loaders.py path/to/file.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.loaders import lazy_load_documents, load_documents  # noqa: E402


def show(label: str, docs) -> None:
    print(f"{label}: {len(docs)} document(s)")
    if docs:
        print(f"  first doc metadata: {docs[0].metadata}")
        print(f"  first 150 chars:    {docs[0].page_content[:150]!r}\n")


def main(source: str) -> None:
    # Eager: everything is loaded into memory immediately -> returns a list.
    show("Eager load", load_documents(source))

    # Lazy: documents are produced one at a time -> returns a generator.
    print("Lazy load (one document at a time):")
    for i, doc in enumerate(lazy_load_documents(source), start=1):
        print(f"  doc {i}: {len(doc.page_content)} characters")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python examples/03_document_loaders.py <file-or-url>")
    main(sys.argv[1])
