"""Ask questions about the ingested documents.

Usage:
    python ask.py "What is the crypto regulation?"   # one-off question
    python ask.py                                    # interactive mode
"""

import argparse

from rag.chain import build_rag_chain
from rag.vectorstore import load_vector_store


def print_result(result: dict) -> None:
    print(f"\n{result['answer']}\n")
    print("Sources:")
    for i, doc in enumerate(result["sources"], start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"{source} (page {page + 1})" if isinstance(page, int) else source
        print(f"  [{i}] {label}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query your documents with RAG.")
    parser.add_argument("question", nargs="?", help="Omit for interactive mode")
    args = parser.parse_args()

    chain = build_rag_chain(load_vector_store())

    if args.question:
        print_result(chain.invoke(args.question))
        return

    print("Interactive mode - type 'exit' to quit.")
    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break
        print_result(chain.invoke(question))


if __name__ == "__main__":
    main()
