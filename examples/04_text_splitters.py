"""Example 04 - Text splitters: why chunk size and overlap matter.

Runs offline - no API key needed.
"""

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

TEXT = (
    "India is a vibrant, diverse nation in South Asia and the seventh-largest "
    "country by land area. With a population of over 1.4 billion people, it is a "
    "rich melting pot of multiple cultures, languages, and major world religions. "
    "Its geography spans from the snowy peaks of the Himalayas in the north down "
    "to tropical coastlines bordered by the Indian Ocean, the Arabian Sea, and the "
    "Bay of Bengal. Famous for landmarks like the Taj Mahal, festivals like Diwali "
    "and Holi, and a fast-growing economy driven by technology and services, India "
    "balances deep historical traditions with modern global progress."
)


def show(name: str, chunks: list[str]) -> None:
    print(f"\n=== {name}: {len(chunks)} chunks ===")
    for i, chunk in enumerate(chunks, start=1):
        print(f"[{i}] ({len(chunk)} chars) {chunk}")


if __name__ == "__main__":
    # Splits on a single separator only (here: spaces).
    char_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50, separator=" ")
    show("CharacterTextSplitter (300/50)", char_splitter.split_text(TEXT))

    # Tries paragraph -> line -> word -> character, keeping text semantically intact.
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    show("RecursiveCharacterTextSplitter (300/50)", recursive_splitter.split_text(TEXT))

    # Tiny chunks make the overlap easy to see.
    tiny = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=10)
    show("Tiny chunks (40/10) - notice the repeated words at the edges", tiny.split_text(TEXT)[:5])
