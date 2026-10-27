"""Example 05 - The same RAG idea with OpenAI + FAISS (alternative stack).

Needs:  pip install -r requirements-optional.txt   and   OPENAI_API_KEY in .env

Usage:
    python examples/05_faiss_openai_variant.py data/your_file.pdf "Your question?"
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.loaders import load_documents  # noqa: E402
from rag.splitter import split_documents  # noqa: E402

load_dotenv()

PROMPT = ChatPromptTemplate.from_template(
    "Answer using only this context:\n{context}\n\nQuestion: {question}"
)


def main(source: str, question: str) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set in your .env file.")

    chunks = split_documents(load_documents(source))
    store = FAISS.from_documents(chunks, OpenAIEmbeddings())  # in-memory vector index
    retriever = store.as_retriever(search_kwargs={"k": 4})

    context = "\n\n".join(d.page_content for d in retriever.invoke(question))
    chain = PROMPT | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    print(chain.invoke({"context": context, "question": question}))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit('Usage: python examples/05_faiss_openai_variant.py <file> "<question>"')
    main(sys.argv[1], sys.argv[2])
