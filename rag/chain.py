"""Step 4 - Retrieve + generate: answer a question using only the retrieved context."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from rag import config

PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store: Chroma, llm=None):
    """Return a chain: question -> {"answer": str, "sources": list[Document]}."""
    retriever = vector_store.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
    llm = llm or ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        google_api_key=config.get_google_api_key(),
    )

    answer_chain = PROMPT | llm | StrOutputParser()

    # Retrieve once, then reuse the same docs for the prompt and for the sources.
    return (
        {"docs": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            answer=lambda x: answer_chain.invoke(
                {"context": format_docs(x["docs"]), "question": x["question"]}
            )
        )
        | (lambda x: {"answer": x["answer"], "sources": x["docs"]})
    )
