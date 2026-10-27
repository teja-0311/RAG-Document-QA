"""Example 02 - LangChain building blocks.

Covers: chat model -> message types -> prompt templates -> output parsers -> chains.
"""

import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import config  # noqa: E402

model = ChatGoogleGenerativeAI(
    model=config.CHAT_MODEL, google_api_key=config.get_google_api_key()
)


def basic_invoke() -> None:
    print("--- 1. Basic invoke ---")
    print(model.invoke("What do you know about LangChain?").content)


def message_types() -> None:
    print("\n--- 2. System / Human / AI messages (conversation history) ---")
    messages = [
        SystemMessage(content="Keep the tone polite and information factual."),
        HumanMessage(content="What is the capital of Maharashtra?"),
        AIMessage(content="Mumbai is the capital of Maharashtra."),
        HumanMessage(content="What is the capital of Andhra Pradesh?"),
    ]
    print(model.invoke(messages).content)


def prompt_template() -> None:
    print("\n--- 3. Prompt template ---")
    prompt = ChatPromptTemplate.from_template(
        "I am preparing for a DSA interview. You are an experienced {language} "
        "developer. Help me with this error: {error}"
    )
    prompt_value = prompt.invoke({"language": "Java", "error": "';' is missing"})
    print(prompt_value)
    print(model.invoke(prompt_value).content)


def chain_with_json_parser() -> None:
    print("\n--- 4. Chain: prompt | model | JSON parser ---")
    parser = JsonOutputParser()
    prompt = PromptTemplate(
        template="Help me resolve this error: {error}.\n{format_instructions}",
        input_variables=["error"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser

    print(chain.invoke({"error": "';' is missing"}))

    print("\nChain graph:")
    chain.get_graph().print_ascii()  # needs `grandalf`


if __name__ == "__main__":
    basic_invoke()
    message_types()
    prompt_template()
    chain_with_json_parser()
