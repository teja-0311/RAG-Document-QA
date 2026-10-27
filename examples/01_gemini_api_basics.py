"""Example 01 - Calling an LLM directly (no LangChain).

Two ways to reach a model:
  1. The official Google Gen AI SDK
  2. A plain HTTP request to OpenRouter (OpenAI-compatible API)

Keys are read from environment variables / `.env` - never hardcode them.
"""

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from google import genai

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import config  # noqa: E402

load_dotenv()


def ask_gemini(prompt: str) -> str:
    client = genai.Client(api_key=config.get_google_api_key())
    response = client.models.generate_content(model=config.CHAT_MODEL, contents=prompt)
    return response.text


def ask_openrouter(prompt: str, model: str = "openrouter/free") -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in your .env file.")

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print("--- Gemini SDK ---")
    print(ask_gemini("Explain how AI works in three sentences."))

    if os.getenv("OPENROUTER_API_KEY"):
        print("\n--- OpenRouter ---")
        print(ask_openrouter("What do you know about India?"))
