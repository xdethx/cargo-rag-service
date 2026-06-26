"""
llm.py — provider-agnostic chat client factory.

get_chat_model() reads LLM_PROVIDER from config and returns the right
LangChain chat client. All model names / keys / endpoints come from config
(environment variables) — never hardcoded here.

Supported providers:
    ollama           — local, free, no API key (good for dev/test)
    anthropic        — Anthropic Claude
    openai_compatible — one path for Grok / Groq / OpenRouter / any OpenAI-compatible API
    gemini           — Google Gemini

check_llm() does a minimal real liveness probe so /health reflects whether the
provider actually responds, not just whether it is configured.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

import src.config as config

# Type alias — any of the four chat clients share the same .invoke() interface.
ChatModel = ChatOllama | ChatAnthropic | ChatOpenAI | ChatGoogleGenerativeAI


def get_chat_model() -> ChatModel:
    """Return the configured LangChain chat client.

    temperature=0 keeps answers deterministic and grounded — the model should
    stick to the retrieved context, not improvise.
    """
    provider = config.LLM_PROVIDER

    if provider == "ollama":
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
        )

    if provider == "anthropic":
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=0,
        )

    if provider == "openai_compatible":
        # One client covers Grok (xAI), Groq, OpenRouter — only the base_url
        # and model differ; the OpenAI-compatible protocol is identical.
        return ChatOpenAI(
            model=config.OPENAI_COMPAT_MODEL,
            base_url=config.OPENAI_COMPAT_BASE_URL,
            api_key=config.OPENAI_COMPAT_API_KEY,
            temperature=0,
        )

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. "
        "Valid values: ollama | anthropic | openai_compatible | gemini"
    )


def check_llm() -> tuple[bool, str]:
    """Minimal liveness probe — actually calls the provider with a one-word query.

    Returns (True, "reachable") when the provider responds, or (False, "<reason>")
    when it doesn't. Used by GET /health so it reflects reality, not just config.

    Note: this sends a real request (one token for Ollama; billed for cloud providers).
    Cloud callers may want to gate /health behind auth or call it sparingly.
    """
    try:
        llm = get_chat_model()
        llm.invoke([HumanMessage(content="ping")])
        return True, "reachable"
    except Exception as exc:  # noqa: BLE001 — surface provider errors as strings
        return False, str(exc)
