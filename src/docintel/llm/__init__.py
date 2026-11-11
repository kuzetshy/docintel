"""LLM layer: the single entry point for chat models and embeddings.

Import from here, not from provider modules:

    from docintel.llm import ChatModel, Message, create_chat_model
"""

from docintel.llm.base import BaseChatModel, ChatModel, Embedder
from docintel.llm.errors import (
    LLMError,
    LLMRateLimitError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from docintel.llm.factory import create_chat_model, create_embedder
from docintel.llm.types import Generation, Message, Usage

__all__ = [
    "BaseChatModel",
    "ChatModel",
    "Embedder",
    "Generation",
    "LLMError",
    "LLMRateLimitError",
    "LLMRequestError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "Message",
    "Usage",
    "create_chat_model",
    "create_embedder",
]
