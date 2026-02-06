"""
MidLens Chatbot
===============
AI-powered medical chatbot with agentic RAG and LLM fallback chain.

Primary:  ``AgenticChatbot`` — sentence-transformer embeddings + FAISS + tool use
Fallback: ``AIChatbot``      — multi-LLM with web/PubMed search
"""

from .agent import create_chatbot, AgenticChatbot
from .fallback import AIChatbot, ChatResponse, Source
from .llm_client import (
    call_gemini,
    call_groq,
    call_llm,
    get_available_providers,
    get_rate_limit_status,
)
from .rate_limiter import RollingWindowRateLimiter

__all__ = [
    # Primary chatbot (agentic RAG)
    "create_chatbot",
    "AgenticChatbot",
    # Fallback chatbot (multi-LLM)
    "AIChatbot",
    "ChatResponse",
    "Source",
    # LLM utilities
    "call_gemini",
    "call_groq",
    "call_llm",
    "get_available_providers",
    "get_rate_limit_status",
    "RollingWindowRateLimiter",
]
