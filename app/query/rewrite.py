"""
Query rewriter — converts conversational questions into standalone queries.

Extracted from the existing create_history_aware_retriever logic.
Can now also be called standalone (e.g., from the CLI or evaluation pipeline).
"""
import logging
from typing import List

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logger = logging.getLogger(__name__)

CONTEXTUALIZE_PROMPT = (
    "Given a chat history and the latest user question which might reference context "
    "in the chat history, formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)


def build_contextualize_prompt() -> ChatPromptTemplate:
    """Return the prompt template for history-aware query rewriting."""
    return ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


def rewrite_query(query: str, chat_history: List[BaseMessage], llm) -> str:
    """
    Rewrite a conversational query into a standalone question.

    Args:
        query: The user's latest message (may reference chat history)
        chat_history: Previous conversation messages
        llm: Any LangChain LLM instance

    Returns:
        A standalone question string
    """
    if not chat_history:
        return query  # No rewriting needed if no history

    prompt = build_contextualize_prompt()
    chain = prompt | llm

    try:
        result = chain.invoke({"input": query, "chat_history": chat_history})
        rewritten = result.content.strip() if hasattr(result, "content") else str(result)
        logger.debug(f"Query rewritten: '{query}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.warning(f"Query rewrite failed: {e}. Using original query.")
        return query
