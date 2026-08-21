"""
Agent State Definition for LangGraph.

This defines the memory/state that is passed between nodes in the graph.
"""
from typing import Annotated, List, Sequence, TypedDict
import operator

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Represents the state of our agentic RAG system.
    """
    # The original or rewritten query
    question: str

    # Conversational history
    chat_history: List[BaseMessage]

    # Documents retrieved from the vector store
    documents: List[Document]

    # The final generated answer
    generation: str

    # Counter for query rewrite retries
    retries: int

    # Debug trace of what the agent decided
    trace: Annotated[List[str], operator.add]

    # Names of all uploaded PDF files (used to build the document manifest in the prompt)
    doc_names: List[str]
