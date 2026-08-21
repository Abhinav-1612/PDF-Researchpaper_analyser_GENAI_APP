"""
Answer generator — Phase 5 upgrade (LangGraph Agentic RAG).

Replaces the linear LangChain with a stateful LangGraph agent that can:
1. Retrieve documents
2. Grade them for relevance
3. Generate an answer IF documents are relevant
4. Rewrite the query and retry IF documents are irrelevant
"""
import logging
from typing import Optional, Dict, Any

from langchain_groq import ChatGroq

from app.core.config import settings
from app.agents.graph import compile_agentic_rag

logger = logging.getLogger(__name__)


class AgenticRAGWrapper:
    """
    Wraps the compiled LangGraph agent to provide the same `.invoke()`
    interface as the old `create_rag_chain`.
    This ensures app.py doesn't break.
    """
    def __init__(self, agent_app):
        self.agent_app = agent_app

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects: {"input": query, "chat_history": [...], "doc_names": [...]}
        Returns: {"answer": text, "context": [Document, ...]}
        """
        question  = inputs["input"]
        chat_history = inputs.get("chat_history", [])
        doc_names = inputs.get("doc_names", [])

        # Phase 8: Configure Langfuse tracing
        callbacks = []
        if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
            try:
                from langfuse.callback import CallbackHandler
                langfuse_handler = CallbackHandler(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST,
                )
                callbacks.append(langfuse_handler)
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse tracing: {e}")

        # Initialize the LangGraph state
        initial_state = {
            "question":     question,
            "chat_history": chat_history,
            "documents":    [],
            "generation":   "",
            "retries":      0,
            "trace":        [],
            "doc_names":    doc_names,
        }

        logger.info(f"Starting Agentic RAG for: '{question}' | docs: {doc_names}")
        final_state = self.agent_app.invoke(initial_state, config={"callbacks": callbacks})

        # Streamlit debug removed

        return {
            "answer":  final_state.get("generation", "I could not find sufficient evidence."),
            "context": final_state.get("documents", [])
        }


def create_rag_chain(
    retriever: Any,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """
    Build a Phase 5 Agentic RAG system using LangGraph.

    Architecture:
      1. LangGraph State Machine
      2. QueryIntelligenceRetriever wrapped in the `retrieve` node
      3. Grading, Generation, and Query Transformation nodes

    Args:
        retriever:   Base retriever (HybridRetriever from Phase 3)
        model_name:  Groq model ID
        temperature: LLM temperature

    Returns:
        AgenticRAGWrapper with `.invoke()` method.
    """
    model_name  = model_name  or settings.DEFAULT_LLM_MODEL
    temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE

    logger.info(f"Building Phase 5 Agentic RAG with model: {model_name}")

    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_retries=settings.LLM_MAX_RETRIES,
    )

    # Wrap the base retriever with query intelligence
    from app.query.intelligence import QueryIntelligenceRetriever
    qi_retriever = QueryIntelligenceRetriever(
        base_retriever=retriever,
        llm=llm,
        enable_multi_query=False,      # Disabled: adds 3 extra LLM calls per query
        enable_decomposition=True,     # Keep: helps with compound questions
        enable_hyde=settings.ENABLE_HYDE,
    )

    # Compile the LangGraph app
    agent_app = compile_agentic_rag(retriever=qi_retriever, llm=llm)
    
    # Return the wrapper for UI compatibility
    return AgenticRAGWrapper(agent_app)
