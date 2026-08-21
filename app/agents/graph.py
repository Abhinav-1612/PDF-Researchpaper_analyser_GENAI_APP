"""
LangGraph State Machine definition.

Routes the flow between nodes based on the state.
"""
import logging
from typing import Any

from langgraph.graph import END, StateGraph

from app.agents.state import AgentState
from app.agents.nodes import AgentNodes
from app.core.config import settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def decide_to_generate(state: AgentState):
    """
    Edge decision: after grading, do we generate or rewrite?
    """
    logger.info("---DECIDE TO GENERATE---")
    filtered_docs = state["documents"]
    retries = state.get("retries", 0)
    
    if not filtered_docs:
        # All documents were deemed irrelevant
        if retries >= MAX_RETRIES:
            logger.info("---MAX RETRIES REACHED. FORCE GENERATE---")
            return "generate"
        logger.info("---ALL DOCUMENTS IRRELEVANT. TRANSFORM QUERY---")
        return "transform_query"
        
    logger.info("---RELEVANT DOCUMENTS FOUND. GENERATE---")
    return "generate"


def compile_agentic_rag(retriever: Any, llm: Any):
    """
    Build and compile the LangGraph workflow.
    """
    nodes = AgentNodes(retriever, llm)
    
    workflow = StateGraph(AgentState)
    
    # Define the nodes
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("transform_query", nodes.transform_query)
    
    # Define the edges (workflow routing)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    # Conditional edge after grading
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        }
    )
    
    # If we rewrite the query, we go back to retrieval
    workflow.add_edge("transform_query", "retrieve")
    
    # After generation, we finish
    workflow.add_edge("generate", END)
    
    # Compile the graph
    app = workflow.compile()
    return app
