"""
Graph Nodes for LangGraph Agent.

Nodes define the actions the agent can take.
Edges (in graph.py) will route between these nodes.
"""
import logging
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser

from app.agents.state import AgentState
from app.agents.prompts import build_grader_prompt, build_rewriter_prompt, build_generator_prompt
from app.core.config import settings

logger = logging.getLogger(__name__)


class AgentNodes:
    """
    Container for the node functions used in our LangGraph state machine.
    """
    def __init__(self, retriever: Any, llm: Any):
        self.retriever = retriever
        self.llm = llm

    def retrieve(self, state: AgentState) -> Dict:
        """
        Node: Retrieve documents based on the current question.
        Uses the QueryIntelligenceRetriever from Phase 4.
        """
        logger.info("---RETRIEVE---")
        question = state["question"]
        
        # We assume the retriever handles History Awareness (from Phase 4 setup)
        # or we could do it here. For simplicity, we just pass the question.
        # The history_aware_retriever needs {"input": question, "chat_history": state["chat_history"]}
        
        # If the retriever passed in is just the base retriever, we invoke it with the string.
        # Let's assume `self.retriever` accepts a string and returns docs.
        try:
             documents = self.retriever.invoke(question)
        except Exception as e:
             logger.error(f"Retrieval error: {e}")
             documents = []
             
        return {"documents": documents, "trace": ["Retrieved documents"]}

    def grade_documents(self, state: AgentState) -> Dict:
        """
        Node: Pass-through — trust the Cross-Encoder Reranker which already
        scores and filters documents by relevance. Adding a second LLM grader
        on top causes false negatives (especially for multi-PDF questions) and
        doubles latency. We only trigger a query rewrite if retrieval returns
        zero documents.
        """
        logger.info("---DOCUMENT GRADE (PASS-THROUGH)---")
        documents = state["documents"]
        trace_msg = f"Accepted all {len(documents)} reranked docs (grader bypassed)"
        logger.info(trace_msg)
        return {"documents": documents, "trace": [trace_msg]}

    def generate(self, state: AgentState) -> Dict:
        """
        Node: Generate the final answer using the filtered relevant documents.
        """
        logger.info("---GENERATE---")
        question    = state["question"]
        documents   = state["documents"]
        chat_history = state["chat_history"]
        doc_names   = state.get("doc_names", [])  # list of uploaded PDF filenames

        # ── Build labelled context so the LLM knows which chunk came from where ── #
        context_parts = []
        for doc in documents:
            src  = doc.metadata.get("document_name", "Unknown Document")
            page = doc.metadata.get("page", "?")
            sect = doc.metadata.get("section", "")
            label = f"[SOURCE: {src} | Page {page}" + (f" | §{sect}" if sect else "") + "]"
            context_parts.append(f"{label}\n{doc.page_content}")
        context = "\n\n---\n\n".join(context_parts)

        # ── Inject document manifest into system prompt ───────────────────────── #
        prompt = build_generator_prompt(doc_names=doc_names)
        chain  = prompt | self.llm | StrOutputParser()

        generation = chain.invoke({
            "context":      context,
            "question":     question,
            "chat_history": chat_history,
        })

        return {"generation": generation, "trace": ["Generated answer"]}

    def transform_query(self, state: AgentState) -> Dict:
        """
        Node: Rewrite the query to produce a better question for retrieval.
        """
        logger.info("---TRANSFORM QUERY---")
        question = state["question"]
        retries = state.get("retries", 0)
        
        prompt = build_rewriter_prompt()
        chain = prompt | self.llm | StrOutputParser()
        
        better_question = chain.invoke({"question": question})
        
        trace_msg = f"Rewrote query (Attempt {retries+1})"
        return {
            "question": better_question, 
            "retries": retries + 1,
            "trace": [trace_msg]
        }
