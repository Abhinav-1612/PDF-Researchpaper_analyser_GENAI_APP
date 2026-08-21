"""
BM25 sparse retrieval — keyword-based search using Okapi BM25.

Complements dense vector search by excelling at exact keyword matches
and rare technical terms that embeddings often miss.

Library: rank_bm25 (lightweight, no server needed)
"""
import logging
import re
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25 index built from a corpus of LangChain Documents.

    Usage:
        retriever = BM25Retriever(child_docs)
        results = retriever.retrieve("what accuracy was reported?", top_k=20)
    """

    def __init__(self, documents: List[Document]):
        if not documents:
            raise ValueError("BM25Retriever requires at least one document.")

        self.documents = documents

        # Tokenise all documents at construction time
        tokenised_corpus = [self._tokenise(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenised_corpus)

        logger.info(f"BM25 index built with {len(documents)} documents")

    # ------------------------------------------------------------------ #
    # Tokenisation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """
        Simple whitespace + punctuation tokeniser.
        Lowercase, strip punctuation, split on whitespace.
        Falls back to [""] so BM25 never gets an empty token list.
        """
        tokens = re.findall(r"\b[a-z0-9][a-z0-9\-]*\b", text.lower())
        return tokens if tokens else [""]

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Tuple[Document, float]]:
        """
        Return top-k documents ranked by BM25 score.

        Args:
            query: Raw user query string
            top_k: Number of results to return

        Returns:
            List of (Document, bm25_score) tuples, sorted descending by score.
            Documents with score == 0 are excluded.
        """
        query_tokens = self._tokenise(query)
        scores = self.bm25.get_scores(query_tokens)

        indexed = [
            (i, float(score))
            for i, score in enumerate(scores)
            if score > 0
        ]
        indexed.sort(key=lambda x: x[1], reverse=True)

        results = [(self.documents[i], score) for i, score in indexed[:top_k]]

        logger.debug(
            f"BM25 retrieved {len(results)} / {top_k} docs "
            f"(max score: {results[0][1]:.3f})" if results else "BM25: no hits"
        )
        return results

    def get_all_scores(self, query: str) -> List[float]:
        """Return raw BM25 scores for all documents (used internally by RRF)."""
        return self.bm25.get_scores(self._tokenise(query)).tolist()
