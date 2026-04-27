"""
argument_lab/core/retriever.py
 
Thin abstraction over the vector index. Agents call retrieve() to get
grounded evidence before generating an argument. The implementation is
swappable (FAISS for MVP, OpenSearch for v2) — agents never import the
index directly.
"""
 
from dataclasses import dataclass
from typing import Protocol
 
 
@dataclass
class RetrievedChunk:
    source_id: str
    excerpt: str
    score: float  # cosine similarity, [0, 1]
 
 
class VectorIndex(Protocol):
    """
    Any object with this interface can be used as the backing index.
    FAISS, ChromaDB, and OpenSearch all satisfy it with a thin wrapper.
    """
    def similarity_search(self, query: str, k: int) -> list[RetrievedChunk]:
        ...
 
 
class Retriever:
    """
    Injected into each agent node at graph compile time via the config.
    Agents call retrieve() with a natural-language query and get back
    chunks they can directly attach as EvidenceRef objects.
    """
 
    def __init__(self, index: VectorIndex, top_k: int = 4):
        self._index = index
        self._top_k = top_k
 
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Returns up to top_k chunks ranked by similarity to the query.
        Never raises — returns an empty list if the index is unavailable,
        which the agent node treats as a hard failure (no evidence = no argument).
        """
        try:
            return self._index.similarity_search(query, k=self._top_k)
        except Exception as exc:
            # Propagate as a typed error so the node can surface it cleanly
            raise RetrieverError(f"Index query failed: {exc}") from exc
 
    def retrieve_multi(self, queries: list[str]) -> list[RetrievedChunk]:
        """
        Runs multiple queries and deduplicates by source_id, keeping the
        highest-scoring chunk per source. Used when an agent formulates
        separate queries for their main claim and their rebuttal.
        """
        seen: dict[str, RetrievedChunk] = {}
        for query in queries:
            for chunk in self.retrieve(query):
                existing = seen.get(chunk.source_id)
                if existing is None or chunk.score > existing.score:
                    seen[chunk.source_id] = chunk
        return sorted(seen.values(), key=lambda c: c.score, reverse=True)
 
 
class RetrieverError(RuntimeError):
    pass