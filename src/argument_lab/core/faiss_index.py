"""
argument_lab/core/faiss_index.py

Concrete implementation of the VectorIndex protocol backed by FAISS
and OpenAI embeddings. This is the production retriever for MVP.

The FaissIndex class is a thin wrapper — it satisfies the VectorIndex
protocol defined in retriever.py without importing from it, keeping the
dependency direction clean (core never imports from scripts).

Usage:
    from argument_lab.core.faiss_index import FaissIndex
    from argument_lab.core.retriever import Retriever

    index = FaissIndex.load("local_data/faiss_index")
    retriever = Retriever(index=index, top_k=4)
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings

from argument_lab.core.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Stored chunk metadata
# ---------------------------------------------------------------------------

@dataclass
class ChunkRecord:
    """
    Everything we need to reconstruct a RetrievedChunk from a FAISS hit.
    The FAISS index stores raw float vectors; metadata lives alongside it
    in a sidecar JSON file.
    """
    source_id: str   # e.g. "doc_03_chunk_12"
    excerpt: str     # the raw text of the chunk
    doc_title: str   # human-readable source label for the metrics dashboard


# ---------------------------------------------------------------------------
# FaissIndex
# ---------------------------------------------------------------------------

class FaissIndex:
    """
    Wraps a flat L2 FAISS index and a parallel list of ChunkRecords.

    The index and metadata are saved/loaded as a pair:
      <path>/index.faiss   — the binary FAISS index
      <path>/metadata.pkl  — pickled list[ChunkRecord]

    Cosine similarity is approximated by L2 distance on unit-normalised
    vectors: similarity = 1 - (l2_distance² / 2), clipped to [0, 1].
    """

    def __init__(
        self,
        index: faiss.Index,
        metadata: list[ChunkRecord],
        embeddings: OpenAIEmbeddings,
    ):
        self._index = index
        self._metadata = metadata
        self._embeddings = embeddings

    # ------------------------------------------------------------------
    # VectorIndex protocol implementation
    # ------------------------------------------------------------------

    def similarity_search(self, query: str, k: int) -> list[RetrievedChunk]:
        """
        Embeds the query, searches the FAISS index, and returns the top-k
        chunks as RetrievedChunk objects with cosine similarity scores.
        """
        if self._index.ntotal == 0:
            return []

        k = min(k, self._index.ntotal)
        query_vec = self._embed_query(query)

        distances, indices = self._index.search(query_vec, k)

        results: list[RetrievedChunk] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue  # FAISS returns -1 for unfilled slots
            record = self._metadata[idx]
            # Convert L2 distance on unit vectors to cosine similarity
            similarity = float(np.clip(1.0 - dist / 2.0, 0.0, 1.0))
            results.append(RetrievedChunk(
                source_id=record.source_id,
                excerpt=record.excerpt,
                score=round(similarity, 4),
            ))

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Saves the FAISS index and metadata sidecar to disk.
        Creates the directory if it doesn't exist.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self._metadata, f)

        print(f"[FaissIndex] Saved {self._index.ntotal} vectors to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FaissIndex":
        """
        Loads a previously saved FaissIndex from disk.
        Raises FileNotFoundError with a helpful message if the index
        doesn't exist yet (run scripts/ingest_corpus.py first).
        """
        path = Path(path)
        index_path = path / "index.faiss"
        meta_path = path / "metadata.pkl"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{path}'. "
                "Run `python setup/ingest_corpus.py` to build it first."
            )

        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
        )
        instance = cls(index=index, metadata=metadata, embeddings=embeddings)
        print(f"[FaissIndex] Loaded {index.ntotal} vectors from {path}")
        return instance

    # ------------------------------------------------------------------
    # Construction (used by ingestion script)
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, chunks: list[ChunkRecord]) -> "FaissIndex":
        """
        Embeds a list of ChunkRecords and builds a new FaissIndex.
        Called by the ingestion script — not at runtime.

        Uses a flat L2 index (IndexFlatL2) — exact search, no approximation.
        Appropriate for MVP corpus sizes (< 100k chunks). Switch to
        IndexIVFFlat for larger corpora.
        """
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
        )

        print(f"[FaissIndex] Embedding {len(chunks)} chunks...")
        texts = [c.excerpt for c in chunks]
        vectors = embeddings.embed_documents(texts)

        matrix = np.array(vectors, dtype=np.float32)
        # Normalise to unit length so L2 distance ≈ cosine distance
        faiss.normalize_L2(matrix)

        dimension = matrix.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(matrix)

        print(f"[FaissIndex] Built index: {index.ntotal} vectors, dim={dimension}")
        return cls(index=index, metadata=chunks, embeddings=embeddings)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self._embeddings.embed_query(query)
        matrix = np.array([vec], dtype=np.float32)
        faiss.normalize_L2(matrix)
        return matrix