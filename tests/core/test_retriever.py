import pytest
from argument_lab.core.retriever import Retriever, RetrievedChunk, RetrieverError

class MockIndex:
    def similarity_search(self, query: str, k: int) -> list[RetrievedChunk]:
        if query == "fail":
            raise RuntimeError("Index failure")
        return [
            RetrievedChunk(source_id="doc_1", excerpt=f"match for {query}", score=0.9),
            RetrievedChunk(source_id="doc_2", excerpt="other", score=0.5)
        ]

def test_retrieve():
    retriever = Retriever(index=MockIndex(), top_k=2)
    chunks = retriever.retrieve("test")
    assert len(chunks) == 2
    assert chunks[0].source_id == "doc_1"

def test_retrieve_failure():
    retriever = Retriever(index=MockIndex(), top_k=2)
    with pytest.raises(RetrieverError):
        retriever.retrieve("fail")

def test_retrieve_multi_dedup():
    class MockDedupIndex:
        def similarity_search(self, query: str, k: int) -> list[RetrievedChunk]:
            if query == "q1":
                return [RetrievedChunk("doc_1", "foo", 0.9)]
            if query == "q2":
                return [RetrievedChunk("doc_1", "foo", 0.95), RetrievedChunk("doc_2", "bar", 0.8)]
            return []
            
    retriever = Retriever(index=MockDedupIndex(), top_k=2)
    chunks = retriever.retrieve_multi(["q1", "q2"])
    
    assert len(chunks) == 2
    assert chunks[0].source_id == "doc_1"
    assert chunks[0].score == 0.95
    assert chunks[1].source_id == "doc_2"
