import pytest
from argument_lab.orchestrator.graph import build_graph
from argument_lab.core.retriever import Retriever, RetrievedChunk

class DummyIndex:
    def similarity_search(self, query: str, k: int) -> list[RetrievedChunk]:
        return [RetrievedChunk("d1", "ex", 0.9)]

def test_build_graph_compiles():
    retriever = Retriever(DummyIndex())
    graph = build_graph(retriever)
    
    assert graph is not None
