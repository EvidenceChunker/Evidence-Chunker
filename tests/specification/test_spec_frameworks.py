from types import SimpleNamespace
import pytest
from evidence_chunker import EvidenceUnit
from evidence_chunker.export import TextChunk
from ._data import EU_KEYS, TEXT_KEYS

def sample_chunks():
    eu=EvidenceUnit(eu_id="paper-p1-0",page_no=1,doc_id="paper",
                    caption_text="Table 1: Revenue",flattened_rows=["Revenue: 100"])
    return eu,TextChunk(SimpleNamespace(text="General discussion"),"paper",0)

@pytest.mark.spec("QA-40")
def test_qa40_langchain_single_and_units_schema():
    # Required for this test: absence is an environment error, not an acceptance pass.
    from langchain_core.documents import Document
    from evidence_chunker.export.langchain import to_langchain,to_langchain_units
    eu,tc=sample_chunks()
    docs=to_langchain([eu,tc])
    assert all(isinstance(d,Document) for d in docs)
    assert set(docs[0].metadata)==EU_KEYS and set(docs[1].metadata)==TEXT_KEYS
    assert docs[0].page_content==eu.text and docs[1].page_content==tc.text
    units=to_langchain_units([eu,tc])
    child=[d for d in units if d.metadata["chunk_id"]==eu.chunk_id]
    ordinary=[d for d in units if d.metadata["chunk_id"]==tc.chunk_id]
    assert [d.page_content for d in child]==eu.retrieval_units
    assert all(set(d.metadata)==EU_KEYS-{"retrieval_text"}|{"parent_text"} for d in child)
    assert all(d.metadata["parent_text"]==eu.text for d in child)
    assert len(ordinary)==1 and set(ordinary[0].metadata)==TEXT_KEYS

@pytest.mark.spec("QA-41")
def test_qa41_real_vectorstore_question_hit_and_dedup():
    # Real in-memory vector store; deterministic term-count embeddings.
    # This checks indexing/ranking/wrapper wiring, not pretrained-model quality.
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import InMemoryVectorStore
    from evidence_chunker.export.langchain import to_langchain_units,EvidenceRetriever
    class TermEmbeddings(Embeddings):
        def embed_query(self,text):
            t=text.lower()
            return [float(t.count("revenue")+t.count("sales")),
                    float(t.count("rain")+t.count("weather")),0.01]
        def embed_documents(self,texts):
            return [self.embed_query(t) for t in texts]
    gold=EvidenceUnit(eu_id="sales",page_no=1,caption_text="Table 1: Revenue",
        flattened_rows=["Sales revenue: 100","Sales revenue: 200"],context_before=["Annual sales revenue"])
    other=EvidenceUnit(eu_id="weather",page_no=2,caption_text="Table 2: Weather",
        flattened_rows=["Rain: 10","Rain: 20"])
    store=InMemoryVectorStore.from_documents(to_langchain_units([gold,other]),TermEmbeddings())
    hits=EvidenceRetriever(store,k=2,fetch_k=20).invoke("What is the sales revenue?")
    ids=[d.metadata["chunk_id"] for d in hits]
    assert ids[0]=="sales"
    assert len(ids)==len(set(ids))==2

@pytest.mark.spec("QA-43")
def test_qa43_llamaindex_wrapper_honors_k():
    from llama_index.core.schema import NodeWithScore,TextNode
    from evidence_chunker.export.llamaindex import EvidenceRetriever
    items=[NodeWithScore(node=TextNode(text=x,metadata={"chunk_id":x}),score=s)
           for x,s in [("a",.9),("a",.8),("b",.7),("c",.6)]]
    class Retriever:
        def retrieve(self,query):
            assert query=="example query"
            return items
    hits=EvidenceRetriever(Retriever(),k=1).retrieve("example query")
    assert hits==[items[0]]

@pytest.mark.spec("QA-44")
@pytest.mark.parametrize("framework",["langchain","llamaindex"])
@pytest.mark.parametrize("scored",[False,True])
def test_qa44_dedupe_all_result_forms_preserves_first_and_order(framework,scored):
    if framework=="langchain":
        from langchain_core.documents import Document
        from evidence_chunker.export.langchain import dedupe_by_chunk_id
        docs=[Document(page_content=str(i),metadata={"chunk_id":cid}) for i,cid in enumerate(["a","a","b"])]
        items=list(zip(docs,[.9,.8,.7])) if scored else docs
    else:
        from llama_index.core.schema import TextNode,NodeWithScore
        from evidence_chunker.export.llamaindex import dedupe_by_chunk_id
        docs=[TextNode(text=str(i),metadata={"chunk_id":cid}) for i,cid in enumerate(["a","a","b"])]
        items=[NodeWithScore(node=d,score=s) for d,s in zip(docs,[.9,.8,.7])] if scored else docs
    hits=dedupe_by_chunk_id(items,k=2)
    assert len(hits)==2
    assert hits[0] is items[0] and hits[1] is items[2]
    if scored:
        scores=[x[1] if framework=="langchain" else x.score for x in hits]
        assert scores==[.9,.7]

