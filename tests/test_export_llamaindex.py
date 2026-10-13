# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
export.llamaindex의 to_llamaindex/to_llamaindex_units/dedupe_by_chunk_id/
EvidenceRetriever가 실제 llama-index-core 설치 환경에서 동작하는지 검증.
"""
import pytest

pytest.importorskip("llama_index.core")

from evidence_chunker.export import TextChunk
from evidence_chunker.export.llamaindex import (
    EvidenceRetriever,
    dedupe_by_chunk_id,
    to_llamaindex,
    to_llamaindex_units,
)
from evidence_chunker.unit import EvidenceUnit


def _make_eu() -> EvidenceUnit:
    eu = EvidenceUnit(eu_id="doc-p1-0", page_no=1, doc_id="doc")
    eu.caption_text = "Table 1: Runtime"
    eu.context_before = ["문맥 단락 1"]
    eu.flattened_rows = ["row1: value1", "row2: value2"]
    return eu


class _FakeHybridChunk:
    text = "일반 본문 청크"


def test_to_llamaindex_one_node_per_chunk():
    eu = _make_eu()
    tc = TextChunk(_FakeHybridChunk(), doc_id="doc", index=0)

    nodes = to_llamaindex([eu, tc])

    assert len(nodes) == 2
    assert nodes[0].id_ == eu.chunk_id == "doc-p1-0"
    assert nodes[0].text == eu.text
    assert nodes[0].metadata["chunk_id"] == eu.chunk_id
    assert nodes[1].id_ == "doc-hybrid-0"


def test_to_llamaindex_units_small_to_big():
    eu = _make_eu()
    tc = TextChunk(_FakeHybridChunk(), doc_id="doc", index=0)

    nodes = to_llamaindex_units([eu, tc])

    eu_nodes = [n for n in nodes if n.metadata["chunk_id"] == eu.chunk_id]
    assert len(eu_nodes) == len(eu.retrieval_units)
    assert all(n.metadata["parent_text"] == eu.text for n in eu_nodes)
    assert {n.id_ for n in eu_nodes} == {f"{eu.chunk_id}-u{i}" for i in range(len(eu.retrieval_units))}

    tc_nodes = [n for n in nodes if n.metadata["chunk_id"] == tc.chunk_id]
    assert len(tc_nodes) == 1
    assert tc_nodes[0].id_ == tc.chunk_id  # is_atomic=True는 통짜로 들어감


def test_dedupe_by_chunk_id_keeps_first_per_parent():
    from llama_index.core.schema import NodeWithScore, TextNode

    nodes = [
        NodeWithScore(node=TextNode(text="a", metadata={"chunk_id": "eu-1"}), score=0.9),
        NodeWithScore(node=TextNode(text="b", metadata={"chunk_id": "eu-1"}), score=0.8),
        NodeWithScore(node=TextNode(text="c", metadata={"chunk_id": "eu-2"}), score=0.7),
    ]

    deduped = dedupe_by_chunk_id(nodes, k=5)

    assert [n.node.metadata["chunk_id"] for n in deduped] == ["eu-1", "eu-2"]


def test_evidence_retriever_wraps_and_dedupes():
    from llama_index.core.schema import NodeWithScore, TextNode

    class _FakeBaseRetriever:
        def retrieve(self, query):
            return [
                NodeWithScore(node=TextNode(text="a", metadata={"chunk_id": "eu-1"}), score=0.9),
                NodeWithScore(node=TextNode(text="b", metadata={"chunk_id": "eu-1"}), score=0.8),
                NodeWithScore(node=TextNode(text="c", metadata={"chunk_id": "eu-2"}), score=0.7),
            ]

    retriever = EvidenceRetriever(_FakeBaseRetriever(), k=5)
    results = retriever.retrieve("query")

    assert [n.node.metadata["chunk_id"] for n in results] == ["eu-1", "eu-2"]
