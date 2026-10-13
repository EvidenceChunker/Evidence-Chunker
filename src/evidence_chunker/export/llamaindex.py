# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
llamaindex.py

RetrievalChunk(EvidenceUnit 또는 export.TextChunk) → LlamaIndex 변환.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import RetrievalChunk

try:
    from llama_index.core.schema import TextNode as LlamaIndexTextNode
except ImportError:
    LlamaIndexTextNode = None  # type: ignore


def to_llamaindex(chunks: list["RetrievalChunk"]) -> list["LlamaIndexTextNode"]:
    """
    RetrievalChunk 리스트 → LlamaIndex TextNode 리스트 (1 chunk = 1 TextNode).
    LlamaIndex RAG 파이프라인에 넘길 때 사용.

    chunk.chunk_id를 TextNode.id_로 명시 지정한다 — EvidenceUnit(eu_id)과
    export.TextChunk("{doc_id}-hybrid-{index}") 둘 다 항상 안정적인 값을
    준다.
    """
    if LlamaIndexTextNode is None:
        raise ImportError("llama-index-core is not installed. pip install llama-index-core")

    return [
        LlamaIndexTextNode(text=c.text, metadata=dict(c.metadata), id_=c.chunk_id)
        for c in chunks
    ]


def to_llamaindex_units(chunks: list["RetrievalChunk"]) -> list["LlamaIndexTextNode"]:
    """
    RetrievalChunk 리스트 → LlamaIndex TextNode 리스트 (행 단위 다중 벡터, small-to-big).
    to_langchain_units()와 동일 목적/분기 기준. metadata["parent_text"](chunk.text)를
    실제 LLM 컨텍스트로, node.text(작은 조각)는 검색 전용으로 쓸 것.
    """
    if LlamaIndexTextNode is None:
        raise ImportError("llama-index-core is not installed. pip install llama-index-core")

    nodes = []
    for c in chunks:
        if c.is_atomic:
            nodes.append(LlamaIndexTextNode(text=c.text, metadata=dict(c.metadata), id_=c.chunk_id))
            continue

        base_metadata = {k: v for k, v in c.metadata.items() if k != "retrieval_text"}
        base_metadata["parent_text"] = c.text
        nodes.extend(
            LlamaIndexTextNode(
                id_=f"{c.chunk_id}-u{i}",
                text=unit_text,
                metadata=dict(base_metadata),
            )
            for i, unit_text in enumerate(c.retrieval_units)
        )
    return nodes


# ---------------------------------------------------------------------------
# max-pool 재랭킹 (검색 결과 후처리)
# ---------------------------------------------------------------------------
#
# export.langchain의 동일 함수와 목적/근거가 같다 — 거기 주석 참고.
# 각 노드의 metadata["chunk_id"]가 부모 EU/청크를 가리킨다(위 to_llamaindex_units()
# 에서 base_metadata로 전파됨. id_는 "{chunk_id}-u{i}"라 청크 단위 그룹핑에는
# 못 쓴다 — 유닛마다 달라서 — 그래서 metadata["chunk_id"]를 따로 써야 한다).

def dedupe_by_chunk_id(
    results: list,
    k: "int | None" = None,
    key: str = "chunk_id",
) -> list:
    """
    이미 관련도 순으로 정렬된 검색 결과에서, 같은 부모 청크(chunk_id)를
    가진 항목 중 가장 앞에 나온 것(=가장 관련도 높은 것)만 남긴다.

    `NodeWithScore`(retriever.retrieve()의 반환 타입) 리스트와 순수
    `TextNode` 리스트 둘 다 받는다 — `getattr(item, "node", item)`로
    구분 없이 처리.

    사용 예시::

        results = retriever.retrieve(query)   # similarity_top_k는 넉넉하게 설정
        top5 = dedupe_by_chunk_id(results, k=5)

    보통은 이 함수를 직접 부르는 대신 EvidenceRetriever를 쓰는 게 더 편하다.
    """
    seen: set = set()
    deduped: list = []
    for item in results:
        node = getattr(item, "node", item)
        cid = node.metadata.get(key)
        if cid is not None:
            if cid in seen:
                continue
            seen.add(cid)
        deduped.append(item)
        if k is not None and len(deduped) >= k:
            break
    return deduped


class EvidenceRetriever:
    """max-pool dedupe가 기본으로 켜진 검색 래퍼.

    LlamaIndex의 기존 retriever(`VectorIndexRetriever` 등)를 감싼다.
    `dedupe=False`로 끄면 기존(유닛 단위 flat 랭킹) 동작 그대로.
    """

    def __init__(self, base_retriever, k: int = 5, dedupe: bool = True) -> None:
        """
        Args:
            base_retriever: `.retrieve(query)`를 제공하는 LlamaIndex retriever.
                생성 시 `similarity_top_k`를 최종 k보다 넉넉하게 잡아둘 것 —
                EU 하나가 retrieval_units개로 쪼개져 있으므로, dedupe 후에도
                k개가 채워지려면 후보를 그만큼 더 가져와야 한다.
            k: dedupe 후 최종 반환할 노드 수.
            dedupe: False면 max-pool 없이 base_retriever 결과 그대로(상위 k개만 자름).
        """
        self.base_retriever = base_retriever
        self.k = k
        self.dedupe = dedupe

    def retrieve(self, query: str) -> list:
        results = self.base_retriever.retrieve(query)
        if self.dedupe:
            return dedupe_by_chunk_id(results, k=self.k)
        return results[: self.k]
