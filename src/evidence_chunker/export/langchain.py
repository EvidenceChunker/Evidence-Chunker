# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
langchain.py

RetrievalChunk(EvidenceUnit 또는 export.TextChunk) → LangChain 변환.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import RetrievalChunk

try:
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    LangChainDocument = None  # type: ignore


def to_langchain(chunks: list["RetrievalChunk"]) -> list["LangChainDocument"]:
    """
    RetrievalChunk 리스트 → LangChain Document 리스트 (1 chunk = 1 Document).
    RAG 파이프라인에 넘길 때 사용.

    임베딩 모델은 보통 입력을 256토큰 안팎에서 truncate하므로,
    page_content(전체 내용) 대신 metadata["retrieval_text"](EvidenceUnit
    전용, table_html 제외한 축약 텍스트)로 벡터스토어를 구성하면 Recall이
    개선된다(LangChain MultiVectorRetriever 등에서 활용).
    """
    if LangChainDocument is None:
        raise ImportError("langchain_core is not installed. pip install langchain-core")

    return [
        LangChainDocument(page_content=c.text, metadata=dict(c.metadata))
        for c in chunks
    ]


def to_langchain_units(chunks: list["RetrievalChunk"]) -> list["LangChainDocument"]:
    """
    RetrievalChunk 리스트 → LangChain Document 리스트 (행 단위 다중 벡터, small-to-big).

    to_langchain()은 chunk 하나를 문서 하나로 통째로 임베딩하는데, 표 안의
    특정 셀 값을 묻는 질의는 같은 표의 나머지 행 데이터에 묻혀 코사인
    유사도에서 밀리는 문제가 있다 (EvidenceUnit.retrieval_units 참고).

    is_atomic=False인(=EvidenceUnit) chunk만 retrieval_units 단위로 쪼갠다.
    is_atomic=True(=TextChunk)는 그 자체가 이미 최소 단위라 to_langchain()과
    동일하게 1개 그대로 반환한다.

    벡터스토어에는 이 작은 Document들을 넣어서 검색 정밀도를 높이고,
    실제 LLM에 넘길 때는 page_content(작은 조각) 대신
    metadata["parent_text"](chunk.text, 표 전체 맥락)를 사용할 것.
    """
    if LangChainDocument is None:
        raise ImportError("langchain_core is not installed. pip install langchain-core")

    docs = []
    for c in chunks:
        if c.is_atomic:
            docs.append(LangChainDocument(page_content=c.text, metadata=dict(c.metadata)))
            continue

        base_metadata = {k: v for k, v in c.metadata.items() if k != "retrieval_text"}
        base_metadata["parent_text"] = c.text
        docs.extend(
            LangChainDocument(page_content=unit_text, metadata=dict(base_metadata))
            for unit_text in c.retrieval_units
        )
    return docs


# to_langchain_units()로 만든 벡터스토어는 EU 하나가 retrieval_units개의
# Document로 쪼개져 들어간다. similarity_search(query, k=10)을 그냥 쓰면
# 같은 EU의 유닛 여러 개가 top-k 슬롯을 몰아 차지해서 실제로 검색되는
# "서로 다른 근거"의 개수가 줄어든다. 같은 chunk_id(부모 EU/청크)끼리
# 최고 점수만 남기고 재랭킹하면(max-pool) 이 손해를 없앨 수 있다.

def dedupe_by_chunk_id(
    results: list,
    k: "int | None" = None,
    key: str = "chunk_id",
) -> list:
    """
    이미 관련도 순으로 정렬된 검색 결과에서, 같은 부모 청크(chunk_id)를
    가진 항목 중 **가장 앞에 나온 것(=가장 관련도 높은 것)만** 남긴다.

    벡터스토어의 similarity_search() 계열 함수는 이미 관련도 내림차순으로
    정렬해서 반환하므로, "정렬된 순서에서 그룹당 첫 등장만 채택"이
    "그룹별 최고 점수로 재랭킹"과 결과가 같다. (Document, score) 튜플
    리스트든 Document 리스트든 둘 다 받는다.

    사용 예시(직접 vectorstore를 쓸 때)::

        results = vectorstore.similarity_search_with_score(query, k=20)
        top5 = dedupe_by_chunk_id(results, k=5)

    보통은 이 함수를 직접 부르는 대신 EvidenceRetriever를 쓰는 게 더 편하다.

    Args:
        results: similarity_search()/similarity_search_with_score() 등이
            반환한, 이미 정렬된 검색 결과. Document 리스트 또는
            (Document, score) 튜플 리스트.
        k: dedupe 후 상위 k개만 반환. None이면 dedupe만 하고 전부 반환.
        key: 그룹핑에 쓸 metadata 키. 기본 "chunk_id".

    Returns:
        입력과 같은 형태의 리스트, 중복 chunk_id 제거 + (k가 있으면) 상위 k개로 자름.
    """
    seen: set = set()
    deduped: list = []
    for item in results:
        doc = item[0] if isinstance(item, tuple) else item
        cid = doc.metadata.get(key)
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

    `vectorstore.as_retriever()` 대신 이걸 쓰면 Recall@5/10이 개선된다.
    LangChain의 `BaseRetriever`를 상속하지 않은 얇은 래퍼라 버전 호환성
    문제가 없다 — `get_relevant_documents()`(구버전 API)와 `invoke()`
    (신버전 API) 둘 다 제공한다.

    dedupe=False로 끄면 기존(유닛 단위 flat 랭킹) 동작으로 되돌아간다.
    """

    def __init__(
        self,
        vectorstore,
        k: int = 5,
        fetch_k: "int | None" = None,
        dedupe: bool = True,
    ) -> None:
        """
        Args:
            vectorstore: `similarity_search_with_score(query, k)`를 제공하는
                LangChain VectorStore(FAISS, Chroma 등).
            k: 최종 반환할 문서 수.
            fetch_k: dedupe 전에 벡터스토어에서 미리 가져올 후보 수. EU
                하나가 retrieval_units개로 쪼개져 있으므로 k보다 넉넉해야
                dedupe 후에도 k개가 채워진다. 기본값은 `max(k * 4, 20)`.
            dedupe: False면 max-pool 없이 기존 flat 랭킹 그대로 반환.
        """
        self.vectorstore = vectorstore
        self.k = k
        self.fetch_k = fetch_k or max(k * 4, 20)
        self.dedupe = dedupe

    def get_relevant_documents(self, query: str) -> list["LangChainDocument"]:
        results = self.vectorstore.similarity_search_with_score(query, k=self.fetch_k)
        if self.dedupe:
            results = dedupe_by_chunk_id(results, k=self.k)
        else:
            results = results[: self.k]
        return [doc for doc, _score in results]

    def invoke(self, query: str) -> list["LangChainDocument"]:
        """LangChain 신버전 Runnable 인터페이스와 이름을 맞춘 별칭."""
        return self.get_relevant_documents(query)
