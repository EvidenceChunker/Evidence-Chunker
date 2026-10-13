# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
export

EvidenceUnit + 일반 본문 청크를 균일하게 다루기 위한 공통 인터페이스.

EvidenceChunker.build_corpus()는 EvidenceUnit(표)와 TextChunk(HybridChunker가
만든 일반 본문 청크)를 한 리스트에 섞어 반환한다. 두 타입 모두 RetrievalChunk
프로토콜(chunk_id/text/retrieval_text/retrieval_units/metadata)을 만족하므로,
export.langchain/export.llamaindex의 to_*() 함수가 타입을 구분하는 isinstance
분기 없이 두 타입을 동일하게 처리할 수 있다.
"""
from __future__ import annotations

from typing import Protocol


# 타입 힌트 전용 프로토콜 — isinstance()로 검사하지 않는다. 데이터 속성만
# 가진 Protocol은 런타임 검사를 지원하지 않고, 애초에 to_*() 함수들이 타입
# 분기 없이 두 타입을 그대로 처리하는 것이 이 프로토콜의 목적이다.
class RetrievalChunk(Protocol):
    chunk_id: str
    is_atomic: bool
    text: str
    retrieval_text: str
    retrieval_units: list[str]
    metadata: dict


class TextChunk:
    """
    표와 무관한 일반 본문 청크(Docling HybridChunker가 만든 chunk)를
    RetrievalChunk 프로토콜에 맞춰 감싼 래퍼.

    is_atomic=True가 EvidenceUnit과의 실질적 차이 — export의
    to_*_units() 함수들이 이 값으로 "쪼갤 대상인지"를 구분한다(일반
    청크는 그 자체로 최소 단위라 쪼갤 게 없음).
    """

    def __init__(self, chunk, doc_id: str, index: int) -> None:
        self.text = chunk.text
        self.retrieval_text = chunk.text
        self.retrieval_units = [chunk.text]
        self.chunk_id = f"{doc_id}-hybrid-{index}"
        self.is_atomic = True

        page_no = None
        try:
            page_no = chunk.meta.doc_items[0].prov[0].page_no
        except (AttributeError, IndexError):
            pass
        self.metadata = {
            "chunk_id": self.chunk_id,  # EvidenceUnit.metadata와 필드명 통일(dedupe_by_chunk_id용)
            "eu_id": None,
            "doc_id": doc_id,
            "page_no": page_no,
            "source": "hybrid",
        }


# attach_context_paragraphs()가 표 주변 단락을 eu.context_before/after로
# EU에 편입시켜도 원본 doc.texts에서는 그 단락을 지우지 않는다 — 그래서
# 별도로 HybridChunker를 돌리면 EU가 이미 흡수한 문단이 일반 청크로도
# 중복 등장해 검색 코퍼스 안에서 서로 경쟁한다(카니발라이제이션).

def filter_consumed_paragraphs(
    chunks: list,
    eu_list: list,
    min_substring_len: int = 20,
) -> list:
    """
    HybridChunker 등이 만든 일반 청크 중, 이미 EU의 context_before/
    context_after로 흡수된 문단과 겹치는(카니발라이제이션) 청크를 제거.

    Args:
        chunks: 텍스트를 가진 청크/Document/Node 리스트. page_content
            (LangChain Document) 또는 text(LlamaIndex TextNode, Docling
            HybridChunker 청크) 속성 중 하나를 가지고 있어야 함.
        eu_list: EvidenceChunker가 만든 EvidenceUnit 리스트.
        min_substring_len: 이 길이 미만인 문자열은 완전 일치만 적용하고
            부분 포함 매칭은 건너뛴다 — 짧은 각주/라벨이 무관한 긴 청크에
            우연히 포함돼 오탐되는 걸 방지.

    Returns:
        카니발라이제이션 청크가 제거된 리스트 (원본 순서 유지).
    """
    def _normalize(text: str) -> str:
        return " ".join((text or "").split())

    consumed_texts = [
        _normalize(p)
        for eu in eu_list
        for p in eu.context_before + eu.context_after
        if p.strip()
    ]

    filtered_chunks = []
    for c in chunks:
        c_text = _normalize(getattr(c, "page_content", None) or getattr(c, "text", None) or "")
        if not c_text:
            continue

        is_consumed = False
        for consumed in consumed_texts:
            if c_text == consumed:
                is_consumed = True
                break
            shorter_len = min(len(c_text), len(consumed))
            if shorter_len < min_substring_len:
                continue
            if c_text in consumed or consumed in c_text:
                is_consumed = True
                break

        if not is_consumed:
            filtered_chunks.append(c)

    return filtered_chunks
