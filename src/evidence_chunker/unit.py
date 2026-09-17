"""
unit.py

검색 단위 EvidenceUnit 데이터클래스 정의.

표 하나에서 뽑은 캡션 / 표 데이터 / 인접 설명 단락 / 각주를 한 덩어리로
들고, 용도별로 세 가지 텍스트 뷰를 property로 제공한다.

    text            : LLM 컨텍스트용 전체 텍스트 (table_html 포함)
    retrieval_text  : 임베딩 검색용 (table_html 제외 — 토큰 예산 절약)
    retrieval_units : 행 단위 다중 벡터(small-to-big)용 검색 단위 목록

표가 아닌 일반 본문 청크는 export.TextChunk가 담당한다. 두 타입 모두
export.RetrievalChunk 프로토콜을 만족하므로 export의 to_*() 함수들이
구분 없이 다룰 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# EvidenceUnit
# ---------------------------------------------------------------------------

@dataclass
class EvidenceUnit:
    """
    표 + 캡션 + 설명단락을 하나로 묶은 검색 단위. 항상 표 기반 EU다.

    text/retrieval_text/retrieval_units는 자동 계산되는 property이므로
    직접 값을 넣지 말 것(각자의 docstring 참고).
    """

    # ------------------------------------------------------------------
    # 식별
    # ------------------------------------------------------------------
    eu_id: str          # "{doc_id}-p{page}-{idx}" 형식
    page_no: int
    doc_id: str = ""    # eu_id 접두사. 여러 PDF를 합칠 때 충돌 방지용

    # parsed.tables 안에서의 전역 위치(parser.base.TableBlock.index와 동일
    # 값). QA 생성기가 정답 표를 meta.table_index로 태깅해두므로, 이 값을
    # 그대로 보존해야 "정답 표 자체를 검색했는가"를 리트리벌 단계에서
    # 판별할 수 있다(페이지 일치만으로는 같은 페이지의 다른 표와 구분 불가).
    table_index: Optional[int] = None

    # 표와 연결된 모든 페이지. context가 인접 페이지에서 가져온 경우 포함.
    page_span: set[int] = field(default_factory=set)

    # ------------------------------------------------------------------
    # 내용
    # ------------------------------------------------------------------
    section_header: Optional[str] = None
    caption_text: Optional[str] = None
    table_html: Optional[str] = None
    footnote_text: Optional[str] = None

    # bbox 거리로 수집(기본, sim_threshold=0.0). sim_threshold>0으로 올리면
    # 코사인 유사도 필터도 함께 적용된 표 위/아래쪽 단락.
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)   # 아래쪽(위와 동일 기준)

    # 셀을 "행헤더 | 열헤더: 값" 자연어 문장으로 변환 (flatten.flatten_to_sentences()).
    flattened_rows: list[str] = field(default_factory=list)

    # 행별 문장 맵(내부 전용) — split.py가 표 분할 시 조각별로 필요한 문장만 골라내는 데 씀.
    row_sentence_map: dict[int, list[str]] = field(default_factory=dict, repr=False)

    # 표 한 줄 요약(caption + 헤더로 자동 생성, LLM 미사용) — 광범위 질의용.
    table_abstract: Optional[str] = None

    # ------------------------------------------------------------------
    # 위치
    # ------------------------------------------------------------------
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # (x1, y1, x2, y2), 0~1 정규화, BOTTOMLEFT 유지(공개 API 계약).
    # 패키지 내부 좌표(parser.base.BBox 등)는 TOPLEFT라 서로 다르다 — 그
    # 값을 그대로 넣지 말고 반드시 geometry.normalize_bbox()를 거칠 것.

    # 512토큰 초과 시 행 단위로 분할된다(split.py).
    is_split: bool = False
    split_index: Optional[int] = None   # 0-based
    total_splits: Optional[int] = None

    # "direct"=RefItem 직접 연결, "inferred"=bbox 추정, "none"=캡션 없음
    caption_confidence: Literal["direct", "inferred", "none"] = "none"

    # caption_confidence만으로는 그림(Figure) 캡션이 표에 잘못 연결된 경우를
    # 못 걸러낸다 — 텍스트 패턴(fig/figure/그림)으로 한 번 더 검증하는
    # 2차 안전망(1차는 caption.py의 구조적 체크).
    @property
    def safe_caption(self) -> str | None:
        """캡션이 Fig/Figure/그림으로 시작하면 None(표 정체성으로 신뢰 안 함)."""
        if not self.caption_text:
            return None
        lower_cap = self.caption_text.strip().lower()
        if lower_cap.startswith(("fig", "figure", "그림")):
            return None
        return self.caption_text

    @property
    def safe_abstract(self) -> str | None:
        """table_abstract는 caption_text로 조립되므로, 캡션이 오염됐다면 같이 제거."""
        if not self.table_abstract:
            return None
        if self.caption_text and not self.safe_caption:
            return self.table_abstract.replace(self.caption_text, "").strip()
        return self.table_abstract

    # safe_caption에서 탈락한 그림 캡션도 키워드로서는 가치가 있어 문맥
    # 단락으로 강등해 살린다(표 정체성 prefix로는 안 씀).
    @property
    def context_before_with_fallback(self) -> list[str]:
        """context_before + safe_caption에서 탈락한 그림 캡션(맨 앞에 삽입)."""
        fallback = list(self.context_before)
        if self.caption_text and not self.safe_caption:
            fallback.insert(0, self.caption_text)
        return fallback

    @property
    def text(self) -> str:
        """LLM 컨텍스트용 전체 텍스트(table_html 포함). 표 정체성(caption)과
        핵심 문맥을 앞쪽에 배치해 임베딩 모델의 positional bias에 대응한다.
        """
        parts = [
            self.safe_caption,
            *self.context_before_with_fallback,
            self.safe_abstract,
            self.section_header,
            self.table_html,
            *self.flattened_rows,
            self.footnote_text,
            *self.context_after,
        ]
        return "\n".join(p for p in parts if p)

    @property
    def retrieval_text(self) -> str:
        """임베딩 검색용 텍스트. table_html은 토큰 제한(256 토큰 내외)을
        고려해 제외한다 — 전체 맥락이 필요하면 text를 쓸 것.
        """
        parts = [
            self.safe_caption,
            *self.context_before_with_fallback,
            self.safe_abstract,
            self.section_header,
            *self.flattened_rows,
            self.footnote_text,
            *self.context_after,
        ]
        return "\n".join(p for p in parts if p)

    @property
    def retrieval_units(self) -> list[str]:
        """표 전체 요약과 세부 문장(문맥/행/각주)을 분리한 검색 단위 목록 —
        특정 셀 질의가 표 전체 벡터에 묻히는 문제를 막는 small-to-big 패턴.
        매칭된 유닛이 속한 EU 전체(eu.text)를 돌려주면 된다.
        """
        units: list[str] = []

        summary = "\n".join(
            p for p in (self.safe_caption, self.safe_abstract, self.section_header) if p
        )
        if summary:
            units.append(summary)

        # 캡션을 prefix로 붙여 각 하위 유닛의 표 정체성을 유지
        prefix = f"[{self.safe_caption}] " if self.safe_caption else ""

        units.extend(f"{prefix}{para}" for para in self.context_before_with_fallback)
        units.extend(f"{prefix}{row}" for row in self.flattened_rows)

        if self.footnote_text:
            units.append(f"{prefix}{self.footnote_text}")

        units.extend(f"{prefix}{para}" for para in self.context_after)

        if not units:
            fallback = self.retrieval_text
            if fallback:
                units.append(fallback)

        return units

    # export.RetrievalChunk 프로토콜 구현 — export.TextChunk와 같은 속성
    # 집합을 노출해야 export 함수들이 isinstance 분기 없이 동작한다.
    @property
    def chunk_id(self) -> str:
        return self.eu_id

    @property
    def is_atomic(self) -> bool:
        """항상 False — EvidenceUnit은 retrieval_units 단위로 쪼갤 수 있는 대상.
        일반 본문 청크(export.TextChunk)만 True.
        """
        return False

    @property
    def metadata(self) -> dict:
        """LangChain/LlamaIndex 문서 메타데이터."""
        return {
            "chunk_id": self.chunk_id,  # export.TextChunk와 그룹핑 키 통일
            "eu_id": self.eu_id,
            "doc_id": self.doc_id,
            "page_no": self.page_no,
            "page_span": sorted(self.page_span) if self.page_span else [self.page_no],
            "table_index": self.table_index,
            "section_header": self.section_header,
            "caption_text": self.caption_text,
            "bbox": list(self.bbox),
            "is_split": self.is_split,
            "split_index": self.split_index,
            "total_splits": self.total_splits,
            "caption_confidence": self.caption_confidence,
            "retrieval_text": self.retrieval_text,
        }
