# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
parser/base.py

Docling 파서에 종속되지 않는 독립적인 내부 문서 데이터 모델 및 파서 프로토콜(Protocol)을 정의한다.

flatten.py/caption.py/context.py/filters.py 등 하위 알고리즘 모듈은 이 모듈이
정의하는 추상화(ParsedDoc/TextBlock/TableBlock)만 바라보고, DoclingDocument
객체를 직접 다루지 않는다 — 파서를 교체하거나 버전을 올려도 이 모듈의
from_doc()/변환 함수만 손보면 되도록 격리한다. TextBlock의 최소 필드
구성(label/text/page_no/bbox)은 test_caption_exceptions.py가 쓰던 프로토콜을
그대로 정식 모델로 승격시킨 것이다.

파서 경계에서 세 가지를 변환한다:
1. 좌표계를 BOTTOMLEFT(Docling 원본)에서 TOPLEFT로 정규화 — 이후 모든
   알고리즘이 "위쪽 = y값이 작다"는 화면 좌표계 직관을 그대로 쓸 수 있게.
2. 텍스트 참조 문자열("#/texts/3")을 정수 인덱스로 치환 — 매번 문자열을
   파싱하지 않고 가벼운 인덱스 비교만으로 처리하기 위함.
3. 파서가 내놓는 다양한 라벨을 우리가 실제로 구분해서 쓰는 5종(text/
   list_item/paragraph/section_header/caption)으로 축소.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, Optional, Protocol


class BlockLabel(str, Enum):
    TEXT = "text"
    LIST_ITEM = "list_item"
    PARAGRAPH = "paragraph"
    SECTION_HEADER = "section_header"
    CAPTION = "caption"
    OTHER = "other"  # 위 5종 외 Docling 라벨(picture, table 등)은 전부 여기로


class BBox(NamedTuple):
    """항상 TOPLEFT 정규화: 시각적으로 위쪽일수록 t(top)가 작다."""
    l: float
    t: float
    r: float
    b: float


@dataclass
class TextBlock:
    index: int          # ParsedDoc.texts 안에서의 위치. cref/self_ref를 대체하는 정수 식별자.
    text: str
    label: BlockLabel
    page_no: int
    bbox: BBox


@dataclass
class TableBlock:
    index: int                          # ParsedDoc.tables 안에서의 위치.
    page_no: int
    bbox: BBox
    cells: list[dict]                   # Docling table_cells 형태 그대로 유지
                                         # (column_header/row_header/start_row_offset_idx/
                                         # end_row_offset_idx/start_col_offset_idx/
                                         # end_col_offset_idx/text). flatten.py가 이미
                                         # 이 dict 형태로만 동작하므로 재설계 범위 밖.
    num_rows: int
    num_cols: int
    html: Optional[str]                 # export_to_html() 결과, 파싱 시점에 미리 계산
    caption_refs: list[int] = field(default_factory=list)   # texts 인덱스
    footnote_refs: list[int] = field(default_factory=list)  # texts 인덱스

    def num_data_rows(self) -> int:
        """헤더 행을 제외한 데이터 행 개수. export_to_dataframe().shape[0]과
        같은 기준 — last_column_values()는 빈 텍스트 행을 걸러내서 개수가
        줄어들 수 있으므로(마지막 열이 빈 행이 있는 경우), min_rows류
        게이트에는 이쪽을 쓸 것.

        행 단위로 헤더를 판정한다(그 행에 column_header 셀이 하나라도
        있으면 헤더 행) — 셀 단위로 판정하면 다단 헤더 표의 좌상단 모서리
        셀이 column_header=False로 찍힌 경우를 데이터 행으로 잘못 센다.
        """
        all_rows: dict[int, bool] = {}
        for c in self.cells:
            row = c.get("start_row_offset_idx")
            if row is None:
                continue
            all_rows[row] = all_rows.get(row, False) or bool(c.get("column_header"))
        return sum(1 for is_header in all_rows.values() if not is_header)

    def last_column_values(self) -> list[str]:
        """마지막 열의 셀 값(행 순서). filters.is_toc_or_lof_decoy 전용.

        Docling의 export_to_dataframe()(pandas 의존) 대신 cells에서 직접
        뽑는다 — 이 메서드가 파서 추상화의 유일한 소비처라, pandas가 알고리즘
        레이어에 새는 걸 막는다.

        헤더 셀(column_header=True)은 제외 — export_to_dataframe()가 첫 행을
        컬럼명으로 쓰고 데이터에서 빼는 것과 동일한 기준.
        """
        last_col = self.num_cols - 1
        by_row: dict[int, str] = {}
        for cell in self.cells:
            if cell.get("column_header"):
                continue
            if cell.get("start_col_offset_idx", -1) != last_col:
                continue
            row = cell.get("start_row_offset_idx", -1)
            text = (cell.get("text") or "").strip()
            if text:
                by_row[row] = text
        return [by_row[r] for r in sorted(by_row)]


@dataclass
class ParsedDoc:
    texts: list[TextBlock]
    tables: list[TableBlock]
    picture_caption_refs: set[int]              # doc.pictures[*].captions -> texts 인덱스 집합
    page_sizes: dict[int, tuple[float, float]]  # page_no -> (width, height)


class PdfParser(Protocol):
    def parse(self, path: str) -> ParsedDoc: ...
