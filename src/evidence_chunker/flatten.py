# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
flatten.py

표 → 자연어 문장/요약 변환 유틸리티.

기능:
  - build_col_header_map  : 다단/병합 헤더 열 매핑 (col_span 처리)
  - build_row_header_map  : 행헤더 행 매핑 (row_span 처리)
  - infer_headers_fallback: Docling이 헤더 태깅 못 했을 때 휴리스틱 추론
  - flatten_to_sentences  : 셀 → "행헤더 | 열헤더: 값" 구조화 문자열 (임베딩용)
  - detect_cell_marker    : 셀 내 각주 마커(*†‡) 감지
  - build_table_abstract  : 표 전체 요약 문자열 (multi-granularity 검색용)
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

# 각주 마커 패턴 (유니코드 포함)
_MARKER_RE = re.compile(r'[*†‡※#]')


def build_col_header_map(cells: list[dict], num_cols: int) -> dict[int, str]:
    """
    column_header=True 셀을 수집해 col_index → 헤더 텍스트 매핑 반환.
    col_span > 1 이면 해당 범위 전체에 텍스트 기록.
    다단 헤더(여러 행에 걸친 헤더)는 ' / '로 이어붙임.

    예:
        상위 헤더 "분기" (col_span=2) + 하위 헤더 "Q1", "Q2"
        → col_map = {0: "분기 / Q1", 1: "분기 / Q2"}
    """
    col_map: dict[int, str] = {}

    for cell in cells:
        if not cell.get("column_header"):
            continue
        text = cell.get("text", "").strip()
        if not text:
            continue
        c_start = cell.get("start_col_offset_idx", 0)
        c_end   = cell.get("end_col_offset_idx", c_start + 1)

        for col in range(c_start, min(c_end, num_cols)):
            if col in col_map:
                col_map[col] = col_map[col] + " / " + text
            else:
                col_map[col] = text

    return col_map


def build_row_header_map(cells: list[dict], num_rows: int) -> dict[int, str]:
    """
    row_header=True 셀을 수집해 row_index → 헤더 텍스트 매핑 반환.
    row_span > 1 이면 해당 범위 전체 행에 동일 텍스트 기록.

    예:
        "APAC" (row_span=3) → rows 2, 3, 4 모두 row_map[r] = "APAC"
    """
    row_map: dict[int, str] = {}

    for cell in cells:
        if not cell.get("row_header"):
            continue
        text = cell.get("text", "").strip()
        if not text:
            continue
        r_start = cell.get("start_row_offset_idx", 0)
        r_end   = cell.get("end_row_offset_idx", r_start + 1)

        for row in range(r_start, min(r_end, num_rows)):
            if row not in row_map:
                row_map[row] = text

    return row_map


def infer_headers_fallback(cells: list[dict], num_cols: int) -> dict[int, str]:
    """
    column_header 태깅이 전혀 없을 때 첫 번째 행을 헤더로 추정.
    첫 행이 전부 숫자면 Col0, Col1, ... 로 대체.
    """
    first_row = sorted(
        [c for c in cells if c.get("start_row_offset_idx", -1) == 0],
        key=lambda x: x.get("start_col_offset_idx", 0),
    )

    def _is_numeric(text: str) -> bool:
        return bool(re.match(r'^[\d\s.,+\-±%()]+$', text.strip()))

    col_map: dict[int, str] = {}
    for cell in first_row:
        col = cell.get("start_col_offset_idx", 0)
        text = cell.get("text", "").strip()
        col_map[col] = text if (text and not _is_numeric(text)) else f"Col{col}"

    return col_map


def detect_cell_marker(text: str) -> tuple[str, Optional[str]]:
    """
    셀 값에서 각주 마커(*†‡ 등)를 분리.
    반환: (cleaned_text, marker_or_None)

    예: "100*" → ("100", "*")
        "O(n²)" → ("O(n²)", None)
    """
    m = _MARKER_RE.search(text)
    if m:
        cleaned = _MARKER_RE.sub("", text).strip()
        return cleaned, m.group()
    return text.strip(), None


def group_sentences_by_row(
    cells: list[dict],
    num_rows: int,
    num_cols: int,
    footnote_text: Optional[str] = None,
) -> dict[int, list[str]]:
    """
    표 셀 데이터를 자연어 문장으로 변환하되, 원본 표의 행 인덱스
    (start_row_offset_idx)별로 묶어서 반환.

    split.py가 표를 행 단위로 분할할 때, 분할된 조각에 해당 행의
    flattened_rows 문장만 함께 실어 보내기 위해 사용한다.

    Returns:
        {row_offset_idx: [문장, ...]}  — 헤더 행은 포함되지 않음
    """
    col_map = build_col_header_map(cells, num_cols)
    row_map = build_row_header_map(cells, num_rows)

    if not col_map:
        col_map = infer_headers_fallback(cells, num_cols)

    grouped: dict[int, list[str]] = defaultdict(list)

    for cell in cells:
        if cell.get("column_header") or cell.get("row_header"):
            continue

        raw_value = cell.get("text", "").strip()
        if not raw_value:
            continue

        row = cell.get("start_row_offset_idx", 0)
        col = cell.get("start_col_offset_idx", 0)

        value, marker = detect_cell_marker(raw_value)
        if not value:
            continue

        col_header = col_map.get(col, f"Col{col}")
        row_header = row_map.get(row)

        # 언어 무관 key:value 포맷 — 각 셀을 검색 가능한 짧은 단위로 쪼개는 게
        # 목적이라 문법 문장("~의 ~는 ~이다")보다 이쪽이 임베딩에 유리하다.
        label_parts = [p for p in (row_header, col_header) if p]
        sentence = f"{' | '.join(label_parts)}: {value}"
        if marker:
            note = footnote_text.strip() if footnote_text else marker
            sentence += f" [{note}]"

        grouped[row].append(sentence)

    return dict(grouped)


def flatten_to_sentences(
    cells: list[dict],
    num_rows: int,
    num_cols: int,
    footnote_text: Optional[str] = None,
) -> list[str]:
    """
    표 셀 데이터를 자연어 문장 목록으로 변환 (행 순서대로 이어붙임).
    행별로 묶인 결과가 필요하면 group_sentences_by_row() 사용.
    """
    grouped = group_sentences_by_row(cells, num_rows, num_cols, footnote_text)
    return [sentence for row in sorted(grouped) for sentence in grouped[row]]


def build_table_abstract(
    caption_text: Optional[str],
    col_map: dict[int, str],
    num_rows: int,
    section_header: Optional[str] = None,
) -> str:
    """
    LLM 없이 규칙 기반으로 표 요약 생성.
    광범위 질의("지역별 매출 표가 어디 있어?")에 대한 검색용.

    포맷(언어 무관 구조 라벨): "[섹션헤더] > [캡션]. Columns: [열헤더 목록]. [N] data row(s)."
    """
    parts: list[str] = []

    if section_header:
        parts.append(f"[{section_header}]")

    if caption_text:
        parts.append(caption_text)

    cols = [v for v in col_map.values() if v]
    if cols:
        parts.append("Columns: " + ", ".join(cols[:6]) + ("..." if len(cols) > 6 else ""))

    data_rows = max(0, num_rows - 1)  # 헤더 행 제외
    if data_rows > 0:
        parts.append(f"{data_rows} data row(s).")

    return " ".join(parts)
