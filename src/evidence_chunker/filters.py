# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
filters.py

EU 생성 대상에서 표를 제외해야 하는 케이스 감지.

기능:
  - find_duplicate_tables : Docling이 표 1개를 TableItem 2개로 중복 감지하는
                            케이스 탐지
  - is_toc_or_lof_decoy   : 목차(ToC)/그림·표 목록(LoF)이 표로 오인식된 경우 감지

parser.base.ParsedDoc/TableBlock 기반 — Docling DoclingDocument를 직접
만지지 않는다.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parser.base import ParsedDoc, TableBlock

_DUP_TOKEN_RE = re.compile(r"[.,()%]+$")


def _cell_text_signature(cells: list[dict]) -> set[str]:
    """셀 텍스트를 정규화된 토큰 집합으로 변환 (중복 표 비교용)."""
    tokens: set[str] = set()
    for cell in cells:
        text = (cell.get("text") or "").strip()
        if not text:
            continue
        for line in re.split(r"[\n,]+", text):
            for tok in line.split():
                tok = _DUP_TOKEN_RE.sub("", tok).lower()
                if len(tok) >= 2:
                    tokens.add(tok)
    return tokens


def find_duplicate_tables(parsed: "ParsedDoc", overlap_ratio: float = 0.6) -> dict[int, int]:
    """
    같은 페이지 내 표 쌍의 셀 텍스트 중복도로 중복 TableItem 감지.

    Docling이 같은 물리적 표를 TableItem 2개로 나눠 인식하는 경우가
    있다(하나는 행이 뭉개진 채 캡션과 연결, 다른 하나는 행은 제대로
    분리됐지만 캡션이 없음). 셀 텍스트 토큰 집합의 포함비율(containment =
    |교집합| / min(|A|,|B|))이 overlap_ratio 이상이면 같은 표로 간주하고,
    행 수가 더 많은(더 세밀하게 구조화된) 쪽만 남긴다.

    Returns:
        {제거할 표의 parsed.tables 인덱스: 남길 표의 parsed.tables 인덱스}
    """
    pages: dict[int, list[int]] = defaultdict(list)
    for table in parsed.tables:
        pages[table.page_no].append(table.index)

    signatures: dict[int, set[str]] = {}
    row_counts: dict[int, int] = {}
    for table in parsed.tables:
        signatures[table.index] = _cell_text_signature(table.cells)
        row_counts[table.index] = table.num_rows

    drop_map: dict[int, int] = {}
    for idxs in pages.values():
        if len(idxs) < 2:
            continue
        for a_pos in range(len(idxs)):
            for b_pos in range(a_pos + 1, len(idxs)):
                a, b = idxs[a_pos], idxs[b_pos]
                if a in drop_map or b in drop_map:
                    continue
                sig_a, sig_b = signatures[a], signatures[b]
                if not sig_a or not sig_b:
                    continue
                containment = len(sig_a & sig_b) / min(len(sig_a), len(sig_b))
                if containment >= overlap_ratio:
                    loser, winner = (
                        (a, b) if row_counts[a] < row_counts[b] else (b, a)
                    )
                    drop_map[loser] = winner

    return drop_map


# is_toc_or_lof_decoy()의 판정 기준.
TOC_MAX_PAGE = 5            # 목차/표 목록은 문서 앞부분에만 나온다고 가정
TOC_NUMERIC_RATIO_THRESHOLD = 0.7   # 마지막 열이 대부분 숫자(= 페이지 번호)
TOC_MIN_ROWS = 3
TOC_HEADER_KEYWORDS = ("content", "list of table", "list of figure")


def _has_toc_like_header(parsed: "ParsedDoc", page_no: int) -> bool:
    """표 바로 앞(같은 페이지 또는 이전 페이지)에 ToC/LoF 계열 헤더가 있는지."""
    from .parser.base import BlockLabel

    for item in parsed.texts:
        if item.label != BlockLabel.SECTION_HEADER:
            continue
        if item.page_no not in (page_no, page_no - 1):
            continue
        text = item.text.strip().lower()
        if any(kw in text for kw in TOC_HEADER_KEYWORDS):
            return True
    return False


def is_toc_or_lof_decoy(
    parsed: "ParsedDoc",
    table: "TableBlock",
    max_page: int = TOC_MAX_PAGE,
    numeric_ratio_threshold: float = TOC_NUMERIC_RATIO_THRESHOLD,
    min_rows: int = TOC_MIN_ROWS,
) -> bool:
    """목차(ToC)나 그림/표 목록(LoF)이 표로 오인식된 경우 True.

    build_evidence_units()가 이 표를 EU 생성 대상에서 건너뛴다(스캔 중
    continue) — 일반 본문 청킹(_build_text_chunks()의 HybridChunker)은
    parsed.tables를 보지 않으므로 영향받지 않는다.
    """
    if table.page_no is None or table.page_no > max_page:
        return False

    if table.num_data_rows() < min_rows or table.num_cols == 0:
        return False

    last_col_values = table.last_column_values()
    if not last_col_values:
        return False

    numeric_ratio = sum(1 for v in last_col_values if v.isdigit()) / len(last_col_values)
    if numeric_ratio < numeric_ratio_threshold:
        return False

    return _has_toc_like_header(parsed, table.page_no)
