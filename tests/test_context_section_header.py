# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""find_section_header()의 섹션 제목 선택 회귀 테스트.

TOPLEFT 좌표계에서는 cy가 클수록 페이지 아래쪽이다. 표가 속한 섹션은 표
바로 앞에 나온 마지막 제목이므로, 이전 페이지에서 제목을 찾을 때는 cy가
가장 큰(= 가장 아래) 제목을 골라야 한다. 이전 구현은 cy 최솟값을 골라
페이지 맨 위 제목을 반환했다(QA-23 / GAP-03).
"""
from evidence_chunker.context import find_section_header
from evidence_chunker.parser.base import BBox, BlockLabel, ParsedDoc, TextBlock

PAGE_SIZE = (600.0, 1000.0)


def _header(index, text, page, cy):
    return TextBlock(
        index, text, BlockLabel.SECTION_HEADER, page,
        BBox(20.0, cy - 5.0, 280.0, cy + 5.0),
    )


def _parsed(texts):
    return ParsedDoc(texts, [], set(), {p: PAGE_SIZE for p in (1, 2, 3)})


def test_previous_page_picks_bottom_most_header():
    """이전 페이지에서는 아래쪽(cy 최대) 제목이 표의 섹션이다."""
    texts = [
        _header(0, "Earlier section", 1, 100.0),
        _header(1, "Continuing section", 1, 800.0),
    ]
    assert find_section_header(_parsed(texts), 2, 100.0) == "Continuing section"


def test_nearest_previous_page_wins():
    """이전 페이지가 여러 개면 가장 가까운 페이지의 마지막 제목을 쓴다."""
    texts = [
        _header(0, "Page one tail", 1, 900.0),
        _header(1, "Page two head", 2, 150.0),
        _header(2, "Page two tail", 2, 700.0),
    ]
    assert find_section_header(_parsed(texts), 3, 120.0) == "Page two tail"


def test_same_page_header_takes_precedence_over_previous_page():
    texts = [
        _header(0, "Previous page tail", 1, 900.0),
        _header(1, "Current section", 2, 300.0),
    ]
    assert find_section_header(_parsed(texts), 2, 400.0) == "Current section"


def test_same_page_nearest_above_is_selected():
    texts = [
        _header(0, "Far above", 1, 100.0),
        _header(1, "Just above", 1, 350.0),
    ]
    assert find_section_header(_parsed(texts), 1, 400.0) == "Just above"


def test_header_below_table_on_same_page_is_ignored():
    texts = [_header(0, "Below the table", 1, 600.0)]
    assert find_section_header(_parsed(texts), 1, 400.0) is None


def test_no_header_returns_none():
    assert find_section_header(_parsed([]), 1, 400.0) is None


def test_unpaged_header_is_ignored():
    texts = [_header(0, "Floating header", -1, 50.0)]
    assert find_section_header(_parsed(texts), 2, 400.0) is None
