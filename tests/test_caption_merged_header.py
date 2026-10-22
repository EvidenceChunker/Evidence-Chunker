# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""표 첫 행 병합 셀 캡션 탐색 회귀 테스트 (QA-15 / GAP-02).

이 fallback은 캡션이 표 안으로 빨려 들어간 경우를 복구하는 규칙이고,
캡션은 표 위에 붙으므로 표 HTML 안에서는 첫 행으로만 나타난다. 따라서
"첫 행"이라는 위치가 이 추론의 유일한 근거다. 이전 구현은 HTML 전체를
훑어서 표 중간의 병합 셀도 캡션으로 채택했다.
"""
import pytest

from evidence_chunker.caption import _find_caption_in_merged_header

CAPTION_ROW = '<tr><td colspan="2">Table 1: Embedded caption</td></tr>'
HEADER_ROW = "<tr><th>Name</th><th>Value</th></tr>"
MID_CAPTION_ROW = '<tr><td colspan="2">Table 9: Quoted data</td></tr>'
DATA_ROW = "<tr><td>A</td><td>10</td></tr>"


def _table(*rows):
    return "<table>" + "".join(rows) + "</table>"


def test_first_row_merged_caption_is_adopted():
    html = _table(CAPTION_ROW, DATA_ROW)
    assert _find_caption_in_merged_header(html) == "Table 1: Embedded caption"


def test_merged_cell_below_first_row_is_ignored():
    """표 중간의 그룹 구분 행이나 다른 표를 인용한 셀을 캡션으로 쓰면 안 된다."""
    html = _table(HEADER_ROW, MID_CAPTION_ROW, DATA_ROW)
    assert _find_caption_in_merged_header(html) is None


def test_caption_row_inside_thead_is_examined():
    html = "<table><thead>" + CAPTION_ROW + "</thead><tbody>" + DATA_ROW + "</tbody></table>"
    assert _find_caption_in_merged_header(html) == "Table 1: Embedded caption"


def test_non_caption_text_in_first_row_is_ignored():
    html = _table('<tr><td colspan="2">Ordinary group heading</td></tr>', DATA_ROW)
    assert _find_caption_in_merged_header(html) is None


def test_first_row_without_colspan_is_ignored():
    html = _table(HEADER_ROW, DATA_ROW)
    assert _find_caption_in_merged_header(html) is None


def test_markup_inside_merged_cell_is_stripped():
    html = _table('<tr><td colspan="3"><b>Table 2:</b> Nested markup</td></tr>', DATA_ROW)
    result = _find_caption_in_merged_header(html)
    assert result is not None
    assert result.startswith("Table 2:") and "Nested markup" in result
    assert "<b>" not in result


@pytest.mark.parametrize("html", [None, "", "<p>표가 아닌 마크업</p>"])
def test_missing_or_rowless_html_returns_none(html):
    assert _find_caption_in_merged_header(html) is None
