# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
parser.base.TableBlock의 헬퍼(num_data_rows, last_column_values)를
합성 데이터로 검증 — PDF 파싱 없이 도는 빠른 회귀 테스트.

num_data_rows()의 다단 헤더 케이스는 실제 PDF(docling_technical_report.pdf
table[0])에서 발견된 실패 사례를 그대로 옮긴 것: 헤더가 2행인 표에서
좌상단 모서리 셀("CPU")이 column_header=False로 찍혀 있어, 셀 단위로
헤더를 판정하면 그 행이 데이터 행으로 잘못 세어져 3이 나왔다(pandas
export_to_dataframe()는 2). 행 단위 판정(그 행에 column_header 셀이
하나라도 있으면 헤더 행)으로 바꿔서 고쳤다.
"""
from evidence_chunker.parser.base import TableBlock, BBox


def _multi_header_table_cells():
    return [
        # 1행 헤더: 모서리 셀만 column_header=False (실제 버그 재현)
        {"start_row_offset_idx": 0, "start_col_offset_idx": 0, "column_header": False, "text": "CPU"},
        {"start_row_offset_idx": 0, "start_col_offset_idx": 1, "column_header": True, "text": "Thread budget"},
        {"start_row_offset_idx": 0, "start_col_offset_idx": 2, "column_header": True, "text": "native backend"},
        {"start_row_offset_idx": 0, "start_col_offset_idx": 5, "column_header": True, "text": "pypdfium backend"},
        # 2행 헤더: 전부 column_header=True
        {"start_row_offset_idx": 1, "start_col_offset_idx": 2, "column_header": True, "text": "TTS"},
        {"start_row_offset_idx": 1, "start_col_offset_idx": 5, "column_header": True, "text": "TTS"},
        {"start_row_offset_idx": 1, "start_col_offset_idx": 7, "column_header": True, "text": "Mem"},
        # 데이터 2행
        {"start_row_offset_idx": 2, "start_col_offset_idx": 0, "row_header": True, "text": "Apple M3 Max"},
        {"start_row_offset_idx": 2, "start_col_offset_idx": 7, "text": "2.56 GB"},
        {"start_row_offset_idx": 3, "start_col_offset_idx": 0, "row_header": True, "text": "Intel Xeon"},
        {"start_row_offset_idx": 3, "start_col_offset_idx": 7, "text": "2.42 GB"},
    ]


def _table(cells, num_rows=4, num_cols=8):
    return TableBlock(
        index=0, page_no=1, bbox=BBox(0, 0, 100, 100),
        cells=cells, num_rows=num_rows, num_cols=num_cols, html=None,
    )


def test_num_data_rows_excludes_multi_row_header():
    table = _table(_multi_header_table_cells())
    assert table.num_data_rows() == 2


def test_num_data_rows_simple_single_header():
    cells = [
        {"start_row_offset_idx": 0, "start_col_offset_idx": 0, "column_header": True, "text": "Name"},
        {"start_row_offset_idx": 0, "start_col_offset_idx": 1, "column_header": True, "text": "Value"},
        {"start_row_offset_idx": 1, "start_col_offset_idx": 0, "text": "a"},
        {"start_row_offset_idx": 1, "start_col_offset_idx": 1, "text": "1"},
        {"start_row_offset_idx": 2, "start_col_offset_idx": 0, "text": "b"},
        {"start_row_offset_idx": 2, "start_col_offset_idx": 1, "text": "2"},
    ]
    table = _table(cells, num_rows=3, num_cols=2)
    assert table.num_data_rows() == 2


def test_last_column_values_excludes_header_cells():
    table = _table(_multi_header_table_cells())
    assert table.last_column_values() == ["2.56 GB", "2.42 GB"]
