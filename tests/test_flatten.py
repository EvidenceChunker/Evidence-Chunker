# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
flatten.py가 Docling과 완전히 무관함을 증명하는 테스트.

"""
from evidence_chunker.flatten import (
    build_col_header_map,
    build_row_header_map,
    build_table_abstract,
    detect_cell_marker,
    group_sentences_by_row,
    infer_headers_fallback,
)


def _simple_cells():
    return [
        {"start_row_offset_idx": 0, "start_col_offset_idx": 0, "column_header": True, "text": "Model"},
        {"start_row_offset_idx": 0, "start_col_offset_idx": 1, "column_header": True, "text": "Score"},
        {"start_row_offset_idx": 1, "start_col_offset_idx": 0, "row_header": True, "text": "YOLOv5"},
        {"start_row_offset_idx": 1, "start_col_offset_idx": 1, "text": "42.3*"},
        {"start_row_offset_idx": 2, "start_col_offset_idx": 0, "row_header": True, "text": "Faster R-CNN"},
        {"start_row_offset_idx": 2, "start_col_offset_idx": 1, "text": "38.1"},
    ]


def test_build_col_header_map():
    col_map = build_col_header_map(_simple_cells(), num_cols=2)
    assert col_map == {0: "Model", 1: "Score"}


def test_build_row_header_map():
    row_map = build_row_header_map(_simple_cells(), num_rows=3)
    assert row_map == {1: "YOLOv5", 2: "Faster R-CNN"}


def test_infer_headers_fallback_when_no_column_header():
    cells = [
        {"start_row_offset_idx": 0, "start_col_offset_idx": 0, "text": "5"},
        {"start_row_offset_idx": 0, "start_col_offset_idx": 1, "text": "Total"},
    ]
    fallback = infer_headers_fallback(cells, num_cols=2)
    assert fallback[0] == "Col0"  # 숫자만 있는 열은 Col{n}으로 대체
    assert fallback[1] == "Total"


def test_detect_cell_marker():
    assert detect_cell_marker("100*") == ("100", "*")
    assert detect_cell_marker("O(n²)") == ("O(n²)", None)


def test_group_sentences_by_row():
    # 값 셀은 col=1(Score 열)에 있으므로 "Score:"가 붙어야 함 — Model은 헤더 열(col=0)
    grouped = group_sentences_by_row(_simple_cells(), num_rows=3, num_cols=2)
    assert grouped[1] == ["YOLOv5 | Score: 42.3 [*]"]
    assert grouped[2] == ["Faster R-CNN | Score: 38.1"]


def test_build_table_abstract_includes_section_header_and_caption():
    abstract = build_table_abstract(
        caption_text="Table 2: Detection results",
        col_map={0: "Model", 1: "Score"},
        num_rows=3,
        section_header="4.1 Benchmarks",
    )
    assert abstract == "[4.1 Benchmarks] Table 2: Detection results Columns: Model, Score 2 data row(s)."


def test_no_docling_import():
    """실제 import 문에만 결합이 없는지 확인 (docstring에 "Docling"이 설명용으로
    등장하는 것과는 구분) — 이 모듈은 doc 파라미터를 안 받으므로 이걸로 충분."""
    import evidence_chunker.flatten as flatten_module
    import inspect

    for line in inspect.getsource(flatten_module).splitlines():
        assert "import docling" not in line and "from docling" not in line, line
