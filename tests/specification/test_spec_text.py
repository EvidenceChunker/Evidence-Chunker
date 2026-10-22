import pytest
from evidence_chunker.flatten import (
    build_col_header_map,build_row_header_map,infer_headers_fallback,
    group_sentences_by_row,build_table_abstract,
)
from ._data import cell

@pytest.mark.spec("QA-25")
def test_qa25_multilevel_columns_and_spanning_rows():
    cells=[
        cell("Period",0,1,column_header=True,col_end=3),
        cell("Q1",1,1,column_header=True),cell("Q2",1,2,column_header=True),
        cell("APAC",2,0,row_header=True,row_end=4),
    ]
    assert build_col_header_map(cells,3)=={1:"Period / Q1",2:"Period / Q2"}
    assert build_row_header_map(cells,4)=={2:"APAC",3:"APAC"}

@pytest.mark.spec("QA-26")
@pytest.mark.parametrize("values,expected",[
    (["Name","Score"],{0:"Name",1:"Score"}),
    (["10","20"],{0:"Col0",1:"Col1"}),
])
def test_qa26_unmarked_header_and_numeric_fallback(values,expected):
    assert infer_headers_fallback([cell(v,0,i) for i,v in enumerate(values)],2)==expected

@pytest.mark.spec("QA-27")
@pytest.mark.parametrize("with_row_header",[False,True])
def test_qa27_cell_text_with_and_without_row_header(with_row_header):
    cells=[cell("Score",0,1,column_header=True),cell("42",1,1)]
    if with_row_header:cells.append(cell("Model A",1,0,row_header=True))
    assert group_sentences_by_row(cells,2,2)=={1:[("Model A | " if with_row_header else "")+"Score: 42"]}

@pytest.mark.spec("QA-28")
@pytest.mark.parametrize("note,expected_suffix",[(None,"[*]"),("Measured once","[Measured once]")])
def test_qa28_footnote_text_or_marker_attached(note,expected_suffix):
    cells=[cell("Score",0,1,column_header=True),cell("42*",1,1),cell("43",2,1)]
    rows=group_sentences_by_row(cells,3,2,note)
    assert rows=={1:[f"Score: 42 {expected_suffix}"],2:["Score: 43"]}

@pytest.mark.spec("QA-29")
def test_qa29_abstract_is_deterministic():
    args=dict(caption_text="Table 1: Values",col_map={0:"Name",1:"Score"},num_rows=3,section_header="Results")
    expected="[Results] Table 1: Values Columns: Name, Score 2 data row(s)."
    assert build_table_abstract(**args)==expected
    assert build_table_abstract(**args)==expected

