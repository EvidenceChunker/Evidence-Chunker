import copy
import pytest
from evidence_chunker import EvidenceUnit
from evidence_chunker.chunker import build_evidence_units
from evidence_chunker.caption import map_table_caption
from evidence_chunker.context import attach_context_paragraphs, find_section_header
from evidence_chunker.filters import find_duplicate_tables, is_toc_or_lof_decoy
from ._data import parsed, table, text, cell

@pytest.mark.spec("QA-09")
@pytest.mark.parametrize("shared,duplicates",[(2,False),(3,True)])
def test_qa09_duplicate_threshold_and_caption_inheritance(shared,duplicates):
    a,b,c=table(0,rows=2),table(1,rows=4),table(2,rows=2)
    first=["alpha","bravo","charlie","delta","echo"]
    second=first[:shared]+["foxtrot","golf","hotel"][:5-shared]
    a.cells=[cell(" ".join(first),1,0)]
    b.cells=[cell(" ".join(second),1,0)]
    c.cells=[cell("india juliet kilo lima mike",1,0)]
    # Direct reference on a remote page prevents winner bbox fallback from hiding donor logic.
    ts=[text(0,"Table 1: Donor caption",label="caption",page=3)]
    a.caption_refs=[0]
    p=parsed([a,b,c],ts)
    assert find_duplicate_tables(p)==({0:1} if duplicates else {})
    eus=build_evidence_units(p,doc_id="dedup")
    assert {e.table_index for e in eus}==({1,2} if duplicates else {0,1,2})
    if duplicates:
        assert next(e for e in eus if e.table_index==1).caption_text=="Table 1: Donor caption"

@pytest.mark.spec("QA-10")
@pytest.mark.parametrize("page,header,values,expected",[
    (5,"Contents",["1","2","3"],True),
    (6,"Contents",["1","2","3"],False),
    (1,"Results",["1","2","3"],False),
    (1,"Contents",["one","two","3"],False),
])
def test_qa10_toc_filter_and_nonmatches(page,header,values,expected):
    t=table(page=page,rows=3)
    for row,value in enumerate(values,1):
        t.cells[2*row+1]["text"]=value
    p=parsed([t],[text(0,header,label="section_header",page=page,center=100)])
    p.page_sizes[page]=(600,1000)
    assert is_toc_or_lof_decoy(p,t) is expected
    eus=build_evidence_units(p)
    assert len(eus)==(0 if expected else 1)

@pytest.mark.spec("QA-13")
@pytest.mark.parametrize("distance,expected",[(199,"inferred"),(200,"inferred"),(201,"none")])
def test_qa13_caption_distance_boundary(distance,expected):
    t=table(top=450,bottom=550)
    p=parsed([t],[text(0,"Table 1: Distance boundary",label="caption",center=450-distance)])
    assert map_table_caption(p,t,0).confidence==expected

@pytest.mark.spec("QA-15")
@pytest.mark.parametrize("label,expected",[("Table 1: Embedded caption","inferred"),("Ordinary group heading","none")])
def test_qa15_merged_first_row_caption(label,expected):
    t=table()
    t.html=f'<table><tr><td colspan="2">{label}</td></tr><tr><td>A</td><td>10</td></tr></table>'
    assert map_table_caption(parsed([t]),t,0).confidence==expected

@pytest.mark.spec("QA-15")
@pytest.mark.spec_gap
def test_qa15_caption_must_be_in_first_row():
    t=table()
    t.html='<table><tr><th>Name</th><th>Value</th></tr><tr><td colspan="2">Table 9: Quoted data</td></tr></table>'
    assert map_table_caption(parsed([t]),t,0).caption_text is None, "GAP-02: first-row requirement not enforced"

@pytest.mark.spec("QA-17")
@pytest.mark.parametrize("caption",["Fig. 1: example","Figure 2: example","그림 3: 예시"])
def test_qa17_safe_caption_preserves_original(caption):
    e=EvidenceUnit(eu_id="x",page_no=1,caption_text=caption)
    assert e.safe_caption is None
    assert e.caption_text==caption

@pytest.mark.spec("QA-18")
def test_qa18_figure_caption_fallback_and_safe_abstract():
    cap="Figure 1: A diagram"
    e=EvidenceUnit(eu_id="x",page_no=1,caption_text=cap,
                   table_abstract=cap+" Columns: Value",context_before=["Original paragraph"])
    assert e.context_before_with_fallback==[cap,"Original paragraph"]
    assert cap not in e.safe_abstract
    assert e.context_before==["Original paragraph"]

@pytest.mark.spec("QA-20")
@pytest.mark.parametrize("direction",["above","below"])
@pytest.mark.parametrize("distance,keep",[(299,True),(300,True),(301,False)])
def test_qa20_context_distance_boundary(direction,distance,keep):
    t=table(top=400,bottom=500)
    center=400-distance if direction=="above" else 500+distance
    p=parsed([t],[text(0,"Context candidate",center=center)])
    e=EvidenceUnit(eu_id="x",page_no=1)
    attach_context_paragraphs(e,p,table_bbox=t.bbox)
    selected=e.context_before if direction=="above" else e.context_after
    assert selected==(["Context candidate"] if keep else [])

@pytest.mark.spec("QA-21")
@pytest.mark.parametrize("direction",["above","below"])
def test_qa21_same_page_section_boundary(direction):
    t=table()
    para_center,header_center=(250,300) if direction=="above" else (650,600)
    ts=[text(0,"Other section paragraph",center=para_center),
        text(1,"Other section",label="section_header",center=header_center)]
    e=EvidenceUnit(eu_id="x",page_no=1)
    attach_context_paragraphs(e,parsed([t],ts),table_bbox=t.bbox)
    assert "Other section paragraph" not in e.context_before+e.context_after

@pytest.mark.spec("QA-22")
@pytest.mark.parametrize("top,bottom,expected",[
    (150,400,["previous"]), (151,400,[]),
    (400,850,["next"]), (400,849,[]),
])
def test_qa22_adjacent_page_context_gate(top,bottom,expected):
    t=table(page=2,top=top,bottom=bottom)
    ts=[text(0,"previous",page=1,center=950),text(1,"next",page=3,center=50)]
    e=EvidenceUnit(eu_id="x",page_no=2)
    attach_context_paragraphs(e,parsed([t],ts),table_bbox=t.bbox)
    assert e.context_before+e.context_after==expected
    assert e.page_span==({1,2} if expected==["previous"] else {2,3} if expected else {2})

@pytest.mark.spec("QA-23")
def test_qa23_same_page_section_and_no_header():
    ts=[text(0,"Old",label="section_header",center=100),
        text(1,"Current",label="section_header",center=300)]
    assert find_section_header(parsed(texts=ts),1,400)=="Current"
    assert find_section_header(parsed(),1,400) is None

@pytest.mark.spec("QA-23")
@pytest.mark.spec_gap
def test_qa23_previous_page_last_header_is_selected():
    ts=[text(0,"Earlier section",label="section_header",page=1,center=100),
        text(1,"Continuing section",label="section_header",page=1,center=800)]
    assert find_section_header(parsed(texts=ts),2,100)=="Continuing section", "GAP-03: previous-page heading ordering"

