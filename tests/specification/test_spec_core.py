from collections import Counter
from types import SimpleNamespace
import pytest
from evidence_chunker import EvidenceChunker, EvidenceUnit
from evidence_chunker.chunker import build_evidence_units
from evidence_chunker.parser.docling import DoclingParser
from evidence_chunker.export import filter_consumed_paragraphs
from ._data import EU_KEYS, StaticParser, parsed, table, text, expected_sentences

@pytest.mark.spec("QA-02")
def test_qa02_default_constructor():
    c=EvidenceChunker()
    assert c.bbox_threshold == 300.0
    assert c.sim_threshold == 0.0
    assert c.parser is None

@pytest.mark.spec("QA-03")
def test_qa03_artifacts_path_is_stored(tmp_path):
    path=str(tmp_path/"models")
    assert EvidenceChunker(artifacts_path=path).artifacts_path == path

@pytest.mark.spec("QA-04")
def test_qa04_parameters_change_context_selection():
    t=table()
    p=parsed([t],[text(0,"near paragraph",center=350),text(1,"far paragraph",center=150)])
    narrow=EvidenceChunker(parser=StaticParser(p),bbox_threshold=100,sim_threshold=0)
    wide=EvidenceChunker(parser=StaticParser(p),bbox_threshold=300,sim_threshold=0)
    a=narrow.chunk("demo.pdf")[0]
    b=wide.chunk("demo.pdf")[0]
    assert a.context_before == ["near paragraph"]
    assert b.context_before == ["near paragraph","far paragraph"]
    assert EvidenceChunker(sim_threshold=0.9).sim_threshold == 0.9

@pytest.mark.spec("QA-05")
def test_qa05_chunk_preserves_normal_and_split_table_data():
    p=parsed([table(0,rows=2),table(1,page=2,rows=24,prefix="L")])
    result=EvidenceChunker(parser=StaticParser(p)).chunk("demo.pdf")
    assert all(isinstance(e,EvidenceUnit) for e in result)
    assert {e.table_index for e in result} == {0,1}
    normal=[e for e in result if e.table_index==0]
    split=[e for e in result if e.table_index==1]
    assert len(normal)==1 and not normal[0].is_split
    assert len(split)>1 and all(e.is_split for e in split)
    assert normal[0].flattened_rows==expected_sentences(2)
    assert [s for e in split for s in e.flattened_rows]==expected_sentences(24,"L")

@pytest.mark.spec("QA-05")
def test_qa05_no_tables_returns_empty_list():
    assert EvidenceChunker(parser=StaticParser(parsed())).chunk("empty.pdf")==[]

@pytest.mark.spec("QA-06")
@pytest.mark.parametrize("doc_id,expected",[(None,"my-paper"),("custom-doc","custom-doc")])
def test_qa06_document_id(doc_id,expected):
    e=EvidenceChunker(parser=StaticParser(parsed([table()]))).chunk("folder/my-paper.pdf",doc_id=doc_id)[0]
    assert e.eu_id==f"{expected}-p1-0"
    assert e.metadata["doc_id"]==expected
    assert e.metadata["eu_id"]==e.eu_id

@pytest.mark.spec("QA-08")
def test_qa08_adapter_preserves_counts_refs_and_coordinates():
    # Adapter input double: tests conversion only, not PDF recognition.
    class Item:
        def __init__(self,data): self.data=data
        def model_dump(self): return self.data
    class RawTable(Item):
        def export_to_html(self,doc): return "<table><tr><td>42</td></tr></table>"
    bbox={"l":10,"t":900,"r":100,"b":800}
    provenance=[{"page_no":1,"bbox":bbox}]
    texts=[Item({"text":"Table 1: Values","label":"caption","prov":provenance})]
    raw=RawTable({"prov":provenance,"data":{"table_cells":[],"num_rows":1,"num_cols":1},
                  "captions":[{"cref":"#/texts/0"}],"footnotes":[]})
    doc=SimpleNamespace(texts=texts,tables=[raw],pictures=[],pages={
        1:Item({"size":{"width":600,"height":1000}}),
        2:Item({"size":{"width":600,"height":1000}}),
    })
    p=DoclingParser().from_doc(doc)
    assert len(p.texts)==1 and len(p.tables)==1 and len(p.page_sizes)==2
    assert p.tables[0].caption_refs==[0]
    assert p.tables[0].bbox==(10,100,100,200)
    assert p.texts[0].bbox==(10,100,100,200)

@pytest.mark.spec("QA-16")
def test_qa16_missing_caption_still_builds_eu():
    e=build_evidence_units(parsed([table()]))[0]
    assert e.caption_text is None and e.caption_confidence=="none"

@pytest.mark.spec("QA-19")
@pytest.mark.parametrize("with_notes",[False,True])
def test_qa19_first_footnote_only(with_notes):
    t=table()
    ts=[text(0,"First footnote",label="other"),text(1,"Second footnote",label="other")]
    t.footnote_refs=[0,1] if with_notes else []
    e=build_evidence_units(parsed([t],ts))[0]
    assert e.footnote_text==("First footnote" if with_notes else None)
    assert "Second footnote" not in e.text

@pytest.mark.spec("QA-30")
def test_qa30_evidence_contains_available_components():
    from ._data import raw_unit
    e=raw_unit(2)
    for value in ["Results","Table 1: Measurement",
                  "Measurements use the same evaluation conditions.",
                  "Only validated measurements are reported.",
                  "Additional observations follow the measurement.",e.table_html]:
        assert value in e.text
    missing=build_evidence_units(parsed([table()]))[0]
    assert missing.caption_text is None and missing.footnote_text is None
    assert missing.context_before==[] and missing.context_after==[]

@pytest.mark.spec("QA-31")
def test_qa31_text_views_and_cell_unit_prefix():
    e=EvidenceUnit(eu_id="x",page_no=1,caption_text="Table 1: Values",
                   table_html="<table>ONLY_HTML</table>",
                   flattened_rows=["A | Value: 10","B | Value: 20"],
                   context_before=["before"])
    assert "<table>" in e.text and "ONLY_HTML" not in e.retrieval_text
    assert "[Table 1: Values] A | Value: 10" in e.retrieval_units
    assert "[Table 1: Values] B | Value: 20" in e.retrieval_units
    assert len([s for s in e.retrieval_units if "| Value:" in s])==2

@pytest.mark.spec("QA-32")
def test_qa32_metadata_keys_unique_ids_and_bbox():
    p=parsed([table(0,rows=2),table(1,page=2,rows=24,prefix="L")])
    eus=EvidenceChunker(parser=StaticParser(p)).chunk("demo.pdf")
    assert len({e.eu_id for e in eus})==len(eus)
    for e in eus:
        assert set(e.metadata)==EU_KEYS
        assert all(0<=v<=1 for v in e.bbox)
        assert e.bbox[0]<=e.bbox[2] and e.bbox[1]<=e.bbox[3]

@pytest.mark.spec("QA-39")
def test_qa39_consumed_paragraph_removed_unrelated_preserved():
    consumed="This paragraph is already part of table evidence."
    a=SimpleNamespace(text=consumed)
    b=SimpleNamespace(text="Independent paragraph with separate content.")
    eu=EvidenceUnit(eu_id="x",page_no=1,context_before=[consumed])
    assert filter_consumed_paragraphs([a,b],[eu])==[b]

