"""Opt-in checks: enabled-but-missing dependencies/data FAIL, never silently skip."""
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import venv
import pytest

@pytest.mark.spec("QA-01")
@pytest.mark.spec_install
def test_qa01_clean_venv_install_and_import(tmp_path,request):
    target=request.config.getoption("--spec-install-target")
    env_dir=tmp_path/"clean"
    venv.EnvBuilder(with_pip=True).create(env_dir)
    python=env_dir/("Scripts/python.exe" if os.name=="nt" else "bin/python")
    env=os.environ.copy()
    env.pop("PYTHONPATH",None);env.pop("PYTHONHOME",None)
    subprocess.run([str(python),"-m","pip","install",target],cwd=tmp_path,env=env,check=True,timeout=1200)
    subprocess.run([str(python),"-m","pip","check"],cwd=tmp_path,env=env,check=True,timeout=120)
    subprocess.run([str(python),"-c","import evidence_chunker; from evidence_chunker import EvidenceChunker, EvidenceUnit"],cwd=tmp_path,env=env,check=True,timeout=120)

@pytest.mark.spec("QA-07")
@pytest.mark.spec_integration
def test_qa07_pdf_parser_matches_human_table_count(spec_pdf_bundle):
    _,oracle,_,doc,_=spec_pdf_bundle
    assert len(doc.tables)==oracle["table_count"]
    assert len(doc.pages)==oracle["page_count"]

@pytest.mark.spec("QA-08")
@pytest.mark.spec_integration
def test_qa08_real_adapter_preserves_counts_refs_and_coords(spec_pdf_bundle):
    _,_,_,doc,parsed=spec_pdf_bundle
    assert len(parsed.tables)==len(doc.tables)
    assert len(parsed.texts)==len(doc.texts)
    assert len(parsed.page_sizes)==len(doc.pages)
    for raw,converted in zip(doc.tables,parsed.tables):
        d=raw.model_dump()
        expected=[int(r["cref"].split("/")[-1]) for r in d.get("captions",[])]
        assert converted.caption_refs==expected
        assert all(isinstance(i,int) and 0<=i<len(parsed.texts) for i in converted.caption_refs)
        prov=d.get("prov") or []
        if prov:
            b=prov[0]["bbox"];h=parsed.page_sizes[converted.page_no][1]
            assert converted.bbox==(b["l"],h-b["t"],b["r"],h-b["b"])

@pytest.mark.spec("QA-05")
@pytest.mark.spec("QA-30")
@pytest.mark.spec_integration
def test_qa05_qa30_real_pdf_content_matches_oracle(spec_pdf_bundle,monkeypatch):
    from bs4 import BeautifulSoup
    from evidence_chunker import EvidenceUnit
    path,oracle,c,doc,parsed=spec_pdf_bundle
    def normalize(s):return " ".join(s.split())
    # Reuse actual parsed PDF; do not replace the EU builder or splitting.
    monkeypatch.setattr(c,"_parse",lambda _:doc)
    units=c.chunk(path,doc_id="acceptance")
    assert units and all(isinstance(e,EvidenceUnit) for e in units)
    expected_tables=[t for t in oracle["tables"] if t.get("expect_eu",True)]
    assert {e.table_index for e in units}=={t["index"] for t in expected_tables}
    for t in oracle["tables"]:
        raw=parsed.tables[t["index"]]
        assert raw.page_no==t["page_no"]
        assert [normalize(v["text"]) for v in raw.cells]==[normalize(v) for v in t["cell_texts"]]
    assert {t["expect_split"] for t in expected_tables}=={False,True}, "Oracle must cover a normal and a split table"
    for t in expected_tables:
        group=[e for e in units if e.table_index==t["index"]]
        assert all(e.page_no==t["page_no"] for e in group)
        assert all(e.is_split is t["expect_split"] for e in group)
        # Independent expected data rows: exclude header rows.
        observed=[]
        for e in group:
            soup=BeautifulSoup(e.table_html or "","html.parser")
            rows=soup.select("tbody tr") if soup.tbody else soup.select("tr")[1:]
            observed.extend([[normalize(v.get_text()) for v in row.find_all(["td","th"])] for row in rows])
        assert observed==t["data_rows"]
        combined="\n".join(e.text for e in group)
        for value in t["required_text"]:
            assert normalize(value) in normalize(combined)
        for field in t["absent_fields"]:
            assert all(not getattr(e,field) for e in group)

@pytest.mark.spec("QA-23")
@pytest.mark.spec_integration
def test_qa23_pdf_section_headers_match_human_oracle(spec_pdf_bundle):
    from evidence_chunker.chunker import build_evidence_units
    _,oracle,_,_,parsed=spec_pdf_bundle
    units=build_evidence_units(parsed)
    expected={t["index"]:t["section_header"] for t in oracle["tables"] if t.get("expect_eu",True)}
    assert {e.table_index:e.section_header for e in units}==expected

@pytest.mark.spec("QA-24")
@pytest.mark.spec_integration
def test_qa24_real_similarity_filter_reduces_candidates():
    from evidence_chunker import EvidenceUnit
    from evidence_chunker.context import attach_context_paragraphs
    from ._data import parsed,table,text
    # Require the real model package; missing package must not trigger silent fallback.
    import sentence_transformers
    caption="Annual revenue and operating profit of the company."
    ts=[text(0,caption,center=350),
        text(1,"Tropical butterflies have colorful wings and feed on nectar.",center=300)]
    t=table()
    p=parsed([t],ts)
    before=EvidenceUnit(eu_id="x",page_no=1,caption_text=caption)
    after=EvidenceUnit(eu_id="x",page_no=1,caption_text=caption)
    attach_context_paragraphs(before,p,sim_threshold=0.0,table_bbox=t.bbox)
    attach_context_paragraphs(after,p,sim_threshold=0.9,table_bbox=t.bbox)
    assert set(after.context_before)<=set(before.context_before)
    assert len(after.context_before)<len(before.context_before)
    assert caption in after.context_before

@pytest.mark.spec("QA-38")
@pytest.mark.spec_integration
def test_qa38_real_build_corpus_contains_tables_and_text(spec_pdf_bundle,monkeypatch):
    from evidence_chunker import EvidenceUnit
    from evidence_chunker.export import TextChunk
    path,_,c,doc,_=spec_pdf_bundle
    monkeypatch.setattr(c,"_parse",lambda _:doc)
    chunks=c.build_corpus(path,doc_id="acceptance")
    assert any(isinstance(x,EvidenceUnit) for x in chunks)
    assert any(isinstance(x,TextChunk) for x in chunks)

