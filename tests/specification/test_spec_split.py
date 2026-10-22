import copy
from bs4 import BeautifulSoup
import pytest
import tiktoken
from evidence_chunker import EvidenceUnit
from evidence_chunker.split import split_eu
from evidence_chunker.tokens import count_tokens, count_eu_tokens
from ._data import raw_unit, expected_sentences

@pytest.mark.spec("QA-33")
def test_qa33_token_count_matches_tiktoken():
    value="한글과 English: 42.5%."
    assert count_tokens(value)==len(tiktoken.get_encoding("cl100k_base").encode(value))

@pytest.mark.spec("QA-33")
@pytest.mark.parametrize("size,strategy",[(512,"single"),(513,"row_split")])
def test_qa33_exact_512_513_boundary(size,strategy):
    # Real tokenizer; no mocked token counter. Padding creates the boundary input.
    e=EvidenceUnit(eu_id="boundary",page_no=1,
        table_html="<table><thead><tr><th>Value</th></tr></thead><tbody><tr><td>42</td></tr></tbody></table>",
        row_sentence_map={1:["Value: 42"]},flattened_rows=["Value: 42"])
    for n in range(1,700):
        e.context_before=["x "*n]
        if count_eu_tokens(e)==size:break
    assert count_eu_tokens(e)==size, "Tokenizer boundary fixture could not be constructed"
    result=split_eu(e)
    assert result.strategy==strategy

@pytest.mark.spec("QA-33")
def test_qa33_larger_limit_reduces_chunk_count():
    e=raw_unit()
    a,b=split_eu(e,limit=256),split_eu(e,limit=512)
    assert a.strategy==b.strategy=="row_split"
    assert len(b.chunks)<len(a.chunks)

@pytest.mark.spec("QA-34")
def test_qa34_data_rows_and_sentences_preserved_once_in_order():
    result=split_eu(raw_unit())
    assert result.strategy=="row_split" and len(result.chunks)>1
    rows=[]
    for e in result.chunks:
        soup=BeautifulSoup(e.table_html,"html.parser")
        rows.extend([[td.get_text() for td in row.find_all("td")] for row in soup.select("tbody tr")])
    assert rows==[[f"R{r:02d}",f"value-{r:02d} alpha beta gamma delta"] for r in range(1,25)]
    assert [s for e in result.chunks for s in e.flattened_rows]==expected_sentences(24)

@pytest.mark.spec("QA-34")
@pytest.mark.spec_gap
def test_qa34_final_text_respects_512_token_limit():
    e=raw_unit()
    result=split_eu(e)
    assert result.strategy=="row_split"
    sizes=[count_eu_tokens(c) for c in result.chunks]
    assert all(n<=512 for n in sizes), f"GAP-01: final EU sizes exceed 512: {sizes}"

@pytest.mark.spec("QA-35")
def test_qa35_repeated_fields_and_final_footnote_field():
    e=raw_unit()
    result=split_eu(e)
    assert result.strategy=="row_split" and len(result.chunks)>1
    for c in result.chunks:
        assert c.caption_text==e.caption_text
        assert c.section_header==e.section_header
        assert c.context_before==e.context_before and c.context_after==e.context_after
        assert c.table_abstract==e.table_abstract
        assert BeautifulSoup(c.table_html,"html.parser").thead.get_text()== "NameValue"
    assert all(not c.footnote_text for c in result.chunks[:-1])
    assert result.chunks[-1].footnote_text==e.footnote_text

@pytest.mark.spec("QA-35")
@pytest.mark.spec_gap
def test_qa35_full_footnote_text_only_in_last_chunk():
    # Literal submitted wording: full note content must not occur in earlier chunks.
    e=raw_unit()
    note="UNIQUE_FULL_FOOTNOTE"
    e.footnote_text=note
    e.row_sentence_map[1]=[expected_sentences(24)[0]+" ["+note+"]"]
    e.flattened_rows=[s for r in sorted(e.row_sentence_map) for s in e.row_sentence_map[r]]
    result=split_eu(e)
    assert result.strategy=="row_split" and len(result.chunks)>1
    assert all(note not in c.text for c in result.chunks[:-1]), "GAP-04: note survives in flattened cell text"
    assert note in result.chunks[-1].text

@pytest.mark.spec("QA-36")
def test_qa36_split_ids_indices_and_total():
    e=raw_unit()
    chunks=split_eu(e).chunks
    assert len(chunks)>1
    assert [c.eu_id for c in chunks]==[f"{e.eu_id}-s{i}" for i in range(1,len(chunks)+1)]
    assert [c.split_index for c in chunks]==list(range(1,len(chunks)+1))
    assert all(c.total_splits==len(chunks) and c.is_split for c in chunks)

@pytest.mark.spec("QA-37")
@pytest.mark.parametrize("case",["huge","single_row"])
def test_qa37_oversized_or_single_row_returns_original(case):
    e=raw_unit(1)
    if case=="huge":
        e.context_before=["word "*4000]
        assert count_eu_tokens(e)>3072
    else:
        payload="word "*650
        e.table_html="<table><thead><tr><th>Value</th></tr></thead><tbody><tr><td>"+payload+"</td></tr></tbody></table>"
        e.flattened_rows=[];e.row_sentence_map={}
        assert 512<count_eu_tokens(e)<=3072
    result=split_eu(e)
    assert result.strategy=="llm_summary"
    assert len(result.chunks)==1 and result.chunks[0] is e
    assert not e.is_split

