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
    # 입력은 행 분할로 실제 한도를 맞출 수 있는 표여야 한다. 데이터 행이 하나뿐인
    # 표는 513토큰이어도 쪼갤 수단이 없어 QA-37의 llm_summary로 빠지며, 그 경로는
    # test_qa37_oversized_or_single_row_returns_original이 따로 검증한다.
    rows=10
    body="".join(f"<tr><td>R{r}</td><td>{r}</td></tr>" for r in range(1,rows+1))
    sentences=[f"R{r} | Value: {r}" for r in range(1,rows+1)]
    e=EvidenceUnit(eu_id="boundary",page_no=1,
        table_html="<table><thead><tr><th>Name</th><th>Value</th></tr></thead><tbody>"+body+"</tbody></table>",
        row_sentence_map={r:[sentences[r-1]] for r in range(1,rows+1)},
        flattened_rows=list(sentences))
    for n in range(1,700):
        e.context_before=["x "*n]
        if count_eu_tokens(e)==size:break
    assert count_eu_tokens(e)==size, "Tokenizer boundary fixture could not be constructed"
    result=split_eu(e)
    assert result.strategy==strategy
    if strategy=="row_split":
        assert all(count_eu_tokens(c)<=512 for c in result.chunks)


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
def test_qa35_footnote_field_is_last_only_but_cell_marker_survives():
    """
    QA-35의 '각주'는 표 전체에 걸리는 주석(footnote_text 필드)을 뜻한다.
    특정 셀에 걸린 각주 마커는 QA-28("각주 마커가 있는 셀 문장에 각주 내용
    첨부")에 따라 그 셀 문장에 유지된다. 조각이 단독으로 검색됐을 때 "이 값에
    왜 표시가 붙었는지"를 잃지 않기 위한 설계이며, 두 항목은 각각 '셀 문장
    변환'과 '분할 반복 요소'라는 서로 다른 단계를 규정한다.
    """
    e=raw_unit()
    note="UNIQUE_FULL_FOOTNOTE"
    e.footnote_text=note
    e.row_sentence_map[1]=[expected_sentences(24)[0]+" ["+note+"]"]
    e.flattened_rows=[s for r in sorted(e.row_sentence_map) for s in e.row_sentence_map[r]]
    result=split_eu(e)
    assert result.strategy=="row_split" and len(result.chunks)>1
    # 필드로서의 각주는 마지막 조각에만
    assert all(not c.footnote_text for c in result.chunks[:-1])
    assert result.chunks[-1].footnote_text==note
    assert note in result.chunks[-1].text
    # 마커가 달린 셀 문장의 각주는 그 셀이 속한 조각에 그대로 유지 (QA-28)
    assert any(note in s for s in result.chunks[0].flattened_rows)



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

