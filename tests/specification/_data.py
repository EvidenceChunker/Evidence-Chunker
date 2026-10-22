"""Small deterministic inputs. Expected values must not come from tested outputs."""
from html import escape
from evidence_chunker.parser.base import BBox, BlockLabel, TextBlock, TableBlock, ParsedDoc

EU_KEYS = {
    "chunk_id", "eu_id", "doc_id", "page_no", "page_span", "table_index",
    "section_header", "caption_text", "bbox", "is_split", "split_index",
    "total_splits", "caption_confidence", "retrieval_text",
}
TEXT_KEYS = {"chunk_id", "eu_id", "doc_id", "page_no", "source"}

def cell(text, row, col, *, column_header=False, row_header=False, row_end=None, col_end=None):
    return dict(text=text, column_header=column_header, row_header=row_header,
                start_row_offset_idx=row, end_row_offset_idx=row_end or row+1,
                start_col_offset_idx=col, end_col_offset_idx=col_end or col+1)

def text(index, value, *, label="text", page=1, center=300, left=20, right=280):
    return TextBlock(index, value, BlockLabel(label), page,
                     BBox(left, center-5, right, center+5))

def table(index=0, *, page=1, rows=2, top=400, bottom=500, prefix="R"):
    cells=[cell("Name",0,0,column_header=True),cell("Value",0,1,column_header=True)]
    body=[]
    for r in range(1, rows+1):
        name=f"{prefix}{r:02d}"
        value=f"value-{r:02d} alpha beta gamma delta"
        cells += [cell(name,r,0,row_header=True),cell(value,r,1)]
        body.append(f"<tr><td>{escape(name)}</td><td>{escape(value)}</td></tr>")
    html="<table><thead><tr><th>Name</th><th>Value</th></tr></thead><tbody>"+"".join(body)+"</tbody></table>"
    return TableBlock(index,page,BBox(20,top,280,bottom),cells,rows+1,2,html)

def parsed(tables=None, texts=None):
    return ParsedDoc(texts or [],tables or [],set(),{p:(600.0,1000.0) for p in (1,2,3)})

def expected_sentences(rows, prefix="R"):
    return [f"{prefix}{r:02d} | Value: value-{r:02d} alpha beta gamma delta" for r in range(1,rows+1)]

class StaticParser:
    def __init__(self, document):
        self.document=document
        self.calls=[]
    def parse(self,path):
        self.calls.append(path)
        return self.document

def raw_unit(rows=24):
    from evidence_chunker.chunker import build_evidence_units
    t=table(rows=rows)
    t.caption_refs=[1]
    t.footnote_refs=[3]
    ts=[
        text(0,"Results",label="section_header",center=100),
        text(1,"Table 1: Measurement",label="caption",center=380),
        text(2,"Measurements use the same evaluation conditions.",center=350),
        text(3,"Only validated measurements are reported.",label="other",center=530),
        text(4,"Additional observations follow the measurement.",center=550),
    ]
    return build_evidence_units(parsed([t],ts),doc_id="fixture")[0]

