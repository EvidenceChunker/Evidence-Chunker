# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
EvidenceChunker(parser=...) 생성자 주입이 실제로 동작한다는 실행 가능한 증거.

README가 약속하는 "PdfParser 프로토콜의 parse() 하나만 구현하면 파서를
갈아끼울 수 있다"는 계약을 실행 가능한 형태로 잠근다 — chunk()가 주입된
parser만 쓰고, 내부에서 DocumentConverter를 몰래 부르지 않는지 확인한다.

이 테스트는 Docling을 아예 import하지 않는 FakePdfParser를 주입해서
chunk()가 끝까지 동작함을 증명한다 — sys.meta_path로 docling import를
막아둔 채로 실행하므로(tests/test_no_docling_dependency.py와 동일 기법),
내부 어딘가 raw DoclingDocument를 몰래 요구하면 이 테스트가 ImportError로
바로 잡아낸다.
"""
import sys

import pytest

from evidence_chunker import EvidenceChunker
from evidence_chunker.parser.base import BBox, ParsedDoc, TableBlock


def _fake_parsed_doc() -> ParsedDoc:
    table = TableBlock(
        index=0,
        page_no=1,
        bbox=BBox(l=10.0, t=20.0, r=110.0, b=60.0),
        cells=[{
            "text": "42",
            "column_header": False,
            "row_header": False,
            "start_row_offset_idx": 0,
            "end_row_offset_idx": 1,
            "start_col_offset_idx": 0,
            "end_col_offset_idx": 1,
        }],
        num_rows=1,
        num_cols=1,
        html="<table><tr><td>42</td></tr></table>",
    )
    return ParsedDoc(texts=[], tables=[table], picture_caption_refs=set(), page_sizes={1: (612.0, 792.0)})


class FakePdfParser:
    """PdfParser 프로토콜만 구현 — Docling을 아예 import하지 않는다."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def parse(self, path: str) -> ParsedDoc:
        self.calls.append(path)
        return _fake_parsed_doc()


_BLOCKED_PREFIXES = ("docling", "docling_core")


class _BlockDocling:
    def find_spec(self, name, path=None, target=None):
        if name in _BLOCKED_PREFIXES or name.startswith(tuple(p + "." for p in _BLOCKED_PREFIXES)):
            raise ImportError(f"{name} blocked for test")
        return None


def test_chunk_uses_injected_parser_without_docling():
    """chunk()가 주입된 parser.parse()만으로 끝까지 동작 — Docling 무관."""
    fake_parser = FakePdfParser()
    blocker = _BlockDocling()
    sys.meta_path.insert(0, blocker)
    try:
        chunker = EvidenceChunker(parser=fake_parser)
        eus = chunker.chunk("fake.pdf", doc_id="doc")
    finally:
        sys.meta_path.remove(blocker)

    assert fake_parser.calls == ["fake.pdf"]
    assert len(eus) == 1
    assert eus[0].eu_id == "doc-p1-0"
    assert eus[0].table_html == "<table><tr><td>42</td></tr></table>"


def test_build_corpus_rejects_injected_parser():
    """build_corpus()는 일반 본문 청킹(HybridChunker)이 Docling에 결합돼 있어
    커스텀 parser를 지원하지 않는다 — 침묵하는 대신 명시적으로 거부한다."""
    chunker = EvidenceChunker(parser=FakePdfParser())
    with pytest.raises(NotImplementedError):
        chunker.build_corpus("fake.pdf")
