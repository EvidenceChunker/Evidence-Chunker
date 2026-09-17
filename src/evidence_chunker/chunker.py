"""
chunker.py

EvidenceChunker 메인 클래스.

Usage:
    from evidence_chunker import EvidenceChunker
    from evidence_chunker.export.langchain import to_langchain, to_langchain_units

    chunker = EvidenceChunker()
    eus = chunker.chunk("paper.pdf")                # List[EvidenceUnit] (표만)

    # build_corpus()는 표(EU) + 일반 본문을 한 번에 합쳐서 반환한다.
    chunks = chunker.build_corpus("paper.pdf")      # List[RetrievalChunk] (표+본문)
    docs = to_langchain(chunks)                     # List[LangChainDocument]

    # 표만 필요하거나 직접 다른 텍스트 청커를 쓰고 싶으면 chunk()만 쓸 것.
    docs_tables_only = to_langchain(chunker.chunk("paper.pdf"))

    # 행 단위 다중 벡터(small-to-big): 표의 특정 셀 값을 묻는 질의에 강함.
    # 벡터스토어에는 이 작은 Document/Node들을 넣고, 검색 후에는
    # metadata["parent_text"](표 전체 맥락)를 LLM에 넘길 것.
    docs = to_langchain_units(chunks)               # List[LangChainDocument]
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .unit import EvidenceUnit
from .split import split_oversized_units

if TYPE_CHECKING:
    from .parser.base import ParsedDoc, PdfParser


def _table_bottomleft_bbox(bbox, page_height: float) -> dict:
    """
    TOPLEFT로 정규화된 TableBlock.bbox를 원본 BOTTOMLEFT 딕셔너리로 복원.

    EvidenceUnit.bbox는 공개 계약상 항상 BOTTOMLEFT를 유지해야 한다(자세한
    이유는 geometry.normalize_bbox 참고). parser.docling._to_bbox가 쓰는
    변환식(page_height - y)은 자기 역함수라, 같은 식을 한 번 더 적용하면
    원본 BOTTOMLEFT 값으로 그대로 되돌아온다.
    """
    return {"l": bbox.l, "t": page_height - bbox.t, "r": bbox.r, "b": page_height - bbox.b}


def build_evidence_units(
    parsed: "ParsedDoc",
    bbox_threshold: float = 300.0,
    sim_threshold: float = 0.0,
    doc_id: str | None = None,
) -> list[EvidenceUnit]:
    """파싱된 문서의 표들을 EvidenceUnit 리스트로 변환.

    Args:
        parsed: 파서가 만든 내부 문서 모델(표/텍스트/페이지 크기).
        bbox_threshold: 표 위아래 단락 수집 범위 (PDF 포인트).
        sim_threshold: 문맥 단락 채택 코사인 유사도 임계값.
        doc_id: eu_id 접두사로 쓸 문서 식별자. None이면 "doc".

    Returns:
        생성된 EvidenceUnit 리스트 (분할 전, split_oversized_units()는
        호출자가 별도로 적용).
    """
    if doc_id is None:
        doc_id = "doc"

    from .geometry import normalize_bbox
    from .context import attach_context_paragraphs
    from .caption import map_table_caption
    from .flatten import (
        build_col_header_map,
        build_table_abstract,
        group_sentences_by_row,
    )
    from .filters import find_duplicate_tables, is_toc_or_lof_decoy

    # Docling이 표 1개를 TableItem 2개로 중복 인식하는 경우 제거되는 쪽의
    # 캡션을 남는 쪽에 물려준다.
    dup_drop_map = find_duplicate_tables(parsed)
    dup_donor_caption = {}
    for loser_idx, winner_idx in dup_drop_map.items():
        donor_mapping = map_table_caption(parsed, parsed.tables[loser_idx], loser_idx)
        if donor_mapping.caption_text:
            dup_donor_caption[winner_idx] = donor_mapping

    eu_list: list[EvidenceUnit] = []
    page_counters: dict[int, int] = {}

    for table_block in parsed.tables:
        table_index = table_block.index
        if table_index in dup_drop_map:
            continue  # 중복 표: 더 세밀하게 구조화된 쪽만 남김

        if is_toc_or_lof_decoy(parsed, table_block):
            continue

        pg = table_block.page_no
        if pg == -1:
            continue

        idx = page_counters.get(pg, 0)
        page_counters[pg] = idx + 1
        eu_id = f"{doc_id}-p{pg}-{idx}"

        cap_mapping = map_table_caption(parsed, table_block, table_index)
        if cap_mapping.confidence == "none" and table_index in dup_donor_caption:
            cap_mapping = dup_donor_caption[table_index]
        caption_text = cap_mapping.caption_text
        caption_confidence = cap_mapping.confidence

        footnote_text = None
        if table_block.footnote_refs:
            footnote_text = parsed.texts[table_block.footnote_refs[0]].text or None

        table_html = table_block.html

        # bbox 정규화 (BOTTOMLEFT 복원 이유는 _table_bottomleft_bbox() 참고)
        width, height = parsed.page_sizes.get(pg, (1.0, 1.0))
        raw_bbox = _table_bottomleft_bbox(table_block.bbox, height)
        norm_bbox = normalize_bbox(raw_bbox, width, height)

        eu = EvidenceUnit(
            eu_id=eu_id,
            page_no=pg,
            doc_id=doc_id,
            table_index=table_index,
            caption_text=caption_text,
            table_html=table_html,
            footnote_text=footnote_text,
            bbox=norm_bbox,
            caption_confidence=caption_confidence,
        )

        # table_bbox를 직접 넘기는 이유는 attach_context_paragraphs() 참고.
        attach_context_paragraphs(
            eu, parsed, bbox_threshold, sim_threshold, table_bbox=table_block.bbox,
        )

        cells = table_block.cells
        num_rows = table_block.num_rows
        num_cols = table_block.num_cols

        eu.row_sentence_map = group_sentences_by_row(cells, num_rows, num_cols, footnote_text)
        eu.flattened_rows = [
            s for row in sorted(eu.row_sentence_map) for s in eu.row_sentence_map[row]
        ]

        col_map = build_col_header_map(cells, num_cols)
        # eu.section_header가 이미 채워져 있어야 함 (순서 변경 x)
        eu.table_abstract = build_table_abstract(caption_text, col_map, num_rows, eu.section_header)

        eu_list.append(eu)

    return eu_list


# ---------------------------------------------------------------------------
# EvidenceChunker
# ---------------------------------------------------------------------------

class EvidenceChunker:
    """
    PDF → Evidence Unit 파이프라인 래퍼.

    PDF를 파싱(기본 Docling, parser 주입 시 교체 가능)해 표를 EvidenceUnit으로
    구성하고, context.attach_context_paragraphs()로 인접 단락·섹션 헤더를
    붙여 반환한다.

    Args:
        parser: PdfParser 프로토콜(parse(path) -> ParsedDoc) 구현체. None이면
            기본 DoclingParser로 파싱(_get_parsed() 참고). chunk()만 교체
            가능 — build_corpus()는 지원 안 함(build_corpus() 참고).
        artifacts_path: Docling 로컬 모델 경로. parser를 직접 넘기면 무시됨.
            None이면 HuggingFace Hub에서 자동 다운로드.
        bbox_threshold: 표 위아래 단락 수집 범위 (PDF 포인트). 기본 300pt.
        sim_threshold: 코사인 유사도 임계값. 기본 0.0(사실상 bbox 단일 게이트). 
                       더 엄격한 필터링이 필요하면 호출자가 올릴 수 있음.
    """

    def __init__(
        self,
        parser: "PdfParser | None" = None,
        artifacts_path: str | None = None,
        bbox_threshold: float = 300.0,
        sim_threshold: float = 0.0,
    ) -> None:
        self.parser = parser
        self.artifacts_path = artifacts_path
        self.bbox_threshold = bbox_threshold
        self.sim_threshold = sim_threshold
        # Docling DocumentConverter는 첫 파싱 시점에 만든다(lazy).
        # parser를 주입하면 아예 만들지 않으므로 Docling 초기화 비용도 들지 않는다.
        self._converter = None

    # ------------------------------------------------------------------
    # 퍼블릭 API
    # ------------------------------------------------------------------

    def chunk(
        self,
        pdf_path: str | Path,
        doc_id: str | None = None,
    ) -> list[EvidenceUnit]:
        """
        PDF에서 표(Evidence Unit)만 추출.

        표와 무관한 일반 본문까지 합친 검색 코퍼스가 필요하면
        build_corpus()를 쓸 것.

        Args:
            pdf_path: PDF 파일 경로.
            doc_id: 문서 식별자. 기본은 파일명 stem(EvidenceUnit.doc_id 참고).

        Returns:
            추출된 EvidenceUnit 리스트. 512토큰(DEFAULT_TOKEN_LIMIT) 초과 EU는
            행 단위로 분할된 상태로 반환된다.
        """
        pdf_path = str(pdf_path)
        doc_id = doc_id or Path(pdf_path).stem
        parsed = self._get_parsed(pdf_path)
        eu_list = build_evidence_units(parsed, self.bbox_threshold, self.sim_threshold, doc_id)
        return split_oversized_units(eu_list)

    def build_corpus(
        self,
        pdf_path: str | Path,
        doc_id: str | None = None,
    ) -> list:
        """
        PDF를 표(EU) + 일반 본문을 합친 검색 코퍼스로 변환.

        EU가 이미 흡수한 문단과 겹치는 일반 청크는 자동 제거되므로
        (중복 등장으로 인한 카니발라이제이션 방지: _build_text_chunks() 참고), 
        이 한 번 호출로 바로 벡터스토어에 넣을 수 있는 최종 코퍼스가 나온다.

        Args:
            pdf_path: PDF 파일 경로
            doc_id:   chunk() 참고

        Returns:
            List[RetrievalChunk] — EvidenceUnit(표)과 export.TextChunk(일반 본문)이 섞인 리스트. 
            둘 다 chunk_id/is_atomic/text/retrieval_text/retrieval_units/metadata를 노출하므로 
            export.to_langchain() 등에 타입 구분 없이 그대로 넘기면 됨. 
            is_atomic=True인 항목이 일반 본문 청크(EU 아님)를 뜻한다.

        Raises:
            NotImplementedError: 생성자에 parser를 넘긴 경우. 
            표만 필요하면 chunk()를 쓸 것(그쪽은 parser 교체 가능)
        """
        from .export import TextChunk
        from .parser.docling import DoclingParser

        if self.parser is not None:
            raise NotImplementedError(
                "build_corpus()는 커스텀 parser를 지원하지 않음 — 일반 본문 "
                "청킹(HybridChunker)이 Docling DocumentConverter 산출물에 직접 결합되어 있기 때문. "
                "표만 필요하면 chunk()를 쓸 것."
            )

        pdf_path = str(pdf_path)
        doc_id = doc_id or Path(pdf_path).stem
        doc = self._parse(pdf_path)
        parsed = DoclingParser().from_doc(doc)
        eu_list = build_evidence_units(parsed, self.bbox_threshold, self.sim_threshold, doc_id)
        eu_list = split_oversized_units(eu_list)
        text_chunks = self._build_text_chunks(doc, eu_list)
        return eu_list + [
            TextChunk(c, doc_id, i) for i, c in enumerate(text_chunks)
        ]

    def _build_text_chunks(self, doc, eu_list: list[EvidenceUnit]) -> list:
        """
        표와 무관한 일반 본문 청크 생성 (Docling HybridChunker), 
        원본 청크 그대로 반환 (export.TextChunk 래핑은 build_corpus()가 수행).

        EU가 이미 흡수한 문단과 겹치는 청크는 export.filter_consumed_paragraphs()로 제거
        """
        from docling.chunking import HybridChunker
        from docling_core.types.doc import DocItemLabel
        from .export import filter_consumed_paragraphs

        all_chunks = list(HybridChunker().chunk(doc))
        is_table_chunk = lambda c: any(
            di.label == DocItemLabel.TABLE for di in c.meta.doc_items
        )
        text_chunks = [c for c in all_chunks if not is_table_chunk(c)]
        return filter_consumed_paragraphs(text_chunks, eu_list)

    # ------------------------------------------------------------------
    # 내부 단계
    # ------------------------------------------------------------------

    def _get_parsed(self, pdf_path: str) -> "ParsedDoc":
        """parser가 주입됐으면 그대로 위임, 아니면 기본 Docling 플로우
        (artifacts_path 반영)로 파싱 후 ParsedDoc으로 변환.

        chunk()가 parser를 완전히 교체 가능한 지점:
        표 추출 알고리즘(caption/context/filters/flatten)이 ParsedDoc만 알고 
        Docling을 모르기 때문(tests/test_no_docling_dependency.py 참고).
        """
        if self.parser is not None:
            return self.parser.parse(pdf_path)
        from .parser.docling import DoclingParser

        doc = self._parse(pdf_path)
        return DoclingParser().from_doc(doc)

    def _parse(self, pdf_path: str):
        """Docling으로 PDF 파싱 → DoclingDocument."""
        if self._converter is None:
            self._converter = self._make_converter()
        result = self._converter.convert(pdf_path)
        return result.document

    def _make_converter(self):
        """DocumentConverter 생성. artifacts_path가 있으면 로컬 모델을 쓴다.

        parser.docling.make_converter()와 역할이 겹치지만, 
        이쪽은 생성자 인자(artifacts_path)만 반영하는 최소 구성
        build_corpus()가 HybridChunker에 그대로 넘길 DoclingDocument를 
        직접 만들어야 해서 컨버터를 이 클래스가 소유한다. 
        표 구조 추출(do_table_structure)은
        Docling 기본값이 이미 True라 따로 지정하지 않는다.
        """
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pipeline_options = PdfPipelineOptions()
        if self.artifacts_path:
            pipeline_options.artifacts_path = self.artifacts_path

        return DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
