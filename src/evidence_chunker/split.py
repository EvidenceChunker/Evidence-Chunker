"""
split.py

토큰 한도를 넘는 EvidenceUnit을 표의 행 단위로 쪼개는 모듈.
파이프라인 진입점은 split_oversized_units(eu_list).

EU 하나의 토큰 수에 따라 세 전략 중 하나가 선택된다:
    single      : 한도 이내 — 원본 그대로 통과
    row_split   : 캡션/헤더/문맥을 유지한 채 본문 행을 나눠 여러 조각으로
    llm_summary : 행 하나만으로도 한도를 넘거나 표가 지나치게 큰 경우.
                  기계적 분할을 포기하고 원본 EU를 그대로 통과시킨다 —
                  이름과 달리 이 모듈이 요약을 수행하지는 않고, 요약이
                  필요하다는 신호로만 쓰인다.

표 HTML을 다시 조립해야 하므로 이 모듈만 beautifulsoup4에 의존한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from bs4 import BeautifulSoup

from .unit import EvidenceUnit
from .tokens import count_tokens, DEFAULT_TOKEN_LIMIT

SPLIT_LIMIT = DEFAULT_TOKEN_LIMIT

# 한도의 6배를 넘는 표는 행 단위로 쪼개도 조각 수가 과하게 늘어나 검색
# 단위로서의 의미가 흐려지므로, 기계적 분할 대상에서 빼고 llm_summary로
# 분류한다(실제 요약은 상위 레이어의 몫).
LLM_SUMMARY_LIMIT = DEFAULT_TOKEN_LIMIT * 6


@dataclass
class SplitResult:
    """split_eu() 한 번의 결과.

    strategy       : "single" | "row_split" | "llm_summary" (모듈 docstring 참고)
    chunks         : 분할 결과. single/llm_summary는 원본 EU 하나만 담는다.
    original_eu_id : 분할 전 EU의 eu_id
    total_tokens   : 분할 전 eu.text의 토큰 수
    """

    strategy: str
    chunks: list[EvidenceUnit]
    original_eu_id: str
    total_tokens: int


def parse_table(table_html: str):
    """표 HTML을 헤더와 본문 행들로 분리.

    <thead>가 있으면 그 안의 행 전체를 헤더로 보고, 없으면 첫 <tr> 한 줄만
    헤더로 본다.

    Returns:
        (header_html, body_rows) — <table>을 못 찾으면 ("", [])
    """
    soup = BeautifulSoup(table_html, "html.parser")

    table = soup.find("table")
    if table is None:
        return "", []

    # -------- 헤더 --------
    thead = table.find("thead")
    if thead:
        header_html = str(thead)
    else:
        rows = table.find_all("tr", recursive=True)
        if not rows:
            return "", []
        header_html = str(rows[0])

    # -------- 본문 행 --------
    tbody = table.find("tbody")
    if tbody:
        body_rows = [str(r) for r in tbody.find_all("tr", recursive=False)]
    else:
        rows = table.find_all("tr")
        if len(rows) <= 1:
            body_rows = []
        else:
            body_rows = [str(r) for r in rows[1:]]

    return header_html, body_rows


def build_table(header_html, rows):
    """헤더 + 주어진 행들로 <table> HTML을 다시 조립.

    header_html이 <thead>면 행들을 <tbody>로 감싸고, 첫 <tr>을 헤더로 쓴
    경우에는 그대로 이어 붙인다 — 원본 표의 구조를 최대한 유지한다.
    """
    if header_html.startswith("<thead"):
        return (
            "<table>"
            + header_html
            + "<tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    return "<table>" + header_html + "".join(rows) + "</table>"


def estimate_chunk_tokens(caption, html, footnote=""):
    """분할 조각 하나의 토큰 수 추정 (캡션 + 표 HTML + 각주).

    문맥 단락(context_before/after)은 모든 조각에 똑같이 복제되므로 여기서
    세지 않는다 — 행을 어디서 끊을지에는 영향이 없지만, 그만큼 완성된
    조각의 실제 토큰 수는 이 추정치보다 크다.
    """
    text = "\n".join(x for x in [caption, html, footnote] if x)
    return count_tokens(text)


def split_eu(
    eu: EvidenceUnit,
    limit=SPLIT_LIMIT,
):
    """EU 하나를 토큰 한도에 맞춰 분할.

    Args:
        eu: 분할 대상 EU.
        limit: 조각 하나의 토큰 상한. 기본 DEFAULT_TOKEN_LIMIT(512).

    Returns:
        SplitResult — 선택된 전략과 조각 목록. row_split 조각은 캡션/섹션
        헤더/문맥 단락/table_abstract/bbox를 부모에서 그대로 물려받고,
        table_html과 flattened_rows만 자기 행 범위로 좁혀진다. 조각의
        eu_id는 "{부모 eu_id}-s{n}"(n은 1부터).
    """
    total = count_tokens(eu.text)

    if total <= limit:
        return SplitResult(
            strategy="single",
            chunks=[eu],
            original_eu_id=eu.eu_id,
            total_tokens=total,
        )

    if total > LLM_SUMMARY_LIMIT:
        return SplitResult(
            strategy="llm_summary",
            chunks=[eu],
            original_eu_id=eu.eu_id,
            total_tokens=total,
        )

    header_html, rows = parse_table(eu.table_html or "")

    if len(rows) == 0:
        return SplitResult(
            strategy="single",
            chunks=[eu],
            original_eu_id=eu.eu_id,
            total_tokens=total,
        )

    # 헤더가 차지하는 원본 행 개수. row_sentence_map의 키(start_row_offset_idx)는
    # 헤더 행까지 포함해서 매겨지므로, body_rows의 0-based 위치를 원본 행
    # 인덱스로 되돌리려면 이 오프셋을 더해야 한다 (단일 헤더 행 가정 —
    # parse_table()도 동일하게 가정함).
    header_row_offset = max(1, header_html.count("<tr"))

    def _flattened_for(positions: list[int]) -> list[str]:
        """분할 조각에 포함된 원본 행들의 flattened_rows 문장만 추출."""
        return [
            sentence
            for pos in positions
            for sentence in eu.row_sentence_map.get(pos + header_row_offset, [])
        ]

    chunks = []
    current_rows = []
    current_positions: list[int] = []
    split_idx = 1

    for pos, row in enumerate(rows):
        # 행 하나만으로도 한도를 넘으면 행 단위 분할로는 해결되지 않으므로
        # 여기까지 만든 조각을 버리고 표 전체를 llm_summary로 넘긴다.
        single_html = build_table(header_html, [row])
        if estimate_chunk_tokens(
            eu.caption_text,
            single_html,
            eu.footnote_text,
        ) > limit:
            return SplitResult(
                strategy="llm_summary",
                chunks=[eu],
                original_eu_id=eu.eu_id,
                total_tokens=total,
            )

        candidate_rows = current_rows + [row]
        candidate_html = build_table(header_html, candidate_rows)
        token = estimate_chunk_tokens(eu.caption_text, candidate_html)

        if token <= limit:
            current_rows.append(row)
            current_positions.append(pos)
        else:
            chunk_html = build_table(header_html, current_rows)
            chunks.append(
                EvidenceUnit(
                    eu_id=f"{eu.eu_id}-s{split_idx}",
                    page_no=eu.page_no,
                    doc_id=eu.doc_id,
                    table_index=eu.table_index,
                    section_header=eu.section_header,
                    caption_text=eu.caption_text,
                    table_html=chunk_html,
                    # 각주는 표 전체에 걸리는 주석이라 조각마다 반복하지 않고
                    # 마지막 조각에만 싣는다(루프 뒤 "마지막 조각" 참고).
                    footnote_text="",
                    # 반면 문맥 단락은 모든 조각에 그대로 복제한다 — 일부
                    # 조각에만 실으면 문맥에 답이 있는 질의에서 그 조각이
                    # 검색돼도 근거가 빠진다.
                    context_before=list(eu.context_before),
                    context_after=list(eu.context_after),
                    page_span=set(eu.page_span),  # 부모 EU의 page_span 그대로 전파
                    flattened_rows=_flattened_for(current_positions),
                    table_abstract=eu.table_abstract,
                    bbox=eu.bbox,
                    is_split=True,
                    split_index=split_idx,
                    total_splits=None,
                    caption_confidence=eu.caption_confidence,
                )
            )
            split_idx += 1
            current_rows = [row]
            current_positions = [pos]

    # 마지막 조각
    if current_rows:
        chunk_html = build_table(header_html, current_rows)
        chunks.append(
            EvidenceUnit(
                eu_id=f"{eu.eu_id}-s{split_idx}",
                page_no=eu.page_no,
                doc_id=eu.doc_id,
                table_index=eu.table_index,
                section_header=eu.section_header,
                caption_text=eu.caption_text,
                table_html=chunk_html,
                footnote_text=eu.footnote_text,
                context_before=list(eu.context_before),
                context_after=list(eu.context_after),
                page_span=set(eu.page_span),  # 앞쪽 조각과 동일하게 전파(마지막 조각만 빠져있던 버그)
                flattened_rows=_flattened_for(current_positions),
                table_abstract=eu.table_abstract,
                bbox=eu.bbox,
                is_split=True,
                split_index=split_idx,
                total_splits=None,
                caption_confidence=eu.caption_confidence,
            )
        )

    total_chunks = len(chunks)
    for c in chunks:
        c.total_splits = total_chunks

    return SplitResult(
        strategy="row_split",
        chunks=chunks,
        original_eu_id=eu.eu_id,
        total_tokens=total,
    )


def split_oversized_units(
    eu_list: list[EvidenceUnit],
    stats: dict | None = None,
) -> list[EvidenceUnit]:
    """
    512토큰(DEFAULT_TOKEN_LIMIT) 초과 EU를 split_eu()로 행 단위 분할.
    한도 이내 EU는 그대로 통과.

    stats를 넘기면 전략별 건수와 분할 규모(split_eus / split_chunks)를
    채워준다. 동작에는 영향 없음(계측 전용).
    """
    result = []
    for eu in eu_list:
        r = split_eu(eu)
        if stats is not None:
            stats[f"strategy:{r.strategy}"] = stats.get(f"strategy:{r.strategy}", 0) + 1
            if r.strategy == "row_split":
                stats["split_eus"] = stats.get("split_eus", 0) + 1
                stats["split_chunks"] = stats.get("split_chunks", 0) + len(r.chunks)
        result.extend(r.chunks)
    return result
