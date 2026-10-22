# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
caption.py

캡션 ↔ 표 매핑 알고리즘.

parser.base.TableBlock.caption_refs를 사용해 표와 캡션 텍스트를 1:1로 연결한다. 
관련 로직은 map_table_caption() 참고.

매핑 우선순위
1. direct (Docling 파서가 직접 연결한 경우)
2. bbox (표 상단 200pt 이내 텍스트 탐색)
3. 인접 페이지 (표가 페이지 최상단일 경우 이전 페이지 하단 탐색)
4. 병합 헤더 (표 내부 첫 행이 캡션 역할을 하는 경우)
5. none

아키텍처 규칙
- parser.base.ParsedDoc / TableBlock 기반으로 동작하며, DoclingDocument를 직접 조작하지 않는다.
- 좌표 비교는 TOPLEFT 기준 (cy가 작을수록 위쪽을 의미).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from .parser.base import BBox, ParsedDoc, TableBlock

# 일반 컨텍스트 탐색(300pt)보다 좁힌다 — 캡션은 표에 바짝 붙어 있으므로
# 반경을 좁혀 무관한 텍스트 유입을 막는다.
CAPTION_SEARCH_PT = 200.0

# 파서가 캡션을 section_header로 잘못 라벨링하는 경우가 흔해 후보에 포함한다.
_CANDIDATE_LABEL_NAMES = {"caption", "section_header", "text"}

# "Table 1", "표1", "<표 3>" 등 일반적인 캡션 패턴. [A-Z]?는 "Table C.1" 같은
# 부록 번호 체계까지 잡기 위함.
_CAPTION_PATTERN = re.compile(r"(table|figure|fig|표|그림)\s*\.?\s*[A-Z]?\.?\s*\d+", re.IGNORECASE)
_MIN_CAPTION_LEN = 8

# 텍스트 맨 앞 20자 이내에서만 매칭 — 전체를 검색하면 본문 중간의 참조
# 문구("...as shown in Figure 2.2...")를 캡션으로 오인한다.
CAPTION_PREFIX_CHARS = 20

# 표가 페이지 상/하단 15% 이내일 때만 인접 페이지를 탐색 — 게이팅 없이
# 뒤지면 멀리 떨어진 무관한 캡션을 잘못 채택할 위험이 크다.
_ADJACENT_PAGE_EDGE_RATIO = 0.15


@dataclass
class CaptionMapping:
    table_index: int
    page_no: int
    caption_text: Optional[str]
    caption_ref: Optional[str]        # texts 인덱스 문자열. fallback으로 찾은 경우 None
    confidence: Literal["direct", "inferred", "none"]
    multi_caption: bool = False       # 표 하나가 captions 참조 2개 이상 보유
    cross_page: bool = False          # 캡션이 표와 다른 페이지에 위치


def _center_y(bbox: "BBox") -> float:
    """bbox의 수직 중심. TOPLEFT라 값이 작을수록 페이지 위쪽이다."""
    return (bbox.t + bbox.b) / 2.0


def _looks_like_caption(text: Optional[str]) -> bool:
    """
    번호가 부여된 캡션 패턴("Table 1", "<표 3>" 등) 포함 여부.

    표뿐 아니라 그림 캡션까지 넓게 인정한다 — 완전히 버리는 대신 fallback에서
    confidence를 "inferred"로 낮춰 정보 손실 없이 활용한다. 앞부분
    CAPTION_PREFIX_CHARS 글자만 검사하는 이유는 그 상수 주석 참고.
    """
    if not text or len(text.strip()) < _MIN_CAPTION_LEN:
        return False
    prefix = text.strip()[:CAPTION_PREFIX_CHARS]
    return bool(_CAPTION_PATTERN.search(prefix))


# direct 검증은 table/표 패턴만 인정한다 — table.captions가 그림 캡션을
# 가리키는 경우까지 direct로 신뢰하면 safe_caption이 못 걸러내고 EU 전체에
# 노이즈가 퍼진다. bbox/인접 페이지 fallback은 넓은 패턴을 허용하되
# confidence를 inferred로 낮춰서 구분한다.
_TABLE_CAPTION_PATTERN = re.compile(r"(table|표)\s*\.?\s*[A-Z]?\.?\s*\d+", re.IGNORECASE)


def _looks_like_table_caption(text: Optional[str]) -> bool:
    """direct 참조 검증 전용: table/표 패턴만 인정(그림 캡션 오매핑 방지)."""
    if not text or len(text.strip()) < _MIN_CAPTION_LEN:
        return False
    prefix = text.strip()[:CAPTION_PREFIX_CHARS]
    return bool(_TABLE_CAPTION_PATTERN.search(prefix))


def _find_caption_by_bbox(
    parsed: "ParsedDoc", table_page: int, table_top_y: float, table_bot_y: float,
    picture_caption_refs: set[int] | None = None,
) -> Optional[str]:
    """
    captions 참조가 없거나 품질 미달일 때, 같은 페이지에서 bbox 거리 기준으로
    캡션처럼 보이는 텍스트를 찾는 fallback.

    picture_caption_refs가 주어지면, 이미 어느 PictureItem의 캡션으로
    구조적으로 확인된 텍스트(index가 그 집합에 있음)는 후보에서 제외한다.
    """
    candidates: list[tuple[float, str]] = []
    picture_caption_refs = picture_caption_refs or set()

    for item in parsed.texts:
        if item.label.value not in _CANDIDATE_LABEL_NAMES:
            continue
        if item.page_no != table_page:
            continue
        if item.index in picture_caption_refs:
            continue

        text = item.text.strip()
        if not _looks_like_caption(text):
            continue

        cy = _center_y(item.bbox)
        if cy < table_top_y:
            dist = table_top_y - cy
        elif cy > table_bot_y:
            dist = cy - table_bot_y
        else:
            dist = 0.0

        if dist <= CAPTION_SEARCH_PT:
            candidates.append((dist, text))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _find_caption_adjacent_page(
    parsed: "ParsedDoc", table_page: int, table_top_y: float, table_bot_y: float,
    picture_caption_refs: set[int] | None = None,
) -> tuple[Optional[str], bool]:
    """
    같은 페이지 bbox fallback도 실패했을 때, 표 바로 앞/뒤 페이지에서 캡션 재탐색.

    페이지가 다르면 좌표계(페이지별 y 원점)가 서로 달라 bbox 거리 비교가 무의미하므로, 
    "이전 페이지에서 가장 아래" / "다음 페이지에서 가장 위"에 있는 캡션 패턴 텍스트를 채택한다.

    Returns:
        (caption_text, found) — 못 찾거나 게이팅에 걸리면 (None, False)
    """
    page_height = parsed.page_sizes.get(table_page, (0.0, 0.0))[1]
    if not page_height:
        return None, False

    near_top = table_top_y <= page_height * _ADJACENT_PAGE_EDGE_RATIO
    near_bottom = table_bot_y >= page_height * (1 - _ADJACENT_PAGE_EDGE_RATIO)
    if not near_top and not near_bottom:
        return None, False

    picture_caption_refs = picture_caption_refs or set()
    prev_candidates: list[tuple[float, str]] = []
    next_candidates: list[tuple[float, str]] = []

    for item in parsed.texts:
        if item.label.value not in _CANDIDATE_LABEL_NAMES:
            continue
        if item.index in picture_caption_refs:
            continue
        text = item.text.strip()
        if not _looks_like_caption(text):
            continue
        cy = _center_y(item.bbox)

        if near_top and item.page_no == table_page - 1:
            prev_candidates.append((cy, text))  # 이전 페이지 맨 아래 = TOPLEFT에서 cy 최댓값
        elif near_bottom and item.page_no == table_page + 1:
            next_candidates.append((cy, text))  # 다음 페이지 맨 위 = TOPLEFT에서 cy 최솟값

    if prev_candidates:
        prev_candidates.sort(key=lambda x: -x[0])
        return prev_candidates[0][1], True

    if next_candidates:
        next_candidates.sort(key=lambda x: x[0])
        return next_candidates[0][1], True

    return None, False


# fallback 4: Docling이 캡션을 별도 텍스트 아이템이 아니라 표 첫 행(병합된
# th/td)으로 렌더링하는 경우 — 위 fallback들은 parsed.texts만 순회하므로
# 이 케이스를 발견하지 못한다.
#
# table.html/cells는 여기서 건드리지 않는다. 두 데이터는 같은 표에서 독립
# 계산되므로 한쪽(html)에서만 캡션 행을 지우면 flatten.py/split.py에서 행
# 인덱스가 어긋난다. 대신 EU.text에 캡션이 한 번 더 남는 경미한 토큰 중복을
# 감수한다.

_MERGED_HEADER_PATTERN = re.compile(
    r'<t[hd][^>]*colspan="(\d+)"[^>]*>\s*(.*?)\s*</t[hd]>',
    re.IGNORECASE | re.DOTALL,
)


# 병합 셀 캡션 탐색은 표 "첫 행"만 본다. 이 규칙은 캡션이 표 안으로 빨려
# 들어간 경우를 복구하려는 것이고, 캡션은 표 위에 붙으므로 표 HTML 안에서는
# 첫 행으로만 나타난다. 범위를 풀면 표 중간의 그룹 구분 행, "Table 9:
# (continued)" 같은 반복 머리글, 다른 표를 가리키는 셀 값까지 캡션으로
# 채택돼 EU 전체(text/retrieval_text/retrieval_units/table_abstract)에
# 잘못된 정체성이 복제된다.
_FIRST_ROW_PATTERN = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)


def _find_caption_in_merged_header(table_html: Optional[str]) -> Optional[str]:
    """
    표 첫 행에 여러 컬럼을 병합한 th/td가 있고, 그 안 텍스트가 캡션
    패턴(_looks_like_table_caption)에 맞으면 채택.

    첫 행에서 못 찾으면 더 아래 행은 보지 않는다 — 윗주석 참고.

    Returns:
        caption_text — 못 찾으면 None
    """
    if not table_html:
        return None
    first_row = _FIRST_ROW_PATTERN.search(table_html)
    if not first_row:
        return None
    m = _MERGED_HEADER_PATTERN.search(first_row.group(1))
    if not m:
        return None
    cell_text = re.sub(r"<[^>]+>", " ", m.group(2)).strip()
    if not _looks_like_table_caption(cell_text):
        return None
    return cell_text


def map_table_caption(parsed: "ParsedDoc", table: "TableBlock", table_index: int) -> CaptionMapping:
    """
    단일 표에 대한 캡션 매핑 (1:1).

    우선순위:
      1. captions 참조가 가리키는 텍스트가 캡션답게 생겼으면 채택 (direct).
         참조가 2개 이상(복수 캡션)이면 캡션 패턴에 맞는 텍스트를 모두 병합.
      2. 참조가 없거나 전부 파편을 가리키면 같은 페이지 bbox 거리로 재탐색 (inferred)
      3. 같은 페이지에서도 못 찾으면 표 바로 앞/뒤 페이지에서 재탐색
         (inferred, cross_page=True)
      4. 표 구조 자체(table.html)에 캡션이 병합된 행으로 들어있는지 확인 (inferred)
      5. 전부 실패하면 캡션 없음 (none)
    """
    table_page = table.page_no
    table_top_y = table.bbox.t
    table_bot_y = table.bbox.b

    cap_refs = table.caption_refs
    multi_caption = len(cap_refs) > 1
    cross_page = False

    picture_caption_refs = parsed.picture_caption_refs

    resolved_texts: list[str] = []
    resolved_refs: list[str] = []
    for idx in cap_refs:
        if idx in picture_caption_refs:
            continue  # 구조적으로 그림 캡션 확인됨 -> direct 후보에서 제외
        candidate_text = parsed.texts[idx].text or None
        if not _looks_like_table_caption(candidate_text):
            continue
        caption_page = parsed.texts[idx].page_no
        if caption_page != -1 and caption_page != table_page:
            cross_page = True
        resolved_texts.append(candidate_text.strip())
        resolved_refs.append(str(idx))

    if resolved_texts:
        return CaptionMapping(
            table_index=table_index,
            page_no=table_page,
            caption_text=" ".join(resolved_texts),
            caption_ref="; ".join(resolved_refs),
            confidence="direct",
            multi_caption=multi_caption,
            cross_page=cross_page,
        )

    fallback_text = _find_caption_by_bbox(parsed, table_page, table_top_y, table_bot_y, picture_caption_refs)
    if fallback_text:
        return CaptionMapping(
            table_index=table_index,
            page_no=table_page,
            caption_text=fallback_text,
            caption_ref=None,
            confidence="inferred",
            multi_caption=multi_caption,
            cross_page=False,
        )

    adjacent_text, found_adjacent = _find_caption_adjacent_page(
        parsed, table_page, table_top_y, table_bot_y, picture_caption_refs
    )
    if found_adjacent:
        return CaptionMapping(
            table_index=table_index,
            page_no=table_page,
            caption_text=adjacent_text,
            caption_ref=None,
            confidence="inferred",
            multi_caption=multi_caption,
            cross_page=True,
        )

    merged_caption = _find_caption_in_merged_header(table.html)
    if merged_caption:
        return CaptionMapping(
            table_index=table_index,
            page_no=table_page,
            caption_text=merged_caption,
            caption_ref=None,
            confidence="inferred",
            multi_caption=multi_caption,
            cross_page=cross_page,
        )

    return CaptionMapping(
        table_index=table_index,
        page_no=table_page,
        caption_text=None,
        caption_ref=None,
        confidence="none",
        multi_caption=multi_caption,
        cross_page=cross_page,
    )


def map_all_captions(parsed: "ParsedDoc") -> list[CaptionMapping]:
    """문서 내 모든 표에 대해 캡션 매핑 수행."""
    return [map_table_caption(parsed, table, table.index) for table in parsed.tables]


def validate_mapping(mappings: list[CaptionMapping]) -> dict:
    """
    1:1 매핑 검증.

    반환:
        total_tables, mapped, unmapped, mapping_rate
        multi_caption_tables : 복수 캡션 참조를 가진 표 인덱스 목록
        cross_page_tables    : 캡션이 다른 페이지에 있는 표 인덱스 목록
        collisions           : 같은 caption_ref를 참조하는 표가 2개 이상인 경우
                                {caption_ref: [table_index, ...]}
    """
    total = len(mappings)
    mapped = sum(1 for m in mappings if m.caption_text)
    unmapped = total - mapped
    multi = [m.table_index for m in mappings if m.multi_caption]
    cross_page = [m.table_index for m in mappings if m.cross_page]
    by_confidence = {
        "direct": sum(1 for m in mappings if m.confidence == "direct"),
        "inferred": sum(1 for m in mappings if m.confidence == "inferred"),
        "none": sum(1 for m in mappings if m.confidence == "none"),
    }

    ref_owners: dict[str, list[int]] = {}
    for m in mappings:
        if m.caption_ref:
            ref_owners.setdefault(m.caption_ref, []).append(m.table_index)
    collisions = {ref: idxs for ref, idxs in ref_owners.items() if len(idxs) > 1}

    return {
        "total_tables": total,
        "mapped": mapped,
        "unmapped": unmapped,
        "mapping_rate": round(mapped / total, 3) if total else 0.0,
        "by_confidence": by_confidence,
        "multi_caption_tables": multi,
        "cross_page_tables": cross_page,
        "collisions": collisions,
    }
