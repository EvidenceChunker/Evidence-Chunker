# Copyright 2026 docling-evidence-chunker contributors
# SPDX-License-Identifier: Apache-2.0
"""
context.py

표 주변의 인접 해설 단락(context)을 탐지하고 병합하는 모듈. 메인 진입점은
attach_context_paragraphs(eu, parsed) — bbox 거리로 후보 단락을 모으고
(sim_threshold > 0이면) 코사인 유사도로 한 번 더 거른 뒤, 섹션 경계와
멀티컬럼 레이아웃 예외를 처리해서 EU에 붙인다.

parser.base.ParsedDoc만 바라보고 DoclingDocument를 직접 다루지 않으며,
좌표는 전부 TOPLEFT 기준이다(parser/base.py 참고) — 원본 BOTTOMLEFT와
상/하 부등호 방향이 반대이니 주의.

sim_threshold 기본값은 0.0이고, 이 경우 _embedding_filter()는
model.encode() 호출 자체를 건너뛴다. 그래서 기본 설정으로 쓰면
sentence-transformers/torch가 없어도 청킹이 그대로 동작한다 — 0보다 큰
값을 명시적으로 넘기는 호출자만 그 의존성이 필요하다(pyproject.toml의
similarity extra 참고).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parser.base import BBox, ParsedDoc, TableBlock
    from .unit import EvidenceUnit

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

CONTEXT_WINDOW_PT = 300.0   # 표 위아래 300 PDF 포인트 이내
CONTEXT_LABEL_NAMES = {"text", "list_item", "paragraph"}
COLUMN_OVERLAP_TOL = 20.0   # 멀티컬럼 컬럼 판별 허용 오차 (pt)
ADJACENT_PAGE_EDGE_RATIO = 0.15  # 페이지 상하단 15% 이내일 때 인접 페이지 탐색

# 문맥 단락이 EU에서 차지할 수 있는 토큰 상한.
#
# 문맥은 분할 조각 전부에 복제되는 고정 비용이라, 제한이 없으면 표 본문보다
# 문맥이 커져서 "행 하나 + 고정 비용"만으로 토큰 한도를 넘는다. 그러면 큰 표가
# 행 분할 대상에서 빠지고 llm_summary로 통째로 흘러간다.
#
# bbox 수집 범위(300pt) 자체는 명세대로 두고, 수집된 단락 중 표에서 가까운
# 것부터 이 예산까지만 싣는다.
CONTEXT_TOKEN_BUDGET = 192


def _center_y(bbox: "BBox") -> float:
    """bbox의 수직 중심. TOPLEFT라 값이 작을수록 페이지 위쪽이다."""
    return (bbox.t + bbox.b) / 2.0


def _same_column(table_bbox: "BBox", para_bbox: "BBox") -> bool:
    """표와 단락의 x 범위가 겹치면 같은 컬럼(단일 컬럼 PDF는 항상 True)."""
    return (
        para_bbox.r >= table_bbox.l - COLUMN_OVERLAP_TOL
        and para_bbox.l <= table_bbox.r + COLUMN_OVERLAP_TOL
    )


def find_section_header(parsed: "ParsedDoc", table_page: int, table_top_y: float) -> str | None:
    """
    표 위쪽에서 가장 가까운 section_header 반환.

    TOPLEFT 좌표계: 시각적으로 표 위 = cy < table_top_y.
    같은 페이지에 없으면 이전 페이지 중 마지막 헤더.
    """
    from .parser.base import BlockLabel

    same_above: list[tuple[float, str]] = []
    prev_pages: list[tuple[int, float, str]] = []

    for item in parsed.texts:
        if item.label != BlockLabel.SECTION_HEADER:
            continue
        if item.page_no == -1:
            continue
        cy = _center_y(item.bbox)
        text = item.text.strip()
        if not text:
            continue

        if item.page_no == table_page and cy < table_top_y:
            same_above.append((table_top_y - cy, text))
        elif item.page_no < table_page:
            prev_pages.append((item.page_no, cy, text))

    if same_above:
        same_above.sort(key=lambda x: x[0])
        return same_above[0][1]

    if prev_pages:
        # 가장 가까운 이전 페이지(page_no 최대)를 고르고, 그 페이지에서 시각적으로
        # 가장 아래 = 가장 나중에 등장한 헤더를 고른다. TOPLEFT에서는 cy가 클수록
        # 페이지 아래쪽이므로, 오름차순 정렬 후 마지막 원소가 (최대 page_no, 최대 cy)다.
        prev_pages.sort(key=lambda x: (x[0], x[1]))
        return prev_pages[-1][2]

    return None


def _has_section_boundary(
    parsed: "ParsedDoc",
    table_page: int,
    table_edge_y: float,
    para_cy: float,
    direction: str,  # "above" | "below"
) -> bool:
    """
    표와 단락 사이에 section_header가 있으면 True (과포함 방지).

    TOPLEFT: above는 para_cy < section_cy < table_edge_y, 
             below는 table_edge_y < section_cy < para_cy
    """
    from .parser.base import BlockLabel

    for item in parsed.texts:
        if item.label != BlockLabel.SECTION_HEADER:
            continue
        if item.page_no != table_page:
            continue
        cy = _center_y(item.bbox)

        if direction == "above" and para_cy < cy < table_edge_y:
            return True
        if direction == "below" and table_edge_y < cy < para_cy:
            return True

    return False


def _collect_by_bbox(
    parsed: "ParsedDoc",
    table_page: int,
    table_bbox: "BBox",
    bbox_threshold: float,
) -> tuple[list[tuple[float, str, int]], list[tuple[float, str, int]]]:
    """
    bbox 거리 기준 단락 수집. 같은 페이지 + 인접 페이지 경계 케이스 포함.
    섹션 경계를 넘는 단락은 항상 제외.

    같은 컬럼 단락을 우선하고, 없으면 컬럼 제한 없이 재탐색한다
    (표가 본문 컬럼 그리드에 딱 안 맞게 배치된 경우 대비,
    section_boundary 체크가 이미 다른 섹션 내용 유입을 막고 있어 안전함).

    Returns:
        before: [(거리, 텍스트, page_no), ...] — 표 위쪽, 거리 오름차순
        after:  [(거리, 텍스트, page_no), ...] — 표 아래쪽, 거리 오름차순
    """
    from .parser.base import BlockLabel

    table_top_y = table_bbox.t
    table_bot_y = table_bbox.b

    before_same_col: list[tuple[float, str, int]] = []
    before_any_col: list[tuple[float, str, int]] = []
    after_same_col: list[tuple[float, str, int]] = []
    after_any_col: list[tuple[float, str, int]] = []

    # TOPLEFT: 페이지 위쪽 = y가 0에 가까움, 아래쪽 = y가 page_height에 가까움.
    page_height = parsed.page_sizes.get(table_page, (0.0, 0.0))[1]
    check_prev = page_height > 0 and table_top_y <= page_height * ADJACENT_PAGE_EDGE_RATIO
    check_next = page_height > 0 and table_bot_y >= page_height * (1 - ADJACENT_PAGE_EDGE_RATIO)

    before: list[tuple[float, str, int]] = []
    after: list[tuple[float, str, int]] = []

    for item in parsed.texts:
        if item.label.value not in CONTEXT_LABEL_NAMES:
            continue
        text = item.text.strip()
        if not text:
            continue
        cy = _center_y(item.bbox)

        if item.page_no == table_page:
            same_col = _same_column(table_bbox, item.bbox)

            if cy < table_top_y:  # TOPLEFT: 표 위
                dist = table_top_y - cy
                if dist > bbox_threshold:
                    continue
                if _has_section_boundary(parsed, table_page, table_top_y, cy, "above"):
                    continue
                (before_same_col if same_col else before_any_col).append((dist, text, table_page))

            elif cy > table_bot_y:  # TOPLEFT: 표 아래
                dist = cy - table_bot_y
                if dist > bbox_threshold:
                    continue
                if _has_section_boundary(parsed, table_page, table_bot_y, cy, "below"):
                    continue
                (after_same_col if same_col else after_any_col).append((dist, text, table_page))

        # 이전 페이지 하단 단락. TOPLEFT는 페이지 하단이 page_height라(페이지마다 다름) 
        # BOTTOMLEFT와 달리 그 페이지 자신의 높이 조회가 필요하다.
        elif check_prev and item.page_no == table_page - 1:
            prev_page_height = parsed.page_sizes.get(table_page - 1, (0.0, 0.0))[1]
            if prev_page_height > 0 and cy >= prev_page_height - bbox_threshold:
                before.append((prev_page_height - cy + 1.0, text, table_page - 1))

        # 다음 페이지 상단 단락. TOPLEFT 상단은 항상 y=0이라 높이 조회가 필요 없다.
        elif check_next and item.page_no == table_page + 1:
            if cy <= bbox_threshold:
                after.append((cy + 1.0, text, table_page + 1))

    before.extend(before_same_col or before_any_col)
    after.extend(after_same_col or after_any_col)

    before.sort(key=lambda x: x[0])
    after.sort(key=lambda x: x[0])
    return before, after


_EMBEDDING_MODEL = None  # 모듈 레벨 캐시 — EU마다 재로드 방지


def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


def _embedding_filter(
    candidates: list[tuple[float, str, int]],
    reference_text: str,
    sim_threshold: float,
) -> list[tuple[str, int]]:
    """
    코사인 유사도 >= sim_threshold 인 단락만 통과.

    reference_text 없으면 전부 통과 (캡션 없는 표 방어).
    sentence-transformers 미설치 시 전부 통과 (graceful degradation).
    sim_threshold <= 0 이면 전부 통과 — 아래 short-circuit 참고.
    """
    if not reference_text:
        return [(text, page) for _, text, page in candidates]

    # 코사인 유사도는 음수로 내려가지 않으므로 sim_threshold<=0이면 결과가
    # 어차피 전부 통과다. 기본값이 0.0이라 이 분기가 거의 항상 타므로, 결과가
    # 정해진 채로 임베딩 모델을 로드/추론하는 무거운 연산은 건너뛴다 — 이
    # import 자체도 여기서 걸리지 않아 기본 설정이면 sentence-transformers
    # 없이도 청킹이 동작한다.
    if sim_threshold <= 0:
        return [(text, page) for _, text, page in candidates]

    try:
        from sentence_transformers import util
        model = _get_embedding_model()
    except ImportError:
        return [(text, page) for _, text, page in candidates]

    if not candidates:
        return []

    ref_emb = model.encode(reference_text, convert_to_tensor=True)
    texts = [text for _, text, _ in candidates]
    pages = [page for _, _, page in candidates]
    cand_embs = model.encode(texts, convert_to_tensor=True)
    scores = util.cos_sim(ref_emb, cand_embs)[0]

    return [(text, page) for text, page, score in zip(texts, pages, scores)
            if score.item() >= sim_threshold]


def _apply_token_budget(
    before_candidates: list[tuple[float, str, int]],
    after_candidates: list[tuple[float, str, int]],
    before_kept: list[tuple[str, int]],
    after_kept: list[tuple[str, int]],
    budget: int = CONTEXT_TOKEN_BUDGET,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """표에서 가까운 단락부터 토큰 예산까지만 남긴다.

    위/아래를 따로 자르지 않고 거리 하나로 줄 세운다 — 설명이 표 아래에만
    몰린 문서에서 위쪽 몫을 비워두는 낭비를 막기 위해서다. 예산을 넘기는
    단락이 나오면 거기서 끊는다(그 뒤 단락은 더 멀다).
    """
    from .tokens import count_tokens

    keep_before = {text for text, _ in before_kept}
    keep_after = {text for text, _ in after_kept}

    merged = [("b", d, t, p) for d, t, p in before_candidates if t in keep_before]
    merged += [("a", d, t, p) for d, t, p in after_candidates if t in keep_after]
    merged.sort(key=lambda x: x[1])

    used = 0
    kept_b: list[tuple[float, str, int]] = []
    kept_a: list[tuple[float, str, int]] = []
    for side, dist, text, page in merged:
        cost = count_tokens(text)
        if used + cost > budget:
            break
        used += cost
        (kept_b if side == "b" else kept_a).append((dist, text, page))

    kept_b.sort(key=lambda x: x[0])
    kept_a.sort(key=lambda x: x[0])
    return ([(t, p) for _, t, p in kept_b], [(t, p) for _, t, p in kept_a])


def attach_context_paragraphs(
    eu: "EvidenceUnit",
    parsed: "ParsedDoc",
    bbox_threshold: float = CONTEXT_WINDOW_PT,
    sim_threshold: float = 0.0,
    table_bbox: "BBox | None" = None,
) -> "EvidenceUnit":
    """
    EU에 인접 설명 단락 + 섹션 헤더 추가.

    Args:
        eu: EU 기본 뼈대
        parsed: 파서 추상화 문서
        bbox_threshold: 표 위아래 수집 범위 (PDF 포인트). 기본 300pt
        sim_threshold: 임베딩 유사도 임계값. 기본 0.0(사실상 무필터).
            더 엄격한 필터가 필요하면 호출자가 직접 값을 올려서 넘기면 됨.
        table_bbox: 이 EU가 속한 표의 bbox. 호출자가 직접 넘길 것 — 표
            dedup으로 eu_id 순번과 parsed.tables 스캔 순서가 어긋날 수
            있어, None일 때 쓰는 eu_id 기반 추정은 부정확할 수 있다.

    Returns:
        context_before / context_after / section_header 가 채워진 EU
    """
    if table_bbox is None:
        # eu_id 기반 추정 — 실제 파이프라인은 항상 table_bbox를 직접 넘기므로
        # 이 경로는 안 탄다. doc_id에 하이픈이 있으면 파싱이 깨지는 갭이 있음.
        try:
            eu_idx = int(eu.eu_id.split("-")[-1])
        except (ValueError, IndexError):
            eu_idx = 0

        page_tables: list["BBox"] = [
            t.bbox for t in parsed.tables if t.page_no == eu.page_no
        ]

        if eu_idx >= len(page_tables):
            return eu

        table_bbox = page_tables[eu_idx]

    if not table_bbox:
        return eu

    table_top_y = table_bbox.t

    eu.section_header = find_section_header(parsed, eu.page_no, table_top_y)

    before_candidates, after_candidates = _collect_by_bbox(
        parsed, eu.page_no, table_bbox, bbox_threshold
    )

    # 기준 텍스트는 캡션뿐이다 — eu.table_abstract는 이 함수가 끝난 뒤에
    # 채워지므로(chunker.build_evidence_units의 호출 순서) 여기서 참조해도
    # 항상 비어 있다. 캡션이 없으면 빈 문자열이 되고, _embedding_filter()가
    # 후보를 전부 통과시킨다.
    reference = eu.caption_text or ""
    before_kept = _embedding_filter(before_candidates, reference, sim_threshold)
    after_kept  = _embedding_filter(after_candidates,  reference, sim_threshold)

    before_kept, after_kept = _apply_token_budget(
        before_candidates, after_candidates, before_kept, after_kept
    )

    eu.context_before = [text for text, _ in before_kept]
    eu.context_after  = [text for text, _ in after_kept]

    # 표 자신의 페이지 + 실제로 채택된 문맥 단락이 걸친 페이지 전부.
    eu.page_span = {eu.page_no} | {p for _, p in before_kept} | {p for _, p in after_kept}

    return eu
